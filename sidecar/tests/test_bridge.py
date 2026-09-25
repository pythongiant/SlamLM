"""End-to-end tests for the SlamLM Python bridge.

These drive the *real* bridge: a `python -m slam_lm_bridge.server` subprocess
built on the real MLX model cache. Nothing is mocked and nothing is fabricated —
every assertion is about a measurement taken from that process.

Run with pytest, from the app root::

    .venv/bin/python -m pytest sidecar/tests/test_bridge.py -q

or as a plain script (same steps, printed in order, plus the live wire lines)::

    PYTHONPATH=sidecar .venv/bin/python sidecar/tests/test_bridge.py

The bridge is spawned with the interpreter running the tests, so any virtualenv
with `mlx-lm` installed works; only `sidecar/` must be the package's parent.

The steps are an ordered scenario that shares one bridge process (load once,
generate, serve HTTP, unload), so they are intentionally not independent.

`SLAM_LM_BRIDGE_ROOT` points the harness at a different copy of the
`slam_lm_bridge` package. The default is the source tree in this repo; the
override exists so a reverted copy outside the repo can be pointed at to show a
regression test failing against the pre-fix code.
"""

from __future__ import annotations

import atexit
import collections
import http.client
import itertools
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

#: The `tests/` directory's parent, i.e. the directory holding `slam_lm_bridge`.
SIDECAR = Path(__file__).resolve().parents[1]
#: The app root: `menubar/` in a source tree, the repository root when published.
APP_ROOT = SIDECAR.parent
#: The directory *containing* the `slam_lm_bridge` package under test.
BRIDGE_ROOT = Path(os.environ.get("SLAM_LM_BRIDGE_ROOT") or SIDECAR)
#: Spawn the bridge with the interpreter that is running these tests — it is by
#: definition the environment `mlx-lm` was installed into, whatever the layout.
PYTHON = Path(sys.executable)

MODEL_ID = "mlx-community/Qwen3-0.6B-4bit"
OTHER_MODEL_ID = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
LOCAL_MODEL_IDS = (
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Qwen3-1.7B-4bit",
    "mlx-community/Qwen1.5-0.5B-Chat-4bit",
    "convaiinnovations/laya",
)
#: 174 MB ONNX embedding export: no config.json, no loadable weights -> skipped.
WEIGHTLESS_MODEL_ID = "Qdrant/all-MiniLM-L6-v2-onnx"

PROMPT = "Say hello in exactly five words."
GENERATE_TOKENS = 24
#: The `chat` rendering is compared through the same tiny window: a handful of
#: decoded tokens is enough to publish a record, so these stay fast.
CHAT_MAX_TOKENS = 4
#: Request ids for these tests, well clear of every id the fixed scenarios use,
#: so their events can never be read as an older request's.
CHAT_REQUEST_BASE = 9000
_CHAT_REQUEST_IDS = itertools.count(CHAT_REQUEST_BASE)
REQUEST_ID = 7
MEGABYTE = 1024 * 1024

#: PROTOCOL.md's canonical display order for the derived model categories.
CANONICAL_CATEGORIES = (
    "Chat",
    "Code",
    "Vision",
    "Embedding",
    "Audio",
    "Instruct",
    "Multilingual",
    "Reasoning",
)

#: Long enough that a following request is still queued behind it.
LONG_PROMPT = "Write a long essay about mountains. " * 40
LONG_MAX_TOKENS = 400

#: Enough output that a resident set which never fell is unmistakable.
RSS_DROP_TOKENS = 64

#: RSS samples per window. macOS reports only truly resident pages and Metal's
#: working set comes and goes between 5 Hz samples, so one sample of a 0.6B
#: model's process can read ~300 MB low; the max (loaded) and min (unloaded)
#: over a short window are the stable measurements.
RSS_WINDOW_SAMPLES = 6


# MARK: - Bridge process harness


class BridgeSession:
    """One live bridge subprocess, with helpers for its wire protocol."""

    def __init__(self, state_path: Path):
        env = dict(os.environ)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{BRIDGE_ROOT}{os.pathsep}{existing}" if existing else str(BRIDGE_ROOT)
        )
        env["SLAM_LM_STATE"] = str(state_path)
        self.state_path = state_path
        self.proc = subprocess.Popen(
            [str(PYTHON), "-m", "slam_lm_bridge.server"],
            cwd=str(APP_ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._condition = threading.Condition()
        self._replies: Dict[int, Dict[str, Any]] = {}
        self._events: List[Dict[str, Any]] = []
        self._raw: List[str] = []
        self._stderr: Deque[str] = collections.deque(maxlen=500)
        self._next_id = 0
        self._exited = False
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

    # -- readers

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            text = line.rstrip("\n")
            if not text:
                continue
            try:
                message: Any = json.loads(text)
            except ValueError:
                message = None
            with self._condition:
                self._raw.append(text)
                if isinstance(message, dict):
                    if "event" in message:
                        self._events.append(message)
                    elif "id" in message:
                        self._replies[int(message["id"])] = message
                self._condition.notify_all()
        with self._condition:
            self._exited = True
            self._condition.notify_all()

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            with self._condition:
                self._stderr.append(line.rstrip("\n"))

    # -- wire

    def next_id(self) -> int:
        with self._condition:
            self._next_id += 1
            return self._next_id

    def send_burst(self, payloads: List[Dict[str, Any]]) -> None:
        """Write several requests at once, without waiting for any reply."""
        assert self.proc.stdin is not None
        self.proc.stdin.write(
            "".join(json.dumps(payload) + "\n" for payload in payloads)
        )
        self.proc.stdin.flush()

    def await_reply(
        self, request_id: int, timeout: float = 180.0, context: Any = None
    ) -> Dict[str, Any]:
        """Wait for the reply to an already-sent `request_id`."""
        payload = context if context is not None else request_id
        deadline = time.time() + timeout
        with self._condition:
            while request_id not in self._replies:
                if self._exited:
                    raise AssertionError(
                        f"bridge exited before replying to {payload}\n"
                        f"stderr:\n{self.stderr_text()}"
                    )
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise AssertionError(
                        f"no reply to {payload} within {timeout}s\n"
                        f"stderr:\n{self.stderr_text()}"
                    )
                self._condition.wait(min(remaining, 0.5))
            return self._replies.pop(request_id)

    def request(self, cmd: str, timeout: float = 180.0, **fields: Any) -> Dict[str, Any]:
        request_id = self.next_id()
        payload = {"id": request_id, "cmd": cmd}
        payload.update(fields)
        self.send_burst([payload])
        return self.await_reply(request_id, timeout, payload)

    def event_count(self) -> int:
        with self._condition:
            return len(self._events)

    @property
    def pid(self) -> int:
        return self.proc.pid

    def events(self, name: str) -> List[Dict[str, Any]]:
        with self._condition:
            return [
                event["data"]
                for event in self._events
                if event.get("event") == name and isinstance(event.get("data"), dict)
            ]

    def event_log(self, name: str, after: int = 0) -> List[Dict[str, Any]]:
        """Every `name` event at or after index `after`, in emission order."""
        with self._condition:
            return [
                dict(event["data"])
                for event in self._events[after:]
                if event.get("event") == name and isinstance(event.get("data"), dict)
            ]

    def wait_for(
        self,
        name: str,
        after: int = 0,
        timeout: float = 60.0,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, Any]:
        """Wait for the first `name` event at or after index `after`."""
        deadline = time.time() + timeout
        with self._condition:
            while True:
                for index in range(after, len(self._events)):
                    event = self._events[index]
                    if event.get("event") != name:
                        continue
                    data = event.get("data")
                    if not isinstance(data, dict):
                        continue
                    if predicate is None or predicate(data):
                        return data
                if self._exited:
                    raise AssertionError(
                        f"bridge exited before emitting {name!r}\n"
                        f"stderr:\n{self.stderr_text()}"
                    )
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise AssertionError(
                        f"no {name!r} event within {timeout}s\n"
                        f"stderr:\n{self.stderr_text()}"
                    )
                self._condition.wait(min(remaining, 0.5))

    def metrics(self) -> List[Dict[str, Any]]:
        return self.events("metrics")

    def metrics_after(self, index: int, timeout: float = 10.0) -> Dict[str, Any]:
        """The next metrics sample emitted after `index` (a fresh measurement)."""
        return self.wait_for("metrics", after=index, timeout=timeout)

    def raw_lines(self, *needles: str) -> List[str]:
        with self._condition:
            lines = list(self._raw)
        return [line for line in lines if all(needle in line for needle in needles)]

    def unclaimed_replies(self) -> Dict[int, Dict[str, Any]]:
        """Replies nobody consumed: any entry means an id was answered twice."""
        with self._condition:
            return dict(self._replies)

    def stderr_text(self) -> str:
        with self._condition:
            return "\n".join(self._stderr)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        for stream in (self.proc.stdout, self.proc.stderr):
            if stream is not None:
                stream.close()


_SESSION: Optional[BridgeSession] = None
_TEMP_DIR: Optional[tempfile.TemporaryDirectory[str]] = None


def session() -> BridgeSession:
    global _SESSION, _TEMP_DIR
    if _SESSION is None:
        _TEMP_DIR = tempfile.TemporaryDirectory(prefix="slam-lm-test-")
        _SESSION = BridgeSession(Path(_TEMP_DIR.name) / "state.json")
        atexit.register(_SESSION.close)
    return _SESSION


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _http(
    url: str, payload: Optional[Dict[str, Any]] = None, stream: bool = False
) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    response = urllib.request.urlopen(request, timeout=180)
    body = response.read().decode("utf-8")
    if stream:
        return response, body
    return response, json.loads(body)


def _status_of(url: str, payload: Optional[Dict[str, Any]] = None) -> int:
    try:
        _http(url, payload)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    return 200


def _weight_bytes(model_path: str) -> int:
    total = 0
    for path in sorted(Path(model_path).rglob("*.safetensors")):
        total += path.stat().st_size
    return total


def ensure_loaded(bridge: BridgeSession) -> None:
    """Load the test model (a no-op when it is already loaded)."""
    reply = bridge.request("load", model=MODEL_ID)
    assert reply["ok"] is True, reply
    assert reply["result"]["model"] == MODEL_ID, reply


# MARK: - Scenario steps


def step_hello(bridge: BridgeSession) -> None:
    reply = bridge.request("hello")
    assert reply["ok"] is True, reply
    result = reply["result"]
    assert result["protocol"] == 1, result
    assert isinstance(result["bridge"], str) and result["bridge"], result

    hardware = result["hardware"]
    assert hardware["chip"].startswith("Apple ") and "M4" in hardware["chip"], hardware
    total = hardware["totalBytes"]
    assert 15e9 < total <= 18e9 and total % (1024**3) == 0, f"total memory: {total}"
    recommended = hardware["recommendedBytes"]
    assert abs(recommended - 12.7e9) / 12.7e9 < 0.1, f"recommended: {recommended}"
    assert recommended < total, hardware
    assert hardware["model"], hardware
    # An Apple GPU has at least one core and far fewer than 128.
    gpu_cores = hardware["gpuCores"]
    assert isinstance(gpu_cores, int) and 0 < gpu_cores <= 128, hardware
    # MLX's active memory lives inside the process's resident set, which in turn
    # cannot exceed the machine's RAM.
    assert 0 <= hardware["memoryActiveBytes"] <= hardware["processRssBytes"], hardware
    assert hardware["processRssBytes"] < total, hardware

    # `hello.categories` is PROTOCOL.md's canonical order, filtered to exactly
    # the categories the discovered models derive; every model's own list is a
    # subset of it.
    categories = result["categories"]
    assert isinstance(categories, list) and categories, result
    assert all(isinstance(entry, str) and entry for entry in categories), categories
    models = bridge.request("catalog")["result"]["models"]
    discovered = {category for model in models for category in model["categories"]}
    expected = [category for category in CANONICAL_CATEGORIES if category in discovered]
    assert categories == expected, (categories, expected)
    for model in models:
        assert set(model["categories"]) <= set(categories), model


def step_catalog(bridge: BridgeSession) -> None:
    result = bridge.request("catalog")["result"]
    models = {model["id"]: model for model in result["models"]}

    for expected in LOCAL_MODEL_IDS:
        assert expected in models, f"{expected} missing from {sorted(models)}"
    assert WEIGHTLESS_MODEL_ID not in models, "weights-less directory was listed"

    for model_id, model in models.items():
        assert model["bytes"] > 0, model
        assert model["quant"] == "fp16" or model["quant"].endswith("-bit"), model
        quant_digits = model["quant"].split("-")[0]
        assert model["quant"] == "fp16" or quant_digits.isdigit(), model
        assert model["categories"], model
        assert model["contextLength"] > 0, model
        assert isinstance(model["hasChatTemplate"], bool), model
        assert model["architecture"], model
        assert isinstance(model["lastUsed"], float), model
        path = Path(model["path"])
        assert path.is_dir(), model
        assert (path / "config.json").exists() or any(
            child.name == "config.json" for child in path.rglob("config.json")
        ), f"no config.json under {path}"
        # bytes is the summed size of the real weight files
        assert model["bytes"] == _weight_bytes(model["path"]), model_id
        assert model["name"] and model["params"], model


def step_load(bridge: BridgeSession) -> None:
    before = bridge.metrics_after(bridge.event_count())
    # With no model loaded the memory fields are still real, and the rates are
    # zeroed rather than invented.
    assert before["model"] is None, before
    assert before["status"] == "idle", before
    assert before["memoryTotalBytes"] > 0, before
    assert 0 <= before["memoryActiveBytes"] <= before["processRssBytes"], before
    assert before["processRssBytes"] < before["memoryTotalBytes"], before
    assert before["decodeTps"] == 0.0, before
    assert before["ttftMs"] == 0.0, before
    assert before["prefillTps"] == 0.0, before
    assert before["loadMs"] == 0.0, before
    assert before["tokensGenerated"] == 0, before
    assert before["requests"] == 0, before

    reply = bridge.request("load", model=MODEL_ID)
    assert reply["ok"] is True, reply
    result = reply["result"]
    assert result["model"] == MODEL_ID, result
    assert result["loadMs"] > 0, result
    assert result["memoryBytes"] > 0, result

    marker = bridge.event_count()
    after = bridge.metrics_after(marker)
    assert after["memoryActiveBytes"] > before["memoryActiveBytes"], (
        f"active memory did not rise: {before['memoryActiveBytes']} -> "
        f"{after['memoryActiveBytes']}"
    )
    assert after["model"] == MODEL_ID, after
    assert after["loadMs"] == result["loadMs"], after
    assert after["status"] in ("ready", "generating"), after

    # A load of the loaded model is a no-op: same loadMs, no new memory.
    again = bridge.request("load", model=MODEL_ID)["result"]
    assert again["loadMs"] == result["loadMs"], again
    assert again["memoryBytes"] == 0, again

    settled = bridge.metrics_after(bridge.event_count())
    assert settled["status"] == "ready", settled
    assert settled["phase"] == "idle", settled


def step_catalog_tracks_last_used(bridge: BridgeSession) -> None:
    ensure_loaded(bridge)
    models = bridge.request("catalog")["result"]["models"]
    by_id = {model["id"]: model for model in models}
    loaded = by_id[MODEL_ID]
    assert loaded["lastUsed"] > 0, loaded
    assert models[0]["id"] == MODEL_ID, [model["id"] for model in models]
    for model_id, model in by_id.items():
        if model_id != MODEL_ID:
            assert model["lastUsed"] == 0.0, (model_id, model["lastUsed"])

    state = json.loads(session().state_path.read_text(encoding="utf-8"))
    assert MODEL_ID in state["lastUsed"], state


def step_generate(bridge: BridgeSession) -> None:
    ensure_loaded(bridge)
    marker = bridge.event_count()
    reply = bridge.request(
        "generate", prompt=PROMPT, max_tokens=GENERATE_TOKENS, request=REQUEST_ID
    )
    assert reply["ok"] is True, reply
    assert reply["result"] == {"request": REQUEST_ID}, reply

    record = bridge.wait_for(
        "request_end", after=marker, predicate=lambda data: data["request"] == REQUEST_ID
    )
    tokens = [
        event
        for event in bridge.events("token")
        if event["request"] == REQUEST_ID
    ]
    text = "".join(token["text"] for token in tokens)

    assert text.strip(), "generated no text"
    assert tokens[-1]["index"] == record["genTokens"], (tokens[-1], record)
    assert all(
        tokens[index + 1]["ttsMs"] >= tokens[index]["ttsMs"]
        for index in range(len(tokens) - 1)
    ), "token timestamps went backwards"
    assert abs(tokens[0]["ttsMs"] - record["ttftMs"]) < 5.0, (tokens[0], record)

    assert record["model"] == MODEL_ID, record
    assert record["promptTokens"] > 0, record
    assert record["genTokens"] > 0, record
    assert record["ttftMs"] > 0, record
    assert record["prefillTps"] > 0, record
    assert record["decodeTps"] > 0, record
    assert record["peakMemBytes"] > 0, record
    assert record["totalMs"] >= record["ttftMs"], record
    assert 0 <= time.time() - record["startedAt"] < 300, record
    assert record["finishReason"] in ("length", "stop"), record

    # The record's rates follow PROTOCOL.md's definitions, recomputed here from
    # the token events the bridge actually emitted.
    span = (tokens[-1]["ttsMs"] - tokens[0]["ttsMs"]) / 1000.0
    assert span > 0, tokens
    expected_decode = (len(tokens) - 1) / span
    assert abs(record["decodeTps"] - expected_decode) < 0.02 * expected_decode, (
        record["decodeTps"],
        expected_decode,
    )
    expected_prefill = record["promptTokens"] / (record["ttftMs"] / 1000.0)
    assert abs(record["prefillTps"] - expected_prefill) < 0.02 * expected_prefill, (
        record["prefillTps"],
        expected_prefill,
    )

    # The same request is visible in the live metrics: session counters move
    # and the trailing 2 s decode window reports a real rate.
    live = bridge.wait_for(
        "metrics",
        after=marker,
        timeout=10.0,
        predicate=lambda data: data["decodeTps"] > 0
        and data["tokensGenerated"] >= record["genTokens"],
    )
    assert live["requests"] >= 1, live
    assert live["ttftMs"] > 0 and live["prefillTps"] > 0, live
    assert live["status"] in ("generating", "ready"), live


def step_http(bridge: BridgeSession) -> None:
    port = _free_port()
    serve = bridge.request("serve", port=port)
    assert serve["ok"] is True, serve
    assert serve["result"]["port"] == port, serve
    base = serve["result"]["url"]
    assert base == f"http://127.0.0.1:{port}", serve

    again = bridge.request("serve", port=port)["result"]
    assert again["url"] == base, again  # idempotent

    _, health = _http(f"{base}/health")
    assert health == {"status": "ok", "model": MODEL_ID}, health

    _, models = _http(f"{base}/v1/models")
    assert [entry["id"] for entry in models["data"]] == [MODEL_ID], models

    marker = bridge.event_count()
    _, completion = _http(
        f"{base}/v1/chat/completions",
        {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Name one colour."}],
            "max_tokens": 24,
        },
    )
    choice = completion["choices"][0]
    assert completion["object"] == "chat.completion", completion
    assert completion["model"] == MODEL_ID, completion
    assert choice["message"]["role"] == "assistant", choice
    assert choice["message"]["content"].strip(), completion
    assert choice["finish_reason"] in ("length", "stop"), choice
    assert completion["usage"]["prompt_tokens"] > 0, completion
    assert completion["usage"]["completion_tokens"] > 0, completion

    # HTTP traffic runs through the same runner: a real request_end event with
    # an HTTP request id, and the token events that fed the streamed response.
    http_record = bridge.wait_for(
        "request_end", after=marker, predicate=lambda data: data["request"] >= 1_000_000
    )
    assert http_record["genTokens"] == completion["usage"]["completion_tokens"], (
        http_record,
        completion,
    )
    http_tokens = [
        event
        for event in bridge.events("token")
        if event["request"] == http_record["request"]
    ]
    assert "".join(token["text"] for token in http_tokens).strip(), http_tokens

    # Streaming: SSE chat.completion.chunk frames, terminated by [DONE].
    stream_marker = bridge.event_count()
    _, body = _http(
        f"{base}/v1/chat/completions",
        {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Count to three."}],
            "max_tokens": 16,
            "stream": True,
        },
        stream=True,
    )
    frames = [
        line[len("data: ") :]
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    assert frames[-1] == "[DONE]", body[-200:]
    chunks = [json.loads(frame) for frame in frames[:-1]]
    assert chunks, body[-200:]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks), chunks[:2]
    streamed = "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
    )
    assert streamed.strip(), streamed
    assert chunks[-1]["choices"][0]["finish_reason"] in ("length", "stop"), chunks[-1]
    bridge.wait_for(
        "request_end", after=stream_marker, predicate=lambda data: data["request"] >= 1_000_000
    )

    # Legacy completions surface.
    _, text_completion = _http(
        f"{base}/v1/completions",
        {"model": MODEL_ID, "prompt": "The colour of the sky is", "max_tokens": 8},
    )
    assert text_completion["object"] == "text_completion", text_completion
    assert text_completion["choices"][0]["text"].strip(), text_completion
    assert text_completion["choices"][0]["finish_reason"] in ("length", "stop"), text_completion

    _, metrics = _http(f"{base}/metrics")
    assert metrics["live"]["model"] == MODEL_ID, metrics["live"]
    assert metrics["live"]["tokensGenerated"] > 0, metrics["live"]
    ids = [record["request"] for record in metrics["requests"]]
    assert http_record["request"] in ids, (http_record["request"], ids)
    assert ids == sorted(ids, reverse=True), ids  # newest first
    assert _status_of(f"{base}/definitely-not-a-path") == 404
    assert _status_of(f"{base}/v1/chat/completions", {"messages": []}) == 400

    # A taken port is reported clearly, and the running server survives it.
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        taken = int(blocker.getsockname()[1])
        conflict = bridge.request("serve", port=taken)
        assert conflict["ok"] is False, conflict
        assert str(taken) in conflict["error"], conflict
        assert _http(f"{base}/health")[1]["status"] == "ok"


def step_unknown_command(bridge: BridgeSession) -> None:
    reply = bridge.request("definitely-not-a-command")
    assert reply["ok"] is False, reply
    assert "definitely-not-a-command" in reply["error"], reply
    assert "result" not in reply, reply


def step_unload(bridge: BridgeSession) -> None:
    ensure_loaded(bridge)
    before = bridge.metrics_after(bridge.event_count())
    assert before["model"] == MODEL_ID, before
    assert before["memoryActiveBytes"] > 100 * MEGABYTE, before

    reply = bridge.request("unload")
    assert reply["ok"] is True and reply["result"] == {}, reply

    marker = bridge.event_count()
    after = bridge.metrics_after(marker)
    drop = before["memoryActiveBytes"] - after["memoryActiveBytes"]
    assert drop > 100 * MEGABYTE, (
        f"unload freed only {drop / MEGABYTE:.1f} MB "
        f"({before['memoryActiveBytes']} -> {after['memoryActiveBytes']})"
    )
    assert after["model"] is None, after
    assert after["status"] == "idle", after
    assert after["loadMs"] == 0.0, after
    assert after["memoryCacheBytes"] == 0, after

    # Unloading again is a no-op, and the bridge still answers.
    assert bridge.request("unload")["result"] == {}
    assert bridge.request("ping")["result"]["pong"] is True
    assert len(bridge.request("catalog")["result"]["models"]) >= len(LOCAL_MODEL_IDS)
    assert bridge.request("cancel")["result"] == {"cancelled": False}

    base = bridge.request("serve", port=_free_port())["result"]["url"]
    assert _http(f"{base}/health")[1]["model"] is None
    assert _status_of(f"{base}/v1/models") == 503
    assert (
        _status_of(
            f"{base}/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hello"}]},
        )
        == 503
    )
    generate = bridge.request("generate", prompt="hello", max_tokens=4, request=99)
    assert generate["ok"] is False, generate
    assert "no model" in generate["error"], generate
    bridge.request("stop_serve")


def step_cancel(bridge: BridgeSession) -> None:
    """A cancelled request still gets exactly one `request_end`."""
    ensure_loaded(bridge)
    marker = bridge.event_count()
    reply = bridge.request("generate", prompt=PROMPT, max_tokens=400, request=11)
    assert reply["ok"] is True, reply
    assert reply["result"] == {"request": 11}, reply

    cancelled = bridge.request("cancel")
    assert cancelled["ok"] is True, cancelled
    assert cancelled["result"] == {"cancelled": True}, cancelled

    record = bridge.wait_for(
        "request_end", after=marker, predicate=lambda data: data["request"] == 11
    )
    assert record["finishReason"] == "cancel", record
    assert record["genTokens"] < 400, record
    assert not bridge.unclaimed_replies(), (
        "a request id received more than one reply: "
        f"{sorted(bridge.unclaimed_replies())}"
    )

    # The runner is still usable afterwards.
    marker = bridge.event_count()
    bridge.request("generate", prompt="Say hi.", max_tokens=8, request=12)
    follow_up = bridge.wait_for(
        "request_end", after=marker, predicate=lambda data: data["request"] == 12
    )
    assert follow_up["finishReason"] in ("length", "stop"), follow_up
    assert follow_up["genTokens"] > 0, follow_up
    assert bridge.request("cancel")["result"] == {"cancelled": False}


def step_replace_model(bridge: BridgeSession) -> None:
    """Loading a different model while one is loaded replaces it."""
    ensure_loaded(bridge)
    other = OTHER_MODEL_ID
    reply = bridge.request("load", model=other)
    assert reply["ok"] is True, reply
    assert reply["result"]["model"] == other, reply
    assert reply["result"]["loadMs"] > 0, reply
    assert reply["result"]["memoryBytes"] > 0, reply

    live = bridge.metrics_after(bridge.event_count())
    assert live["model"] == other, live
    assert live["memoryActiveBytes"] > 100 * MEGABYTE, live

    by_id = {model["id"]: model for model in bridge.request("catalog")["result"]["models"]}
    assert by_id[other]["lastUsed"] > 0, by_id[other]

    assert bridge.request("unload")["result"] == {}
    assert bridge.metrics_after(bridge.event_count())["model"] is None


def _ps_rss_bytes(pid: int) -> int:
    """The kernel's own resident set size for `pid`, in bytes."""
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=True
    )
    return int(out.stdout.strip()) * 1024


def _assert_rss_matches_ps(bridge: BridgeSession, sample: Dict[str, Any]) -> int:
    """`processRssBytes` is the measurement `ps` reports, within 2% (or 16 MB)."""
    measured = _ps_rss_bytes(bridge.pid)
    tolerance = max(16 * MEGABYTE, measured // 50)
    assert abs(sample["processRssBytes"] - measured) <= tolerance, (
        f"processRssBytes {sample['processRssBytes']} != ps rss {measured} "
        f"(tolerance {tolerance})"
    )
    return measured


def _rss_window(bridge: BridgeSession, samples: int = RSS_WINDOW_SAMPLES) -> List[int]:
    """That many fresh `processRssBytes` samples, in emission order."""
    return [
        bridge.metrics_after(bridge.event_count())["processRssBytes"]
        for _ in range(samples)
    ]


def step_process_rss_is_live(bridge: BridgeSession) -> None:
    """`processRssBytes` is the live resident set, not a high-water mark.

    `ru_maxrss` is monotone and never falls, so a high-water-mark
    implementation reports the same figure on both sides of an unload, stops
    matching `ps`, and cannot lose resident memory.
    """
    ensure_loaded(bridge)
    marker = bridge.event_count()
    assert bridge.request(
        "generate", prompt=PROMPT, max_tokens=RSS_DROP_TOKENS, request=21
    )["result"] == {"request": 21}
    record = bridge.wait_for(
        "request_end", after=marker, predicate=lambda data: data["request"] == 21
    )
    assert record["genTokens"] > 0, record

    loaded_samples = _rss_window(bridge)
    loaded = bridge.metrics_after(bridge.event_count())
    assert loaded["model"] == MODEL_ID, loaded
    loaded_ps = _assert_rss_matches_ps(bridge, loaded)

    assert bridge.request("unload")["result"] == {}
    after_samples = _rss_window(bridge)
    after = bridge.metrics_after(bridge.event_count())
    after_ps = _assert_rss_matches_ps(bridge, after)
    assert after["model"] is None, after

    loaded_rss = max(loaded_samples)
    after_rss = min(after_samples)
    assert after_rss < loaded_rss, (
        "processRssBytes did not fall across an unload (a high-water mark?) — "
        f"loaded samples {loaded_samples} vs after {after_samples}"
    )
    drop = loaded_rss - after_rss
    # The fall for exactly these steps measures anywhere from ~0.22 to ~0.73 of
    # the loaded resident set on this machine (Metal's working set is not fully
    # resident between samples), so the bounds are the conservative end of that
    # observed range; the strict comparison above is what a high-water mark
    # cannot satisfy, whatever the magnitude.
    assert drop >= 48 * MEGABYTE, (
        f"unload released only {drop / MEGABYTE:.1f} MB "
        f"({loaded_rss} -> {after_rss}, ps {loaded_ps} -> {after_ps})"
    )
    assert drop >= 0.15 * loaded_rss, (
        f"unload released only {drop / loaded_rss:.0%} of the resident set "
        f"({loaded_rss} -> {after_rss}, ps {loaded_ps} -> {after_ps})"
    )


def step_keep_alive_survives_unknown_post(bridge: BridgeSession) -> None:
    """A rejected POST must not desynchronise the next keep-alive request."""
    port = _free_port()
    served = bridge.request("serve", port=port)["result"]
    assert served["port"] == port, served
    body = json.dumps({"messages": [{"role": "user", "content": "hello"}]})
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        connection.request(
            "POST",
            "/definitely-not-a-path",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        first = connection.getresponse()
        first_text = first.read().decode("utf-8", "replace")
        assert first.status == 404, (first.status, first_text[:200])
        first_body = json.loads(first_text)
        assert first_body["error"]["code"] == 404, first_body

        # Immediately, on the same socket: with the body left unread the server
        # parses those leftover bytes as the next request line.
        connection.request("GET", "/health")
        second = connection.getresponse()
        second_text = second.read().decode("utf-8", "replace")
        assert second.status == 200, (second.status, second_text[:200])
        second_body = json.loads(second_text)
        assert second_body["status"] == "ok", second_body

        # The connection stays usable for further request/response pairs.
        connection.request(
            "POST",
            "/definitely-not-a-path",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        third = connection.getresponse()
        third_text = third.read().decode("utf-8", "replace")
        assert third.status == 404, (third.status, third_text[:200])

        connection.request("GET", "/health")
        fourth = connection.getresponse()
        fourth_text = fourth.read().decode("utf-8", "replace")
        assert fourth.status == 200, (fourth.status, fourth_text[:200])
        assert json.loads(fourth_text) == second_body, (fourth_text, second_body)
    finally:
        connection.close()

    assert bridge.request("stop_serve")["result"] == {"stopped": True}


def step_stop_serve_releases_the_port(bridge: BridgeSession) -> None:
    """`stop_serve` reports a bool and really closes the listening socket."""
    port = _free_port()
    served = bridge.request("serve", port=port)["result"]
    assert served["port"] == port, served
    with socket.create_connection(("127.0.0.1", port), timeout=10):
        pass  # it really is listening

    stopped = bridge.request("stop_serve")
    assert stopped["ok"] is True, stopped
    assert stopped["result"] == {"stopped": True}, stopped

    deadline = time.time() + 5.0
    while True:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=2).close()
        except OSError:
            break
        if time.time() >= deadline:
            raise AssertionError(
                f"port {port} still accepts connections after stop_serve"
            )
        time.sleep(0.05)

    # Stopping again is a no-op, and the API can be brought back up.
    assert bridge.request("stop_serve")["result"] == {"stopped": False}
    resumed = bridge.request("serve", port=port)["result"]
    assert resumed["port"] == port, resumed
    assert _http(f"{resumed['url']}/health")[1]["status"] == "ok"
    assert bridge.request("stop_serve")["result"] == {"stopped": True}


def step_cancel_reports_a_queued_request(bridge: BridgeSession) -> None:
    """`cancel` reports true for a request still queued behind a longer one."""
    ensure_loaded(bridge)
    marker = bridge.event_count()
    front = bridge.request(
        "generate", prompt=LONG_PROMPT, max_tokens=LONG_MAX_TOKENS, request=31
    )
    assert front["result"] == {"request": 31}, front
    time.sleep(0.05)  # the worker is inside the front generation by now
    queued = bridge.request(
        "generate", prompt="Now write about rivers.", max_tokens=32, request=32
    )
    assert queued["result"] == {"request": 32}, queued

    cancelled = bridge.request("cancel")
    assert cancelled["ok"] is True, cancelled
    assert cancelled["result"] == {"cancelled": True}, cancelled

    front_record = bridge.wait_for(
        "request_end", after=marker, predicate=lambda data: data["request"] == 31
    )
    assert front_record["finishReason"] == "cancel", front_record

    queued_record = bridge.wait_for(
        "request_end", after=marker, predicate=lambda data: data["request"] == 32
    )
    assert queued_record["finishReason"] == "cancel", queued_record
    # It was cancelled before it ever reached the model.
    assert queued_record["genTokens"] == 0, queued_record
    assert queued_record["ttftMs"] == 0.0, queued_record

    ends = [
        entry
        for entry in bridge.events("request_end")
        if entry["request"] in (31, 32)
    ]
    assert sorted(entry["request"] for entry in ends) == [31, 32], ends
    assert not bridge.unclaimed_replies(), sorted(bridge.unclaimed_replies())
    assert bridge.request("cancel")["result"] == {"cancelled": False}

    # The same cancel while the request has not been picked up at all: the
    # `generate` and the `cancel` go out in one write, so the runner sees the
    # request queued and not yet active. It is cancelled all the same, so the
    # reply has to say so.
    marker = bridge.event_count()
    for offset in range(3):
        request_id = 81 + offset
        generate_id = bridge.next_id()
        cancel_id = bridge.next_id()
        bridge.send_burst(
            [
                {
                    "id": generate_id,
                    "cmd": "generate",
                    "prompt": LONG_PROMPT,
                    "max_tokens": LONG_MAX_TOKENS,
                    "request": request_id,
                },
                {"id": cancel_id, "cmd": "cancel"},
            ]
        )
        accepted = bridge.await_reply(generate_id)
        assert accepted["result"] == {"request": request_id}, accepted
        cancelled = bridge.await_reply(cancel_id)
        assert cancelled["ok"] is True, cancelled
        assert cancelled["result"] == {"cancelled": True}, (offset, cancelled)
        record = bridge.wait_for(
            "request_end",
            after=marker,
            predicate=lambda data: data["request"] == request_id,
        )
        assert record["finishReason"] == "cancel", record
        ends = [
            entry
            for entry in bridge.events("request_end")
            if entry["request"] == request_id
        ]
        assert len(ends) == 1, ends


def step_ttft_includes_the_queue_wait(bridge: BridgeSession) -> None:
    """`ttftMs` is measured from `submit`, so it includes time spent queued."""
    ensure_loaded(bridge)
    marker = bridge.event_count()
    front = bridge.request(
        "generate", prompt=LONG_PROMPT, max_tokens=LONG_MAX_TOKENS, request=41
    )
    assert front["result"] == {"request": 41}, front
    time.sleep(0.05)

    started = time.perf_counter()
    queued = bridge.request(
        "generate", prompt=PROMPT, max_tokens=GENERATE_TOKENS, request=42
    )
    assert queued["result"] == {"request": 42}, queued
    first = bridge.wait_for(
        "token", after=marker, predicate=lambda data: data["request"] == 42
    )
    first_token_ms = (time.perf_counter() - started) * 1000.0

    record = bridge.wait_for(
        "request_end", after=marker, predicate=lambda data: data["request"] == 42
    )
    front_record = bridge.wait_for(
        "request_end", after=marker, predicate=lambda data: data["request"] == 41
    )

    assert record["finishReason"] in ("length", "stop"), record
    assert record["genTokens"] > 0, record
    assert abs(first["ttsMs"] - record["ttftMs"]) < 5.0, (first, record)
    # Both clocks start just before `generate`, so they agree; a worker-pickup
    # clock would report only this request's own prefill (tens of ms).
    assert abs(record["ttftMs"] - first_token_ms) <= 0.25 * first_token_ms, (
        f"ttftMs {record['ttftMs']:.1f} vs {first_token_ms:.1f} ms since submit"
    )
    # The wait really was the front request's runtime, not a rounding artefact.
    assert record["ttftMs"] >= 0.5 * front_record["totalMs"], (record, front_record)


def step_state_events_follow_the_protocol(bridge: BridgeSession) -> None:
    """`state` carries PROTOCOL.md's four fields through a load and a generate."""
    bridge.request("unload")  # a known state, so the load really transitions
    marker = bridge.event_count()
    assert bridge.request("load", model=MODEL_ID)["ok"] is True
    loaded_states = bridge.event_log("state", marker)

    for state in loaded_states:
        assert set(state) == {"status", "phase", "model", "message"}, state
    assert [state["status"] for state in loaded_states] == ["loading", "ready"], (
        loaded_states
    )
    loading, ready = loaded_states
    assert loading["phase"] == "load", loading
    assert loading["model"] is None, loading  # nothing is loaded at that point
    assert isinstance(loading["message"], str), loading
    assert MODEL_ID in loading["message"], loading
    assert ready["phase"] == "idle", ready
    assert ready["model"] == MODEL_ID, ready
    assert ready["message"] is None, ready

    marker = bridge.event_count()
    assert bridge.request(
        "generate", prompt=PROMPT, max_tokens=GENERATE_TOKENS, request=51
    )["result"] == {"request": 51}
    record = bridge.wait_for(
        "request_end", after=marker, predicate=lambda data: data["request"] == 51
    )
    assert record["finishReason"] in ("length", "stop"), record
    bridge.wait_for(
        "state", after=marker, predicate=lambda data: data["status"] == "ready"
    )
    gen_states = bridge.event_log("state", marker)

    for state in gen_states:
        assert set(state) == {"status", "phase", "model", "message"}, state
        assert state["status"] in ("generating", "ready"), state
        assert state["model"] == MODEL_ID, state
    assert [state["status"] for state in gen_states] == [
        "generating",
        "generating",
        "ready",
    ], gen_states
    assert [state["phase"] for state in gen_states] == ["prefill", "decode", "idle"], (
        gen_states
    )
    assert gen_states[-1]["message"] is None, gen_states


def step_load_during_generate_cancels_once(bridge: BridgeSession) -> None:
    """A `load` during a generation ends that request exactly once."""
    ensure_loaded(bridge)
    marker = bridge.event_count()
    assert bridge.request(
        "generate", prompt=LONG_PROMPT, max_tokens=LONG_MAX_TOKENS, request=61
    )["result"] == {"request": 61}
    time.sleep(0.05)

    loaded = bridge.request("load", model=OTHER_MODEL_ID)
    assert loaded["ok"] is True, loaded
    assert loaded["result"]["model"] == OTHER_MODEL_ID, loaded

    record = bridge.wait_for(
        "request_end", after=marker, predicate=lambda data: data["request"] == 61
    )
    assert record["finishReason"] == "cancel", record

    # Usable afterwards: the replacement model really generates.
    marker = bridge.event_count()
    assert bridge.request(
        "generate", prompt="Say hi.", max_tokens=8, request=62
    )["result"] == {"request": 62}
    follow_up = bridge.wait_for(
        "request_end", after=marker, predicate=lambda data: data["request"] == 62
    )
    assert follow_up["finishReason"] in ("length", "stop"), follow_up
    assert follow_up["genTokens"] > 0, follow_up
    assert follow_up["model"] == OTHER_MODEL_ID, follow_up

    ends = [entry for entry in bridge.events("request_end") if entry["request"] == 61]
    assert len(ends) == 1, ends


def step_unload_during_generate_cancels_once(bridge: BridgeSession) -> None:
    """An `unload` during a generation ends that request exactly once."""
    ensure_loaded(bridge)
    marker = bridge.event_count()
    assert bridge.request(
        "generate", prompt=LONG_PROMPT, max_tokens=LONG_MAX_TOKENS, request=71
    )["result"] == {"request": 71}
    time.sleep(0.05)

    assert bridge.request("unload")["result"] == {}
    assert bridge.metrics_after(bridge.event_count())["model"] is None

    record = bridge.wait_for(
        "request_end", after=marker, predicate=lambda data: data["request"] == 71
    )
    assert record["finishReason"] == "cancel", record

    # The runner is usable again once a model is loaded back.
    ensure_loaded(bridge)
    marker = bridge.event_count()
    assert bridge.request(
        "generate", prompt="Say hi.", max_tokens=8, request=72
    )["result"] == {"request": 72}
    follow_up = bridge.wait_for(
        "request_end", after=marker, predicate=lambda data: data["request"] == 72
    )
    assert follow_up["finishReason"] in ("length", "stop"), follow_up
    assert follow_up["genTokens"] > 0, follow_up

    ends = [entry for entry in bridge.events("request_end") if entry["request"] == 71]
    assert len(ends) == 1, ends


def _generate_record(
    bridge: BridgeSession, prompt: str, **fields: Any
) -> Dict[str, Any]:
    """One `generate` round trip, returning the single record it published."""
    request_id = next(_CHAT_REQUEST_IDS)
    marker = bridge.event_count()
    reply = bridge.request(
        "generate",
        timeout=60.0,
        prompt=prompt,
        max_tokens=CHAT_MAX_TOKENS,
        request=request_id,
        **fields,
    )
    assert reply["ok"] is True, reply
    assert reply["result"] == {"request": request_id}, reply
    return bridge.wait_for(
        "request_end",
        after=marker,
        timeout=60.0,
        predicate=lambda data: data["request"] == request_id,
    )


def _assert_working_record(record: Dict[str, Any]) -> None:
    assert record["model"] == MODEL_ID, record
    assert record["promptTokens"] > 0, record
    assert record["genTokens"] > 0, record
    assert record["ttftMs"] > 0, record
    assert record["finishReason"] in ("length", "stop"), record


_TOKENIZER: Any = None


def _expected_prompt_tokens() -> Dict[str, int]:
    """What the model's own tokenizer says the two prompt forms cost.

    Loaded once, straight from the local files the catalog reported, so the
    counts are the tokenizer's own answer and not a copy of the bridge's.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        models = session().request("catalog")["result"]["models"]
        path = Path(next(model["path"] for model in models if model["id"] == MODEL_ID))
        assert (path / "tokenizer.json").is_file(), path
        # A source checkout of mlx-lm may not be installed; the app root is where
        # it would live. Harmless when mlx-lm came from a wheel.
        if (APP_ROOT / "mlx_lm").is_dir() and str(APP_ROOT) not in sys.path:
            sys.path.insert(0, str(APP_ROOT))
        from mlx_lm.tokenizer_utils import load as load_tokenizer

        _TOKENIZER = load_tokenizer(path)
    tokenizer = _TOKENIZER
    # mlx-lm's own rule for the verbatim prompt (see stream_generate).
    add_special = tokenizer.bos_token is None or not PROMPT.startswith(tokenizer.bos_token)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {
        "raw": len(tokenizer.encode(PROMPT, add_special_tokens=add_special)),
        "chat": len(tokenizer.encode(rendered, add_special_tokens=add_special)),
    }


def step_generate_chat_renders_the_template(bridge: BridgeSession) -> None:
    """`chat: true` prefills the chat template's rendering of the prompt.

    The Qwen3 template wraps the message in special tokens, so the identical
    text costs strictly more prompt tokens than the raw prompt does -- and
    exactly what the model's own tokenizer says the rendered template costs.
    """
    ensure_loaded(bridge)
    raw = _generate_record(bridge, PROMPT)
    chat = _generate_record(bridge, PROMPT, chat=True)
    _assert_working_record(raw)
    _assert_working_record(chat)

    expected = _expected_prompt_tokens()
    print(
        f"promptTokens for {PROMPT!r}: raw={raw['promptTokens']} chat={chat['promptTokens']} "
        f"(tokenizer: raw={expected['raw']} chat={expected['chat']})"
    )
    assert chat["promptTokens"] > raw["promptTokens"], (chat, raw)
    assert raw["promptTokens"] == expected["raw"], (raw["promptTokens"], expected)
    assert chat["promptTokens"] == expected["chat"], (chat["promptTokens"], expected)

    # The chat path streams exactly like the raw one: `token` events for its own
    # request id, with timestamps that never go backwards and the record's own
    # count on the last one.
    tokens = [
        event for event in bridge.events("token") if event["request"] == chat["request"]
    ]
    assert tokens, chat
    assert tokens[-1]["index"] == chat["genTokens"], (tokens[-1], chat)
    assert all(
        tokens[index + 1]["ttsMs"] >= tokens[index]["ttsMs"]
        for index in range(len(tokens) - 1)
    ), "token timestamps went backwards"


def step_generate_without_chat_keeps_the_raw_prompt(bridge: BridgeSession) -> None:
    """The default did not change: no `chat` still prefills the prompt verbatim."""
    ensure_loaded(bridge)
    omitted = _generate_record(bridge, PROMPT)
    explicit = _generate_record(bridge, PROMPT, chat=False)
    _assert_working_record(omitted)
    _assert_working_record(explicit)

    raw_tokens = _expected_prompt_tokens()["raw"]
    print(
        f"promptTokens: omitted={omitted['promptTokens']} chat=False={explicit['promptTokens']} "
        f"(raw prompt tokenizes to {raw_tokens})"
    )
    assert omitted["promptTokens"] == explicit["promptTokens"], (omitted, explicit)
    assert omitted["promptTokens"] == raw_tokens, (omitted["promptTokens"], raw_tokens)


def step_generate_chat_rejects_the_empty_raw_prompt(bridge: BridgeSession) -> None:
    """Validation still runs on the raw prompt, not on the rendered one."""
    ensure_loaded(bridge)
    marker = bridge.event_count()
    request_id = next(_CHAT_REQUEST_IDS)
    reply = bridge.request(
        "generate",
        prompt="   ",
        max_tokens=CHAT_MAX_TOKENS,
        request=request_id,
        chat=True,
    )
    assert reply["ok"] is False, reply
    assert "non-empty prompt" in reply["error"], reply

    bridge.request("unload")  # drains the runner's queue
    ends = [
        entry
        for entry in bridge.event_log("request_end", marker)
        if entry["request"] == request_id
    ]
    assert ends == [], ends


def step_generate_chat_junk_values(bridge: BridgeSession) -> None:
    """A junk `chat` value is only ever read for its truthiness."""
    ensure_loaded(bridge)
    raw = _generate_record(bridge, PROMPT)
    truthy = [_generate_record(bridge, PROMPT, chat=value) for value in (1, "yes")]
    falsy = [_generate_record(bridge, PROMPT, chat=value) for value in (None, {})]
    for record in truthy + falsy:
        _assert_working_record(record)

    print(
        f"promptTokens: raw={raw['promptTokens']} chat=1 {truthy[0]['promptTokens']} "
        f"chat='yes' {truthy[1]['promptTokens']} chat=None {falsy[0]['promptTokens']} "
        f"chat={{}} {falsy[1]['promptTokens']}"
    )
    assert truthy[0]["promptTokens"] == truthy[1]["promptTokens"], truthy
    assert truthy[0]["promptTokens"] > raw["promptTokens"], (truthy[0], raw)
    for record in falsy:
        assert record["promptTokens"] == raw["promptTokens"], (record, raw)


def step_generate_chat_without_a_model(bridge: BridgeSession) -> None:
    """`chat` does not bypass the missing-model contract."""
    bridge.request("unload")
    assert bridge.metrics_after(bridge.event_count())["model"] is None

    marker = bridge.event_count()
    request_id = next(_CHAT_REQUEST_IDS)
    reply = bridge.request(
        "generate",
        prompt=PROMPT,
        max_tokens=CHAT_MAX_TOKENS,
        request=request_id,
        chat=True,
    )
    assert reply["ok"] is False, reply
    assert "no model" in reply["error"], reply

    # `unload` runs on the runner's own thread, so a generation that had been
    # queued anyway would have published its record before this returns.
    bridge.request("unload")
    ends = [
        entry
        for entry in bridge.event_log("request_end", marker)
        if entry["request"] == request_id
    ]
    assert ends == [], ends


# MARK: - Machine-wide memory (Activity Monitor's numbers)

#: The bridge's sample and the test's own `vm_stat` read are a moment apart on a
#: live 16 GB machine: free and cached pages move with every app, and the
#: compressor rewrites pages between the two reads. 512 MB (~3% of RAM) absorbs
#: that drift and still catches a wrong definition — leaving `cached` out of
#: `systemUsedBytes` understates it by gigabytes, and reporting pages rather
#: than bytes is off by 16384x.
SYSTEM_MEMORY_TOLERANCE = 512 * MEGABYTE
#: Swap is read twice, a moment apart; it moves in whole pages while the test
#: runs. 1 GB (~8% of this machine's 13 GB of swap, or all of a small one) is
#: slack for that, and a pages-for-bytes mix-up is off by four orders of
#: magnitude.
SWAP_TOLERANCE = 1024 * MEGABYTE

SYSCTL = shutil.which("sysctl") or "/usr/sbin/sysctl"
VM_STAT = shutil.which("vm_stat") or "/usr/bin/vm_stat"

#: Every machine-wide field the bridge must carry, in PROTOCOL.md's order.
SYSTEM_MEMORY_KEYS = (
    "systemUsedBytes",
    "systemAppBytes",
    "systemWiredBytes",
    "systemCompressedBytes",
    "systemCachedBytes",
    "systemSwapBytes",
)


def _sysctl_value(key: str) -> str:
    out = subprocess.run(
        [SYSCTL, "-n", key], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _vm_stat_pages() -> Dict[str, int]:
    """`vm_stat`'s per-label page counts, read from the OS by this process."""
    out = subprocess.run([VM_STAT], capture_output=True, text=True, check=True)
    counts: Dict[str, int] = {}
    for line in out.stdout.splitlines():
        label, sep, value = line.partition(":")
        digits = value.strip().rstrip(".").strip()
        if sep and digits.isdigit():
            counts[label.strip()] = int(digits)
    return counts


def _os_system_memory() -> Dict[str, int]:
    """The bridge's figures, recomputed from the OS without the bridge."""
    page = int(_sysctl_value("vm.pagesize"))
    physical = int(_sysctl_value("hw.memsize"))
    pages = _vm_stat_pages()
    cached = (pages["File-backed pages"] + pages["Pages purgeable"]) * page
    return {
        "pagesize": page,
        "memsize": physical,
        "used": physical - pages["Pages free"] * page - cached,
        "cached": cached,
        "app": pages["Anonymous pages"] * page,
        "wired": pages["Pages wired down"] * page,
        "compressed": pages["Pages occupied by compressor"] * page,
    }


def _os_swap_used() -> int:
    """`sysctl -n vm.swapusage`'s used figure in bytes, honouring its scale."""
    tokens = _sysctl_value("vm.swapusage").replace("=", " ").split()
    value = tokens[tokens.index("used") + 1]
    unit = value[-1].upper()
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}.get(unit)
    number = value[:-1] if scale is not None else value
    return int(float(number) * (scale if scale is not None else 1))


def step_system_memory_matches_the_os(bridge: BridgeSession) -> None:
    """The six fields are the machine's own numbers, not this process's.

    Compared against an independent read by the test process: `sysctl -n
    hw.memsize`, `sysctl -n vm.pagesize`, a parsed `vm_stat` and `sysctl -n
    vm.swapusage`. Both sides are live, hence `SYSTEM_MEMORY_TOLERANCE`.
    """
    os_memory = _os_system_memory()
    sample = bridge.metrics_after(bridge.event_count())
    assert sample["memoryTotalBytes"] == os_memory["memsize"], (sample, os_memory)
    for field, key in (("systemUsedBytes", "used"), ("systemCachedBytes", "cached")):
        measured = sample[field]
        expected = os_memory[key]
        assert abs(measured - expected) <= SYSTEM_MEMORY_TOLERANCE, (
            f"{field}: bridge {measured} vs vm_stat {expected} "
            f"(tolerance {SYSTEM_MEMORY_TOLERANCE})"
        )
    os_swap = _os_swap_used()
    assert abs(sample["systemSwapBytes"] - os_swap) <= max(
        SWAP_TOLERANCE, os_swap // 8
    ), f"systemSwapBytes: bridge {sample['systemSwapBytes']} vs sysctl {os_swap}"


def step_system_memory_invariants(bridge: BridgeSession) -> None:
    """One sample: plausible, self-consistent, and tied to physical RAM."""
    sample = bridge.metrics_after(bridge.event_count())
    total = sample["memoryTotalBytes"]
    assert total == int(_sysctl_value("hw.memsize")), sample
    used = sample["systemUsedBytes"]
    assert 0 < used < total, sample
    assert sample["systemAppBytes"] > 0, sample
    assert sample["systemWiredBytes"] > 0, sample
    assert sample["systemCachedBytes"] > 0, sample
    assert used + sample["systemCachedBytes"] <= total, sample
    assert sample["systemCompressedBytes"] >= 0, sample
    assert sample["systemSwapBytes"] >= 0, sample
    # The four page-derived fields are separate reads, not one value copied
    # across: on any real machine at least two of them differ.
    distinct = {
        sample["systemAppBytes"],
        sample["systemWiredBytes"],
        sample["systemCompressedBytes"],
        sample["systemCachedBytes"],
    }
    assert len(distinct) > 1, sample


def step_hello_and_metrics_agree_on_system_memory(bridge: BridgeSession) -> None:
    """A decoder can rely on all six keys in `hello` and in every `metrics`."""
    hardware = bridge.request("hello")["result"]["hardware"]
    first = bridge.wait_for("metrics", timeout=30)  # the session's first sample
    for key in SYSTEM_MEMORY_KEYS:
        assert key in hardware and isinstance(hardware[key], int), hardware
        assert key in first and isinstance(first[key], int), first
    assert hardware["totalBytes"] == first["memoryTotalBytes"], (hardware, first)


def step_system_memory_is_live(bridge: BridgeSession) -> None:
    """Two samples 1.5 s apart differ: a live read, not a cached constant."""
    first = bridge.metrics_after(bridge.event_count())
    time.sleep(1.5)
    second = bridge.metrics_after(bridge.event_count())
    before = tuple(first[key] for key in SYSTEM_MEMORY_KEYS)
    after = tuple(second[key] for key in SYSTEM_MEMORY_KEYS)
    assert before != after, f"system memory never moved: {before}"


STEPS: List[tuple[str, Callable[[BridgeSession], None]]] = [
    ("hello", step_hello),
    ("catalog", step_catalog),
    ("load", step_load),
    ("catalog tracks last used model", step_catalog_tracks_last_used),
    ("generate 24 tokens", step_generate),
    ("cancel a request", step_cancel),
    ("http api", step_http),
    ("unknown command", step_unknown_command),
    ("unload", step_unload),
    ("replace model", step_replace_model),
    ("process rss is live", step_process_rss_is_live),
    ("keep-alive after unknown POST", step_keep_alive_survives_unknown_post),
    ("stop_serve releases the port", step_stop_serve_releases_the_port),
    ("cancel a queued request", step_cancel_reports_a_queued_request),
    ("ttft includes the queue wait", step_ttft_includes_the_queue_wait),
    ("state events", step_state_events_follow_the_protocol),
    ("load during generate", step_load_during_generate_cancels_once),
    ("unload during generate", step_unload_during_generate_cancels_once),
    ("generate with chat", step_generate_chat_renders_the_template),
    ("generate without chat", step_generate_without_chat_keeps_the_raw_prompt),
    ("generate chat rejects an empty prompt", step_generate_chat_rejects_the_empty_raw_prompt),
    ("generate with junk chat values", step_generate_chat_junk_values),
    ("generate with chat and no model", step_generate_chat_without_a_model),
    ("system memory matches the os", step_system_memory_matches_the_os),
    ("system memory invariants", step_system_memory_invariants),
    ("hello and metrics agree", step_hello_and_metrics_agree_on_system_memory),
    ("system memory is live", step_system_memory_is_live),
]


# MARK: - pytest entry points


def test_hello_reports_real_hardware() -> None:
    step_hello(session())


def test_catalog_lists_real_local_models() -> None:
    step_catalog(session())


def test_load_raises_active_memory() -> None:
    step_load(session())


def test_catalog_sorts_by_last_used() -> None:
    step_catalog_tracks_last_used(session())


def test_generate_produces_real_tokens_and_record() -> None:
    step_generate(session())


def test_cancel_ends_the_request_once() -> None:
    step_cancel(session())


def test_http_api_serves_the_loaded_model() -> None:
    step_http(session())


def test_unknown_command_fails() -> None:
    step_unknown_command(session())


def test_unload_frees_memory() -> None:
    step_unload(session())


def test_loading_another_model_replaces_it() -> None:
    step_replace_model(session())


def test_process_rss_is_live_and_falls_on_unload() -> None:
    step_process_rss_is_live(session())


def test_keep_alive_survives_post_to_unknown_path() -> None:
    step_keep_alive_survives_unknown_post(session())


def test_stop_serve_releases_the_port() -> None:
    step_stop_serve_releases_the_port(session())


def test_cancel_reports_a_queued_request() -> None:
    step_cancel_reports_a_queued_request(session())


def test_ttft_includes_the_queue_wait() -> None:
    step_ttft_includes_the_queue_wait(session())


def test_state_events_follow_the_protocol() -> None:
    step_state_events_follow_the_protocol(session())


def test_load_during_generate_cancels_once() -> None:
    step_load_during_generate_cancels_once(session())


def test_unload_during_generate_cancels_once() -> None:
    step_unload_during_generate_cancels_once(session())


def test_generate_chat_renders_the_tokenizer_template() -> None:
    step_generate_chat_renders_the_template(session())


def test_generate_without_chat_keeps_the_raw_prompt() -> None:
    step_generate_without_chat_keeps_the_raw_prompt(session())


def test_generate_chat_rejects_the_empty_raw_prompt() -> None:
    step_generate_chat_rejects_the_empty_raw_prompt(session())


def test_generate_chat_junk_values_are_only_truthy() -> None:
    step_generate_chat_junk_values(session())


def test_generate_chat_without_a_model_fails() -> None:
    step_generate_chat_without_a_model(session())


def test_system_memory_matches_the_os() -> None:
    step_system_memory_matches_the_os(session())


def test_system_memory_invariants() -> None:
    step_system_memory_invariants(session())


def test_hello_and_metrics_agree_on_system_memory() -> None:
    step_hello_and_metrics_agree_on_system_memory(session())


def test_system_memory_is_live() -> None:
    step_system_memory_is_live(session())


# MARK: - Plain-script entry point


def _report_metrics_line(bridge: BridgeSession) -> Optional[str]:
    for line in bridge.raw_lines('"event":"metrics"', '"decodeTps":'):
        try:
            data = json.loads(line)["data"]
        except (ValueError, KeyError, TypeError):
            continue
        if data.get("decodeTps", 0) > 0:
            return line
    return None


def main() -> int:
    failures = 0
    for name, step in STEPS:
        started = time.time()
        try:
            step(session())
        except Exception as exc:  # noqa: BLE001 - a report, not a handler
            failures += 1
            print(f"FAIL {name}: {exc}")
            print(session().stderr_text()[-2000:])
        else:
            print(f"ok   {name} ({time.time() - started:.2f}s)")

    bridge = session()
    print("\nreal metrics event line (decodeTps > 0):")
    print(_report_metrics_line(bridge) or "(none captured)")
    print("\nreal request_end event line:")
    records = bridge.raw_lines('"event":"request_end"')
    print(records[0] if records else "(none captured)")
    print(f"\nstdout lines captured: {len(bridge.raw_lines())}")
    bridge.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

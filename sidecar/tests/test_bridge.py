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
import dataclasses
import http.client
import itertools
import json
import os
import re
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
#: The tool loop's request ids, clear of every other scenario's.
TOOL_REQUEST_BASE = 9500
_TOOL_REQUEST_IDS = itertools.count(TOOL_REQUEST_BASE)
REQUEST_ID = 7
MEGABYTE = 1024 * 1024

#: The tree the bridge's file tools are confined to ($SLAM_LM_TOOL_ROOT): a real
#: temp directory holding a known file and a nested one, so `read_file`,
#: `list_directory` and `search_files` are exercised offline and deterministically.
TOOL_FILE_NAME = "note.txt"
TOOL_FILE_TEXT = "The secret word is platypus.\n"
TOOL_NESTED_DIR = "nested"
TOOL_NESTED_NAME = "deep.txt"
TOOL_NESTED_TEXT = "a nested file\n"
#: A path that is always outside that tree.
OUTSIDE_PATH = "/etc/hosts"

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

    def __init__(
        self,
        state_path: Path,
        tool_root: Optional[Path] = None,
        env_extra: Optional[Dict[str, str]] = None,
    ):
        env = dict(os.environ)
        if env_extra:
            env.update(env_extra)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{BRIDGE_ROOT}{os.pathsep}{existing}" if existing else str(BRIDGE_ROOT)
        )
        env["SLAM_LM_STATE"] = str(state_path)
        # The file tools are confined to this tree, so every path the tests
        # exercise is real, local and deterministic — /etc/hosts is outside it.
        env["SLAM_LM_TOOL_ROOT"] = str(tool_root if tool_root is not None else tool_root_path())
        self.state_path = state_path
        self.tool_root = Path(env["SLAM_LM_TOOL_ROOT"])
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


def _temp_dir() -> Path:
    """One disposable temp directory for the state file and the tool tree."""
    global _TEMP_DIR
    if _TEMP_DIR is None:
        _TEMP_DIR = tempfile.TemporaryDirectory(prefix="slam-lm-test-")
        atexit.register(_TEMP_DIR.cleanup)
    return Path(_TEMP_DIR.name)


def tool_root_path() -> Path:
    """The real temp tree the bridge's file tools are confined to."""
    root = _temp_dir() / "tools"
    if not root.exists():
        root.mkdir()
        (root / TOOL_FILE_NAME).write_text(TOOL_FILE_TEXT, encoding="utf-8")
        nested = root / TOOL_NESTED_DIR
        nested.mkdir()
        (nested / TOOL_NESTED_NAME).write_text(TOOL_NESTED_TEXT, encoding="utf-8")
    return root


def session() -> BridgeSession:
    global _SESSION
    if _SESSION is None:
        _SESSION = BridgeSession(_temp_dir() / "state.json")
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
    marker = bridge.event_count()
    bridge.request("unload")
    # The sampler runs at 5 Hz while the unload is still releasing memory, so a
    # sample captured mid-unload can arrive after it finished. Take the first
    # sample that reports the unloaded state, not whichever lands next.
    after = bridge.wait_for(
        "metrics",
        after=marker,
        timeout=10.0,
        predicate=lambda data: data["model"] is None,
    )
    assert after["status"] == "idle", after

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


# MARK: - Tools

#: Every tool round is given an explicit budget: the calls below are short, so
#: this keeps each request bounded instead of letting a chatty model fill the
#: context window. It is generous because a thinking model spends tokens on its
#: reasoning before it emits the call (200-280 for these prompts, and the odd
#: run reasons further than that).
TOOL_MAX_TOKENS = 512

#: How many times a forced call is asked for. The model is greedy, but MLX is
#: not bit-deterministic, so the odd run reasons its way past the budget without
#: emitting the call; asking again is what keeps these tests stable.
TOOL_ATTEMPTS = 5
#: A plain request that the model answers itself gets fewer retries: it either
#: reaches for the tool or it does not, and that is the thing under test.
TOOL_CALL_ATTEMPTS = 3

#: PROTOCOL.md's exact keys for a `tool` event, in either phase.
TOOL_EVENT_KEYS = {
    "request",
    "round",
    "phase",
    "name",
    "arguments",
    "ok",
    "summary",
    "detail",
}


def _tool_request(
    bridge: BridgeSession,
    prompt: str,
    request_id: int,
    *,
    timeout: float = 120.0,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """One `tools: true` request: its record, its `tool` events, its tokens."""
    marker = bridge.event_count()
    reply = bridge.request(
        "generate",
        timeout=60.0,
        prompt=prompt,
        request=request_id,
        tools=True,
        max_tokens=TOOL_MAX_TOKENS,
    )
    assert reply["ok"] is True, reply
    assert reply["result"] == {"request": request_id}, reply
    record = bridge.wait_for(
        "request_end",
        after=marker,
        timeout=timeout,
        predicate=lambda data: data["request"] == request_id,
    )
    events = [
        event for event in bridge.event_log("tool", marker) if event["request"] == request_id
    ]
    tokens = [
        event for event in bridge.events("token") if event["request"] == request_id
    ]
    return record, events, tokens


def _tool_call(
    bridge: BridgeSession, prompts: List[str]
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Send each prompt in turn; return the first result that asked for a tool."""
    result: Optional[tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]] = None
    for prompt in prompts:
        request_id = next(_TOOL_REQUEST_IDS)
        result = _tool_request(bridge, prompt, request_id)
        if any(event["phase"] == "call" for event in result[1]):
            break
    assert result is not None
    return result


def _asked_tool_call(
    bridge: BridgeSession, prompt: str, *, attempts: int = TOOL_CALL_ATTEMPTS
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """A plain request: the model has to decide for itself to use a tool."""
    return _tool_call(bridge, [prompt] * attempts)


def _forced_tool_call(
    bridge: BridgeSession,
    instruction: str,
    call_json: str,
    *,
    attempts: int = TOOL_ATTEMPTS,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Ask the model for one specific call until it makes one.

    The call is echoed back in the format the model is shown, which is what
    makes a 0.6B model reach for a tool it would otherwise only describe.
    Nothing about the *execution* is faked: the bridge parses the model's own
    output, runs the real tool against the real file system and reports what it
    returned.
    """
    # A 0.6B model can reason its way past the budget without ever emitting the
    # call, and repeating the same wording can reproduce that. The shorter ask is
    # the fallback: the model still produces the call itself.
    asked = f"{instruction} Reply with only this tool call and nothing else: {call_json}"
    fallback = f"Reply with only this tool call and nothing else: {call_json}"
    return _tool_call(bridge, [asked] + [fallback] * (attempts - 1))


def _tool_calls(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [event for event in events if event["phase"] == "call"]


def _tool_results(
    events: List[Dict[str, Any]], name: Optional[str] = None
) -> List[Dict[str, Any]]:
    return [
        event
        for event in events
        if event["phase"] == "result" and (name is None or event["name"] == name)
    ]


def step_tools_run_a_real_file_call(bridge: BridgeSession) -> None:
    """`tools: true` reports the call and the result, and the tool really ran."""
    ensure_loaded(bridge)
    record, events, tokens = _forced_tool_call(
        bridge,
        f"Call read_file on {TOOL_FILE_NAME}.",
        f'<tool_call>{{"name": "read_file", "arguments": {{"path": "{TOOL_FILE_NAME}"}}}}</tool_call>',
    )
    calls = _tool_calls(events)
    results = _tool_results(events)
    assert calls, f"the model never asked for a tool: {record}"
    assert len(results) == len(calls), events

    call = calls[0]
    assert set(call) == TOOL_EVENT_KEYS, call
    assert call["request"] == record["request"], (call, record)
    assert call["round"] == 1, call
    assert call["name"] == "read_file", call
    # The model may spell out a documented default (`max_bytes`) or leave it
    # out; what it may not do is invent an argument.
    arguments = call["arguments"]
    assert isinstance(arguments, dict), call
    assert arguments.get("path") == TOOL_FILE_NAME, call
    assert set(arguments) <= {"path", "max_bytes"}, call
    assert call["ok"] is True, call
    assert TOOL_FILE_NAME in call["summary"], call
    assert call["detail"] is None, call

    result = results[0]
    assert set(result) == TOOL_EVENT_KEYS, result
    assert result["phase"] == "result", result
    assert result["round"] == call["round"], (result, call)
    assert result["name"] == call["name"], (result, call)
    assert result["arguments"] is None, result
    assert result["ok"] is True, result
    # The tool read the real file: its exact bytes are in the detail.
    assert TOOL_FILE_TEXT.strip() in (result["detail"] or ""), result
    assert str(len(TOOL_FILE_TEXT.encode("utf-8"))) in result["summary"], result

    # The record counts every executed call, and it finished cleanly.
    assert record["toolCalls"] == len(calls), (record, calls)
    assert record["finishReason"] in ("length", "stop"), record
    assert record["promptTokens"] > 0 and record["genTokens"] > 0, record

    # The streamed answer is the *final* round's text: a call round's
    # scaffolding is reported as a `tool` event, never as answer text.
    answer = "".join(token["text"] for token in tokens)
    assert answer.strip(), f"the answer round streamed nothing: {record}"
    assert "<tool_call>" not in answer, answer
    assert tokens[-1]["index"] == record["genTokens"], (tokens[-1], record)
    assert all(
        tokens[index + 1]["ttsMs"] >= tokens[index]["ttsMs"]
        for index in range(len(tokens) - 1)
    ), "token timestamps went backwards"
    print(f"read_file result: {result['summary']}; answer {answer[:80]!r}")


def step_tools_list_and_describe_real_paths(bridge: BridgeSession) -> None:
    """`list_directory` and `file_info` return the real entries and metadata."""
    ensure_loaded(bridge)
    _, events, _ = _forced_tool_call(
        bridge,
        "Call list_directory with the path \".\".",
        '<tool_call>{"name": "list_directory", "arguments": {"path": "."}}</tool_call>',
    )
    listings = _tool_results(events, "list_directory")
    assert listings, f"the model never called list_directory: {events}"
    listing = listings[0]
    assert listing["ok"] is True, listing
    detail = listing["detail"] or ""
    assert TOOL_FILE_NAME in detail, listing
    assert TOOL_NESTED_DIR in detail, listing
    assert "file" in detail and "directory" in detail, listing
    print(f"list_directory: {listing['summary']}")

    _, events, _ = _forced_tool_call(
        bridge,
        f"Call file_info on {TOOL_FILE_NAME}.",
        f'<tool_call>{{"name": "file_info", "arguments": {{"path": "{TOOL_FILE_NAME}"}}}}</tool_call>',
    )
    infos = _tool_results(events, "file_info")
    assert infos, f"the model never called file_info: {events}"
    info = infos[0]
    assert info["ok"] is True, info
    detail = info["detail"] or ""
    assert "kind: file" in detail, info
    assert f"bytes: {len(TOOL_FILE_TEXT.encode('utf-8'))}" in detail, info
    stamp = detail.split("modified: ")[-1].strip()
    assert "T" in stamp and ("+" in stamp or stamp.endswith("Z")), detail
    print(f"file_info: {info['summary']}")


def step_tools_find_a_nested_file(bridge: BridgeSession) -> None:
    """`search_files` globs recursively inside the tool root."""
    ensure_loaded(bridge)
    _, events, _ = _forced_tool_call(
        bridge,
        'Call search_files with pattern "**/*.txt".',
        '<tool_call>{"name": "search_files", "arguments": {"pattern": "**/*.txt"}}</tool_call>',
    )
    results = _tool_results(events, "search_files")
    assert results, f"the model never called search_files: {events}"
    result = results[0]
    assert result["ok"] is True, result
    detail = result["detail"] or ""
    # The nested file is one level down: only a recursive search finds it.
    assert TOOL_NESTED_NAME in detail, result
    assert f"{TOOL_NESTED_DIR}/{TOOL_NESTED_NAME}" in detail, result
    assert TOOL_FILE_NAME in detail, result
    print(f"search_files: {result['summary']}")


def step_tools_refuse_a_path_outside_the_root(bridge: BridgeSession) -> None:
    """A path outside $SLAM_LM_TOOL_ROOT is refused, and the request survives."""
    ensure_loaded(bridge)
    record, events, _ = _forced_tool_call(
        bridge,
        f"Call read_file with exactly this path: {OUTSIDE_PATH}.",
        f'<tool_call>{{"name": "read_file", "arguments": {{"path": "{OUTSIDE_PATH}"}}}}</tool_call>',
    )
    results = _tool_results(events)
    assert results, f"the model never asked for a tool: {events}"
    refused = results[0]
    assert refused["ok"] is False, refused
    assert "outside the tool root" in refused["summary"], refused
    assert OUTSIDE_PATH in refused["summary"], refused
    assert refused["arguments"] is None, refused

    # The refusal is fed back, not fatal: the request still ends with a record.
    assert record["finishReason"] in ("length", "stop"), record
    assert record["toolCalls"] == len(_tool_calls(events)), (record, events)
    print(f"refused: {refused['summary']}")


def step_tools_report_an_unknown_tool(bridge: BridgeSession) -> None:
    """A tool the model invented is an `ok:false` result, not a crash."""
    ensure_loaded(bridge)
    record, events, _ = _forced_tool_call(
        bridge,
        "Call the delete_file tool.",
        f'<tool_call>{{"name": "delete_file", "arguments": {{"path": "{TOOL_FILE_NAME}"}}}}</tool_call>',
    )
    results = _tool_results(events, "delete_file")
    assert results, f"the model never called the invented tool: {events}"
    result = results[0]
    assert result["ok"] is False, result
    assert "unknown tool" in result["summary"], result
    assert "delete_file" in (result["detail"] or ""), result

    # Nothing was written, and the bridge answered the request regardless.
    assert (bridge.tool_root / TOOL_FILE_NAME).read_text(encoding="utf-8") == TOOL_FILE_TEXT
    assert record["finishReason"] in ("length", "stop"), record
    assert record["toolCalls"] == len(_tool_calls(events)), (record, events)
    print(f"unknown tool: {result['summary']}")


def step_tools_search_the_real_web(bridge: BridgeSession) -> None:
    """`web_search` reaches the real web, or fails naming every provider."""
    ensure_loaded(bridge)
    _, events, _ = _forced_tool_call(
        bridge,
        'Call web_search with the query "mlx lm" and max_results 3.',
        '<tool_call>{"name": "web_search", "arguments": {"query": "mlx lm", "max_results": 3}}</tool_call>',
    )
    results = _tool_results(events, "web_search")
    assert results, f"the model never called web_search: {events}"
    successes = [result for result in results if result["ok"]]
    if successes:
        # Success means real, non-empty results from a named provider.
        result = successes[0]
        detail = result["detail"] or ""
        assert detail.startswith("provider: "), result
        assert "\n1. " in detail, result
        assert "http" in detail, result
        assert result["summary"] != "0 results", result
        print(f"web_search: {result['summary']}\n{detail[:300]}")
        return

    # Every call failed: the detail must name each provider with its real
    # reason, and no result may be dressed up as an empty success. DuckDuckGo
    # rate-limits this machine, so a failed ladder is a legitimate outcome.
    result = results[-1]
    detail = result["detail"] or ""
    reasons = [line for line in detail.splitlines() if line.startswith("- ")]
    assert result["summary"] == "every search provider failed", result
    assert len(reasons) > 1, result
    assert all(reason.split(": ", 1)[1].strip() for reason in reasons), result
    print(f"web_search: {result['summary']}\n{detail}")


def step_http_accepts_tools(bridge: BridgeSession) -> None:
    """`tools` works over HTTP too: the completion runs the real tool loop.

    The answer arrives through the same sink as any other completion, so a tool
    request's final round is streamed to the HTTP client rather than swallowed
    with the scaffolding.
    """
    ensure_loaded(bridge)
    port = _free_port()
    base = bridge.request("serve", port=port)["result"]["url"]
    marker = bridge.event_count()
    _, completion = _http(
        f"{base}/v1/chat/completions",
        {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": f"Call read_file on {TOOL_FILE_NAME}. Reply with only this tool call "
                    "and nothing else: "
                    f'<tool_call>{{"name": "read_file", "arguments": {{"path": "{TOOL_FILE_NAME}"}}}}</tool_call>',
                }
            ],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
            "max_tokens": TOOL_MAX_TOKENS,
        },
    )
    record = bridge.wait_for(
        "request_end",
        after=marker,
        timeout=120.0,
        predicate=lambda data: data["request"] >= 1_000_000,
    )
    events = [
        event for event in bridge.event_log("tool", marker) if event["request"] == record["request"]
    ]
    assert [event for event in events if event["phase"] == "call"], events
    assert [event for event in events if event["phase"] == "result"], events
    assert record["toolCalls"] == len([e for e in events if e["phase"] == "call"]), (
        record,
        events,
    )
    # Every round's tokens are in the usage; the visible answer is the last one.
    assert completion["usage"]["completion_tokens"] == record["genTokens"], (
        completion,
        record,
    )
    content = completion["choices"][0]["message"]["content"]
    assert content.strip(), completion
    assert "<tool_call>" not in content, content
    print(f"http tools: toolCalls={record['toolCalls']} completion={content[:60]!r}")


def step_tools_offer_the_tools_through_the_chat_template(
    bridge: BridgeSession,
) -> None:
    """A plain request makes these models call a tool, because they are offered.

    The tools go to the tokenizer's chat template as `tools=` schemas, which is
    what puts them in the model's system turn. Measured on this model: with the
    written instructions only, the same request narrates a fake search instead
    of calling; with the template's own tools block it emits the real call.
    """
    ensure_loaded(bridge)
    record, events, tokens = _asked_tool_call(
        bridge, "What is the latest MLX release version? Use your web_search tool."
    )
    calls = _tool_calls(events)
    assert calls, (
        "the model answered a plain request without calling a tool: "
        f"genTokens={record['genTokens']}, finishReason={record['finishReason']}"
    )
    call = calls[0]
    assert call["name"] == "web_search", calls
    query = (call["arguments"] or {}).get("query")
    assert isinstance(query, str) and query.strip(), calls
    results = _tool_results(events, "web_search")
    assert results, events
    assert isinstance(results[0]["ok"], bool), results
    print(f"offered through the template: {call['summary']} -> {results[0]['summary']}")


def step_tools_fallback_written_instructions(bridge: BridgeSession) -> None:
    """A template that ignores `tools=` still gets working tools.

    Qwen1.5's chat template has no notion of tools, so the bridge falls back to
    the written instructions and the plain `tool` role messages. The prompt is
    the fence spelling *inside* a sentence, which is what this model actually
    emits (measured: an inline ```tool block after a line of prose).
    """
    loaded = bridge.request("load", model=OTHER_MODEL_ID)["result"]
    assert loaded["model"] == OTHER_MODEL_ID, loaded
    try:
        prompt = (
            f"To read the file {TOOL_FILE_NAME} you can use the following command: "
            f'```tool {{"name": "read_file", "arguments": {{"path": "{TOOL_FILE_NAME}"}}}}```'
        )
        record, events, _ = _tool_call(bridge, [prompt] * TOOL_CALL_ATTEMPTS)
        calls = _tool_calls(events)
        assert calls, f"the model never called a tool: genTokens={record['genTokens']}"
        assert calls[0]["name"] == "read_file", calls
        assert calls[0]["arguments"] == {"path": TOOL_FILE_NAME}, calls
        results = _tool_results(events, "read_file")
        assert results, events
        assert results[0]["ok"] is True, results[0]
        # The inline fence was parsed, and the tool then read the real file.
        assert TOOL_FILE_TEXT.strip() in (results[0]["detail"] or ""), results[0]
        assert record["finishReason"] in ("length", "stop"), record
        assert record["toolCalls"] == len(calls), (record, calls)
        print(
            f"written instructions on {OTHER_MODEL_ID}: "
            f"{calls[0]['summary']} -> {results[0]['summary']}"
        )
    finally:
        ensure_loaded(bridge)


def step_generate_without_a_token_budget(bridge: BridgeSession) -> None:
    """No `max_tokens` from the app: a normal answer is not truncated.

    PROTOCOL.md has no user-facing budget, so the thing to prove is that an
    answer the model finishes on its own is reported as a natural stop — the
    panel sends no number, and nothing cuts the reply short.
    """
    ensure_loaded(bridge)
    request_id = next(_CHAT_REQUEST_IDS)
    marker = bridge.event_count()
    reply = bridge.request(
        "generate",
        timeout=120.0,
        prompt="Name the three primary colours, then stop.",
        request=request_id,
        chat=True,
    )
    assert reply["ok"] is True, reply
    assert reply["result"] == {"request": request_id}, reply
    record = bridge.wait_for(
        "request_end",
        after=marker,
        timeout=300.0,
        predicate=lambda data: data["request"] == request_id,
    )
    assert record["finishReason"] == "stop", record
    assert record["genTokens"] > 0, record
    assert record["totalMs"] > 0 and record["decodeTps"] > 0, record

    tokens = [
        event for event in bridge.events("token") if event["request"] == request_id
    ]
    assert tokens, record
    assert tokens[-1]["index"] == record["genTokens"], (tokens[-1], record)
    assert "".join(token["text"] for token in tokens).strip(), record
    print(
        f"no budget: {record['genTokens']} tokens, finishReason={record['finishReason']}, "
        f"{record['totalMs'] / 1000.0:.1f}s"
    )


def step_generate_ceiling_bounds_a_runaway(tmp_state: Path) -> None:
    """The safety ceiling is what stops a model that would not stop.

    A model can fail to emit its stop token, and the request that prompted this
    test is proof: with no ceiling the bridge generated until the context window
    was full, minutes of work and a multi-gigabyte KV cache on a 16 GB machine,
    which drove the host into swap and left the panel waiting forever.
    `SLAM_LM_MAX_TOKENS` lowers the ceiling so the bound is observable quickly.
    """
    bridge = BridgeSession(tmp_state / "state.json", env_extra={"SLAM_LM_MAX_TOKENS": "24"})
    try:
        ensure_loaded(bridge)
        request_id = next(_CHAT_REQUEST_IDS)
        marker = bridge.event_count()
        reply = bridge.request(
            "generate",
            timeout=120.0,
            prompt="Write a long essay about mountains. " * 40,
            request=request_id,
            chat=True,
        )
        assert reply["ok"] is True, reply
        record = bridge.wait_for(
            "request_end",
            after=marker,
            timeout=300.0,
            predicate=lambda data: data["request"] == request_id,
        )
        assert record["genTokens"] <= 24, record
        assert record["genTokens"] > 0, record
        assert record["finishReason"] == "length", record
        print(
            f"ceiling: stopped at {record['genTokens']} tokens "
            f"(ceiling 24), finishReason={record['finishReason']}"
        )


    finally:
        bridge.close()


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
    ("tools: a real file call", step_tools_run_a_real_file_call),
    ("tools: list and describe real paths", step_tools_list_and_describe_real_paths),
    ("tools: find a nested file", step_tools_find_a_nested_file),
    ("tools: refuse a path outside the root", step_tools_refuse_a_path_outside_the_root),
    ("tools: report an unknown tool", step_tools_report_an_unknown_tool),
    ("tools: search the real web", step_tools_search_the_real_web),
    ("tools: offered through the chat template", step_tools_offer_the_tools_through_the_chat_template),
    ("tools: written-instruction fallback", step_tools_fallback_written_instructions),
    ("http accepts tools", step_http_accepts_tools),
    ("generate with no token budget", step_generate_without_a_token_budget),
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


def test_tools_call_and_result_events_are_the_contract() -> None:
    step_tools_run_a_real_file_call(session())


def test_tools_list_directory_and_file_info_are_real() -> None:
    step_tools_list_and_describe_real_paths(session())


def test_tools_search_files_finds_a_nested_file() -> None:
    step_tools_find_a_nested_file(session())


def test_tools_refuse_a_path_outside_the_root() -> None:
    step_tools_refuse_a_path_outside_the_root(session())


def test_tools_report_an_unknown_tool() -> None:
    step_tools_report_an_unknown_tool(session())


def test_tools_web_search_hits_the_real_endpoint() -> None:
    step_tools_search_the_real_web(session())


# MARK: - The web_search provider ladder

#: A query whose answer has to come from the real web.
SEARCH_QUERY = "mlx lm apple silicon"
#: A factual query Wikipedia answers directly, and the name the article it lists
#: for it carries: `nvidia ceo` returns `Jensen Huang`, whose extract says so.
WIKIPEDIA_QUERY = "nvidia ceo"
WIKIPEDIA_NAME = "Jensen Huang"
#: Every Wikipedia result URL is that article's own canonical address.
WIKIPEDIA_PREFIX = "https://en.wikipedia.org/"
#: The two lines one rendered result occupies: `1. A title`, then its own URL.
_RESULT_TITLE_RE = re.compile(r"^\d+\. (\S.*)$", re.MULTILINE)
_RESULT_URL_RE = re.compile(r"^   (https?://\S+)$", re.MULTILINE)
#: The reasons that mean "no provider could answer *here*", rather than "the
#: provider answered and the parser saw nothing": a transport failure, or a
#: rate-limit refusal. DuckDuckGo answers this machine 202 and Brave answers 429
#: after a burst; neither is a defect in the parse.
_UNREACHABLE_REASONS = ("unreachable", "timed out", "HTTP 202", "HTTP 429")


def _tools_module() -> Any:
    """The `slam_lm_bridge.tools` module under test, imported in this process."""
    root = str(BRIDGE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import slam_lm_bridge.tools as tools

    return tools


def _pytest() -> Any:
    """`pytest`, imported late: the plain-script entry point does not need it."""
    import pytest

    return pytest


def test_web_search_answers_from_this_machine() -> None:
    """A real query returns real results, and the detail names the provider.

    The ladder is keyless HTML endpoints, and DuckDuckGo rate-limits this
    machine, so this is the test that the *capability* works here right now.
    Only no provider being able to answer at all — every reason a transport
    failure or a rate-limit refusal — is a skip; a provider that answered and
    produced nothing is a failure, because a broken parse is the defect this
    ladder exists to survive.
    """
    tools = _tools_module()
    call = tools.ToolCall("web_search", {"query": SEARCH_QUERY, "max_results": 3})
    result = tools.ToolRegistry().execute(call)
    detail = result.detail or ""

    if not result.ok:
        reasons = re.findall(r"^- ([\w-]+): (.+)$", detail, re.MULTILINE)
        assert reasons, (result.summary, detail)
        if all(any(bad in reason for bad in _UNREACHABLE_REASONS) for _, reason in reasons):
            _pytest().skip(f"no search provider can answer from this machine: {detail}")
        _pytest().fail(f"no provider produced a result: {detail}")

    header, _, results = detail.partition("\n")
    assert header.startswith("provider: "), detail
    answered = header[len("provider: ") :].strip()
    assert answered in {entry.name for entry in tools.SEARCH_PROVIDERS}, detail

    titles = _RESULT_TITLE_RE.findall(results)
    urls = _RESULT_URL_RE.findall(results)
    assert titles and urls, detail
    assert all(title.strip() for title in titles), detail
    assert all(url.startswith("http") for url in urls), detail
    # Every rendered result carries exactly one URL line, so the summary's count
    # is the number of real results the tool is claiming.
    assert result.summary == f"{len(urls)} results", (result.summary, detail)
    print(f"web_search answered by {answered}: {detail[:300]}")


def test_web_search_wikipedia_answers_a_factual_query(monkeypatch: Any) -> None:
    """The last provider is a real factual source, exercised on its own.

    Every other provider is dropped from the ladder, so the only endpoint this
    test touches is Wikipedia: the MediaWiki search API for the hits, then each
    hit's summary for a clean extract and the article's canonical URL. Wikipedia
    being unreachable *here* is a skip; a reachable Wikipedia that cannot answer
    `nvidia ceo` is a failure, because that is the provider this ladder ends on.
    """
    tools = _tools_module()
    wikipedia = tuple(
        provider for provider in tools.SEARCH_PROVIDERS if provider.name == "wikipedia"
    )
    assert len(wikipedia) == 1, tools.SEARCH_PROVIDERS
    monkeypatch.setattr(tools, "SEARCH_PROVIDERS", wikipedia)

    call = tools.ToolCall("web_search", {"query": WIKIPEDIA_QUERY, "max_results": 3})
    result = tools.ToolRegistry().execute(call)
    detail = result.detail or ""

    if not result.ok:
        reasons = re.findall(r"^- ([\w-]+): (.+)$", detail, re.MULTILINE)
        assert reasons, (result.summary, detail)
        if all(any(bad in reason for bad in _UNREACHABLE_REASONS) for _, reason in reasons):
            _pytest().skip(f"Wikipedia is unreachable from this machine: {detail}")
        _pytest().fail(f"Wikipedia produced no result for {WIKIPEDIA_QUERY!r}: {detail}")

    assert detail.startswith("provider: wikipedia\n"), detail
    urls = _RESULT_URL_RE.findall(detail)
    assert urls, detail
    assert all(url.startswith(WIKIPEDIA_PREFIX) for url in urls), detail
    rendered = detail.partition("\n")[2]
    assert WIKIPEDIA_NAME in rendered, detail
    assert result.summary == f"{len(urls)} results", (result.summary, detail)
    print(f"wikipedia: {detail[:300]}")


def test_web_search_falls_through_a_page_with_no_results(monkeypatch: Any) -> None:
    """A 200 holding no result link is a fall-through, not the end of the search.

    The first entry is pointed at a real 200 page that carries no result link
    (`duckduckgo.com/robots.txt`) and the rest at a closed local port. Both real
    reasons have to appear in the detail, and the walk has to reach the closed
    port at all, which is only possible if the empty 200 did not end the search
    — the defect this ladder fixes. The empty 200 is a challenge, so it is
    retried the full number of times; the refused connections are hard failures
    and get exactly one attempt; the detail reports both counts.
    """
    tools = _tools_module()
    first, *rest = tools.SEARCH_PROVIDERS
    assert rest, tools.SEARCH_PROVIDERS
    monkeypatch.setattr(
        tools,
        "SEARCH_PROVIDERS",
        (
            dataclasses.replace(
                first, template="https://duckduckgo.com/robots.txt?{query}"
            ),
            *(
                dataclasses.replace(provider, template="http://127.0.0.1:1/search?{query}")
                for provider in rest
            ),
        ),
    )

    call = tools.ToolCall("web_search", {"query": "mlx lm"})
    result = tools.ToolRegistry().execute(call)
    detail = result.detail or ""

    assert result.ok is False, result
    if f"- {first.name}: unreachable" in detail:
        _pytest().skip(f"duckduckgo.com is unreachable from this machine: {detail}")
    assert f"- {first.name}: HTTP 200 but the page held no result link" in detail, detail
    assert "unreachable" in detail, detail
    assert len([line for line in detail.splitlines() if line.startswith("- ")]) == 1 + len(
        rest
    ), detail
    # How many attempts each provider got, reported per provider: the challenge
    # was retried, every hard failure was not.
    retries = tools.CHALLENGE_RETRIES + 1
    assert re.search(
        rf"^- {re.escape(first.name)}: .* after {retries} attempts$", detail, re.MULTILINE
    ), detail
    for provider in rest:
        assert re.search(
            rf"^- {re.escape(provider.name)}: unreachable: .* after 1 attempt$",
            detail,
            re.MULTILINE,
        ), detail
    assert not _RESULT_TITLE_RE.search(detail), detail
    assert not _RESULT_URL_RE.search(detail), detail


def test_web_search_names_every_provider_when_they_all_fail(monkeypatch: Any) -> None:
    """When the whole ladder fails, the detail names each provider and why.

    Every entry is pointed at a closed local port: `urllib` really connects,
    really gets refused, and the reason reported is the socket's own. Nothing
    is faked, and the walk is proven to reach every provider rather than
    stopping at the first failure.
    """
    tools = _tools_module()
    unreachable = tuple(
        dataclasses.replace(provider, template="http://127.0.0.1:1/search?{query}")
        for provider in tools.SEARCH_PROVIDERS
    )
    assert len(unreachable) > 1, unreachable
    monkeypatch.setattr(tools, "SEARCH_PROVIDERS", unreachable)

    call = tools.ToolCall("web_search", {"query": "mlx lm", "max_results": 3})
    result = tools.ToolRegistry().execute(call)
    detail = result.detail or ""

    assert result.ok is False, result
    assert result.summary == "every search provider failed", result
    for provider in unreachable:
        assert f"- {provider.name}: unreachable: " in detail, detail
    assert not _RESULT_TITLE_RE.search(detail), detail
    assert not _RESULT_URL_RE.search(detail), detail


def test_web_search_failure_text_forbids_answering_from_memory(
    monkeypatch: Any,
) -> None:
    """A total failure tells the model not to answer from its own knowledge.

    Every provider is pointed at a closed local port, so the ladder really
    fails. The tool message the model reads is `model_text`, and it has to
    carry the instruction that the lookup did not happen — the reason a
    fabricated answer like a remembered CEO must not be given.
    """
    tools = _tools_module()
    monkeypatch.setattr(
        tools,
        "SEARCH_PROVIDERS",
        tuple(
            dataclasses.replace(provider, template="http://127.0.0.1:1/search?{query}")
            for provider in tools.SEARCH_PROVIDERS
        ),
    )

    call = tools.ToolCall("web_search", {"query": "who is the ceo of nvidia"})
    result = tools.ToolRegistry().execute(call)
    text = result.model_text()

    assert result.ok is False, result
    assert result.summary == "every search provider failed", result
    assert tools.SEARCH_FAILURE_INSTRUCTION in text, text
    assert (
        "could not look the answer up rather than answering from your own knowledge"
        in text
    ), text
    # The instruction is the model's; the detail still carries no fake result.
    assert not _RESULT_TITLE_RE.search(text), text
    assert not _RESULT_URL_RE.search(text), text


def test_tools_are_offered_through_the_chat_template() -> None:
    step_tools_offer_the_tools_through_the_chat_template(session())


def test_tools_fall_back_to_written_instructions() -> None:
    step_tools_fallback_written_instructions(session())


def test_http_api_accepts_tools() -> None:
    step_http_accepts_tools(session())


def test_generate_without_max_tokens_is_unlimited() -> None:
    step_generate_without_a_token_budget(session())


def test_generate_ceiling_bounds_a_runaway() -> None:
    step_generate_ceiling_bounds_a_runaway(_temp_dir())


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


def test_parse_calls_accepts_both_spellings() -> None:
    """PROTOCOL.md's two call spellings, anywhere in the text, in order."""
    parse_calls = _tools_module().parse_calls

    def pairs(text: str) -> List[tuple[str, Any]]:
        return [(call.name, call.arguments) for call in parse_calls(text)]

    # The tag form, after thinking text on the same line.
    assert pairs(
        'I will look that up.<tool_call>{"name": "read_file", '
        '"arguments": {"path": "note.txt"}}</tool_call>'
    ) == [("read_file", {"path": "note.txt"})]
    # Case-insensitive, and tolerant of the tag's own newlines.
    assert pairs(
        '<TOOL_CALL>\n{"name": "file_info", "arguments": {"path": "a"}}\n</TOOL_CALL>'
    ) == [("file_info", {"path": "a"})]
    # The fenced form, with the JSON on its own line...
    assert pairs(
        'Let me check.\n```tool\n{"name": "web_search", "arguments": {"query": "x"}}\n```'
    ) == [("web_search", {"query": "x"})]
    # ...and inline inside a sentence, which is what a real 0.5B model emits.
    assert pairs(
        "To find all files on your desktop, you can use the following command:  "
        '```tool {"name": "list_directory", "arguments": {"path": "your desktop"}} ```'
    ) == [("list_directory", {"path": "your desktop"})]
    # Every call in one reply, in order, across both spellings.
    assert pairs(
        '<tool_call>{"name": "a", "arguments": {}}</tool_call> then '
        '```tool {"name": "b", "arguments": {"x": 1}}```'
    ) == [("a", {}), ("b", {"x": 1})]
    # Malformed JSON and prose are not calls, and a fence must say `tool`.
    assert pairs('<tool_call>{not json}</tool_call>') == []
    assert pairs('```toolbar\n{"name": "a", "arguments": {}}\n```') == []
    assert pairs("The capital of France is Paris.") == []


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

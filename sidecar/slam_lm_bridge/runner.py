"""The model runner: one worker thread owns the model and the MLX stream.

Commands are serialised through a job queue so a load can never overlap a
generate, and cancellation is a sequence number rather than a flag — a `cancel`
that arrives between `generate` and the worker picking it up still wins.

All MLX work happens on the worker thread. Other threads only enqueue jobs
(and, for HTTP, drain a token sink the worker pushes into).
"""

from __future__ import annotations

import gc
import importlib.util
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import mlx.core as mx

from . import catalog
from .metrics import HardwareProbe, RunnerStats, live_metrics
from .protocol import RequestRecord, StatePayload, TokenPayload, monotonic_ms, sanitise_float

#: Prompt used for the post-load warmup: it exercises the real prefill path so
#: the first user prompt pays no one-off compile cost (mlx-lm's own `generate`
#: does the same with a real prompt).
WARMUP_PROMPT = "hi"
WARMUP_TOKENS = 1

#: Bounded request history for `GET /metrics` (newest first).
REQUEST_HISTORY_LIMIT = 100

_load_fn: Optional[Callable[..., Any]] = None
_stream_generate_fn: Optional[Callable[..., Any]] = None


class BridgeError(Exception):
    """A condition the app should see as `ok:false`."""


def _ensure_mlx_lm_importable() -> None:
    """Make the mlx-lm checkout importable when cwd is not the repo root.

    The bridge lives in the repo during development
    (`menubar/sidecar/slam_lm_bridge/`) but inside the app bundle when shipped
    (`Contents/Resources/sidecar/slam_lm_bridge/`), so the checkout is found by
    searching upwards for a directory that actually contains ``mlx_lm``.
    ``$SLAM_LM_REPO`` wins when the app already knows the answer.
    """
    if importlib.util.find_spec("mlx_lm") is not None:
        return
    candidates: list[Path] = []
    explicit = os.environ.get("SLAM_LM_REPO")
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(Path(__file__).resolve().parents[:9])
    for root in candidates:
        if (root / "mlx_lm" / "__init__.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return


def _mlx_lm() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Lazily import mlx-lm: `hello`/`catalog` answer long before this is needed."""
    global _load_fn, _stream_generate_fn
    if _load_fn is None or _stream_generate_fn is None:
        _ensure_mlx_lm_importable()
        try:
            from mlx_lm.generate import stream_generate
            from mlx_lm.utils import load
        except ImportError as exc:  # pragma: no cover - environment problem
            raise BridgeError(f"mlx-lm is not importable: {exc}") from exc
        _load_fn = load
        _stream_generate_fn = stream_generate
    return _load_fn, _stream_generate_fn


@dataclass
class _Job:
    fn: Callable[[], Any]
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None

    def run(self) -> None:
        try:
            self.result = self.fn()
        except BaseException as exc:  # reported to the caller, never swallowed
            self.error = exc


@dataclass
class _GenerateRequest:
    request: int
    prompt: str
    max_tokens: int
    cancel_seq: int
    submitted_perf: float
    submitted_wall: float
    sink: Optional[Callable[[Dict[str, Any]], None]] = None


@dataclass
class _Loaded:
    model_id: Optional[str] = None
    load_ms: float = 0.0


@dataclass
class _Outcome:
    """One finished generation: its protocol record plus any error message."""

    record: RequestRecord
    error: Optional[str] = None


class Runner:
    """Owns the loaded model, the MLX stream and the request lifecycle."""

    def __init__(
        self,
        writer: Any,
        stats: RunnerStats,
        probe: HardwareProbe,
        store: Any = None,
        hf_home: Optional[Path] = None,
        extra_dirs: Optional[List[Path]] = None,
    ):
        self._writer = writer
        self._stats = stats
        self._probe = probe
        self._store = store
        self._hf_home = hf_home
        self._extra_dirs = extra_dirs

        self._jobs: "queue.Queue[Optional[_Job]]" = queue.Queue()

        # Guarded by _lock: everything other threads may read.
        self._lock = threading.Lock()
        self._loaded = _Loaded()
        self._cancel_seq = 0
        self._active_request: Optional[int] = None
        # Generations submitted but not yet finished. Unlike `_active_request`
        # this covers a request still waiting in the worker's queue, which is
        # what `cancel` has to report on.
        self._outstanding = 0

        # Written only by the worker thread. `_tokenizer` is also read by
        # chat_prompt() from HTTP threads, which only reads a reference.
        self._model: Any = None
        self._tokenizer: Any = None

        self._history: deque[RequestRecord] = deque(maxlen=REQUEST_HISTORY_LIMIT)

    # MARK: - Lifecycle

    def start(self) -> None:
        threading.Thread(target=self._loop, name="slam-lm-runner", daemon=True).start()

    def _loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            job.run()
            job.done.set()

    def stop(self) -> None:
        """Ask the worker thread to finish once its queue drains."""
        self._jobs.put(None)

    def _call(self, fn: Callable[[], Any]) -> Any:
        """Run `fn` on the worker thread and wait for its result."""
        job = _Job(fn)
        self._jobs.put(job)
        job.done.wait()
        if job.error is not None:
            if isinstance(job.error, BridgeError):
                raise job.error
            raise BridgeError(str(job.error) or job.error.__class__.__name__)
        return job.result

    # MARK: - Reads other threads need

    @property
    def loaded_model_id(self) -> Optional[str]:
        with self._lock:
            return self._loaded.model_id

    def live(self) -> Dict[str, Any]:
        return live_metrics(self._stats, self._probe).to_wire()

    def chat_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Render chat messages for the loaded model's tokenizer.

        The tokenizer's own chat template when it has one, otherwise the
        message contents joined with newlines (PROTOCOL.md's HTTP API rule).
        """
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise BridgeError("no model loaded")
        if getattr(tokenizer, "has_chat_template", False):
            try:
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except Exception as exc:
                self._writer.log(
                    "warn", f"chat template failed, joining messages instead: {exc}"
                )
        return "\n".join(str(message.get("content", "")) for message in messages)

    def request_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [record.to_wire() for record in self._history]

    # MARK: - Commands

    def load(self, model_id: str) -> Dict[str, Any]:
        model_id = (model_id or "").strip()
        if not model_id:
            raise BridgeError("load requires a model id")
        self._bump_cancel()  # a load replaces whatever is running
        return self._call(lambda: self._load(model_id))

    def unload(self) -> Dict[str, Any]:
        self._bump_cancel()
        return self._call(self._unload)

    def submit(
        self,
        prompt: str,
        max_tokens: int,
        request: int,
        sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        chat: bool = False,
    ) -> int:
        """Queue one generation; returns the request id it will report.

        With `chat` the prompt is first rendered as a single `user` message
        through `chat_prompt`, so the worker prefills exactly what the HTTP API
        would send. The raw text is what gets validated, and `chat` is only
        consulted after every other check has passed, so a falsy or missing
        value keeps the historical verbatim-prompt behaviour.
        """
        if not isinstance(prompt, str):
            raise BridgeError("generate requires a prompt string")
        if not prompt.strip():
            raise BridgeError("generate requires a non-empty prompt")
        try:
            tokens = int(max_tokens)
        except (TypeError, ValueError):
            raise BridgeError("generate requires max_tokens as an integer") from None
        if tokens < 1:
            raise BridgeError("max_tokens must be at least 1")
        if not isinstance(request, int) or request <= 0:
            raise BridgeError("generate requires a positive request id")

        with self._lock:
            if self._loaded.model_id is None:
                raise BridgeError("no model loaded")
            cancel_seq = self._cancel_seq
            self._outstanding += 1

        try:
            if chat:
                # Rendered here, before queueing: `_GenerateRequest.prompt` then
                # carries the text the tokenizer's chat template produced, so
                # the worker and every event it emits are untouched.
                prompt = self.chat_prompt([{"role": "user", "content": prompt}])
        except BaseException:
            with self._lock:
                self._outstanding -= 1
            raise

        req = _GenerateRequest(
            request=request,
            prompt=prompt,
            max_tokens=tokens,
            cancel_seq=cancel_seq,
            submitted_perf=time.perf_counter(),
            submitted_wall=time.time(),
            sink=sink,
        )

        def run() -> Any:
            try:
                return self._generate(req)
            finally:
                with self._lock:
                    self._outstanding -= 1

        self._jobs.put(_Job(run))
        return request

    def cancel(self) -> bool:
        """Abort a queued or running generation; True when there was one.

        A request that is still sitting in the worker's queue counts: the
        sequence number it captured at submit time means it will be cancelled as
        soon as it is picked up, so reporting `False` here would contradict the
        `request_end` that follows.
        """
        with self._lock:
            in_flight = self._outstanding > 0 or self._active_request is not None
            self._cancel_seq += 1
        return in_flight

    def _bump_cancel(self) -> None:
        with self._lock:
            self._cancel_seq += 1

    def _cancelled(self, seq: int) -> bool:
        with self._lock:
            return self._cancel_seq != seq

    # MARK: - Worker-thread implementation

    def _state(self, status: str, phase: str, message: Optional[str] = None) -> None:
        with self._lock:
            model_id = self._loaded.model_id
        payload: StatePayload = self._stats.set_state(status, phase, model_id, message)
        self._writer.send_event("state", payload.to_wire())

    def _load(self, model_id: str) -> Dict[str, Any]:
        with self._lock:
            already = self._loaded.model_id == model_id and self._model is not None
            previous_load_ms = self._loaded.load_ms
        if already:
            return {"model": model_id, "loadMs": previous_load_ms, "memoryBytes": 0}

        if self._model is not None:
            self._unload(quiet=True)

        self._state("loading", "load", f"loading {model_id}")
        self._writer.log("info", f"loading {model_id}")

        load, _ = _mlx_lm()
        path = catalog.local_path(model_id, self._hf_home, self._extra_dirs) or model_id
        before = int(mx.get_active_memory())
        started = time.perf_counter()
        try:
            model, tokenizer = load(path)
            self._warmup(model, tokenizer)
        except Exception as exc:
            self._model = None
            self._tokenizer = None
            with self._lock:
                self._loaded = _Loaded()
            self._stats.set_load_ms(0.0)
            gc.collect()
            mx.clear_cache()
            self._state("error", "idle", str(exc))
            self._writer.log("error", f"load failed for {model_id}: {exc}")
            raise BridgeError(f"failed to load {model_id}: {exc}") from None

        load_ms = monotonic_ms(started)
        self._model = model
        self._tokenizer = tokenizer
        with self._lock:
            self._loaded = _Loaded(model_id=model_id, load_ms=load_ms)
        self._stats.set_load_ms(load_ms)
        self._stats.clear_token_window()

        if self._store is not None:
            self._store.mark_used(model_id)

        memory_bytes = max(0, int(mx.get_active_memory()) - before)
        self._state("ready", "idle", None)
        self._writer.log(
            "info", f"loaded {model_id} in {load_ms:.0f} ms ({memory_bytes} bytes active)"
        )
        return {"model": model_id, "loadMs": load_ms, "memoryBytes": memory_bytes}

    def _drain_generation_stream(self) -> None:
        """Wait for the work mlx-lm just queued.

        mlx-lm owns the stream and we deliberately pass none in — the released
        package does not accept a `stream` argument, and both versions default to
        a thread-local stream, so the worker thread already has its own. That
        leaves the stream addressed by name, which is what mlx-lm itself uses.
        """
        try:
            from mlx_lm.generate import generation_stream
        except ImportError:  # pragma: no cover - mlx-lm moved its internals
            mx.synchronize()
            return
        mx.synchronize(generation_stream)

    def _warmup(self, model: Any, tokenizer: Any) -> None:
        """One token through the real prefill path, with no telemetry."""
        _, stream_generate = _mlx_lm()
        generator = stream_generate(model, tokenizer, WARMUP_PROMPT, max_tokens=WARMUP_TOKENS)
        try:
            next(generator)
        except StopIteration:
            pass
        finally:
            generator.close()
            self._drain_generation_stream()

    def _unload(self, quiet: bool = False) -> Dict[str, Any]:
        model_id = self._loaded.model_id
        if model_id is None and self._model is None:
            return {}
        if not quiet:
            self._state("loading", "load", "unloading")
        self._model = None
        self._tokenizer = None
        with self._lock:
            self._loaded = _Loaded()
            self._active_request = None
        self._stats.set_load_ms(0.0)
        self._stats.clear_token_window()
        gc.collect()
        mx.clear_cache()
        if not quiet:
            self._state("idle", "idle", None)
        if model_id:
            self._writer.log("info", f"unloaded {model_id}")
        return {}

    def _generate(self, req: _GenerateRequest) -> RequestRecord:
        """Run one request, then publish exactly one `request_end` for it."""
        try:
            outcome = self._run_generation(req)
        except BaseException as exc:  # never lose a request, not even a bug
            message = f"{exc.__class__.__name__}: {exc}"
            self._writer.log("error", f"generation failed before it started: {message}")
            outcome = _Outcome(
                record=self._error_record(req, message),
                error=message,
            )
        self._publish(req, outcome)
        return outcome.record

    def _publish(self, req: _GenerateRequest, outcome: _Outcome) -> None:
        record = outcome.record
        # The HTTP sink goes first: a caller waiting on it must always be
        # released, whatever happens to the app-facing stdout stream.
        if req.sink is not None:
            req.sink(
                {"kind": "end", "record": record.to_wire(), "error": outcome.error}
            )
        self._stats.finish_request()
        with self._lock:
            self._history.appendleft(record)
            self._active_request = None
        self._writer.send_event("request_end", record.to_wire())
        self._state("ready", "idle", outcome.error)

    def _error_record(self, req: _GenerateRequest, message: str) -> RequestRecord:
        with self._lock:
            model_id = self._loaded.model_id or ""
        return RequestRecord(
            request=req.request,
            model=model_id,
            promptTokens=0,
            genTokens=0,
            ttftMs=0.0,
            prefillTps=0.0,
            decodeTps=0.0,
            peakMemBytes=int(mx.get_peak_memory()),
            startedAt=time.time(),
            totalMs=0.0,
            finishReason="error",
        )

    def _run_generation(self, req: _GenerateRequest) -> _Outcome:
        model, tokenizer, model_id = self._model, self._tokenizer, self._loaded.model_id
        if model is None or tokenizer is None or model_id is None:
            raise BridgeError("no model loaded")
        _, stream_generate = _mlx_lm()

        with self._lock:
            self._active_request = req.request
        self._state("generating", "prefill", None)

        started_at = req.submitted_wall
        started = req.submitted_perf
        prompt_tokens = 0
        gen_tokens = 0
        ttft_ms = 0.0
        first: Optional[float] = None
        last: Optional[float] = None
        last_response: Any = None
        finish_reason: Optional[str] = None
        error_message: Optional[str] = None
        generator: Any = None

        try:
            if self._cancelled(req.cancel_seq):
                # Cancelled between submit and start: never touch the model.
                finish_reason = "cancel"
            else:
                generator = stream_generate(
                    model,
                    tokenizer,
                    req.prompt,
                    max_tokens=req.max_tokens,
                )
                for response in generator:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                        ttft_ms = monotonic_ms(started)
                        prompt_tokens = int(response.prompt_tokens)
                        prefill_tps = (
                            prompt_tokens / (ttft_ms / 1000.0) if ttft_ms > 0 else 0.0
                        )
                        self._stats.note_ttft(ttft_ms, sanitise_float(prefill_tps))
                        self._state("generating", "decode", None)

                    last = now
                    last_response = response
                    gen_tokens = int(response.generation_tokens)
                    prompt_tokens = int(response.prompt_tokens)
                    self._stats.note_token(now)
                    payload = TokenPayload(
                        request=req.request,
                        index=gen_tokens,
                        text=response.text,
                        ttsMs=monotonic_ms(started),
                    )
                    self._writer.send_event("token", payload.to_wire())
                    if req.sink is not None:
                        req.sink(
                            {
                                "kind": "token",
                                "index": gen_tokens,
                                "text": response.text,
                            }
                        )

                    if self._cancelled(req.cancel_seq):
                        finish_reason = "cancel"
                        break
                else:
                    raw = getattr(last_response, "finish_reason", None)
                    finish_reason = raw if raw in ("length", "stop") else "length"
        except Exception as exc:  # a real inference failure is a real record
            finish_reason = "error"
            error_message = f"{exc.__class__.__name__}: {exc}"
            self._writer.log("error", f"generation failed: {error_message}")
        finally:
            if generator is not None:
                try:
                    generator.close()
                except Exception:
                    pass

        total_ms = monotonic_ms(started)
        # PROTOCOL.md defines the per-request rate as (genTokens - 1) /
        # (t_last - t_first) over the token yields, so that is what is
        # reported; mlx-lm's own generation_tps is the fallback for the cases
        # where no interval exists (a single token, or a stream that failed).
        decode_tps = 0.0
        if gen_tokens > 1 and first is not None and last is not None and last > first:
            decode_tps = sanitise_float((gen_tokens - 1) / (last - first))
        if decode_tps <= 0.0:
            decode_tps = sanitise_float(getattr(last_response, "generation_tps", 0.0) or 0.0)
        # Prefill is likewise pinned to the wall-clock TTFT definition, with
        # mlx-lm's prompt_tps used only when no TTFT could be measured.
        prefill_tps = (
            sanitise_float(prompt_tokens / (ttft_ms / 1000.0))
            if ttft_ms > 0 and prompt_tokens > 0
            else sanitise_float(getattr(last_response, "prompt_tps", 0.0) or 0.0)
        )
        # Peak memory has no protocol formula; MLX's own high-water mark is the
        # measurement, with the process-wide value as the fallback.
        reported_peak = sanitise_float(getattr(last_response, "peak_memory", 0.0) or 0.0)
        peak_bytes = (
            int(reported_peak * 1e9) if reported_peak > 0 else int(mx.get_peak_memory())
        )

        record = RequestRecord(
            request=req.request,
            model=model_id,
            promptTokens=prompt_tokens,
            genTokens=gen_tokens,
            ttftMs=sanitise_float(ttft_ms),
            prefillTps=prefill_tps,
            decodeTps=decode_tps,
            peakMemBytes=peak_bytes,
            startedAt=started_at,
            totalMs=sanitise_float(total_ms),
            finishReason=finish_reason or "length",
        )
        return _Outcome(record=record, error=error_message)


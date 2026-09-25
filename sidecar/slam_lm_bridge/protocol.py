"""Wire protocol for the SlamLM bridge.

This module is the Python mirror of ``menubar/PROTOCOL.md``: it owns the
encode/decode helpers, the thread-safe stdout writer and the dataclass layer
for every type that crosses the wire. Field names are camelCase on purpose —
they *are* the wire keys, so nothing has to be renamed on the way out.

Nothing here measures anything; metrics live in ``metrics.py``.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

PROTOCOL_VERSION = 1
BRIDGE_NAME = "slam-lm-bridge"

#: Canonical display order for categories. Used for `Model.categories` and,
#: filtered to what was discovered, for `hello.categories`.
CATEGORY_ORDER = (
    "Chat",
    "Code",
    "Vision",
    "Embedding",
    "Audio",
    "Instruct",
    "Multilingual",
    "Reasoning",
)


# MARK: - Encode / decode


def encode(payload: Dict[str, Any]) -> str:
    """Encode one wire line: compact JSON, UTF-8, no embedded newlines.

    ``allow_nan=False`` keeps non-JSON floats (NaN/Inf) out of the protocol;
    every producer of numbers must sanitise them rather than leak them here.
    """
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def decode(line: str) -> Dict[str, Any]:
    """Decode one wire line into a dict. Raises ``ValueError`` on bad input."""
    parsed = json.loads(line)
    if not isinstance(parsed, dict):
        raise ValueError("wire line is not a JSON object")
    return parsed


def reply(id: int, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": id, "ok": True, "result": result}


def error_reply(id: int, message: str) -> Dict[str, Any]:
    return {"id": id, "ok": False, "error": message}


def event(name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {"event": name, "data": data}


class LineWriter:
    """Serialises stdout (and stderr logging) behind one lock.

    Concurrent writer threads — the sampler, the runner worker, the command
    dispatcher and every HTTP handler — therefore never interleave a line.
    """

    def __init__(self, stream: Any = None, err: Any = None):
        self._stream = stream if stream is not None else sys.stdout
        self._err = err if err is not None else sys.stderr
        self._lock = threading.Lock()

    # -- raw send

    def send(self, payload: Dict[str, Any]) -> None:
        try:
            line = encode(payload)
        except (TypeError, ValueError) as exc:  # a producer bug, not a wire case
            self._log("error", f"could not encode {payload!r}: {exc}")
            rid = payload.get("id")
            if isinstance(rid, int) and "result" in payload:
                line = encode(error_reply(rid, f"encode failure: {exc}"))
            else:
                return
        with self._lock:
            try:
                self._stream.write(line + "\n")
                self._stream.flush()
            except OSError as exc:
                # stdout is gone (the app exited); nothing can be reported.
                self._err.write(f"[{BRIDGE_NAME}] error: stdout write failed: {exc}\n")
                self._err.flush()

    # -- envelopes

    def send_reply(self, id: int, result: Dict[str, Any]) -> None:
        self.send(reply(id, result))

    def send_error(self, id: int, message: str) -> None:
        self.send(error_reply(id, message))

    def send_event(self, name: str, data: Dict[str, Any]) -> None:
        self.send(event(name, data))

    # -- logging (stderr is never parsed; the `log` event feeds the app's strip)

    def _log(self, level: str, message: str) -> None:
        with self._lock:
            self._err.write(f"[{BRIDGE_NAME}] {level}: {message}\n")
            self._err.flush()

    def log(self, level: str, message: str, surface: bool = True) -> None:
        self._log(level, message)
        if surface:
            self.send_event("log", LogPayload(level=level, message=message).to_wire())


# MARK: - Wire types


class _Wire:
    """Mixin giving every wire dataclass its exact-key dict form."""

    def to_wire(self) -> Dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]


@dataclass
class Model(_Wire):
    id: str
    name: str
    params: str
    quant: str
    bytes: int
    path: str
    categories: List[str]
    architecture: str
    contextLength: int
    hasChatTemplate: bool
    lastUsed: float


@dataclass
class Hardware(_Wire):
    model: str
    chip: str
    gpuCores: int
    totalBytes: int
    recommendedBytes: int
    memoryActiveBytes: int
    memoryPeakBytes: int
    cacheBytes: int
    processRssBytes: int
    systemUsedBytes: int
    systemAppBytes: int
    systemWiredBytes: int
    systemCompressedBytes: int
    systemCachedBytes: int
    systemSwapBytes: int


@dataclass
class LiveMetrics(_Wire):
    ts: float
    memoryActiveBytes: int
    memoryPeakBytes: int
    memoryCacheBytes: int
    memoryTotalBytes: int
    memoryRecommendedBytes: int
    processRssBytes: int
    systemUsedBytes: int
    systemAppBytes: int
    systemWiredBytes: int
    systemCompressedBytes: int
    systemCachedBytes: int
    systemSwapBytes: int
    decodeTps: float
    prefillTps: float
    ttftMs: float
    tokensGenerated: int
    requests: int
    status: str
    phase: str
    model: Optional[str]
    loadMs: float


@dataclass
class RequestRecord(_Wire):
    request: int
    model: str
    promptTokens: int
    genTokens: int
    ttftMs: float
    prefillTps: float
    decodeTps: float
    peakMemBytes: int
    startedAt: float
    totalMs: float
    finishReason: str


@dataclass
class StatePayload(_Wire):
    status: str
    phase: str
    model: Optional[str]
    message: Optional[str] = None


@dataclass
class TokenPayload(_Wire):
    request: int
    index: int
    text: str
    ttsMs: float


@dataclass
class LogPayload(_Wire):
    level: str
    message: str


#: Requests arriving over HTTP get ids from this offset so they can never be
#: confused with ids chosen by the app for its own `generate` commands.
HTTP_REQUEST_BASE = 1_000_000


def sanitise_float(value: Any) -> float:
    """Clamp NaN/Inf to 0.0 so `encode` can never see a non-JSON float."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if out != out or out in (float("inf"), float("-inf")):
        return 0.0
    return out


def monotonic_ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000.0

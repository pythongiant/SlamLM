"""The bridge itself: newline-delimited JSON over stdin/stdout.

stdin is read on a background thread; commands are dispatched one at a time, in
order, so a load can never be reordered against an unload. Every request gets
exactly one reply envelope, events interleave freely, and stdout is written
through a single lock so concurrent writer threads never split a line. Logs go
to stderr (and to the `log` event, which is what the app's log strip shows).
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__, catalog
from .http_api import HttpApi, HttpApiError
from .metrics import HardwareProbe, MetricsSampler, RunnerStats
from .protocol import (
    BRIDGE_NAME,
    HTTP_REQUEST_BASE,
    LineWriter,
    PROTOCOL_VERSION,
    decode,
)
from .runner import BridgeError, Runner
from .state import StateStore

DEFAULT_PORT = 8712


class Bridge:
    """Wires the runner, the sampler, the HTTP API and the command surface."""

    def __init__(
        self,
        writer: Optional[LineWriter] = None,
        hf_home: Optional[Path] = None,
        extra_dirs: Optional[List[Path]] = None,
    ):
        self.writer = writer if writer is not None else LineWriter(sys.stdout)
        self.hf_home = hf_home
        self.extra_dirs = extra_dirs

        self.store = StateStore(warn=lambda message: self.writer.log("warn", message))
        self.stats = RunnerStats()
        self.probe = HardwareProbe()
        self.runner = Runner(
            writer=self.writer,
            stats=self.stats,
            probe=self.probe,
            store=self.store,
            hf_home=hf_home,
            extra_dirs=extra_dirs,
        )
        self.http = HttpApi(
            self.runner,
            log=lambda level, message: self.writer.log(level, message),
            id_source=self.next_request_id,
        )
        self.sampler = MetricsSampler(
            self.stats,
            emit=lambda sample: self.writer.send_event("metrics", sample.to_wire()),
            probe=self.probe,
            on_error=lambda message: self.writer.log("warn", message),
        )
        self._id_lock = threading.Lock()
        self._request_seq = HTTP_REQUEST_BASE

    # MARK: - Lifecycle

    def start(self) -> None:
        self.runner.start()
        self.probe.warm()
        self.sampler.start()
        self.writer.log("info", f"{BRIDGE_NAME}/{__version__} ready (protocol 1)")

    def shutdown(self) -> None:
        self.sampler.stop()
        self.http.stop()
        self.runner.stop()

    # MARK: - Command surface

    def next_request_id(self) -> int:
        with self._id_lock:
            self._request_seq += 1
            return self._request_seq

    def scan_models(self) -> List[Any]:
        return catalog.scan_models(
            self.hf_home, self.extra_dirs, self.store.last_used()
        )

    def handle(self, message: Dict[str, Any]) -> None:
        request_id = message.get("id")
        if not isinstance(request_id, int) or isinstance(request_id, bool):
            self.writer.log(
                "warn", f"ignoring a message without an integer id: {message!r}"
            )
            return
        command = message.get("cmd")
        try:
            result = self._execute(command, message)
        except (BridgeError, HttpApiError) as exc:
            self.writer.send_error(request_id, str(exc))
        except Exception as exc:  # never leave a request unanswered
            self.writer.log("error", f"{command!r} failed: {exc!r}")
            self.writer.send_error(request_id, f"{command} failed: {exc}")
        else:
            self.writer.send_reply(request_id, result)

    def _execute(self, command: Any, message: Dict[str, Any]) -> Dict[str, Any]:
        if command == "hello":
            models = self.scan_models()
            return {
                "protocol": PROTOCOL_VERSION,
                "bridge": f"{BRIDGE_NAME}/{__version__}",
                "hardware": self.probe.sample().to_wire(),
                "categories": catalog.hello_categories(models),
            }
        if command == "catalog":
            return {"models": [model.to_wire() for model in self.scan_models()]}
        if command == "load":
            return self.runner.load(str(message.get("model") or ""))
        if command == "unload":
            return self.runner.unload()
        if command == "generate":
            request_id = message.get("request")
            if not isinstance(request_id, int) or isinstance(request_id, bool) or request_id <= 0:
                request_id = self.next_request_id()
            return {
                "request": self.runner.submit(
                    message.get("prompt"),
                    # Absent, null and any falsy junk all mean no limit.
                    message.get("max_tokens"),
                    request_id,
                    # Absent, null and any falsy junk all mean the raw prompt.
                    chat=bool(message.get("chat") or False),
                    tools=bool(message.get("tools") or False),
                )
            }
        if command == "cancel":
            return {"cancelled": self.runner.cancel()}
        if command == "serve":
            port = message.get("port")
            if port is None:
                port = DEFAULT_PORT
            try:
                port = int(port)
            except (TypeError, ValueError):
                raise BridgeError("serve requires an integer port") from None
            return self.http.start(port)
        if command == "stop_serve":
            return {"stopped": self.http.stop()}
        if command == "ping":
            return {"pong": True, "ts": time.time()}
        raise BridgeError(f"unknown command: {command!r}")

    # MARK: - stdio loop

    def run(self) -> int:
        self.start()
        lines: "queue.Queue[Optional[str]]" = queue.Queue()
        reader = threading.Thread(
            target=self._read_stdin, args=(lines,), name="slam-lm-stdin", daemon=True
        )
        reader.start()
        try:
            while True:
                line = lines.get()
                if line is None:
                    break
                self._dispatch_line(line)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
        return 0

    def _read_stdin(self, lines: "queue.Queue[Optional[str]]") -> None:
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                lines.put(line)
        except (OSError, ValueError) as exc:
            self.writer.log("warn", f"stdin read failed: {exc}")
        finally:
            lines.put(None)

    def _dispatch_line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        try:
            message = decode(text)
        except ValueError as exc:
            self.writer.log("warn", f"ignoring malformed request line: {exc}")
            return
        self.handle(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slam_lm_bridge",
        description="SlamLM bridge: local MLX inference over newline-delimited JSON.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="also start the OpenAI-compatible HTTP API on this port at boot",
    )
    parser.add_argument(
        "--hf-home",
        default=None,
        help="model store to scan (default: $HF_HOME or ~/.cache/huggingface/hub)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="extra model stores to scan, colon separated (default: $SLAM_LM_MODELS)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    extra_dirs = (
        [Path(part) for part in args.models.split(":") if part]
        if args.models
        else None
    )
    bridge = Bridge(
        writer=LineWriter(sys.stdout),
        hf_home=Path(args.hf_home) if args.hf_home else None,
        extra_dirs=extra_dirs,
    )
    if args.port is not None:
        try:
            bridge.http.start(args.port)
        except HttpApiError as exc:
            bridge.writer.log("error", f"could not start the HTTP API: {exc}")
    return bridge.run()


if __name__ == "__main__":
    raise SystemExit(main())

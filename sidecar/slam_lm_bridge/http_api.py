"""OpenAI-compatible HTTP API over the loaded model.

Standard library only (``http.server``). Every request goes through the same
``Runner`` path as an app-initiated `generate`, so HTTP traffic produces the
same `token` / `request_end` events and feeds the same metrics.

Bound to ``127.0.0.1`` on the requested port; ``503`` when no model is loaded,
``404`` for unknown paths.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional

from .protocol import BRIDGE_NAME, HTTP_REQUEST_BASE
from .runner import BridgeError

#: A client that asks for more than this gets this much: a cap on an explicit
#: request, never a default for a request that named no budget.
MAX_MAX_TOKENS = 4096


class HttpApiError(Exception):
    """A condition that maps onto an HTTP status code."""

    def __init__(self, status: int, message: str, kind: str = "invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.kind = kind


def message_text(message: Dict[str, Any]) -> str:
    """Flatten one chat message's content into text."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def normalise_messages(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise HttpApiError(400, "`messages` must be a non-empty array")
    messages: List[Dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict) or not isinstance(entry.get("role"), str):
            raise HttpApiError(400, "each message needs a `role`")
        message = dict(entry)
        message["content"] = message_text(entry)
        messages.append(message)
    return messages


def _max_tokens(body: Dict[str, Any]) -> Optional[int]:
    """PROTOCOL.md: an absent `max_tokens` is no limit, not a budget."""
    raw = body.get("max_tokens")
    if raw is None:
        raw = body.get("max_completion_tokens")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise HttpApiError(400, "`max_tokens` must be an integer") from None
    if value < 1:
        raise HttpApiError(400, "`max_tokens` must be at least 1")
    return min(value, MAX_MAX_TOKENS)


class HttpApi:
    """Owns the HTTP server thread and the request-id space for HTTP traffic."""

    def __init__(
        self,
        runner: Any,
        host: str = "127.0.0.1",
        log: Optional[Callable[[str, str], None]] = None,
        id_source: Optional[Callable[[], int]] = None,
    ):
        self._runner = runner
        self._host = host
        self._log = log
        self._id_source = id_source
        self._lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._port: Optional[int] = None
        self._request_seq = HTTP_REQUEST_BASE

    # MARK: - Lifecycle

    @property
    def runner(self) -> Any:
        return self._runner

    @property
    def running(self) -> bool:
        with self._lock:
            return self._server is not None

    @property
    def port(self) -> Optional[int]:
        with self._lock:
            return self._port

    @property
    def url(self) -> Optional[str]:
        port = self.port
        return f"http://{self._host}:{port}" if port else None

    def start(self, port: int) -> Dict[str, Any]:
        """Idempotent: same port returns the running server, a new port rebinds.

        The new port is bound before the old server is stopped, so a failed
        bind leaves a working server untouched.
        """
        with self._lock:
            if self._server is not None and self._port == port:
                return {"port": int(port), "url": f"http://{self._host}:{port}"}

        handler = _make_handler(self)
        try:
            server = ThreadingHTTPServer((self._host, int(port)), handler)
        except OSError as exc:
            raise HttpApiError(
                503,
                f"cannot bind {self._host}:{port} — {exc}",
                kind="server_error",
            ) from None
        server.daemon_threads = True
        bound = int(server.server_address[1])

        with self._lock:
            previous, self._server = self._server, server
            self._port = bound
        if previous is not None:
            previous.shutdown()
            previous.server_close()

        threading.Thread(
            target=server.serve_forever, name="slam-lm-http", daemon=True
        ).start()
        url = f"http://{self._host}:{bound}"
        if self._log is not None:
            self._log("info", f"serving the OpenAI API on {url}")
        return {"port": bound, "url": url}

    def stop(self) -> bool:
        with self._lock:
            server, self._server = self._server, None
            self._port = None
        if server is None:
            return False
        server.shutdown()
        server.server_close()
        if self._log is not None:
            self._log("info", "stopped the OpenAI API")
        return True

    # MARK: - Request plumbing

    def next_request_id(self) -> int:
        if self._id_source is not None:
            return self._id_source()
        with self._lock:
            self._request_seq += 1
            return self._request_seq

    def log_message(self, level: str, message: str) -> None:
        if self._log is not None:
            self._log(level, message)

    def render_chat_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """The model's chat template when it has one, else newline-joined content."""
        return self._runner.chat_prompt(messages)


def _make_handler(api: HttpApi) -> type:
    runner = api.runner

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"{BRIDGE_NAME}/http"
        timeout = 600

        # -- logging: stderr only, never stdout

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            api.log_message("info", "http " + (format % args))

        def log_error(self, format: str, *args: Any) -> None:  # noqa: A002
            api.log_message("warn", "http " + (format % args))

        # -- verbs

        def do_GET(self) -> None:  # noqa: N802
            try:
                self._route_get()
            except HttpApiError as exc:
                self._send_error(exc)
            except Exception as exc:  # pragma: no cover - defensive
                self._send_error(HttpApiError(500, str(exc), "server_error"))

        def do_POST(self) -> None:  # noqa: N802
            try:
                self._route_post()
            except HttpApiError as exc:
                self._send_error(exc)
            except Exception as exc:  # pragma: no cover - defensive
                self._send_error(HttpApiError(500, str(exc), "server_error"))

        # -- routes

        def _route_get(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/health":
                self._send_json(
                    200, {"status": "ok", "model": runner.loaded_model_id}
                )
                return
            if path == "/metrics":
                self._send_json(
                    200,
                    {"live": runner.live(), "requests": runner.request_history()},
                )
                return
            if path == "/v1/models":
                model_id = runner.loaded_model_id
                if model_id is None:
                    raise HttpApiError(503, "no model is loaded", "server_error")
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": model_id,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "local",
                            }
                        ],
                    },
                )
                return
            raise HttpApiError(404, f"unknown path {path!r}", "not_found")

        def _route_post(self) -> None:
            path = self.path.split("?", 1)[0]
            if path not in ("/v1/chat/completions", "/v1/completions"):
                self._drain_body()
                raise HttpApiError(404, f"unknown path {path!r}", "not_found")

            body = self._read_json_body()
            if runner.loaded_model_id is None:
                raise HttpApiError(503, "no model is loaded", "server_error")

            # Only prompt shape, token budget, `tools` and `stream` are honoured;
            # sampling knobs (`temperature`, `top_p`, …) are not plumbed through
            # to mlx-lm, so generation runs with mlx-lm's own default sampler.
            stream = bool(body.get("stream") or False)
            max_tokens = _max_tokens(body)
            tools = bool(body.get("tools") or False)

            if path == "/v1/chat/completions":
                messages = normalise_messages(body.get("messages"))
                prompt = api.render_chat_prompt(messages)
                object_name, chunk_object = "chat.completion", "chat.completion.chunk"
            else:
                raw_prompt = body.get("prompt")
                if not isinstance(raw_prompt, str):
                    raise HttpApiError(400, "`prompt` must be a string")
                prompt = raw_prompt
                object_name, chunk_object = "text_completion", "text_completion"

            if not prompt.strip():
                raise HttpApiError(400, "the rendered prompt is empty")

            request_id = api.next_request_id()
            sink: "queue.Queue[Dict[str, Any]]" = queue.Queue()
            try:
                runner.submit(
                    prompt, max_tokens, request_id, sink=sink.put_nowait, tools=tools
                )
            except BridgeError as exc:
                status = 503 if "no model" in str(exc) else 400
                raise HttpApiError(status, str(exc), "server_error") from None

            model_id = runner.loaded_model_id or ""
            created = int(time.time())
            is_chat = object_name == "chat.completion"
            completion_id = f"{'chatcmpl' if is_chat else 'cmpl'}-{request_id}"

            if stream:
                self._stream_response(
                    sink, model_id, completion_id, created, is_chat, chunk_object
                )
            else:
                self._collect_response(
                    sink, model_id, completion_id, created, object_name
                )

        # -- bodies

        def _drain_body(self) -> None:
            """Consume a request body this handler will not parse.

            Leaving body bytes in the socket desynchronises the next request on
            a keep-alive connection - the leftovers get read as a request line -
            so read what the client declared, or close the connection when the
            length cannot be known.
            """
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                self.close_connection = True
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self.close_connection = True
                return
            remaining = max(0, length)
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    break
                remaining -= len(chunk)

        def _read_json_body(self) -> Dict[str, Any]:
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length or 0)
            except ValueError:
                raise HttpApiError(400, "invalid Content-Length") from None
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                raise HttpApiError(400, "request body is not valid JSON") from None
            if not isinstance(parsed, dict):
                raise HttpApiError(400, "request body must be a JSON object")
            return parsed

        def _collect_response(
            self,
            sink: Any,
            model_id: str,
            completion_id: str,
            created: int,
            object_name: str,
        ) -> None:
            text = ""
            record: Optional[Dict[str, Any]] = None
            error: Optional[str] = None
            while record is None:
                item = sink.get()
                if item["kind"] == "token":
                    text += item["text"]
                else:
                    record = item["record"]
                    error = item.get("error")
            if record["finishReason"] == "error":
                raise HttpApiError(500, error or "generation failed", "server_error")

            usage = {
                "prompt_tokens": record["promptTokens"],
                "completion_tokens": record["genTokens"],
                "total_tokens": record["promptTokens"] + record["genTokens"],
            }
            if object_name == "chat.completion":
                payload = {
                    "id": completion_id,
                    "object": object_name,
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": record["finishReason"],
                        }
                    ],
                    "usage": usage,
                }
            else:
                payload = {
                    "id": completion_id,
                    "object": object_name,
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "text": text,
                            "finish_reason": record["finishReason"],
                        }
                    ],
                    "usage": usage,
                }
            self._send_json(200, payload)

        def _stream_response(
            self,
            sink: Any,
            model_id: str,
            completion_id: str,
            created: int,
            is_chat: bool,
            chunk_object: str,
        ) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            closed = False

            def chunk(payload: Dict[str, Any]) -> None:
                nonlocal closed
                if closed:
                    return
                data = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode(
                    "utf-8"
                )
                try:
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    closed = True  # client hung up: finish the request regardless

            record: Optional[Dict[str, Any]] = None
            while record is None:
                item = sink.get()
                if item["kind"] == "token":
                    delta = (
                        {"delta": {"content": item["text"]}, "index": 0,
                         "finish_reason": None}
                        if is_chat
                        else {"text": item["text"], "index": 0, "finish_reason": None}
                    )
                    chunk(
                        {
                            "id": completion_id,
                            "object": chunk_object,
                            "created": created,
                            "model": model_id,
                            "choices": [delta],
                        }
                    )
                else:
                    record = item["record"]

            final = (
                {"delta": {}, "index": 0, "finish_reason": record["finishReason"]}
                if is_chat
                else {"text": "", "index": 0, "finish_reason": record["finishReason"]}
            )
            chunk(
                {
                    "id": completion_id,
                    "object": chunk_object,
                    "created": created,
                    "model": model_id,
                    "choices": [final],
                    "usage": {
                        "prompt_tokens": record["promptTokens"],
                        "completion_tokens": record["genTokens"],
                        "total_tokens": record["promptTokens"] + record["genTokens"],
                    },
                }
            )
            if not closed:
                done = b"data: [DONE]\n\n"
                try:
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(done), done))
                    self.wfile.write(b"0\r\n\r\n")  # terminal chunk
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            self.close_connection = True

        # -- writers

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, OSError):
                self.close_connection = True

        def _send_error(self, error: HttpApiError) -> None:
            self._send_json(
                error.status,
                {
                    "error": {
                        "message": error.message,
                        "type": error.kind,
                        "code": error.status,
                    }
                },
            )

    return Handler

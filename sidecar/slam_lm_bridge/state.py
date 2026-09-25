"""Per-model state that survives restarts.

``~/.slam-lm/state.json`` holds one thing the catalog needs and the filesystem
cannot tell us: when each model was last used, so the picker can sort by
recency. Writes are atomic (temp file + rename) and every failure is tolerated
— a bridge that cannot remember recency is still a working bridge.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

#: Environment override, used by the tests to keep the real user state clean.
STATE_ENV = "SLAM_LM_STATE"
DEFAULT_STATE_DIR = ".slam-lm"
DEFAULT_STATE_FILE = "state.json"


def default_state_path() -> Path:
    override = os.environ.get(STATE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / DEFAULT_STATE_DIR / DEFAULT_STATE_FILE


class StateStore:
    """Loads, updates and atomically rewrites the bridge's state file."""

    def __init__(
        self,
        path: Optional[Path] = None,
        warn: Optional[Callable[[str], None]] = None,
    ):
        self.path = Path(path) if path is not None else default_state_path()
        self._warn = warn
        self._lock = threading.Lock()
        self._last_used: Dict[str, float] = self._read()

    # -- io

    def _read(self) -> Dict[str, float]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return {}  # missing/unreadable: start empty
        try:
            data: Any = json.loads(raw)
        except ValueError:
            self._warn_soft(f"ignoring corrupt state file {self.path}")
            return {}
        if not isinstance(data, dict):
            return {}
        entries = data.get("lastUsed")
        if not isinstance(entries, dict):
            return {}
        out: Dict[str, float] = {}
        for key, value in entries.items():
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                continue  # skip the bad row, keep the rest
        return out

    def _write(self) -> None:
        payload = {
            "version": 1,
            "updatedAt": time.time(),
            "lastUsed": self._last_used,
        }
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, self.path)
        except OSError as exc:
            self._warn_soft(f"could not persist state to {self.path}: {exc}")
            try:
                tmp.unlink()
            except OSError:
                pass

    def _warn_soft(self, message: str) -> None:
        if self._warn is not None:
            try:
                self._warn(message)
            except Exception:
                pass

    # -- api

    def last_used(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._last_used)

    def mark_used(self, model_id: str, when: Optional[float] = None) -> float:
        stamp = time.time() if when is None else float(when)
        with self._lock:
            self._last_used[model_id] = stamp
            self._write()
        return stamp

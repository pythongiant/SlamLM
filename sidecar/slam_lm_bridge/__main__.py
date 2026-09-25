"""``python -m slam_lm_bridge`` — the same entry point as ``.server``."""

from __future__ import annotations

from .server import main

if __name__ == "__main__":
    raise SystemExit(main())

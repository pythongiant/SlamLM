"""SlamLM Python bridge.

Local MLX inference plus telemetry for the macOS menu bar app. The wire
contract lives in ``menubar/PROTOCOL.md``; ``protocol.py`` is its Python
mirror. Entry points::

    python -m slam_lm_bridge            # same as below
    python -m slam_lm_bridge.server
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

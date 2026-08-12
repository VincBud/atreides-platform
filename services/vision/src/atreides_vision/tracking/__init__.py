"""Multi-object tracking: the Atreides interface and its adapters.

Import :class:`MultiObjectTracker` from here. Concrete adapters
(:mod:`atreides_vision.tracking.bytetrack`) are imported explicitly by the
composition root (the CLI) and nowhere else.
"""

from __future__ import annotations

from atreides_vision.tracking.base import MultiObjectTracker

__all__ = ["MultiObjectTracker"]

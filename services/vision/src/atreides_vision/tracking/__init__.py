"""Multi-object tracking: the Atreides interface and its adapters.

Import :class:`MultiObjectTracker` from here. Concrete adapters
(:mod:`atreides_vision.tracking.bytetrack`, :mod:`atreides_vision.tracking.ocsort`)
are selected by the composition root (the CLI) and imported nowhere else.

Both current adapters are motion-only: no appearance descriptors, no
re-identification, no pixels used for identity. Track ids stay ephemeral and
session-local.
"""

from __future__ import annotations

from atreides_vision.tracking.base import MultiObjectTracker

__all__ = ["MultiObjectTracker"]

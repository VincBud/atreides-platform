"""The Atreides multi-object tracker interface.

Track ids produced by implementations are ephemeral and session-local. They
associate boxes across frames of one video from one camera and mean nothing
outside that scope — see :mod:`atreides_vision.models`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from atreides_vision.models import Detection, Track

__all__ = ["MultiObjectTracker"]


@runtime_checkable
class MultiObjectTracker(Protocol):
    """Associates per-frame detections into tracks over time."""

    @property
    def name(self) -> str:
        """Human-readable tracker name, used in startup logs."""
        ...

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        """Advance the tracker by one frame.

        Args:
            detections: People detected in this frame, in source-frame pixels.
            frame: The ``(H, W, 3)`` BGR frame the detections came from.
                Appearance-based trackers need it; motion-only trackers ignore
                it. It is part of the interface so appearance-based trackers can
                be dropped in later without changing the pipeline.

        Returns:
            Confirmed tracks for this frame.
        """
        ...

    def reset(self) -> None:
        """Drop all track state. Called between videos/sessions."""
        ...

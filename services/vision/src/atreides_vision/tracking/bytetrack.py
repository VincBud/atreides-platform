"""ByteTrack adapter — the default :class:`MultiObjectTracker` implementation.

``trackers`` is imported here and nowhere outside
:mod:`atreides_vision.tracking`. The conversion in both directions

``list[Detection]`` -> ``sv.Detections`` -> ByteTrack -> ``sv.Detections`` -> ``list[Track]``

is shared with the other adapters via :mod:`atreides_vision.tracking.conversion`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from atreides_vision.config import TrackerConfig
from atreides_vision.errors import TrackerError
from atreides_vision.models import Detection, Track
from atreides_vision.tracking.conversion import (
    UNASSIGNED_TRACKER_ID,
    detections_to_supervision,
    supervision_to_tracks,
)

logger = logging.getLogger(__name__)

__all__ = ["UNASSIGNED_TRACKER_ID", "ByteTrackBackend", "ByteTrackTracker"]


class ByteTrackBackend(Protocol):
    """The slice of the third-party ByteTrack surface this adapter uses."""

    def update(self, detections: Any) -> Any: ...

    def reset(self) -> None: ...


def _default_backend_factory(config: TrackerConfig) -> ByteTrackBackend:
    """Construct a real ByteTrack tracker from the ``trackers`` package."""
    try:
        from trackers import ByteTrackTracker as _ByteTrackTracker
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise TrackerError(
            "The 'trackers' package is not installed. Run 'uv sync' from the repository root."
        ) from exc

    try:
        return _ByteTrackTracker(
            lost_track_buffer=config.lost_track_buffer,
            frame_rate=config.frame_rate,
            track_activation_threshold=config.track_activation_threshold,
            minimum_consecutive_frames=config.minimum_consecutive_frames,
            minimum_iou_threshold=config.minimum_iou_threshold,
        )
    except Exception as exc:
        raise TrackerError(f"Failed to initialise ByteTrack: {exc}") from exc


class ByteTrackTracker:
    """Tracks people across frames using ByteTrack.

    ByteTrack is motion-only: it uses a Kalman filter and IoU association and
    never looks at pixels. That is a deliberate fit for this milestone — no
    appearance descriptors are computed, so nothing biometric is derived, and
    ids cannot survive beyond this stream.

    Args:
        config: Tracker settings.
        backend_factory: Builds the underlying ByteTrack instance. Overridden in
            tests with a fake.
    """

    def __init__(
        self,
        config: TrackerConfig,
        backend_factory: Callable[[TrackerConfig], ByteTrackBackend] = _default_backend_factory,
    ) -> None:
        self._config = config
        self._backend = backend_factory(config)

    @property
    def name(self) -> str:
        return "ByteTrack"

    @property
    def config(self) -> TrackerConfig:
        return self._config

    @property
    def backend(self) -> ByteTrackBackend:
        """The underlying tracker, for read-only diagnostics.

        Exposed so :mod:`atreides_vision.tracking.diagnostics` can copy out
        internal tracklet state. Nothing in the pipeline writes through it.
        """
        return self._backend

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        """Advance ByteTrack by one frame. See :class:`MultiObjectTracker`.

        ``frame`` is accepted for interface compatibility and deliberately not
        forwarded: ByteTrack performs no camera-motion compensation and warns
        when handed a frame.
        """
        del frame  # ByteTrack is motion-only; passing a frame would only warn.

        sv_detections = detections_to_supervision(detections)
        try:
            tracked = self._backend.update(sv_detections)
        except Exception as exc:
            raise TrackerError(f"ByteTrack update failed: {exc}") from exc
        return supervision_to_tracks(tracked)

    def reset(self) -> None:
        """Drop all ByteTrack state."""
        self._backend.reset()

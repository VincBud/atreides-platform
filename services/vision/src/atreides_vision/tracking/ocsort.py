"""OC-SORT adapter — an alternative :class:`MultiObjectTracker` implementation.

Added to A/B against ByteTrack on fixed-CCTV footage, where a person briefly
occluded by another often comes back with a new track id. OC-SORT is designed
for exactly that failure: it scores association on direction consistency as
well as IoU, and recovers a track from its last observation before the gap
rather than from the Kalman filter's drifted prediction.

Like ByteTrack it is **motion-only**. No appearance descriptors, no re-identification,
no pixels used for identity. The frame is accepted for interface compatibility
and deliberately not forwarded. Track ids remain ephemeral and session-local.

Conversion in both directions is shared with the ByteTrack adapter via
:mod:`atreides_vision.tracking.conversion`, so an A/B compares trackers and
nothing else.
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
    detections_to_supervision,
    supervision_to_tracks,
)

logger = logging.getLogger(__name__)

__all__ = ["OcSortBackend", "OcSortTracker"]


class OcSortBackend(Protocol):
    """The slice of the third-party OC-SORT surface this adapter uses."""

    def update(self, detections: Any) -> Any: ...

    def reset(self) -> None: ...


def _default_backend_factory(config: TrackerConfig) -> OcSortBackend:
    """Construct a real OC-SORT tracker from the ``trackers`` package."""
    try:
        from trackers import OCSORTTracker as _OCSORTTracker
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise TrackerError(
            "The 'trackers' package is not installed. Run 'uv sync' from the repository root."
        ) from exc

    try:
        return _OCSORTTracker(
            lost_track_buffer=config.lost_track_buffer,
            frame_rate=config.frame_rate,
            minimum_consecutive_frames=config.minimum_consecutive_frames,
            minimum_iou_threshold=config.ocsort_minimum_iou_threshold,
            direction_consistency_weight=config.direction_consistency_weight,
            high_conf_det_threshold=config.high_conf_det_threshold,
            delta_t=config.delta_t,
        )
    except Exception as exc:
        raise TrackerError(f"Failed to initialise OC-SORT: {exc}") from exc


class OcSortTracker:
    """Tracks people across frames using OC-SORT.

    Note that OC-SORT has no equivalent of ByteTrack's
    ``track_activation_threshold``; ``high_conf_det_threshold`` is the
    confidence above which a detection may start a new track.

    Args:
        config: Tracker settings.
        backend_factory: Builds the underlying OC-SORT instance. Overridden in
            tests with a fake.
    """

    def __init__(
        self,
        config: TrackerConfig,
        backend_factory: Callable[[TrackerConfig], OcSortBackend] = _default_backend_factory,
    ) -> None:
        self._config = config
        self._backend = backend_factory(config)

    @property
    def name(self) -> str:
        return "OC-SORT"

    @property
    def config(self) -> TrackerConfig:
        return self._config

    @property
    def backend(self) -> OcSortBackend:
        """The underlying tracker, for read-only diagnostics.

        Exposed so :mod:`atreides_vision.tracking.diagnostics` can copy out
        internal tracklet state. Nothing in the pipeline writes through it.
        """
        return self._backend

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        """Advance OC-SORT by one frame. See :class:`MultiObjectTracker`.

        ``frame`` is accepted for interface compatibility and deliberately not
        forwarded: OC-SORT performs no camera-motion compensation and warns
        when handed a frame.
        """
        del frame  # OC-SORT is motion-only; passing a frame would only warn.

        sv_detections = detections_to_supervision(detections)
        try:
            tracked = self._backend.update(sv_detections)
        except Exception as exc:
            raise TrackerError(f"OC-SORT update failed: {exc}") from exc
        return supervision_to_tracks(tracked)

    def reset(self) -> None:
        """Drop all OC-SORT state."""
        self._backend.reset()

"""ByteTrack adapter — the initial :class:`MultiObjectTracker` implementation.

This module is the **only** place in Atreides allowed to import ``trackers`` or
``supervision``. It owns both directions of the conversion:

``list[Detection]`` -> ``sv.Detections`` -> ByteTrack -> ``sv.Detections`` -> ``list[Track]``
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from atreides_vision.config import TrackerConfig
from atreides_vision.errors import TrackerError
from atreides_vision.models import BoundingBox, Detection, Track, detections_to_arrays

logger = logging.getLogger(__name__)

__all__ = ["ByteTrackBackend", "ByteTrackTracker"]

#: ByteTrack marks detections it did not assign to a confirmed track with -1.
UNASSIGNED_TRACKER_ID = -1


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

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        """Advance ByteTrack by one frame. See :class:`MultiObjectTracker`.

        ``frame`` is accepted for interface compatibility and deliberately not
        forwarded: ByteTrack performs no camera-motion compensation and warns
        when handed a frame.
        """
        del frame  # ByteTrack is motion-only; passing a frame would only warn.

        sv_detections = self._to_supervision(detections)
        try:
            tracked = self._backend.update(sv_detections)
        except Exception as exc:
            raise TrackerError(f"ByteTrack update failed: {exc}") from exc
        return self._to_tracks(tracked)

    def reset(self) -> None:
        """Drop all ByteTrack state."""
        self._backend.reset()

    @staticmethod
    def _to_supervision(detections: list[Detection]) -> Any:
        """Convert Atreides detections into ``sv.Detections``."""
        import supervision as sv

        if not detections:
            return sv.Detections.empty()
        xyxy, confidence, class_id = detections_to_arrays(detections)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    @staticmethod
    def _to_tracks(tracked: Any) -> list[Track]:
        """Convert ByteTrack's ``sv.Detections`` output into Atreides tracks.

        Detections ByteTrack could not confirm carry ``tracker_id == -1`` and are
        dropped: the pipeline only ever sees confirmed tracks.
        """
        tracker_ids = getattr(tracked, "tracker_id", None)
        if tracker_ids is None:
            return []
        tracker_ids = np.asarray(tracker_ids).reshape(-1)
        if tracker_ids.size == 0:
            return []

        xyxy = np.asarray(tracked.xyxy, dtype=np.float64).reshape(-1, 4)
        raw_confidence = getattr(tracked, "confidence", None)
        confidence: np.ndarray | None = None
        if raw_confidence is not None:
            confidence = np.asarray(raw_confidence, dtype=np.float64).reshape(-1)

        tracks: list[Track] = []
        for index in range(tracker_ids.size):
            track_id = int(tracker_ids[index])
            if track_id == UNASSIGNED_TRACKER_ID:
                continue
            try:
                box = BoundingBox.from_xyxy(xyxy[index])
            except ValueError:
                logger.debug("Skipping malformed tracked box %s", xyxy[index])
                continue
            score: float | None = None
            if confidence is not None and index < confidence.size:
                value = float(confidence[index])
                if np.isfinite(value):
                    score = min(max(value, 0.0), 1.0)
            tracks.append(Track(track_id=track_id, bbox_xyxy=box, confidence=score))
        return tracks

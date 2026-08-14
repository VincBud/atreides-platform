"""Conversion between Atreides domain objects and ``supervision.Detections``.

Shared by every tracker adapter. Both directions live here rather than in each
adapter for a specific reason: when two trackers are compared A/B on the same
clip, any difference in how their input or output is converted would confound
the result. One tested implementation makes the comparison a comparison of
trackers alone.

Together with the adapters, this module confines ``supervision`` and
``trackers`` to :mod:`atreides_vision.tracking`. Nothing outside that package
imports either.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from atreides_vision.models import BoundingBox, Detection, Track, detections_to_arrays

logger = logging.getLogger(__name__)

__all__ = [
    "UNASSIGNED_TRACKER_ID",
    "detections_to_supervision",
    "supervision_to_tracks",
]

#: Trackers mark detections they did not assign to a confirmed track with -1.
UNASSIGNED_TRACKER_ID = -1


def detections_to_supervision(detections: list[Detection]) -> Any:
    """Convert Atreides detections into ``sv.Detections``."""
    import supervision as sv

    if not detections:
        return sv.Detections.empty()
    xyxy, confidence, class_id = detections_to_arrays(detections)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


def supervision_to_tracks(tracked: Any) -> list[Track]:
    """Convert a tracker's ``sv.Detections`` output into Atreides tracks.

    Detections the tracker could not confirm carry ``tracker_id == -1`` and are
    dropped, so the pipeline only ever sees confirmed tracks. Malformed boxes
    are skipped rather than failing the frame.
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

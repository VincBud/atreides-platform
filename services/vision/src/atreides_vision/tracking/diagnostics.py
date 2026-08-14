"""Read-only introspection of tracker internals, for diagnosing ID switches.

Everything here **observes**. Nothing mutates tracker state, nothing changes
association behaviour, and the third-party ``trackers`` package is untouched.
The pipeline runs identically whether or not diagnostics are enabled — the only
difference is that some already-computed state is copied out and written to a
file.

What is safe to read
--------------------
``OCSORTTracklet.get_state_bbox()`` is a pure read of the current Kalman
estimate: OC-SORT's own ``update()`` calls it to build the association matrix,
so calling it again returns the same value without advancing the filter.
``predict()`` is *not* safe and is never called here.

The remaining fields (``age``, ``time_since_update``, ``last_observation``,
``tracker_id``, ...) are plain attributes on the tracklet.

What is **not** available
-------------------------
The tracker computes its IoU and direction-consistency matrices as locals
inside ``update()`` and does not retain them, so per-pair association costs
cannot be read out without modifying the package. This module therefore records
the *inputs* to that decision — every detection, every track's predicted box and
last observation — from which IoU can be recomputed offline. No association
cost is invented here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self

import numpy as np

from atreides_vision.errors import MetricsOutputError
from atreides_vision.models import Detection, Track

logger = logging.getLogger(__name__)

__all__ = [
    "InternalTrackSnapshot",
    "JsonlTrackingDebugWriter",
    "TrackingDebugRecord",
    "TrackingDebugWriter",
    "snapshot_internal_tracks",
]


@dataclass(frozen=True, slots=True)
class InternalTrackSnapshot:
    """One tracklet's state as the tracker holds it, after a frame's update.

    Every field is copied from the tracker; none is inferred, except ``state``
    which is a label derived solely from ``time_since_update``.
    """

    #: Tracker-assigned id. ``-1`` means the tracklet exists but has not yet
    #: reached ``minimum_consecutive_frames`` and so is not reported publicly.
    track_id: int
    #: ``"tracked"`` when this frame updated it, ``"lost"`` when it did not.
    state: str
    age: int
    time_since_update: int
    #: Last real detection assigned to this track, ``[x1, y1, x2, y2]``.
    last_observation_bbox: list[float] | None
    #: Current Kalman estimate — the box the tracker associates against.
    predicted_bbox: list[float] | None
    #: Consecutive matched frames; drives maturity and thus id assignment.
    consecutive_updates: int
    #: OC-SORT's ORU flag: ``False`` once the track has missed a frame and its
    #: filter state has been frozen for later re-update.
    observed: bool | None
    #: Normalised direction vector ``[dy, dx]``, or ``None`` before two matches.
    velocity: list[float] | None

    def to_record(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "state": self.state,
            "age": self.age,
            "time_since_update": self.time_since_update,
            "last_observation_bbox": self.last_observation_bbox,
            "predicted_bbox": self.predicted_bbox,
            "consecutive_updates": self.consecutive_updates,
            "observed": self.observed,
            "velocity": self.velocity,
        }


def _as_bbox(value: Any) -> list[float] | None:
    """Convert a tracker bbox to a plain rounded list, or ``None``."""
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < 4 or not np.all(np.isfinite(array[:4])):
        return None
    return [round(float(v), 2) for v in array[:4]]


def snapshot_internal_tracks(tracker: object) -> tuple[InternalTrackSnapshot, ...]:
    """Copy out the internal tracklets of a tracker adapter, if it exposes them.

    Duck-typed and defensive: a tracker that does not expose a backend, or whose
    tracklets lack these attributes, yields an empty tuple rather than raising.
    Diagnostics must never be able to break a run.
    """
    backend = getattr(tracker, "backend", None)
    if backend is None:
        return ()
    tracklets = getattr(backend, "tracks", None)
    if tracklets is None:
        return ()

    snapshots: list[InternalTrackSnapshot] = []
    for tracklet in tracklets:
        try:
            time_since_update = int(getattr(tracklet, "time_since_update", 0))
            get_state_bbox = getattr(tracklet, "get_state_bbox", None)
            # Pure read: OC-SORT itself calls this to build its association
            # matrix. predict() would advance the filter and is never called.
            predicted = _as_bbox(get_state_bbox()) if callable(get_state_bbox) else None
            velocity = getattr(tracklet, "velocity", None)
            snapshots.append(
                InternalTrackSnapshot(
                    track_id=int(getattr(tracklet, "tracker_id", -1)),
                    state="tracked" if time_since_update == 0 else "lost",
                    age=int(getattr(tracklet, "age", 0)),
                    time_since_update=time_since_update,
                    last_observation_bbox=_as_bbox(getattr(tracklet, "last_observation", None)),
                    predicted_bbox=predicted,
                    consecutive_updates=int(
                        getattr(tracklet, "number_of_successful_consecutive_updates", 0)
                    ),
                    observed=getattr(tracklet, "_observed", None),
                    velocity=None
                    if velocity is None
                    else [round(float(v), 4) for v in np.asarray(velocity).reshape(-1)],
                )
            )
        except Exception:  # diagnostics must never break a run
            logger.debug("Skipping a tracklet that could not be introspected", exc_info=True)
    return tuple(snapshots)


def _detection_records(detections: tuple[Detection, ...]) -> list[dict[str, Any]]:
    return [
        {
            "bbox": [round(v, 2) for v in detection.bbox_xyxy.as_tuple()],
            "confidence": round(detection.confidence, 4),
        }
        for detection in detections
    ]


@dataclass(frozen=True, slots=True)
class TrackingDebugRecord:
    """Everything observable about one frame's tracking decision.

    ``detections`` always means *what entered the tracker*, which is what it
    meant before deduplication existed — so existing readers keep working.
    When deduplication ran, the record additionally carries the pre-suppression
    detections and what was removed; when it did not, the record's shape is
    unchanged from before the feature existed.
    """

    frame_index: int
    detections: tuple[Detection, ...]
    reported_tracks: tuple[Track, ...]
    internal_tracks: tuple[InternalTrackSnapshot, ...]
    #: Detections before suppression. ``None`` when deduplication did not run.
    raw_detections: tuple[Detection, ...] | None = None
    #: Detections suppressed this frame. Empty unless deduplication ran.
    removed_detections: tuple[Detection, ...] = ()

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "frame": self.frame_index,
            "detections": _detection_records(self.detections),
            "reported_tracks": [
                {
                    "track_id": track.track_id,
                    "bbox": [round(v, 2) for v in track.bbox_xyxy.as_tuple()],
                    "confidence": None if track.confidence is None else round(track.confidence, 4),
                }
                for track in self.reported_tracks
            ],
            "internal_tracks": [snapshot.to_record() for snapshot in self.internal_tracks],
        }
        if self.raw_detections is not None:
            # Present only when deduplication ran, so a run without it produces
            # exactly the record shape earlier tooling already reads.
            record["raw_detections"] = _detection_records(self.raw_detections)
            record["tracker_input_detections"] = _detection_records(self.detections)
            record["removed_detections"] = _detection_records(self.removed_detections)
            record["detections_removed_by_deduplication"] = len(self.removed_detections)
        return record


class TrackingDebugWriter(Protocol):
    """A destination for per-frame tracking diagnostics."""

    def write(self, record: TrackingDebugRecord) -> None: ...

    def release(self) -> None: ...


class JsonlTrackingDebugWriter:
    """Appends one JSON object per processed frame to a ``.jsonl`` file.

    Unlike the zone metrics stream this is not sampled: diagnosing a single
    frame's association needs every frame around it.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._records_written = 0
        self._released = False

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("w", encoding="utf-8")
        except OSError as exc:
            raise MetricsOutputError(
                f"Could not open tracking debug output {self._path}: {exc}"
            ) from exc

    @property
    def path(self) -> Path:
        return self._path

    @property
    def records_written(self) -> int:
        return self._records_written

    def write(self, record: TrackingDebugRecord) -> None:
        if self._released:
            raise MetricsOutputError(f"Cannot write to a released debug writer: {self._path}")
        try:
            self._handle.write(json.dumps(record.to_record(), separators=(",", ":")) + "\n")
        except OSError as exc:
            raise MetricsOutputError(
                f"Could not write tracking debug to {self._path}: {exc}"
            ) from exc
        self._records_written += 1

    def release(self) -> None:
        """Flush and close. Idempotent, and safe after an exception."""
        if not getattr(self, "_released", False):
            self._handle.close()
            self._released = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

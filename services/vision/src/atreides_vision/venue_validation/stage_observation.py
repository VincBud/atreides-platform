"""V0.4.10a — camera-local perception/tracking observability seam.

Nothing here changes what the camera-local pipeline decides. This module reads
facts the pipeline has already computed — via the existing, already-committed
:class:`~atreides_vision.tracking.diagnostics.TrackingDebugRecord` diagnostics
seam — and reshapes them, per camera per frame, into an immutable, generic
snapshot a validation observer can inspect. It is **observability only**:

* No WILDTRACK, no ground truth, no scoring, no coverage/attribution/residual
  metric. That is V0.4.10b's job, not this module's.
* No RF-DETR, ``trackers``-library, ``supervision``, ``torch`` or ``ndarray``
  type ever appears in this module's public surface — everything here is
  already-neutral Atreides domain data
  (:class:`~atreides_vision.models.Detection`,
  :class:`~atreides_vision.models.Track`,
  :class:`~atreides_vision.tracking.diagnostics.InternalTrackSnapshot`) turned
  into even plainer value types.
* No cadence experiment, no signed residual vector (V0.4.10b), no dedup rule
  change: detections are reported exactly as
  :class:`~atreides_vision.tracking.diagnostics.TrackingDebugRecord` already
  recorded them, truthfully partitioned into kept and suppressed.

Position is the one piece of derived data this module adds, and it adds
nothing new: every position is the existing
:func:`~atreides_vision.behavior.coordinates.bottom_center_anchor` floor
anchor, run through the caller's own already-built
:class:`~atreides_vision.behavior.coordinates.MotionProjector` — the same
anchor and the same projector the behaviour layer already uses for the
subjects that survive tracking. An anchor beyond the calibration horizon
raises :class:`~atreides_vision.errors.ProjectionError`; this module catches
that narrowly and records ``position=None``, exactly the pattern
:meth:`~atreides_vision.behavior.analyzer.BehaviorAnalyzer.update` already
uses. No pixel, no calibration object and no image ever crosses into a
snapshot — only the :class:`~atreides_vision.behavior.coordinates.MotionSpace`
the projector itself declares.
"""

from __future__ import annotations

from dataclasses import dataclass

from atreides_vision.behavior.coordinates import (
    MotionPoint,
    MotionProjector,
    MotionSpace,
    bottom_center_anchor,
)
from atreides_vision.errors import ProjectionError, ValidationError
from atreides_vision.models import BoundingBox
from atreides_vision.timing import StreamFrame, StreamRef
from atreides_vision.tracking.diagnostics import TrackingDebugRecord

__all__ = [
    "CameraStageSnapshot",
    "LatestTrackingRecord",
    "StageDetectionSnapshot",
    "StageTrackSnapshot",
    "StageTrackletSnapshot",
    "build_camera_stage_snapshot",
]


def _project(bbox: BoundingBox | None, projector: MotionProjector) -> MotionPoint | None:
    """The floor position of ``bbox``'s bottom-centre anchor, or ``None``.

    ``None`` covers two truthful cases identically: there was no box to
    project, or the anchor fell beyond the calibration horizon. Neither is a
    fabricated substitute position.
    """
    if bbox is None:
        return None
    try:
        return projector.project(bottom_center_anchor(bbox))
    except ProjectionError:
        return None


def _bbox_from_xyxy(values: list[float] | None) -> BoundingBox | None:
    if values is None:
        return None
    x1, y1, x2, y2 = values
    return BoundingBox.from_xyxy((x1, y1, x2, y2))


@dataclass(frozen=True, slots=True)
class StageDetectionSnapshot:
    """One detection as it stood between detection and tracking.

    ``suppressed`` is read straight from
    :class:`~atreides_vision.tracking.diagnostics.TrackingDebugRecord`'s own
    partition of what deduplication removed — never re-derived by re-running
    the dedup rule here.
    """

    bbox: BoundingBox
    confidence: float
    #: The bottom-centre anchor projected into the camera's declared motion
    #: space, or ``None`` if it fell beyond the calibration horizon.
    position: MotionPoint | None
    #: ``True`` for a detection deduplication removed before the tracker saw
    #: it. Always ``False`` when deduplication did not run.
    suppressed: bool


@dataclass(frozen=True, slots=True)
class StageTrackSnapshot:
    """One publicly reported track, for this frame."""

    track_id: int
    bbox: BoundingBox
    confidence: float | None
    position: MotionPoint | None


@dataclass(frozen=True, slots=True)
class StageTrackletSnapshot:
    """One internal tracker tracklet, whether or not it is public yet.

    ``track_id`` of ``-1`` is a real, tracker-assigned fact — a tracklet the
    tracker holds but has not yet matured into a publicly reported
    :class:`~atreides_vision.models.Track` — and is preserved verbatim rather
    than filtered out, so a caller can see confirmation delay happening.
    """

    track_id: int
    state: str
    age: int
    time_since_update: int
    consecutive_updates: int
    observed: bool | None
    last_observation_position: MotionPoint | None
    predicted_position: MotionPoint | None


@dataclass(frozen=True, slots=True)
class CameraStageSnapshot:
    """One camera's perception/tracking stages, for one frame it contributed.

    ``space`` is the projector's own declared
    :class:`~atreides_vision.behavior.coordinates.MotionSpace` — never
    inferred from whether any position happened to project — so an
    uncalibrated camera always reports ``NORMALIZED_IMAGE`` here, even on a
    frame where every position is ``None``.
    """

    stream: StreamRef
    frame_index: int
    venue_seconds: float
    space: MotionSpace
    detections: tuple[StageDetectionSnapshot, ...]
    tracks: tuple[StageTrackSnapshot, ...]
    tracklets: tuple[StageTrackletSnapshot, ...]


class LatestTrackingRecord:
    """A single-slot, synchronous capture of one frame's tracking diagnostics.

    Implements :class:`~atreides_vision.tracking.diagnostics.TrackingDebugWriter`
    so it can be injected into :class:`~atreides_vision.video.processor.VideoProcessor`
    exactly like :class:`~atreides_vision.tracking.diagnostics.JsonlTrackingDebugWriter`
    already is. There is no queue, no history and no async handoff: the
    processor calls :meth:`write` synchronously, once, from inside
    ``process()``, before ``process()`` returns — so a caller that calls
    :meth:`take` immediately after ``process()`` returns is guaranteed to
    retrieve exactly that call's record, deterministically, with no race.

    :meth:`take` is destructive: it clears the slot, so a caller that calls it
    twice for one frame, or that never wrote a frame first, gets a loud
    :class:`~atreides_vision.errors.ValidationError` rather than a silently
    stale or empty record.
    """

    def __init__(self) -> None:
        self._record: TrackingDebugRecord | None = None

    def write(self, record: TrackingDebugRecord) -> None:
        self._record = record

    def release(self) -> None:
        """No-op: there is no handle to flush or close."""

    def take(self) -> TrackingDebugRecord:
        """The record from the most recent :meth:`write`, consuming it.

        Raises:
            ValidationError: if no record is waiting — the processor has not
                run for this frame, or this frame's record was already taken.
        """
        if self._record is None:
            raise ValidationError(
                "LatestTrackingRecord.take() found no record waiting. A caller must "
                "process exactly one frame through the wired VideoProcessor, and take() "
                "exactly once, before taking again."
            )
        record, self._record = self._record, None
        return record


def build_camera_stage_snapshot(
    record: TrackingDebugRecord,
    frame: StreamFrame,
    projector: MotionProjector,
) -> CameraStageSnapshot:
    """Freeze one frame's tracking diagnostics into a :class:`CameraStageSnapshot`.

    Pure and read-only: it neither calls ``predict()`` nor mutates the tracker,
    and it neither detects, tracks nor associates anything itself. Every
    detection/track/tracklet fact is copied straight out of ``record``; the
    only thing computed here is each fact's projected floor position.

    Args:
        record: This frame's diagnostics, as captured by a
            :class:`LatestTrackingRecord` wired into the
            :class:`~atreides_vision.video.processor.VideoProcessor` that
            produced ``frame``.
        frame: The exact frame ``record`` was produced from.
        projector: This stream's motion projector — calibrated or normalized,
            whichever the runtime actually built for this camera.

    Raises:
        ValidationError: if ``record`` was not produced from ``frame`` — a
            caller pairing two runtime objects that were never produced
            together, never a legitimate observation outcome.
    """
    if record.frame_index != frame.frame_index:
        raise ValidationError(
            f"Cannot build a camera stage snapshot for stream '{frame.stream}': the debug "
            f"record's frame_index ({record.frame_index}) does not match the frame's "
            f"({frame.frame_index}). This record was not produced from this frame."
        )

    if record.raw_detections is not None:
        # Deduplication ran: report every detection it saw, truthfully
        # partitioned by the record's own account of what it removed.
        source_detections = record.raw_detections
        removed_ids = {id(detection) for detection in record.removed_detections}
    else:
        # Deduplication did not run: nothing was ever a candidate for
        # suppression, so every detection that reached the tracker is kept.
        source_detections = record.detections
        removed_ids = set()

    detections = tuple(
        StageDetectionSnapshot(
            bbox=detection.bbox_xyxy,
            confidence=detection.confidence,
            position=_project(detection.bbox_xyxy, projector),
            suppressed=id(detection) in removed_ids,
        )
        for detection in source_detections
    )

    tracks = tuple(
        StageTrackSnapshot(
            track_id=track.track_id,
            bbox=track.bbox_xyxy,
            confidence=track.confidence,
            position=_project(track.bbox_xyxy, projector),
        )
        for track in record.reported_tracks
    )

    tracklets = tuple(
        StageTrackletSnapshot(
            track_id=tracklet.track_id,
            state=tracklet.state,
            age=tracklet.age,
            time_since_update=tracklet.time_since_update,
            consecutive_updates=tracklet.consecutive_updates,
            observed=tracklet.observed,
            last_observation_position=_project(
                _bbox_from_xyxy(tracklet.last_observation_bbox), projector
            ),
            predicted_position=_project(_bbox_from_xyxy(tracklet.predicted_bbox), projector),
        )
        for tracklet in record.internal_tracks
    )

    return CameraStageSnapshot(
        stream=frame.stream,
        frame_index=frame.frame_index,
        venue_seconds=frame.venue_seconds,
        space=projector.space,
        detections=detections,
        tracks=tracks,
        tracklets=tracklets,
    )

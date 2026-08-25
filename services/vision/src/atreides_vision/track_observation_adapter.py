"""Runtime -> ``TrackObservation`` adapter — V0.4.7.

V0.4.0 through V0.4.6 built the whole venue-time, association, fusion and
identity stack, and every one of them documented the same acknowledged gap:
:mod:`atreides_vision.venue_pipeline`'s own module docstring names it
outright — "Building an observation is the caller's responsibility; this
pipeline's contract begins at the observation." Nothing before this milestone
converts an already-running camera-local pipeline's own state — a decoded
:class:`~atreides_vision.timing.StreamFrame`, a projected behaviour
snapshot, a camera's calibration and integrity state — into the exact
:class:`~atreides_vision.association.models.TrackObservation` that
:class:`~atreides_vision.venue_pipeline.VenuePipeline` consumes.

This module is that adapter, and nothing more.

Where this sits
----------------
::

    StreamFrame (timing)          -- V0.4.0, unchanged
    BehaviorSubjectSnapshot        -- V0.3, unchanged
    GroundPlaneProjector           -- V0.3.1, unchanged
    CalibrationIntegrityGuard      -- V0.3.3, unchanged
        \\
         -> build_track_observation()   <- V0.4.7 builds this
        /
    TrackObservation                -- V0.4.1 contract, unchanged
      -> VenuePipeline.process_instant()  -- V0.4.4, unchanged

Every field this module writes already exists somewhere in the runtime named
above; nothing here measures, projects, associates, or invents a value. Where
a required field genuinely cannot be read from that runtime, the caller's
input is rejected rather than guessed.

Field -> source of truth
-------------------------
- ``subject`` — ``snapshot.subject``, exactly as
  :meth:`~atreides_vision.behavior.analyzer.BehaviorAnalyzer.update` already
  constructed it from the camera-local tracker's own ``(camera_id,
  track_id)``. Never remapped, never a new identifier.
- ``stream`` — ``frame.stream``, the :class:`~atreides_vision.timing.StreamRef`
  already carried end to end by :class:`~atreides_vision.timing.StreamFrame`
  since V0.4.0. Its ``stream_epoch`` comes from the venue/session manifest
  that produced ``frame`` — this adapter never derives one.
- ``frame_index`` / ``venue_seconds`` / ``timing`` — read from ``frame``
  unchanged. ``atreides_vision.video.processor.VideoProcessor.process``
  already threads ``frame.venue_seconds`` into the behaviour analyzer as its
  own ``timestamp_seconds``, which is why ``snapshot.timestamp_seconds`` is
  cross-checked against ``frame.venue_seconds`` below rather than trusted on
  its own: the check catches a caller pairing a snapshot with the wrong
  frame, it does not compute a second value.
- ``position`` — ``snapshot.position``, the same
  :class:`~atreides_vision.behavior.coordinates.MotionPoint` the analyzer's
  own projector already produced. Never re-projected.
- ``spatial`` — a fresh
  :class:`~atreides_vision.association.models.CameraSpatialContext` built
  from ``calibration`` (``None`` for the uncalibrated, normalized-image
  case) and ``integrity_guard``. ``camera_id`` comes from ``frame.stream``,
  never from the projector. ``trust`` is the documented
  :class:`~atreides_vision.association.models.GeometryTrust` narrowing of
  :class:`~atreides_vision.calibration.integrity.models.IntegrityState`,
  honouring the guard's own sticky suppression: once a guarded run has
  withheld metric output for detected drift, this adapter reports
  ``UNTRUSTED`` even if a later check happens to read consistent again,
  because the guard itself never re-extends trust without a re-calibration.

What this milestone does not do
--------------------------------
No association, no fusion, no venue-local identity, no ReID or appearance
matching, no trajectory smoothing or filtering, no topology or epoch
inference, no new calibration algorithm, and no repair of a calibration or
integrity finding already reported. A structurally impossible combination
(mismatched cameras, a snapshot paired with a frame it was not built from, a
metric position without matching calibration) is rejected with
:class:`~atreides_vision.errors.AssociationError` — the same error
``TrackObservation`` itself already raises for the same class of problem —
never silently coerced into a value that looks valid.
"""

from __future__ import annotations

from atreides_vision.association import CameraSpatialContext, GeometryTrust, TrackObservation
from atreides_vision.association.models import VenueRegion
from atreides_vision.behavior.models import BehaviorSubjectSnapshot
from atreides_vision.calibration import GroundPlaneProjector
from atreides_vision.calibration.integrity import CalibrationIntegrityGuard, IntegrityState
from atreides_vision.errors import AssociationError
from atreides_vision.timing import StreamFrame

__all__ = ["build_track_observation"]


def _geometry_trust(integrity_guard: CalibrationIntegrityGuard | None) -> GeometryTrust:
    """The documented :class:`GeometryTrust` narrowing of one camera's integrity state.

    No guard deployed is not evidence the camera moved, and it is not evidence
    it did not: it collapses to ``UNVERIFIED``, exactly like ``UNCHECKED`` and
    ``INSUFFICIENT_EVIDENCE`` already do. A guard that has permanently withheld
    metric output because guarded-mode drift was detected reports ``UNTRUSTED``
    regardless of what its most recent individual check read, because
    :attr:`CalibrationIntegrityGuard.metric_output_allowed` never re-extends
    trust on its own — only a re-calibration does, and that is an operator
    action this adapter does not perform.
    """
    if integrity_guard is None:
        return GeometryTrust.UNVERIFIED
    if integrity_guard.state is IntegrityState.DRIFT_DETECTED:
        return GeometryTrust.UNTRUSTED
    if not integrity_guard.metric_output_allowed:
        return GeometryTrust.UNTRUSTED
    if integrity_guard.state is IntegrityState.CONSISTENT:
        return GeometryTrust.VERIFIED
    return GeometryTrust.UNVERIFIED


def build_track_observation(
    snapshot: BehaviorSubjectSnapshot,
    frame: StreamFrame,
    *,
    calibration: GroundPlaneProjector | None = None,
    integrity_guard: CalibrationIntegrityGuard | None = None,
) -> TrackObservation:
    """Adapt one camera-local behaviour snapshot into a venue ``TrackObservation``.

    Args:
        snapshot: One subject's already-projected observation for this frame,
            as :meth:`~atreides_vision.behavior.analyzer.BehaviorAnalyzer.update`
            produced it.
        frame: The exact :class:`~atreides_vision.timing.StreamFrame` that
            produced ``snapshot`` — the source of the stream, frame index,
            venue time and timing uncertainty this observation carries.
        calibration: This camera's ground-plane projector, when the camera is
            calibrated. ``None`` for a camera still measured in normalized
            image coordinates, matching
            :attr:`~atreides_vision.behavior.coordinates.MotionSpace.GROUND_PLANE_METRES`
            being reserved rather than universal.
        integrity_guard: This camera's calibration integrity guard, when one
            is deployed. ``None`` produces ``GeometryTrust.UNVERIFIED`` —
            nobody confirmed the viewpoint, which is the same honest answer
            :class:`~atreides_vision.association.models.CameraSpatialContext`
            already defaults to.

    Raises:
        AssociationError: if ``snapshot`` and ``frame`` disagree about which
            camera or which frame produced them, if a calibrated projector is
            supplied for a normalized-space snapshot (or omitted for a
            metric one), or if ``calibration`` names a different camera than
            ``frame``. Every one of these is a caller pairing two runtime
            objects that were never produced together, not an ordinary
            association outcome — building the observation anyway would
            silently attach one camera's calibration or venue time to
            another's track.
    """
    camera_id = frame.stream.camera_id
    if snapshot.subject.camera_id != camera_id:
        raise AssociationError(
            f"Cannot build a TrackObservation for subject '{snapshot.subject}' from stream "
            f"'{frame.stream}': the subject's camera does not match the frame's. A snapshot "
            "and a frame that disagree about the camera were never produced by the same "
            "camera-local pipeline pass."
        )
    if snapshot.frame_index != frame.frame_index:
        raise AssociationError(
            f"Cannot build a TrackObservation for subject '{snapshot.subject}': the snapshot's "
            f"frame_index ({snapshot.frame_index}) does not match the frame's "
            f"({frame.frame_index}). This snapshot was not produced from this frame."
        )
    if snapshot.timestamp_seconds != frame.venue_seconds:
        raise AssociationError(
            f"Cannot build a TrackObservation for subject '{snapshot.subject}': the snapshot's "
            f"timestamp_seconds ({snapshot.timestamp_seconds!r}) does not match the frame's "
            f"venue_seconds ({frame.venue_seconds!r}). This snapshot was not produced from this "
            "frame's venue time."
        )
    if calibration is not None and calibration.camera_id != camera_id:
        raise AssociationError(
            f"Cannot build a TrackObservation for camera '{camera_id}': the supplied "
            f"calibration belongs to camera '{calibration.camera_id}'."
        )

    space = snapshot.position.space
    if space.is_physical:
        if calibration is None:
            raise AssociationError(
                f"Subject '{snapshot.subject}' is positioned in {space.value}, which requires a "
                "ground-plane calibration to interpret, but none was supplied. Guessing a scale "
                "would fabricate a metre value nobody measured."
            )
        if calibration.space is not space:
            raise AssociationError(
                f"Subject '{snapshot.subject}' is positioned in {space.value} but the supplied "
                f"calibration projects into {calibration.space.value}."
            )
        venue_id: str | None = calibration.venue_id
        surface_id: str | None = calibration.calibration.surface_id
        calibrated_region: VenueRegion | None = VenueRegion.covering(
            (float(x), float(y)) for x, y in calibration.calibration.venue_points()
        )
    else:
        if calibration is not None:
            raise AssociationError(
                f"Subject '{snapshot.subject}' is positioned in {space.value}, which is not "
                "physical, but a ground-plane calibration was supplied for it. A calibrated "
                "projector and an uncalibrated position cannot both describe the same subject."
            )
        venue_id = None
        surface_id = None
        calibrated_region = None

    spatial = CameraSpatialContext(
        camera_id=camera_id,
        space=space,
        venue_id=venue_id,
        surface_id=surface_id,
        trust=_geometry_trust(integrity_guard),
        calibrated_region=calibrated_region,
    )

    return TrackObservation(
        subject=snapshot.subject,
        stream=frame.stream,
        frame_index=frame.frame_index,
        venue_seconds=frame.venue_seconds,
        timing=frame.timing,
        position=snapshot.position,
        spatial=spatial,
    )

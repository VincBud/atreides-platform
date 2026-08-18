"""Builders for association tests: observations, contexts and a legible config.

Every gate in the association layer is a rule about *context* — which venue
frame, which surface, which clock, which camera. Spelling all of that out at
each call site would bury the one field a test is actually about, so the
builders default everything to a mutually comparable pair of cameras and each
test overrides exactly the field it exercises.

The config here is not the shipped default. Its numbers are chosen so that every
radius in the suite is exact in binary floating point and can be checked by
hand: separation, tolerance and allowance are all small multiples of a half.
"""

from __future__ import annotations

from atreides_vision.association import (
    AssociationGatingConfig,
    CameraSpatialContext,
    GeometryTrust,
    TrackObservation,
    VenueRegion,
)
from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.timing import StreamRef, TimeUncertainty

METRIC = MotionSpace.GROUND_PLANE_METRES
NORMALIZED = MotionSpace.NORMALIZED_IMAGE

#: Half-width used by both cameras unless a test says otherwise. With two
#: streams the slack is 0.5 s, so a co-timed pair gets a motion allowance of
#: 0.5 * 2.0 = 1.0 m and an admissible radius of exactly 2.0 m.
HALF_WIDTH_SECONDS = 0.25

GATING = AssociationGatingConfig(
    max_pairing_gap_seconds=0.5,
    max_subject_speed_m_per_s=2.0,
    position_tolerance_m=1.0,
    ambiguity_margin_m=0.5,
)


def context(
    camera_id: str = "cam_a",
    *,
    venue_id: str | None = "venue_01",
    surface_id: str | None = "main_floor",
    space: MotionSpace = METRIC,
    trust: GeometryTrust = GeometryTrust.VERIFIED,
    calibrated_region: VenueRegion | None = None,
) -> CameraSpatialContext:
    return CameraSpatialContext(
        camera_id=camera_id,
        space=space,
        venue_id=venue_id if space.is_physical else None,
        surface_id=surface_id if space.is_physical else None,
        trust=trust,
        calibrated_region=calibrated_region,
    )


def observation(
    camera_id: str = "cam_a",
    track_id: int = 1,
    *,
    x: float = 0.0,
    y: float = 0.0,
    venue_seconds: float = 10.0,
    half_width_seconds: float | None = HALF_WIDTH_SECONDS,
    stream_epoch: int = 0,
    frame_index: int = 1,
    space: MotionSpace = METRIC,
    spatial: CameraSpatialContext | None = None,
) -> TrackObservation:
    """One camera-local track at one venue instant, comparable by default."""
    resolved = spatial if spatial is not None else context(camera_id, space=space)
    timing = (
        TimeUncertainty.unknown()
        if half_width_seconds is None
        else TimeUncertainty.declared(half_width_seconds)
    )
    return TrackObservation(
        subject=SubjectRef(camera_id, track_id),
        stream=StreamRef(camera_id, stream_epoch),
        frame_index=frame_index,
        venue_seconds=venue_seconds,
        timing=timing,
        position=MotionPoint(x=x, y=y, space=resolved.space),
        spatial=resolved,
    )

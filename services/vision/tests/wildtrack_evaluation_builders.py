"""Synthetic builders for the offline WILDTRACK evaluator tests (V0.4.9b-D1).

Everything here is constructed in memory: no disk, no real WILDTRACK corpus, no
detector or tracker. Ground-truth positions are set directly (the ``positionID``
is a valid placeholder and never re-derived), so a test controls the exact
ground-plane metre geometry the evaluator's matcher sees.
"""

from __future__ import annotations

from collections.abc import Sequence

from association_builders import NORMALIZED
from association_builders import observation as _observation
from atreides_vision.association import (
    AssociationOutcome,
    CandidatePair,
    DecisionReason,
    GeometricEvidence,
    GeometryTrust,
    PairAssessment,
    SupportSummary,
)
from atreides_vision.association.models import (
    AssociationRound,
    PairDecision,
    TrackObservation,
    observation_order_key,
    tracklet_key,
)
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.fusion.models import IntegrationResult
from atreides_vision.identity.models import VenueSubjectId
from atreides_vision.occupancy.models import OccupancyContribution, OccupancySnapshot
from atreides_vision.venue_validation.datasets.wildtrack.annotations import (
    WildtrackBoundingBox,
    WildtrackFrameAnnotations,
    WildtrackGroundPosition,
    WildtrackPersonAnnotation,
    WildtrackPersonId,
)
from atreides_vision.venue_validation.datasets.wildtrack.cameras import CAMERA_IDS
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.ground_truth import (
    WildtrackGroundTruth,
)
from atreides_vision.venue_validation.datasets.wildtrack.time import WildtrackFrameId
from atreides_vision.venue_validation.observer import (
    VenueSubjectMembershipSnapshot,
    VenueValidationInstantSnapshot,
)

__all__ = [
    "BBOX",
    "ground_truth_of",
    "gt_frame",
    "gt_person",
    "instant_seconds",
    "membership",
    "metric_obs",
    "missed_decision",
    "non_metric_obs",
    "occupancy_of",
    "snapshot_for",
    "subject_id",
    "tracklet_key",
]

#: WILDTRACK's declared scored cadence is 10 Hz / 5 = 2.0 fps.
_DECLARED_FPS = 2.0

BBOX = WildtrackBoundingBox(0.0, 0.0, 1.0, 1.0)


def instant_seconds(instant_index: int) -> float:
    return instant_index / _DECLARED_FPS


def subject_id(sequence: int) -> VenueSubjectId:
    return VenueSubjectId(sequence=sequence)


def gt_person(
    person_id: int,
    *,
    x: float,
    y: float,
    cameras: Sequence[str],
) -> WildtrackPersonAnnotation:
    """A ground-truth person at ``(x, y)`` metres, visible in ``cameras``."""
    seen = set(cameras)
    return WildtrackPersonAnnotation(
        person_id=WildtrackPersonId(person_id),
        position=WildtrackGroundPosition(position_id=0, x_m=x, y_m=y),
        views={camera: (BBOX if camera in seen else None) for camera in CAMERA_IDS},
    )


def gt_frame(frame_value: int, *persons: WildtrackPersonAnnotation) -> WildtrackFrameAnnotations:
    ordered = tuple(sorted(persons, key=lambda person: person.person_id.value))
    return WildtrackFrameAnnotations(frame_id=WildtrackFrameId(frame_value), persons=ordered)


def ground_truth_of(*frames: WildtrackFrameAnnotations) -> WildtrackGroundTruth:
    return WildtrackGroundTruth(frames=tuple(frames))


def metric_obs(
    camera_id: str,
    track_id: int,
    *,
    x: float = 0.0,
    y: float = 0.0,
    venue_seconds: float = 0.0,
    stream_epoch: int = 0,
) -> TrackObservation:
    return _observation(
        camera_id,
        track_id,
        x=x,
        y=y,
        venue_seconds=venue_seconds,
        stream_epoch=stream_epoch,
    )


def non_metric_obs(
    camera_id: str,
    track_id: int,
    *,
    venue_seconds: float = 0.0,
    stream_epoch: int = 0,
) -> TrackObservation:
    return _observation(
        camera_id,
        track_id,
        venue_seconds=venue_seconds,
        stream_epoch=stream_epoch,
        space=NORMALIZED,
    )


def membership(
    sequence: int,
    *members: TrackObservation,
) -> VenueSubjectMembershipSnapshot:
    return VenueSubjectMembershipSnapshot(
        subject_id=subject_id(sequence),
        members=tuple(tracklet_key(member) for member in members),
    )


def occupancy_of(
    instant_index: int,
    *,
    raw_active_track_count: int = 0,
    contributions: Sequence[tuple[int, Sequence[TrackObservation]]] = (),
    unresolved: Sequence[TrackObservation] = (),
) -> OccupancySnapshot:
    contributing = tuple(
        OccupancyContribution(
            subject_id=subject_id(sequence),
            members=tuple(SubjectRef(o.camera_id, o.subject.track_id) for o in members),
        )
        for sequence, members in contributions
    )
    return OccupancySnapshot(
        venue_seconds=instant_seconds(instant_index),
        raw_active_track_count=raw_active_track_count,
        contributing_subjects=contributing,
        unresolved_tracklets=tuple(SubjectRef(o.camera_id, o.subject.track_id) for o in unresolved),
    )


def missed_decision(
    first: TrackObservation,
    second: TrackObservation,
    *,
    reason: DecisionReason,
    outcome: AssociationOutcome = AssociationOutcome.UNASSOCIATED,
) -> PairDecision:
    """A comparable pair the runtime did not associate, for the given reason.

    ``outcome`` defaults to ``UNASSOCIATED`` but a caller can ask for
    ``AMBIGUOUS`` (e.g. a ``support_not_sustained`` pair still being weighed);
    both are missed associations to the evaluator.
    """
    pair = CandidatePair(first, second)
    assessment = PairAssessment(
        pair=pair,
        evidence=GeometricEvidence(
            separation_m=1.0,
            position_tolerance_m=1.0,
            motion_allowance_m=1.0,
            nominal_gap_seconds=0.0,
            timing_slack_seconds=0.5,
            trust=GeometryTrust.VERIFIED,
        ),
    )
    return PairDecision(
        assessment=assessment,
        outcome=outcome,
        reasons=(reason,),
        support=SupportSummary(
            samples=1,
            total_samples=1,
            span_seconds=0.0,
            started_seconds=instant_seconds(0),
            last_seconds=instant_seconds(0),
            trust=GeometryTrust.VERIFIED,
            max_separation_m=1.0,
            min_rival_margin_m=None,
        ),
    )


def _decision_observations(
    decisions: Sequence[PairDecision],
) -> tuple[TrackObservation, ...]:
    gathered: list[TrackObservation] = []
    for decision in decisions:
        gathered.append(decision.pair.first)
        gathered.append(decision.pair.second)
    return tuple(gathered)


def snapshot_for(
    instant_index: int,
    *,
    observations: Sequence[TrackObservation] = (),
    decisions: Sequence[PairDecision] = (),
    memberships: Sequence[VenueSubjectMembershipSnapshot] = (),
    unowned: Sequence[TrackObservation] = (),
    occupancy: OccupancySnapshot | None = None,
    integration: IntegrationResult | None = None,
) -> VenueValidationInstantSnapshot:
    """A whole venue-instant snapshot aligned to ``instant_index``.

    ``observations`` must already contain every observation referenced by a
    decision pair; they are re-sorted into canonical order here.
    """
    seconds = instant_seconds(instant_index)
    by_key: dict[tuple[str, int, int, int], TrackObservation] = {}
    for source in (*observations, *_decision_observations(decisions)):
        by_key[observation_order_key(source)] = source
    ordered_obs = tuple(by_key[key] for key in sorted(by_key))
    ordered_decisions = tuple(sorted(decisions, key=lambda decision: decision.key))
    association_round = AssociationRound(
        venue_seconds=seconds,
        observations=ordered_obs,
        decisions=ordered_decisions,
    )
    occ = occupancy if occupancy is not None else occupancy_of(instant_index)
    integ = (
        integration
        if integration is not None
        else IntegrationResult(venue_seconds=seconds, outcomes=())
    )
    return VenueValidationInstantSnapshot(
        venue_seconds=seconds,
        observations=ordered_obs,
        association_round=association_round,
        integration=integ,
        memberships=tuple(memberships),
        unowned_tracklets=tuple(tracklet_key(o) for o in unowned),
        occupancy=occ,
    )

"""Builders for fusion integration tests: rounds with deliberately shaped decisions.

V0.4.2's own decision engine is exercised end to end in
``test_association_engine.py``; this module goes one level lower and builds
:class:`~atreides_vision.association.models.PairDecision` directly, so a
fusion test can put two specific tracklets into exactly the outcome it needs
to exercise — including outcomes (two independently pre-existing venue
subjects meeting through a fresh association) that would take an
unreasonably contrived geometry to coax out of the real engine.
"""

from __future__ import annotations

from association_builders import observation
from atreides_vision.association import (
    AssociationOutcome,
    AssociationRound,
    CandidatePair,
    DecisionConfidence,
    DecisionReason,
    GateReason,
    GateRejection,
    GeometricEvidence,
    GeometryTrust,
    PairAssessment,
    PairDecision,
    SupportSummary,
    TrackObservation,
    observation_order_key,
)

__all__ = [
    "ambiguous_decision",
    "associated_decision",
    "not_comparable_decision",
    "obs",
    "round_of",
    "unassociated_decision",
]


def obs(camera_id: str, track_id: int, *, venue_seconds: float = 10.0) -> TrackObservation:
    return observation(camera_id, track_id, venue_seconds=venue_seconds)


def _evidence(separation_m: float) -> GeometricEvidence:
    return GeometricEvidence(
        separation_m=separation_m,
        position_tolerance_m=1.0,
        motion_allowance_m=1.0,
        nominal_gap_seconds=0.0,
        timing_slack_seconds=0.5,
        trust=GeometryTrust.VERIFIED,
    )


def _support(**overrides: object) -> SupportSummary:
    fields: dict[str, object] = {
        "samples": 3,
        "total_samples": 3,
        "span_seconds": 0.25,
        "started_seconds": 10.0,
        "last_seconds": 10.25,
        "trust": GeometryTrust.VERIFIED,
        "max_separation_m": 0.5,
        "min_rival_margin_m": None,
    }
    fields.update(overrides)
    return SupportSummary(**fields)  # type: ignore[arg-type]


def associated_decision(first: TrackObservation, second: TrackObservation) -> PairDecision:
    """A genuine, fully-earned positive association between two observations."""
    pair = CandidatePair(first, second)
    assessment = PairAssessment(pair=pair, evidence=_evidence(0.5))
    return PairDecision(
        assessment=assessment,
        outcome=AssociationOutcome.ASSOCIATED,
        reasons=(DecisionReason.SUSTAINED_MUTUAL_AGREEMENT,),
        support=_support(),
        confidence=DecisionConfidence.CORROBORATED,
    )


def unassociated_decision(first: TrackObservation, second: TrackObservation) -> PairDecision:
    pair = CandidatePair(first, second)
    assessment = PairAssessment(pair=pair, evidence=_evidence(40.0))
    return PairDecision(
        assessment=assessment,
        outcome=AssociationOutcome.UNASSOCIATED,
        reasons=(DecisionReason.GEOMETRY_EXCLUDED,),
        support=_support(max_separation_m=40.0),
    )


def ambiguous_decision(first: TrackObservation, second: TrackObservation) -> PairDecision:
    pair = CandidatePair(first, second)
    assessment = PairAssessment(pair=pair, evidence=_evidence(0.5))
    return PairDecision(
        assessment=assessment,
        outcome=AssociationOutcome.AMBIGUOUS,
        reasons=(DecisionReason.RIVAL_NOT_SEPARATED,),
        support=_support(min_rival_margin_m=0.1),
    )


def not_comparable_decision(first: TrackObservation, second: TrackObservation) -> PairDecision:
    pair = CandidatePair(first, second)
    assessment = PairAssessment(
        pair=pair, rejections=(GateRejection(GateReason.SAME_CAMERA, "one camera"),)
    )
    return PairDecision(
        assessment=assessment,
        outcome=AssociationOutcome.NOT_COMPARABLE,
        reasons=(DecisionReason.HARD_GATE_REJECTED,),
    )


def round_of(venue_seconds: float, *decisions: PairDecision) -> AssociationRound:
    """An :class:`AssociationRound` built directly from prepared decisions.

    Observations and decisions are each re-sorted into their own canonical
    order, matching what :class:`AssociationRound` itself requires — a caller
    lists decisions in whatever order is convenient for the test and this
    builder makes them reproducible.
    """
    observations: dict[tuple[str, int, int, int], TrackObservation] = {}
    for decision in decisions:
        for observation_ in (decision.pair.first, decision.pair.second):
            observations[observation_order_key(observation_)] = observation_
    ordered_observations = tuple(observations[key] for key in sorted(observations))
    ordered_decisions = tuple(sorted(decisions, key=lambda decision: decision.key))
    return AssociationRound(
        venue_seconds=venue_seconds, observations=ordered_observations, decisions=ordered_decisions
    )

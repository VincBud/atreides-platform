"""Preference, mutuality and group consistency — the rules that refuse a merge.

Geometry can only say "not impossible". These functions are what turns that into
"and nobody else fits, and both sides say so, and the group this would form does
not disagree with itself". Most of the tests below are therefore refusals: a
one-sided preference, a rival too close to call, and transitive evidence that
would join tracklets nothing ever compared.
"""

from __future__ import annotations

from association_builders import DECIDING, GATING, observation
from atreides_vision.association import (
    CameraPreference,
    CandidatePair,
    DecisionReason,
    GateReason,
    GateRejection,
    GeometricEvidence,
    GeometryTrust,
    PairAssessment,
    TrackletKey,
    TrackObservation,
    camera_preferences,
    candidate_key,
    gate_observation,
    inconsistent_pairs,
    pair_agreement,
)

A1: TrackletKey = ("cam_a", 0, 1)
A2: TrackletKey = ("cam_a", 0, 2)
B1: TrackletKey = ("cam_b", 0, 1)
C1: TrackletKey = ("cam_c", 0, 1)


def preference(
    camera_id: str,
    partner: TrackletKey,
    *,
    separation_m: float = 0.5,
    rival_margin_m: float | None = None,
    ambiguous: bool = False,
) -> CameraPreference:
    return CameraPreference(
        camera_id=camera_id,
        partner=partner,
        separation_m=separation_m,
        rival_margin_m=rival_margin_m,
        ambiguous=ambiguous,
    )


def measured(
    first: TrackObservation, second: TrackObservation, separation_m: float
) -> PairAssessment:
    """An assessment that cleared every hard gate and measured a separation."""
    return PairAssessment(
        pair=CandidatePair(first, second),
        evidence=GeometricEvidence(
            separation_m=separation_m,
            position_tolerance_m=1.0,
            motion_allowance_m=1.0,
            nominal_gap_seconds=0.0,
            timing_slack_seconds=0.5,
            trust=GeometryTrust.VERIFIED,
        ),
    )


def never_compared(first: TrackObservation, second: TrackObservation) -> PairAssessment:
    """An assessment a hard gate refused, so no separation exists for it."""
    return PairAssessment(
        pair=CandidatePair(first, second),
        rejections=(GateRejection(GateReason.DIFFERENT_SURFACE, "no shared surface"),),
    )


class TestWhatAnObservationPrefers:
    def test_the_nearest_candidate_on_each_camera_is_preferred(self) -> None:
        subject = observation("cam_a", 1, x=0.0, y=0.0)
        pool = [
            subject,
            observation("cam_b", 1, x=0.5, y=0.0),
            observation("cam_b", 2, x=1.5, y=0.0),
            observation("cam_c", 1, x=0.25, y=0.0),
        ]

        preferences = camera_preferences(gate_observation(subject, pool, GATING))

        assert [one.camera_id for one in preferences] == ["cam_b", "cam_c"]
        assert preferences[0].partner == B1
        assert preferences[0].rival_margin_m == 1.0

    def test_candidates_on_different_cameras_are_not_rivals(self) -> None:
        """One person in three views produces three consistent preferences."""
        subject = observation("cam_a", 1, x=0.0, y=0.0)
        pool = [
            subject,
            observation("cam_b", 1, x=0.25, y=0.0),
            observation("cam_c", 1, x=0.25, y=0.0),
        ]

        preferences = camera_preferences(gate_observation(subject, pool, GATING))

        assert [one.rival_margin_m for one in preferences] == [None, None]

    def test_a_camera_the_gating_layer_called_ambiguous_is_flagged(self) -> None:
        subject = observation("cam_a", 1, x=0.0, y=0.0)
        pool = [
            subject,
            observation("cam_b", 1, x=0.5, y=0.0),
            observation("cam_b", 2, x=0.75, y=0.0),
        ]

        preferences = camera_preferences(gate_observation(subject, pool, GATING))

        assert preferences[0].ambiguous
        assert preferences[0].rival_margin_m == 0.25

    def test_one_camera_prefers_nobody(self) -> None:
        subject = observation("cam_a", 1, x=0.0, y=0.0)
        pool = [subject, observation("cam_a", 2, x=0.25, y=0.0)]

        assert camera_preferences(gate_observation(subject, pool, GATING)) == ()

    def test_a_geometrically_excluded_counterpart_is_never_preferred(self) -> None:
        subject = observation("cam_a", 1, x=0.0, y=0.0)
        pool = [subject, observation("cam_b", 1, x=9.0, y=0.0)]

        assert camera_preferences(gate_observation(subject, pool, GATING)) == ()


class TestMutualAgreement:
    PAIR = CandidatePair(observation("cam_a", 1), observation("cam_b", 1))

    def test_both_sides_preferring_each_other_agrees(self) -> None:
        agreement = pair_agreement(
            self.PAIR,
            {A1: (preference("cam_b", B1),), B1: (preference("cam_a", A1),)},
            DECIDING,
        )

        assert agreement.agrees
        assert agreement.reason is None
        assert agreement.rival_margin_m is None

    def test_a_one_sided_preference_is_not_an_agreement(self) -> None:
        agreement = pair_agreement(
            self.PAIR,
            {A1: (preference("cam_b", B1),), B1: (preference("cam_a", A2),)},
            DECIDING,
        )

        assert not agreement.agrees
        assert agreement.reason is DecisionReason.NOT_MUTUAL_BEST

    def test_a_side_that_prefers_nobody_on_that_camera_is_not_an_agreement(self) -> None:
        agreement = pair_agreement(self.PAIR, {A1: (preference("cam_b", B1),)}, DECIDING)

        assert agreement.reason is DecisionReason.NOT_MUTUAL_BEST

    def test_a_rival_too_close_to_call_refuses_the_agreement(self) -> None:
        agreement = pair_agreement(
            self.PAIR,
            {
                A1: (preference("cam_b", B1, rival_margin_m=2.0),),
                B1: (preference("cam_a", A1, rival_margin_m=0.25),),
            },
            DECIDING,
        )

        assert not agreement.agrees
        assert agreement.reason is DecisionReason.RIVAL_NOT_SEPARATED
        assert agreement.rival_margin_m == 0.25

    def test_a_rival_exactly_at_the_acceptance_margin_is_separated_enough(self) -> None:
        agreement = pair_agreement(
            self.PAIR,
            {
                A1: (preference("cam_b", B1, rival_margin_m=DECIDING.acceptance_margin_m),),
                B1: (preference("cam_a", A1),),
            },
            DECIDING,
        )

        assert agreement.agrees
        assert agreement.rival_margin_m == DECIDING.acceptance_margin_m

    def test_an_ambiguous_camera_refuses_the_agreement_whatever_the_margin(self) -> None:
        agreement = pair_agreement(
            self.PAIR,
            {
                A1: (preference("cam_b", B1, rival_margin_m=5.0, ambiguous=True),),
                B1: (preference("cam_a", A1, rival_margin_m=5.0),),
            },
            DECIDING,
        )

        assert agreement.reason is DecisionReason.RIVAL_NOT_SEPARATED

    def test_the_answer_does_not_depend_on_the_order_preferences_were_built(self) -> None:
        first = {A1: (preference("cam_b", B1),), B1: (preference("cam_a", A1),)}
        second = {B1: (preference("cam_a", A1),), A1: (preference("cam_b", B1),)}

        assert pair_agreement(self.PAIR, first, DECIDING) == pair_agreement(
            self.PAIR, second, DECIDING
        )


class TestGroupConsistency:
    """Pairwise agreement must not become a merge nobody measured."""

    a1 = observation("cam_a", 1)
    a2 = observation("cam_a", 2)
    b1 = observation("cam_b", 1)
    c1 = observation("cam_c", 1)

    def test_a_consistent_three_camera_group_stands(self) -> None:
        accepted = [
            candidate_key(CandidatePair(self.a1, self.b1)),
            candidate_key(CandidatePair(self.b1, self.c1)),
        ]
        assessments = {
            candidate_key(one.pair): one
            for one in (
                measured(self.a1, self.b1, 0.5),
                measured(self.b1, self.c1, 0.5),
                measured(self.a1, self.c1, 0.75),
            )
        }

        assert inconsistent_pairs(accepted, assessments) == ()

    def test_a_transitive_group_its_own_geometry_excludes_is_refused_whole(self) -> None:
        accepted = [
            candidate_key(CandidatePair(self.a1, self.b1)),
            candidate_key(CandidatePair(self.a1, self.c1)),
        ]
        assessments = {
            candidate_key(one.pair): one
            for one in (
                measured(self.a1, self.b1, 1.5),
                measured(self.a1, self.c1, 1.5),
                measured(self.b1, self.c1, 3.0),
            )
        }

        refused = inconsistent_pairs(accepted, assessments)

        assert [key for key, _ in refused] == sorted(accepted)
        assert {reason for _, reason in refused} == {DecisionReason.INCONSISTENT_EVIDENCE}

    def test_a_group_holding_two_tracklets_from_one_camera_is_refused(self) -> None:
        accepted = [
            candidate_key(CandidatePair(self.a1, self.b1)),
            candidate_key(CandidatePair(self.b1, self.c1)),
            candidate_key(CandidatePair(self.a2, self.c1)),
        ]
        assessments = {
            candidate_key(one.pair): one
            for one in (
                measured(self.a1, self.b1, 0.5),
                measured(self.b1, self.c1, 0.5),
                measured(self.a2, self.c1, 0.5),
                never_compared(self.a1, self.a2),
            )
        }

        refused = inconsistent_pairs(accepted, assessments)

        assert len(refused) == 3
        assert {reason for _, reason in refused} == {DecisionReason.EXCLUSIVITY_CONFLICT}

    def test_a_pair_that_was_never_comparable_is_not_a_contradiction(self) -> None:
        """Cameras with no shared coverage are a deployment, not a disagreement."""
        accepted = [
            candidate_key(CandidatePair(self.a1, self.b1)),
            candidate_key(CandidatePair(self.b1, self.c1)),
        ]
        assessments = {
            candidate_key(one.pair): one
            for one in (
                measured(self.a1, self.b1, 0.5),
                measured(self.b1, self.c1, 0.5),
                never_compared(self.a1, self.c1),
            )
        }

        assert inconsistent_pairs(accepted, assessments) == ()

    def test_one_contradictory_group_does_not_refuse_an_unrelated_one(self) -> None:
        d1 = observation("cam_d", 1)
        e1 = observation("cam_e", 1)
        contradictory = [
            candidate_key(CandidatePair(self.a1, self.b1)),
            candidate_key(CandidatePair(self.a1, self.c1)),
        ]
        innocent = candidate_key(CandidatePair(d1, e1))
        assessments = {
            candidate_key(one.pair): one
            for one in (
                measured(self.a1, self.b1, 1.5),
                measured(self.a1, self.c1, 1.5),
                measured(self.b1, self.c1, 3.0),
                measured(d1, e1, 0.5),
            )
        }

        refused = inconsistent_pairs([*contradictory, innocent], assessments)

        assert innocent not in [key for key, _ in refused]

    def test_the_refusals_come_back_in_canonical_order_whatever_went_in(self) -> None:
        accepted = [
            candidate_key(CandidatePair(self.a1, self.b1)),
            candidate_key(CandidatePair(self.a1, self.c1)),
        ]
        assessments = {
            candidate_key(one.pair): one
            for one in (
                measured(self.a1, self.b1, 1.5),
                measured(self.a1, self.c1, 1.5),
                measured(self.b1, self.c1, 3.0),
            )
        }

        assert inconsistent_pairs(accepted, assessments) == inconsistent_pairs(
            list(reversed(accepted)), assessments
        )

"""The deterministic association core, end to end, in both directions.

A true association found, a genuine non-match refused, an ambiguous pair left
unmerged, a transitive group that would have merged three cameras on evidence
nobody measured, evidence that expires, a single camera degrading cleanly, and
byte-identical output across two runs and across camera enumeration order.

Every position here is in the abstract motion space the projector produces. With
the suite's gating config a co-timed pair is admissible out to exactly 2.0 m,
and ticks are 0.125 s apart so three of them span exactly the 0.25 s the
decision config requires.
"""

from __future__ import annotations

import json
from itertools import permutations

import pytest

from association_builders import DECIDING, GATING, context, observation
from atreides_vision.association import (
    AssociationEngine,
    AssociationOutcome,
    AssociationRound,
    DecisionConfidence,
    DecisionReason,
    GeometryTrust,
    PairDecision,
    TrackObservation,
)
from atreides_vision.behavior.coordinates import MotionSpace
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.errors import AssociationError

STEP = 0.125
START = 10.0

#: One pair, one metre apart in two views: admissible, mutual, no rival.
ONE_PAIR = (("cam_a", 1, 0.0), ("cam_b", 1, 0.5))


def placed(
    seconds: float, placements: tuple[tuple[str, int, float], ...]
) -> list[TrackObservation]:
    return [
        observation(camera_id, track_id, x=x, y=0.0, venue_seconds=seconds)
        for camera_id, track_id, x in placements
    ]


def engine() -> AssociationEngine:
    return AssociationEngine(GATING, DECIDING)


def ticks(
    core: AssociationEngine,
    placements: tuple[tuple[str, int, float], ...],
    count: int = 3,
    start: float = START,
) -> list[AssociationRound]:
    """Observe the same placements at ``count`` successive venue instants."""
    rounds = []
    for index in range(count):
        seconds = start + index * STEP
        rounds.append(core.observe(seconds, placed(seconds, placements)))
    return rounds


def decision_for(round_: AssociationRound, first: SubjectRef, second: SubjectRef) -> PairDecision:
    for decision in round_.decisions:
        if decision.subjects == (first, second) or decision.subjects == (second, first):
            return decision
    raise AssertionError(f"no decision for {first} and {second}")


A1 = SubjectRef("cam_a", 1)
A2 = SubjectRef("cam_a", 2)
B1 = SubjectRef("cam_b", 1)
B2 = SubjectRef("cam_b", 2)
C1 = SubjectRef("cam_c", 1)


class TestATrueAssociationIsEarned:
    def test_agreement_must_hold_before_it_is_believed(self) -> None:
        rounds = ticks(engine(), ONE_PAIR)

        assert [round_.outcome_for(A1) for round_ in rounds] == [
            AssociationOutcome.AMBIGUOUS,
            AssociationOutcome.AMBIGUOUS,
            AssociationOutcome.ASSOCIATED,
        ]
        assert rounds[0].decisions[0].reasons == (DecisionReason.SUPPORT_NOT_SUSTAINED,)

    def test_an_association_states_the_rule_and_the_support_that_earned_it(self) -> None:
        final = ticks(engine(), ONE_PAIR)[-1]

        decision = decision_for(final, A1, B1)

        assert decision.reasons == (DecisionReason.SUSTAINED_MUTUAL_AGREEMENT,)
        assert decision.support is not None
        assert (decision.support.samples, decision.support.span_seconds) == (3, 0.25)
        assert decision.confidence is DecisionConfidence.CORROBORATED

    def test_an_association_resolves_back_to_its_camera_and_track_observations(self) -> None:
        final = ticks(engine(), ONE_PAIR)[-1]

        assert final.partners_of(A1) == (B1,)
        record = json.loads(json.dumps(final.to_record()))
        assessment = record["decisions"][0]["assessment"]
        for side, camera_id in (("first", "cam_a"), ("second", "cam_b")):
            contributor = assessment["pair"][side]
            assert contributor["camera_id"] == camera_id
            assert contributor["track_id"] == 1
            assert contributor["stream_epoch"] == 0
            assert contributor["frame_index"] == 1
        assert assessment["evidence"]["separation_m"] == 0.5
        assert record["decisions"][0]["support"]["total_samples"] == 3
        assert record["decisions"][0]["confidence"] == "corroborated"

    def test_unverified_geometry_associates_only_provisionally(self) -> None:
        """Weaker calibration lowers the confidence rather than being swallowed."""
        core = engine()
        for index in range(3):
            seconds = START + index * STEP
            core.observe(
                seconds,
                [
                    observation("cam_a", 1, x=0.0, venue_seconds=seconds),
                    observation(
                        "cam_b",
                        1,
                        x=0.5,
                        venue_seconds=seconds,
                        spatial=context("cam_b", trust=GeometryTrust.UNVERIFIED),
                    ),
                ],
            )
        final = core.observe(
            START + 3 * STEP,
            [
                observation("cam_a", 1, x=0.0, venue_seconds=START + 3 * STEP),
                observation(
                    "cam_b",
                    1,
                    x=0.5,
                    venue_seconds=START + 3 * STEP,
                    spatial=context("cam_b", trust=GeometryTrust.UNVERIFIED),
                ),
            ],
        )

        decision = decision_for(final, A1, B1)

        assert decision.is_associated
        assert decision.confidence is DecisionConfidence.PROVISIONAL


class TestRefusals:
    def test_a_genuine_non_match_stays_unassociated_however_long_it_is_watched(self) -> None:
        rounds = ticks(engine(), (("cam_a", 1, 0.0), ("cam_b", 1, 9.0)), count=6)

        assert {round_.outcome_for(A1) for round_ in rounds} == {AssociationOutcome.UNASSOCIATED}
        assert rounds[-1].decisions[0].reasons == (DecisionReason.GEOMETRY_EXCLUDED,)

    def test_two_candidates_too_close_to_tell_apart_are_left_unmerged(self) -> None:
        rounds = ticks(
            engine(), (("cam_a", 1, 0.0), ("cam_b", 1, 0.5), ("cam_b", 2, 0.75)), count=6
        )

        assert rounds[-1].outcome_for(A1) is AssociationOutcome.AMBIGUOUS
        assert decision_for(rounds[-1], A1, B1).reasons == (DecisionReason.RIVAL_NOT_SEPARATED,)

    def test_one_observation_cannot_be_associated_with_two_on_one_camera(self) -> None:
        final = ticks(engine(), (("cam_a", 1, 0.0), ("cam_b", 1, 0.5), ("cam_b", 2, 1.75)))[-1]

        assert final.partners_of(A1) == (B1,)
        assert decision_for(final, A1, B2).reasons == (DecisionReason.NOT_MUTUAL_BEST,)
        assert decision_for(final, B1, B2).outcome is AssociationOutcome.NOT_COMPARABLE

    def test_a_run_broken_by_a_contradiction_must_be_earned_again(self) -> None:
        core = engine()
        ticks(core, ONE_PAIR, count=2)
        broken = core.observe(
            START + 2 * STEP, placed(START + 2 * STEP, (("cam_a", 1, 0.0), ("cam_b", 1, 9.0)))
        )
        resumed = ticks(core, ONE_PAIR, count=2, start=START + 3 * STEP)

        assert broken.outcome_for(A1) is AssociationOutcome.UNASSOCIATED
        assert [round_.outcome_for(A1) for round_ in resumed] == [
            AssociationOutcome.AMBIGUOUS,
            AssociationOutcome.AMBIGUOUS,
        ]


class TestGroupsOfCameras:
    def test_three_cameras_that_agree_produce_three_pairwise_associations(self) -> None:
        final = ticks(engine(), (("cam_a", 1, 0.0), ("cam_b", 1, 0.5), ("cam_c", 1, 1.0)))[-1]

        assert len(final.associated) == 3
        assert final.partners_of(A1) == (B1, C1)

    def test_a_group_its_own_geometry_contradicts_is_not_merged_transitively(self) -> None:
        """A near B and A near C, but B and C excluded: nobody measured A=B=C."""
        final = ticks(engine(), (("cam_a", 1, 0.0), ("cam_b", 1, -1.5), ("cam_c", 1, 1.5)))[-1]

        assert final.associated == ()
        assert decision_for(final, A1, B1).reasons == (DecisionReason.INCONSISTENT_EVIDENCE,)
        assert decision_for(final, B1, C1).outcome is AssociationOutcome.UNASSOCIATED

    def test_a_chain_that_would_place_one_camera_twice_in_a_group_is_refused(self) -> None:
        final = ticks(
            engine(),
            (("cam_a", 1, 0.0), ("cam_a", 2, 1.8), ("cam_b", 1, 0.1), ("cam_c", 1, 1.7)),
        )[-1]

        assert final.associated == ()
        assert decision_for(final, A1, B1).reasons == (DecisionReason.EXCLUSIVITY_CONFLICT,)
        assert decision_for(final, B1, C1).reasons == (DecisionReason.EXCLUSIVITY_CONFLICT,)


class TestEvidenceExpires:
    def test_a_gap_nobody_observed_is_not_time_spent_agreeing(self) -> None:
        core = engine()
        ticks(core, ONE_PAIR, count=2)
        after_gap = ticks(core, ONE_PAIR, count=1, start=11.0)[-1]

        support = decision_for(after_gap, A1, B1).support
        assert support is not None
        assert (support.samples, support.total_samples) == (1, 3)
        assert after_gap.outcome_for(A1) is AssociationOutcome.AMBIGUOUS

    def test_a_pair_nobody_has_seen_since_the_ttl_stops_being_tracked(self) -> None:
        core = engine()
        ticks(core, ONE_PAIR, count=2)

        core.observe(12.0, placed(12.0, (("cam_a", 1, 0.0),)))

        assert core.tracked_pairs == 0

    def test_a_pair_that_stops_being_comparable_loses_its_run(self) -> None:
        """Uncalibrated metres end the run rather than freezing it for later."""
        core = engine()
        ticks(core, ONE_PAIR, count=2)
        uncalibrated = START + 2 * STEP
        refused = core.observe(
            uncalibrated,
            [
                observation("cam_a", 1, x=0.0, venue_seconds=uncalibrated),
                observation(
                    "cam_b",
                    1,
                    x=0.5,
                    venue_seconds=uncalibrated,
                    space=MotionSpace.NORMALIZED_IMAGE,
                ),
            ],
        )
        resumed = ticks(core, ONE_PAIR, count=1, start=START + 3 * STEP)[-1]

        assert refused.outcome_for(A1) is AssociationOutcome.NOT_COMPARABLE
        support = decision_for(resumed, A1, B1).support
        assert support is not None
        assert support.total_samples == 1


class TestOneCameraIsTheSamePath:
    def test_a_single_camera_produces_no_comparable_pair_and_no_error(self) -> None:
        final = ticks(engine(), (("cam_a", 1, 0.0), ("cam_a", 2, 5.0)))[-1]

        assert final.outcome_for(A1) is AssociationOutcome.NOT_COMPARABLE
        assert decision_for(final, A1, A2).reasons == (DecisionReason.HARD_GATE_REJECTED,)

    def test_a_single_camera_accumulates_no_evidence(self) -> None:
        core = engine()
        ticks(core, (("cam_a", 1, 0.0), ("cam_a", 2, 5.0)), count=4)

        assert core.tracked_pairs == 0

    def test_one_observation_alone_decides_nothing(self) -> None:
        final = ticks(engine(), (("cam_a", 1, 0.0),))[-1]

        assert final.decisions == ()
        assert final.outcome_for(A1) is AssociationOutcome.NOT_COMPARABLE


class TestDeterminism:
    PLACEMENTS = (
        ("cam_a", 1, 0.0),
        ("cam_b", 1, 0.5),
        ("cam_c", 1, 1.0),
        ("cam_b", 2, 6.0),
    )

    def test_two_runs_of_the_same_input_are_byte_identical(self) -> None:
        first = ticks(engine(), self.PLACEMENTS)[-1]
        second = ticks(engine(), self.PLACEMENTS)[-1]

        assert json.dumps(first.to_record()) == json.dumps(second.to_record())

    def test_the_camera_enumeration_order_changes_nothing(self) -> None:
        expected = json.dumps(ticks(engine(), self.PLACEMENTS)[-1].to_record())

        for order in permutations(self.PLACEMENTS):
            assert json.dumps(ticks(engine(), order)[-1].to_record()) == expected

    def test_decisions_and_observations_come_back_in_canonical_order(self) -> None:
        final = ticks(engine(), tuple(reversed(self.PLACEMENTS)))[-1]

        assert [decision.key for decision in final.decisions] == sorted(
            decision.key for decision in final.decisions
        )
        assert [one.subject.camera_id for one in final.observations] == [
            "cam_a",
            "cam_b",
            "cam_b",
            "cam_c",
        ]


class TestTheEngineRefusesImpossibleInput:
    def test_venue_instants_must_advance(self) -> None:
        core = engine()
        core.observe(10.0, placed(10.0, ONE_PAIR))

        with pytest.raises(AssociationError, match="must advance"):
            core.observe(9.5, placed(9.5, ONE_PAIR))

    def test_one_tracklet_cannot_be_in_two_places_at_one_instant(self) -> None:
        core = engine()

        with pytest.raises(AssociationError, match="appears twice"):
            core.observe(
                10.0,
                [
                    observation("cam_a", 1, x=0.0, venue_seconds=10.0),
                    observation("cam_a", 1, x=5.0, venue_seconds=10.0),
                ],
            )

    def test_a_restarted_stream_is_not_the_same_tracklet_twice(self) -> None:
        """Track 1 before and after an epoch change is not one tracklet replayed."""
        core = engine()

        round_ = core.observe(
            10.0,
            [
                observation("cam_a", 1, x=0.0, venue_seconds=10.0),
                observation("cam_a", 1, x=5.0, venue_seconds=10.0, stream_epoch=1),
            ],
        )

        assert len(round_.observations) == 2

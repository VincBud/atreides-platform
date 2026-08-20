"""V0.4.4: the runtime seam wiring association, fusion and venue identity together.

Each layer already has its own exhaustive tests — this module protects only the
contract the pipeline itself adds: that calling ``AssociationEngine.observe``
and ``fusion.integrate_round`` in sequence, per venue instant, produces the
right venue-local identity state, never a guessed one, independent of input
ordering, and idempotently on repetition.
"""

from __future__ import annotations

import json
from itertools import permutations

import pytest

from association_builders import DECIDING, GATING, observation
from atreides_vision.association import AssociationOutcome, TrackObservation
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.errors import AssociationError
from atreides_vision.fusion import IntegrationOutcome
from atreides_vision.identity import VenueSubjectId
from atreides_vision.venue_pipeline import VenueInstantResult, VenuePipeline
from identity_builders import CONFIG

STEP = 0.125
START = 10.0

#: One pair, one metre apart in two views: admissible, mutual, no rival.
ONE_PAIR = (("cam_a", 1, 0.0), ("cam_b", 1, 0.5))

A1 = SubjectRef("cam_a", 1)
A2 = SubjectRef("cam_a", 2)
B1 = SubjectRef("cam_b", 1)
B2 = SubjectRef("cam_b", 2)
C1 = SubjectRef("cam_c", 1)


def pipeline() -> VenuePipeline:
    return VenuePipeline(gating=GATING, decisions=DECIDING, identity=CONFIG)


def placed(
    seconds: float, placements: tuple[tuple[str, int, float], ...]
) -> list[TrackObservation]:
    return [
        observation(camera_id, track_id, x=x, y=0.0, venue_seconds=seconds)
        for camera_id, track_id, x in placements
    ]


def ticks(
    pipe: VenuePipeline,
    placements: tuple[tuple[str, int, float], ...],
    count: int = 3,
    start: float = START,
) -> list[VenueInstantResult]:
    """Process the same placements at ``count`` successive venue instants."""
    results = []
    for index in range(count):
        seconds = start + index * STEP
        results.append(pipe.process_instant(seconds, placed(seconds, placements)))
    return results


class TestAssociationBecomesVenueIdentity:
    def test_sustained_two_camera_agreement_creates_one_venue_subject(self) -> None:
        results = ticks(pipeline(), ONE_PAIR)

        assert [r.round.outcome_for(A1) for r in results] == [
            AssociationOutcome.AMBIGUOUS,
            AssociationOutcome.AMBIGUOUS,
            AssociationOutcome.ASSOCIATED,
        ]
        for result in results[:2]:
            assert result.active_subjects == ()
            assert result.integration.applied == ()

        final = results[-1]
        assert len(final.active_subjects) == 1
        assert final.integration.applied != ()

    def test_no_venue_subject_before_evidence_is_sustained(self) -> None:
        results = ticks(pipeline(), ONE_PAIR, count=2)

        for result in results:
            assert result.active_subjects == ()
            assert result.integration.applied == ()

    def test_three_cameras_that_agree_are_fused_into_one_venue_subject(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, (("cam_a", 1, 0.0), ("cam_b", 1, 0.5), ("cam_c", 1, 1.0)))[-1]

        assert len(final.active_subjects) == 1
        assert final.round.associated != ()
        assert final.integration.applied != ()

        subject_id = final.active_subjects[0]
        subject = pipe.registry.get(subject_id)
        assert subject.camera_ids == ("cam_a", "cam_b", "cam_c")


class TestNonPositiveOutcomesNeverMutateIdentity:
    def test_ambiguous_evidence_creates_no_venue_subject(self) -> None:
        results = ticks(
            pipeline(), (("cam_a", 1, 0.0), ("cam_b", 1, 0.5), ("cam_b", 2, 0.75)), count=6
        )

        assert results[-1].round.outcome_for(A1) is AssociationOutcome.AMBIGUOUS
        for result in results:
            assert result.active_subjects == ()
            assert all(
                outcome.outcome is IntegrationOutcome.IGNORED
                for outcome in result.integration.outcomes
            )

    def test_unassociated_evidence_creates_no_venue_subject(self) -> None:
        results = ticks(pipeline(), (("cam_a", 1, 0.0), ("cam_b", 1, 9.0)), count=6)

        assert {r.round.outcome_for(A1) for r in results} == {AssociationOutcome.UNASSOCIATED}
        for result in results:
            assert result.active_subjects == ()

    def test_not_comparable_evidence_creates_no_venue_subject(self) -> None:
        # a single camera: every candidate pair is refused by the same-camera gate
        results = ticks(pipeline(), (("cam_a", 1, 0.0), ("cam_a", 2, 5.0)), count=4)

        for result in results:
            assert result.round.outcome_for(A1) is AssociationOutcome.NOT_COMPARABLE
            assert result.active_subjects == ()
            assert all(
                outcome.outcome is IntegrationOutcome.IGNORED
                for outcome in result.integration.outcomes
            )


class TestDistinctVenueSubjectsAreNeverAutoMerged:
    def test_two_independently_created_subjects_are_never_merged_by_a_later_association(
        self,
    ) -> None:
        pipe = pipeline()
        first = ticks(pipe, ONE_PAIR)[-1]
        subject_one = first.active_subjects[0]

        second = ticks(
            pipe,
            (("cam_a", 2, 0.0), ("cam_c", 1, 0.5)),
            count=3,
            start=START + 3 * STEP,
        )[-1]
        assert len(second.active_subjects) == 2
        subject_two = next(s for s in second.active_subjects if s != subject_one)

        # force cam_b/1 (owned by subject_one) and cam_c/1 (owned by subject_two)
        # to sustain agreement with each other: fusion must refuse, never merge.
        result = None
        for index in range(6, 10):
            seconds = START + index * STEP
            result = pipe.process_instant(
                seconds, placed(seconds, (("cam_b", 1, 0.0), ("cam_c", 1, 0.5)))
            )
        assert result is not None
        refused = [
            o for o in result.integration.outcomes if o.outcome is IntegrationOutcome.REFUSED
        ]
        assert refused
        assert set(refused[0].subject_ids) == {subject_one, subject_two}
        # both original subjects remain, untouched and unmerged
        assert set(result.active_subjects) == {subject_one, subject_two}


class TestSameCameraExclusivityAtVenueLevel:
    def test_a_venue_subject_cannot_gain_a_second_active_tracklet_on_one_camera(self) -> None:
        pipe = pipeline()
        first = ticks(pipe, ONE_PAIR)[-1]
        subject_id = first.active_subjects[0]
        subject = pipe.registry.get(subject_id)
        assert subject.camera_ids == ("cam_a", "cam_b")

        # cam_b/1's stream ends; a new tracklet, cam_b/2, appears in its place
        # and independently earns association with cam_a/1. It must never be
        # folded into subject_id: that camera slot is already claimed.
        result = None
        for index in range(3, 7):
            seconds = START + index * STEP
            result = pipe.process_instant(
                seconds, placed(seconds, (("cam_a", 1, 0.0), ("cam_b", 2, 0.5)))
            )
        assert result is not None
        assert result.round.outcome_for(A1) is AssociationOutcome.ASSOCIATED

        refused = [
            o for o in result.integration.outcomes if o.outcome is IntegrationOutcome.REFUSED
        ]
        assert refused
        assert refused[0].subject_ids == (subject_id,)

        # the original membership is untouched
        unchanged = pipe.registry.get(subject_id)
        assert unchanged.camera_ids == ("cam_a", "cam_b")


class TestRepeatedProcessingIsIdempotent:
    def test_continuing_to_observe_an_associated_pair_produces_only_no_ops(self) -> None:
        pipe = pipeline()
        results = ticks(pipe, ONE_PAIR, count=6)
        subject_id = results[2].active_subjects[0]

        for result in results[3:]:
            assert result.active_subjects == (subject_id,)
            assert all(
                outcome.outcome in (IntegrationOutcome.NO_OP, IntegrationOutcome.APPLIED)
                for outcome in result.integration.outcomes
            )


class TestTimingUncertaintyReachesTheExistingGates:
    def test_unestablished_timing_does_not_shortcut_the_support_threshold(self) -> None:
        pipe = pipeline()
        results = []
        for index in range(3):
            seconds = START + index * STEP
            obs = [
                observation("cam_a", 1, x=0.0, venue_seconds=seconds, half_width_seconds=None),
                observation("cam_b", 1, x=0.5, venue_seconds=seconds, half_width_seconds=None),
            ]
            results.append(pipe.process_instant(seconds, obs))

        # unknown timing is a gate parameter, not a shortcut around evidence:
        # the pair still needs its own sustained support before it associates.
        assert results[0].active_subjects == ()
        assert results[1].active_subjects == ()


class TestSingleCameraIsAValidRuntimePath:
    def test_a_single_camera_session_produces_no_venue_subjects_and_no_error(self) -> None:
        results = ticks(pipeline(), (("cam_a", 1, 0.0), ("cam_a", 2, 5.0)), count=5)

        for result in results:
            assert result.active_subjects == ()
            assert result.round.associated == ()


class TestZeroFrameAndEndedStreams:
    def test_an_empty_instant_is_a_legal_no_op(self) -> None:
        pipe = pipeline()

        result = pipe.process_instant(START, [])

        assert result.observations == ()
        assert result.round.decisions == ()
        assert result.integration.outcomes == ()
        assert result.active_subjects == ()

    def test_an_ended_stream_stops_contributing_without_fabricating_observations(self) -> None:
        pipe = pipeline()
        results = ticks(pipe, ONE_PAIR)
        subject_id = results[-1].active_subjects[0]

        # cam_b's stream ends; only cam_a still produces frames.
        after = pipe.process_instant(
            START + 3 * STEP, placed(START + 3 * STEP, (("cam_a", 1, 0.0),))
        )

        assert after.active_subjects == (subject_id,)
        subject = pipe.registry.get(subject_id)
        assert subject.camera_ids == ("cam_a", "cam_b")


class TestLifecycleAndExpiryUseVenueTimeOnly:
    def test_expire_retires_an_idle_subject(self) -> None:
        pipe = pipeline()
        results = ticks(pipe, ONE_PAIR)
        subject_id = results[-1].active_subjects[0]

        # idle_expiry_seconds=10.0 in CONFIG: comfortably past retirement,
        # short of the additional retention_seconds needed to purge.
        purged = pipe.expire(results[-1].venue_seconds + CONFIG.idle_expiry_seconds + 1.0)

        assert subject_id not in pipe.registry.active_subjects()
        assert subject_id not in purged
        assert pipe.registry.get(subject_id).is_retired

    def test_expiry_does_not_run_implicitly_inside_process_instant(self) -> None:
        pipe = pipeline()
        results = ticks(pipe, ONE_PAIR)
        subject_id = results[-1].active_subjects[0]

        # far past idle_expiry_seconds, but expire() was never called
        assert subject_id in pipe.registry.active_subjects()


class TestDeterminismAcrossOrderingAndReplay:
    PLACEMENTS = (
        ("cam_a", 1, 0.0),
        ("cam_b", 1, 0.5),
        ("cam_c", 1, 1.0),
        ("cam_b", 2, 6.0),
    )

    def test_two_runs_of_the_same_input_are_byte_identical(self) -> None:
        first = ticks(pipeline(), self.PLACEMENTS)[-1]
        second = ticks(pipeline(), self.PLACEMENTS)[-1]

        assert json.dumps(first.round.to_record()) == json.dumps(second.round.to_record())
        assert first.active_subjects == second.active_subjects

    def test_observation_enumeration_order_within_an_instant_changes_nothing(self) -> None:
        expected = ticks(pipeline(), self.PLACEMENTS)[-1]

        for order in permutations(self.PLACEMENTS):
            got = ticks(pipeline(), order)[-1]
            assert json.dumps(got.round.to_record()) == json.dumps(expected.round.to_record())
            assert got.active_subjects == expected.active_subjects


class TestThePipelinePreservesEngineFailureSemantics:
    def test_venue_instants_must_still_advance(self) -> None:
        pipe = pipeline()
        pipe.process_instant(START, placed(START, ONE_PAIR))

        with pytest.raises(AssociationError, match="must advance"):
            pipe.process_instant(START - 0.5, placed(START - 0.5, ONE_PAIR))

    def test_one_tracklet_cannot_be_in_two_places_at_one_instant(self) -> None:
        pipe = pipeline()

        with pytest.raises(AssociationError, match="appears twice"):
            pipe.process_instant(
                START,
                [
                    observation("cam_a", 1, x=0.0, venue_seconds=START),
                    observation("cam_a", 1, x=5.0, venue_seconds=START),
                ],
            )


class TestPublicSurfaceStaysSmall:
    def test_the_registry_property_exposes_no_mutable_internal_dict(self) -> None:
        pipe = pipeline()

        # the registry is the real, encapsulated V0.4.3a object: it has its
        # own read-only surface, not a bare dict handed out for convenience.
        assert not isinstance(pipe.registry, dict)
        assert hasattr(pipe.registry, "active_subjects")

    def test_venue_subject_id_stays_a_distinct_type_from_a_track_id(self) -> None:
        results = ticks(pipeline(), ONE_PAIR)
        subject_id = results[-1].active_subjects[0]

        assert isinstance(subject_id, VenueSubjectId)
        assert not isinstance(subject_id, int)

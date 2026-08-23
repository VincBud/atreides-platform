"""V0.4.6: duplicate suppression against a real association/fusion/identity run.

Reuses the exact runtime fixtures and scenarios ``test_venue_pipeline.py``
already establishes for V0.4.4 — every occupancy scenario here is built by
actually running :class:`~atreides_vision.venue_pipeline.VenuePipeline`
through a realistic sequence of venue instants, not by constructing a final
:class:`~atreides_vision.occupancy.OccupancySnapshot` by hand.
"""

from __future__ import annotations

import json
from itertools import permutations

from association_builders import DECIDING, GATING, observation
from atreides_vision.association import AssociationOutcome, TrackObservation
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.fusion import IntegrationOutcome
from atreides_vision.occupancy import compute_occupancy
from atreides_vision.venue_pipeline import VenueInstantResult, VenuePipeline
from identity_builders import CONFIG

STEP = 0.125
START = 10.0

#: One pair, one metre apart in two views: admissible, mutual, no rival.
ONE_PAIR = (("cam_a", 1, 0.0), ("cam_b", 1, 0.5))


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
    results = []
    for index in range(count):
        seconds = start + index * STEP
        results.append(pipe.process_instant(seconds, placed(seconds, placements)))
    return results


class TestZeroAndSingleObservation:
    def test_an_empty_instant_has_zero_occupancy(self) -> None:
        pipe = pipeline()
        result = pipe.process_instant(START, [])

        snapshot = compute_occupancy(result, pipe.registry)

        assert snapshot.raw_active_track_count == 0
        assert snapshot.occupancy_count == 0
        assert snapshot.suppressed_duplicate_count == 0

    def test_one_lone_tracklet_counts_as_one(self) -> None:
        pipe = pipeline()
        result = pipe.process_instant(START, placed(START, (("cam_a", 1, 0.0),)))

        snapshot = compute_occupancy(result, pipe.registry)

        assert snapshot.raw_active_track_count == 1
        assert snapshot.occupancy_count == 1
        assert snapshot.suppressed_duplicate_count == 0
        assert snapshot.unresolved_tracklets == (SubjectRef("cam_a", 1),)


class TestUnrelatedTracksAreNeverSuppressed:
    def test_two_cameras_too_far_apart_to_associate_both_count_independently(self) -> None:
        pipe = pipeline()
        results = ticks(pipe, (("cam_a", 1, 0.0), ("cam_b", 1, 9.0)), count=6)
        final = results[-1]
        assert final.round.outcome_for(SubjectRef("cam_a", 1)) is AssociationOutcome.UNASSOCIATED

        snapshot = compute_occupancy(final, pipe.registry)

        assert snapshot.raw_active_track_count == 2
        assert snapshot.occupancy_count == 2
        assert snapshot.suppressed_duplicate_count == 0
        assert snapshot.contributing_subjects == ()


class TestSustainedFusionSuppressesExactlyOne:
    def test_two_confirmed_camera_views_of_one_person_count_as_one(self) -> None:
        pipe = pipeline()
        results = ticks(pipe, ONE_PAIR)
        final = results[-1]
        assert final.integration.applied != ()

        snapshot = compute_occupancy(final, pipe.registry)

        assert snapshot.raw_active_track_count == 2
        assert snapshot.occupancy_count == 1
        assert snapshot.suppressed_duplicate_count == 1
        assert len(snapshot.contributing_subjects) == 1
        contribution = snapshot.contributing_subjects[0]
        assert contribution.suppressed_count == 1
        assert set(contribution.members) == {SubjectRef("cam_a", 1), SubjectRef("cam_b", 1)}

    def test_no_suppression_before_association_evidence_is_sustained(self) -> None:
        pipe = pipeline()
        results = ticks(pipe, ONE_PAIR, count=2)

        for result in results:
            snapshot = compute_occupancy(result, pipe.registry)
            assert snapshot.raw_active_track_count == 2
            assert snapshot.occupancy_count == 2
            assert snapshot.suppressed_duplicate_count == 0

    def test_three_confirmed_camera_views_of_one_person_count_as_one(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, (("cam_a", 1, 0.0), ("cam_b", 1, 0.5), ("cam_c", 1, 1.0)))[-1]

        snapshot = compute_occupancy(final, pipe.registry)

        assert snapshot.raw_active_track_count == 3
        assert snapshot.occupancy_count == 1
        assert snapshot.suppressed_duplicate_count == 2
        assert len(snapshot.contributing_subjects) == 1
        assert snapshot.contributing_subjects[0].suppressed_count == 2

    def test_the_worked_example_shape_scales_to_several_independent_subjects(self) -> None:
        pipe = pipeline()
        first = ticks(pipe, ONE_PAIR)[-1]
        subject_one = first.active_subjects[0]

        second = ticks(
            pipe,
            (("cam_a", 2, 0.0), ("cam_c", 1, 0.5)),
            count=3,
            start=START + 3 * STEP,
        )[-1]
        subject_two = next(s for s in second.active_subjects if s != subject_one)

        # sustain both existing pairings undisturbed alongside a lone tracklet
        result = None
        for index in range(6, 9):
            seconds = START + index * STEP
            result = pipe.process_instant(
                seconds,
                placed(
                    seconds,
                    (
                        ("cam_a", 1, 0.0),
                        ("cam_b", 1, 0.5),
                        ("cam_a", 2, 0.0),
                        ("cam_c", 1, 0.5),
                        ("cam_d", 9, 20.0),
                    ),
                ),
            )
        assert result is not None
        assert set(result.active_subjects) == {subject_one, subject_two}

        snapshot = compute_occupancy(result, pipe.registry)

        assert snapshot.raw_active_track_count == 5
        assert snapshot.occupancy_count == 3  # subject_one + subject_two + the lone tracklet
        assert snapshot.suppressed_duplicate_count == 2
        assert snapshot.unresolved_tracklets == (SubjectRef("cam_d", 9),)


class TestAllFourAssociationOutcomesAndRefusedFusion:
    def test_ambiguous_evidence_never_suppresses(self) -> None:
        pipe = pipeline()
        results = ticks(pipe, (("cam_a", 1, 0.0), ("cam_b", 1, 0.5), ("cam_b", 2, 0.75)), count=6)
        final = results[-1]
        assert final.round.outcome_for(SubjectRef("cam_a", 1)) is AssociationOutcome.AMBIGUOUS

        snapshot = compute_occupancy(final, pipe.registry)

        assert snapshot.raw_active_track_count == 3
        assert snapshot.occupancy_count == 3
        assert snapshot.suppressed_duplicate_count == 0

    def test_unassociated_evidence_never_suppresses(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, (("cam_a", 1, 0.0), ("cam_b", 1, 9.0)), count=6)[-1]

        snapshot = compute_occupancy(final, pipe.registry)

        assert snapshot.suppressed_duplicate_count == 0
        assert snapshot.occupancy_count == snapshot.raw_active_track_count

    def test_not_comparable_evidence_never_suppresses(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, (("cam_a", 1, 0.0), ("cam_a", 2, 5.0)), count=4)[-1]
        assert final.round.outcome_for(SubjectRef("cam_a", 1)) is AssociationOutcome.NOT_COMPARABLE

        snapshot = compute_occupancy(final, pipe.registry)

        assert snapshot.raw_active_track_count == 2
        assert snapshot.occupancy_count == 2
        assert snapshot.suppressed_duplicate_count == 0

    def test_a_refused_fusion_outcome_never_causes_a_cross_subject_suppression(self) -> None:
        pipe = pipeline()
        first = ticks(pipe, ONE_PAIR)[-1]
        subject_one = first.active_subjects[0]

        second = ticks(
            pipe,
            (("cam_a", 2, 0.0), ("cam_c", 1, 0.5)),
            count=3,
            start=START + 3 * STEP,
        )[-1]
        subject_two = next(s for s in second.active_subjects if s != subject_one)

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

        snapshot = compute_occupancy(result, pipe.registry)

        # cam_b/1 (subject_one) and cam_c/1 (subject_two) each still count
        # once each, as two distinct, never-merged venue subjects.
        assert snapshot.raw_active_track_count == 2
        assert snapshot.occupancy_count == 2
        assert snapshot.suppressed_duplicate_count == 0
        assert {c.subject_id for c in snapshot.contributing_subjects} == {
            subject_one,
            subject_two,
        }


class TestActiveVersusHistoricalMembership:
    def test_a_camera_that_stops_observing_drops_out_of_current_occupancy(self) -> None:
        pipe = pipeline()
        results = ticks(pipe, ONE_PAIR)
        subject_id = results[-1].active_subjects[0]

        # cam_b's stream ends; only cam_a still produces frames. The
        # registry still remembers cam_b/1 as historical membership.
        after = pipe.process_instant(
            START + 3 * STEP, placed(START + 3 * STEP, (("cam_a", 1, 0.0),))
        )
        assert after.active_subjects == (subject_id,)
        subject = pipe.registry.get(subject_id)
        assert subject.camera_ids == ("cam_a", "cam_b")

        snapshot = compute_occupancy(after, pipe.registry)

        assert snapshot.raw_active_track_count == 1
        assert snapshot.occupancy_count == 1
        assert snapshot.suppressed_duplicate_count == 0
        # the subject still contributes, but with only its currently
        # observed member — never the historical cam_b tracklet.
        assert len(snapshot.contributing_subjects) == 1
        assert snapshot.contributing_subjects[0].members == (SubjectRef("cam_a", 1),)

    def test_registry_retention_is_not_current_presence(self) -> None:
        pipe = pipeline()
        results = ticks(pipe, ONE_PAIR)
        subject_id = results[-1].active_subjects[0]

        # nobody is observed this instant at all; the registry still holds
        # the subject as active (retained provenance), but occupancy must
        # not count anyone.
        empty = pipe.process_instant(START + 3 * STEP, [])
        assert subject_id in pipe.registry.active_subjects()

        snapshot = compute_occupancy(empty, pipe.registry)

        assert snapshot.raw_active_track_count == 0
        assert snapshot.occupancy_count == 0
        assert snapshot.contributing_subjects == ()


class TestLifecycleAndExpiry:
    def test_a_retired_subjects_tracklets_are_never_double_counted_after_retirement(
        self,
    ) -> None:
        pipe = pipeline()
        results = ticks(pipe, ONE_PAIR)
        subject_id = results[-1].active_subjects[0]

        pipe.expire(results[-1].venue_seconds + CONFIG.idle_expiry_seconds + 1.0)
        assert pipe.registry.get(subject_id).is_retired

        # cam_a/1 reappears afterward as a fresh, independently observed
        # tracklet; it must not resurrect the retired subject's grouping.
        seconds = results[-1].venue_seconds + CONFIG.idle_expiry_seconds + 2.0
        later = pipe.process_instant(seconds, placed(seconds, (("cam_a", 1, 0.0),)))

        snapshot = compute_occupancy(later, pipe.registry)

        assert snapshot.raw_active_track_count == 1
        assert snapshot.occupancy_count == 1
        assert snapshot.contributing_subjects == ()
        assert snapshot.unresolved_tracklets == (SubjectRef("cam_a", 1),)


class TestSingleCameraIsAValidRuntimePath:
    def test_a_single_camera_with_many_tracks_counts_each_independently(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, (("cam_a", 1, 0.0), ("cam_a", 2, 5.0), ("cam_a", 3, 10.0)), count=5)[-1]

        snapshot = compute_occupancy(final, pipe.registry)

        assert snapshot.raw_active_track_count == 3
        assert snapshot.occupancy_count == 3
        assert snapshot.suppressed_duplicate_count == 0
        assert snapshot.contributing_subjects == ()


class TestDeterminism:
    PLACEMENTS = (
        ("cam_a", 1, 0.0),
        ("cam_b", 1, 0.5),
        ("cam_c", 1, 1.0),
        ("cam_b", 2, 6.0),
    )

    def test_two_independent_runs_produce_identical_snapshots(self) -> None:
        pipe1 = pipeline()
        final1 = ticks(pipe1, self.PLACEMENTS)[-1]
        snapshot1 = compute_occupancy(final1, pipe1.registry)

        pipe2 = pipeline()
        final2 = ticks(pipe2, self.PLACEMENTS)[-1]
        snapshot2 = compute_occupancy(final2, pipe2.registry)

        assert json.dumps(snapshot1.to_record(), sort_keys=True) == json.dumps(
            snapshot2.to_record(), sort_keys=True
        )

    def test_observation_enumeration_order_does_not_change_the_snapshot(self) -> None:
        expected_pipe = pipeline()
        expected = compute_occupancy(
            ticks(expected_pipe, self.PLACEMENTS)[-1], expected_pipe.registry
        )

        for order in permutations(self.PLACEMENTS):
            pipe = pipeline()
            got = compute_occupancy(ticks(pipe, order)[-1], pipe.registry)
            assert json.dumps(got.to_record(), sort_keys=True) == json.dumps(
                expected.to_record(), sort_keys=True
            )

    def test_repeated_computation_over_the_same_result_is_idempotent(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, ONE_PAIR)[-1]

        first = compute_occupancy(final, pipe.registry)
        second = compute_occupancy(final, pipe.registry)

        assert first.to_record() == second.to_record()


class TestOccupancyNeverMutatesRuntimeState:
    def test_computing_a_snapshot_repeatedly_changes_nothing_about_the_registry(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, ONE_PAIR)[-1]

        before = pipe.registry.active_subjects()
        compute_occupancy(final, pipe.registry)
        compute_occupancy(final, pipe.registry)
        after = pipe.registry.active_subjects()

        assert before == after

    def test_computing_a_snapshot_does_not_alter_the_pipeline_result_object(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, ONE_PAIR)[-1]
        active_before = final.active_subjects
        applied_before = final.integration.applied

        compute_occupancy(final, pipe.registry)

        assert final.active_subjects == active_before
        assert final.integration.applied == applied_before

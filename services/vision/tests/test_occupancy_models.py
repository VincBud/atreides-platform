"""V0.4.6: the occupancy domain types enforce their own arithmetic invariant.

``OccupancyContribution`` and ``OccupancySnapshot`` are constructed directly
here to check the invariant itself, independent of how a snapshot is
produced at runtime — ``test_occupancy_analyzer.py`` covers that separately.
"""

from __future__ import annotations

import pytest

from atreides_vision.behavior.models import SubjectRef
from atreides_vision.errors import OccupancyError
from atreides_vision.identity import VenueSubjectId
from atreides_vision.occupancy import OccupancyContribution, OccupancySnapshot

A1 = SubjectRef("cam_a", 1)
B1 = SubjectRef("cam_b", 1)
C1 = SubjectRef("cam_c", 1)
D1 = SubjectRef("cam_d", 1)

SUBJECT_ONE = VenueSubjectId(1)
SUBJECT_TWO = VenueSubjectId(2)


class TestOccupancyContributionInvariants:
    def test_a_contribution_requires_at_least_one_member(self) -> None:
        with pytest.raises(OccupancyError, match="zero currently observed"):
            OccupancyContribution(subject_id=SUBJECT_ONE, members=())

    def test_a_contribution_rejects_the_same_tracklet_listed_twice(self) -> None:
        with pytest.raises(OccupancyError, match="more than once"):
            OccupancyContribution(subject_id=SUBJECT_ONE, members=(A1, A1))

    def test_a_single_member_contribution_suppresses_nothing(self) -> None:
        contribution = OccupancyContribution(subject_id=SUBJECT_ONE, members=(A1,))
        assert contribution.suppressed_count == 0

    def test_an_n_member_contribution_suppresses_n_minus_one(self) -> None:
        contribution = OccupancyContribution(subject_id=SUBJECT_ONE, members=(A1, B1, C1))
        assert contribution.suppressed_count == 2

    def test_to_record_names_every_field_truthfully(self) -> None:
        contribution = OccupancyContribution(subject_id=SUBJECT_ONE, members=(A1, B1))
        record = contribution.to_record()
        assert record["venue_subject_id"] == str(SUBJECT_ONE)
        assert record["members"] == [str(A1), str(B1)]
        assert record["suppressed_count"] == 1


class TestOccupancySnapshotArithmeticInvariant:
    def test_the_worked_example_from_the_milestone_spec(self) -> None:
        # raw = 50 across three cameras (18 + 15 + 17); six duplicate
        # tracklets suppressed by confirmed fusion -> occupancy = 44.
        contributing = tuple(
            OccupancyContribution(
                subject_id=VenueSubjectId(index + 1),
                members=(SubjectRef("cam_a", index), SubjectRef("cam_b", index)),
            )
            for index in range(1, 7)
        )
        unresolved = tuple(SubjectRef("cam_c", track_id) for track_id in range(100, 138))

        snapshot = OccupancySnapshot(
            venue_seconds=10.0,
            raw_active_track_count=50,
            contributing_subjects=contributing,
            unresolved_tracklets=unresolved,
        )

        assert snapshot.raw_active_track_count == 50
        assert snapshot.active_venue_subject_count == 6
        assert snapshot.suppressed_duplicate_count == 6
        assert snapshot.occupancy_count == 44
        assert (
            snapshot.raw_active_track_count
            == snapshot.occupancy_count + snapshot.suppressed_duplicate_count
        )

    def test_zero_observations_is_a_legal_zero_occupancy_snapshot(self) -> None:
        snapshot = OccupancySnapshot(
            venue_seconds=0.0,
            raw_active_track_count=0,
            contributing_subjects=(),
            unresolved_tracklets=(),
        )
        assert snapshot.occupancy_count == 0
        assert snapshot.suppressed_duplicate_count == 0

    def test_one_unfused_tracklet_counts_as_one(self) -> None:
        snapshot = OccupancySnapshot(
            venue_seconds=0.0,
            raw_active_track_count=1,
            contributing_subjects=(),
            unresolved_tracklets=(A1,),
        )
        assert snapshot.occupancy_count == 1
        assert snapshot.suppressed_duplicate_count == 0

    def test_negative_raw_count_is_rejected(self) -> None:
        with pytest.raises(OccupancyError, match="non-negative"):
            OccupancySnapshot(
                venue_seconds=0.0,
                raw_active_track_count=-1,
                contributing_subjects=(),
                unresolved_tracklets=(),
            )

    def test_contributing_subjects_out_of_canonical_order_is_rejected(self) -> None:
        contributing = (
            OccupancyContribution(subject_id=SUBJECT_TWO, members=(A1,)),
            OccupancyContribution(subject_id=SUBJECT_ONE, members=(B1,)),
        )
        with pytest.raises(OccupancyError, match="canonical VenueSubjectId order"):
            OccupancySnapshot(
                venue_seconds=0.0,
                raw_active_track_count=2,
                contributing_subjects=contributing,
                unresolved_tracklets=(),
            )

    def test_the_same_venue_subject_contributing_twice_is_rejected(self) -> None:
        contributing = (
            OccupancyContribution(subject_id=SUBJECT_ONE, members=(A1,)),
            OccupancyContribution(subject_id=SUBJECT_ONE, members=(B1,)),
        )
        with pytest.raises(OccupancyError, match="more than once"):
            OccupancySnapshot(
                venue_seconds=0.0,
                raw_active_track_count=2,
                contributing_subjects=contributing,
                unresolved_tracklets=(),
            )

    def test_a_tracklet_claimed_by_two_contributions_is_rejected(self) -> None:
        contributing = (
            OccupancyContribution(subject_id=SUBJECT_ONE, members=(A1,)),
            OccupancyContribution(subject_id=SUBJECT_TWO, members=(A1,)),
        )
        with pytest.raises(OccupancyError, match="claimed by more than one"):
            OccupancySnapshot(
                venue_seconds=0.0,
                raw_active_track_count=1,
                contributing_subjects=contributing,
                unresolved_tracklets=(),
            )

    def test_a_tracklet_both_a_member_and_unresolved_is_rejected(self) -> None:
        contributing = (OccupancyContribution(subject_id=SUBJECT_ONE, members=(A1,)),)
        with pytest.raises(OccupancyError, match="counted exactly once"):
            OccupancySnapshot(
                venue_seconds=0.0,
                raw_active_track_count=1,
                contributing_subjects=contributing,
                unresolved_tracklets=(A1,),
            )

    def test_a_raw_count_that_does_not_reconcile_is_rejected(self) -> None:
        with pytest.raises(OccupancyError, match="but 1 distinct"):
            OccupancySnapshot(
                venue_seconds=0.0,
                raw_active_track_count=2,
                contributing_subjects=(),
                unresolved_tracklets=(A1,),
            )

    def test_to_record_exposes_every_derived_count_by_its_true_name(self) -> None:
        contributing = (OccupancyContribution(subject_id=SUBJECT_ONE, members=(A1, B1)),)
        snapshot = OccupancySnapshot(
            venue_seconds=12.5,
            raw_active_track_count=3,
            contributing_subjects=contributing,
            unresolved_tracklets=(D1,),
        )
        record = snapshot.to_record()
        assert record["venue_seconds"] == 12.5
        assert record["raw_active_track_count"] == 3
        assert record["occupancy_count"] == 2
        assert record["suppressed_duplicate_count"] == 1
        assert record["active_venue_subject_count"] == 1
        assert record["contributing_subjects"] == [
            {
                "venue_subject_id": str(SUBJECT_ONE),
                "members": [str(A1), str(B1)],
                "suppressed_count": 1,
            }
        ]
        assert record["unresolved_tracklets"] == [str(D1)]

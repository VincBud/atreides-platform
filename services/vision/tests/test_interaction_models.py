"""Pair identity, coordinate compatibility and the interaction domain types.

The identity rules here are the milestone's safety boundary: a pair that should
not exist must be impossible to construct, not merely unlikely to be built.
"""

from __future__ import annotations

import math

import pytest

from atreides_vision.behavior.coordinates import MotionSpace
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.errors import InteractionError
from atreides_vision.interaction.models import (
    InteractionFrame,
    InteractionSignal,
    InteractionSignalType,
    InteractionSnapshot,
    PairGeometry,
    SpatialContext,
    SubjectPairRef,
    interaction_unit_for,
)

NORMALIZED = MotionSpace.NORMALIZED_IMAGE
METRIC = MotionSpace.GROUND_PLANE_METRES


def metric_context(
    camera_id: str = "cam_01", venue_id: str = "venue_01", surface_id: str = "main_floor"
) -> SpatialContext:
    return SpatialContext(camera_id, METRIC, venue_id, surface_id)


class TestSubjectPairRef:
    def test_canonical_ordering(self) -> None:
        """(7, 12) and (12, 7) are one pair, not two."""
        low, high = SubjectRef("cam_01", 7), SubjectRef("cam_01", 12)
        assert SubjectPairRef(low, high) == SubjectPairRef(high, low)
        assert SubjectPairRef(high, low).track_ids == (7, 12)

    def test_ordering_survives_hashing(self) -> None:
        """Pair history is keyed by this, so both orders must hit one entry."""
        low, high = SubjectRef("cam_01", 3), SubjectRef("cam_01", 9)
        store = {SubjectPairRef(low, high): "state"}
        assert store[SubjectPairRef(high, low)] == "state"
        assert len({SubjectPairRef(low, high), SubjectPairRef(high, low)}) == 1

    def test_a_subject_cannot_pair_with_itself(self) -> None:
        subject = SubjectRef("cam_01", 7)
        with pytest.raises(InteractionError, match="cannot interact with itself"):
            SubjectPairRef(subject, subject)

    def test_cross_camera_pairs_are_rejected(self) -> None:
        """The core safety rule: one person in two cameras must not self-pair."""
        with pytest.raises(InteractionError, match="Cross-camera interaction pairs"):
            SubjectPairRef(SubjectRef("cam_a", 1), SubjectRef("cam_b", 9))

    def test_the_rejection_explains_why(self) -> None:
        with pytest.raises(InteractionError) as caught:
            SubjectPairRef(SubjectRef("cam_a", 1), SubjectRef("cam_b", 1))
        assert "paired with themselves" in str(caught.value)

    def test_same_track_id_on_two_cameras_is_still_rejected(self) -> None:
        with pytest.raises(InteractionError, match="Cross-camera"):
            SubjectPairRef(SubjectRef("cam_a", 7), SubjectRef("cam_b", 7))

    def test_exposes_the_shared_camera(self) -> None:
        pair = SubjectPairRef(SubjectRef("cam_01", 1), SubjectRef("cam_01", 2))
        assert pair.camera_id == "cam_01"

    def test_membership_and_partner_lookup(self) -> None:
        first, second = SubjectRef("cam_01", 4), SubjectRef("cam_01", 8)
        pair = SubjectPairRef(first, second)
        assert pair.contains(first) and pair.contains(second)
        assert not pair.contains(SubjectRef("cam_01", 99))
        assert pair.other_than(first) == second
        assert pair.other_than(second) == first

    def test_partner_lookup_rejects_a_stranger(self) -> None:
        pair = SubjectPairRef(SubjectRef("cam_01", 1), SubjectRef("cam_01", 2))
        with pytest.raises(InteractionError, match="is not part of pair"):
            pair.other_than(SubjectRef("cam_01", 3))

    def test_track_id_churn_creates_a_different_pair(self) -> None:
        """A re-acquired track is a new subject, so its pairs are new pairs."""
        anchor = SubjectRef("cam_01", 1)
        before = SubjectPairRef(anchor, SubjectRef("cam_01", 7))
        after = SubjectPairRef(anchor, SubjectRef("cam_01", 31))
        assert before != after
        assert len({before, after}) == 2


class TestSpatialContext:
    def test_normalized_needs_no_venue(self) -> None:
        context = SpatialContext("cam_01", NORMALIZED)
        assert not context.is_metric

    def test_metric_requires_venue_and_surface(self) -> None:
        """Assuming a shared plane is the fabrication this prevents."""
        with pytest.raises(InteractionError, match="needs both a venue_id and a surface_id"):
            SpatialContext("cam_01", METRIC)
        with pytest.raises(InteractionError, match="venue_id and a surface_id"):
            SpatialContext("cam_01", METRIC, venue_id="venue_01")
        with pytest.raises(InteractionError, match="venue_id and a surface_id"):
            SpatialContext("cam_01", METRIC, surface_id="main_floor")

    def test_metric_context_is_physical(self) -> None:
        assert metric_context().is_metric

    @pytest.mark.parametrize("camera_id", ["", "   "])
    def test_rejects_a_blank_camera(self, camera_id: str) -> None:
        with pytest.raises(InteractionError, match="non-empty string"):
            SpatialContext(camera_id, NORMALIZED)

    def test_identical_contexts_are_compatible(self) -> None:
        assert metric_context().is_compatible_with(metric_context())
        assert SpatialContext("cam_01", NORMALIZED).is_compatible_with(
            SpatialContext("cam_01", NORMALIZED)
        )

    def test_different_spaces_are_incompatible(self) -> None:
        reason = SpatialContext("cam_01", NORMALIZED).incompatibility_with(metric_context())
        assert reason is not None and "different coordinate spaces" in reason

    def test_normalized_across_cameras_is_incompatible(self) -> None:
        """Frame fractions on two cameras have no relationship at all."""
        reason = SpatialContext("cam_a", NORMALIZED).incompatibility_with(
            SpatialContext("cam_b", NORMALIZED)
        )
        assert reason is not None and "different cameras" in reason

    def test_metric_across_cameras_is_incompatible_in_this_milestone(self) -> None:
        """Same venue frame, but pairing them risks pairing a person with themselves."""
        reason = metric_context("cam_a").incompatibility_with(metric_context("cam_b"))
        assert reason is not None and "different cameras" in reason

    def test_different_venues_are_incompatible(self) -> None:
        reason = metric_context(venue_id="club").incompatibility_with(
            metric_context(venue_id="festival")
        )
        assert reason is not None and "different venues" in reason

    def test_different_surfaces_are_incompatible(self) -> None:
        """A balcony and the floor can share x/y and be metres apart vertically."""
        reason = metric_context(surface_id="main_floor").incompatibility_with(
            metric_context(surface_id="balcony")
        )
        assert reason is not None and "different surfaces" in reason

    def test_compatible_contexts_report_no_reason(self) -> None:
        assert metric_context().incompatibility_with(metric_context()) is None


class TestSignalTypes:
    def test_only_neutral_measurements_exist(self) -> None:
        """V0.3.4 measures; it does not judge."""
        assert {member.value for member in InteractionSignalType} == {
            "closing_speed",
            "relative_motion_heading",
            "proximity_duration",
        }

    def test_no_semantic_type_exists(self) -> None:
        names = " ".join(member.value for member in InteractionSignalType)
        for forbidden in (
            "aggress",
            "fight",
            "threat",
            "conflict",
            "escalat",
            "suspicious",
            "harass",
            "stalk",
            "risk",
            "danger",
        ):
            assert forbidden not in names

    @pytest.mark.parametrize("signal_type", list(InteractionSignalType))
    def test_every_type_has_a_unit_in_every_space(self, signal_type: InteractionSignalType) -> None:
        for space in MotionSpace:
            assert interaction_unit_for(signal_type, space)

    def test_distance_units_follow_the_space(self) -> None:
        assert interaction_unit_for(InteractionSignalType.CLOSING_SPEED, METRIC) == "m/s"
        assert (
            interaction_unit_for(InteractionSignalType.CLOSING_SPEED, NORMALIZED)
            == "normalized_frame_unit/s"
        )

    def test_angles_and_durations_are_space_independent(self) -> None:
        for space in MotionSpace:
            assert (
                interaction_unit_for(InteractionSignalType.RELATIVE_MOTION_HEADING, space) == "rad"
            )
            assert interaction_unit_for(InteractionSignalType.PROXIMITY_DURATION, space) == "s"


class TestPairGeometry:
    def test_carries_distance_and_unit(self) -> None:
        geometry = PairGeometry(distance=0.84, space=METRIC)
        assert geometry.distance_unit == "m"
        assert geometry.to_record()["distance"]["unit"] == "m"

    def test_zero_distance_is_valid(self) -> None:
        """Two tracks at the same projected point are still two subjects."""
        assert PairGeometry(distance=0.0, space=METRIC).distance == 0.0

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_rejects_impossible_distances(self, bad: float) -> None:
        with pytest.raises(InteractionError, match="finite and non-negative"):
            PairGeometry(distance=bad, space=METRIC)


class TestInteractionSignal:
    def test_derives_its_unit(self) -> None:
        signal = InteractionSignal(InteractionSignalType.CLOSING_SPEED, 1.12, METRIC)
        assert signal.unit == "m/s"
        assert signal.to_record() == {
            "type": "closing_speed",
            "value": pytest.approx(1.12),
            "unit": "m/s",
        }

    def test_rejects_non_finite_values(self) -> None:
        with pytest.raises(InteractionError, match="must be finite"):
            InteractionSignal(InteractionSignalType.CLOSING_SPEED, float("inf"), METRIC)

    def test_negative_closing_speed_is_valid(self) -> None:
        """Separation is a legitimate measurement, not an error."""
        signal = InteractionSignal(InteractionSignalType.CLOSING_SPEED, -2.0, METRIC)
        assert signal.value == -2.0


class TestInteractionSnapshotAndFrame:
    def pair(self, first: int = 1, second: int = 2) -> SubjectPairRef:
        return SubjectPairRef(SubjectRef("cam_01", first), SubjectRef("cam_01", second))

    def snapshot(self, first: int = 1, second: int = 2) -> InteractionSnapshot:
        return InteractionSnapshot(
            pair=self.pair(first, second),
            frame_index=4,
            timestamp_seconds=0.16,
            geometry=PairGeometry(1.5, METRIC),
            signals=(
                InteractionSignal(InteractionSignalType.CLOSING_SPEED, 0.5, METRIC),
                InteractionSignal(InteractionSignalType.RELATIVE_MOTION_HEADING, math.pi, METRIC),
            ),
        )

    def test_snapshot_queries(self) -> None:
        snapshot = self.snapshot()
        assert snapshot.space is METRIC
        assert snapshot.distance == pytest.approx(1.5)
        assert len(snapshot) == 2
        assert snapshot.value_for(InteractionSignalType.CLOSING_SPEED) == pytest.approx(0.5)
        assert snapshot.value_for(InteractionSignalType.PROXIMITY_DURATION) is None

    def test_frame_queries(self) -> None:
        frame = InteractionFrame.from_snapshots(
            4, 0.16, [self.snapshot(1, 2), self.snapshot(1, 3)], candidate_pairs=3
        )
        assert len(frame) == 2
        assert frame.candidate_pairs == 3
        assert frame.for_pair(self.pair(1, 3)) is not None
        assert frame.for_pair(self.pair(2, 3)) is None

    def test_frame_lookup_is_order_independent(self) -> None:
        frame = InteractionFrame.from_snapshots(4, 0.16, [self.snapshot(1, 2)])
        reversed_pair = SubjectPairRef(SubjectRef("cam_01", 2), SubjectRef("cam_01", 1))
        assert frame.for_pair(reversed_pair) is not None

    def test_empty_frame(self) -> None:
        frame = InteractionFrame(frame_index=1, timestamp_seconds=0.0)
        assert len(frame) == 0
        assert list(frame) == []

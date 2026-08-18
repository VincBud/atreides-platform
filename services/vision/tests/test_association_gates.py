"""The hard comparability gates: every rule that can refuse a cross-camera pair.

A gate answers a question about the measurement, never about the people, and its
refusal is ``NOT_COMPARABLE`` — nothing was measured, so nothing was found. Each
test below drives exactly one gate and leaves every other precondition satisfied,
so a failure names the rule that broke.
"""

from __future__ import annotations

import pytest

from association_builders import GATING, NORMALIZED, context, observation
from atreides_vision.association import (
    HARD_GATE_ORDER,
    CandidatePair,
    GateReason,
    GeometryTrust,
    TrackObservation,
    VenueRegion,
    comparability_rejections,
)

NEAR_REGION = VenueRegion.covering([(0.0, 0.0), (10.0, 10.0)])
FAR_REGION = VenueRegion.covering([(200.0, 0.0), (210.0, 10.0)])


def reasons(first: TrackObservation, second: TrackObservation) -> set[GateReason]:
    """The set of gates that refuse this pair, under the test config."""
    pair = CandidatePair(first, second)
    return {rejection.reason for rejection in comparability_rejections(pair, GATING)}


class TestComparablePair:
    def test_two_calibrated_co_timed_cameras_clear_every_gate(self) -> None:
        pair = CandidatePair(observation("cam_a", 1), observation("cam_b", 4, x=0.5))

        assert comparability_rejections(pair, GATING) == ()

    def test_clearing_the_gates_says_nothing_about_the_people(self) -> None:
        """Two strangers 20 m apart still clear every *comparability* gate."""
        pair = CandidatePair(observation("cam_a", 1), observation("cam_b", 4, x=20.0))

        assert comparability_rejections(pair, GATING) == ()


class TestDifferentCameraRequirement:
    def test_two_tracks_on_one_camera_are_not_a_cross_camera_candidate(self) -> None:
        assert reasons(observation("cam_a", 1), observation("cam_a", 2)) == {GateReason.SAME_CAMERA}

    def test_the_same_camera_across_a_stream_epoch_is_still_the_same_camera(self) -> None:
        """Fragment stitching is a different capability, not a degenerate merge."""
        assert GateReason.SAME_CAMERA in reasons(
            observation("cam_a", 1, stream_epoch=0),
            observation("cam_a", 2, stream_epoch=1),
        )


class TestCalibratedMetresRequirement:
    def test_normalized_image_coordinates_cannot_cross_cameras(self) -> None:
        assert reasons(
            observation("cam_a", 1, space=NORMALIZED),
            observation("cam_b", 1, space=NORMALIZED),
        ) == {GateReason.NOT_GROUND_PLANE_METRES}

    def test_one_uncalibrated_camera_is_enough_to_refuse(self) -> None:
        assert reasons(observation("cam_a", 1, space=NORMALIZED), observation("cam_b", 1)) == {
            GateReason.NOT_GROUND_PLANE_METRES
        }

    def test_an_uncalibrated_pair_is_not_also_accused_of_venue_mismatch(self) -> None:
        """A gate that cannot be evaluated is skipped, never guessed."""
        rejected = reasons(
            observation("cam_a", 1, space=NORMALIZED),
            observation("cam_b", 1, space=NORMALIZED),
        )

        assert GateReason.DIFFERENT_VENUE_FRAME not in rejected
        assert GateReason.DIFFERENT_SURFACE not in rejected


class TestSameVenueAndSurfaceRequirement:
    def test_different_venue_frames_are_unrelated_coordinate_systems(self) -> None:
        assert reasons(
            observation("cam_a", 1),
            observation("cam_b", 1, spatial=context("cam_b", venue_id="venue_02")),
        ) == {GateReason.DIFFERENT_VENUE_FRAME}

    def test_different_surfaces_are_different_planes(self) -> None:
        """A balcony and the floor below share coordinates and not a storey."""
        assert reasons(
            observation("cam_a", 1),
            observation("cam_b", 1, spatial=context("cam_b", surface_id="balcony")),
        ) == {GateReason.DIFFERENT_SURFACE}

    def test_both_mismatches_are_reported_together(self) -> None:
        """Fixing the venue frame is wasted work if the surfaces also differ."""
        elsewhere = context("cam_b", venue_id="venue_02", surface_id="balcony")

        assert reasons(observation("cam_a", 1), observation("cam_b", 1, spatial=elsewhere)) == {
            GateReason.DIFFERENT_VENUE_FRAME,
            GateReason.DIFFERENT_SURFACE,
        }


class TestCalibrationTrustRequirement:
    def test_a_camera_that_moved_gates_everything_it_touches(self) -> None:
        moved = context("cam_b", trust=GeometryTrust.UNTRUSTED)

        assert reasons(observation("cam_a", 1), observation("cam_b", 1, spatial=moved)) == {
            GateReason.CAMERA_GEOMETRY_UNTRUSTED
        }

    def test_an_unverified_camera_still_compares(self) -> None:
        """Nobody checked is not the same fact as the camera moved."""
        unchecked = context("cam_b", trust=GeometryTrust.UNVERIFIED)

        assert reasons(observation("cam_a", 1), observation("cam_b", 1, spatial=unchecked)) == set()


class TestVenueTimeRequirement:
    def test_an_unaligned_stream_has_no_common_clock(self) -> None:
        assert reasons(
            observation("cam_a", 1, half_width_seconds=None), observation("cam_b", 1)
        ) == {GateReason.VENUE_TIME_NOT_ESTABLISHED}

    def test_both_unaligned_streams_are_named_once(self) -> None:
        pair = CandidatePair(
            observation("cam_a", 1, half_width_seconds=None),
            observation("cam_b", 1, half_width_seconds=None),
        )

        rejections = comparability_rejections(pair, GATING)

        assert [rejection.reason for rejection in rejections] == [
            GateReason.VENUE_TIME_NOT_ESTABLISHED
        ]
        assert "cam_a/1" in rejections[0].detail
        assert "cam_b/1" in rejections[0].detail

    def test_an_unaligned_pair_is_never_also_judged_on_time_or_region(self) -> None:
        """With no slack there is no honest window and no honest radius."""
        assert reasons(
            observation("cam_a", 1, venue_seconds=10.0, half_width_seconds=None),
            observation("cam_b", 1, venue_seconds=900.0),
        ) == {GateReason.VENUE_TIME_NOT_ESTABLISHED}


class TestSimultaneityRequirement:
    def test_observations_far_apart_in_venue_time_are_refused(self) -> None:
        """4.0 s apart, 0.5 s of slack, 0.5 s window: 3.5 s remains."""
        assert reasons(
            observation("cam_a", 1, venue_seconds=10.0),
            observation("cam_b", 1, venue_seconds=14.0),
        ) == {GateReason.NOT_SIMULTANEOUS}

    def test_the_window_is_measured_after_uncertainty_is_given_away(self) -> None:
        """1.0 s apart minus 0.5 s of slack is exactly the 0.5 s window."""
        assert (
            reasons(
                observation("cam_a", 1, venue_seconds=10.0),
                observation("cam_b", 1, venue_seconds=11.0),
            )
            == set()
        )

    def test_a_hair_beyond_the_window_is_refused(self) -> None:
        assert reasons(
            observation("cam_a", 1, venue_seconds=10.0),
            observation("cam_b", 1, venue_seconds=11.002),
        ) == {GateReason.NOT_SIMULTANEOUS}

    def test_wider_uncertainty_admits_a_pair_a_tighter_one_refuses(self) -> None:
        """Uncertainty weakens a gate; it must never manufacture a rejection."""
        tight = reasons(
            observation("cam_a", 1, venue_seconds=10.0, half_width_seconds=0.25),
            observation("cam_b", 1, venue_seconds=13.0, half_width_seconds=0.25),
        )
        loose = reasons(
            observation("cam_a", 1, venue_seconds=10.0, half_width_seconds=2.0),
            observation("cam_b", 1, venue_seconds=13.0, half_width_seconds=2.0),
        )

        assert tight == {GateReason.NOT_SIMULTANEOUS}
        assert loose == set()


class TestCalibratedRegionRequirement:
    def test_regions_beyond_reach_of_each_other_cannot_share_a_person(self) -> None:
        assert reasons(
            observation("cam_a", 1, spatial=context("cam_a", calibrated_region=NEAR_REGION)),
            observation("cam_b", 1, spatial=context("cam_b", calibrated_region=FAR_REGION)),
        ) == {GateReason.NO_SHARED_CALIBRATED_REGION}

    def test_adjacent_regions_within_reach_are_not_refused(self) -> None:
        """The gate excludes the unreachable, not merely the non-overlapping."""
        adjacent = VenueRegion.covering([(11.0, 0.0), (20.0, 10.0)])

        assert (
            reasons(
                observation("cam_a", 1, spatial=context("cam_a", calibrated_region=NEAR_REGION)),
                observation("cam_b", 1, spatial=context("cam_b", calibrated_region=adjacent)),
            )
            == set()
        )

    def test_an_undeclared_region_disables_the_gate_rather_than_assuming_one(self) -> None:
        assert (
            reasons(
                observation("cam_a", 1, spatial=context("cam_a", calibrated_region=NEAR_REGION)),
                observation("cam_b", 1),
            )
            == set()
        )

    def test_overlapping_regions_are_never_refused(self) -> None:
        assert (
            reasons(
                observation("cam_a", 1, spatial=context("cam_a", calibrated_region=NEAR_REGION)),
                observation("cam_b", 1, spatial=context("cam_b", calibrated_region=NEAR_REGION)),
            )
            == set()
        )


class TestRejectionRecording:
    def test_rejections_are_recorded_in_the_documented_gate_order(self) -> None:
        broken = context(
            "cam_a", venue_id="venue_02", surface_id="balcony", trust=GeometryTrust.UNTRUSTED
        )
        pair = CandidatePair(
            observation("cam_a", 1, venue_seconds=10.0),
            observation("cam_a", 2, venue_seconds=40.0, spatial=broken),
        )

        produced = [rejection.reason for rejection in comparability_rejections(pair, GATING)]

        assert produced == sorted(produced, key=HARD_GATE_ORDER.index)
        assert set(produced) == {
            GateReason.SAME_CAMERA,
            GateReason.DIFFERENT_VENUE_FRAME,
            GateReason.DIFFERENT_SURFACE,
            GateReason.CAMERA_GEOMETRY_UNTRUSTED,
            GateReason.NOT_SIMULTANEOUS,
        }

    def test_every_gate_reason_appears_in_the_documented_order(self) -> None:
        assert set(HARD_GATE_ORDER) == set(GateReason)
        assert len(HARD_GATE_ORDER) == len(GateReason)

    def test_every_rejection_carries_a_sentence_for_an_operator(self) -> None:
        pair = CandidatePair(observation("cam_a", 1), observation("cam_a", 2))

        for rejection in comparability_rejections(pair, GATING):
            assert rejection.detail.strip()
            assert rejection.to_record()["reason"] == rejection.reason.value

    def test_gate_order_is_independent_of_which_camera_was_enumerated_first(self) -> None:
        first = observation("cam_a", 1, venue_seconds=10.0)
        second = observation(
            "cam_b", 2, venue_seconds=40.0, spatial=context("cam_b", surface_id="balcony")
        )

        forward = comparability_rejections(CandidatePair(first, second), GATING)
        backward = comparability_rejections(CandidatePair(second, first), GATING)

        assert forward == backward


@pytest.mark.parametrize("reason", list(GateReason))
def test_no_gate_reason_names_a_person_or_a_conclusion(reason: GateReason) -> None:
    """Gate names describe the measurement's preconditions, never the people."""
    assert not any(
        word in reason.value
        for word in ("person", "identity", "match", "merge", "same_subject", "duplicate")
    )

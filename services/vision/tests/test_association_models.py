"""The association domain types: what can be said, and what cannot be expressed.

These are the milestone's safety boundary. A verdict that over-claims must be
impossible to construct, not merely unlikely to be written — so most of these
tests assert that something *raises*.
"""

from __future__ import annotations

import dataclasses

import pytest

from association_builders import METRIC, NORMALIZED, context, observation
from atreides_vision.association import (
    AssociationOutcome,
    CameraSpatialContext,
    CandidatePair,
    GateReason,
    GateRejection,
    GeometricEvidence,
    GeometryTrust,
    ObservationGating,
    PairAssessment,
    TrackObservation,
    VenueRegion,
    observation_order_key,
)
from atreides_vision.behavior.coordinates import MotionPoint
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.errors import AssociationError
from atreides_vision.timing import StreamRef, TimeUncertainty


def evidence(
    separation_m: float = 1.0,
    *,
    position_tolerance_m: float = 1.0,
    motion_allowance_m: float = 1.0,
    nominal_gap_seconds: float = 0.0,
    timing_slack_seconds: float = 0.5,
    trust: GeometryTrust = GeometryTrust.VERIFIED,
) -> GeometricEvidence:
    return GeometricEvidence(
        separation_m=separation_m,
        position_tolerance_m=position_tolerance_m,
        motion_allowance_m=motion_allowance_m,
        nominal_gap_seconds=nominal_gap_seconds,
        timing_slack_seconds=timing_slack_seconds,
        trust=trust,
    )


class TestAssociationOutcome:
    def test_exactly_four_outcomes_exist(self) -> None:
        """The vocabulary is the contract: a consumer must handle all four."""
        assert {outcome.value for outcome in AssociationOutcome} == {
            "associated",
            "ambiguous",
            "unassociated",
            "not_comparable",
        }

    def test_a_closed_gate_and_an_excluded_measurement_are_different_answers(self) -> None:
        """'We were not allowed to look' must never read as 'we looked and saw nobody'."""
        pair = CandidatePair(observation("cam_a", 1), observation("cam_b", 2))
        gated = PairAssessment(
            pair=pair, rejections=(GateRejection(GateReason.SAME_CAMERA, "one camera"),)
        )
        excluded = PairAssessment(pair=pair, evidence=evidence(separation_m=40.0))

        assert gated.geometric_outcome != excluded.geometric_outcome


class TestGeometryTrust:
    @pytest.mark.parametrize(
        ("trust", "permitted"),
        [
            (GeometryTrust.VERIFIED, True),
            (GeometryTrust.UNVERIFIED, True),
            (GeometryTrust.UNTRUSTED, False),
        ],
    )
    def test_only_detected_movement_forbids_comparison(
        self, trust: GeometryTrust, permitted: bool
    ) -> None:
        """An unchecked camera is not a moved camera; the two must not collapse."""
        assert trust.permits_comparison is permitted

    def test_worst_of_takes_the_weaker_side_in_either_order(self) -> None:
        verified, unverified = GeometryTrust.VERIFIED, GeometryTrust.UNVERIFIED
        assert verified.worst_of(unverified) is unverified
        assert unverified.worst_of(verified) is unverified
        assert unverified.worst_of(GeometryTrust.UNTRUSTED) is GeometryTrust.UNTRUSTED
        assert verified.worst_of(verified) is verified


class TestVenueRegion:
    def test_covering_is_the_bounding_box_of_its_points(self) -> None:
        region = VenueRegion.covering([(1.0, 2.0), (-3.0, 5.0), (0.0, 0.0)])

        assert (region.min_x, region.min_y, region.max_x, region.max_y) == (-3.0, 0.0, 1.0, 5.0)

    def test_negative_coordinates_are_ordinary(self) -> None:
        """The venue origin is wherever the operator put it."""
        region = VenueRegion.covering([(-9.0, -4.0), (-2.0, -1.0)])

        assert region.min_x == -9.0

    def test_gap_between_separated_regions_is_the_axis_gap(self) -> None:
        left = VenueRegion.covering([(0.0, 0.0), (10.0, 10.0)])
        right = VenueRegion.covering([(13.0, 0.0), (20.0, 10.0)])

        assert left.gap_to(right) == pytest.approx(3.0)
        assert right.gap_to(left) == pytest.approx(3.0)

    def test_gap_between_diagonally_separated_regions_is_euclidean(self) -> None:
        lower = VenueRegion.covering([(0.0, 0.0), (1.0, 1.0)])
        upper = VenueRegion.covering([(4.0, 5.0), (6.0, 7.0)])

        assert lower.gap_to(upper) == pytest.approx(5.0)

    def test_overlapping_regions_have_no_gap(self) -> None:
        first = VenueRegion.covering([(0.0, 0.0), (10.0, 10.0)])
        second = VenueRegion.covering([(5.0, 5.0), (15.0, 15.0)])

        assert first.gap_to(second) == 0.0

    def test_an_empty_region_is_refused(self) -> None:
        """'Covers nothing' and 'we do not know what it covers' are different facts."""
        with pytest.raises(AssociationError, match="at least one point"):
            VenueRegion.covering([])

    def test_non_finite_coordinates_are_refused(self) -> None:
        with pytest.raises(AssociationError, match="finite metres"):
            VenueRegion(min_x=0.0, min_y=0.0, max_x=float("inf"), max_y=1.0)

    def test_inverted_bounds_are_refused(self) -> None:
        with pytest.raises(AssociationError, match="min <= max"):
            VenueRegion(min_x=5.0, min_y=0.0, max_x=1.0, max_y=1.0)


class TestCameraSpatialContext:
    def test_metric_context_requires_a_venue_frame_and_a_surface(self) -> None:
        with pytest.raises(AssociationError, match="venue_id and a surface_id"):
            CameraSpatialContext(camera_id="cam_a", space=METRIC, venue_id="venue_01")

    def test_normalized_context_needs_neither(self) -> None:
        assert not CameraSpatialContext(camera_id="cam_a", space=NORMALIZED).is_metric

    def test_a_region_without_metres_is_refused(self) -> None:
        """A region is venue metres; an uncalibrated space has none."""
        with pytest.raises(AssociationError, match="declares a calibrated region"):
            CameraSpatialContext(
                camera_id="cam_a",
                space=NORMALIZED,
                calibrated_region=VenueRegion.covering([(0.0, 0.0), (1.0, 1.0)]),
            )

    def test_trust_defaults_to_unverified(self) -> None:
        """Nobody confirmed this viewpoint is the honest default, not VERIFIED."""
        assert CameraSpatialContext("cam_a", NORMALIZED).trust is GeometryTrust.UNVERIFIED

    def test_empty_camera_id_is_refused(self) -> None:
        with pytest.raises(AssociationError, match="non-empty string"):
            CameraSpatialContext(camera_id="  ", space=NORMALIZED)


class TestTrackObservation:
    def test_carries_no_pixels_and_no_appearance(self) -> None:
        """Nothing above the projector may describe how somebody looks."""
        names = {field.name for field in dataclasses.fields(TrackObservation)}

        assert names == {
            "subject",
            "stream",
            "frame_index",
            "venue_seconds",
            "timing",
            "position",
            "spatial",
        }

    def test_subject_stream_and_context_must_agree_on_the_camera(self) -> None:
        with pytest.raises(AssociationError, match="must describe one camera"):
            TrackObservation(
                subject=SubjectRef("cam_a", 1),
                stream=StreamRef("cam_b", 0),
                frame_index=1,
                venue_seconds=1.0,
                timing=TimeUncertainty.declared(0.25),
                position=MotionPoint(0.0, 0.0, METRIC),
                spatial=context("cam_a"),
            )

    def test_position_space_must_match_the_declared_context(self) -> None:
        with pytest.raises(AssociationError, match="positioned in"):
            TrackObservation(
                subject=SubjectRef("cam_a", 1),
                stream=StreamRef("cam_a", 0),
                frame_index=1,
                venue_seconds=1.0,
                timing=TimeUncertainty.declared(0.25),
                position=MotionPoint(0.0, 0.0, NORMALIZED),
                spatial=context("cam_a", space=METRIC),
            )

    def test_venue_time_cannot_precede_the_session(self) -> None:
        with pytest.raises(AssociationError, match="finite and non-negative"):
            observation(venue_seconds=-0.5)

    def test_record_preserves_camera_track_and_stream_epoch(self) -> None:
        record = observation("cam_b", 7, stream_epoch=2, frame_index=41).to_record()

        assert record["camera_id"] == "cam_b"
        assert record["track_id"] == 7
        assert record["stream_epoch"] == 2
        assert record["frame_index"] == 41

    def test_record_states_the_timing_basis_and_never_invents_a_half_width(self) -> None:
        record = observation(half_width_seconds=None).to_record()

        assert record["timing_basis"] == "not_established"
        assert record["timing_half_width_seconds"] is None


class TestCandidatePair:
    def test_the_same_two_observations_are_one_candidate_in_either_order(self) -> None:
        first, second = observation("cam_a", 1), observation("cam_b", 9)

        assert CandidatePair(first, second) == CandidatePair(second, first)
        assert len({CandidatePair(first, second), CandidatePair(second, first)}) == 1

    def test_canonical_order_follows_camera_then_epoch_then_track(self) -> None:
        low = observation("cam_a", 12)
        high = observation("cam_b", 3)

        pair = CandidatePair(high, low)

        assert pair.first is low
        assert pair.second is high
        assert pair.cameras == ("cam_a", "cam_b")

    def test_ordering_is_total_across_epochs_and_frames(self) -> None:
        assert observation_order_key(observation("cam_a", 3, stream_epoch=1, frame_index=2)) == (
            "cam_a",
            1,
            3,
            2,
        )

    def test_an_observation_cannot_be_a_candidate_for_itself(self) -> None:
        subject = observation("cam_a", 1)

        with pytest.raises(AssociationError, match="cannot be a candidate for itself"):
            CandidatePair(subject, subject)

    def test_same_camera_pairs_are_constructible_so_they_can_be_reported(self) -> None:
        """A rejection has to be representable to be recordable."""
        pair = CandidatePair(observation("cam_a", 1), observation("cam_a", 2))

        assert pair.cameras == ("cam_a", "cam_a")

    def test_other_than_returns_the_counterpart(self) -> None:
        first, second = observation("cam_a", 1), observation("cam_b", 9)
        pair = CandidatePair(first, second)

        assert pair.other_than(first.subject) is second
        assert pair.other_than(second.subject) is first

    def test_other_than_refuses_a_stranger(self) -> None:
        pair = CandidatePair(observation("cam_a", 1), observation("cam_b", 9))

        with pytest.raises(AssociationError, match="is not part of candidate"):
            pair.other_than(SubjectRef("cam_c", 4))


class TestGeometricEvidence:
    def test_radius_is_tolerance_plus_motion_allowance(self) -> None:
        assert evidence(position_tolerance_m=1.5, motion_allowance_m=4.0).admissible_radius_m == 5.5

    def test_a_pair_exactly_on_the_radius_is_admissible(self) -> None:
        """The boundary belongs to the permissive side: a gate only excludes."""
        on_radius = evidence(separation_m=2.0, position_tolerance_m=1.0, motion_allowance_m=1.0)

        assert on_radius.is_admissible
        assert on_radius.margin_m == 0.0

    def test_a_pair_just_beyond_the_radius_is_excluded(self) -> None:
        beyond = evidence(separation_m=2.001, position_tolerance_m=1.0, motion_allowance_m=1.0)

        assert not beyond.is_admissible
        assert beyond.margin_m < 0

    def test_negative_measurements_are_refused(self) -> None:
        with pytest.raises(AssociationError, match="finite and non-negative"):
            evidence(separation_m=-1.0)

    def test_record_shows_the_radius_that_admitted_the_pair(self) -> None:
        """Without the radius, a pair admitted on 30 s of clock drift looks tight."""
        record = evidence(separation_m=1.0, motion_allowance_m=60.0).to_record()

        assert record["separation_m"] == 1.0
        assert record["admissible_radius_m"] == 61.0
        assert record["timing_slack_seconds"] == 0.5


class TestPairAssessment:
    def test_a_rejected_pair_carries_no_measurement(self) -> None:
        pair = CandidatePair(observation("cam_a", 1), observation("cam_b", 2))

        with pytest.raises(AssociationError, match="Measurements across a closed gate"):
            PairAssessment(
                pair=pair,
                rejections=(GateRejection(GateReason.SAME_CAMERA, "detail"),),
                evidence=evidence(),
            )

    def test_a_comparable_pair_must_have_been_measured(self) -> None:
        pair = CandidatePair(observation("cam_a", 1), observation("cam_b", 2))

        with pytest.raises(AssociationError, match="unreported blind spot"):
            PairAssessment(pair=pair)

    def test_a_rejected_pair_is_not_comparable(self) -> None:
        pair = CandidatePair(observation("cam_a", 1), observation("cam_a", 2))
        assessment = PairAssessment(
            pair=pair, rejections=(GateRejection(GateReason.SAME_CAMERA, "one camera"),)
        )

        assert not assessment.is_comparable
        assert assessment.geometric_outcome is AssociationOutcome.NOT_COMPARABLE
        assert assessment.reasons() == (GateReason.SAME_CAMERA,)

    def test_an_excluded_pair_is_unassociated_not_incomparable(self) -> None:
        pair = CandidatePair(observation("cam_a", 1), observation("cam_b", 2))
        assessment = PairAssessment(pair=pair, evidence=evidence(separation_m=40.0))

        assert assessment.is_comparable
        assert assessment.geometric_outcome is AssociationOutcome.UNASSOCIATED

    def test_a_surviving_pair_has_no_outcome_at_all(self) -> None:
        """Surviving a gate is not evidence; geometry defers rather than merges."""
        pair = CandidatePair(observation("cam_a", 1), observation("cam_b", 2))
        assessment = PairAssessment(pair=pair, evidence=evidence(separation_m=0.5))

        assert assessment.geometric_outcome is None


class TestGateRejection:
    def test_a_rejection_must_explain_itself(self) -> None:
        with pytest.raises(AssociationError, match="must explain itself"):
            GateRejection(GateReason.SAME_CAMERA, "   ")


class TestObservationGating:
    def test_geometry_cannot_assert_an_association(self) -> None:
        """The milestone's central rule, enforced by construction."""
        with pytest.raises(AssociationError, match="cannot establish ASSOCIATED"):
            ObservationGating(
                observation=observation(),
                geometric_outcome=AssociationOutcome.ASSOCIATED,
            )

    def test_unresolved_ambiguity_cannot_be_reported_as_resolved(self) -> None:
        with pytest.raises(AssociationError, match="must not be reported as a resolved outcome"):
            ObservationGating(
                observation=observation(),
                ambiguous_cameras=("cam_b",),
                geometric_outcome=None,
            )

    def test_ambiguity_must_name_the_camera_it_could_not_resolve(self) -> None:
        with pytest.raises(AssociationError, match="without naming a camera"):
            ObservationGating(
                observation=observation(),
                geometric_outcome=AssociationOutcome.AMBIGUOUS,
            )

    def test_unassociated_cannot_be_claimed_while_candidates_survive(self) -> None:
        pair = CandidatePair(observation("cam_a", 1), observation("cam_b", 2))
        surviving = PairAssessment(pair=pair, evidence=evidence(separation_m=0.5))

        with pytest.raises(AssociationError, match="reports UNASSOCIATED"):
            ObservationGating(
                observation=observation(),
                assessments=(surviving,),
                candidates=(surviving,),
                geometric_outcome=AssociationOutcome.UNASSOCIATED,
            )

    def test_an_empty_gating_is_a_valid_not_comparable_result(self) -> None:
        gating = ObservationGating(
            observation=observation(),
            geometric_outcome=AssociationOutcome.NOT_COMPARABLE,
        )

        assert gating.subject == SubjectRef("cam_a", 1)
        assert gating.comparable == ()

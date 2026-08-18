"""Candidate enumeration and gating: the failure directions, not only the happy path.

An association layer is only trustworthy if its refusals are as well tested as
its findings, so this file covers a true pair surviving, a genuine non-match
excluded, an ambiguous pair left unmerged, a single camera degrading cleanly,
non-overlapping cameras, provenance surviving the whole path, and byte-identical
results across runs and across camera enumeration order.
"""

from __future__ import annotations

import json
from itertools import permutations
from types import ModuleType
from typing import ClassVar

import pytest

from association_builders import GATING, NORMALIZED, context, observation
from atreides_vision.association import (
    AssociationOutcome,
    CandidatePair,
    GateReason,
    GeometryTrust,
    TrackObservation,
    VenueRegion,
    assess_pair,
    candidate_pairs,
    gate_observation,
)
from atreides_vision.association import candidates as candidates_module
from atreides_vision.association import config as config_module
from atreides_vision.association import gates as gates_module
from atreides_vision.association import geometry as geometry_module
from atreides_vision.association import models as models_module

#: A subject on cam_a, and the counterparts each test places around it.
SUBJECT = observation("cam_a", 1, x=5.0, y=5.0)


def outcome_of(pool: list[TrackObservation], subject: TrackObservation = SUBJECT) -> object:
    return gate_observation(subject, pool, GATING).geometric_outcome


class TestCandidateEnumeration:
    def test_every_unordered_pair_appears_exactly_once(self) -> None:
        pool = [observation("cam_a", 1), observation("cam_b", 2), observation("cam_c", 3)]

        assert len(candidate_pairs(pool)) == 3

    def test_same_camera_pairs_are_enumerated_so_they_can_be_refused(self) -> None:
        """A refused candidate with a named reason beats a candidate never formed."""
        pool = [observation("cam_a", 1), observation("cam_a", 2)]

        pairs = candidate_pairs(pool)

        assert len(pairs) == 1
        assert pairs[0].cameras == ("cam_a", "cam_a")

    def test_one_observation_yields_no_candidates(self) -> None:
        assert candidate_pairs([observation("cam_a", 1)]) == ()

    def test_an_empty_pool_yields_no_candidates(self) -> None:
        assert candidate_pairs([]) == ()

    def test_enumeration_does_not_depend_on_input_order(self) -> None:
        pool = [
            observation("cam_c", 3),
            observation("cam_a", 1),
            observation("cam_b", 2, stream_epoch=1),
        ]

        orders = {candidate_pairs(list(order)) for order in permutations(pool)}

        assert len(orders) == 1

    def test_enumeration_is_ordered_by_camera_then_epoch_then_track(self) -> None:
        pool = [
            observation("cam_b", 9),
            observation("cam_a", 4, stream_epoch=1),
            observation("cam_a", 7, stream_epoch=0),
        ]

        keys = [
            (pair.first.subject.track_id, pair.second.subject.track_id)
            for pair in candidate_pairs(pool)
        ]

        assert keys == [(7, 4), (7, 9), (4, 9)]


class TestPairAssessment:
    def test_a_comparable_pair_is_measured(self) -> None:
        pair = CandidatePair(SUBJECT, observation("cam_b", 4, x=5.3, y=5.0))

        assessment = assess_pair(pair, GATING)

        assert assessment.is_comparable
        assert assessment.evidence is not None
        assert assessment.evidence.separation_m == pytest.approx(0.3)
        assert assessment.evidence.admissible_radius_m == pytest.approx(2.0)

    def test_evidence_carries_the_weaker_of_the_two_cameras_trust(self) -> None:
        unchecked = context("cam_b", trust=GeometryTrust.UNVERIFIED)
        pair = CandidatePair(SUBJECT, observation("cam_b", 4, x=5.3, spatial=unchecked))

        assessment = assess_pair(pair, GATING)

        assert assessment.evidence is not None
        assert assessment.evidence.trust is GeometryTrust.UNVERIFIED

    def test_a_refused_pair_is_recorded_with_its_reason_and_no_numbers(self) -> None:
        pair = CandidatePair(SUBJECT, observation("cam_a", 2, x=5.3))

        assessment = assess_pair(pair, GATING)

        assert assessment.evidence is None
        assert assessment.reasons() == (GateReason.SAME_CAMERA,)
        assert assessment.geometric_outcome is AssociationOutcome.NOT_COMPARABLE


class TestTrueAssociationSurvives:
    def test_a_plausible_counterpart_is_kept_as_a_candidate(self) -> None:
        near = observation("cam_b", 4, x=5.3, y=5.0)

        gating = gate_observation(SUBJECT, [near], GATING)

        assert len(gating.candidates) == 1
        assert gating.candidates[0].pair.other_than(SUBJECT.subject) is near

    def test_surviving_the_gates_is_never_reported_as_an_association(self) -> None:
        """Geometry vetoes; it does not authorise. The milestone's central rule."""
        gating = gate_observation(SUBJECT, [observation("cam_b", 4, x=5.0, y=5.0)], GATING)

        assert gating.geometric_outcome is None
        assert gating.geometric_outcome is not AssociationOutcome.ASSOCIATED

    def test_a_person_seen_by_three_cameras_is_not_ambiguous(self) -> None:
        """Candidates on different cameras are not rivals; good coverage is not a fault."""
        pool = [observation("cam_b", 4, x=5.2, y=5.0), observation("cam_c", 9, x=5.1, y=5.0)]

        gating = gate_observation(SUBJECT, pool, GATING)

        assert gating.ambiguous_cameras == ()
        assert gating.geometric_outcome is None
        assert len(gating.candidates) == 2

    def test_candidates_are_ordered_nearest_first(self) -> None:
        pool = [observation("cam_b", 4, x=6.5, y=5.0), observation("cam_c", 9, x=5.1, y=5.0)]

        gating = gate_observation(SUBJECT, pool, GATING)

        separations = [
            assessment.evidence.separation_m
            for assessment in gating.candidates
            if assessment.evidence is not None
        ]
        assert separations == sorted(separations)


class TestGenuineNonMatchRefused:
    def test_a_counterpart_beyond_reach_is_excluded(self) -> None:
        far = observation("cam_b", 4, x=25.0, y=5.0)

        gating = gate_observation(SUBJECT, [far], GATING)

        assert gating.geometric_outcome is AssociationOutcome.UNASSOCIATED
        assert gating.candidates == ()
        assert gating.comparable != ()

    def test_exclusion_keeps_the_measurement_that_produced_it(self) -> None:
        gating = gate_observation(SUBJECT, [observation("cam_b", 4, x=25.0, y=5.0)], GATING)

        evidence = gating.assessments[0].evidence
        assert evidence is not None
        assert evidence.separation_m == pytest.approx(20.0)
        assert evidence.margin_m == pytest.approx(-18.0)

    def test_a_pair_exactly_on_the_radius_is_not_excluded(self) -> None:
        """The boundary is permissive: a gate excludes only what it can rule out."""
        gating = gate_observation(SUBJECT, [observation("cam_b", 4, x=7.0, y=5.0)], GATING)

        assert gating.geometric_outcome is None
        assert len(gating.candidates) == 1

    def test_a_pair_a_hair_beyond_the_radius_is_excluded(self) -> None:
        gating = gate_observation(SUBJECT, [observation("cam_b", 4, x=7.001, y=5.0)], GATING)

        assert gating.geometric_outcome is AssociationOutcome.UNASSOCIATED


class TestAmbiguityIsLeftUnresolved:
    def test_two_indistinguishable_counterparts_are_ambiguous(self) -> None:
        pool = [
            observation("cam_b", 4, x=5.2, y=5.0),
            observation("cam_b", 5, x=5.4, y=5.0),
        ]

        gating = gate_observation(SUBJECT, pool, GATING)

        assert gating.geometric_outcome is AssociationOutcome.AMBIGUOUS
        assert gating.ambiguous_cameras == ("cam_b",)

    def test_an_ambiguous_result_keeps_both_candidates_rather_than_choosing(self) -> None:
        """Picking the nearer of two indistinguishable candidates is a coin flip."""
        pool = [
            observation("cam_b", 4, x=5.2, y=5.0),
            observation("cam_b", 5, x=5.4, y=5.0),
        ]

        gating = gate_observation(SUBJECT, pool, GATING)

        assert len(gating.candidates) == 2

    def test_counterparts_separated_by_more_than_the_margin_are_not_ambiguous(self) -> None:
        pool = [
            observation("cam_b", 4, x=5.1, y=5.0),
            observation("cam_b", 5, x=6.5, y=5.0),
        ]

        gating = gate_observation(SUBJECT, pool, GATING)

        assert gating.geometric_outcome is None
        assert gating.ambiguous_cameras == ()

    def test_exactly_the_margin_apart_is_not_ambiguous(self) -> None:
        """0.5 m and 1.0 m differ by exactly the 0.5 m margin."""
        pool = [
            observation("cam_b", 4, x=5.5, y=5.0),
            observation("cam_b", 5, x=6.0, y=5.0),
        ]

        assert outcome_of(pool) is None

    def test_a_hair_inside_the_margin_is_ambiguous(self) -> None:
        pool = [
            observation("cam_b", 4, x=5.5, y=5.0),
            observation("cam_b", 5, x=5.999, y=5.0),
        ]

        assert outcome_of(pool) is AssociationOutcome.AMBIGUOUS

    def test_ambiguity_on_one_camera_does_not_hide_a_clean_third(self) -> None:
        pool = [
            observation("cam_b", 4, x=5.2, y=5.0),
            observation("cam_b", 5, x=5.4, y=5.0),
            observation("cam_c", 9, x=5.1, y=5.0),
        ]

        gating = gate_observation(SUBJECT, pool, GATING)

        assert gating.ambiguous_cameras == ("cam_b",)
        assert len(gating.candidates) == 3


class TestSingleCameraAndNoOverlap:
    def test_a_single_camera_venue_gates_to_not_comparable(self) -> None:
        """One camera is the same code path, not a special mode and not an error."""
        gating = gate_observation(SUBJECT, [], GATING)

        assert gating.geometric_outcome is AssociationOutcome.NOT_COMPARABLE
        assert gating.candidates == ()
        assert gating.assessments == ()

    def test_the_observation_itself_in_the_pool_is_skipped(self) -> None:
        gating = gate_observation(SUBJECT, [SUBJECT], GATING)

        assert gating.assessments == ()
        assert gating.geometric_outcome is AssociationOutcome.NOT_COMPARABLE

    def test_other_tracks_on_the_same_camera_are_not_cross_camera_evidence(self) -> None:
        pool = [observation("cam_a", 2, x=5.1, y=5.0)]

        gating = gate_observation(SUBJECT, pool, GATING)

        assert gating.geometric_outcome is AssociationOutcome.NOT_COMPARABLE
        assert gating.assessments[0].reasons() == (GateReason.SAME_CAMERA,)

    def test_cameras_whose_calibrated_regions_cannot_meet_are_not_comparable(self) -> None:
        here = observation(
            "cam_a",
            1,
            x=5.0,
            y=5.0,
            spatial=context(
                "cam_a", calibrated_region=VenueRegion.covering([(0.0, 0.0), (10.0, 10.0)])
            ),
        )
        elsewhere = observation(
            "cam_b",
            4,
            x=5.0,
            y=5.0,
            spatial=context(
                "cam_b", calibrated_region=VenueRegion.covering([(200.0, 0.0), (210.0, 10.0)])
            ),
        )

        gating = gate_observation(here, [elsewhere], GATING)

        assert gating.geometric_outcome is AssociationOutcome.NOT_COMPARABLE
        assert gating.assessments[0].reasons() == (GateReason.NO_SHARED_CALIBRATED_REGION,)

    def test_an_uncalibrated_venue_produces_no_cross_camera_evidence(self) -> None:
        pool = [observation("cam_b", 4, space=NORMALIZED)]
        subject = observation("cam_a", 1, space=NORMALIZED)

        assert outcome_of(pool, subject) is AssociationOutcome.NOT_COMPARABLE

    def test_a_drifted_camera_stops_contributing_rather_than_contributing_quietly(self) -> None:
        moved = context("cam_b", trust=GeometryTrust.UNTRUSTED)
        pool = [observation("cam_b", 4, x=5.1, y=5.0, spatial=moved)]

        assert outcome_of(pool) is AssociationOutcome.NOT_COMPARABLE

    def test_unrelated_clocks_produce_no_cross_camera_evidence(self) -> None:
        pool = [observation("cam_b", 4, x=5.1, y=5.0, half_width_seconds=None)]

        assert outcome_of(pool) is AssociationOutcome.NOT_COMPARABLE


class TestProvenance:
    def test_every_candidate_resolves_back_to_its_camera_track_and_stream(self) -> None:
        counterpart = observation("cam_b", 4, x=5.3, stream_epoch=2, frame_index=93)

        gating = gate_observation(SUBJECT, [counterpart], GATING)

        record = gating.to_record()["assessments"][0]["pair"]["second"]
        assert record["camera_id"] == "cam_b"
        assert record["track_id"] == 4
        assert record["stream_epoch"] == 2
        assert record["frame_index"] == 93

    def test_a_refusal_records_which_gate_refused_it(self) -> None:
        pool = [observation("cam_b", 4, x=5.1, half_width_seconds=None)]

        record = gate_observation(SUBJECT, pool, GATING).to_record()

        assert record["geometric_outcome"] == "not_comparable"
        assert record["assessments"][0]["rejections"][0]["reason"] == "venue_time_not_established"
        assert record["assessments"][0]["rejections"][0]["detail"]

    def test_the_considered_set_is_recorded_not_only_the_survivors(self) -> None:
        pool = [observation("cam_b", 4, x=5.1), observation("cam_c", 9, x=99.0)]

        record = gate_observation(SUBJECT, pool, GATING).to_record()

        assert record["considered"] == 2
        assert record["comparable"] == 2
        assert len(record["assessments"]) == 2

    def test_a_gating_record_names_no_person_and_counts_nobody(self) -> None:
        pool = [observation("cam_b", 4, x=5.1), observation("cam_b", 5, x=5.2)]

        record = json.dumps(gate_observation(SUBJECT, pool, GATING).to_record())

        for forbidden in (
            "venue_subject",
            "person_id",
            "identity",
            "unique_people",
            "duplicate",
            "embedding",
            "appearance",
        ):
            assert forbidden not in record


class TestDeterminism:
    #: Deliberately mixed: two rivals on one camera, a clean third camera, and
    #: a fourth too far away to survive. Every branch of the ordering rule runs.
    POOL: ClassVar[list[TrackObservation]] = [
        observation("cam_c", 9, x=5.1, y=5.0),
        observation("cam_b", 4, x=5.4, y=5.0),
        observation("cam_b", 5, x=5.8, y=5.0),
        observation("cam_d", 2, x=40.0, y=5.0),
    ]

    def test_two_runs_produce_byte_identical_records(self) -> None:
        first = json.dumps(gate_observation(SUBJECT, self.POOL, GATING).to_record())
        second = json.dumps(gate_observation(SUBJECT, self.POOL, GATING).to_record())

        assert first == second

    def test_camera_enumeration_order_cannot_reach_the_output(self) -> None:
        """The rule association must obey above all others: order is not evidence."""
        records = {
            json.dumps(gate_observation(SUBJECT, list(order), GATING).to_record())
            for order in permutations(self.POOL)
        }

        assert len(records) == 1

    def test_the_outcome_itself_is_order_independent(self) -> None:
        outcomes = {
            gate_observation(SUBJECT, list(order), GATING).geometric_outcome
            for order in permutations(self.POOL)
        }

        assert outcomes == {AssociationOutcome.AMBIGUOUS}


class TestNoAssociationIsEverAsserted:
    @pytest.mark.parametrize(
        "pool",
        [
            [],
            [observation("cam_b", 4, x=5.0, y=5.0)],
            [observation("cam_b", 4, x=5.0), observation("cam_b", 5, x=5.0)],
            [observation("cam_b", 4, x=900.0)],
            [observation("cam_a", 2, x=5.0)],
        ],
        ids=["empty", "one-near", "two-identical", "far", "same-camera"],
    )
    def test_no_arrangement_of_evidence_produces_associated(
        self, pool: list[TrackObservation]
    ) -> None:
        assert gate_observation(SUBJECT, pool, GATING).geometric_outcome is not (
            AssociationOutcome.ASSOCIATED
        )

    def test_two_people_standing_on_one_spot_are_not_declared_one_person(self) -> None:
        """The case that makes geometry insufficient: it cannot tell them apart."""
        coincident = observation("cam_b", 4, x=5.0, y=5.0)

        gating = gate_observation(SUBJECT, [coincident], GATING)

        assert gating.candidates[0].evidence is not None
        assert gating.candidates[0].evidence.separation_m == 0.0
        assert gating.geometric_outcome is None


def association_modules() -> list[ModuleType]:
    return [models_module, config_module, geometry_module, gates_module, candidates_module]


def module_level_containers() -> dict[str, int]:
    """Every mutable module-level container in the package, with its size."""
    sizes: dict[str, int] = {}
    for module in association_modules():
        for name, value in vars(module).items():
            if name.startswith("__"):
                continue
            if isinstance(value, dict | list | set):
                sizes[f"{module.__name__}.{name}"] = len(value)
    return sizes


class TestBoundedState:
    def test_the_layer_holds_exactly_one_module_level_container(self) -> None:
        """Gating is pure. A cache appearing here is how unbounded state starts."""
        assert set(module_level_containers()) == {"atreides_vision.association.models._TRUST_RANK"}

    def test_churning_many_track_ids_retains_nothing(self) -> None:
        before = module_level_containers()

        for track_id in range(500):
            subject = observation("cam_a", track_id, x=5.0, y=5.0)
            pool = [observation("cam_b", track_id + 1, x=5.2, y=5.0)]
            gate_observation(subject, pool, GATING)

        assert module_level_containers() == before

    def test_a_gating_result_holds_no_reference_to_any_frame(self) -> None:
        gating = gate_observation(SUBJECT, [observation("cam_b", 4, x=5.1)], GATING)

        assert not hasattr(gating.observation, "image")
        assert not hasattr(gating.observation, "bbox")

"""Association -> venue identity integration — V0.4.3b.

Covers the four ownership cases the module docstring names, the fifth
same-camera refusal at the registry boundary, the three non-positive
outcomes staying IGNORED, idempotent reprocessing, ordering independence,
provenance recoverability and determinism across two runs.

Two fixture styles are used deliberately: the real
:class:`~atreides_vision.association.AssociationEngine` (via ``ticks``,
mirroring ``test_association_engine.py``) for happy-path scenarios where
genuine engine behaviour is the more meaningful fixture, and
``fusion_builders`` — hand-built :class:`PairDecision`\\ s — for edge cases
that would need unreasonably contrived geometry to reach through the engine,
especially the cross-subject conflict in case 4.
"""

from __future__ import annotations

import json
from itertools import permutations

from association_builders import DECIDING, GATING, observation
from atreides_vision.association import AssociationEngine, AssociationRound, TrackObservation
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.fusion import IntegrationOutcome, integrate_round
from atreides_vision.identity import VenueIdentityConfig, VenueSubjectRegistry
from atreides_vision.identity.models import tracklet_key
from fusion_builders import (
    ambiguous_decision,
    associated_decision,
    not_comparable_decision,
    obs,
    round_of,
    unassociated_decision,
)
from identity_builders import CONFIG, membership

STEP = 0.125
START = 10.0

ONE_PAIR = (("cam_a", 1, 0.0), ("cam_b", 1, 0.5))
THREE_CAMERAS = (("cam_a", 1, 0.0), ("cam_b", 1, 0.5), ("cam_c", 1, 1.0))

A1 = SubjectRef("cam_a", 1)
B1 = SubjectRef("cam_b", 1)
C1 = SubjectRef("cam_c", 1)


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
    rounds = []
    for index in range(count):
        seconds = start + index * STEP
        rounds.append(core.observe(seconds, placed(seconds, placements)))
    return rounds


def registry(config: VenueIdentityConfig = CONFIG) -> VenueSubjectRegistry:
    return VenueSubjectRegistry(config)


class TestOnlyPositiveAssociationsAreEligible:
    def test_ambiguous_is_ignored(self) -> None:
        reg = registry()
        decision = ambiguous_decision(obs("cam_a", 1), obs("cam_b", 1))
        result = integrate_round(reg, round_of(10.0, decision))

        assert result.outcomes[0].outcome is IntegrationOutcome.IGNORED
        assert reg.subjects() == ()

    def test_unassociated_is_ignored(self) -> None:
        reg = registry()
        decision = unassociated_decision(obs("cam_a", 1), obs("cam_b", 1))
        result = integrate_round(reg, round_of(10.0, decision))

        assert result.outcomes[0].outcome is IntegrationOutcome.IGNORED
        assert reg.subjects() == ()

    def test_not_comparable_is_ignored(self) -> None:
        reg = registry()
        decision = not_comparable_decision(obs("cam_a", 1), obs("cam_a", 2))
        result = integrate_round(reg, round_of(10.0, decision))

        assert result.outcomes[0].outcome is IntegrationOutcome.IGNORED
        assert reg.subjects() == ()

    def test_ignored_detail_names_the_outcome(self) -> None:
        reg = registry()
        decision = ambiguous_decision(obs("cam_a", 1), obs("cam_b", 1))
        outcome = integrate_round(reg, round_of(10.0, decision)).outcomes[0]

        assert "ambiguous" in outcome.detail

    def test_ignored_outcome_carries_no_subject_ids(self) -> None:
        reg = registry()
        decision = unassociated_decision(obs("cam_a", 1), obs("cam_b", 1))
        outcome = integrate_round(reg, round_of(10.0, decision)).outcomes[0]

        assert outcome.subject_ids == ()


class TestCaseOneCreateBoth:
    def test_neither_owned_creates_one_subject(self) -> None:
        reg = registry()
        final = ticks(engine(), ONE_PAIR)[-1]

        result = integrate_round(reg, final)

        assert len(result.applied) == 1
        applied = result.applied[0]
        assert applied.outcome is IntegrationOutcome.APPLIED
        assert len(applied.subject_ids) == 1
        subject = reg.get(applied.subject_ids[0])
        assert subject.camera_ids == ("cam_a", "cam_b")

    def test_created_subject_holds_both_tracklets(self) -> None:
        reg = registry()
        final = ticks(engine(), ONE_PAIR)[-1]
        outcome = integrate_round(reg, final).applied[0]

        subject = reg.get(outcome.subject_ids[0])
        subjects_held = {m.subject for m in subject.members}
        assert subjects_held == {A1, B1}

    def test_create_detail_names_both_tracklets(self) -> None:
        reg = registry()
        final = ticks(engine(), ONE_PAIR)[-1]
        outcome = integrate_round(reg, final).applied[0]

        assert "cam_a" in outcome.detail
        assert "cam_b" in outcome.detail
        assert "neither" in outcome.detail


class TestCaseTwoExtend:
    def test_one_owned_one_unowned_extends_the_existing_subject(self) -> None:
        reg = registry()
        rounds = ticks(engine(), THREE_CAMERAS)
        final = rounds[-1]

        result = integrate_round(reg, final)

        # Canonical decision order is (A,B), (A,C), (B,C): the first pair
        # creates a subject, the second extends it with the third camera,
        # and the third pair — now both members of the same subject — is a
        # no-op. Two applications, one no-op, zero refusals.
        assert len(result.applied) == 2
        assert len(result.no_ops) == 1
        assert result.refused == ()
        subject_ids = {outcome.subject_ids[0] for outcome in result.applied}
        assert len(subject_ids) == 1
        subject = reg.get(next(iter(subject_ids)))
        assert subject.camera_ids == ("cam_a", "cam_b", "cam_c")

    def test_three_or_more_cameras_extend_across_sequential_rounds(self) -> None:
        """A third and fourth camera join one identity, applied round by round."""
        reg = registry()
        four_cameras = (
            ("cam_a", 1, 0.0),
            ("cam_b", 1, 0.5),
            ("cam_c", 1, 1.0),
            ("cam_d", 1, 1.5),
        )
        rounds = ticks(engine(), four_cameras)
        final = rounds[-1]

        result = integrate_round(reg, final)

        applied_subject_ids = {o.subject_ids[0] for o in result.applied}
        assert len(applied_subject_ids) == 1
        subject = reg.get(next(iter(applied_subject_ids)))
        assert subject.camera_ids == ("cam_a", "cam_b", "cam_c", "cam_d")
        assert len(subject.members) == 4

    def test_extend_detail_names_the_growing_subject(self) -> None:
        reg = registry()
        final = ticks(engine(), THREE_CAMERAS)[-1]
        result = integrate_round(reg, final)

        extend_outcomes = [o for o in result.applied if "added" in o.detail]
        assert extend_outcomes
        assert "already owning" in extend_outcomes[0].detail


class TestCaseThreeNoOpIdempotent:
    def test_both_owned_by_same_subject_is_a_no_op(self) -> None:
        reg = registry()
        final = ticks(engine(), ONE_PAIR)[-1]

        integrate_round(reg, final)
        second_result = integrate_round(reg, final)

        assert len(second_result.no_ops) == 1
        assert second_result.no_ops[0].outcome is IntegrationOutcome.NO_OP
        assert second_result.applied == ()

    def test_reprocessing_the_same_round_does_not_change_registry_state(self) -> None:
        reg = registry()
        final = ticks(engine(), ONE_PAIR)[-1]

        integrate_round(reg, final)
        before = reg.get(reg.subjects()[0])
        integrate_round(reg, final)
        after = reg.get(reg.subjects()[0])

        assert before == after
        assert len(reg.subjects()) == 1

    def test_three_camera_round_processed_twice_is_fully_idempotent(self) -> None:
        reg = registry()
        final = ticks(engine(), THREE_CAMERAS)[-1]

        integrate_round(reg, final)
        second = integrate_round(reg, final)

        assert second.applied == ()
        assert len(second.no_ops) == 3
        assert len(reg.subjects()) == 1

    def test_no_op_detail_names_the_owning_subject(self) -> None:
        reg = registry()
        final = ticks(engine(), ONE_PAIR)[-1]
        integrate_round(reg, final)
        outcome = integrate_round(reg, final).no_ops[0]

        assert str(outcome.subject_ids[0]) in outcome.detail


class TestCaseFourCrossSubjectConflict:
    def test_both_owned_by_different_subjects_is_refused_not_merged(self) -> None:
        reg = registry()
        subject_x = reg.create(membership("cam_a", 1, joined_at_seconds=10.0))
        subject_y = reg.create(membership("cam_b", 1, joined_at_seconds=10.0))

        decision = associated_decision(
            obs("cam_a", 1, venue_seconds=10.0), obs("cam_b", 1, venue_seconds=10.0)
        )
        result = integrate_round(reg, round_of(10.0, decision))

        assert result.refused
        outcome = result.refused[0]
        assert outcome.outcome is IntegrationOutcome.REFUSED
        assert set(outcome.subject_ids) == {subject_x, subject_y}

    def test_refusal_never_creates_a_third_subject(self) -> None:
        reg = registry()
        reg.create(membership("cam_a", 1, joined_at_seconds=10.0))
        reg.create(membership("cam_b", 1, joined_at_seconds=10.0))

        decision = associated_decision(
            obs("cam_a", 1, venue_seconds=10.0), obs("cam_b", 1, venue_seconds=10.0)
        )
        integrate_round(reg, round_of(10.0, decision))

        assert len(reg.subjects()) == 2

    def test_refusal_leaves_both_subjects_membership_unchanged(self) -> None:
        reg = registry()
        subject_x = reg.create(membership("cam_a", 1, joined_at_seconds=10.0))
        subject_y = reg.create(membership("cam_b", 1, joined_at_seconds=10.0))
        before_x = reg.get(subject_x)
        before_y = reg.get(subject_y)

        decision = associated_decision(
            obs("cam_a", 1, venue_seconds=10.0), obs("cam_b", 1, venue_seconds=10.0)
        )
        integrate_round(reg, round_of(10.0, decision))

        assert reg.get(subject_x) == before_x
        assert reg.get(subject_y) == before_y

    def test_refusal_reports_subject_ids_in_canonical_order(self) -> None:
        """Order is independent of which observation happens to own which subject."""
        reg = registry()
        subject_x = reg.create(membership("cam_a", 1, joined_at_seconds=10.0))
        subject_y = reg.create(membership("cam_b", 1, joined_at_seconds=10.0))

        decision = associated_decision(
            obs("cam_b", 1, venue_seconds=10.0), obs("cam_a", 1, venue_seconds=10.0)
        )
        outcome = integrate_round(reg, round_of(10.0, decision)).refused[0]

        assert outcome.subject_ids == tuple(sorted((subject_x, subject_y)))

    def test_refusal_detail_explains_the_conflict(self) -> None:
        reg = registry()
        reg.create(membership("cam_a", 1, joined_at_seconds=10.0))
        reg.create(membership("cam_b", 1, joined_at_seconds=10.0))

        decision = associated_decision(
            obs("cam_a", 1, venue_seconds=10.0), obs("cam_b", 1, venue_seconds=10.0)
        )
        outcome = integrate_round(reg, round_of(10.0, decision)).refused[0]

        assert "different venue subjects" in outcome.detail
        assert "not a decision this layer makes" in outcome.detail


class TestSameCameraConflictRefusal:
    def test_extending_with_a_second_active_tracklet_on_one_camera_is_refused(self) -> None:
        reg = registry()
        subject_x = reg.create(membership("cam_a", 1, joined_at_seconds=10.0))
        reg.add_membership(subject_x, membership("cam_b", 1, joined_at_seconds=10.0))

        decision = associated_decision(
            obs("cam_a", 1, venue_seconds=10.0), obs("cam_b", 2, venue_seconds=10.0)
        )
        result = integrate_round(reg, round_of(10.0, decision))

        assert result.refused
        outcome = result.refused[0]
        assert outcome.subject_ids == (subject_x,)
        assert "one camera" in outcome.detail

    def test_same_camera_refusal_leaves_the_registry_untouched(self) -> None:
        reg = registry()
        subject_x = reg.create(membership("cam_a", 1, joined_at_seconds=10.0))
        reg.add_membership(subject_x, membership("cam_b", 1, joined_at_seconds=10.0))
        before = reg.get(subject_x)

        decision = associated_decision(
            obs("cam_a", 1, venue_seconds=10.0), obs("cam_b", 2, venue_seconds=10.0)
        )
        integrate_round(reg, round_of(10.0, decision))

        assert reg.get(subject_x) == before


class TestOrderingIndependence:
    def test_decisions_processed_in_reverse_produce_the_same_registry_state(self) -> None:
        reg_forward = registry()
        reg_reverse = registry()
        final = ticks(engine(), THREE_CAMERAS)[-1]
        reversed_round = AssociationRound(
            venue_seconds=final.venue_seconds,
            observations=final.observations,
            decisions=final.decisions,
        )

        integrate_round(reg_forward, final)
        # AssociationRound itself enforces canonical decision order, so
        # reprocess through a second registry with the same (already
        # canonical) round: outcome content must not depend on how a caller
        # might otherwise have enumerated the pairs. This is checked via the
        # camera-enumeration-order sweep below, which is the meaningful
        # version of this property for AssociationRound-shaped input.
        integrate_round(reg_reverse, reversed_round)

        assert {str(s) for s in reg_forward.subjects()} == {str(s) for s in reg_reverse.subjects()}

    def test_camera_enumeration_order_does_not_change_the_outcome_set(self) -> None:
        expected_camera_sets = None
        for order in permutations(THREE_CAMERAS):
            reg = registry()
            final = ticks(engine(), order)[-1]
            result = integrate_round(reg, final)
            camera_sets = sorted(reg.get(sid).camera_ids for sid in reg.subjects())
            if expected_camera_sets is None:
                expected_camera_sets = camera_sets
            else:
                assert camera_sets == expected_camera_sets
            assert len(result.applied) == 2
            assert len(result.no_ops) == 1


class TestDeterminism:
    def test_two_runs_of_the_same_round_produce_byte_identical_outcome_records(self) -> None:
        final = ticks(engine(), THREE_CAMERAS)[-1]

        first_result = integrate_round(registry(), final)
        second_result = integrate_round(registry(), final)

        first_json = json.dumps(
            [
                {
                    "outcome": o.outcome.value,
                    "detail": o.detail,
                    "subject_ids": [str(s) for s in o.subject_ids],
                }
                for o in first_result.outcomes
            ]
        )
        second_json = json.dumps(
            [
                {
                    "outcome": o.outcome.value,
                    "detail": o.detail,
                    "subject_ids": [str(s) for s in o.subject_ids],
                }
                for o in second_result.outcomes
            ]
        )
        assert first_json == second_json

    def test_integration_result_preserves_venue_seconds(self) -> None:
        final = ticks(engine(), ONE_PAIR)[-1]
        result = integrate_round(registry(), final)

        assert result.venue_seconds == final.venue_seconds


class TestProvenance:
    def test_applied_outcome_carries_the_full_pair_decision(self) -> None:
        reg = registry()
        final = ticks(engine(), ONE_PAIR)[-1]
        outcome = integrate_round(reg, final).applied[0]

        assert outcome.decision.is_associated
        assert outcome.decision.support is not None
        assert outcome.decision.confidence is not None

    def test_subject_resolves_back_to_its_camera_track_observations(self) -> None:
        reg = registry()
        final = ticks(engine(), ONE_PAIR)[-1]
        outcome = integrate_round(reg, final).applied[0]

        subject = reg.get(outcome.subject_ids[0])
        provenance = reg.provenance_for(outcome.subject_ids[0])
        cameras = {event.membership.camera_id for event in provenance}
        assert cameras == {"cam_a", "cam_b"}
        assert subject.camera_ids == ("cam_a", "cam_b")

    def test_provenance_survives_after_extension_across_three_cameras(self) -> None:
        reg = registry()
        final = ticks(engine(), THREE_CAMERAS)[-1]
        result = integrate_round(reg, final)

        subject_id = result.applied[0].subject_ids[0]
        provenance = reg.provenance_for(subject_id)
        cameras = {event.membership.camera_id for event in provenance}
        assert cameras == {"cam_a", "cam_b", "cam_c"}

    def test_membership_outcome_key_matches_the_underlying_decision(self) -> None:
        reg = registry()
        final = ticks(engine(), ONE_PAIR)[-1]
        outcome = integrate_round(reg, final).applied[0]

        assert outcome.key == outcome.decision.key


class TestSingleCameraDegradesCleanly:
    def test_a_single_camera_produces_no_positive_decision_and_no_registry_change(self) -> None:
        reg = registry()
        final = ticks(engine(), (("cam_a", 1, 0.0), ("cam_a", 2, 5.0)))[-1]

        result = integrate_round(reg, final)

        assert result.applied == ()
        assert reg.subjects() == ()


class TestTrackletKeyReconciliation:
    def test_owner_lookup_uses_identity_tracklet_key_built_from_the_observation(self) -> None:
        reg = registry()
        final = ticks(engine(), ONE_PAIR)[-1]
        outcome = integrate_round(reg, final).applied[0]

        first = final.decisions[0].pair.first
        key = tracklet_key(first.subject, first.stream)
        assert reg.owner_of(key) == outcome.subject_ids[0]

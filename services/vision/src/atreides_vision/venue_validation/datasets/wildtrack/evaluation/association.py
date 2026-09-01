"""Per-instant association scoring for the offline WILDTRACK evaluator.

This module consumes only *factual* values: an immutable
:class:`~atreides_vision.venue_validation.observer.VenueValidationInstantSnapshot`
(the runtime's own frozen prediction) and WILDTRACK ground truth. It imports
nothing from :mod:`atreides_vision.association` beyond the frozen domain model
(:class:`~atreides_vision.association.models.AssociationOutcome`,
:class:`~atreides_vision.association.models.DecisionReason`, ``tracklet_key``);
it never touches the association engine, candidate builder, gates or matcher.

Outcome semantics: a cross-camera pair is *missed* whenever its decision is not
``ASSOCIATED``. ``AMBIGUOUS`` (including a ``support_not_sustained`` pair that is
still being weighed) and ``UNASSOCIATED`` are both missed associations, counted
together in ``missed_association_count`` and split out on their own axes
(``ambiguous_pair_count``, ``support_not_sustained_count``).
``support_not_sustained`` is an ``AMBIGUOUS`` reason and is never relabelled an
``UNASSOCIATED`` reason.

Totality is enforced, never inferred: for a ground-truth cross-camera pair whose
two cameras each have a matched runtime tracklet, the association round *must*
carry a decision for that tracklet pair. Its absence is a contract violation and
raises :class:`~atreides_vision.errors.ValidationError` — a missing decision is
never read as ``UNASSOCIATED`` and there is no ``NO_DECISION`` outcome.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from atreides_vision.association.models import (
    AssociationOutcome,
    DecisionReason,
    PairDecision,
    TrackletKey,
    tracklet_key,
)
from atreides_vision.errors import ValidationError
from atreides_vision.identity.models import VenueSubjectId
from atreides_vision.venue_validation.datasets.wildtrack.annotations import (
    WildtrackFrameAnnotations,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.models import (
    CameraMatch,
    FalseMergeDiagnostic,
    InstantAssociationEvaluation,
)
from atreides_vision.venue_validation.observer import VenueValidationInstantSnapshot

__all__ = ["score_instant_association"]

CandidateKey = tuple[TrackletKey, TrackletKey]

_MICROMETRES_PER_METRE = 1_000_000


def _local_match_distance_m(
    camera_matches: Mapping[str, CameraMatch], key: TrackletKey
) -> float | None:
    match = camera_matches.get(key[0])
    if match is None:
        return None
    micrometres = match.match_distance_micrometres_for_observation(key)
    if micrometres is None:
        return None
    return micrometres / _MICROMETRES_PER_METRE


def _candidate_key(first: TrackletKey, second: TrackletKey) -> CandidateKey:
    return (first, second) if first <= second else (second, first)


def _owner_by_tracklet(
    snapshot: VenueValidationInstantSnapshot,
) -> dict[TrackletKey, VenueSubjectId]:
    owner: dict[TrackletKey, VenueSubjectId] = {}
    for membership in snapshot.memberships:
        for member in membership.members:
            owner[member] = membership.subject_id
    return owner


def _cross_camera_pairs(
    frame: WildtrackFrameAnnotations,
    selected: Sequence[str],
) -> list[tuple[int, str, str]]:
    selected_set = set(selected)
    pairs: list[tuple[int, str, str]] = []
    for person in frame.persons:
        cameras = [camera for camera in person.visible_camera_ids if camera in selected_set]
        for i in range(len(cameras)):
            for j in range(i + 1, len(cameras)):
                pairs.append((person.person_id.value, cameras[i], cameras[j]))
    return pairs


def score_instant_association(
    snapshot: VenueValidationInstantSnapshot,
    frame: WildtrackFrameAnnotations,
    camera_matches: Mapping[str, CameraMatch],
    *,
    selected_camera_ids: Sequence[str],
) -> InstantAssociationEvaluation:
    """Score one venue instant's cross-camera association against ground truth."""
    decision_by_key: dict[CandidateKey, PairDecision] = {}
    for decision in snapshot.association_round.decisions:
        first_key = tracklet_key(decision.pair.first)
        second_key = tracklet_key(decision.pair.second)
        decision_by_key[_candidate_key(first_key, second_key)] = decision

    owner = _owner_by_tracklet(snapshot)

    true_positive_merges = 0
    false_merges = 0
    unresolved_for_scoring = 0
    evaluable_pair_count = 0
    evaluable_associated_count = 0
    missed_association_count = 0
    ambiguous_pair_count = 0
    support_not_sustained_count = 0
    not_evaluable_pair_count = 0
    end_to_end_pair_count = 0
    end_to_end_satisfied_count = 0
    gate_reason_counts: Counter[str] = Counter()
    decision_reason_counts: Counter[str] = Counter()

    for person_id, camera_a, camera_b in _cross_camera_pairs(frame, selected_camera_ids):
        key_a = camera_matches[camera_a].observation_for_person(person_id)
        key_b = camera_matches[camera_b].observation_for_person(person_id)

        end_to_end_pair_count += 1

        if key_a is None or key_b is None:
            not_evaluable_pair_count += 1
            continue

        candidate = _candidate_key(key_a, key_b)
        pair_decision = decision_by_key.get(candidate)
        if pair_decision is None:
            raise ValidationError(
                "The association round carries no decision for co-present cross-camera "
                f"tracklets {key_a} and {key_b} (WILDTRACK person {person_id}). A missing "
                "decision is a totality violation, not an implicit non-association."
            )

        evaluable_pair_count += 1
        if pair_decision.outcome is AssociationOutcome.ASSOCIATED:
            evaluable_associated_count += 1
        else:
            missed_association_count += 1
            if pair_decision.outcome is AssociationOutcome.AMBIGUOUS:
                ambiguous_pair_count += 1
            if DecisionReason.SUPPORT_NOT_SUSTAINED in pair_decision.reasons:
                support_not_sustained_count += 1
            for rejection in pair_decision.assessment.rejections:
                gate_reason_counts[rejection.reason.value] += 1
            for reason in pair_decision.reasons:
                decision_reason_counts[reason.value] += 1

        owner_a = owner.get(key_a)
        owner_b = owner.get(key_b)
        if owner_a is not None and owner_a == owner_b:
            end_to_end_satisfied_count += 1

    gt_position_by_person = {
        person.person_id.value: (person.position.x_m, person.position.y_m)
        for person in frame.persons
    }
    false_merges_detail: list[FalseMergeDiagnostic] = []

    for decision in snapshot.association_round.associated:
        obs_a = decision.pair.first
        obs_b = decision.pair.second
        key_a = tracklet_key(obs_a)
        key_b = tracklet_key(obs_b)
        person_a = _person_for(camera_matches, key_a)
        person_b = _person_for(camera_matches, key_b)
        if person_a is None or person_b is None:
            unresolved_for_scoring += 1
        elif person_a == person_b:
            true_positive_merges += 1
        else:
            false_merges += 1
            endpoints = sorted(
                (
                    (key_a, obs_a, person_a),
                    (key_b, obs_b, person_b),
                ),
                key=lambda endpoint: endpoint[0],
            )
            first_endpoint, second_endpoint = endpoints
            first_key, first_obs, first_person = first_endpoint
            second_key, second_obs, second_person = second_endpoint
            first_distance = _local_match_distance_m(camera_matches, first_key)
            second_distance = _local_match_distance_m(camera_matches, second_key)
            false_merges_detail.append(
                FalseMergeDiagnostic(
                    dataset_frame_id=frame.frame_id.value,
                    venue_seconds=snapshot.venue_seconds,
                    first_tracklet=first_key,
                    second_tracklet=second_key,
                    first_camera_id=first_obs.camera_id,
                    second_camera_id=second_obs.camera_id,
                    first_person_id=first_person,
                    second_person_id=second_person,
                    first_runtime_position=(first_obs.position.x, first_obs.position.y),
                    second_runtime_position=(second_obs.position.x, second_obs.position.y),
                    first_ground_truth_position=gt_position_by_person[first_person],
                    second_ground_truth_position=gt_position_by_person[second_person],
                    first_local_match_distance_m=first_distance,
                    second_local_match_distance_m=second_distance,
                    outcome=AssociationOutcome.ASSOCIATED.value,
                    reasons=tuple(reason.value for reason in decision.reasons),
                )
            )

    return InstantAssociationEvaluation(
        true_positive_merges=true_positive_merges,
        false_merges=false_merges,
        unresolved_for_scoring=unresolved_for_scoring,
        evaluable_pair_count=evaluable_pair_count,
        evaluable_associated_count=evaluable_associated_count,
        missed_association_count=missed_association_count,
        ambiguous_pair_count=ambiguous_pair_count,
        support_not_sustained_count=support_not_sustained_count,
        not_evaluable_pair_count=not_evaluable_pair_count,
        end_to_end_pair_count=end_to_end_pair_count,
        end_to_end_satisfied_count=end_to_end_satisfied_count,
        gate_reason_counts=dict(gate_reason_counts),
        decision_reason_counts=dict(decision_reason_counts),
        false_merges_detail=tuple(
            sorted(
                false_merges_detail,
                key=lambda diagnostic: (
                    diagnostic.dataset_frame_id,
                    diagnostic.first_tracklet,
                    diagnostic.second_tracklet,
                ),
            )
        ),
    )


def _person_for(
    camera_matches: Mapping[str, CameraMatch],
    key: TrackletKey,
) -> int | None:
    camera_id = key[0]
    match = camera_matches.get(camera_id)
    if match is None:
        return None
    return match.person_for_observation(key)

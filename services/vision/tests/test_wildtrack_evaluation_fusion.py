"""Per-instant fusion / venue-subject purity scoring for the WILDTRACK evaluator.

Both failure directions are pinned analytically: an over-merge (one venue
subject spanning two WILDTRACK people) and an under-merge (one WILDTRACK person
fragmented across two venue subjects), plus a matched tracklet that no subject
owns, and provenance that survives fusion.
"""

from __future__ import annotations

from atreides_vision.association.models import AssociationOutcome, DecisionReason
from atreides_vision.venue_validation.datasets.wildtrack.annotations import (
    WildtrackFrameAnnotations,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.config import EvaluationConfig
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.evaluator import (
    evaluate_instant,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.models import (
    InstantFusionEvaluation,
)
from atreides_vision.venue_validation.observer import VenueValidationInstantSnapshot
from fusion_builders import associated_decision
from wildtrack_evaluation_builders import (
    ground_truth_of,
    gt_frame,
    gt_person,
    membership,
    metric_obs,
    missed_decision,
    snapshot_for,
    tracklet_key,
)

CONFIG = EvaluationConfig(selected_camera_ids=("C1", "C2"))


def _fusion(
    snapshot: VenueValidationInstantSnapshot, frame: WildtrackFrameAnnotations
) -> InstantFusionEvaluation:
    return evaluate_instant(snapshot, ground_truth_of(frame), CONFIG, instant_index=0).fusion


def test_a_correctly_fused_subject_is_pure_and_not_fragmented() -> None:
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    b = metric_obs("C2", 2, x=0.0, y=0.0)
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1", "C2")))
    snapshot = snapshot_for(
        0,
        observations=[a, b],
        decisions=[associated_decision(a, b)],
        memberships=[membership(1, a, b)],
    )

    result = _fusion(snapshot, frame)

    assert result.evaluable_subject_count == 1
    assert result.impure_subject_count == 0
    assert result.persons_with_owned_match == 1
    assert result.fragmented_person_count == 0
    assert result.fusion_impurity.value == 0.0


def test_an_over_merge_of_two_people_into_one_subject_is_impure() -> None:
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    b = metric_obs("C2", 2, x=5.0, y=5.0)
    frame = gt_frame(
        0,
        gt_person(1, x=0.0, y=0.0, cameras=("C1",)),
        gt_person(2, x=5.0, y=5.0, cameras=("C2",)),
    )
    snapshot = snapshot_for(0, observations=[a, b], memberships=[membership(1, a, b)])

    result = _fusion(snapshot, frame)

    assert result.evaluable_subject_count == 1
    assert result.impure_subject_count == 1
    assert result.fusion_impurity.value == 1.0


def test_a_person_split_across_two_subjects_is_fragmented() -> None:
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    b = metric_obs("C2", 2, x=0.0, y=0.0)
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1", "C2")))
    snapshot = snapshot_for(
        0,
        observations=[a, b],
        decisions=[
            missed_decision(
                a,
                b,
                reason=DecisionReason.SUPPORT_NOT_SUSTAINED,
                outcome=AssociationOutcome.AMBIGUOUS,
            )
        ],
        memberships=[membership(1, a), membership(2, b)],
    )

    result = _fusion(snapshot, frame)

    assert result.persons_with_owned_match == 1
    assert result.fragmented_person_count == 1
    assert result.fragmentation_rate.value == 1.0
    assert result.impure_subject_count == 0


def test_a_subject_with_no_gt_resolved_member_is_not_evaluable() -> None:
    a = metric_obs("C1", 1, x=50.0, y=50.0)
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1",)))
    snapshot = snapshot_for(0, observations=[a], memberships=[membership(1, a)])

    result = _fusion(snapshot, frame)

    assert result.evaluable_subject_count == 0
    assert result.fusion_impurity.value is None


def test_a_matched_tracklet_no_subject_owns_is_reported_as_matched_unowned() -> None:
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1",)))
    snapshot = snapshot_for(0, observations=[a], unowned=[a])

    result = _fusion(snapshot, frame)

    assert result.matched_tracklet_count == 1
    assert result.matched_unowned_count == 1
    assert result.matched_unowned_rate.value == 1.0


def test_matched_unowned_is_deduplicated_to_person_instances_at_person_level() -> None:
    # One WILDTRACK person seen by both cameras; both runtime tracklets match it
    # and neither is owned by a venue subject.
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    b = metric_obs("C2", 2, x=0.0, y=0.0)
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1", "C2")))
    snapshot = snapshot_for(
        0,
        observations=[a, b],
        decisions=[associated_decision(a, b)],
        unowned=[a, b],
    )

    result = _fusion(snapshot, frame)

    assert result.matched_unowned_count == 2  # two tracklets
    assert result.matched_unowned_person_count == 1  # one WILDTRACK person


def test_provenance_resolves_back_to_the_contributing_camera_tracks() -> None:
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    b = metric_obs("C2", 2, x=0.0, y=0.0)
    snapshot = snapshot_for(
        0,
        observations=[a, b],
        decisions=[associated_decision(a, b)],
        memberships=[membership(1, a, b)],
    )

    (only,) = snapshot.memberships
    assert set(only.members) == {tracklet_key(a), tracklet_key(b)}

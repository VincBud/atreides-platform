"""Per-instant association scoring for the offline WILDTRACK evaluator (V0.4.9b-D1).

The failure directions are covered explicitly: a true association found, a
genuine wrong merge caught, a real pair the runtime refused (missed), an
ambiguous pair left unmerged, a co-present pair with no decision at all
(a totality violation), and a pair that is simply not evaluable.
"""

from __future__ import annotations

import pytest

from atreides_vision.association.models import (
    AssociationOutcome,
    DecisionReason,
    PairDecision,
    TrackObservation,
)
from atreides_vision.errors import ValidationError
from atreides_vision.venue_validation.datasets.wildtrack.annotations import (
    WildtrackFrameAnnotations,
    WildtrackPersonAnnotation,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.config import EvaluationConfig
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.evaluator import (
    evaluate_instant,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.models import (
    InstantAssociationEvaluation,
)
from atreides_vision.venue_validation.observer import VenueValidationInstantSnapshot
from fusion_builders import ambiguous_decision, associated_decision, not_comparable_decision
from wildtrack_evaluation_builders import (
    ground_truth_of,
    gt_frame,
    gt_person,
    membership,
    metric_obs,
    missed_decision,
    snapshot_for,
)

CONFIG = EvaluationConfig(selected_camera_ids=("C1", "C2"))


def _score(
    snapshot: VenueValidationInstantSnapshot, frame: WildtrackFrameAnnotations
) -> InstantAssociationEvaluation:
    return evaluate_instant(snapshot, ground_truth_of(frame), CONFIG, instant_index=0).association


def test_a_true_cross_camera_association_is_scored_as_found_and_end_to_end() -> None:
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    b = metric_obs("C2", 2, x=0.0, y=0.0)
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1", "C2")))
    snapshot = snapshot_for(
        0,
        observations=[a, b],
        decisions=[associated_decision(a, b)],
        memberships=[membership(1, a, b)],
    )

    result = _score(snapshot, frame)

    assert result.evaluable_pair_count == 1
    assert result.evaluable_associated_count == 1
    assert result.true_positive_merges == 1
    assert result.false_merges == 0
    assert result.end_to_end_pair_count == 1
    assert result.end_to_end_satisfied_count == 1
    assert result.association_recall_evaluable.value == 1.0
    assert result.association_recall_end_to_end.value == 1.0
    assert result.fused_identity_recall_end_to_end.value == 1.0


def test_a_wrong_merge_of_two_different_people_is_counted_as_a_false_merge() -> None:
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    b = metric_obs("C2", 2, x=0.0, y=0.0)
    # Two different WILDTRACK people that happen to sit at the same metre point,
    # each seen by only one camera -> no genuine cross-camera pair exists.
    frame = gt_frame(
        0,
        gt_person(1, x=0.0, y=0.0, cameras=("C1",)),
        gt_person(2, x=0.0, y=0.0, cameras=("C2",)),
    )
    snapshot = snapshot_for(0, observations=[a, b], decisions=[associated_decision(a, b)])

    result = _score(snapshot, frame)

    assert result.false_merges == 1
    assert result.true_positive_merges == 0
    assert result.evaluable_pair_count == 0
    assert result.association_precision.value == 0.0


def test_a_false_merge_produces_one_ordered_gt_resolved_diagnostic() -> None:
    a = metric_obs("C1", 7, x=0.0, y=0.0)
    b = metric_obs("C2", 3, x=0.0, y=0.0)
    frame = gt_frame(
        0,
        gt_person(1, x=0.0, y=0.0, cameras=("C1",)),
        gt_person(2, x=0.0, y=0.0, cameras=("C2",)),
    )
    snapshot = snapshot_for(0, observations=[a, b], decisions=[associated_decision(a, b)])

    result = _score(snapshot, frame)

    (diagnostic,) = result.false_merges_detail
    assert diagnostic.first_tracklet <= diagnostic.second_tracklet
    assert diagnostic.first_tracklet == ("C1", 0, 7)
    assert diagnostic.second_tracklet == ("C2", 0, 3)
    assert diagnostic.first_person_id == 1
    assert diagnostic.second_person_id == 2
    assert diagnostic.outcome == "associated"
    assert diagnostic.first_ground_truth_position == (0.0, 0.0)


def test_a_true_positive_merge_produces_no_diagnostic() -> None:
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    b = metric_obs("C2", 2, x=0.0, y=0.0)
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1", "C2")))
    snapshot = snapshot_for(
        0,
        observations=[a, b],
        decisions=[associated_decision(a, b)],
        memberships=[membership(1, a, b)],
    )

    result = _score(snapshot, frame)

    assert result.false_merges == 0
    assert result.false_merges_detail == ()


def test_an_unresolved_associated_pair_produces_no_diagnostic() -> None:
    # Both tracklets sit far from every annotated person -> not GT-resolvable.
    a = metric_obs("C1", 1, x=50.0, y=50.0)
    b = metric_obs("C2", 2, x=50.0, y=50.0)
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1", "C2")))
    snapshot = snapshot_for(0, observations=[a, b], decisions=[associated_decision(a, b)])

    result = _score(snapshot, frame)

    assert result.unresolved_for_scoring == 1
    assert result.false_merges == 0
    assert result.false_merges_detail == ()


def test_a_refused_real_pair_is_a_missed_association_with_its_reason_recorded() -> None:
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    b = metric_obs("C2", 2, x=0.0, y=0.0)
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1", "C2")))
    snapshot = snapshot_for(
        0,
        observations=[a, b],
        decisions=[missed_decision(a, b, reason=DecisionReason.GEOMETRY_EXCLUDED)],
    )

    result = _score(snapshot, frame)

    assert result.evaluable_pair_count == 1
    assert result.evaluable_associated_count == 0
    assert result.missed_association_count == 1
    assert result.decision_reason_counts == {"geometry_excluded": 1}
    assert result.association_recall_evaluable.value == 0.0


def test_support_not_sustained_is_tallied_on_its_own_axis() -> None:
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
    )

    result = _score(snapshot, frame)

    assert result.support_not_sustained_count == 1
    assert result.missed_association_count == 1
    # support_not_sustained is an AMBIGUOUS reason, never an UNASSOCIATED one.
    assert result.ambiguous_pair_count == 1


def test_missed_association_rate_complements_association_recall_evaluable() -> None:
    pairs: list[TrackObservation] = []
    persons: list[WildtrackPersonAnnotation] = []
    decisions: list[PairDecision] = []
    for index in range(4):
        x = float(index * 10)
        first = metric_obs("C1", 2 * index + 1, x=x, y=0.0)
        second = metric_obs("C2", 2 * index + 2, x=x, y=0.0)
        pairs.extend((first, second))
        persons.append(gt_person(index + 1, x=x, y=0.0, cameras=("C1", "C2")))
    decisions.append(associated_decision(pairs[0], pairs[1]))
    decisions.append(missed_decision(pairs[2], pairs[3], reason=DecisionReason.GEOMETRY_EXCLUDED))
    decisions.append(ambiguous_decision(pairs[4], pairs[5]))
    decisions.append(
        missed_decision(
            pairs[6],
            pairs[7],
            reason=DecisionReason.SUPPORT_NOT_SUSTAINED,
            outcome=AssociationOutcome.AMBIGUOUS,
        )
    )
    frame = gt_frame(0, *persons)
    snapshot = snapshot_for(0, observations=pairs, decisions=decisions)

    result = _score(snapshot, frame)

    assert result.evaluable_pair_count == 4
    assert result.evaluable_associated_count == 1
    assert result.missed_association_count == 3
    assert result.association_recall_evaluable.value == 0.25
    assert result.missed_association_rate.value == 0.75


def test_an_ambiguous_pair_is_left_unmerged_and_raises_the_ambiguity_rate() -> None:
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    b = metric_obs("C2", 2, x=0.0, y=0.0)
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1", "C2")))
    snapshot = snapshot_for(0, observations=[a, b], decisions=[ambiguous_decision(a, b)])

    result = _score(snapshot, frame)

    assert result.ambiguous_pair_count == 1
    assert result.evaluable_associated_count == 0
    assert result.ambiguity_rate.value == 1.0


def test_a_not_comparable_decision_still_counts_and_records_its_gate() -> None:
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    b = metric_obs("C2", 2, x=0.0, y=0.0)
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1", "C2")))
    snapshot = snapshot_for(0, observations=[a, b], decisions=[not_comparable_decision(a, b)])

    result = _score(snapshot, frame)

    assert result.evaluable_pair_count == 1
    assert result.gate_reason_counts == {"same_camera": 1}


def test_a_co_present_pair_with_no_decision_is_a_totality_violation() -> None:
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    b = metric_obs("C2", 2, x=0.0, y=0.0)
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1", "C2")))
    snapshot = snapshot_for(0, observations=[a, b])

    with pytest.raises(ValidationError):
        _score(snapshot, frame)


def test_a_pair_without_a_runtime_match_on_both_cameras_is_not_evaluable() -> None:
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1", "C2")))
    snapshot = snapshot_for(0, observations=[a])

    result = _score(snapshot, frame)

    assert result.not_evaluable_pair_count == 1
    assert result.evaluable_pair_count == 0
    assert result.end_to_end_pair_count == 1


def test_a_single_camera_person_creates_no_cross_camera_pair() -> None:
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1",)))
    snapshot = snapshot_for(0, observations=[a])

    result = _score(snapshot, frame)

    assert result.end_to_end_pair_count == 0
    assert result.evaluable_pair_count == 0


def test_association_scoring_is_independent_of_selected_camera_order() -> None:
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    b = metric_obs("C2", 2, x=0.0, y=0.0)
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1", "C2")))
    snapshot = snapshot_for(
        0,
        observations=[a, b],
        decisions=[associated_decision(a, b)],
        memberships=[membership(1, a, b)],
    )

    forward = evaluate_instant(
        snapshot, ground_truth_of(frame), CONFIG, instant_index=0
    ).association
    reversed_ = evaluate_instant(
        snapshot,
        ground_truth_of(frame),
        EvaluationConfig(selected_camera_ids=("C2", "C1")),
        instant_index=0,
    ).association

    assert forward.to_record() == reversed_.to_record()

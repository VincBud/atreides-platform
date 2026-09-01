"""Model-level checks for the offline WILDTRACK evaluator (V0.4.9b-D1).

These pin the null-safe ratio, the per-camera runtime/ground-truth matching
(:func:`build_camera_matches`), and the small aggregate helpers. Every expected
value is computed by hand from the synthetic geometry, never snapshotted.
"""

from __future__ import annotations

from atreides_vision.venue_validation.datasets.wildtrack.evaluation.models import (
    CameraCoverage,
    InstantAssociationEvaluation,
    Ratio,
    build_camera_matches,
    mean,
    rmse,
)
from wildtrack_evaluation_builders import (
    gt_frame,
    gt_person,
    metric_obs,
    non_metric_obs,
    snapshot_for,
    tracklet_key,
)

_RADIUS = 0.75


def test_ratio_reports_an_undefined_value_as_none_not_zero() -> None:
    undefined = Ratio.of(0, 0)
    assert undefined.value is None
    assert undefined.to_record() == {"numerator": 0, "denominator": 0, "value": None}


def test_ratio_value_and_record_rounding() -> None:
    ratio = Ratio.of(1, 3)
    assert ratio.value == 1 / 3
    assert ratio.to_record()["value"] == round(1 / 3, 6)


def test_mean_and_rmse_are_none_only_when_empty() -> None:
    assert mean([]) is None
    assert rmse([]) is None
    assert mean([-1, 1, 3]) == 1.0
    assert rmse([3, 4]) == 12.5**0.5


def test_build_camera_matches_is_camera_local_and_counts_non_metric_separately() -> None:
    obs_c1 = metric_obs("C1", 1, x=0.0, y=0.0)
    obs_c1_img = non_metric_obs("C1", 2)
    obs_c2 = metric_obs("C2", 9, x=50.0, y=50.0)
    snapshot = snapshot_for(0, observations=[obs_c1, obs_c1_img, obs_c2])
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1",)))

    matches = build_camera_matches(
        snapshot, frame, selected_camera_ids=("C1", "C2"), radius_metres=_RADIUS
    )

    c1 = matches["C1"]
    assert c1.metric_observation_count == 1
    assert c1.non_metric_observation_count == 1
    assert c1.runtime_observation_count == 2
    assert c1.matched_count == 1
    assert c1.person_for_observation(tracklet_key(obs_c1)) == 1
    assert c1.observation_for_person(1) == tracklet_key(obs_c1)
    # C2 saw nobody in ground truth and its observation is far away.
    assert matches["C2"].matched_count == 0
    assert matches["C2"].person_for_observation(tracklet_key(obs_c2)) is None


def test_build_camera_matches_keeps_persons_in_ascending_id_order() -> None:
    near = metric_obs("C1", 7, x=0.0, y=0.0)
    far = metric_obs("C1", 8, x=10.0, y=0.0)
    snapshot = snapshot_for(0, observations=[near, far])
    frame = gt_frame(
        0,
        gt_person(5, x=10.0, y=0.0, cameras=("C1",)),
        gt_person(2, x=0.0, y=0.0, cameras=("C1",)),
    )

    match = build_camera_matches(
        snapshot, frame, selected_camera_ids=("C1",), radius_metres=_RADIUS
    )["C1"]

    assert match.person_ids == (2, 5)
    assert match.person_for_observation(tracklet_key(near)) == 2
    assert match.person_for_observation(tracklet_key(far)) == 5


def test_the_match_radius_is_inclusive_at_exactly_the_bound() -> None:
    obs = metric_obs("C1", 1, x=0.0, y=0.0)
    snapshot = snapshot_for(0, observations=[obs])
    frame = gt_frame(0, gt_person(1, x=_RADIUS, y=0.0, cameras=("C1",)))

    match = build_camera_matches(
        snapshot, frame, selected_camera_ids=("C1",), radius_metres=_RADIUS
    )["C1"]

    assert match.matched_count == 1


def test_camera_coverage_ratios_are_null_safe() -> None:
    coverage = CameraCoverage(
        camera_id="C1",
        runtime_observation_count=0,
        metric_observation_count=0,
        non_metric_observation_count=0,
        gt_person_count=0,
        matched_count=0,
        total_distance_micrometres=0,
    )
    assert coverage.runtime_coverage.value is None
    assert coverage.ground_truth_coverage.value is None


def _association(
    *,
    evaluable_pair_count: int,
    evaluable_associated_count: int,
    end_to_end_pair_count: int,
    end_to_end_satisfied_count: int,
) -> InstantAssociationEvaluation:
    return InstantAssociationEvaluation(
        true_positive_merges=evaluable_associated_count,
        false_merges=0,
        unresolved_for_scoring=0,
        evaluable_pair_count=evaluable_pair_count,
        evaluable_associated_count=evaluable_associated_count,
        missed_association_count=evaluable_pair_count - evaluable_associated_count,
        ambiguous_pair_count=0,
        support_not_sustained_count=0,
        not_evaluable_pair_count=end_to_end_pair_count - evaluable_pair_count,
        end_to_end_pair_count=end_to_end_pair_count,
        end_to_end_satisfied_count=end_to_end_satisfied_count,
        gate_reason_counts={},
        decision_reason_counts={},
    )


def test_association_and_fused_identity_recall_are_separate_layers() -> None:
    # 10 GT-positive pairs; 5 evaluable; 3 PairDecision==ASSOCIATED successes;
    # 2 of those end up under one VenueSubjectId after fusion.
    result = _association(
        evaluable_pair_count=5,
        evaluable_associated_count=3,
        end_to_end_pair_count=10,
        end_to_end_satisfied_count=2,
    )

    assert result.association_recall_evaluable.value == 3 / 5
    assert result.association_recall_end_to_end.value == 3 / 10
    assert result.fused_identity_recall_end_to_end.value == 2 / 10


def test_changing_subject_ownership_moves_only_fused_identity_recall() -> None:
    before = _association(
        evaluable_pair_count=5,
        evaluable_associated_count=3,
        end_to_end_pair_count=10,
        end_to_end_satisfied_count=2,
    )
    # Same PairDecision outcomes; one more pair now shares a VenueSubjectId.
    after = _association(
        evaluable_pair_count=5,
        evaluable_associated_count=3,
        end_to_end_pair_count=10,
        end_to_end_satisfied_count=3,
    )

    assert after.association_recall_evaluable.value == before.association_recall_evaluable.value
    assert after.association_recall_end_to_end.value == before.association_recall_end_to_end.value
    assert after.fused_identity_recall_end_to_end.value == 3 / 10
    assert (
        after.fused_identity_recall_end_to_end.value
        != before.fused_identity_recall_end_to_end.value
    )

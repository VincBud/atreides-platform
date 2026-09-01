"""Per-instant and aggregate occupancy scoring for the WILDTRACK evaluator.

The duplicate-suppression direction is explicit: two camera-local tracks of one
person collapse to a fused occupancy of one, and the suppressed count is
surfaced rather than hidden. Ground truth is restricted to the scored cameras.
"""

from __future__ import annotations

from atreides_vision.venue_validation.datasets.wildtrack.annotations import (
    WildtrackFrameAnnotations,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.config import EvaluationConfig
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.evaluator import (
    aggregate_evaluation,
    evaluate_instant,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.models import (
    InstantOccupancyEvaluation,
)
from atreides_vision.venue_validation.observer import VenueValidationInstantSnapshot
from wildtrack_evaluation_builders import (
    ground_truth_of,
    gt_frame,
    gt_person,
    metric_obs,
    occupancy_of,
    snapshot_for,
)

CONFIG = EvaluationConfig(selected_camera_ids=("C1", "C2"))


def _occupancy(
    snapshot: VenueValidationInstantSnapshot, frame: WildtrackFrameAnnotations
) -> InstantOccupancyEvaluation:
    return evaluate_instant(snapshot, ground_truth_of(frame), CONFIG, instant_index=0).occupancy


def test_ground_truth_is_restricted_to_the_scored_cameras() -> None:
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C3",)))
    snapshot = snapshot_for(0)

    result = _occupancy(snapshot, frame)

    assert result.ground_truth_observable_count == 0
    assert result.raw_active_track_count == 0
    assert result.fused_occupancy_count == 0


def test_two_camera_tracks_of_one_person_collapse_to_a_single_fused_count() -> None:
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1", "C2")))
    occupancy = occupancy_of(
        0,
        raw_active_track_count=2,
        contributions=[(1, [metric_obs("C1", 1), metric_obs("C2", 2)])],
    )
    snapshot = snapshot_for(0, occupancy=occupancy)

    result = _occupancy(snapshot, frame)

    assert result.ground_truth_observable_count == 1
    assert result.raw_active_track_count == 2
    assert result.fused_occupancy_count == 1
    assert result.suppressed_duplicate_count == 1
    assert result.raw_error == 1
    assert result.fused_error == 0


def test_occupancy_summary_aggregates_error_and_suppression_analytically() -> None:
    person_frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1", "C2")))
    empty_frame = gt_frame(5)
    ground_truth = ground_truth_of(person_frame, empty_frame)

    first = snapshot_for(
        0,
        occupancy=occupancy_of(
            0,
            raw_active_track_count=2,
            contributions=[(1, [metric_obs("C1", 1), metric_obs("C2", 2)])],
        ),
    )
    second = snapshot_for(1)

    instants = [
        evaluate_instant(first, ground_truth, CONFIG, instant_index=0),
        evaluate_instant(second, ground_truth, CONFIG, instant_index=1),
    ]
    summary = aggregate_evaluation(instants, CONFIG).occupancy

    assert summary.instant_count == 2
    assert summary.raw_mae == 0.5
    assert summary.fused_mae == 0.0
    assert summary.raw_bias == 0.5
    assert summary.fused_bias == 0.0
    assert summary.raw_rmse == 0.5**0.5
    assert summary.fused_rmse == 0.0
    assert summary.exact_fused_count == 2
    assert summary.fused_over_count == 0
    assert summary.fused_under_count == 0
    assert summary.max_abs_raw_error == 1
    assert summary.max_abs_fused_error == 0
    assert summary.improved_count == 1
    assert summary.unchanged_count == 1
    assert summary.worsened_count == 0
    assert summary.total_suppressed_duplicates == 1
    assert summary.mean_suppressed_duplicates == 0.5
    assert summary.exact_fused_fraction.value == 1.0
    assert summary.fused_minus_raw_mae == -0.5

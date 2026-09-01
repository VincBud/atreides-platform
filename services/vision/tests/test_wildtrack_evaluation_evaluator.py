"""Streaming lifecycle and determinism of :class:`WildtrackEvaluator` (V0.4.9b-D1).

The evaluator is an ordered observer: it aligns the ``k``-th venue instant to
``ground_truth.frame(k)``, refuses a misaligned or miscounted stream, and
produces a result that does not depend on the order the scored cameras are
listed in.
"""

from __future__ import annotations

import pytest

from atreides_vision.errors import ValidationError
from atreides_vision.venue_validation.datasets.wildtrack.annotations import (
    WildtrackFrameAnnotations,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.config import EvaluationConfig
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.evaluator import (
    WildtrackEvaluator,
    aggregate_evaluation,
)
from atreides_vision.venue_validation.observer import VenueValidationInstantSnapshot
from fusion_builders import associated_decision
from wildtrack_evaluation_builders import (
    ground_truth_of,
    gt_frame,
    gt_person,
    membership,
    metric_obs,
    snapshot_for,
)

CONFIG = EvaluationConfig(selected_camera_ids=("C1", "C2"))


def _person_instant() -> VenueValidationInstantSnapshot:
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    b = metric_obs("C2", 2, x=0.0, y=0.0)
    return snapshot_for(
        0,
        observations=[a, b],
        decisions=[associated_decision(a, b)],
        memberships=[membership(1, a, b)],
    )


def _frame() -> WildtrackFrameAnnotations:
    return gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1", "C2")))


def test_a_single_aligned_instant_finalizes_into_a_summary() -> None:
    evaluator = WildtrackEvaluator(CONFIG, ground_truth_of(_frame()))
    evaluator(_person_instant())

    summary = evaluator.finalize()

    assert summary.instant_count == 1
    assert summary.association.true_positive_merges == 1


def test_too_many_instants_is_refused() -> None:
    evaluator = WildtrackEvaluator(CONFIG, ground_truth_of(_frame()))
    evaluator(_person_instant())

    with pytest.raises(ValidationError, match="too many"):
        evaluator(snapshot_for(1))


def test_too_few_instants_is_refused_at_finalize() -> None:
    evaluator = WildtrackEvaluator(CONFIG, ground_truth_of(_frame(), gt_frame(5)))
    evaluator(_person_instant())

    with pytest.raises(ValidationError, match="too few"):
        evaluator.finalize()


def test_a_misaligned_venue_clock_is_refused() -> None:
    evaluator = WildtrackEvaluator(CONFIG, ground_truth_of(_frame()))

    with pytest.raises(ValidationError, match="misaligned"):
        evaluator(snapshot_for(2))


def test_finalize_is_single_use() -> None:
    evaluator = WildtrackEvaluator(CONFIG, ground_truth_of(_frame()))
    evaluator(_person_instant())
    evaluator.finalize()

    with pytest.raises(ValidationError):
        evaluator.finalize()


def test_an_instant_after_finalize_is_refused() -> None:
    evaluator = WildtrackEvaluator(CONFIG, ground_truth_of(_frame()))
    evaluator(_person_instant())
    evaluator.finalize()

    with pytest.raises(ValidationError, match="the run is over"):
        evaluator(snapshot_for(1))


def test_aggregate_evaluation_needs_at_least_one_instant() -> None:
    with pytest.raises(ValidationError):
        aggregate_evaluation([], CONFIG)


def test_scoring_is_independent_of_selected_camera_order() -> None:
    forward = WildtrackEvaluator(
        EvaluationConfig(selected_camera_ids=("C1", "C2")), ground_truth_of(_frame())
    )
    reverse = WildtrackEvaluator(
        EvaluationConfig(selected_camera_ids=("C2", "C1")), ground_truth_of(_frame())
    )
    forward(_person_instant())
    reverse(_person_instant())

    a = forward.finalize()
    b = reverse.finalize()

    assert a.association.to_record() == b.association.to_record()
    assert a.fusion.to_record() == b.fusion.to_record()
    assert a.occupancy.to_record() == b.occupancy.to_record()
    # Only the per-camera coverage list is order-sensitive, by construction.
    assert {c.camera_id for c in a.camera_coverage} == {c.camera_id for c in b.camera_coverage}

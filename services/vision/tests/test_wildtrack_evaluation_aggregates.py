"""Aggregate local-coverage and match-distance statistics (V0.4.9b-D1).

These are pure, closed-form measurements over camera-local GT<->runtime matches:
:class:`LocalCoverage` sums the per-camera counts, :class:`MatchDistanceStats`
summarises every successful match distance. Both report ``None`` for an
undefined result rather than a misleading zero.
"""

from __future__ import annotations

from atreides_vision.venue_validation.datasets.wildtrack.evaluation.models import (
    CameraCoverage,
    LocalCoverage,
    MatchDistanceStats,
)

_UM = 1_000_000


def _camera(camera_id: str, *, metric: int, gt: int, matched: int) -> CameraCoverage:
    return CameraCoverage(
        camera_id=camera_id,
        runtime_observation_count=metric,
        metric_observation_count=metric,
        non_metric_observation_count=0,
        gt_person_count=gt,
        matched_count=matched,
        total_distance_micrometres=0,
    )


def test_local_coverage_sums_camera_counts_and_derives_the_unmatched_residuals() -> None:
    coverage = LocalCoverage.from_cameras(
        [
            _camera("C1", metric=4, gt=5, matched=3),
            _camera("C2", metric=2, gt=3, matched=2),
        ]
    )

    assert coverage.gt_visible_observations == 8  # 5 + 3
    assert coverage.runtime_observations == 6  # 4 + 2
    assert coverage.matched_observations == 5  # 3 + 2
    assert coverage.unmatched_gt_observations == 3  # 8 - 5
    assert coverage.unmatched_runtime_observations == 1  # 6 - 5
    assert coverage.local_observation_coverage.value == 5 / 8


def test_local_observation_coverage_is_none_when_no_gt_is_visible() -> None:
    coverage = LocalCoverage.from_cameras([_camera("C1", metric=2, gt=0, matched=0)])

    assert coverage.local_observation_coverage.value is None
    serialised = coverage.to_record()["local_observation_coverage"]
    assert isinstance(serialised, dict)
    assert serialised["value"] is None


def test_match_distance_stats_are_all_none_when_nothing_matched() -> None:
    stats = MatchDistanceStats.from_micrometres(())

    assert stats.mean_m is None
    assert stats.median_m is None
    assert stats.p95_m is None
    assert stats.max_m is None


def test_match_distance_stats_for_a_single_match_collapse_to_that_distance() -> None:
    stats = MatchDistanceStats.from_micrometres([_UM // 4])

    assert stats.mean_m == 0.25
    assert stats.median_m == 0.25
    assert stats.p95_m == 0.25
    assert stats.max_m == 0.25


def test_median_is_the_mean_of_the_two_central_values_for_an_even_sample() -> None:
    stats = MatchDistanceStats.from_micrometres([4 * _UM, 1 * _UM, 3 * _UM, 2 * _UM])

    assert stats.median_m == 2.5  # mean(2, 3) after ascending sort
    assert stats.mean_m == 2.5
    assert stats.max_m == 4.0


def test_median_is_the_central_value_for_an_odd_sample() -> None:
    stats = MatchDistanceStats.from_micrometres([5 * _UM, 1 * _UM, 3 * _UM])

    assert stats.median_m == 3.0


def test_p95_uses_nearest_rank_ceil_on_the_ascending_sorted_sample() -> None:
    # n = 10, rank = ceil(0.95 * 10) = 10 -> the largest value.
    stats = MatchDistanceStats.from_micrometres([i * _UM for i in range(10, 0, -1)])
    assert stats.p95_m == 10.0

    # n = 20, rank = ceil(0.95 * 20) = 19 -> the 19th smallest (value 19).
    stats_20 = MatchDistanceStats.from_micrometres([i * _UM for i in range(1, 21)])
    assert stats_20.p95_m == 19.0


def test_match_distance_stats_do_not_depend_on_input_order() -> None:
    ascending = MatchDistanceStats.from_micrometres([1 * _UM, 2 * _UM, 7 * _UM, 9 * _UM])
    shuffled = MatchDistanceStats.from_micrometres([9 * _UM, 1 * _UM, 7 * _UM, 2 * _UM])

    assert ascending.to_record() == shuffled.to_record()

"""Deterministic matcher for the offline WILDTRACK evaluator (V0.4.9b-D1).

The vectors here are worked by hand in integer micrometres so the three
ordering guarantees — maximum cardinality first, minimum total distance second,
canonical lower-row/lower-column tie-break third — are each pinned to an exact
expected assignment rather than a snapshot of current output.
"""

from __future__ import annotations

from atreides_vision.venue_validation.datasets.wildtrack.evaluation.matching import (
    match_by_distance,
    match_points,
)

_RADIUS_UM = 750_000


def test_cardinality_beats_a_cheaper_smaller_matching() -> None:
    # Greedy by row would take the cheap (0, 0) edge and strand row 1.
    distances = [
        [100_000, 700_000],
        [100_000, 750_001],
    ]

    match = match_by_distance(distances, radius_micrometres=_RADIUS_UM)

    assert match.pairs == ((0, 1), (1, 0))
    assert match.total_distance_micrometres == 800_000


def test_equal_cost_swap_resolves_to_the_canonical_diagonal() -> None:
    distances = [
        [500_000, 500_000],
        [500_000, 500_000],
    ]

    match = match_by_distance(distances, radius_micrometres=_RADIUS_UM)

    assert match.pairs == ((0, 0), (1, 1))
    assert match.total_distance_micrometres == 1_000_000


def test_the_radius_is_inclusive_at_exactly_the_bound() -> None:
    on_bound = match_by_distance([[750_000]], radius_micrometres=_RADIUS_UM)
    just_over = match_by_distance([[750_001]], radius_micrometres=_RADIUS_UM)

    assert on_bound.pairs == ((0, 0),)
    assert on_bound.total_distance_micrometres == 750_000
    assert just_over.pairs == ()
    assert just_over.total_distance_micrometres == 0


def test_assignment_is_one_to_one_under_contention() -> None:
    # Both rows want column 0; column 1 is out of range for everyone.
    distances = [
        [10_000, 900_000],
        [20_000, 900_000],
    ]

    match = match_by_distance(distances, radius_micrometres=_RADIUS_UM)

    assert match.pairs == ((0, 0),)
    assert len(match.matched_columns) == 1


def test_empty_and_one_sided_inputs_match_nothing() -> None:
    assert match_points([], [], radius_metres=0.75).pairs == ()
    assert match_points([], [(0.0, 0.0)], radius_metres=0.75).pairs == ()
    assert match_points([(0.0, 0.0)], [], radius_metres=0.75).pairs == ()


def test_match_points_rounds_to_integer_micrometres() -> None:
    match = match_points([(0.0, 0.0)], [(0.3, 0.4)], radius_metres=0.75)

    assert match.pairs == ((0, 0),)
    assert match.total_distance_micrometres == 500_000


def test_result_is_identical_across_row_ordering() -> None:
    forward = match_by_distance(
        [[100_000, 700_000], [100_000, 750_001]], radius_micrometres=_RADIUS_UM
    )
    # Same graph, rows swapped: the canonical result must be the mirror image.
    swapped = match_by_distance(
        [[100_000, 750_001], [100_000, 700_000]], radius_micrometres=_RADIUS_UM
    )

    assert forward.pairs == ((0, 1), (1, 0))
    assert swapped.pairs == ((0, 0), (1, 1))
    assert forward.total_distance_micrometres == swapped.total_distance_micrometres

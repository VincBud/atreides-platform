"""Spatial aggregate formulas, checked against layouts with known answers.

Every case uses a geometry whose centroid, spread, separation, density or
coherence can be worked out by hand, so a wrong formula moves a number a reader
can verify independently.
"""

from __future__ import annotations

import math

import pytest

from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.behavior.models import SubjectMotion
from atreides_vision.crowd.geometry import (
    centroid,
    local_density,
    max_pair_distance,
    mean_pair_distance,
    motion_heading_coherence,
    neighbours_within,
    rms_radius,
    separation,
)
from atreides_vision.errors import CrowdError

METRIC = MotionSpace.GROUND_PLANE_METRES
NORMALIZED = MotionSpace.NORMALIZED_IMAGE


def point(x: float, y: float, space: MotionSpace = METRIC) -> MotionPoint:
    return MotionPoint(x, y, space)


def motion(dx: float, dy: float, space: MotionSpace = METRIC) -> SubjectMotion:
    return SubjectMotion(dx=dx, dy=dy, elapsed_seconds=1.0, space=space)


#: A 2x2 square at the origin. Centroid (1, 1); every corner is sqrt(2) away.
SQUARE = [point(0.0, 0.0), point(2.0, 0.0), point(2.0, 2.0), point(0.0, 2.0)]


class TestSeparation:
    def test_three_four_five(self) -> None:
        assert separation(point(0.0, 0.0), point(3.0, 4.0)) == pytest.approx(5.0)

    def test_identical_positions_are_zero_apart(self) -> None:
        assert separation(point(2.0, 2.0), point(2.0, 2.0)) == 0.0

    def test_is_symmetric(self) -> None:
        first, second = point(1.0, 2.0), point(4.0, 6.0)
        assert separation(first, second) == separation(second, first)

    def test_rejects_mixed_spaces(self) -> None:
        with pytest.raises(CrowdError, match="Cannot measure separation"):
            separation(point(0.0, 0.0, METRIC), point(1.0, 1.0, NORMALIZED))


class TestCentroid:
    def test_square(self) -> None:
        result = centroid(SQUARE)
        assert (result.x, result.y) == pytest.approx((1.0, 1.0))

    def test_simple_triangle(self) -> None:
        """Centroid of (0,0), (6,0), (0,6) is the mean of the vertices."""
        result = centroid([point(0.0, 0.0), point(6.0, 0.0), point(0.0, 6.0)])
        assert (result.x, result.y) == pytest.approx((2.0, 2.0))

    def test_single_position_is_its_own_centroid(self) -> None:
        result = centroid([point(3.0, 7.0)])
        assert (result.x, result.y) == pytest.approx((3.0, 7.0))

    def test_identical_positions(self) -> None:
        result = centroid([point(2.0, 2.0)] * 4)
        assert (result.x, result.y) == pytest.approx((2.0, 2.0))

    def test_carries_the_space(self) -> None:
        assert centroid([point(0.0, 0.0, NORMALIZED)]).space is NORMALIZED

    def test_large_coordinates(self) -> None:
        result = centroid([point(1_000.0, 2_000.0), point(3_000.0, 4_000.0)])
        assert (result.x, result.y) == pytest.approx((2_000.0, 3_000.0))

    def test_negative_coordinates(self) -> None:
        """Venue origins are wherever the operator put them."""
        result = centroid([point(-4.0, -2.0), point(-2.0, -4.0)])
        assert (result.x, result.y) == pytest.approx((-3.0, -3.0))

    def test_rejects_an_empty_set(self) -> None:
        with pytest.raises(CrowdError, match="at least one position"):
            centroid([])

    def test_rejects_mixed_spaces(self) -> None:
        with pytest.raises(CrowdError, match="different coordinate spaces"):
            centroid([point(0.0, 0.0, METRIC), point(1.0, 1.0, NORMALIZED)])


class TestRmsRadius:
    def test_square_is_the_half_diagonal(self) -> None:
        """Every corner of a 2x2 square is sqrt(2) from the centre."""
        assert rms_radius(SQUARE) == pytest.approx(math.sqrt(2.0))

    def test_three_in_a_row(self) -> None:
        """Offsets -1, 0, +1 give sqrt((1+0+1)/3)."""
        row = [point(0.0, 0.0), point(1.0, 0.0), point(2.0, 0.0)]
        assert rms_radius(row) == pytest.approx(math.sqrt(2.0 / 3.0))

    def test_coincident_positions_have_no_spread(self) -> None:
        assert rms_radius([point(1.0, 1.0)] * 3) == pytest.approx(0.0)

    def test_a_single_position_has_no_spread(self) -> None:
        assert rms_radius([point(5.0, 5.0)]) == pytest.approx(0.0)

    def test_accepts_a_precomputed_centre(self) -> None:
        assert rms_radius(SQUARE, centroid(SQUARE)) == pytest.approx(rms_radius(SQUARE))

    def test_scales_with_the_layout(self) -> None:
        doubled = [point(p.x * 2, p.y * 2) for p in SQUARE]
        assert rms_radius(doubled) == pytest.approx(2 * rms_radius(SQUARE))

    def test_rejects_an_empty_set(self) -> None:
        with pytest.raises(CrowdError, match="at least one position"):
            rms_radius([])


class TestPairDistances:
    def test_square_mean_and_max(self) -> None:
        """Four sides of 2, two diagonals of 2*sqrt(2), over six pairs."""
        expected_mean = (4 * 2.0 + 2 * 2.0 * math.sqrt(2.0)) / 6
        assert mean_pair_distance(SQUARE) == pytest.approx(expected_mean)
        assert max_pair_distance(SQUARE) == pytest.approx(2.0 * math.sqrt(2.0))

    def test_three_in_a_row(self) -> None:
        """Pairs of 1, 1 and 2 give a mean of 4/3."""
        row = [point(0.0, 0.0), point(1.0, 0.0), point(2.0, 0.0)]
        assert mean_pair_distance(row) == pytest.approx(4.0 / 3.0)
        assert max_pair_distance(row) == pytest.approx(2.0)

    def test_self_distances_are_excluded(self) -> None:
        """Two coincident points have one pair at zero, not three."""
        assert mean_pair_distance([point(1.0, 1.0), point(1.0, 1.0)]) == pytest.approx(0.0)

    def test_a_single_position_has_no_pairs(self) -> None:
        assert mean_pair_distance([point(0.0, 0.0)]) is None
        assert max_pair_distance([point(0.0, 0.0)]) is None

    def test_an_empty_set_has_no_pairs(self) -> None:
        assert mean_pair_distance([]) is None
        assert max_pair_distance([]) is None

    def test_max_is_at_least_the_mean(self) -> None:
        mean = mean_pair_distance(SQUARE)
        largest = max_pair_distance(SQUARE)
        assert mean is not None and largest is not None
        assert largest >= mean


class TestNeighbours:
    def test_counts_within_the_radius(self) -> None:
        others = [point(1.0, 0.0), point(2.0, 0.0), point(10.0, 0.0)]
        assert neighbours_within(point(0.0, 0.0), others, 2.0) == 2

    def test_the_boundary_counts_as_inside(self) -> None:
        assert neighbours_within(point(0.0, 0.0), [point(2.0, 0.0)], 2.0) == 1

    def test_just_beyond_the_boundary_does_not(self) -> None:
        assert neighbours_within(point(0.0, 0.0), [point(2.0000001, 0.0)], 2.0) == 0

    def test_an_isolated_subject_has_none(self) -> None:
        assert neighbours_within(point(0.0, 0.0), [point(50.0, 50.0)], 2.0) == 0

    def test_rejects_a_non_positive_radius(self) -> None:
        with pytest.raises(CrowdError, match="must be positive"):
            neighbours_within(point(0.0, 0.0), [], 0.0)


class TestLocalDensity:
    def test_the_subject_itself_is_counted(self) -> None:
        """An alone subject has a real density, not a zero."""
        assert local_density(0, 2.0, METRIC) == pytest.approx(1.0 / (math.pi * 4.0))

    def test_two_neighbours(self) -> None:
        assert local_density(2, 2.0, METRIC) == pytest.approx(3.0 / (math.pi * 4.0))

    @pytest.mark.parametrize("count", [0, 1, 5, 20])
    def test_matches_the_stated_formula(self, count: int) -> None:
        radius = 1.5
        assert local_density(count, radius, METRIC) == pytest.approx(
            (1 + count) / (math.pi * radius * radius)
        )

    def test_a_larger_radius_dilutes(self) -> None:
        assert local_density(3, 4.0, METRIC) < local_density(3, 2.0, METRIC)

    def test_rejects_a_non_positive_radius(self) -> None:
        with pytest.raises(CrowdError, match="must be positive"):
            local_density(1, 0.0, METRIC)

    def test_rejects_a_negative_count(self) -> None:
        with pytest.raises(CrowdError, match="non-negative"):
            local_density(-1, 2.0, METRIC)

    def test_a_dense_lattice(self) -> None:
        """A 5x5 unit lattice: the centre point sees its 4-neighbourhood at r=1."""
        lattice = [point(float(x), float(y)) for x in range(5) for y in range(5)]
        middle = point(2.0, 2.0)
        others = [p for p in lattice if (p.x, p.y) != (2.0, 2.0)]
        count = neighbours_within(middle, others, 1.0)
        assert count == 4
        assert local_density(count, 1.0, METRIC) == pytest.approx(5.0 / math.pi)


class TestMotionHeadingCoherence:
    def test_identical_headings_are_fully_coherent(self) -> None:
        assert motion_heading_coherence([motion(1.0, 0.0), motion(2.0, 0.0)], 0.1) == (
            pytest.approx(1.0)
        )

    def test_magnitude_does_not_matter_only_direction(self) -> None:
        """Unit vectors are summed, so a fast walker does not outvote a slow one."""
        assert motion_heading_coherence(
            [motion(10.0, 0.0), motion(0.5, 0.0)], 0.1
        ) == pytest.approx(1.0)

    def test_opposing_headings_cancel(self) -> None:
        assert motion_heading_coherence([motion(1.0, 0.0), motion(-1.0, 0.0)], 0.1) == (
            pytest.approx(0.0)
        )

    def test_perpendicular_headings(self) -> None:
        """Two unit vectors at 90 degrees sum to sqrt(2), over 2 members."""
        assert motion_heading_coherence([motion(1.0, 0.0), motion(0.0, 1.0)], 0.1) == (
            pytest.approx(math.sqrt(2.0) / 2.0)
        )

    def test_evenly_spread_headings_cancel(self) -> None:
        """Four members at 90-degree spacing sum to zero."""
        spread = [motion(1.0, 0.0), motion(0.0, 1.0), motion(-1.0, 0.0), motion(0.0, -1.0)]
        assert motion_heading_coherence(spread, 0.1) == pytest.approx(0.0, abs=1e-12)

    def test_it_wraps_across_the_pi_boundary(self) -> None:
        """+170 and -170 degrees are 20 degrees apart, so nearly coherent."""
        first = motion(math.cos(math.radians(170)), math.sin(math.radians(170)))
        second = motion(math.cos(math.radians(-170)), math.sin(math.radians(-170)))
        value = motion_heading_coherence([first, second], 0.1)
        assert value is not None
        assert value == pytest.approx(math.cos(math.radians(10)))

    def test_it_stays_within_zero_and_one(self) -> None:
        for degrees in range(0, 360, 15):
            angle = math.radians(degrees)
            value = motion_heading_coherence(
                [motion(1.0, 0.0), motion(math.cos(angle), math.sin(angle))], 0.1
            )
            assert value is not None
            assert 0.0 <= value <= 1.0

    def test_stationary_members_are_excluded(self) -> None:
        """A still person has no direction, so cannot disagree with anyone."""
        assert motion_heading_coherence(
            [motion(1.0, 0.0), motion(2.0, 0.0), motion(0.0, 0.0)], 0.1
        ) == pytest.approx(1.0)

    def test_one_mover_omits_the_signal(self) -> None:
        """Not zero: one heading has nothing to be coherent with."""
        assert motion_heading_coherence([motion(1.0, 0.0), motion(0.0, 0.0)], 0.1) is None

    def test_no_movers_omits_the_signal(self) -> None:
        assert motion_heading_coherence([motion(0.0, 0.0), motion(0.0, 0.0)], 0.1) is None

    def test_an_empty_set_omits_the_signal(self) -> None:
        assert motion_heading_coherence([], 0.1) is None

    def test_movement_below_the_floor_is_excluded(self) -> None:
        assert motion_heading_coherence([motion(0.05, 0.0), motion(1.0, 0.0)], 0.1) is None

    def test_rejects_mixed_spaces(self) -> None:
        with pytest.raises(CrowdError, match="across coordinate spaces"):
            motion_heading_coherence([motion(1.0, 0.0, METRIC), motion(1.0, 0.0, NORMALIZED)], 0.1)

    def test_rejects_a_non_positive_floor(self) -> None:
        with pytest.raises(CrowdError, match="must be positive"):
            motion_heading_coherence([motion(1.0, 0.0)], 0.0)


class TestNormalizedUnits:
    def test_geometry_works_in_normalized_space(self) -> None:
        corners = ((0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3))
        square = [point(x, y, NORMALIZED) for x, y in corners]
        result = centroid(square)
        assert (result.x, result.y) == pytest.approx((0.2, 0.2))
        assert rms_radius(square) == pytest.approx(math.hypot(0.1, 0.1))

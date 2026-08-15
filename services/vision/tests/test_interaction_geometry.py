"""Pairwise formulas, checked against scenarios with known analytic answers.

Every trajectory here is generated from a closed-form position function, so the
expected separation, closing speed and relative heading are known by arithmetic
rather than eyeballed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.behavior.models import SubjectMotion
from atreides_vision.errors import InteractionError
from atreides_vision.interaction.geometry import (
    closing_speed,
    pair_distance,
    proximity_since,
    relative_motion_heading,
)

METRIC = MotionSpace.GROUND_PLANE_METRES
NORMALIZED = MotionSpace.NORMALIZED_IMAGE
FPS = 25.0
DT = 1.0 / FPS


@dataclass(frozen=True, slots=True)
class Sample:
    """Minimal :class:`TimedDistance`, so the formulas are tested alone."""

    timestamp_seconds: float
    distance: float


def separation_series(start: float, rate: float, count: int = 26, dt: float = DT) -> list[Sample]:
    """Separations changing at a constant ``rate`` per second."""
    return [Sample(index * dt, start + rate * index * dt) for index in range(count)]


def motion(dx: float, dy: float, space: MotionSpace = METRIC) -> SubjectMotion:
    return SubjectMotion(dx=dx, dy=dy, elapsed_seconds=1.0, space=space)


class TestPairDistance:
    def test_three_four_five(self) -> None:
        assert pair_distance(
            MotionPoint(0.0, 0.0, METRIC), MotionPoint(3.0, 4.0, METRIC)
        ) == pytest.approx(5.0)

    def test_zero_distance(self) -> None:
        """Two tracks at the same projected point — valid, not an error."""
        point = MotionPoint(2.0, 2.0, METRIC)
        assert pair_distance(point, point) == 0.0

    def test_is_symmetric(self) -> None:
        first, second = MotionPoint(1.0, 2.0, METRIC), MotionPoint(4.0, 6.0, METRIC)
        assert pair_distance(first, second) == pair_distance(second, first)

    def test_normalized_distances_are_frame_fractions(self) -> None:
        assert pair_distance(
            MotionPoint(0.1, 0.1, NORMALIZED), MotionPoint(0.4, 0.5, NORMALIZED)
        ) == pytest.approx(0.5)

    def test_rejects_mixed_spaces(self) -> None:
        """A metre minus a frame fraction is nonsense, not a number."""
        with pytest.raises(InteractionError, match="Cannot measure separation"):
            pair_distance(MotionPoint(0.0, 0.0, METRIC), MotionPoint(1.0, 1.0, NORMALIZED))


class TestClosingSpeed:
    def test_constant_separation_is_zero(self) -> None:
        assert closing_speed(separation_series(3.0, 0.0)) == pytest.approx(0.0)

    @pytest.mark.parametrize("rate", [0.5, 1.0, 2.0])
    def test_approaching_is_positive(self, rate: float) -> None:
        """Sign convention: a shrinking gap is a positive closing speed."""
        assert closing_speed(separation_series(8.0, -rate)) == pytest.approx(rate)

    @pytest.mark.parametrize("rate", [0.5, 1.0, 2.0])
    def test_separating_is_negative(self, rate: float) -> None:
        assert closing_speed(separation_series(1.0, rate)) == pytest.approx(-rate)

    def test_needs_two_samples(self) -> None:
        assert closing_speed(separation_series(3.0, -1.0, count=1)) is None

    def test_needs_elapsed_time(self) -> None:
        """Two samples at the same instant give no rate, not a division by zero."""
        assert closing_speed([Sample(1.0, 3.0), Sample(1.0, 2.0)]) is None

    def test_is_frame_rate_independent(self) -> None:
        slow = separation_series(5.0, -1.0, count=11, dt=0.1)
        fast = separation_series(5.0, -1.0, count=61, dt=1 / 60)
        assert closing_speed(slow) == pytest.approx(closing_speed(fast), rel=1e-9)

    def test_a_windowed_mean_reports_the_net_rate(self) -> None:
        """Approach then retreat within one window nets out, by design."""
        approach = separation_series(4.0, -1.0, count=13)
        retreat = [
            Sample(
                approach[-1].timestamp_seconds + (index + 1) * DT,
                approach[-1].distance + 1.0 * (index + 1) * DT,
            )
            for index in range(12)
        ]
        combined = approach + retreat
        assert combined[-1].distance == pytest.approx(combined[0].distance, abs=0.05)
        assert closing_speed(combined) == pytest.approx(0.0, abs=0.05)


class TestRelativeMotionHeading:
    def test_same_direction_is_zero(self) -> None:
        assert relative_motion_heading(motion(1.0, 0.0), motion(2.0, 0.0), 0.1) == (
            pytest.approx(0.0)
        )

    def test_opposite_directions_is_pi(self) -> None:
        assert relative_motion_heading(motion(1.0, 0.0), motion(-1.0, 0.0), 0.1) == (
            pytest.approx(math.pi)
        )

    def test_perpendicular_is_half_pi(self) -> None:
        assert relative_motion_heading(motion(1.0, 0.0), motion(0.0, 1.0), 0.1) == (
            pytest.approx(math.pi / 2)
        )

    def test_is_unsigned(self) -> None:
        """Which of the two is turning relative to the other carries no meaning."""
        clockwise = relative_motion_heading(motion(1.0, 0.0), motion(0.0, 1.0), 0.1)
        counter = relative_motion_heading(motion(1.0, 0.0), motion(0.0, -1.0), 0.1)
        assert clockwise == pytest.approx(counter)

    def test_wraps_near_the_pi_boundary(self) -> None:
        """+170 deg and -170 deg differ by 20 deg, not 340."""
        first = motion(math.cos(math.radians(170)), math.sin(math.radians(170)))
        second = motion(math.cos(math.radians(-170)), math.sin(math.radians(-170)))
        assert relative_motion_heading(first, second, 0.1) == pytest.approx(math.radians(20))

    def test_result_is_within_zero_to_pi(self) -> None:
        for degrees in range(0, 360, 15):
            angle = math.radians(degrees)
            value = relative_motion_heading(
                motion(1.0, 0.0), motion(math.cos(angle), math.sin(angle)), 0.1
            )
            assert value is not None
            assert 0.0 <= value <= math.pi + 1e-12

    def test_a_stationary_subject_omits_the_signal(self) -> None:
        """Not zero: "going nowhere" is not "going the same way"."""
        assert relative_motion_heading(motion(1.0, 0.0), motion(0.0, 0.0), 0.1) is None

    def test_both_stationary_omits_the_signal(self) -> None:
        assert relative_motion_heading(motion(0.0, 0.0), motion(0.0, 0.0), 0.1) is None

    def test_missing_motion_omits_the_signal(self) -> None:
        assert relative_motion_heading(None, motion(1.0, 0.0), 0.1) is None
        assert relative_motion_heading(motion(1.0, 0.0), None, 0.1) is None

    def test_movement_below_the_floor_omits_the_signal(self) -> None:
        assert relative_motion_heading(motion(0.05, 0.0), motion(1.0, 0.0), 0.1) is None

    def test_rejects_mixed_spaces(self) -> None:
        with pytest.raises(InteractionError, match="across coordinate spaces"):
            relative_motion_heading(motion(1.0, 0.0, METRIC), motion(1.0, 0.0, NORMALIZED), 0.1)


class TestProximitySince:
    def test_starts_a_spell_at_the_window_start(self) -> None:
        """The spell dates from the evidence, not the frame it was confirmed."""
        assert (
            proximity_since(
                None, distance=1.0, threshold=1.5, window_start_seconds=2.0, continuous=True
            )
            == 2.0
        )

    def test_continues_an_existing_spell(self) -> None:
        assert (
            proximity_since(
                2.0, distance=1.0, threshold=1.5, window_start_seconds=9.0, continuous=True
            )
            == 2.0
        )

    def test_separating_ends_the_spell(self) -> None:
        assert (
            proximity_since(
                2.0, distance=3.0, threshold=1.5, window_start_seconds=9.0, continuous=True
            )
            is None
        )

    def test_the_boundary_counts_as_close(self) -> None:
        """``distance == threshold`` is inside, defined rather than incidental."""
        assert (
            proximity_since(
                2.0, distance=1.5, threshold=1.5, window_start_seconds=9.0, continuous=True
            )
            == 2.0
        )

    def test_just_beyond_the_boundary_ends_it(self) -> None:
        assert (
            proximity_since(
                2.0,
                distance=1.5000001,
                threshold=1.5,
                window_start_seconds=9.0,
                continuous=True,
            )
            is None
        )

    def test_a_discontinuity_ends_the_spell(self) -> None:
        """We cannot claim they stayed together through time nobody watched."""
        assert (
            proximity_since(
                2.0, distance=0.5, threshold=1.5, window_start_seconds=9.0, continuous=False
            )
            is None
        )

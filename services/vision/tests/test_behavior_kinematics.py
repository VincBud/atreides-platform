"""Kinematic formulas, checked against synthetic trajectories with known answers.

Every trajectory here is generated from a closed-form position function, so the
expected speed, acceleration and turn rate are known exactly rather than
eyeballed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest

from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.behavior.kinematics import (
    acceleration,
    heading_change_rate,
    heading_radians,
    mean_speed,
    net_displacement,
    path_length,
    span_center_seconds,
    span_seconds,
    stationary_since,
    wrap_angle,
)

NORMALIZED = MotionSpace.NORMALIZED_IMAGE
FPS = 25.0
DT = 1.0 / FPS


@dataclass(frozen=True, slots=True)
class Sample:
    """Minimal :class:`TimedPosition`, so the formulas are tested alone."""

    timestamp_seconds: float
    position: MotionPoint


def trajectory(
    positions: Callable[[float], tuple[float, float]],
    count: int,
    dt: float = DT,
    start: float = 0.0,
) -> list[Sample]:
    """Sample a position function at a fixed rate."""
    samples = []
    for index in range(count):
        t = start + index * dt
        x, y = positions(t)
        samples.append(Sample(t, MotionPoint(x, y, NORMALIZED)))
    return samples


def still(x: float = 0.5, y: float = 0.5) -> Callable[[float], tuple[float, float]]:
    return lambda _t: (x, y)


def constant_velocity(vx: float, vy: float = 0.0) -> Callable[[float], tuple[float, float]]:
    return lambda t: (0.1 + vx * t, 0.5 + vy * t)


def constant_acceleration(a: float) -> Callable[[float], tuple[float, float]]:
    """Starts at rest, accelerates along +x. Speed at time t is ``a * t``."""
    return lambda t: (0.1 + 0.5 * a * t * t, 0.5)


class TestPrimitives:
    def test_path_length_sums_segments(self) -> None:
        samples = trajectory(constant_velocity(0.2), 26)  # 1.0 s at 0.2 per s
        assert path_length(samples) == pytest.approx(0.2)

    def test_path_length_counts_movement_in_place(self) -> None:
        """Someone pacing back and forth has travelled, even if not gone far."""
        samples = [
            Sample(0.0, MotionPoint(0.5, 0.5, NORMALIZED)),
            Sample(0.5, MotionPoint(0.6, 0.5, NORMALIZED)),
            Sample(1.0, MotionPoint(0.5, 0.5, NORMALIZED)),
        ]
        assert path_length(samples) == pytest.approx(0.2)
        assert net_displacement(samples) == pytest.approx((0.0, 0.0))

    @pytest.mark.parametrize("count", [0, 1])
    def test_degenerate_inputs(self, count: int) -> None:
        samples = trajectory(still(), count)
        assert path_length(samples) == 0.0
        assert net_displacement(samples) == (0.0, 0.0)
        assert span_seconds(samples) == 0.0
        assert mean_speed(samples) is None

    def test_span_center_is_the_midpoint_of_the_span(self) -> None:
        samples = trajectory(still(), 5, dt=0.1)
        assert span_center_seconds(samples) == pytest.approx(0.2)

    def test_span_center_needs_a_sample(self) -> None:
        with pytest.raises(ValueError, match="at least one sample"):
            span_center_seconds([])

    @pytest.mark.parametrize(
        ("angle", "expected"),
        [
            (0.0, 0.0),
            (math.pi / 2, math.pi / 2),
            (math.pi, math.pi),
            (-math.pi, math.pi),
            (3 * math.pi / 2, -math.pi / 2),
            (math.tau + 0.3, 0.3),
        ],
    )
    def test_wrap_angle(self, angle: float, expected: float) -> None:
        assert wrap_angle(angle) == pytest.approx(expected)

    def test_wrap_angle_makes_the_pi_boundary_a_small_turn(self) -> None:
        """+170 deg to -170 deg is a 20 deg turn, not a 340 deg one."""
        turn = wrap_angle(math.radians(-170) - math.radians(170))
        assert abs(turn) == pytest.approx(math.radians(20))


class TestSpeed:
    def test_zero_movement_is_zero_speed(self) -> None:
        assert mean_speed(trajectory(still(), 26)) == pytest.approx(0.0)

    @pytest.mark.parametrize("velocity", [0.05, 0.2, 1.5])
    def test_constant_velocity_recovers_the_velocity(self, velocity: float) -> None:
        samples = trajectory(constant_velocity(velocity), 26)
        assert mean_speed(samples) == pytest.approx(velocity)

    def test_diagonal_movement_uses_euclidean_distance(self) -> None:
        samples = trajectory(constant_velocity(0.3, 0.4), 26)
        assert mean_speed(samples) == pytest.approx(0.5)

    def test_needs_two_samples(self) -> None:
        assert mean_speed(trajectory(still(), 1)) is None

    def test_needs_elapsed_time(self) -> None:
        """Two samples at the same instant give no speed, not a division by zero."""
        samples = [
            Sample(1.0, MotionPoint(0.1, 0.1, NORMALIZED)),
            Sample(1.0, MotionPoint(0.2, 0.2, NORMALIZED)),
        ]
        assert mean_speed(samples) is None

    def test_is_frame_rate_independent(self) -> None:
        """The same real motion at 10 and 60 FPS reports the same speed."""
        slow = trajectory(constant_velocity(0.2), 11, dt=0.1)
        fast = trajectory(constant_velocity(0.2), 61, dt=1 / 60)
        assert mean_speed(slow) == pytest.approx(mean_speed(fast), rel=1e-9)


class TestAcceleration:
    def test_constant_velocity_has_zero_acceleration(self) -> None:
        samples = trajectory(constant_velocity(0.2), 26)
        assert acceleration(samples) == pytest.approx(0.0, abs=1e-12)

    def test_zero_movement_has_zero_acceleration(self) -> None:
        assert acceleration(trajectory(still(), 26)) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("rate", [0.1, 0.5, -0.0])
    def test_constant_acceleration_is_recovered_exactly(self, rate: float) -> None:
        """Split-half differencing is exact for constant acceleration."""
        samples = trajectory(constant_acceleration(rate), 26)
        assert acceleration(samples) == pytest.approx(rate, rel=1e-9, abs=1e-12)

    def test_deceleration_is_negative(self) -> None:
        """Fast for the first half, slow for the second."""
        samples = trajectory(constant_velocity(0.4), 13)
        tail_start = samples[-1]
        samples += [
            Sample(
                tail_start.timestamp_seconds + (index + 1) * DT,
                MotionPoint(tail_start.position.x + 0.02 * (index + 1) * DT, 0.5, NORMALIZED),
            )
            for index in range(13)
        ]
        result = acceleration(samples)
        assert result is not None and result < 0

    def test_needs_two_samples_in_each_half(self) -> None:
        assert acceleration(trajectory(constant_velocity(0.2), 3)) is None

    def test_is_frame_rate_independent(self) -> None:
        slow = trajectory(constant_acceleration(0.3), 11, dt=0.1)
        fast = trajectory(constant_acceleration(0.3), 61, dt=1 / 60)
        assert acceleration(slow) == pytest.approx(acceleration(fast), rel=1e-9)


class TestHeading:
    def test_direction_of_travel(self) -> None:
        assert heading_radians(trajectory(constant_velocity(0.2), 26), 0.01) == pytest.approx(0.0)
        assert heading_radians(trajectory(constant_velocity(0.0, 0.2), 26), 0.01) == pytest.approx(
            math.pi / 2
        )

    def test_is_undefined_below_the_displacement_floor(self) -> None:
        """A standing person has no direction of travel — jitter is not a heading."""
        assert heading_radians(trajectory(still(), 26), 0.01) is None

    def test_straight_line_has_no_turn(self) -> None:
        samples = trajectory(constant_velocity(0.2), 26)
        assert heading_change_rate(samples, 0.01) == pytest.approx(0.0, abs=1e-12)

    def test_a_right_angle_turn_is_measured(self) -> None:
        """One second travelling +x, then one second travelling +y."""
        first = trajectory(constant_velocity(0.2), 26)
        pivot = first[-1]
        second = [
            Sample(
                pivot.timestamp_seconds + (index + 1) * DT,
                MotionPoint(
                    pivot.position.x, pivot.position.y + 0.2 * (index + 1) * DT, NORMALIZED
                ),
            )
            for index in range(25)
        ]
        rate = heading_change_rate(first + second, 0.01)
        assert rate is not None
        # Half-window centres are ~1.0 s apart, so ~90 deg of turn per second.
        assert abs(rate) == pytest.approx(math.pi / 2, rel=0.25)

    def test_turn_direction_has_a_sign(self) -> None:
        def turning(sign: float) -> list[Sample]:
            first = trajectory(constant_velocity(0.2), 26)
            pivot = first[-1]
            return first + [
                Sample(
                    pivot.timestamp_seconds + (index + 1) * DT,
                    MotionPoint(
                        pivot.position.x,
                        pivot.position.y + sign * 0.2 * (index + 1) * DT,
                        NORMALIZED,
                    ),
                )
                for index in range(25)
            ]

        clockwise = heading_change_rate(turning(1.0), 0.01)
        counter = heading_change_rate(turning(-1.0), 0.01)
        assert clockwise is not None and counter is not None
        assert clockwise > 0 > counter
        assert clockwise == pytest.approx(-counter)

    def test_a_reversal_is_reported_as_a_half_turn_not_a_full_one(self) -> None:
        """Angle wrapping keeps a 180 deg reversal at 180 deg, never 360 deg."""
        out = trajectory(constant_velocity(0.2), 26)
        pivot = out[-1]
        back = [
            Sample(
                pivot.timestamp_seconds + (index + 1) * DT,
                MotionPoint(pivot.position.x - 0.2 * (index + 1) * DT, 0.5, NORMALIZED),
            )
            for index in range(25)
        ]
        samples = out + back
        rate = heading_change_rate(samples, 0.01)
        assert rate is not None

        # Uniform sampling puts the two half-window centres exactly half the
        # total span apart, so the turn itself is rate * (span / 2).
        turn = rate * span_seconds(samples) / 2.0
        assert abs(turn) == pytest.approx(math.pi)
        assert abs(turn) <= math.pi + 1e-9

    def test_no_turn_rate_when_movement_is_negligible(self) -> None:
        assert heading_change_rate(trajectory(still(), 26), 0.01) is None

    def test_needs_two_samples_in_each_half(self) -> None:
        assert heading_change_rate(trajectory(constant_velocity(0.2), 3), 0.001) is None


class TestStationarySince:
    def test_starts_a_spell_at_the_window_start(self) -> None:
        """The spell dates from the evidence, not from the frame it was confirmed."""
        assert (
            stationary_since(
                None, speed=0.0, window_start_seconds=2.0, threshold=0.02, continuous=True
            )
            == 2.0
        )

    def test_continues_an_existing_spell(self) -> None:
        assert (
            stationary_since(
                2.0, speed=0.001, window_start_seconds=9.0, threshold=0.02, continuous=True
            )
            == 2.0
        )

    def test_movement_ends_the_spell(self) -> None:
        assert (
            stationary_since(
                2.0, speed=0.5, window_start_seconds=9.0, threshold=0.02, continuous=True
            )
            is None
        )

    def test_the_threshold_is_exclusive(self) -> None:
        assert (
            stationary_since(
                None, speed=0.02, window_start_seconds=1.0, threshold=0.02, continuous=True
            )
            is None
        )

    def test_a_discontinuity_ends_the_spell(self) -> None:
        """We cannot claim someone stood still through an interval we could not see."""
        assert (
            stationary_since(
                2.0, speed=0.0, window_start_seconds=9.0, threshold=0.02, continuous=False
            )
            is None
        )

    def test_an_unknown_speed_ends_the_spell(self) -> None:
        assert (
            stationary_since(
                2.0, speed=None, window_start_seconds=9.0, threshold=0.02, continuous=True
            )
            is None
        )


class TestDeterminism:
    def test_repeated_evaluation_is_identical(self) -> None:
        samples: Sequence[Sample] = trajectory(constant_acceleration(0.4), 26)
        for _ in range(5):
            assert mean_speed(samples) == mean_speed(samples)
            assert acceleration(samples) == acceleration(samples)
            assert heading_change_rate(samples, 0.01) == heading_change_rate(samples, 0.01)

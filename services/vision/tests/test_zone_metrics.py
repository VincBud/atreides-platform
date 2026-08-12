"""Motion level bounds and end-to-end metric assembly."""

from __future__ import annotations

import pytest

from atreides_vision.zones.config import ZoneAnalysisConfig
from atreides_vision.zones.geometry import NormalizedPoint
from atreides_vision.zones.metrics import MotionLevelEstimator, ZoneMetricsCalculator
from atreides_vision.zones.models import Zone, ZoneLayout, ZoneMembership, ZonePolygon
from atreides_vision.zones.state import TrackMovementStore, ZoneStateStore

ZONE = Zone(
    id="dancefloor",
    name="Dancefloor",
    polygon=ZonePolygon.from_pairs([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
)
LAYOUT = ZoneLayout(camera_id="cam_01", zones=(ZONE,))


class TestMotionLevelBounds:
    def test_starts_at_zero_for_an_unseen_zone(self) -> None:
        assert MotionLevelEstimator().level("anything") == 0.0

    def test_no_movement_is_zero(self) -> None:
        estimator = MotionLevelEstimator()
        assert estimator.update("z", 0.0) == 0.0

    @pytest.mark.parametrize("speed", [0.0, 0.01, 0.15, 1.0, 50.0, 1e9])
    def test_always_within_zero_and_one(self, speed: float) -> None:
        estimator = MotionLevelEstimator()
        for _ in range(200):
            level = estimator.update("z", speed)
            assert 0.0 <= level <= 1.0

    def test_negative_speed_cannot_push_it_below_zero(self) -> None:
        """Defensive: speed should never be negative, but the bound must hold anyway."""
        estimator = MotionLevelEstimator()
        assert estimator.update("z", -5.0) == 0.0

    def test_saturates_at_the_reference_speed(self) -> None:
        config = ZoneAnalysisConfig(motion_reference_speed=0.15, motion_smoothing_frames=1)
        estimator = MotionLevelEstimator(config)
        assert estimator.update("z", 0.15) == pytest.approx(1.0)
        assert estimator.update("z", 0.30) == pytest.approx(1.0)

    def test_scales_linearly_below_the_reference(self) -> None:
        config = ZoneAnalysisConfig(motion_reference_speed=0.2, motion_smoothing_frames=1)
        estimator = MotionLevelEstimator(config)
        assert estimator.update("z", 0.1) == pytest.approx(0.5)


class TestMotionLevelSmoothing:
    def test_rises_gradually_towards_the_target(self) -> None:
        config = ZoneAnalysisConfig(motion_reference_speed=0.1, motion_smoothing_frames=20)
        estimator = MotionLevelEstimator(config)
        estimator.update("z", 0.0)  # seed at rest
        first = estimator.update("z", 0.1)
        assert 0.0 < first < 1.0, "one frame of movement must not jump straight to 1.0"

        for _ in range(200):
            estimator.update("z", 0.1)
        assert estimator.level("z") == pytest.approx(1.0, abs=0.01)

    def test_decays_back_towards_zero(self) -> None:
        config = ZoneAnalysisConfig(motion_reference_speed=0.1, motion_smoothing_frames=10)
        estimator = MotionLevelEstimator(config)
        for _ in range(100):
            estimator.update("z", 0.1)
        for _ in range(300):
            estimator.update("z", 0.0)
        assert estimator.level("z") == pytest.approx(0.0, abs=0.01)

    def test_zones_are_smoothed_independently(self) -> None:
        estimator = MotionLevelEstimator()
        for _ in range(50):
            estimator.update("busy", 0.15)
            estimator.update("quiet", 0.0)
        assert estimator.level("busy") > estimator.level("quiet")

    def test_forget_resets_a_zone(self) -> None:
        estimator = MotionLevelEstimator()
        estimator.update("z", 0.15)
        estimator.forget("z")
        assert estimator.level("z") == 0.0


class TestZoneMetricsCalculator:
    def build(
        self, grace: int = 0
    ) -> tuple[ZoneMetricsCalculator, ZoneStateStore, TrackMovementStore]:
        config = ZoneAnalysisConfig(exit_grace_frames=grace, motion_smoothing_frames=1)
        return (
            ZoneMetricsCalculator(config),
            ZoneStateStore(LAYOUT.zone_ids, config),
            TrackMovementStore(config),
        )

    def test_empty_zone_snapshot(self) -> None:
        calculator, state, movement = self.build()
        snapshot = calculator.snapshot(ZONE, state, movement, timestamp_seconds=0.0)
        assert snapshot.occupancy == 0
        assert snapshot.entries == 0
        assert snapshot.mean_normalized_speed == 0.0
        assert snapshot.motion_level == 0.0
        assert snapshot.average_dwell_seconds == 0.0

    def test_snapshot_reflects_state_and_movement(self) -> None:
        calculator, state, movement = self.build()
        state.update((ZoneMembership(ZONE.id, frozenset({1, 2})),), 1, 0.0)
        movement.observe(1, NormalizedPoint(0.1, 0.5), 1, 0.0)
        movement.observe(1, NormalizedPoint(0.2, 0.5), 2, 1.0)
        movement.observe(2, NormalizedPoint(0.6, 0.5), 1, 0.0)
        movement.observe(2, NormalizedPoint(0.6, 0.5), 2, 1.0)
        state.update((ZoneMembership(ZONE.id, frozenset({1, 2})),), 2, 1.0)

        snapshot = calculator.snapshot(ZONE, state, movement, timestamp_seconds=1.0)
        assert snapshot.occupancy == 2
        assert snapshot.entries == 2
        # One track moving at 0.1/s, one stationary.
        assert snapshot.mean_normalized_speed == pytest.approx(0.05)
        assert snapshot.max_dwell_seconds == pytest.approx(1.0)
        assert snapshot.zone_name == "Dancefloor"

    def test_report_covers_every_configured_zone(self) -> None:
        quiet = Zone(
            id="bar",
            name="Bar",
            polygon=ZonePolygon.from_pairs([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]]),
        )
        layout = ZoneLayout(camera_id="cam_01", zones=(ZONE, quiet))
        config = ZoneAnalysisConfig()
        calculator = ZoneMetricsCalculator(config)
        state = ZoneStateStore(layout.zone_ids, config)
        movement = TrackMovementStore(config)

        report = calculator.report(layout, state, movement, frame_index=5, timestamp_seconds=0.2)
        assert [s.zone_id for s in report.snapshots] == ["dancefloor", "bar"]
        assert report.frame_index == 5
        assert report.timestamp_seconds == 0.2

    def test_motion_level_stays_bounded_through_the_calculator(self) -> None:
        calculator, state, movement = self.build()
        state.update((ZoneMembership(ZONE.id, frozenset({1})),), 1, 0.0)
        # Absurdly fast movement: 5 frame-widths in a hundredth of a second.
        movement.observe(1, NormalizedPoint(0.0, 0.0), 1, 0.0)
        movement.observe(1, NormalizedPoint(1.0, 1.0), 2, 0.01)
        snapshot = calculator.snapshot(ZONE, state, movement, timestamp_seconds=0.01)
        assert 0.0 <= snapshot.motion_level <= 1.0
        assert snapshot.motion_level == 1.0

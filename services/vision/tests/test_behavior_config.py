"""Behaviour configuration validation and defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.config import (
    DEFAULT_DEDUPLICATE_DETECTIONS,
    DEFAULT_DETECTION_CONFIDENCE,
    DEFAULT_OCSORT_HIGH_CONF_THRESHOLD,
    DEFAULT_TRACKER,
    PipelineConfig,
    Tracker,
    TrackerConfig,
    VisionConfig,
)
from atreides_vision.errors import ConfigurationError


class TestDefaults:
    def test_defaults_are_valid(self) -> None:
        config = BehaviorAnalysisConfig()
        assert config.speed_window_seconds == 1.0
        assert config.stationary_speed_threshold == 0.02
        assert config.max_observation_gap_seconds == 0.5

    def test_history_covers_every_signal_window(self) -> None:
        config = BehaviorAnalysisConfig()
        assert config.history_window_seconds >= config.longest_signal_window_seconds

    def test_longest_window_is_the_maximum(self) -> None:
        config = BehaviorAnalysisConfig(
            speed_window_seconds=0.5,
            acceleration_window_seconds=1.5,
            heading_window_seconds=1.0,
            history_window_seconds=2.0,
        )
        assert config.longest_signal_window_seconds == 1.5

    def test_is_immutable(self) -> None:
        config = BehaviorAnalysisConfig()
        with pytest.raises(AttributeError):
            config.speed_window_seconds = 2.0  # type: ignore[misc]


class TestValidation:
    @pytest.mark.parametrize(
        "field",
        [
            "speed_window_seconds",
            "acceleration_window_seconds",
            "heading_window_seconds",
            "max_observation_gap_seconds",
            "stationary_speed_threshold",
            "min_heading_displacement",
        ],
    )
    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_rejects_non_positive_durations_and_thresholds(self, field: str, value: float) -> None:
        with pytest.raises(ConfigurationError, match=f"{field} must be positive"):
            BehaviorAnalysisConfig(**{field: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "field",
        ["max_samples_per_subject", "stale_subject_ttl_frames", "max_tracked_subjects"],
    )
    @pytest.mark.parametrize("value", [0, -5])
    def test_rejects_non_positive_counts(self, field: str, value: int) -> None:
        with pytest.raises(ConfigurationError, match=f"{field} must be >= 1"):
            BehaviorAnalysisConfig(**{field: value})  # type: ignore[arg-type]

    def test_rejects_a_history_window_shorter_than_a_signal_window(self) -> None:
        """Otherwise a signal is silently computed over less history than configured."""
        with pytest.raises(ConfigurationError, match="must be at least the longest signal window"):
            BehaviorAnalysisConfig(speed_window_seconds=3.0, history_window_seconds=1.0)

    def test_history_window_equal_to_the_longest_is_allowed(self) -> None:
        assert (
            BehaviorAnalysisConfig(
                speed_window_seconds=2.0,
                acceleration_window_seconds=2.0,
                heading_window_seconds=2.0,
                history_window_seconds=2.0,
            ).history_window_seconds
            == 2.0
        )


class TestVisionConfigIntegration:
    def test_vision_config_carries_behavior_settings(self, tmp_path: Path) -> None:
        config = VisionConfig(
            pipeline=PipelineConfig(input_path=tmp_path / "in.mp4", output_path=tmp_path / "o.mp4")
        )
        assert isinstance(config.behavior, BehaviorAnalysisConfig)

    def test_v021_validated_defaults_are_untouched(self) -> None:
        """V0.3 must not move a single validated V0.2.1 value."""
        assert DEFAULT_DETECTION_CONFIDENCE == 0.30
        assert DEFAULT_OCSORT_HIGH_CONF_THRESHOLD == 0.30
        assert DEFAULT_TRACKER is Tracker.OCSORT
        assert DEFAULT_DEDUPLICATE_DETECTIONS is True

        tracker = TrackerConfig()
        assert tracker.ocsort_minimum_iou_threshold == 0.3
        assert tracker.lost_track_buffer == 30
        assert tracker.minimum_consecutive_frames == 2
        assert tracker.direction_consistency_weight == 0.2
        assert tracker.delta_t == 3

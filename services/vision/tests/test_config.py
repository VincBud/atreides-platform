"""Configuration validation and device-resolution policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from atreides_vision.config import (
    DEFAULT_DEDUPLICATE_DETECTIONS,
    DEFAULT_DETECTION_CONFIDENCE,
    DEFAULT_OCSORT_HIGH_CONF_THRESHOLD,
    DEFAULT_TRACKER,
    DetectorConfig,
    Device,
    PipelineConfig,
    Tracker,
    TrackerConfig,
    VisionConfig,
    VisualizationConfig,
    resolve_device,
)
from atreides_vision.errors import ConfigurationError, DeviceUnavailableError


@dataclass(frozen=True)
class StubProbe:
    """An :class:`AcceleratorProbe` with hard-coded answers."""

    cuda: bool = False
    mps: bool = False

    def cuda_available(self) -> bool:
        return self.cuda

    def mps_available(self) -> bool:
        return self.mps


class TestResolveDevice:
    def test_cpu_is_always_available(self) -> None:
        assert resolve_device(Device.CPU, StubProbe()) == "cpu"

    def test_auto_prefers_cuda_then_mps_then_cpu(self) -> None:
        assert resolve_device(Device.AUTO, StubProbe(cuda=True, mps=True)) == "cuda"
        assert resolve_device(Device.AUTO, StubProbe(cuda=False, mps=True)) == "mps"
        assert resolve_device(Device.AUTO, StubProbe(cuda=False, mps=False)) == "cpu"

    def test_explicit_accelerator_is_honoured_when_available(self) -> None:
        assert resolve_device(Device.CUDA, StubProbe(cuda=True)) == "cuda"
        assert resolve_device(Device.MPS, StubProbe(mps=True)) == "mps"

    def test_missing_cuda_raises_with_actionable_message(self) -> None:
        with pytest.raises(DeviceUnavailableError) as excinfo:
            resolve_device(Device.CUDA, StubProbe(cuda=False))
        message = str(excinfo.value)
        assert "CUDA" in message
        assert "--device cpu" in message

    def test_missing_mps_raises_with_actionable_message(self) -> None:
        with pytest.raises(DeviceUnavailableError) as excinfo:
            resolve_device(Device.MPS, StubProbe(mps=False))
        assert "--device cpu" in str(excinfo.value)

    def test_probe_that_raises_is_treated_as_unavailable(self) -> None:
        class ExplodingProbe:
            def cuda_available(self) -> bool:
                raise RuntimeError("driver on fire")

            def mps_available(self) -> bool:
                return False

        with pytest.raises(RuntimeError):
            # A probe raising is the probe's bug; TorchAcceleratorProbe swallows
            # the equivalent failure itself (covered below).
            resolve_device(Device.AUTO, ExplodingProbe())

    def test_torch_probe_never_raises(self) -> None:
        from atreides_vision.config import TorchAcceleratorProbe

        probe = TorchAcceleratorProbe()
        assert isinstance(probe.cuda_available(), bool)
        assert isinstance(probe.mps_available(), bool)


class TestDetectorConfig:
    def test_defaults_target_coco_person(self) -> None:
        assert DetectorConfig().person_class_id == 1

    @pytest.mark.parametrize("confidence", [0.0, -0.2, 1.5])
    def test_rejects_out_of_range_confidence(self, confidence: float) -> None:
        with pytest.raises(ConfigurationError, match="confidence must be in"):
            DetectorConfig(confidence_threshold=confidence)

    def test_accepts_boundary_confidence(self) -> None:
        assert DetectorConfig(confidence_threshold=1.0).confidence_threshold == 1.0

    def test_rejects_non_positive_resolution(self) -> None:
        with pytest.raises(ConfigurationError, match="resolution must be positive"):
            DetectorConfig(resolution=0)


class TestTrackerConfig:
    def test_rejects_non_positive_frame_rate(self) -> None:
        with pytest.raises(ConfigurationError, match="frame_rate must be positive"):
            TrackerConfig(frame_rate=0.0)

    def test_rejects_bad_activation_threshold(self) -> None:
        with pytest.raises(ConfigurationError, match="track_activation_threshold"):
            TrackerConfig(track_activation_threshold=1.2)

    def test_rejects_zero_consecutive_frames(self) -> None:
        with pytest.raises(ConfigurationError, match="minimum_consecutive_frames"):
            TrackerConfig(minimum_consecutive_frames=0)

    def test_ocsort_v021_validated_defaults(self) -> None:
        """The frozen V0.2.1 candidate, asserted field by field."""
        config = TrackerConfig()
        assert config.ocsort_minimum_iou_threshold == 0.3
        assert config.direction_consistency_weight == 0.2
        assert config.high_conf_det_threshold == 0.30
        assert config.high_conf_det_threshold == DEFAULT_OCSORT_HIGH_CONF_THRESHOLD
        assert config.delta_t == 3
        assert config.lost_track_buffer == 30
        assert config.minimum_consecutive_frames == 2

    def test_ocsort_iou_is_a_separate_field_from_bytetrack(self) -> None:
        """They share a value today; they must not share a knob.

        Tuning one must never move the other, or an A/B is contaminated.
        """
        assert TrackerConfig().minimum_iou_threshold == 0.3
        assert TrackerConfig().ocsort_minimum_iou_threshold == 0.3

        tuned = TrackerConfig(ocsort_minimum_iou_threshold=0.2)
        assert tuned.ocsort_minimum_iou_threshold == 0.2
        assert tuned.minimum_iou_threshold == 0.3, "ByteTrack must be unaffected"

    def test_ocsort_activation_stays_at_or_below_the_detector_threshold(self) -> None:
        """A higher tracker threshold would silently discard passing detections."""
        assert TrackerConfig().high_conf_det_threshold <= DetectorConfig().confidence_threshold

    @pytest.mark.parametrize("threshold", [-0.01, 1.01])
    def test_rejects_bad_ocsort_iou_threshold(self, threshold: float) -> None:
        with pytest.raises(ConfigurationError, match="ocsort_minimum_iou_threshold"):
            TrackerConfig(ocsort_minimum_iou_threshold=threshold)

    @pytest.mark.parametrize("weight", [-0.1, 1.5])
    def test_rejects_bad_direction_consistency_weight(self, weight: float) -> None:
        with pytest.raises(ConfigurationError, match="direction_consistency_weight"):
            TrackerConfig(direction_consistency_weight=weight)

    @pytest.mark.parametrize("threshold", [-0.01, 1.01])
    def test_rejects_bad_high_conf_det_threshold(self, threshold: float) -> None:
        with pytest.raises(ConfigurationError, match="high_conf_det_threshold"):
            TrackerConfig(high_conf_det_threshold=threshold)

    def test_rejects_zero_delta_t(self) -> None:
        with pytest.raises(ConfigurationError, match="delta_t"):
            TrackerConfig(delta_t=0)


class TestTrackerChoice:
    def test_values_match_the_cli_spelling(self) -> None:
        assert Tracker.BYTETRACK.value == "bytetrack"
        assert Tracker.OCSORT.value == "ocsort"

    def test_parsed_from_a_string(self) -> None:
        assert Tracker("ocsort") is Tracker.OCSORT

    def test_unknown_tracker_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="deepsort"):
            Tracker("deepsort")


class TestVisualizationConfig:
    def test_rejects_short_trail(self) -> None:
        with pytest.raises(ConfigurationError, match="trail_length must be >= 2"):
            VisualizationConfig(trail_length=1)

    def test_rejects_zero_ttl(self) -> None:
        with pytest.raises(ConfigurationError, match="trail_ttl_frames"):
            VisualizationConfig(trail_ttl_frames=0)

    def test_rejects_zero_max_trails(self) -> None:
        with pytest.raises(ConfigurationError, match="max_trails"):
            VisualizationConfig(max_trails=0)


class TestPipelineConfig:
    def test_rejects_zero_max_frames(self) -> None:
        with pytest.raises(ConfigurationError, match="max_frames"):
            PipelineConfig(Path("in.mp4"), Path("out.mp4"), max_frames=0)

    def test_rejects_zero_log_interval(self) -> None:
        with pytest.raises(ConfigurationError, match="log_every_n_frames"):
            PipelineConfig(Path("in.mp4"), Path("out.mp4"), log_every_n_frames=0)

    def test_none_max_frames_means_whole_video(self) -> None:
        assert PipelineConfig(Path("in.mp4"), Path("out.mp4")).max_frames is None


class TestVisionConfig:
    def test_sub_configs_default(self) -> None:
        config = VisionConfig(pipeline=PipelineConfig(Path("in.mp4"), Path("out.mp4")))
        assert config.detector.confidence_threshold == DEFAULT_DETECTION_CONFIDENCE
        assert config.tracker.frame_rate == 30.0
        assert config.visualization.trail_length == 25


class TestV021ValidatedDefaults:
    """A standard run must resolve to the V0.2.1 validated candidate.

    These constants were fixed by a three-scene fixed-CCTV validation. Changing
    one is a deliberate act that should fail here first, not a silent drift.
    """

    def test_detector_confidence(self) -> None:
        assert DEFAULT_DETECTION_CONFIDENCE == 0.30
        assert DetectorConfig().confidence_threshold == 0.30

    def test_tracker_is_ocsort(self) -> None:
        assert DEFAULT_TRACKER is Tracker.OCSORT

    def test_ocsort_high_conf_threshold(self) -> None:
        assert DEFAULT_OCSORT_HIGH_CONF_THRESHOLD == 0.30
        assert TrackerConfig().high_conf_det_threshold == 0.30

    def test_deduplication_is_on(self) -> None:
        assert DEFAULT_DEDUPLICATE_DETECTIONS is True
        assert PipelineConfig(Path("in.mp4"), Path("out.mp4")).deduplicate_detections is True

    def test_untouched_ocsort_parameters_kept_their_values(self) -> None:
        """V0.2.1 promoted defaults; it did not retune the tracker."""
        config = TrackerConfig()
        assert config.ocsort_minimum_iou_threshold == 0.30
        assert config.lost_track_buffer == 30
        assert config.minimum_consecutive_frames == 2
        assert config.direction_consistency_weight == 0.2
        assert config.delta_t == 3

    def test_whole_candidate_resolves_from_a_bare_config(self) -> None:
        config = VisionConfig(pipeline=PipelineConfig(Path("in.mp4"), Path("out.mp4")))
        assert config.detector.confidence_threshold == 0.30
        assert config.tracker.high_conf_det_threshold == 0.30
        assert config.pipeline.deduplicate_detections is True

    def test_bytetrack_remains_available(self) -> None:
        """Promoting OC-SORT must not remove the alternative."""
        assert Tracker.BYTETRACK in set(Tracker)
        assert Tracker("bytetrack") is Tracker.BYTETRACK

    def test_dedup_policy_is_rule_b3(self) -> None:
        from atreides_vision.detection.deduplication import RULE_B3

        assert (RULE_B3.min_iou, RULE_B3.min_area_similarity, RULE_B3.max_center_distance) == (
            0.85,
            0.95,
            0.03,
        )

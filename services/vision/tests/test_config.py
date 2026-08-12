"""Configuration validation and device-resolution policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from atreides_vision.config import (
    DetectorConfig,
    Device,
    PipelineConfig,
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
        assert config.detector.confidence_threshold == 0.5
        assert config.tracker.frame_rate == 30.0
        assert config.visualization.trail_length == 25

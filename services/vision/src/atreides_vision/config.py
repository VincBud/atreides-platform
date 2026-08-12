"""Configuration objects and compute-device resolution.

Configuration is explicit, validated at construction, and passed down the call
chain. There is no module-level mutable state anywhere in Atreides Vision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from atreides_vision.errors import ConfigurationError, DeviceUnavailableError

logger = logging.getLogger(__name__)

__all__ = [
    "AcceleratorProbe",
    "DetectorConfig",
    "Device",
    "PipelineConfig",
    "TorchAcceleratorProbe",
    "TrackerConfig",
    "VisionConfig",
    "VisualizationConfig",
    "resolve_device",
]

#: COCO category id for "person" in the RF-DETR pretrained checkpoints.
#: Pretrained RF-DETR emits raw COCO category ids (1-90, sparse), not 0-based indices.
COCO_PERSON_CLASS_ID = 1


class Device(StrEnum):
    """Compute device requested by the operator."""

    AUTO = "auto"
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"


class AcceleratorProbe(Protocol):
    """Reports which accelerators this machine can actually use.

    Injected into :func:`resolve_device` so device policy can be tested without
    torch and without the hardware in question.
    """

    def cuda_available(self) -> bool: ...

    def mps_available(self) -> bool: ...


class TorchAcceleratorProbe:
    """Default probe, backed by torch. Imports torch lazily and defensively."""

    def cuda_available(self) -> bool:
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is a hard dependency in practice
            return False
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            logger.debug("CUDA availability probe raised; treating CUDA as unavailable")
            return False

    def mps_available(self) -> bool:
        try:
            import torch
        except ImportError:  # pragma: no cover
            return False
        try:
            return bool(torch.backends.mps.is_available())
        except Exception:
            logger.debug("MPS availability probe raised; treating MPS as unavailable")
            return False


def resolve_device(requested: Device, probe: AcceleratorProbe | None = None) -> str:
    """Resolve a requested :class:`Device` to a concrete torch device string.

    ``auto`` prefers CUDA, then MPS, then CPU. An explicitly requested
    accelerator that is unavailable raises :class:`DeviceUnavailableError` with
    an actionable message rather than failing later inside the model.

    Args:
        requested: The device the operator asked for.
        probe: Accelerator availability source. Defaults to a torch-backed probe.

    Returns:
        One of ``"cuda"``, ``"mps"`` or ``"cpu"``.

    Raises:
        DeviceUnavailableError: if a specific accelerator was requested and is
            not usable on this machine.
    """
    probe = probe or TorchAcceleratorProbe()

    if requested is Device.CPU:
        return "cpu"

    if requested is Device.CUDA:
        if not probe.cuda_available():
            raise DeviceUnavailableError(
                "CUDA was requested but no CUDA device is available on this machine. "
                "Re-run with --device cpu (always supported), --device mps on Apple "
                "Silicon, or --device auto to pick the best available backend."
            )
        return "cuda"

    if requested is Device.MPS:
        if not probe.mps_available():
            raise DeviceUnavailableError(
                "MPS (Apple Silicon GPU) was requested but is not available on this "
                "machine. Re-run with --device cpu (always supported) or --device auto "
                "to pick the best available backend."
            )
        return "mps"

    # Device.AUTO
    if probe.cuda_available():
        return "cuda"
    if probe.mps_available():
        return "mps"
    logger.info("No GPU accelerator detected; falling back to CPU.")
    return "cpu"


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Detector-agnostic settings. The RF-DETR adapter reads these."""

    confidence_threshold: float = 0.5
    device: str = "cpu"
    person_class_id: int = COCO_PERSON_CLASS_ID
    #: Optional inference resolution override; ``None`` keeps the model default.
    resolution: int | None = None
    #: Fuse/compile the model for inference where the backend supports it.
    optimize_for_inference: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ConfigurationError(
                f"confidence must be in (0, 1], got {self.confidence_threshold}"
            )
        if self.resolution is not None and self.resolution <= 0:
            raise ConfigurationError(f"resolution must be positive, got {self.resolution}")


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Tracker-agnostic settings. The ByteTrack adapter reads these."""

    #: Source frame rate, used by the tracker's motion model.
    frame_rate: float = 30.0
    #: How many frames a lost track is kept alive before being discarded.
    lost_track_buffer: int = 30
    #: Detections above this confidence can start a new track.
    track_activation_threshold: float = 0.5
    #: Frames a new track must persist before it is reported.
    minimum_consecutive_frames: int = 2
    #: Minimum IoU for a detection to be associated with an existing track.
    minimum_iou_threshold: float = 0.3

    def __post_init__(self) -> None:
        if self.frame_rate <= 0:
            raise ConfigurationError(f"frame_rate must be positive, got {self.frame_rate}")
        if self.lost_track_buffer < 0:
            raise ConfigurationError(
                f"lost_track_buffer must be non-negative, got {self.lost_track_buffer}"
            )
        if not 0.0 <= self.track_activation_threshold <= 1.0:
            raise ConfigurationError(
                "track_activation_threshold must be in [0, 1], "
                f"got {self.track_activation_threshold}"
            )
        if self.minimum_consecutive_frames < 1:
            raise ConfigurationError(
                f"minimum_consecutive_frames must be >= 1, got {self.minimum_consecutive_frames}"
            )


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    """Overlay appearance and trajectory length."""

    #: Number of recent centre points kept and drawn per track.
    trail_length: int = 25
    #: Frames a track may be absent before its trajectory is discarded.
    trail_ttl_frames: int = 45
    #: Hard ceiling on simultaneously retained trajectories (memory bound).
    max_trails: int = 512
    draw_confidence: bool = True
    box_thickness: int = 2
    trail_thickness: int = 2
    font_scale: float = 0.5

    def __post_init__(self) -> None:
        if self.trail_length < 2:
            raise ConfigurationError(f"trail_length must be >= 2, got {self.trail_length}")
        if self.trail_ttl_frames < 1:
            raise ConfigurationError(f"trail_ttl_frames must be >= 1, got {self.trail_ttl_frames}")
        if self.max_trails < 1:
            raise ConfigurationError(f"max_trails must be >= 1, got {self.max_trails}")


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Input/output paths and run-level behaviour."""

    input_path: Path
    output_path: Path
    #: Stop after this many frames. ``None`` processes the whole video.
    max_frames: int | None = None
    #: Emit a progress log line every N processed frames.
    log_every_n_frames: int = 100

    def __post_init__(self) -> None:
        if self.max_frames is not None and self.max_frames < 1:
            raise ConfigurationError(f"max_frames must be >= 1, got {self.max_frames}")
        if self.log_every_n_frames < 1:
            raise ConfigurationError(
                f"log_every_n_frames must be >= 1, got {self.log_every_n_frames}"
            )


@dataclass(frozen=True, slots=True)
class VisionConfig:
    """Complete configuration for one SEE + TRACK run."""

    pipeline: PipelineConfig
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

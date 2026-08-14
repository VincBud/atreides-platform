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

from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.errors import ConfigurationError, DeviceUnavailableError
from atreides_vision.zones.config import ZoneAnalysisConfig

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DEDUPLICATE_DETECTIONS",
    "DEFAULT_DETECTION_CONFIDENCE",
    "DEFAULT_OCSORT_HIGH_CONF_THRESHOLD",
    "DEFAULT_TRACKER",
    "AcceleratorProbe",
    "BehaviorAnalysisConfig",
    "DetectorConfig",
    "Device",
    "PipelineConfig",
    "TorchAcceleratorProbe",
    "Tracker",
    "TrackerConfig",
    "VisionConfig",
    "VisualizationConfig",
    "ZoneAnalysisConfig",
    "resolve_device",
]

#: COCO category id for "person" in the RF-DETR pretrained checkpoints.
#: Pretrained RF-DETR emits raw COCO category ids (1-90, sparse), not 0-based indices.
COCO_PERSON_CLASS_ID = 1

# The V0.2.1 validated candidate, named once. The dataclass defaults below and
# the CLI both read these, so `--help` can never drift from actual behaviour.
# `slots=True` hides dataclass defaults from class-attribute access, which is
# why these are module constants rather than something the CLI introspects.
#
# Validated on three fixed cameras: elevated dense daylight (1080p/30), night
# under strong artificial light (1080p/30), and a head-on corridor (384x288/25).
# See docs/vision/V0.2.1_FIXED_CCTV_VALIDATION.md. Development defaults, not a
# claim of production-wide optimality.

#: Detector confidence floor. See :class:`DetectorConfig`.
DEFAULT_DETECTION_CONFIDENCE: float = 0.30
#: Confidence above which OC-SORT may start a new track. See :class:`TrackerConfig`.
DEFAULT_OCSORT_HIGH_CONF_THRESHOLD: float = 0.30
#: Whether geometric detection deduplication runs. See :class:`PipelineConfig`.
DEFAULT_DEDUPLICATE_DETECTIONS: bool = True


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

    #: V0.2.1 validated default. Lowered from 0.50 after a three-scene fixed-CCTV
    #: validation: 0.50 dropped a large share of mid- and far-field people, and
    #: 0.30 recovered them without introducing false positives that survived
    #: manual inspection. The extra weak detections it admits are duplicate-prone,
    #: which is what :data:`~atreides_vision.detection.deduplication.RULE_B3` is for.
    confidence_threshold: float = DEFAULT_DETECTION_CONFIDENCE
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


class Tracker(StrEnum):
    """Multi-object tracker implementation to run.

    OC-SORT is the V0.2.1 default: it scores association on direction
    consistency as well as IoU, which held ids through short occlusions and
    through 3-4x box scale change where the ByteTrack baseline fragmented.
    ByteTrack remains fully supported and selectable for comparison.
    """

    BYTETRACK = "bytetrack"
    OCSORT = "ocsort"


#: The tracker a standard run uses. See :class:`Tracker`.
DEFAULT_TRACKER: Tracker = Tracker.OCSORT


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Settings shared by the tracker adapters.

    The first block applies to every tracker. The second is read only by
    OC-SORT, which has no equivalent of ``track_activation_threshold``.
    """

    #: Source frame rate, used by the tracker's motion model.
    frame_rate: float = 30.0
    #: How many frames a lost track is kept alive before being discarded.
    lost_track_buffer: int = 30
    #: Detections above this confidence can start a new track. ByteTrack only.
    track_activation_threshold: float = 0.5
    #: Frames a new track must persist before it is reported.
    minimum_consecutive_frames: int = 2
    #: Minimum IoU for a detection to be associated with an existing track.
    #: ByteTrack only — OC-SORT reads ``ocsort_minimum_iou_threshold`` so the two
    #: can be tuned independently without disturbing each other's baseline.
    minimum_iou_threshold: float = 0.3

    #: OC-SORT only. Minimum IoU for association. Kept separate from the
    #: ByteTrack field of the same meaning so the two can be tuned
    #: independently without contaminating each other's baseline.
    #: A 0.2 experiment reduced total ids but did not fix the occlusion case
    #: under investigation, so this is back at the production candidate value.
    ocsort_minimum_iou_threshold: float = 0.3
    #: OC-SORT only. How much direction consistency counts alongside IoU when
    #: associating a detection with a track. This is what lets OC-SORT re-attach
    #: a person who was briefly occluded rather than opening a new id.
    direction_consistency_weight: float = 0.2
    #: OC-SORT only. Confidence above which a detection may start a new track.
    #: Kept at or below the detector's own threshold so no detection that
    #: reaches the tracker is silently ineligible to become a track. V0.2.1
    #: moved this to 0.30 alongside the detector: left at 0.50 it would have
    #: discarded the very mid- and far-field people the lower detector
    #: threshold was adopted to recover.
    high_conf_det_threshold: float = DEFAULT_OCSORT_HIGH_CONF_THRESHOLD
    #: OC-SORT only. How many frames back the "last known observation" used for
    #: velocity direction is taken from.
    delta_t: int = 3

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
        if not 0.0 <= self.ocsort_minimum_iou_threshold <= 1.0:
            raise ConfigurationError(
                "ocsort_minimum_iou_threshold must be in [0, 1], "
                f"got {self.ocsort_minimum_iou_threshold}"
            )
        if not 0.0 <= self.direction_consistency_weight <= 1.0:
            raise ConfigurationError(
                "direction_consistency_weight must be in [0, 1], "
                f"got {self.direction_consistency_weight}"
            )
        if not 0.0 <= self.high_conf_det_threshold <= 1.0:
            raise ConfigurationError(
                f"high_conf_det_threshold must be in [0, 1], got {self.high_conf_det_threshold}"
            )
        if self.delta_t < 1:
            raise ConfigurationError(f"delta_t must be >= 1, got {self.delta_t}")


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
    #: Draw zone polygons and their metric panels when zones are configured.
    draw_zones: bool = True
    #: Opacity of the tint filling a zone polygon. 0 draws outlines only.
    zone_fill_alpha: float = 0.12
    zone_outline_thickness: int = 2
    #: Include cumulative entry/exit counts in each zone's panel.
    draw_zone_counters: bool = True

    def __post_init__(self) -> None:
        if self.trail_length < 2:
            raise ConfigurationError(f"trail_length must be >= 2, got {self.trail_length}")
        if self.trail_ttl_frames < 1:
            raise ConfigurationError(f"trail_ttl_frames must be >= 1, got {self.trail_ttl_frames}")
        if self.max_trails < 1:
            raise ConfigurationError(f"max_trails must be >= 1, got {self.max_trails}")
        if not 0.0 <= self.zone_fill_alpha <= 1.0:
            raise ConfigurationError(
                f"zone_fill_alpha must be in [0, 1], got {self.zone_fill_alpha}"
            )


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Input/output paths and run-level behaviour."""

    input_path: Path
    output_path: Path
    #: Stop after this many frames. ``None`` processes the whole video.
    max_frames: int | None = None
    #: Emit a progress log line every N processed frames.
    log_every_n_frames: int = 100
    #: V0.2.1 validated default: on. Suppress duplicate person detections
    #: geometrically before tracking, using the fixed conservative
    #: :data:`~atreides_vision.detection.deduplication.RULE_B3` policy. Across
    #: three validation scenes it removed 0.08-0.13% of detections, every
    #: manually inspected removal was two boxes on one person, and no case was
    #: found where it merged two genuine people. Disable with
    #: ``--no-deduplicate-detections`` to reproduce pre-V0.2.1 behaviour.
    deduplicate_detections: bool = DEFAULT_DEDUPLICATE_DETECTIONS

    def __post_init__(self) -> None:
        if self.max_frames is not None and self.max_frames < 1:
            raise ConfigurationError(f"max_frames must be >= 1, got {self.max_frames}")
        if self.log_every_n_frames < 1:
            raise ConfigurationError(
                f"log_every_n_frames must be >= 1, got {self.log_every_n_frames}"
            )


@dataclass(frozen=True, slots=True)
class VisionConfig:
    """Complete configuration for one run.

    ``zones`` tunes zone analysis but does not enable it: zone processing runs
    only when the pipeline is given a zone analyzer, which the CLI builds from
    ``--zones``. Without it the run is plain V0.1 SEE + TRACK.

    ``behavior`` works the same way and is likewise inert on its own: V0.3.0
    signals are produced only when the pipeline is given a
    :class:`~atreides_vision.behavior.analyzer.BehaviorAnalyzer`, which no CLI
    flag builds yet.
    """

    pipeline: PipelineConfig
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    zones: ZoneAnalysisConfig = field(default_factory=ZoneAnalysisConfig)
    behavior: BehaviorAnalysisConfig = field(default_factory=BehaviorAnalysisConfig)

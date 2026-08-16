"""Result types for the validation harness, and the shapes they serialise to.

Every metric type here is a **measurement of Atreides' own output**. None of
them is a verdict. There is deliberately no ``passed`` field, no grade, no
threshold and no acceptance band anywhere in this module: the evidence this
milestone gathers does not support one, and a boolean would be read as though it
did.

Each aggregate that could be mistaken for ground truth carries its limitation in
the type itself — see :class:`~atreides_vision.validation.track_metrics.TrackMetrics`
in particular — so a consumer reading the JSON summary sees the caveat next to
the number rather than in a document it may never open.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from atreides_vision.validation.behavior_metrics import BehaviorCoverageMetrics
from atreides_vision.validation.crowd_metrics import CrowdMetrics
from atreides_vision.validation.ground_truth import ManualGroundTruth
from atreides_vision.validation.interaction_metrics import InteractionMetrics
from atreides_vision.validation.manifest import ValidationClip
from atreides_vision.validation.track_metrics import TrackMetrics

__all__ = [
    "VALIDATION_SUMMARY_SCHEMA_VERSION",
    "ClipValidationResult",
    "OutputVolumeMetrics",
    "PerformanceMetrics",
    "RunProvenance",
    "StreamVolume",
    "ValidationSummary",
]

#: Version of the ``validation_summary.json`` contract. Independent of the
#: measurement schemas (behaviour 0.3.3, interaction 0.3.4, crowd 0.3.5): this
#: describes a report *about* those streams, and evolves for its own reasons.
VALIDATION_SUMMARY_SCHEMA_VERSION = "0.3.6"

_PRECISION = 4


def _round(value: float | None, digits: int = _PRECISION) -> float | None:
    return None if value is None else round(value, digits)


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """Exactly what produced one clip's outputs.

    Recorded per clip rather than once per run so a summary assembled from runs
    made at different times cannot silently compare results produced under
    different settings. Anything that would change the numbers belongs here.
    """

    detector: str
    detector_confidence: float
    tracker: str
    tracker_high_conf_threshold: float
    tracker_minimum_iou_threshold: float
    tracker_lost_track_buffer: int
    tracker_minimum_consecutive_frames: int
    tracker_direction_consistency_weight: float
    tracker_delta_t: int
    deduplicate_detections: bool
    deduplication_rule: str
    device: str
    coordinate_space: str
    distance_unit: str
    is_physical_space: bool
    camera_id: str
    input_path: str
    #: ``None`` for every clip without a measured calibration, which is all of
    #: them today. Present only when the manifest supplied one.
    calibration_path: str | None
    behavior_config: dict[str, Any]
    interaction_config: dict[str, Any]
    crowd_config: dict[str, Any]
    #: ``None`` means every analysed frame was written — the mode validation
    #: requires, because sampled output cannot support temporal metrics.
    output_sample_interval_seconds: float | None
    behavior_schema_version: str
    interaction_schema_version: str
    crowd_schema_version: str
    package_version: str
    #: True when the run used the frozen V0.2.1 detector/tracker defaults. False
    #: means the numbers are not comparable with the rest of the report.
    at_v021_defaults: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "detector_confidence": self.detector_confidence,
            "tracker": self.tracker,
            "tracker_high_conf_threshold": self.tracker_high_conf_threshold,
            "tracker_minimum_iou_threshold": self.tracker_minimum_iou_threshold,
            "tracker_lost_track_buffer": self.tracker_lost_track_buffer,
            "tracker_minimum_consecutive_frames": self.tracker_minimum_consecutive_frames,
            "tracker_direction_consistency_weight": self.tracker_direction_consistency_weight,
            "tracker_delta_t": self.tracker_delta_t,
            "deduplicate_detections": self.deduplicate_detections,
            "deduplication_rule": self.deduplication_rule,
            "device": self.device,
            "coordinate_space": self.coordinate_space,
            "distance_unit": self.distance_unit,
            "is_physical_space": self.is_physical_space,
            "camera_id": self.camera_id,
            "input_path": self.input_path,
            "calibration_path": self.calibration_path,
            "behavior_config": self.behavior_config,
            "interaction_config": self.interaction_config,
            "crowd_config": self.crowd_config,
            "output_sample_interval_seconds": self.output_sample_interval_seconds,
            "schema_versions": {
                "behavior": self.behavior_schema_version,
                "interaction": self.interaction_schema_version,
                "crowd": self.crowd_schema_version,
            },
            "package_version": self.package_version,
            "at_v021_defaults": self.at_v021_defaults,
        }


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Local wall-clock throughput for one clip.

    **Machine-specific development measurements.** They describe this laptop,
    this device backend, this build and this footage, with annotated-video
    encoding included in the loop. They do not establish production real-time
    capability, and comparing two of them is only meaningful when their
    :class:`RunProvenance` records agree.
    """

    source_frames: int | None
    source_fps: float
    source_duration_seconds: float | None
    frames_processed: int
    wall_clock_seconds: float
    device: str

    @property
    def processed_fps(self) -> float | None:
        if self.wall_clock_seconds <= 0:
            return None
        return self.frames_processed / self.wall_clock_seconds

    @property
    def realtime_factor(self) -> float | None:
        """``processed_fps / source_fps``. Above 1.0 means faster than playback."""
        processed = self.processed_fps
        if processed is None or self.source_fps <= 0:
            return None
        return processed / self.source_fps

    @property
    def processed_duration_seconds(self) -> float:
        """Source seconds actually processed — the denominator for every rate."""
        if self.source_fps <= 0:
            return 0.0
        return self.frames_processed / self.source_fps

    def to_record(self) -> dict[str, Any]:
        return {
            "source_frames": self.source_frames,
            "source_fps": _round(self.source_fps),
            "source_duration_seconds": _round(self.source_duration_seconds),
            "frames_processed": self.frames_processed,
            "processed_duration_seconds": _round(self.processed_duration_seconds),
            "wall_clock_seconds": _round(self.wall_clock_seconds),
            "processed_fps": _round(self.processed_fps),
            "realtime_factor": _round(self.realtime_factor),
            "device": self.device,
            "measurement_scope": (
                "machine-specific development measurement, including annotated-video "
                "encoding; not a production real-time capability claim"
            ),
        }


@dataclass(frozen=True, slots=True)
class StreamVolume:
    """What one output stream cost, and how many records it holds."""

    name: str
    path: str
    compressed: bool
    size_bytes: int
    #: From the gzip trailer where it can be read cheaply; ``None`` otherwise.
    uncompressed_size_bytes: int | None
    record_count: int

    def records_per_source_second(self, processed_duration_seconds: float) -> float | None:
        if processed_duration_seconds <= 0:
            return None
        return self.record_count / processed_duration_seconds

    def to_record(self, processed_duration_seconds: float) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "compressed": self.compressed,
            "size_bytes": self.size_bytes,
            "size_mib": _round(self.size_bytes / (1024 * 1024)),
            "uncompressed_size_bytes": self.uncompressed_size_bytes,
            "record_count": self.record_count,
            "records_per_source_second": _round(
                self.records_per_source_second(processed_duration_seconds)
            ),
        }


@dataclass(frozen=True, slots=True)
class OutputVolumeMetrics:
    """Total structured-output cost for one clip.

    The rendered MP4 is reported separately and excluded from the structured
    total: it is a review artifact, not part of the machine-readable contract a
    future consumer would store.
    """

    streams: tuple[StreamVolume, ...]
    rendered_video_bytes: int | None = None

    @property
    def total_bytes(self) -> int:
        return sum(stream.size_bytes for stream in self.streams)

    def megabytes_per_source_minute(self, processed_duration_seconds: float) -> float | None:
        if processed_duration_seconds <= 0:
            return None
        minutes = processed_duration_seconds / 60.0
        return (self.total_bytes / (1024 * 1024)) / minutes

    def to_record(self, processed_duration_seconds: float) -> dict[str, Any]:
        return {
            "streams": [stream.to_record(processed_duration_seconds) for stream in self.streams],
            "total_structured_bytes": self.total_bytes,
            "total_structured_mib": _round(self.total_bytes / (1024 * 1024)),
            "structured_mib_per_source_minute": _round(
                self.megabytes_per_source_minute(processed_duration_seconds)
            ),
            "rendered_video_bytes": self.rendered_video_bytes,
            "note": ("sizes are as stored; structured totals exclude the rendered review video"),
        }


@dataclass(frozen=True, slots=True)
class ClipValidationResult:
    """Everything measured for one clip.

    Proxy metrics and manual ground truth are separate fields, never merged into
    one bag, so no serialisation of this type can present a proxy as though a
    human had confirmed it.
    """

    clip: ValidationClip
    provenance: RunProvenance
    performance: PerformanceMetrics
    tracks: TrackMetrics
    behavior: BehaviorCoverageMetrics
    interaction: InteractionMetrics
    crowd: CrowdMetrics
    output_volume: OutputVolumeMetrics
    ground_truth: ManualGroundTruth
    #: Paths to the deterministic review sample, relative to the artifact root.
    review_sample: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        processed = self.performance.processed_duration_seconds
        return {
            "clip": self.clip.to_record(),
            "provenance": self.provenance.to_record(),
            "performance": self.performance.to_record(),
            "proxy_metrics": {
                "measurement_class": "automated_proxy_system_metrics",
                "derived_from": "atreides v0.3 output streams, no human annotation",
                "tracking": self.tracks.to_record(),
                "behavior_coverage": self.behavior.to_record(),
                "interaction": self.interaction.to_record(),
                "crowd": self.crowd.to_record(),
            },
            "output_volume": self.output_volume.to_record(processed),
            "manual_ground_truth": self.ground_truth.to_record(),
            "review_sample": self.review_sample,
        }


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Every clip's result, plus what is true of the run as a whole."""

    results: tuple[ClipValidationResult, ...]
    manifest_path: str | None = None
    generated_by: str = "atreides-vision validation"

    @property
    def all_at_v021_defaults(self) -> bool:
        """Whether every clip ran on the frozen defaults, and so is comparable."""
        return all(result.provenance.at_v021_defaults for result in self.results)

    @property
    def any_physical_space(self) -> bool:
        """Whether any clip produced physically calibrated measurements."""
        return any(result.provenance.is_physical_space for result in self.results)

    @property
    def shared_camera_ids(self) -> tuple[str, ...]:
        """Camera ids contributing more than one clip to this run, sorted.

        Expected and permitted: one physical camera supplies many independent
        clips. Surfaced because it changes what a reader may do with the
        numbers — per-clip counts from these clips must not be added together,
        since nothing relates their track ids to each other.
        """
        counts = Counter(result.clip.camera_id for result in self.results)
        return tuple(sorted(camera for camera, count in counts.items() if count > 1))

    def clip_ids_for_camera(self, camera_id: str) -> tuple[str, ...]:
        """Clip ids captured by one camera, in results order. Grouping only."""
        return tuple(
            result.clip.clip_id for result in self.results if result.clip.camera_id == camera_id
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": VALIDATION_SUMMARY_SCHEMA_VERSION,
            "generated_by": self.generated_by,
            "manifest_path": self.manifest_path,
            "milestone": "V0.3.6 real-world validation & signal stack freeze",
            "measurement_classes": {
                "automated_proxy_system_metrics": (
                    "derived from Atreides' own output streams. Not detection recall, not an "
                    "ID-switch rate, and not proof that any specific tracking failure occurred."
                ),
                "manual_ground_truth": (
                    "derived from human annotation only. Absent unless a human filled in the "
                    "annotation templates for that clip."
                ),
            },
            "physical_units": {
                "any_clip_physically_calibrated": self.any_physical_space,
                "note": (
                    "clips without a measured calibration are reported in normalized image "
                    "units. No metre, m/s or subjects/m^2 figure is derived for them."
                ),
            },
            "comparability": {
                "all_clips_at_v021_defaults": self.all_at_v021_defaults,
                "note": (
                    "cross-clip comparison is only meaningful when every clip ran under the "
                    "same detector/tracker configuration."
                ),
            },
            "clip_isolation": {
                "cameras_with_multiple_clips": {
                    camera: list(self.clip_ids_for_camera(camera))
                    for camera in self.shared_camera_ids
                },
                "note": (
                    "several clips may share a camera_id: one physical camera supplies many "
                    "independent clips. Sharing one asserts only that the footage came from "
                    "the same camera. Every clip is processed by its own detector, tracker "
                    "and analyzers, written to its own streams, and measured into its own "
                    "result; no track id, history or metric is carried across a clip "
                    "boundary. Per-clip counts from clips sharing a camera_id must therefore "
                    "not be summed."
                ),
            },
            "clips": [result.to_record() for result in self.results],
        }

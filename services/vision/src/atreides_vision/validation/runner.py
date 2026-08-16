"""Executing the V0.3 stack over a manifest of clips, and recording exactly how.

The runner is an **orchestrator**. It builds the same components the CLI's
``run`` command builds, in the same way, and measures the result. It contains no
measurement logic of its own: if it did, validation would be measuring the
harness rather than the pipeline.

Three deliberate differences from an operational run, each of which exists
because the metrics require it:

**Nothing is sampled.** Every analysed frame is written to every stream. Track
lifetimes, observation gaps, pair episodes and composition persistence are all
temporal, and a sampled stream cannot support any of them — a track present on
frames 100-130 and sampled once a second appears as a single observation with a
lifetime of zero.

**All three streams are always on.** Behaviour, interaction and crowd output are
independent flags operationally; validation needs all three from one pass so the
cross-stream metrics (pair episodes attributed to member-track endings) describe
one run rather than three.

**Output is gzipped.** Unsampled 1080p output is large. Compression changes no
number in the stream and the reader handles it transparently.

Nothing else changes. The detector, tracker, deduplication and every threshold
are the V0.2.1 validated defaults, and the runner refuses to be the place
somebody quietly tunes one: it takes no threshold arguments at all.

Testability
-----------
The detector and tracker arrive through factories so the whole runner can be
exercised with fakes, on a synthetic video, with no model weights and no
network. That is the same property the pipeline's own adapters exist to provide.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from atreides_vision import __version__
from atreides_vision.behavior.analyzer import BehaviorAnalyzer
from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.behavior.coordinates import NormalizedFrameProjector
from atreides_vision.behavior.reporting import (
    BEHAVIOR_SCHEMA_VERSION,
    BehaviorRunMetadata,
    CalibrationProvenance,
    EmitEveryFrame,
    JsonlBehaviorWriter,
    SourceProvenance,
)
from atreides_vision.calibration.fitting import CalibrationFit
from atreides_vision.calibration.loader import load_calibration_fit
from atreides_vision.calibration.projector import GroundPlaneProjector
from atreides_vision.config import (
    DEFAULT_DEDUPLICATE_DETECTIONS,
    DEFAULT_DETECTION_CONFIDENCE,
    DEFAULT_OCSORT_HIGH_CONF_THRESHOLD,
    DEFAULT_TRACKER,
    DetectorConfig,
    PipelineConfig,
    TrackerConfig,
    VisionConfig,
    VisualizationConfig,
)
from atreides_vision.crowd.analyzer import SpatialClusterAnalyzer
from atreides_vision.crowd.config import CrowdAnalysisConfig
from atreides_vision.crowd.reporting import CROWD_SCHEMA_VERSION, CrowdRunMetadata, JsonlCrowdWriter
from atreides_vision.detection.base import PersonDetector
from atreides_vision.errors import AtreidesVisionError, ValidationError
from atreides_vision.interaction.analyzer import InteractionAnalyzer
from atreides_vision.interaction.config import InteractionAnalysisConfig
from atreides_vision.interaction.models import SpatialContext
from atreides_vision.interaction.reporting import (
    INTERACTION_SCHEMA_VERSION,
    InteractionRunMetadata,
    JsonlInteractionWriter,
)
from atreides_vision.tracking.base import MultiObjectTracker
from atreides_vision.validation.manifest import ValidationClip, ValidationManifest
from atreides_vision.validation.models import PerformanceMetrics, RunProvenance
from atreides_vision.validation.review_samples import (
    DEFAULT_CONTINUITY_WINDOW_SECONDS,
    DEFAULT_CONTINUITY_WINDOWS,
    DEFAULT_SAMPLE_FRAMES,
    ReviewSamplePlan,
    build_sample_plan,
    extract_sample_frames,
    write_contact_sheet,
)
from atreides_vision.validation.storage import (
    ValidationLayout,
    write_json,
)
from atreides_vision.video.processor import VideoProcessor
from atreides_vision.video.sink import Mp4VideoSink
from atreides_vision.video.source import FileVideoSource, VideoMetadata

logger = logging.getLogger(__name__)

__all__ = [
    "BEHAVIOR_STREAM_NAME",
    "CROWD_STREAM_NAME",
    "INTERACTION_STREAM_NAME",
    "ClipRunOutcome",
    "DetectorFactory",
    "TrackerFactory",
    "ValidationRunner",
    "default_detector_factory",
    "default_tracker_factory",
]

BEHAVIOR_STREAM_NAME = "behavior"
INTERACTION_STREAM_NAME = "interaction"
CROWD_STREAM_NAME = "crowd"

#: The deduplication policy in effect. Named rather than described numerically
#: so a provenance record cannot drift from the policy the pipeline applies.
_DEDUPLICATION_RULE = "RULE_B3"


class DetectorFactory(Protocol):
    """Builds the detector for one run. Injected so tests need no weights."""

    def __call__(self, config: DetectorConfig) -> PersonDetector: ...


class TrackerFactory(Protocol):
    """Builds the tracker for one run."""

    def __call__(self, config: TrackerConfig) -> MultiObjectTracker: ...


def default_detector_factory(config: DetectorConfig) -> PersonDetector:
    """RF-DETR Nano, loaded. Imported lazily so tests never touch the weights."""
    from atreides_vision.detection.rfdetr import RFDetrPersonDetector

    detector = RFDetrPersonDetector(config)
    detector.load()
    return detector


def default_tracker_factory(config: TrackerConfig) -> MultiObjectTracker:
    """OC-SORT, the V0.2.1 validated tracker."""
    from atreides_vision.tracking.ocsort import OcSortTracker

    return OcSortTracker(config)


@dataclass(frozen=True, slots=True)
class ClipRunOutcome:
    """What one clip's processing produced, and under what configuration."""

    clip: ValidationClip
    provenance: RunProvenance
    performance: PerformanceMetrics
    rendered_video_path: Path
    behavior_path: Path
    interaction_path: Path
    crowd_path: Path
    sample_plan: ReviewSamplePlan
    unique_track_ids: int
    detections_removed_by_deduplication: int

    def to_record(self) -> dict[str, Any]:
        return {
            "clip": self.clip.to_record(),
            "provenance": self.provenance.to_record(),
            "performance": self.performance.to_record(),
            "artifacts": {
                "rendered_video": self.rendered_video_path.name,
                "behavior_stream": self.behavior_path.name,
                "interaction_stream": self.interaction_path.name,
                "crowd_stream": self.crowd_path.name,
            },
            "pipeline_counters": {
                "unique_track_ids": self.unique_track_ids,
                "detections_removed_by_deduplication": (self.detections_removed_by_deduplication),
            },
            "sample_plan": self.sample_plan.to_record(),
        }


class ValidationRunner:
    """Processes each clip in a manifest and writes its run artifacts.

    Args:
        layout: Where artifacts go.
        device: Resolved torch device string, recorded in provenance.
        detector_factory: Builds the detector. ``None`` resolves to RF-DETR Nano
            at construction, via the module-level default, so a test can replace
            that default without reaching into this signature.
        tracker_factory: Builds the tracker. ``None`` resolves to OC-SORT.
        max_frames: Optional per-clip frame cap, for a fast smoke run. Recorded
            in the performance record so a capped run is never mistaken for a
            full one.
        clock: Monotonic clock, injectable so timing is deterministic in tests.
    """

    def __init__(
        self,
        layout: ValidationLayout,
        device: str = "cpu",
        detector_factory: DetectorFactory | None = None,
        tracker_factory: TrackerFactory | None = None,
        max_frames: int | None = None,
        sample_frames: int = DEFAULT_SAMPLE_FRAMES,
        continuity_windows: int = DEFAULT_CONTINUITY_WINDOWS,
        continuity_window_seconds: float = DEFAULT_CONTINUITY_WINDOW_SECONDS,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._layout = layout
        self._device = device
        # Resolved here rather than as a default argument value, so the module
        # attribute is read at construction and can be replaced under test.
        self._detector_factory = detector_factory or default_detector_factory
        self._tracker_factory = tracker_factory or default_tracker_factory
        self._max_frames = max_frames
        self._sample_frames = sample_frames
        self._continuity_windows = continuity_windows
        self._continuity_window_seconds = continuity_window_seconds
        self._clock = clock

    def run_manifest(self, manifest: ValidationManifest) -> list[ClipRunOutcome]:
        """Process every clip, in manifest order.

        A clip that fails stops the run. A partial validation report presented
        as a complete one is the failure mode worth preventing: the summary and
        the comparison table both read as "these are the scenes", and a silently
        missing scene would change what the report appears to say.
        """
        outcomes: list[ClipRunOutcome] = []
        for position, clip in enumerate(manifest, start=1):
            logger.info(
                "[%d/%d] %s (%s) -> %s",
                position,
                len(manifest),
                clip.clip_id,
                clip.scene_type,
                clip.input_path,
            )
            outcomes.append(self.run_clip(clip))
        return outcomes

    def run_clip(self, clip: ValidationClip) -> ClipRunOutcome:
        """Process one clip and write its run artifacts.

        Raises:
            ValidationError: wrapping any pipeline failure, naming the clip. A
                bare pipeline error twenty minutes into a four-clip run does not
                say which clip produced it.
        """
        run_dir = self._layout.run_dir(clip.clip_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        calibration_fit = (
            None if clip.calibration_path is None else load_calibration_fit(clip.calibration_path)
        )
        if calibration_fit is not None and calibration_fit.camera_id != clip.camera_id:
            raise ValidationError(
                f"Clip '{clip.clip_id}': the manifest says camera '{clip.camera_id}' but its "
                f"calibration says '{calibration_fit.camera_id}'. One of them describes "
                "different footage; correct the mismatch rather than letting one win."
            )

        try:
            outcome = self._process(clip, run_dir, calibration_fit)
        except AtreidesVisionError as exc:
            raise ValidationError(f"Clip '{clip.clip_id}' failed to process: {exc}") from exc

        self._write_review_artifacts(clip, outcome)
        write_json(run_dir / "run.json", outcome.to_record())
        return outcome

    def _process(
        self, clip: ValidationClip, run_dir: Path, calibration_fit: CalibrationFit | None
    ) -> ClipRunOutcome:
        rendered_path = run_dir / "tracked.mp4"
        behavior_path = run_dir / "behavior.jsonl.gz"
        interaction_path = run_dir / "interaction.jsonl.gz"
        crowd_path = run_dir / "crowd.jsonl.gz"

        with FileVideoSource(clip.input_path) as source:
            metadata = source.metadata
            config = self._build_config(clip, rendered_path, metadata)

            behavior_analyzer = _build_behavior_analyzer(clip.camera_id, metadata, calibration_fit)
            interaction_analyzer = _build_interaction_analyzer(
                clip.camera_id, behavior_analyzer, calibration_fit
            )
            crowd_analyzer = _build_crowd_analyzer(
                clip.camera_id, behavior_analyzer, calibration_fit
            )

            detector = self._detector_factory(config.detector)
            tracker = self._tracker_factory(config.tracker)

            with ExitStack() as stack:
                sink = stack.enter_context(
                    Mp4VideoSink(
                        path=rendered_path,
                        width=metadata.width,
                        height=metadata.height,
                        fps=metadata.fps,
                    )
                )
                behavior_writer = stack.enter_context(
                    JsonlBehaviorWriter(
                        behavior_path,
                        _behavior_metadata(clip, behavior_analyzer, metadata, calibration_fit),
                        # Unsampled, always: every temporal metric in this
                        # harness needs a record per analysed frame.
                        emission_policy=EmitEveryFrame(),
                    )
                )
                interaction_writer = stack.enter_context(
                    JsonlInteractionWriter(
                        interaction_path,
                        _interaction_metadata(interaction_analyzer),
                        emission_policy=EmitEveryFrame(),
                    )
                )
                crowd_writer = stack.enter_context(
                    JsonlCrowdWriter(
                        crowd_path,
                        _crowd_metadata(crowd_analyzer),
                        emission_policy=EmitEveryFrame(),
                    )
                )

                processor = VideoProcessor(
                    detector=detector,
                    tracker=tracker,
                    config=config,
                    clock=self._clock,
                    behavior_analyzer=behavior_analyzer,
                    behavior_writer=behavior_writer,
                    interaction_analyzer=interaction_analyzer,
                    interaction_writer=interaction_writer,
                    crowd_analyzer=crowd_analyzer,
                    crowd_writer=crowd_writer,
                )
                started = self._clock()
                result = processor.run(source, sink)
                wall_clock = self._clock() - started

        performance = PerformanceMetrics(
            source_frames=metadata.frame_count,
            source_fps=metadata.fps,
            source_duration_seconds=metadata.duration_seconds,
            frames_processed=result.frames_processed,
            wall_clock_seconds=wall_clock,
            device=self._device,
        )
        sample_plan = build_sample_plan(
            clip_id=clip.clip_id,
            frames_processed=result.frames_processed,
            source_fps=metadata.fps,
            sample_frames=self._sample_frames,
            continuity_windows=self._continuity_windows,
            window_seconds=self._continuity_window_seconds,
        )

        return ClipRunOutcome(
            clip=clip,
            provenance=_provenance(
                clip,
                config,
                behavior_analyzer,
                interaction_analyzer,
                crowd_analyzer,
                self._device,
            ),
            performance=performance,
            rendered_video_path=rendered_path,
            behavior_path=behavior_path,
            interaction_path=interaction_path,
            crowd_path=crowd_path,
            sample_plan=sample_plan,
            unique_track_ids=result.unique_track_ids,
            detections_removed_by_deduplication=result.detections_removed_by_deduplication,
        )

    def _build_config(
        self, clip: ValidationClip, rendered_path: Path, metadata: VideoMetadata
    ) -> VisionConfig:
        """Frozen V0.2.1 defaults, with only the source frame rate varying.

        There is deliberately no path by which a caller can change a detector or
        tracker threshold here. Tuning is out of scope for this milestone, and a
        harness that accepted overrides would eventually be used for one.
        """
        return VisionConfig(
            pipeline=PipelineConfig(
                input_path=clip.input_path,
                output_path=rendered_path,
                max_frames=self._max_frames,
                deduplicate_detections=DEFAULT_DEDUPLICATE_DETECTIONS,
            ),
            detector=DetectorConfig(
                confidence_threshold=DEFAULT_DETECTION_CONFIDENCE, device=self._device
            ),
            tracker=TrackerConfig(frame_rate=metadata.fps),
            visualization=VisualizationConfig(),
        )

    def _write_review_artifacts(self, clip: ValidationClip, outcome: ClipRunOutcome) -> None:
        """Extract the deterministic stills and tile them, next to the templates.

        Annotation templates themselves are written by the analysis pass, which
        is also what reads them back — keeping the two together means the rows
        an annotator sees always match the plan the report expects.
        """
        gt_dir = self._layout.ground_truth_dir(clip.clip_id)
        frames_dir = gt_dir / "frames"
        write_json(gt_dir / "sample_plan.json", outcome.sample_plan.to_record())

        frame_paths = extract_sample_frames(
            outcome.rendered_video_path, outcome.sample_plan, frames_dir
        )
        write_contact_sheet(frame_paths, gt_dir / "contact_sheet.jpg", outcome.sample_plan)
        logger.info("  review sample: %d frame(s) -> %s", len(frame_paths), frames_dir)


def _build_behavior_analyzer(
    camera_id: str, metadata: VideoMetadata, calibration_fit: CalibrationFit | None
) -> BehaviorAnalyzer:
    """The same pairing rule the CLI applies: projector and thresholds together.

    Without a measured calibration this is normalized image space, which is the
    case for every current validation clip — and the reason no physical unit
    appears anywhere in their reports.
    """
    if calibration_fit is None:
        return BehaviorAnalyzer(
            camera_id=camera_id,
            projector=NormalizedFrameProjector(metadata.width, metadata.height),
            config=BehaviorAnalysisConfig(),
        )
    return BehaviorAnalyzer(
        camera_id=camera_id,
        projector=GroundPlaneProjector(
            calibration_fit, frame_width=metadata.width, frame_height=metadata.height
        ),
        config=BehaviorAnalysisConfig.for_ground_plane_metres(),
    )


def _spatial_context(
    camera_id: str, analyzer: BehaviorAnalyzer, calibration_fit: CalibrationFit | None
) -> SpatialContext:
    if calibration_fit is None:
        return SpatialContext(camera_id=camera_id, space=analyzer.space)
    calibration = calibration_fit.calibration
    return SpatialContext(
        camera_id=camera_id,
        space=analyzer.space,
        venue_id=calibration.venue_id,
        surface_id=calibration.surface_id,
    )


def _build_interaction_analyzer(
    camera_id: str, analyzer: BehaviorAnalyzer, calibration_fit: CalibrationFit | None
) -> InteractionAnalyzer:
    config = (
        InteractionAnalysisConfig.for_normalized_image()
        if calibration_fit is None
        else InteractionAnalysisConfig.for_ground_plane_metres()
    )
    return InteractionAnalyzer(_spatial_context(camera_id, analyzer, calibration_fit), config)


def _build_crowd_analyzer(
    camera_id: str, analyzer: BehaviorAnalyzer, calibration_fit: CalibrationFit | None
) -> SpatialClusterAnalyzer:
    config = (
        CrowdAnalysisConfig.for_normalized_image()
        if calibration_fit is None
        else CrowdAnalysisConfig.for_ground_plane_metres()
    )
    return SpatialClusterAnalyzer(_spatial_context(camera_id, analyzer, calibration_fit), config)


def _behavior_metadata(
    clip: ValidationClip,
    analyzer: BehaviorAnalyzer,
    metadata: VideoMetadata,
    calibration_fit: CalibrationFit | None,
) -> BehaviorRunMetadata:
    provenance = None
    if calibration_fit is not None:
        quality = calibration_fit.quality
        calibration = calibration_fit.calibration
        provenance = CalibrationProvenance(
            venue_id=calibration.venue_id,
            surface_id=calibration.surface_id,
            reference_width=calibration.reference_frame.width,
            reference_height=calibration.reference_frame.height,
            correspondence_count=quality.point_count,
            mean_venue_error_m=quality.mean_venue_error_m,
            max_venue_error_m=quality.max_venue_error_m,
            mean_image_error_px=quality.mean_image_error_px,
            max_image_error_px=quality.max_image_error_px,
            relative_error=quality.relative_error,
            venue_extent_m=quality.venue_extent_m,
        )
    return BehaviorRunMetadata(
        camera_id=clip.camera_id,
        space=analyzer.space,
        source=SourceProvenance(
            path=str(clip.input_path),
            width=metadata.width,
            height=metadata.height,
            fps=metadata.fps,
        ),
        calibration=provenance,
    )


def _interaction_metadata(analyzer: InteractionAnalyzer) -> InteractionRunMetadata:
    context = analyzer.context
    return InteractionRunMetadata(
        camera_id=context.camera_id,
        space=context.space,
        venue_id=context.venue_id,
        surface_id=context.surface_id,
        proximity_threshold=analyzer.config.proximity_threshold,
    )


def _crowd_metadata(analyzer: SpatialClusterAnalyzer) -> CrowdRunMetadata:
    context = analyzer.context
    config = analyzer.config
    return CrowdRunMetadata(
        camera_id=context.camera_id,
        space=context.space,
        venue_id=context.venue_id,
        surface_id=context.surface_id,
        cluster_radius=config.cluster_radius,
        density_radius=config.density_radius,
        minimum_cluster_size=config.minimum_cluster_size,
    )


def _provenance(
    clip: ValidationClip,
    config: VisionConfig,
    behavior: BehaviorAnalyzer,
    interaction: InteractionAnalyzer,
    crowd: SpatialClusterAnalyzer,
    device: str,
) -> RunProvenance:
    """Every setting that could change a number in this clip's report.

    Read off the constructed objects rather than off the intended values, so a
    provenance record cannot claim a configuration the run did not actually use.
    """
    tracker = config.tracker
    behavior_config = behavior.config
    interaction_config = interaction.config
    crowd_config = crowd.config

    at_defaults = (
        config.detector.confidence_threshold == DEFAULT_DETECTION_CONFIDENCE
        and tracker.high_conf_det_threshold == DEFAULT_OCSORT_HIGH_CONF_THRESHOLD
        and config.pipeline.deduplicate_detections is DEFAULT_DEDUPLICATE_DETECTIONS
    )

    return RunProvenance(
        detector="RF-DETR Nano",
        detector_confidence=config.detector.confidence_threshold,
        tracker=str(DEFAULT_TRACKER.value),
        tracker_high_conf_threshold=tracker.high_conf_det_threshold,
        tracker_minimum_iou_threshold=tracker.ocsort_minimum_iou_threshold,
        tracker_lost_track_buffer=tracker.lost_track_buffer,
        tracker_minimum_consecutive_frames=tracker.minimum_consecutive_frames,
        tracker_direction_consistency_weight=tracker.direction_consistency_weight,
        tracker_delta_t=tracker.delta_t,
        deduplicate_detections=config.pipeline.deduplicate_detections,
        deduplication_rule=(
            _DEDUPLICATION_RULE if config.pipeline.deduplicate_detections else "disabled"
        ),
        device=device,
        coordinate_space=behavior.space.value,
        distance_unit=behavior.space.distance_unit,
        is_physical_space=behavior.space.is_physical,
        camera_id=clip.camera_id,
        input_path=str(clip.input_path),
        calibration_path=(None if clip.calibration_path is None else str(clip.calibration_path)),
        behavior_config={
            "speed_window_seconds": behavior_config.speed_window_seconds,
            "acceleration_window_seconds": behavior_config.acceleration_window_seconds,
            "heading_window_seconds": behavior_config.heading_window_seconds,
            "min_heading_displacement": behavior_config.min_heading_displacement,
            "stationary_speed_threshold": behavior_config.stationary_speed_threshold,
            "max_observation_gap_seconds": behavior_config.max_observation_gap_seconds,
            "history_window_seconds": behavior_config.history_window_seconds,
        },
        interaction_config={
            "closing_speed_window_seconds": interaction_config.closing_speed_window_seconds,
            "proximity_threshold": interaction_config.proximity_threshold,
            "max_pair_distance": interaction_config.max_pair_distance,
            "min_heading_displacement": interaction_config.min_heading_displacement,
            "max_observation_gap_seconds": interaction_config.max_observation_gap_seconds,
            "history_window_seconds": interaction_config.history_window_seconds,
        },
        crowd_config={
            "cluster_radius": crowd_config.cluster_radius,
            "density_radius": crowd_config.density_radius,
            "minimum_cluster_size": crowd_config.minimum_cluster_size,
            "minimum_moving_members": crowd_config.minimum_moving_members,
            "min_motion_displacement": crowd_config.min_motion_displacement,
            "max_observation_gap_seconds": crowd_config.max_observation_gap_seconds,
        },
        output_sample_interval_seconds=None,
        behavior_schema_version=BEHAVIOR_SCHEMA_VERSION,
        interaction_schema_version=INTERACTION_SCHEMA_VERSION,
        crowd_schema_version=CROWD_SCHEMA_VERSION,
        package_version=__version__,
        at_v021_defaults=at_defaults,
    )


def iter_stream_names() -> Iterator[str]:
    """The structured streams a validation run produces, in report order."""
    yield BEHAVIOR_STREAM_NAME
    yield INTERACTION_STREAM_NAME
    yield CROWD_STREAM_NAME


def stream_paths(run_dir: Path) -> Sequence[tuple[str, Path]]:
    """The structured stream files for one clip's run directory."""
    return (
        (BEHAVIOR_STREAM_NAME, run_dir / "behavior.jsonl.gz"),
        (INTERACTION_STREAM_NAME, run_dir / "interaction.jsonl.gz"),
        (CROWD_STREAM_NAME, run_dir / "crowd.jsonl.gz"),
    )

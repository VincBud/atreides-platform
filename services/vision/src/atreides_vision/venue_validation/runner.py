"""The end-to-end venue validation runner — V0.4.8.

Orchestration, and only orchestration. Every measurement below belongs to a
layer that already owns it::

    manifest + sources        -> V0.4.0 venue ingestion (timebase, cursors, session)
    per-camera frames         -> V0.1-V0.3 VideoProcessor (detect, track, behaviour)
    calibration + integrity   -> V0.3.1 / V0.3.3, injected per camera
    snapshot + frame          -> V0.4.7 build_track_observation()  (the only seam)
    observations              -> V0.4.4 VenuePipeline.process_instant()
    instant result + registry -> V0.4.6 compute_occupancy()
    records                   -> V0.4.5 and V0.4.6 writers
    counts                    -> this module

Nothing here detects, projects, associates, fuses, counts occupancy or invents
a coordinate. If a value is not readable from the runtime above, it is absent
from the summary rather than estimated.

One composition-root detail carries the whole design: ``VideoProcessor.process``
drives the injected :class:`~atreides_vision.behavior.analyzer.BehaviorAnalyzer`
synchronously, so immediately after a frame is processed
:attr:`BehaviorAnalyzer.latest_frame` *is* that frame's snapshots. Because this
module also constructed the projector and integrity guard it injected, all four
inputs :func:`build_track_observation` requires are in hand without a single
change to the camera-local layers.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from atreides_vision.association.models import AssociationOutcome, TrackObservation
from atreides_vision.behavior.analyzer import BehaviorAnalyzer
from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.behavior.coordinates import NormalizedFrameProjector
from atreides_vision.calibration.fitting import CalibrationFit
from atreides_vision.calibration.integrity import (
    CalibrationIntegrityConfig,
    CalibrationIntegrityGuard,
    CalibrationIntegrityMonitor,
    IntegrityPolicyMode,
)
from atreides_vision.calibration.loader import load_calibration_fit
from atreides_vision.calibration.projector import GroundPlaneProjector
from atreides_vision.config import (
    PipelineConfig,
    TrackerConfig,
    VisionConfig,
)
from atreides_vision.detection.base import PersonDetector
from atreides_vision.fusion.models import IntegrationOutcome
from atreides_vision.occupancy.analyzer import compute_occupancy
from atreides_vision.occupancy.reporting import (
    JsonlOccupancyReportWriter,
    OccupancyReportMetadata,
)
from atreides_vision.timing import StreamFrame, StreamRef
from atreides_vision.track_observation_adapter import build_track_observation
from atreides_vision.tracking.base import MultiObjectTracker
from atreides_vision.venue.manifest import VenueStreamEntry
from atreides_vision.venue.models import stream_sort_key
from atreides_vision.venue.session import VenueSession, VideoSourceCursor
from atreides_vision.venue.timebase import VenueTimebase
from atreides_vision.venue_pipeline import VenuePipeline
from atreides_vision.venue_pipeline_reporting import (
    JsonlVenuePipelineReportWriter,
    VenuePipelineReportMetadata,
)
from atreides_vision.venue_validation.config import (
    VenueValidationCameraConfig,
    VenueValidationConfig,
)
from atreides_vision.venue_validation.models import (
    VenueValidationStreamSummary,
    VenueValidationSummary,
    stream_summary_sort_key,
)
from atreides_vision.venue_validation.reporting import write_validation_summary
from atreides_vision.video.processor import VideoProcessor
from atreides_vision.video.source import (
    FileVideoSource,
    VideoMetadata,
    VideoSource,
    read_reference_frame,
)

logger = logging.getLogger(__name__)

__all__ = ["open_stream_source", "run_venue_validation"]

#: Annotated frames are produced by the existing processor and discarded here:
#: this harness validates the measurement path, not the rendering. The path is
#: never opened, and never reaches an artifact.
_DISCARDED_OUTPUT = Path(os.devnull)


def open_stream_source(entry: VenueStreamEntry) -> VideoSource:
    """Open one manifest stream as a real video file.

    The default source factory. It is a parameter of the runner for the same
    reason the detector and tracker are: a test proving the venue loop's timing
    and ordering must not need a decodable H.264 file, a model or a GPU.
    """
    return FileVideoSource(entry.input_path, declared_fps=entry.declared_fps)


@dataclass(slots=True)
class _StreamRuntime:
    """One camera-local pass, and the objects the adapter needs beside it.

    Held per ``(camera_id, stream_epoch)``: tracking, trails and the behaviour
    analyzer's history are all camera-local, and the processor itself refuses a
    frame from another stream.
    """

    stream: StreamRef
    metadata: VideoMetadata
    processor: VideoProcessor
    sink: _DiscardSink
    behavior: BehaviorAnalyzer
    projector: GroundPlaneProjector | None
    integrity_guard: CalibrationIntegrityGuard | None
    frames_processed: int = field(default=0)
    observations_emitted: int = field(default=0)


class _DiscardSink:
    """A :class:`~atreides_vision.video.sink.VideoSink` that keeps nothing.

    The processor annotates every frame whether or not anyone will watch it;
    writing an MP4 per camera would make a validation run's cost dominated by
    encoding output nobody reads.
    """

    def __init__(self) -> None:
        self.frames_written = 0
        self.released = False

    @property
    def path(self) -> Path:
        return _DISCARDED_OUTPUT

    def write(self, frame: object) -> None:
        self.frames_written += 1

    def release(self) -> None:
        self.released = True


def _load_calibration(camera: VenueValidationCameraConfig | None) -> CalibrationFit | None:
    """This camera's calibration, or ``None`` for an uncalibrated camera.

    A fit injected directly (``calibration_fit``) is used as-is; a document path
    (``calibration_path``) is loaded and solved. The config guarantees at most
    one of the two is set. Both routes yield the same :class:`CalibrationFit`
    and feed the same projector.
    """
    if camera is None:
        return None
    if camera.calibration_fit is not None:
        return camera.calibration_fit
    if camera.calibration_path is not None:
        return load_calibration_fit(camera.calibration_path)
    return None


def _build_integrity_guard(
    camera: VenueValidationCameraConfig | None,
) -> CalibrationIntegrityGuard | None:
    """This camera's integrity guard, built exactly as the CLI builds it.

    A missing or unreadable reference fails the run here rather than leaving
    monitoring quietly disabled on a session that asked to be guarded.
    """
    if camera is None or camera.integrity_reference_path is None:
        return None
    config = CalibrationIntegrityConfig(
        policy_mode=(
            IntegrityPolicyMode.REPORT_ONLY
            if camera.integrity_report_only
            else IntegrityPolicyMode.GUARDED
        )
    )
    monitor = CalibrationIntegrityMonitor(
        reference_frame=read_reference_frame(camera.integrity_reference_path),
        config=config,
        reference_id=str(camera.integrity_reference_path),
    )
    return CalibrationIntegrityGuard(monitor, config)


def _build_stream_runtime(
    stream: StreamRef,
    entry: VenueStreamEntry,
    metadata: VideoMetadata,
    validation: VenueValidationConfig,
    camera: VenueValidationCameraConfig | None,
    fit: CalibrationFit | None,
    detector: PersonDetector,
    tracker: MultiObjectTracker,
) -> _StreamRuntime:
    """Compose one stream's camera-local pipeline.

    The projector is built here, per stream, because it is bound to that
    stream's declared frame geometry — and kept here, because the behaviour
    analyzer exposes its space but not its projector, and
    :func:`build_track_observation` needs the projector itself.
    """
    projector: GroundPlaneProjector | None = None
    if fit is None:
        behavior = BehaviorAnalyzer(
            camera_id=stream.camera_id,
            projector=NormalizedFrameProjector(metadata.width, metadata.height),
            config=BehaviorAnalysisConfig(),
        )
    else:
        projector = GroundPlaneProjector(
            fit, frame_width=metadata.width, frame_height=metadata.height
        )
        behavior = BehaviorAnalyzer(
            camera_id=stream.camera_id,
            projector=projector,
            config=BehaviorAnalysisConfig.for_ground_plane_metres(),
        )

    guard = _build_integrity_guard(camera)
    vision_config = VisionConfig(
        pipeline=PipelineConfig(
            input_path=entry.input_path,
            output_path=_DISCARDED_OUTPUT,
            deduplicate_detections=validation.deduplicate_detections,
        ),
        detector=validation.detector,
        tracker=TrackerConfig(frame_rate=metadata.fps),
    )
    processor = VideoProcessor(
        detector=detector,
        tracker=tracker,
        config=vision_config,
        behavior_analyzer=behavior,
        integrity_guard=guard,
    )
    return _StreamRuntime(
        stream=stream,
        metadata=metadata,
        processor=processor,
        sink=_DiscardSink(),
        behavior=behavior,
        projector=projector,
        integrity_guard=guard,
    )


def run_venue_validation(
    config: VenueValidationConfig,
    *,
    detector: PersonDetector,
    tracker_factory: Callable[[TrackerConfig], MultiObjectTracker],
    source_factory: Callable[[VenueStreamEntry], VideoSource] = open_stream_source,
) -> VenueValidationSummary:
    """Run one venue session end to end and write its validation artifacts.

    One invocation is one session: sources are opened, the venue schedule is
    walked once, and every source and writer is released before this returns —
    on the failure path too.

    Args:
        config: The session, its cameras' spatial configuration, and where the
            artifacts go.
        detector: The person detector, shared across cameras. Detection is
            per-frame and stateless; tracking is not, which is why the tracker
            arrives as a factory instead.
        tracker_factory: Builds a fresh tracker for each stream, from that
            stream's own frame rate. A tracker shared between two cameras would
            continue one camera's track ids onto another's people.
        source_factory: Opens one manifest stream. Defaults to a real file.

    Returns:
        The run's execution statistics. They describe what ran — never how
        accurate it was.

    Raises:
        AtreidesVisionError: whatever the underlying layer raises, unchanged.
            A run that fails writes no summary: an empty successful-looking
            summary is worse than no summary at all.
    """
    manifest = config.manifest
    logger.info(
        "V0.4.8 venue validation — session '%s': %d camera(s), %d stream(s)",
        manifest.session_id,
        len(manifest),
        manifest.stream_count,
    )

    timebase = VenueTimebase()
    pipeline = VenuePipeline(identity=config.identity)
    association_counts: Counter[AssociationOutcome] = Counter()
    integration_counts: Counter[IntegrationOutcome] = Counter()
    occupancy_counts: list[int] = []
    suppressed_duplicates = 0
    venue_instants = 0
    first_venue_seconds: float | None = None
    last_venue_seconds: float | None = None

    with ExitStack() as stack:
        cursors: list[VideoSourceCursor] = []
        runtimes: dict[StreamRef, _StreamRuntime] = {}

        for camera in manifest:
            camera_config = config.camera_config(camera.camera_id)
            # Loaded once per camera: one calibration describes one camera's
            # view of one plane, not one recording of it.
            fit = _load_calibration(camera_config)
            for epoch, entry in enumerate(camera.sources):
                stream = StreamRef(camera_id=camera.camera_id, stream_epoch=epoch)
                source = source_factory(entry)
                stack.callback(source.release)
                metadata = source.metadata
                timebase.register(
                    stream,
                    fps_source=metadata.fps_source,
                    offset_seconds=entry.start_offset_seconds,
                    offset_accuracy_seconds=entry.offset_accuracy_seconds,
                )
                cursors.append(VideoSourceCursor(stream, source, timebase))
                runtimes[stream] = _build_stream_runtime(
                    stream,
                    entry,
                    metadata,
                    config,
                    camera_config,
                    fit,
                    detector,
                    tracker_factory(TrackerConfig(frame_rate=metadata.fps)),
                )

        session = VenueSession(manifest.session_id, timebase, cursors)

        association_writer = stack.enter_context(
            JsonlVenuePipelineReportWriter(
                config.association_report_path,
                VenuePipelineReportMetadata(session_id=session.session_id),
            )
        )
        occupancy_writer = stack.enter_context(
            JsonlOccupancyReportWriter(
                config.occupancy_report_path,
                OccupancyReportMetadata(session_id=session.session_id),
            )
        )

        # Frame geometry is a property of a stream, taken once, from the same
        # metadata the projector was built from.
        for runtime in _canonical(runtimes):
            runtime.processor.begin(runtime.metadata, runtime.sink)

        for tick in session.ticks():
            observations: list[TrackObservation] = []
            # Already canonical: VenueTick refuses to hold contributions in any
            # other order, so nothing here depends on manifest camera order.
            for frame in tick.contributions:
                runtime = runtimes[frame.stream]
                runtime.processor.process(frame, runtime.sink)
                runtime.frames_processed += 1
                observations.extend(_observations_for(runtime, frame))

            result = pipeline.process_instant(tick.venue_seconds, observations)
            # Immediately after the instant, before any expiry: the registry
            # still holds exactly the memberships this instant produced.
            snapshot = compute_occupancy(result, pipeline.registry)

            association_writer.write_instant(result, pipeline.registry)
            occupancy_writer.write_snapshot(snapshot)

            association_counts.update(decision.outcome for decision in result.round.decisions)
            integration_counts.update(outcome.outcome for outcome in result.integration.outcomes)
            occupancy_counts.append(snapshot.occupancy_count)
            suppressed_duplicates += snapshot.suppressed_duplicate_count
            venue_instants += 1
            if first_venue_seconds is None:
                first_venue_seconds = tick.venue_seconds
            last_venue_seconds = tick.venue_seconds

        for runtime in _canonical(runtimes):
            runtime.processor.finish()

        summary = VenueValidationSummary(
            session_id=session.session_id,
            streams=tuple(
                sorted(
                    (_stream_summary(runtime) for runtime in runtimes.values()),
                    key=stream_summary_sort_key,
                )
            ),
            venue_instants=venue_instants,
            source_frames_processed=sum(r.frames_processed for r in runtimes.values()),
            observations_emitted=sum(r.observations_emitted for r in runtimes.values()),
            association_outcomes=dict(association_counts),
            integration_outcomes=dict(integration_counts),
            venue_subjects_created=len(pipeline.registry.subjects()),
            venue_subjects_current=len(pipeline.registry.active_subjects()),
            occupancy_snapshots=len(occupancy_counts),
            min_occupancy=min(occupancy_counts) if occupancy_counts else None,
            max_occupancy=max(occupancy_counts) if occupancy_counts else None,
            mean_occupancy=(
                round(sum(occupancy_counts) / len(occupancy_counts), 3)
                if occupancy_counts
                else None
            ),
            suppressed_duplicates=suppressed_duplicates,
            first_venue_seconds=first_venue_seconds,
            last_venue_seconds=last_venue_seconds,
            session_alignment=session.alignment,
        )

    # Written only once every source and stream writer has closed cleanly, so
    # a summary on disk means the run it describes actually completed.
    write_validation_summary(config.summary_path, summary)
    _log_completion(summary, config)
    return summary


def _canonical(runtimes: dict[StreamRef, _StreamRuntime]) -> list[_StreamRuntime]:
    """Stream runtimes in canonical order, never in dict insertion order."""
    return sorted(runtimes.values(), key=lambda runtime: stream_sort_key(runtime.stream))


def _observations_for(runtime: _StreamRuntime, frame: StreamFrame) -> list[TrackObservation]:
    """Adapt the frame just processed into venue observations.

    ``VideoProcessor.process`` has already driven the behaviour analyzer for
    this exact frame, so ``latest_frame`` is that frame's snapshots. Every
    observation goes through V0.4.7's adapter — this runner never builds a
    ``TrackObservation`` itself, and the adapter's own checks reject a snapshot
    paired with a frame that did not produce it.
    """
    behavior_frame = runtime.behavior.latest_frame
    if behavior_frame is None:
        return []
    observations = [
        build_track_observation(
            snapshot,
            frame,
            calibration=runtime.projector,
            integrity_guard=runtime.integrity_guard,
        )
        for snapshot in behavior_frame.snapshots
    ]
    runtime.observations_emitted += len(observations)
    return observations


def _stream_summary(runtime: _StreamRuntime) -> VenueValidationStreamSummary:
    return VenueValidationStreamSummary(
        stream=runtime.stream,
        fps=runtime.metadata.fps,
        fps_source=runtime.metadata.fps_source,
        frames_processed=runtime.frames_processed,
        observations_emitted=runtime.observations_emitted,
        calibrated=runtime.projector is not None,
        integrity_monitored=runtime.integrity_guard is not None,
    )


def _log_completion(summary: VenueValidationSummary, config: VenueValidationConfig) -> None:
    logger.info("Done.")
    logger.info("Venue instants:      %d", summary.venue_instants)
    logger.info("Source frames:       %d", summary.source_frames_processed)
    logger.info("Observations:        %d", summary.observations_emitted)
    for stream in summary.zero_frame_streams:
        logger.info("  %-16s contributed no frames", str(stream))
    logger.info("Association report:  %s", config.association_report_path)
    logger.info("Occupancy report:    %s", config.occupancy_report_path)
    logger.info("Summary:             %s", config.summary_path)
    logger.info(
        "Note:        execution statistics only. This session is NOT_ANNOTATED, so nothing "
        "here is an association, occupancy or duplicate-suppression accuracy figure."
    )

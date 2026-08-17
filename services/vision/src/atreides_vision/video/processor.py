"""Pipeline orchestration: read -> detect -> track -> zones -> annotate -> write.

The processor knows nothing about RF-DETR, ByteTrack, or any model. It talks to
:class:`~atreides_vision.detection.base.PersonDetector`,
:class:`~atreides_vision.tracking.base.MultiObjectTracker`,
:class:`~atreides_vision.video.source.VideoSource` and
:class:`~atreides_vision.video.sink.VideoSink` — all Atreides-owned interfaces.
That is what makes the ML components replaceable and this class testable with
fakes.

Zone analysis is optional and arrives the same way: as one collaborator, the
:class:`~atreides_vision.zones.analyzer.ZoneAnalyzer`, which owns everything
about polygons and metrics. Without it the loop is exactly the V0.1 pipeline.

V0.3 behaviour signals attach identically, as an optional
:class:`~atreides_vision.behavior.analyzer.BehaviorAnalyzer`. It observes the
same tracks and changes nothing that is drawn or written: a run without one is
byte-for-byte what it was before.

Frames arrive already timed
---------------------------
Since V0.4.0 the processor no longer derives time. It is handed
:class:`~atreides_vision.timing.StreamFrame` values that already carry their
venue seconds, and every analyzer is driven from those. That is what makes the
same processor usable for one clip and for one camera of a venue: a stream that
stops and restarts moves *source* seconds backwards, and any measurement built
on them would see time reverse.

:meth:`VideoProcessor.run` is the single-source path and is unchanged in
behaviour — it supplies its own frames, with venue time equal to source time —
while :meth:`VideoProcessor.begin`, :meth:`VideoProcessor.process` and
:meth:`VideoProcessor.finish` are the same loop opened up so an outer venue loop
can drive one processor per camera. ``process()`` returns ``None`` on purpose: a
per-frame result type would be the beginning of a cross-camera correspondence
surface, which this milestone does not have.

**One processor follows one stream.** Everything a pass accumulates is
camera-local — one tracker, one trajectory store, one set of track ids — so the
pass binds to its first frame's
:class:`~atreides_vision.timing.StreamRef` and refuses frames from anywhere
else, the same discipline the venue scheduler applies one layer down.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from atreides_vision.behavior.analyzer import BehaviorAnalyzer
from atreides_vision.behavior.models import BehaviorFrame
from atreides_vision.behavior.reporting import BehaviorWriter
from atreides_vision.calibration.integrity.policy import CalibrationIntegrityGuard
from atreides_vision.config import VisionConfig
from atreides_vision.crowd.analyzer import SpatialClusterAnalyzer
from atreides_vision.crowd.models import CrowdFrame
from atreides_vision.crowd.reporting import CrowdWriter
from atreides_vision.detection.base import PersonDetector
from atreides_vision.detection.deduplication import deduplicate_with_report
from atreides_vision.errors import ConfigurationError
from atreides_vision.interaction.analyzer import InteractionAnalyzer
from atreides_vision.interaction.models import InteractionFrame
from atreides_vision.interaction.reporting import InteractionWriter
from atreides_vision.models import Track
from atreides_vision.timing import StreamFrame, StreamRef, TimeUncertainty
from atreides_vision.tracking.base import MultiObjectTracker
from atreides_vision.tracking.diagnostics import (
    TrackingDebugRecord,
    TrackingDebugWriter,
    snapshot_internal_tracks,
)
from atreides_vision.video.sink import VideoSink
from atreides_vision.video.source import VideoMetadata, VideoSource
from atreides_vision.video.trajectories import TrajectoryStore
from atreides_vision.visualization.overlay import Annotator, StatusOverlay, ZoneOverlay
from atreides_vision.zones.analyzer import ZoneAnalyzer
from atreides_vision.zones.models import ZoneReport
from atreides_vision.zones.reporting import MetricsSampler, MetricsWriter

logger = logging.getLogger(__name__)

__all__ = ["FpsMeter", "ProcessingResult", "VideoProcessor"]

#: Stream label for :meth:`VideoProcessor.run`, which processes one source with
#: no venue context. Nothing reads it — camera identity reaches the signal
#: layers through their own construction, and no output record carries it — but
#: a :class:`~atreides_vision.timing.StreamFrame` must name a stream, and
#: inventing a camera id here would be a claim about a deployment. A venue run
#: labels its frames from the manifest instead.
_SINGLE_SOURCE_STREAM = StreamRef(camera_id="single_source", stream_epoch=0)


class FpsMeter:
    """Rolling throughput estimate over the most recent frames.

    A rolling window is what an operator wants on screen: it reflects current
    throughput rather than an average dragged down by model warm-up.
    """

    def __init__(self, window: int = 30, clock: Callable[[], float] = time.perf_counter) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self._timestamps: deque[float] = deque(maxlen=window)
        self._clock = clock

    def tick(self) -> None:
        """Record that one frame finished processing."""
        self._timestamps.append(self._clock())

    @property
    def fps(self) -> float:
        """Frames per second over the window, or ``0.0`` before it can be known."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    """Summary of a completed run."""

    frames_processed: int
    elapsed_seconds: float
    output_path: Path
    unique_track_ids: int
    #: Final zone metrics, or ``None`` when the run had no zone analyzer.
    final_zone_report: ZoneReport | None = None
    #: Records written to the metrics stream, if one was attached.
    metrics_records_written: int = 0
    #: Detections suppressed by deduplication. Always 0 when it is disabled.
    detections_removed_by_deduplication: int = 0
    #: Final behaviour signals, or ``None`` when the run had no behaviour analyzer.
    final_behavior_frame: BehaviorFrame | None = None
    #: Behaviour signals emitted across the run. Always 0 without an analyzer.
    behavior_signals_emitted: int = 0
    #: Frames whose measurement output was withheld because camera drift was
    #: detected. Counted once per frame across every enabled stream, so it means
    #: the same thing whether behaviour, interaction or both were being written.
    #: Always 0 without integrity monitoring, or when the camera stayed put.
    behavior_frames_suppressed: int = 0
    #: Final pairwise measurements, or ``None`` without an interaction analyzer.
    final_interaction_frame: InteractionFrame | None = None
    #: Pair measurements produced across the run. Always 0 without an analyzer.
    interaction_pairs_measured: int = 0
    #: Final spatial clusters, or ``None`` without a crowd analyzer.
    final_crowd_frame: CrowdFrame | None = None
    #: Cluster measurements produced across the run. Always 0 without an analyzer.
    crowd_clusters_measured: int = 0

    @property
    def effective_fps(self) -> float:
        """Overall throughput including model warm-up."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frames_processed / self.elapsed_seconds


@dataclass(slots=True)
class _RunState:
    """Everything one pass over a stream accumulates.

    Held on the processor rather than in local variables because a frame is now
    a call — :meth:`VideoProcessor.process` — and the state has to survive
    between calls. :meth:`VideoProcessor.finish` discards it, so nothing carries
    over into a later run.

    Bounded by construction: the trajectory store and FPS meter are already
    capped, and everything else here is a counter or a single latest frame. A
    long run costs no more than a short one.
    """

    trajectories: TrajectoryStore
    fps_meter: FpsMeter
    sampler: MetricsSampler
    output_path: Path
    started_at: float
    #: The stream's declared frame geometry, bound by
    #: :meth:`VideoProcessor.begin`. ``None`` until then, which is why
    #: :meth:`VideoProcessor.process` refuses to run without it.
    frame_width: int | None = None
    frame_height: int | None = None
    #: The stream this pass follows, bound by its first frame. Everything the
    #: pass accumulates — the tracker, the trails, the track-id set — is
    #: camera-local, so a frame from anywhere else does not belong in it.
    stream: StreamRef | None = None
    #: The venue time of the previous frame, which the next must be after.
    last_venue_seconds: float | None = None
    seen_track_ids: set[int] = field(default_factory=set)
    frames_processed: int = 0
    metrics_records: int = 0
    detections_removed: int = 0
    zone_report: ZoneReport | None = None
    behavior_frame: BehaviorFrame | None = None
    interaction_frame: InteractionFrame | None = None
    crowd_frame: CrowdFrame | None = None


class VideoProcessor:
    """Runs the SEE + TRACK pipeline over one video source.

    With a ``zone_analyzer`` attached it additionally runs V0.2 zone
    intelligence; with a ``behavior_analyzer`` it additionally measures V0.3
    behaviour signals. All are optional collaborators, so a V0.1 run is
    unchanged and a V0.2 run stays exactly as it was.
    """

    def __init__(
        self,
        detector: PersonDetector,
        tracker: MultiObjectTracker,
        config: VisionConfig,
        annotator: Annotator | None = None,
        clock: Callable[[], float] = time.perf_counter,
        zone_analyzer: ZoneAnalyzer | None = None,
        metrics_writer: MetricsWriter | None = None,
        tracking_debug_writer: TrackingDebugWriter | None = None,
        behavior_analyzer: BehaviorAnalyzer | None = None,
        behavior_writer: BehaviorWriter | None = None,
        integrity_guard: CalibrationIntegrityGuard | None = None,
        interaction_analyzer: InteractionAnalyzer | None = None,
        interaction_writer: InteractionWriter | None = None,
        crowd_analyzer: SpatialClusterAnalyzer | None = None,
        crowd_writer: CrowdWriter | None = None,
    ) -> None:
        self._detector = detector
        self._tracker = tracker
        self._config = config
        self._annotator = annotator or Annotator(config.visualization)
        self._clock = clock
        self._zone_analyzer = zone_analyzer
        self._metrics_writer = metrics_writer
        self._tracking_debug_writer = tracking_debug_writer
        self._behavior_analyzer = behavior_analyzer
        self._behavior_writer = behavior_writer
        self._integrity_guard = integrity_guard
        self._interaction_analyzer = interaction_analyzer
        self._interaction_writer = interaction_writer
        self._crowd_analyzer = crowd_analyzer
        self._crowd_writer = crowd_writer
        self._state: _RunState | None = None

    def run(self, source: VideoSource, sink: VideoSink) -> ProcessingResult:
        """Process every frame of ``source`` into ``sink``.

        The caller owns the lifetimes of ``source`` and ``sink``; this method
        neither opens nor releases them, so a failure mid-run still leaves the
        caller's ``with`` blocks to finalise the output file.

        The single-source path: this supplies its own timed frames, in which
        venue time equals source time, and is otherwise exactly
        :meth:`process` per frame followed by :meth:`finish`.
        """
        # A run is self-contained, as it always has been: any state left behind
        # by a pass that raised part-way through belongs to that pass, not this
        # one, and must not be counted into this run's result.
        self._state = None
        # Begun before the first frame is decoded, so the elapsed time covers
        # the whole pass and an injected clock is consumed in the same order.
        self.begin(source.metadata, sink)
        for frame in self._supply(source):
            self.process(frame, sink)
        return self.finish()

    def begin(self, metadata: VideoMetadata, sink: VideoSink) -> None:
        """Open a pass over one stream, adopting its declared frame geometry.

        Called once before the stream's first frame — by :meth:`run` from the
        source it is about to read, and by a venue loop from the metadata of the
        source behind that stream.

        The geometry is taken here, once, rather than per frame: it is a
        property of the stream, not of any one image, and the composition root
        has already built this camera's projector from the very same
        :class:`~atreides_vision.video.source.VideoMetadata`.
        """
        state = self._begin(sink)
        if state.frame_width is not None and (
            state.frame_width != metadata.width or state.frame_height != metadata.height
        ):
            raise ConfigurationError(
                f"This pass was opened at {state.frame_width}x{state.frame_height} and is "
                f"being re-opened at {metadata.width}x{metadata.height}. Zone geometry is "
                "normalised against these dimensions, so changing them mid-pass would move "
                "every zone boundary under the subjects already measured. Finish the pass "
                "first."
            )
        state.frame_width = metadata.width
        state.frame_height = metadata.height

    def process(self, frame: StreamFrame, sink: VideoSink) -> None:
        """Run one already-timed frame through the pipeline.

        The per-frame step of :meth:`run`, exposed so a venue loop can drive one
        processor per camera from a merged tick schedule. Every analyzer is
        driven from ``frame.venue_seconds``, which is monotonic across a stream
        restart where source seconds are not.

        Returns ``None`` deliberately. A per-frame result type would be the
        obvious hook to hang cross-camera correspondence on, and V0.4.0 performs
        no association: what a venue loop needs from this call is the side
        effects — the annotated frame and the measurement streams — not a value.

        Raises:
            ConfigurationError: if :meth:`begin` has not bound this stream's
                frame geometry, or the frame comes from a stream other than the
                one this pass is following, or its venue time does not advance.
        """
        state = self._begin(sink)
        self._bind(state, frame)

        frame_index = frame.frame_index
        timestamp_seconds = frame.venue_seconds
        image = frame.image
        # The container's declared resolution is authoritative here, for
        # consistency with the projectors. The composition root builds this
        # camera's NormalizedFrameProjector or GroundPlaneProjector from the
        # same VideoMetadata, so behaviour positions and the zone labels
        # attached to them must be normalised against the same dimensions — one
        # site using the decoded array instead would compute zone membership in
        # a different space from the metres it is attached to.
        #
        # A container that misreports its resolution is therefore a real and
        # unhandled case: rotation metadata is reported pre-rotation, so a
        # portrait clip can announce its dimensions transposed, and today every
        # one of those sites is wrong together rather than inconsistently.
        # Correcting it means changing the projector construction sites and this
        # one in a single bounded task — not one of them here.
        frame_width, frame_height = self._require_geometry(state)

        deduplicate = self._config.pipeline.deduplicate_detections
        raw_detections = self._detector.detect(image)

        # Optional, off by default: geometric duplicate suppression sits
        # between detection and tracking, so the tracker never sees two
        # boxes on one person. Anonymous single-frame geometry only.
        if deduplicate:
            report = deduplicate_with_report(raw_detections)
            detections = list(report.kept)
            removed_detections = report.removed
            state.detections_removed += report.removed_count
        else:
            detections = raw_detections
            removed_detections = ()

        tracks = self._tracker.update(detections, image)

        if self._tracking_debug_writer is not None:
            # Purely observational: reads state the tracker has already
            # computed. Enabled only by --tracking-debug-output.
            self._tracking_debug_writer.write(
                TrackingDebugRecord(
                    frame_index=frame_index,
                    detections=tuple(detections),
                    reported_tracks=tuple(tracks),
                    internal_tracks=snapshot_internal_tracks(self._tracker),
                    raw_detections=tuple(raw_detections) if deduplicate else None,
                    removed_detections=removed_detections,
                )
            )

        state.trajectories.update(tracks, frame_index)
        state.seen_track_ids.update(track.track_id for track in tracks)

        zone_overlay: ZoneOverlay | None = None
        if self._zone_analyzer is not None:
            state.zone_report = self._zone_analyzer.update(
                tracks,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            zone_overlay = ZoneOverlay.from_report(
                self._zone_analyzer.layout.zones, state.zone_report
            )
            if self._metrics_writer is not None and state.sampler.should_sample(timestamp_seconds):
                self._metrics_writer.write(state.zone_report)
                state.metrics_records += 1

        if self._behavior_analyzer is not None:
            # Measurement only: the signals are written to their own stream
            # and never drawn, so the rendered video is identical with and
            # without behaviour enabled. Zone ids arrive as opaque labels —
            # the behaviour layer never learns what a zone is, which is why
            # this works with or without zones configured.
            behavior_frame = self._behavior_analyzer.update(
                tracks,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                zone_ids_by_track=(
                    None if self._zone_analyzer is None else self._zone_analyzer.zone_ids_by_track
                ),
            )
            state.behavior_frame = behavior_frame

            # Integrity runs on its own cadence, so this is a no-op on most
            # frames. The guard owns the schedule, the thresholds and the
            # policy; the processor only relays the finding and honours the
            # verdict.
            if self._integrity_guard is not None:
                assessment = self._integrity_guard.evaluate(image, frame_index, timestamp_seconds)
                if assessment is not None and self._behavior_writer is not None:
                    self._behavior_writer.write_integrity(assessment)

            # One trust decision, honoured by both streams. Pairwise metres
            # are derived from the same positions, so output the calibration
            # has disowned must not reappear as an interaction record.
            output_trusted = (
                self._integrity_guard is None or self._integrity_guard.metric_output_allowed
            )

            if self._integrity_guard is not None and not output_trusted:
                # The camera has demonstrably moved. Withholding the output
                # is the point: the calibration's metres no longer describe
                # this camera, and emitting them anyway would look entirely
                # normal downstream. Counted per frame, not per stream, so
                # the figure means the same thing however many are enabled.
                self._integrity_guard.note_suppressed_frame()

            if self._behavior_writer is not None and output_trusted:
                # Streamed per frame, never accumulated: a long run costs no
                # more memory than a short one.
                self._behavior_writer.write_frame(behavior_frame)

            if self._interaction_analyzer is not None:
                # Pair history updates every frame regardless of whether the
                # result is serialised, so closing speed and proximity
                # duration are always built on the complete record.
                state.interaction_frame = self._interaction_analyzer.update(behavior_frame)
                if self._interaction_writer is not None and output_trusted:
                    self._interaction_writer.write_frame(state.interaction_frame)

            if self._crowd_analyzer is not None:
                # A third consumer of the same BehaviorFrame — the frame is
                # analysed once and fanned out, never re-projected. Cluster
                # persistence likewise updates every frame regardless of
                # emission.
                state.crowd_frame = self._crowd_analyzer.update(behavior_frame)
                if self._crowd_writer is not None and output_trusted:
                    self._crowd_writer.write_frame(state.crowd_frame)

        state.fps_meter.tick()
        status = StatusOverlay(
            frame_index=frame_index,
            processing_fps=state.fps_meter.fps,
            track_count=len(tracks),
            zone_count=None if self._zone_analyzer is None else len(self._zone_analyzer.layout),
        )
        annotated = self._annotator.annotate(
            image, tracks, state.trajectories, status, zone_overlay
        )
        sink.write(annotated)

        state.frames_processed += 1
        if state.frames_processed % self._config.pipeline.log_every_n_frames == 0:
            self._log_progress(
                state.frames_processed, tracks, state.fps_meter.fps, state.zone_report
            )

    def finish(self) -> ProcessingResult:
        """Close the pass and summarise it, discarding the run state.

        Safe with no frames at all: an empty source is reported as such rather
        than silently producing a plausible-looking zero-frame summary.
        """
        state = self._begin(None)
        self._state = None

        elapsed = self._clock() - state.started_at
        if state.frames_processed == 0:
            logger.warning("No frames were decoded from the input video.")

        return ProcessingResult(
            frames_processed=state.frames_processed,
            elapsed_seconds=elapsed,
            output_path=state.output_path,
            unique_track_ids=len(state.seen_track_ids),
            final_zone_report=state.zone_report,
            metrics_records_written=state.metrics_records,
            detections_removed_by_deduplication=state.detections_removed,
            final_behavior_frame=state.behavior_frame,
            behavior_signals_emitted=(
                0 if self._behavior_analyzer is None else self._behavior_analyzer.signals_emitted
            ),
            behavior_frames_suppressed=(
                0 if self._integrity_guard is None else self._integrity_guard.suppressed_frames
            ),
            final_interaction_frame=state.interaction_frame,
            interaction_pairs_measured=(
                0
                if self._interaction_analyzer is None
                else self._interaction_analyzer.pairs_analysed
            ),
            final_crowd_frame=state.crowd_frame,
            crowd_clusters_measured=(
                0 if self._crowd_analyzer is None else self._crowd_analyzer.clusters_measured
            ),
        )

    def _bind(self, state: _RunState, frame: StreamFrame) -> None:
        """Hold the pass to one stream, moving forwards.

        The same discipline :func:`~atreides_vision.venue.scheduler.merge_streams`
        applies one layer down, applied here because everything this pass
        accumulates is camera-local: one tracker, one trajectory store, one
        track-id set. A frame from a second camera would be tracked as a
        continuation of the first; a frame from the next epoch of the *same*
        camera would be worse, because its frame index restarts at 1 and the
        trail TTL reads that index as monotonic.
        """
        if state.stream is None:
            state.stream = frame.stream
        elif frame.stream != state.stream:
            raise ConfigurationError(
                f"This pass is following stream {state.stream} and was handed a frame from "
                f"{frame.stream}. Tracking, trails and the track-id set are all camera-local, "
                "so frames from another camera — or another recording of the same camera — "
                "would be measured as a continuation of this one. Use one processor per stream."
            )

        if state.last_venue_seconds is not None and frame.venue_seconds <= state.last_venue_seconds:
            raise ConfigurationError(
                f"Stream {frame.stream} offered frame {frame.frame_index} at venue second "
                f"{frame.venue_seconds!r}, which is not after the previous frame at "
                f"{state.last_venue_seconds!r}. Speed, dwell, cluster persistence and trail "
                "ageing all read venue time as monotonic."
            )
        state.last_venue_seconds = frame.venue_seconds

    def _require_geometry(self, state: _RunState) -> tuple[int, int]:
        """The stream's declared frame geometry, or a refusal to guess at it."""
        if state.frame_width is None or state.frame_height is None:
            raise ConfigurationError(
                "This pass has no frame geometry. Call begin(metadata, sink) with the "
                "stream's VideoMetadata before its first frame: zone geometry is normalised "
                "against the declared dimensions, the same ones this camera's projector was "
                "built from, and they are not guessed from a decoded frame."
            )
        return state.frame_width, state.frame_height

    def _begin(self, sink: VideoSink | None) -> _RunState:
        """Start a pass if one is not already running, and return its state."""
        if self._state is not None:
            return self._state

        visualization = self._config.visualization
        self._state = _RunState(
            trajectories=TrajectoryStore(
                max_length=visualization.trail_length,
                ttl_frames=visualization.trail_ttl_frames,
                max_trails=visualization.max_trails,
            ),
            # Shares the processor's clock so an injected clock makes a run —
            # and therefore the rendered status panel — fully deterministic
            # under test.
            fps_meter=FpsMeter(clock=self._clock),
            sampler=MetricsSampler(self._config.zones.metrics_sample_interval_seconds),
            output_path=(
                self._config.pipeline.output_path
                if sink is None
                else getattr(sink, "path", self._config.pipeline.output_path)
            ),
            started_at=self._clock(),
        )
        return self._state

    def _supply(self, source: VideoSource) -> Iterator[StreamFrame]:
        """Time the frames of one source, honouring ``--max-frames``.

        The frame cap lives here rather than in :meth:`process` because it is a
        property of how much of a source to read, not of how a frame is handled.

        Source timestamps come from the real frame rate. Dwell and speed are
        only frame-rate correct because of this; nothing assumes 30 FPS.
        """
        metadata = source.metadata
        source_fps = metadata.fps if metadata.fps > 0 else 1.0
        max_frames = self._config.pipeline.max_frames
        # One source, so nothing is aligned to anything: alignment describes a
        # relation between streams, and there is no second stream here.
        timing = TimeUncertainty.unknown()

        for frame_index, image in enumerate(source.frames(), start=1):
            if max_frames is not None and frame_index > max_frames:
                logger.info("Reached --max-frames limit of %d; stopping early.", max_frames)
                break

            source_seconds = (frame_index - 1) / source_fps
            yield StreamFrame(
                stream=_SINGLE_SOURCE_STREAM,
                frame_index=frame_index,
                source_seconds=source_seconds,
                # The venue timeline of a one-source run *is* this stream's
                # timeline: the offset is zero, so this is bit-identical to the
                # timestamp the pipeline used before the venue seam existed.
                venue_seconds=source_seconds,
                timing=timing,
                image=image,
            )

    def _log_progress(
        self,
        frames_processed: int,
        tracks: Sequence[Track],
        fps: float,
        zone_report: ZoneReport | None,
    ) -> None:
        """One periodic progress line, with zone occupancy when available."""
        if zone_report is None:
            logger.info(
                "Processed %d frames | %d active tracks | %.1f FPS",
                frames_processed,
                len(tracks),
                fps,
            )
            return
        occupancy = " ".join(
            f"{snapshot.zone_id}={snapshot.occupancy}" for snapshot in zone_report.snapshots
        )
        logger.info(
            "Processed %d frames | %d active tracks | %.1f FPS | %s",
            frames_processed,
            len(tracks),
            fps,
            occupancy or "no zones",
        )

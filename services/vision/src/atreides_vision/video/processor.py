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
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from atreides_vision.behavior.analyzer import BehaviorAnalyzer
from atreides_vision.behavior.models import BehaviorFrame
from atreides_vision.behavior.reporting import BehaviorWriter
from atreides_vision.config import VisionConfig
from atreides_vision.detection.base import PersonDetector
from atreides_vision.detection.deduplication import deduplicate_with_report
from atreides_vision.models import Track
from atreides_vision.tracking.base import MultiObjectTracker
from atreides_vision.tracking.diagnostics import (
    TrackingDebugRecord,
    TrackingDebugWriter,
    snapshot_internal_tracks,
)
from atreides_vision.video.sink import VideoSink
from atreides_vision.video.source import VideoSource
from atreides_vision.video.trajectories import TrajectoryStore
from atreides_vision.visualization.overlay import Annotator, StatusOverlay, ZoneOverlay
from atreides_vision.zones.analyzer import ZoneAnalyzer
from atreides_vision.zones.models import ZoneReport
from atreides_vision.zones.reporting import MetricsSampler, MetricsWriter

logger = logging.getLogger(__name__)

__all__ = ["FpsMeter", "ProcessingResult", "VideoProcessor"]


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

    @property
    def effective_fps(self) -> float:
        """Overall throughput including model warm-up."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frames_processed / self.elapsed_seconds


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

    def run(self, source: VideoSource, sink: VideoSink) -> ProcessingResult:
        """Process every frame of ``source`` into ``sink``.

        The caller owns the lifetimes of ``source`` and ``sink``; this method
        neither opens nor releases them, so a failure mid-run still leaves the
        caller's ``with`` blocks to finalise the output file.
        """
        visualization = self._config.visualization
        trajectories = TrajectoryStore(
            max_length=visualization.trail_length,
            ttl_frames=visualization.trail_ttl_frames,
            max_trails=visualization.max_trails,
        )
        # Shares the processor's clock so an injected clock makes a run — and
        # therefore the rendered status panel — fully deterministic under test.
        fps_meter = FpsMeter(clock=self._clock)
        seen_track_ids: set[int] = set()

        max_frames = self._config.pipeline.max_frames
        log_every = self._config.pipeline.log_every_n_frames

        metadata = source.metadata
        # Source timestamps come from the real frame rate. Dwell and speed are
        # only frame-rate correct because of this; nothing assumes 30 FPS.
        source_fps = metadata.fps if metadata.fps > 0 else 1.0
        sampler = MetricsSampler(self._config.zones.metrics_sample_interval_seconds)
        metrics_records = 0
        zone_report: ZoneReport | None = None
        behavior_frame: BehaviorFrame | None = None

        deduplicate = self._config.pipeline.deduplicate_detections
        detections_removed = 0

        frames_processed = 0
        started_at = self._clock()

        for frame_index, frame in enumerate(source.frames(), start=1):
            if max_frames is not None and frames_processed >= max_frames:
                logger.info("Reached --max-frames limit of %d; stopping early.", max_frames)
                break

            timestamp_seconds = (frame_index - 1) / source_fps

            raw_detections = self._detector.detect(frame)

            # Optional, off by default: geometric duplicate suppression sits
            # between detection and tracking, so the tracker never sees two
            # boxes on one person. Anonymous single-frame geometry only.
            if deduplicate:
                report = deduplicate_with_report(raw_detections)
                detections = list(report.kept)
                removed_detections = report.removed
                detections_removed += report.removed_count
            else:
                detections = raw_detections
                removed_detections = ()

            tracks = self._tracker.update(detections, frame)

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

            trajectories.update(tracks, frame_index)
            seen_track_ids.update(track.track_id for track in tracks)

            zone_overlay: ZoneOverlay | None = None
            if self._zone_analyzer is not None:
                zone_report = self._zone_analyzer.update(
                    tracks,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    frame_width=metadata.width,
                    frame_height=metadata.height,
                )
                zone_overlay = ZoneOverlay.from_report(
                    self._zone_analyzer.layout.zones, zone_report
                )
                if self._metrics_writer is not None and sampler.should_sample(timestamp_seconds):
                    self._metrics_writer.write(zone_report)
                    metrics_records += 1

            if self._behavior_analyzer is not None:
                # Measurement only: the signals are written to their own stream
                # and never drawn, so the rendered video is identical with and
                # without behaviour enabled. Zone context is not passed through
                # yet, which is why behaviour analysis works with or without
                # zones configured.
                behavior_frame = self._behavior_analyzer.update(
                    tracks,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                )
                if self._behavior_writer is not None:
                    # Streamed per frame, never accumulated: a long run costs no
                    # more memory than a short one.
                    self._behavior_writer.write_frame(behavior_frame)

            fps_meter.tick()
            status = StatusOverlay(
                frame_index=frame_index,
                processing_fps=fps_meter.fps,
                track_count=len(tracks),
                zone_count=None if self._zone_analyzer is None else len(self._zone_analyzer.layout),
            )
            annotated = self._annotator.annotate(frame, tracks, trajectories, status, zone_overlay)
            sink.write(annotated)

            frames_processed += 1
            if frames_processed % log_every == 0:
                self._log_progress(frames_processed, tracks, fps_meter.fps, zone_report)

        elapsed = self._clock() - started_at
        if frames_processed == 0:
            logger.warning("No frames were decoded from the input video.")

        return ProcessingResult(
            frames_processed=frames_processed,
            elapsed_seconds=elapsed,
            output_path=getattr(sink, "path", self._config.pipeline.output_path),
            unique_track_ids=len(seen_track_ids),
            final_zone_report=zone_report,
            metrics_records_written=metrics_records,
            detections_removed_by_deduplication=detections_removed,
            final_behavior_frame=behavior_frame,
            behavior_signals_emitted=(
                0 if self._behavior_analyzer is None else self._behavior_analyzer.signals_emitted
            ),
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

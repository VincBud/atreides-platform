"""Pipeline orchestration: read -> detect -> track -> annotate -> write.

The processor knows nothing about RF-DETR, ByteTrack, or any model. It talks to
:class:`~atreides_vision.detection.base.PersonDetector`,
:class:`~atreides_vision.tracking.base.MultiObjectTracker`,
:class:`~atreides_vision.video.source.VideoSource` and
:class:`~atreides_vision.video.sink.VideoSink` — all Atreides-owned interfaces.
That is what makes the ML components replaceable and this class testable with
fakes.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from atreides_vision.config import VisionConfig
from atreides_vision.detection.base import PersonDetector
from atreides_vision.tracking.base import MultiObjectTracker
from atreides_vision.video.sink import VideoSink
from atreides_vision.video.source import VideoSource
from atreides_vision.video.trajectories import TrajectoryStore
from atreides_vision.visualization.overlay import Annotator, StatusOverlay

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

    @property
    def effective_fps(self) -> float:
        """Overall throughput including model warm-up."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frames_processed / self.elapsed_seconds


class VideoProcessor:
    """Runs the SEE + TRACK pipeline over one video source."""

    def __init__(
        self,
        detector: PersonDetector,
        tracker: MultiObjectTracker,
        config: VisionConfig,
        annotator: Annotator | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._detector = detector
        self._tracker = tracker
        self._config = config
        self._annotator = annotator or Annotator(config.visualization)
        self._clock = clock

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
        fps_meter = FpsMeter()
        seen_track_ids: set[int] = set()

        max_frames = self._config.pipeline.max_frames
        log_every = self._config.pipeline.log_every_n_frames

        frames_processed = 0
        started_at = self._clock()

        for frame_index, frame in enumerate(source.frames(), start=1):
            if max_frames is not None and frames_processed >= max_frames:
                logger.info("Reached --max-frames limit of %d; stopping early.", max_frames)
                break

            detections = self._detector.detect(frame)
            tracks = self._tracker.update(detections, frame)

            trajectories.update(tracks, frame_index)
            seen_track_ids.update(track.track_id for track in tracks)

            fps_meter.tick()
            status = StatusOverlay(
                frame_index=frame_index,
                processing_fps=fps_meter.fps,
                track_count=len(tracks),
            )
            annotated = self._annotator.annotate(frame, tracks, trajectories, status)
            sink.write(annotated)

            frames_processed += 1
            if frames_processed % log_every == 0:
                logger.info(
                    "Processed %d frames | %d active tracks | %.1f FPS",
                    frames_processed,
                    len(tracks),
                    fps_meter.fps,
                )

        elapsed = self._clock() - started_at
        if frames_processed == 0:
            logger.warning("No frames were decoded from the input video.")

        return ProcessingResult(
            frames_processed=frames_processed,
            elapsed_seconds=elapsed,
            output_path=getattr(sink, "path", self._config.pipeline.output_path),
            unique_track_ids=len(seen_track_ids),
        )

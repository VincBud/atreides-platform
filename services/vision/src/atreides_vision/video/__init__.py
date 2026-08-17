"""Video I/O and pipeline orchestration."""

from __future__ import annotations

from atreides_vision.timing import FrameRateSource
from atreides_vision.video.processor import ProcessingResult, VideoProcessor
from atreides_vision.video.sink import Mp4VideoSink, VideoSink
from atreides_vision.video.source import FileVideoSource, VideoMetadata, VideoSource
from atreides_vision.video.trajectories import TrajectoryStore

__all__ = [
    "FileVideoSource",
    # Re-exported: it is what makes VideoMetadata.fps_source interpretable, so a
    # consumer of the metadata should not have to know which module defines it.
    "FrameRateSource",
    "Mp4VideoSink",
    "ProcessingResult",
    "TrajectoryStore",
    "VideoMetadata",
    "VideoProcessor",
    "VideoSink",
    "VideoSource",
]

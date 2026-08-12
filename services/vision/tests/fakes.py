"""In-memory implementations of the Atreides interfaces, for tests.

These exist to prove the point of the abstractions: the whole pipeline can be
driven without a neural network, a video file, or a GPU.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from atreides_vision.models import Detection, Track
from atreides_vision.video.source import VideoMetadata

__all__ = ["FakeDetector", "FakeSink", "FakeSource", "FakeTracker"]


class FakeDetector:
    """A :class:`PersonDetector` that replays a scripted list of detections.

    The last scripted entry repeats once the script runs out, so a one-element
    script means "the same detection on every frame".
    """

    def __init__(self, per_frame: Sequence[Sequence[Detection]]) -> None:
        self._per_frame = [list(frame) for frame in per_frame]
        self.calls = 0

    @property
    def name(self) -> str:
        return "Fake Detector"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if not self._per_frame:
            self.calls += 1
            return []
        index = min(self.calls, len(self._per_frame) - 1)
        self.calls += 1
        return list(self._per_frame[index])


class FakeTracker:
    """A :class:`MultiObjectTracker` that assigns stable ids by detection order."""

    def __init__(self) -> None:
        self.calls = 0
        self.reset_calls = 0

    @property
    def name(self) -> str:
        return "Fake Tracker"

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        self.calls += 1
        return [
            Track(track_id=index, bbox_xyxy=detection.bbox_xyxy, confidence=detection.confidence)
            for index, detection in enumerate(detections)
        ]

    def reset(self) -> None:
        self.reset_calls += 1


class FakeSource:
    """An in-memory :class:`VideoSource`."""

    def __init__(self, frames: Sequence[np.ndarray], metadata: VideoMetadata | None = None) -> None:
        self._frames = list(frames)
        height, width = self._frames[0].shape[:2] if self._frames else (120, 160)
        self._metadata = metadata or VideoMetadata(
            width=width, height=height, fps=20.0, frame_count=len(self._frames)
        )
        self.released = False

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    def frames(self) -> Iterator[np.ndarray]:
        yield from self._frames

    def release(self) -> None:
        self.released = True


class FakeSink:
    """An in-memory :class:`VideoSink` that retains everything written to it."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path("artifacts/fake.mp4")
        self.frames: list[np.ndarray] = []
        self.released = False

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True

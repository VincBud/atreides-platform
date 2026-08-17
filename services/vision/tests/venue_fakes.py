"""In-memory streams for venue ingestion tests.

Venue ingestion is about time and ordering, so none of these decode anything.
The whole fan-in, including heterogeneous frame rates, start offsets, stream
restarts and bounded state, is exercised without a video file, a model or a GPU.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from atreides_vision.timing import FrameRateSource, StreamFrame, StreamRef, TimeUncertainty
from atreides_vision.video.source import VideoMetadata

__all__ = [
    "FRAME_H",
    "FRAME_W",
    "CountingSource",
    "FakeStreamCursor",
    "frame_image",
    "stream_frames",
]

FRAME_H, FRAME_W = 6, 8


def frame_image(value: int) -> np.ndarray:
    """A tiny distinguishable frame, so a wrong frame is visible in a failure."""
    return np.full((FRAME_H, FRAME_W, 3), value % 256, dtype=np.uint8)


def stream_frames(
    stream: StreamRef,
    count: int,
    fps: float,
    offset_seconds: float = 0.0,
    timing: TimeUncertainty | None = None,
) -> list[StreamFrame]:
    """``count`` frames of one stream, timed exactly as a real cursor times them."""
    carried = TimeUncertainty.unknown() if timing is None else timing
    frames: list[StreamFrame] = []
    for frame_index in range(1, count + 1):
        source_seconds = (frame_index - 1) / fps
        frames.append(
            StreamFrame(
                stream=stream,
                frame_index=frame_index,
                source_seconds=source_seconds,
                venue_seconds=source_seconds + offset_seconds,
                timing=carried,
                image=frame_image(frame_index),
            )
        )
    return frames


class FakeStreamCursor:
    """A :class:`StreamCursor` over pre-built frames, holding one at a time.

    ``frames_materialised`` counts how many frames it has handed out, so a test
    can prove the merge never runs ahead of the schedule.
    """

    def __init__(self, stream: StreamRef, frames: Sequence[StreamFrame]) -> None:
        self._stream = stream
        self._frames = list(frames)
        self._position = 0
        self._pending: StreamFrame | None = None
        self.peeks = 0
        self.frames_materialised = 0

    @property
    def stream(self) -> StreamRef:
        return self._stream

    @property
    def frames_consumed(self) -> int:
        return self._position

    def peek(self) -> StreamFrame | None:
        self.peeks += 1
        if self._pending is None and self._position < len(self._frames):
            self._pending = self._frames[self._position]
            self.frames_materialised += 1
        return self._pending

    def advance(self) -> None:
        assert self._pending is not None, "advance() without a peeked frame"
        self._pending = None
        self._position += 1


class CountingSource:
    """A :class:`VideoSource` that records how many frames were actually pulled.

    The difference between frames pulled and frames consumed is exactly the
    number held in flight, which is what bounds the venue loop's memory.
    """

    def __init__(
        self,
        frame_count: int,
        fps: float = 25.0,
        fps_source: FrameRateSource = FrameRateSource.CONTAINER,
        width: int = FRAME_W,
        height: int = FRAME_H,
    ) -> None:
        self._frame_count = frame_count
        self._metadata = VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            fps_source=fps_source,
        )
        self.frames_pulled = 0
        self.released = False

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    def frames(self) -> Iterator[np.ndarray]:
        for index in range(1, self._frame_count + 1):
            self.frames_pulled += 1
            yield frame_image(index)

    def release(self) -> None:
        self.released = True

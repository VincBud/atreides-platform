"""Video output. Writes annotated frames to an MP4 file."""

from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

import cv2
import numpy as np

from atreides_vision.errors import VideoSinkError

logger = logging.getLogger(__name__)

__all__ = ["Mp4VideoSink", "VideoSink"]

#: Codec used for output. mp4v is available in every stock OpenCV wheel.
DEFAULT_FOURCC = "mp4v"


class VideoSink(Protocol):
    """A writable destination for annotated frames."""

    def write(self, frame: np.ndarray) -> None: ...

    def release(self) -> None: ...


class Mp4VideoSink:
    """Writes annotated frames to an MP4, preserving source resolution and FPS.

    Always use it as a context manager, or call :meth:`release` from a
    ``finally`` block — an unreleased writer produces a truncated, unplayable
    file::

        with Mp4VideoSink(path, width, height, fps) as sink:
            sink.write(frame)

    Raises:
        VideoSinkError: if the output cannot be created, or a frame's size does
            not match the size the sink was opened with.
    """

    def __init__(
        self,
        path: Path | str,
        width: int,
        height: int,
        fps: float,
        fourcc: str = DEFAULT_FOURCC,
    ) -> None:
        if width <= 0 or height <= 0:
            raise VideoSinkError(f"Output resolution must be positive, got {width}x{height}")
        if fps <= 0:
            raise VideoSinkError(f"Output FPS must be positive, got {fps}")

        self._path = Path(path)
        self._width = width
        self._height = height
        self._fps = float(fps)
        self._frames_written = 0
        self._released = False

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VideoSinkError(
                f"Could not create output directory {self._path.parent}: {exc}"
            ) from exc

        self._writer = cv2.VideoWriter(
            str(self._path),
            cv2.VideoWriter.fourcc(*fourcc),
            self._fps,
            (width, height),
        )
        if not self._writer.isOpened():
            self._writer.release()
            raise VideoSinkError(
                f"Could not open video writer for {self._path} "
                f"(codec '{fourcc}', {width}x{height} @ {self._fps:.2f} FPS). "
                "Check that the output directory is writable and the path ends in .mp4."
            )

    @property
    def path(self) -> Path:
        return self._path

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def write(self, frame: np.ndarray) -> None:
        """Append one BGR frame to the output file."""
        if self._released:
            raise VideoSinkError(f"Cannot write to a released sink: {self._path}")
        height, width = frame.shape[:2]
        if (width, height) != (self._width, self._height):
            raise VideoSinkError(
                f"Frame size {width}x{height} does not match the sink's "
                f"{self._width}x{self._height}. Annotation must not resize frames."
            )
        self._writer.write(frame)
        self._frames_written += 1

    def release(self) -> None:
        """Finalise the file. Idempotent, and safe to call after an exception."""
        if not getattr(self, "_released", False):
            self._writer.release()
            self._released = True
            logger.debug("Released video sink %s after %d frames", self._path, self._frames_written)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Runs on the exception path too, so a crash still leaves a readable file.
        self.release()

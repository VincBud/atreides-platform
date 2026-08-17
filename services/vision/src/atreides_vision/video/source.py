"""Video input. Wraps OpenCV capture behind an Atreides-owned interface."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

import cv2
import numpy as np

from atreides_vision.errors import VideoSourceError
from atreides_vision.timing import FrameRateSource

logger = logging.getLogger(__name__)

__all__ = ["FileVideoSource", "VideoMetadata", "VideoSource", "read_reference_frame"]

#: Frame rate assumed when a container reports a nonsensical value.
DEFAULT_FALLBACK_FPS = 25.0


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Properties of a video stream, read once when it is opened."""

    width: int
    height: int
    fps: float
    #: Total frames if the container reports a usable value, else ``None``.
    frame_count: int | None
    #: Where :attr:`fps` came from. Every source-relative timestamp is derived
    #: from that number, so a fallback constant and a real container value must
    #: not look alike downstream. Defaults to ``CONTAINER`` so existing callers
    #: constructing metadata directly are unaffected.
    fps_source: FrameRateSource = FrameRateSource.CONTAINER

    @property
    def resolution(self) -> str:
        """``"1920x1080"`` — for logging."""
        return f"{self.width}x{self.height}"

    @property
    def duration_seconds(self) -> float | None:
        if self.frame_count is None or self.fps <= 0:
            return None
        return self.frame_count / self.fps


def read_reference_frame(path: Path | str) -> np.ndarray:
    """Read a single frame from an image file, or the first frame of a video.

    Calibration diagnostics need one still to draw on, and an operator will
    reasonably reach for either a screenshot or the clip itself. Images are
    tried first because ``cv2.imread`` simply returns ``None`` for a video
    container, which makes the fallback unambiguous.

    Args:
        path: An image or video file.

    Returns:
        A ``(H, W, 3)`` uint8 BGR frame.

    Raises:
        VideoSourceError: if the file is missing, is not a file, or cannot be
            decoded as either an image or a video.
    """
    frame_path = Path(path)
    if not frame_path.exists():
        raise VideoSourceError(f"Reference frame not found: {frame_path}")
    if not frame_path.is_file():
        raise VideoSourceError(f"Reference frame path is not a file: {frame_path}")

    image = cv2.imread(str(frame_path))
    if image is not None and image.size > 0:
        return np.asarray(image)

    with FileVideoSource(frame_path) as source:
        for frame in source.frames():
            return frame

    raise VideoSourceError(
        f"Could not read a frame from {frame_path}: it decoded as neither an image nor a "
        "video with at least one frame."
    )


class VideoSource(Protocol):
    """A readable source of video frames."""

    @property
    def metadata(self) -> VideoMetadata: ...

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames as ``(H, W, 3)`` uint8 BGR arrays until the stream ends."""
        ...

    def release(self) -> None:
        """Release the underlying resource. Safe to call more than once."""
        ...


class FileVideoSource:
    """Reads frames from a video file (MP4 and anything else OpenCV supports).

    Usable as a context manager, which is the recommended form::

        with FileVideoSource(path) as source:
            for frame in source.frames():
                ...

    ``declared_fps`` overrides whatever the container reports. Real CCTV exports
    routinely carry a nonsensical or plainly wrong frame rate, and every
    source-relative timestamp is derived from that number; letting an operator
    state the true rate is the only honest remedy, and it is recorded as
    :attr:`~atreides_vision.timing.FrameRateSource.DECLARED` rather than being
    passed off as the container's own value.

    Raises:
        VideoSourceError: if the path does not exist, is not a file, cannot be
            decoded, or ``declared_fps`` is not a positive finite number. The
            failure is always explicit — never a silent zero-frame run.
    """

    def __init__(self, path: Path | str, declared_fps: float | None = None) -> None:
        if declared_fps is not None and (not np.isfinite(declared_fps) or declared_fps <= 0):
            raise VideoSourceError(
                f"declared_fps must be a positive, finite frame rate, got {declared_fps!r}. "
                "Omit it to use the rate the container reports."
            )
        self._declared_fps = declared_fps

        self._path = Path(path)
        if not self._path.exists():
            raise VideoSourceError(
                f"Input video not found: {self._path}. "
                "Place your MP4 in data/input/ and pass it with --input."
            )
        if not self._path.is_file():
            raise VideoSourceError(f"Input path is not a file: {self._path}")

        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            self._capture.release()
            raise VideoSourceError(
                f"Could not open video: {self._path}. The file may be corrupt or use a "
                "codec this OpenCV build cannot decode. H.264 MP4 is the safe choice."
            )

        self._metadata = self._read_metadata()
        if self._metadata.width <= 0 or self._metadata.height <= 0:
            self.release()
            raise VideoSourceError(
                f"Video reports an invalid resolution ({self._metadata.width}x"
                f"{self._metadata.height}): {self._path}"
            )
        self._released = False

    def _read_metadata(self) -> VideoMetadata:
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        raw_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if self._declared_fps is not None:
            logger.info(
                "Using the declared frame rate %.4f FPS for %s; the container reported %r.",
                self._declared_fps,
                self._path,
                raw_fps,
            )
            fps = self._declared_fps
            fps_source = FrameRateSource.DECLARED
        elif not np.isfinite(raw_fps) or raw_fps <= 0:
            logger.warning(
                "Video reports an unusable frame rate (%r); assuming %.1f FPS for output timing. "
                "Timestamps derived from an assumed rate drift by an unknown, growing amount — "
                "declare the real rate for this source if it matters.",
                raw_fps,
                DEFAULT_FALLBACK_FPS,
            )
            fps = DEFAULT_FALLBACK_FPS
            fps_source = FrameRateSource.ASSUMED
        else:
            fps = raw_fps
            fps_source = FrameRateSource.CONTAINER

        raw_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = raw_count if raw_count > 0 else None

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            fps_source=fps_source,
        )

    @property
    def path(self) -> Path:
        return self._path

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    def frames(self) -> Iterator[np.ndarray]:
        """Yield decoded BGR frames until the stream is exhausted."""
        if self._released:
            raise VideoSourceError(f"Video source has already been released: {self._path}")
        while True:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                return
            yield frame

    def release(self) -> None:
        if not getattr(self, "_released", False):
            self._capture.release()
            self._released = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

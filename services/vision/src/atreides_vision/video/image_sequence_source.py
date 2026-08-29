"""A :class:`VideoSource` backed by an ordered list of still-image files.

Some multi-camera video arrives as numbered image files rather than as a
container — public datasets such as WILDTRACK ship synchronized frames that
way, and so do some CCTV export tools. This source lets those frames enter the
ordinary Atreides runtime (``VideoProcessor`` -> detection -> tracking ->
behaviour -> ``VenueSession``) through the existing :class:`VideoSource`
protocol, with no change to any camera-local layer and no second video
abstraction. It lives beside :class:`FileVideoSource` because, like it, it is
the one place OpenCV decoding is allowed.

It is deliberately minimal:

* it decodes the files it is handed, **in the order it is handed them** — it
  never re-sorts, resamples, interpolates or duplicates a frame;
* every frame must share the first frame's pixel dimensions, or the run fails
  rather than silently adapting;
* the frame rate is **declared** by the caller (there is no container to read
  one from) and recorded as
  :attr:`~atreides_vision.timing.FrameRateSource.DECLARED`, so
  ``source_seconds = (frame_index - 1) / declared_fps`` downstream is an honest
  elapsed time and not a guess.

No wall clock is read anywhere.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

import cv2
import numpy as np

from atreides_vision.errors import VideoSourceError
from atreides_vision.timing import FrameRateSource
from atreides_vision.video.source import VideoMetadata

__all__ = ["ImageSequenceVideoSource"]


class ImageSequenceVideoSource:
    """Presents an ordered list of image files as a single readable stream.

    Args:
        image_paths: The frames, in the exact order they should be yielded.
            Every path must already exist and be a file; the sequence must be
            non-empty. Ordering is the caller's responsibility — this class
            preserves it verbatim and never re-sorts.
        declared_fps: The true capture rate of the sequence, a positive finite
            number. There is no container to read a rate from, so the caller
            states it and it is recorded as ``DECLARED``.

    Raises:
        VideoSourceError: if ``image_paths`` is empty, if any path is missing or
            not a file, if ``declared_fps`` is not a positive finite number, or
            if the first image cannot be decoded. Per-frame decode failures and
            mid-sequence dimension changes are raised from :meth:`frames`.
    """

    def __init__(self, image_paths: Sequence[Path], *, declared_fps: float) -> None:
        if not np.isfinite(declared_fps) or declared_fps <= 0:
            raise VideoSourceError(
                f"declared_fps must be a positive, finite frame rate, got {declared_fps!r}."
            )
        paths = tuple(Path(p) for p in image_paths)
        if not paths:
            raise VideoSourceError(
                "ImageSequenceVideoSource needs at least one image. An empty sequence has no "
                "frame geometry and no frames; it is a caller error, not an empty recording."
            )
        for path in paths:
            if not path.exists():
                raise VideoSourceError(f"Image sequence frame not found: {path}")
            if not path.is_file():
                raise VideoSourceError(f"Image sequence frame is not a file: {path}")

        self._paths = paths
        self._declared_fps = float(declared_fps)
        first = self._decode(paths[0])
        height, width = first.shape[:2]
        self._metadata = VideoMetadata(
            width=int(width),
            height=int(height),
            fps=self._declared_fps,
            frame_count=len(paths),
            fps_source=FrameRateSource.DECLARED,
        )
        self._released = False
        self._consumed = False

    @staticmethod
    def _decode(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise VideoSourceError(
                f"Could not decode image sequence frame: {path}. The file is missing, "
                "truncated or not an image format this OpenCV build reads."
            )
        return np.asarray(image)

    @property
    def paths(self) -> tuple[Path, ...]:
        return self._paths

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    def frames(self) -> Iterator[np.ndarray]:
        """Yield each image as a ``(H, W, 3)`` uint8 BGR array, in order.

        Raises:
            VideoSourceError: if the source has been released, if :meth:`frames`
                has already been consumed once, if an image fails to decode, or
                if an image's dimensions differ from the first frame's.
        """
        if self._released:
            raise VideoSourceError(
                f"Image sequence source has already been released: {self._paths[0]} and "
                f"{len(self._paths) - 1} more."
            )
        if self._consumed:
            raise VideoSourceError(
                "Image sequence source frames() has already been consumed. Like a video "
                "container, this stream is read once."
            )
        self._consumed = True
        expected = (self._metadata.height, self._metadata.width)
        for path in self._paths:
            image = self._decode(path)
            if image.shape[:2] != expected:
                raise VideoSourceError(
                    f"Image sequence frame {path} is {image.shape[1]}x{image.shape[0]}, but the "
                    f"sequence started at {self._metadata.width}x{self._metadata.height}. A "
                    "stream cannot change resolution mid-flight."
                )
            yield image

    def release(self) -> None:
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

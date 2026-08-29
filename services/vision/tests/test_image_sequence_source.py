"""Contract tests for :class:`ImageSequenceVideoSource`.

Tiny synthetic images written with OpenCV — never the real WILDTRACK corpus.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from atreides_vision.errors import VideoSourceError
from atreides_vision.timing import FrameRateSource, StreamRef
from atreides_vision.venue.session import VideoSourceCursor
from atreides_vision.venue.timebase import VenueTimebase
from atreides_vision.video import ImageSequenceVideoSource

WIDTH, HEIGHT = 16, 12


def _write_sequence(
    directory: Path, count: int, *, width: int = WIDTH, height: int = HEIGHT
) -> list[Path]:
    paths: list[Path] = []
    for index in range(count):
        path = directory / f"{index:04d}.png"
        frame = np.full((height, width, 3), fill_value=index + 1, dtype=np.uint8)
        assert cv2.imwrite(str(path), frame)
        paths.append(path)
    return paths


def test_metadata_reports_declared_rate_and_geometry(tmp_path: Path) -> None:
    paths = _write_sequence(tmp_path, 3)
    source = ImageSequenceVideoSource(paths, declared_fps=2.0)
    meta = source.metadata
    assert (meta.width, meta.height) == (WIDTH, HEIGHT)
    assert meta.fps == 2.0
    assert meta.fps_source is FrameRateSource.DECLARED
    assert meta.frame_count == 3


def test_frames_are_bgr_uint8_in_given_order(tmp_path: Path) -> None:
    paths = _write_sequence(tmp_path, 4)
    with ImageSequenceVideoSource(paths, declared_fps=2.0) as source:
        frames = list(source.frames())
    assert [f.shape for f in frames] == [(HEIGHT, WIDTH, 3)] * 4
    assert all(f.dtype == np.uint8 for f in frames)
    assert [int(f[0, 0, 0]) for f in frames] == [1, 2, 3, 4]


def test_order_is_preserved_verbatim_not_resorted(tmp_path: Path) -> None:
    paths = _write_sequence(tmp_path, 3)
    reversed_paths = list(reversed(paths))
    with ImageSequenceVideoSource(reversed_paths, declared_fps=2.0) as source:
        first_pixels = [int(f[0, 0, 0]) for f in source.frames()]
    assert first_pixels == [3, 2, 1]


def test_source_seconds_through_cursor_step_by_half_second(tmp_path: Path) -> None:
    paths = _write_sequence(tmp_path, 5)
    source = ImageSequenceVideoSource(paths, declared_fps=2.0)
    stream = StreamRef("cam_a", 0)
    timebase = VenueTimebase()
    timebase.register(stream, fps_source=source.metadata.fps_source)
    cursor = VideoSourceCursor(stream, source, timebase)
    observed: list[float] = []
    while (frame := cursor.peek()) is not None:
        observed.append(frame.source_seconds)
        cursor.advance()
    assert observed == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_empty_sequence_is_rejected_explicitly(tmp_path: Path) -> None:
    with pytest.raises(VideoSourceError, match="at least one image"):
        ImageSequenceVideoSource([], declared_fps=2.0)


def test_non_positive_declared_fps_is_rejected(tmp_path: Path) -> None:
    paths = _write_sequence(tmp_path, 1)
    with pytest.raises(VideoSourceError, match="declared_fps"):
        ImageSequenceVideoSource(paths, declared_fps=0.0)


def test_missing_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(VideoSourceError, match="not found"):
        ImageSequenceVideoSource([tmp_path / "nope.png"], declared_fps=2.0)


def test_unreadable_image_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    with pytest.raises(VideoSourceError, match="decode"):
        ImageSequenceVideoSource([bad], declared_fps=2.0)


def test_mid_sequence_dimension_change_is_rejected(tmp_path: Path) -> None:
    paths = _write_sequence(tmp_path, 2)
    odd = tmp_path / "0002.png"
    assert cv2.imwrite(str(odd), np.zeros((HEIGHT + 4, WIDTH, 3), dtype=np.uint8))
    paths.append(odd)
    source = ImageSequenceVideoSource(paths, declared_fps=2.0)
    with pytest.raises(VideoSourceError, match="resolution mid-flight"):
        list(source.frames())


def test_release_is_idempotent_and_blocks_frames(tmp_path: Path) -> None:
    paths = _write_sequence(tmp_path, 2)
    source = ImageSequenceVideoSource(paths, declared_fps=2.0)
    source.release()
    source.release()
    with pytest.raises(VideoSourceError, match="released"):
        list(source.frames())


def test_frames_consumed_once(tmp_path: Path) -> None:
    paths = _write_sequence(tmp_path, 2)
    source = ImageSequenceVideoSource(paths, declared_fps=2.0)
    list(source.frames())
    with pytest.raises(VideoSourceError, match="consumed"):
        list(source.frames())


def test_deterministic_frame_bytes_across_two_reads(tmp_path: Path) -> None:
    paths = _write_sequence(tmp_path, 3)
    a = list(ImageSequenceVideoSource(paths, declared_fps=2.0).frames())
    b = list(ImageSequenceVideoSource(paths, declared_fps=2.0).frames())
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))

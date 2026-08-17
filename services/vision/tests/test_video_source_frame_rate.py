"""Where a source's frame rate comes from, and what happens when it is unusable.

Every source-relative timestamp in the system is ``(frame_index - 1) / fps``, so
this one number decides whether a measured second is a second. These tests cover
the three provenances a real deployment produces: a container that reports a
usable rate, a container that reports nonsense and gets a fallback, and an
operator who states the true rate because the container lies.

No weights, no network. The nonsense-frame-rate case cannot be written by
``cv2.VideoWriter`` — it refuses to open with an invalid rate — so that branch is
driven through a stand-in capture, and everything else uses real synthetic video.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from atreides_vision.errors import VenueSessionError, VideoSourceError
from atreides_vision.timing import FrameRateSource, StreamRef
from atreides_vision.venue.timebase import VenueTimebase
from atreides_vision.video.source import DEFAULT_FALLBACK_FPS, FileVideoSource

WIDTH, HEIGHT = 64, 48
CAM = StreamRef("cam_a", 0)


class StandInCapture:
    """The smallest surface :class:`FileVideoSource` uses from a capture.

    Exists to produce the one thing OpenCV will not write for us: a container
    whose reported frame rate is unusable.
    """

    def __init__(self, fps: float, frame_count: int = 3) -> None:
        self._fps = fps
        self._frame_count = frame_count
        self._read = 0
        self.released = False

    def isOpened(self) -> bool:
        """Named for the cv2 method it stands in for."""
        return True

    def get(self, prop: int) -> float:
        return {
            cv2.CAP_PROP_FRAME_WIDTH: float(WIDTH),
            cv2.CAP_PROP_FRAME_HEIGHT: float(HEIGHT),
            cv2.CAP_PROP_FPS: self._fps,
            cv2.CAP_PROP_FRAME_COUNT: float(self._frame_count),
        }[prop]

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._read >= self._frame_count:
            return False, None
        self._read += 1
        return True, np.full((HEIGHT, WIDTH, 3), self._read, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class CaptureOpener:
    """Opens a real file whose properties come from a stand-in capture."""

    def __init__(self, path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._path = path
        self._monkeypatch = monkeypatch

    def _install(self, fps: float) -> None:
        # Patched on the cv2 module the video layer holds a reference to, and
        # undone by monkeypatch at teardown.
        self._monkeypatch.setattr(cv2, "VideoCapture", lambda _path: StandInCapture(fps=fps))

    def reporting(self, fps: float) -> FileVideoSource:
        """A source whose container reports ``fps``."""
        self._install(fps)
        return FileVideoSource(self._path)

    def declaring(self, fps: float, declared_fps: float) -> FileVideoSource:
        """The same, with the operator declaring the real rate."""
        self._install(fps)
        return FileVideoSource(self._path, declared_fps=declared_fps)


@pytest.fixture
def container(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CaptureOpener:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"a real file, whose bytes this test never decodes")
    return CaptureOpener(path, monkeypatch)


class TestContainerFrameRate:
    def test_a_usable_container_rate_is_used_and_marked_as_such(
        self, synthetic_video: Path
    ) -> None:
        with FileVideoSource(synthetic_video) as source:
            metadata = source.metadata

        assert metadata.fps == pytest.approx(20.0, abs=0.5)
        assert metadata.fps_source is FrameRateSource.CONTAINER
        assert metadata.fps_source.is_trustworthy

    @pytest.mark.parametrize("reported", [0.0, -1.0, math.nan, math.inf])
    def test_an_unusable_container_rate_falls_back_and_is_marked_assumed(
        self, container: CaptureOpener, reported: float, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            caplog.at_level(logging.WARNING, logger="atreides_vision.video.source"),
            container.reporting(reported) as source,
        ):
            metadata = source.metadata

        assert metadata.fps == DEFAULT_FALLBACK_FPS
        assert metadata.fps_source is FrameRateSource.ASSUMED
        assert not metadata.fps_source.is_trustworthy
        assert any("unusable frame rate" in record.message for record in caplog.records)

    def test_the_fallback_warning_names_the_remedy(
        self, container: CaptureOpener, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            caplog.at_level(logging.WARNING, logger="atreides_vision.video.source"),
            container.reporting(0.0),
        ):
            pass

        assert any("declare the real rate" in record.message for record in caplog.records)

    def test_the_frames_still_decode_after_a_fallback(self, container: CaptureOpener) -> None:
        with container.reporting(0.0) as source:
            frames = list(source.frames())

        assert len(frames) == 3
        assert all(frame.shape == (HEIGHT, WIDTH, 3) for frame in frames)


class TestDeclaredFrameRate:
    def test_a_declared_rate_overrides_the_container(self, synthetic_video: Path) -> None:
        with FileVideoSource(synthetic_video, declared_fps=12.5) as source:
            metadata = source.metadata

        assert metadata.fps == 12.5
        assert metadata.fps_source is FrameRateSource.DECLARED
        assert metadata.fps_source.is_trustworthy

    def test_declaring_the_rate_replaces_an_assumed_one(self, container: CaptureOpener) -> None:
        """The remedy the fallback warning points at, exercised end to end."""
        with container.declaring(0.0, 30.0) as source:
            metadata = source.metadata

        assert metadata.fps == 30.0
        assert metadata.fps_source is FrameRateSource.DECLARED

    def test_the_declaration_is_logged_with_what_the_container_claimed(
        self, container: CaptureOpener, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            caplog.at_level(logging.INFO, logger="atreides_vision.video.source"),
            container.declaring(0.0, 30.0),
        ):
            pass

        assert any("declared frame rate" in record.message for record in caplog.records)

    @pytest.mark.parametrize("declared", [0.0, -25.0, math.nan, math.inf, -math.inf])
    def test_an_unusable_declaration_is_refused(
        self, synthetic_video: Path, declared: float
    ) -> None:
        with pytest.raises(VideoSourceError, match="positive, finite frame rate"):
            FileVideoSource(synthetic_video, declared_fps=declared)

    def test_the_refusal_says_how_to_opt_out(self, synthetic_video: Path) -> None:
        with pytest.raises(VideoSourceError, match="Omit it"):
            FileVideoSource(synthetic_video, declared_fps=0.0)

    def test_omitting_the_declaration_changes_nothing_else(self, synthetic_video: Path) -> None:
        with FileVideoSource(synthetic_video) as source:
            metadata = source.metadata

        assert metadata.fps_source is FrameRateSource.CONTAINER
        assert (metadata.width, metadata.height) == (160, 120)
        assert metadata.frame_count == 10


class TestProvenanceReachesTheVenueTimebase:
    """The flag is not decoration: it decides what the timebase will believe.

    A real NVR export reaches the timebase through exactly this path, so these
    exercise the flag as the source actually sets it rather than as a test
    constructs it.
    """

    def test_an_assumed_rate_is_placed_but_never_gains_an_accuracy(
        self, container: CaptureOpener, caplog: pytest.LogCaptureFixture
    ) -> None:
        with container.reporting(0.0) as source:
            fps_source = source.metadata.fps_source

        timebase = VenueTimebase()
        with caplog.at_level(logging.WARNING, logger="atreides_vision.venue.timebase"):
            timebase.register(
                CAM, fps_source=fps_source, offset_seconds=10.0, offset_accuracy_seconds=0.5
            )

        # Placed, so the deployment is representable at all...
        assert timebase.offset_seconds(CAM) == 10.0
        # ...and unaligned, so no accuracy is claimed for a guessed rate.
        assert not timebase.uncertainty(CAM).is_established
        assert any("discarding" in record.getMessage() for record in caplog.records)

    def test_declaring_the_real_rate_keeps_the_accuracy(self, container: CaptureOpener) -> None:
        with container.declaring(0.0, 25.0) as source:
            fps_source = source.metadata.fps_source

        timebase = VenueTimebase()
        timebase.register(
            CAM, fps_source=fps_source, offset_seconds=10.0, offset_accuracy_seconds=0.5
        )

        assert timebase.uncertainty(CAM).is_established

    def test_an_offset_with_no_accuracy_is_refused_whatever_the_rate(
        self, container: CaptureOpener
    ) -> None:
        with container.reporting(0.0) as source:
            fps_source = source.metadata.fps_source

        with pytest.raises(VenueSessionError, match="needs an accuracy"):
            VenueTimebase().register(CAM, fps_source=fps_source, offset_seconds=10.0)

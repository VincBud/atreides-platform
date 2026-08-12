"""Video source metadata and sink lifecycle, using synthetic footage only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atreides_vision.errors import VideoSinkError, VideoSourceError
from atreides_vision.video.sink import Mp4VideoSink
from atreides_vision.video.source import FileVideoSource, VideoMetadata


class TestVideoMetadata:
    def test_resolution_string(self) -> None:
        assert VideoMetadata(1920, 1080, 25.0, 100).resolution == "1920x1080"

    def test_duration_when_frame_count_known(self) -> None:
        assert VideoMetadata(640, 360, 25.0, 250).duration_seconds == 10.0

    def test_duration_unknown_without_frame_count(self) -> None:
        assert VideoMetadata(640, 360, 25.0, None).duration_seconds is None


class TestFileVideoSource:
    def test_missing_file_raises_clear_error(self, tmp_path: Path) -> None:
        with pytest.raises(VideoSourceError) as excinfo:
            FileVideoSource(tmp_path / "nope.mp4")
        assert "not found" in str(excinfo.value)
        assert "data/input/" in str(excinfo.value)

    def test_directory_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(VideoSourceError, match="not a file"):
            FileVideoSource(tmp_path)

    def test_undecodable_file_raises_instead_of_yielding_nothing(self, tmp_path: Path) -> None:
        junk = tmp_path / "junk.mp4"
        junk.write_bytes(b"this is definitely not an mp4 container")
        with pytest.raises(VideoSourceError, match="Could not open video"):
            FileVideoSource(junk)

    def test_reads_metadata_from_synthetic_video(self, synthetic_video: Path) -> None:
        with FileVideoSource(synthetic_video) as source:
            metadata = source.metadata
            assert (metadata.width, metadata.height) == (160, 120)
            assert metadata.fps == pytest.approx(20.0, abs=0.5)
            assert metadata.frame_count == 10

    def test_iterates_all_frames(self, synthetic_video: Path) -> None:
        with FileVideoSource(synthetic_video) as source:
            frames = list(source.frames())
        assert len(frames) == 10
        assert all(frame.shape == (120, 160, 3) for frame in frames)
        assert all(frame.dtype == np.uint8 for frame in frames)

    def test_context_manager_releases(self, synthetic_video: Path) -> None:
        source = FileVideoSource(synthetic_video)
        with source:
            pass
        with pytest.raises(VideoSourceError, match="already been released"):
            list(source.frames())

    def test_release_is_idempotent(self, synthetic_video: Path) -> None:
        source = FileVideoSource(synthetic_video)
        source.release()
        source.release()


class TestMp4VideoSink:
    def test_writes_a_readable_file(self, tmp_path: Path) -> None:
        output = tmp_path / "out.mp4"
        with Mp4VideoSink(output, width=64, height=48, fps=10.0) as sink:
            for _ in range(5):
                sink.write(np.zeros((48, 64, 3), dtype=np.uint8))
            assert sink.frames_written == 5

        assert output.exists() and output.stat().st_size > 0
        with FileVideoSource(output) as source:
            assert source.metadata.width == 64
            assert source.metadata.height == 48
            assert len(list(source.frames())) == 5

    def test_creates_missing_output_directory(self, tmp_path: Path) -> None:
        output = tmp_path / "nested" / "deeper" / "out.mp4"
        with Mp4VideoSink(output, width=32, height=32, fps=5.0) as sink:
            sink.write(np.zeros((32, 32, 3), dtype=np.uint8))
        assert output.exists()

    def test_rejects_mismatched_frame_size(self, tmp_path: Path) -> None:
        with (
            Mp4VideoSink(tmp_path / "out.mp4", width=64, height=48, fps=10.0) as sink,
            pytest.raises(VideoSinkError, match="does not match"),
        ):
            sink.write(np.zeros((10, 10, 3), dtype=np.uint8))

    @pytest.mark.parametrize(
        ("width", "height", "fps"),
        [(0, 48, 10.0), (64, -1, 10.0), (64, 48, 0.0)],
    )
    def test_rejects_invalid_geometry(
        self, tmp_path: Path, width: int, height: int, fps: float
    ) -> None:
        with pytest.raises(VideoSinkError):
            Mp4VideoSink(tmp_path / "out.mp4", width=width, height=height, fps=fps)

    def test_released_on_exception_so_file_is_finalised(self, tmp_path: Path) -> None:
        output = tmp_path / "out.mp4"
        sink = Mp4VideoSink(output, width=32, height=32, fps=5.0)
        with pytest.raises(RuntimeError), sink:
            sink.write(np.zeros((32, 32, 3), dtype=np.uint8))
            raise RuntimeError("pipeline blew up mid-run")

        assert output.exists() and output.stat().st_size > 0
        with pytest.raises(VideoSinkError, match="released sink"):
            sink.write(np.zeros((32, 32, 3), dtype=np.uint8))

    def test_preserves_source_geometry_end_to_end(
        self, synthetic_video: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "copy.mp4"
        with FileVideoSource(synthetic_video) as source:
            metadata = source.metadata
            with Mp4VideoSink(
                output, width=metadata.width, height=metadata.height, fps=metadata.fps
            ) as sink:
                for frame in source.frames():
                    sink.write(frame)

        with FileVideoSource(output) as result:
            assert result.metadata.resolution == metadata.resolution
            assert result.metadata.fps == pytest.approx(metadata.fps, abs=0.5)

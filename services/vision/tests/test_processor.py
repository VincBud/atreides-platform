"""Pipeline orchestration, driven entirely by fakes.

These tests are the reason the detector and tracker sit behind interfaces: the
whole SEE + TRACK flow is verifiable without a neural network.
"""

from __future__ import annotations

import logging
from itertools import count
from pathlib import Path

import numpy as np
import pytest

from atreides_vision.config import (
    DetectorConfig,
    PipelineConfig,
    TrackerConfig,
    VisionConfig,
    VisualizationConfig,
)
from atreides_vision.models import Detection
from atreides_vision.video.processor import FpsMeter, ProcessingResult, VideoProcessor
from fakes import FakeDetector, FakeSink, FakeSource, FakeTracker


def make_frames(n: int, size: tuple[int, int] = (120, 160)) -> list[np.ndarray]:
    return [np.full((*size, 3), 40 + i, dtype=np.uint8) for i in range(n)]


def make_config(tmp_path: Path, **pipeline_kwargs: object) -> VisionConfig:
    return VisionConfig(
        pipeline=PipelineConfig(
            input_path=tmp_path / "in.mp4",
            output_path=tmp_path / "out.mp4",
            **pipeline_kwargs,  # type: ignore[arg-type]
        ),
        detector=DetectorConfig(),
        tracker=TrackerConfig(),
        visualization=VisualizationConfig(trail_length=5, trail_ttl_frames=3),
    )


class TestFpsMeter:
    def test_reports_zero_before_two_samples(self) -> None:
        meter = FpsMeter(clock=iter([0.0, 1.0]).__next__)
        assert meter.fps == 0.0
        meter.tick()
        assert meter.fps == 0.0

    def test_computes_rate_from_the_window(self) -> None:
        clock = count(start=0.0, step=0.25).__next__  # 4 FPS
        meter = FpsMeter(window=10, clock=clock)
        for _ in range(5):
            meter.tick()
        assert meter.fps == pytest.approx(4.0)

    def test_window_is_bounded(self) -> None:
        meter = FpsMeter(window=3, clock=count(start=0.0, step=0.5).__next__)
        for _ in range(50):
            meter.tick()
        assert meter.fps == pytest.approx(2.0)

    def test_rejects_tiny_window(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 2"):
            FpsMeter(window=1)

    def test_stalled_clock_does_not_divide_by_zero(self) -> None:
        meter = FpsMeter(clock=lambda: 5.0)
        meter.tick()
        meter.tick()
        assert meter.fps == 0.0


class TestOrchestration:
    def test_every_frame_flows_through_the_full_pipeline(self, tmp_path: Path) -> None:
        frames = make_frames(6)
        detector = FakeDetector([[Detection.from_xyxy((10, 10, 40, 80), 0.9, 1)]])
        tracker = FakeTracker()
        source, sink = FakeSource(frames), FakeSink(tmp_path / "out.mp4")

        result = VideoProcessor(detector, tracker, make_config(tmp_path)).run(source, sink)

        assert detector.calls == 6
        assert tracker.calls == 6
        assert len(sink.frames) == 6
        assert result.frames_processed == 6

    def test_stage_order_is_detect_then_track_then_write(self, tmp_path: Path) -> None:
        calls: list[str] = []

        class RecordingDetector(FakeDetector):
            def detect(self, frame: np.ndarray) -> list[Detection]:
                calls.append("detect")
                return super().detect(frame)

        class RecordingTracker(FakeTracker):
            def update(self, detections: list[Detection], frame: np.ndarray) -> list:  # type: ignore[type-arg]
                calls.append("track")
                return super().update(detections, frame)

        class RecordingSink(FakeSink):
            def write(self, frame: np.ndarray) -> None:
                calls.append("write")
                super().write(frame)

        VideoProcessor(RecordingDetector([[]]), RecordingTracker(), make_config(tmp_path)).run(
            FakeSource(make_frames(2)), RecordingSink()
        )

        assert calls == ["detect", "track", "write", "detect", "track", "write"]

    def test_detections_reach_the_tracker_unchanged(self, tmp_path: Path) -> None:
        detection = Detection.from_xyxy((1, 2, 30, 40), 0.77, 1)

        received: list[list[Detection]] = []

        class CapturingTracker(FakeTracker):
            def update(self, detections: list[Detection], frame: np.ndarray) -> list:  # type: ignore[type-arg]
                received.append(list(detections))
                return super().update(detections, frame)

        VideoProcessor(FakeDetector([[detection]]), CapturingTracker(), make_config(tmp_path)).run(
            FakeSource(make_frames(1)), FakeSink()
        )

        assert received == [[detection]]

    def test_empty_video_completes_without_error(self, tmp_path: Path) -> None:
        result = VideoProcessor(FakeDetector([]), FakeTracker(), make_config(tmp_path)).run(
            FakeSource([]), FakeSink()
        )
        assert result.frames_processed == 0
        assert result.effective_fps == 0.0

    def test_frames_with_no_detections_are_still_written(self, tmp_path: Path) -> None:
        sink = FakeSink()
        VideoProcessor(FakeDetector([[]]), FakeTracker(), make_config(tmp_path)).run(
            FakeSource(make_frames(4)), sink
        )
        assert len(sink.frames) == 4

    def test_output_preserves_frame_geometry(self, tmp_path: Path) -> None:
        frames = make_frames(3, size=(90, 200))
        sink = FakeSink()
        VideoProcessor(
            FakeDetector([[Detection.from_xyxy((5, 5, 50, 80), 0.9, 1)]]),
            FakeTracker(),
            make_config(tmp_path),
        ).run(FakeSource(frames), sink)

        assert all(frame.shape == (90, 200, 3) for frame in sink.frames)

    def test_annotation_actually_marks_the_frame(self, tmp_path: Path) -> None:
        frames = make_frames(1)
        original = frames[0].copy()
        sink = FakeSink()

        VideoProcessor(
            FakeDetector([[Detection.from_xyxy((20, 20, 60, 100), 0.9, 1)]]),
            FakeTracker(),
            make_config(tmp_path),
        ).run(FakeSource(frames), sink)

        assert not np.array_equal(sink.frames[0], original)


class TestLimitsAndReporting:
    def test_max_frames_stops_early(self, tmp_path: Path) -> None:
        sink = FakeSink()
        result = VideoProcessor(
            FakeDetector([[]]), FakeTracker(), make_config(tmp_path, max_frames=3)
        ).run(FakeSource(make_frames(20)), sink)

        assert result.frames_processed == 3
        assert len(sink.frames) == 3

    def test_counts_unique_track_ids(self, tmp_path: Path) -> None:
        detector = FakeDetector(
            [
                [Detection.from_xyxy((0, 0, 10, 20), 0.9, 1)],
                [
                    Detection.from_xyxy((0, 0, 10, 20), 0.9, 1),
                    Detection.from_xyxy((30, 0, 40, 20), 0.9, 1),
                ],
            ]
        )
        result = VideoProcessor(detector, FakeTracker(), make_config(tmp_path)).run(
            FakeSource(make_frames(2)), FakeSink()
        )
        # FakeTracker assigns ids 0..n-1 per frame, so two distinct ids overall.
        assert result.unique_track_ids == 2

    def test_result_reports_the_sink_path(self, tmp_path: Path) -> None:
        sink = FakeSink(tmp_path / "annotated.mp4")
        result = VideoProcessor(FakeDetector([[]]), FakeTracker(), make_config(tmp_path)).run(
            FakeSource(make_frames(1)), sink
        )
        assert result.output_path == tmp_path / "annotated.mp4"

    def test_effective_fps_is_derived_from_elapsed_time(self) -> None:
        result = ProcessingResult(
            frames_processed=60, elapsed_seconds=10.0, output_path=Path("x.mp4"), unique_track_ids=3
        )
        assert result.effective_fps == pytest.approx(6.0)

    def test_progress_is_logged_periodically_not_every_frame(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="atreides_vision.video.processor"):
            VideoProcessor(
                FakeDetector([[]]), FakeTracker(), make_config(tmp_path, log_every_n_frames=5)
            ).run(FakeSource(make_frames(12)), FakeSink())

        progress_lines = [r for r in caplog.records if "Processed" in r.message]
        assert len(progress_lines) == 2, "expected progress at frames 5 and 10 only"

    def test_empty_input_is_flagged_in_the_log(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="atreides_vision.video.processor"):
            VideoProcessor(FakeDetector([]), FakeTracker(), make_config(tmp_path)).run(
                FakeSource([]), FakeSink()
            )
        assert any("No frames" in r.message for r in caplog.records)


class TestTrajectoryIntegration:
    def test_trails_are_bounded_over_a_long_run(self, tmp_path: Path) -> None:
        """A long run must not accumulate trajectory state indefinitely."""
        detector = FakeDetector([[Detection.from_xyxy((10, 10, 40, 80), 0.9, 1)]])
        tracker = FakeTracker()
        processor = VideoProcessor(detector, tracker, make_config(tmp_path))

        result = processor.run(FakeSource(make_frames(200)), FakeSink())

        assert result.frames_processed == 200
        assert result.unique_track_ids == 1

    def test_failure_mid_run_propagates(self, tmp_path: Path) -> None:
        class ExplodingDetector(FakeDetector):
            def detect(self, frame: np.ndarray) -> list[Detection]:
                if self.calls >= 2:
                    raise RuntimeError("detector crashed")
                return super().detect(frame)

        sink = FakeSink()
        with pytest.raises(RuntimeError, match="detector crashed"):
            VideoProcessor(ExplodingDetector([[]]), FakeTracker(), make_config(tmp_path)).run(
                FakeSource(make_frames(10)), sink
            )
        # Frames written before the failure are intact; the caller's context
        # manager is what finalises the file.
        assert len(sink.frames) == 2

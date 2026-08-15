"""Behaviour analysis through VideoProcessor, driven by fake detector and tracker.

The load-bearing tests here are the regression guarantees: attaching a
behaviour analyzer must not change a single pixel of the output, and omitting
one must leave the V0.1/V0.2 pipeline exactly as it was.
"""

from __future__ import annotations

import subprocess
import sys
from itertools import count
from pathlib import Path

import numpy as np
import pytest

from atreides_vision.behavior.analyzer import BehaviorAnalyzer
from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.behavior.coordinates import MotionSpace, NormalizedFrameProjector
from atreides_vision.behavior.models import BehaviorFrame, SignalType, SubjectRef
from atreides_vision.behavior.reporting import IntegrityFinding
from atreides_vision.config import PipelineConfig, VisionConfig, VisualizationConfig
from atreides_vision.models import BoundingBox, Detection
from atreides_vision.video.processor import VideoProcessor
from atreides_vision.video.source import VideoMetadata
from atreides_vision.zones.analyzer import ZoneAnalyzer
from atreides_vision.zones.models import Zone, ZoneLayout, ZonePolygon
from fakes import FakeDetector, FakeSink, FakeSource, FakeTracker

FRAME_W, FRAME_H = 400, 300
FPS = 25.0
CAMERA = "cam_01"


def frames(count: int) -> list[np.ndarray]:
    return [np.full((FRAME_H, FRAME_W, 3), 60, dtype=np.uint8) for _ in range(count)]


def source(count: int) -> FakeSource:
    return FakeSource(
        frames(count),
        VideoMetadata(width=FRAME_W, height=FRAME_H, fps=FPS, frame_count=count),
    )


def detection_at(x_norm: float) -> Detection:
    cx = x_norm * FRAME_W
    return Detection(
        bbox_xyxy=BoundingBox(cx - 10, 100.0, cx + 10, 150.0), confidence=0.9, class_id=1
    )


def make_config(tmp_path: Path) -> VisionConfig:
    return VisionConfig(
        pipeline=PipelineConfig(tmp_path / "in.mp4", tmp_path / "out.mp4"),
        visualization=VisualizationConfig(trail_length=5),
    )


def make_analyzer(**overrides: object) -> BehaviorAnalyzer:
    return BehaviorAnalyzer(
        camera_id=CAMERA,
        projector=NormalizedFrameProjector(FRAME_W, FRAME_H),
        config=BehaviorAnalysisConfig(**overrides),  # type: ignore[arg-type]
    )


class RecordingBehaviorWriter:
    """A :class:`BehaviorWriter` that keeps what it was handed."""

    def __init__(self) -> None:
        self.frames: list[BehaviorFrame] = []
        self.integrity: list[IntegrityFinding] = []
        self.released = False

    def write_frame(self, frame: BehaviorFrame) -> None:
        self.frames.append(frame)

    def write_integrity(self, assessment: IntegrityFinding) -> None:
        self.integrity.append(assessment)

    def release(self) -> None:
        self.released = True


def walking_script(steps: int, speed_per_frame: float = 0.008) -> list[list[Detection]]:
    return [[detection_at(0.1 + speed_per_frame * step)] for step in range(steps)]


class TestRegression:
    """Behaviour analysis must be invisible to everything that existed before."""

    def test_no_analyzer_means_no_behavior_output(self, tmp_path: Path) -> None:
        result = VideoProcessor(
            FakeDetector([[detection_at(0.25)]]), FakeTracker(), make_config(tmp_path)
        ).run(source(5), FakeSink())

        assert result.final_behavior_frame is None
        assert result.behavior_signals_emitted == 0

    def test_attaching_an_analyzer_does_not_change_a_single_pixel(self, tmp_path: Path) -> None:
        """The strongest guarantee available: identical rendered output."""

        def render(with_behavior: bool) -> list[np.ndarray]:
            sink = FakeSink()
            VideoProcessor(
                FakeDetector(walking_script(20)),
                FakeTracker(),
                make_config(tmp_path),
                clock=count(start=0.0, step=0.04).__next__,
                behavior_analyzer=make_analyzer() if with_behavior else None,
            ).run(source(20), sink)
            return sink.frames

        with_signals, without_signals = render(True), render(False)
        assert len(with_signals) == len(without_signals) == 20
        assert all(np.array_equal(a, b) for a, b in zip(with_signals, without_signals, strict=True))

    def test_frame_count_and_track_count_are_unaffected(self, tmp_path: Path) -> None:
        plain = VideoProcessor(
            FakeDetector(walking_script(10)), FakeTracker(), make_config(tmp_path)
        ).run(source(10), FakeSink())
        instrumented = VideoProcessor(
            FakeDetector(walking_script(10)),
            FakeTracker(),
            make_config(tmp_path),
            behavior_analyzer=make_analyzer(),
        ).run(source(10), FakeSink())

        assert plain.frames_processed == instrumented.frames_processed
        assert plain.unique_track_ids == instrumented.unique_track_ids

    def test_zone_analysis_is_unaffected(self, tmp_path: Path) -> None:
        layout = ZoneLayout(
            camera_id=CAMERA,
            zones=(
                Zone(
                    id="left",
                    name="Left",
                    polygon=ZonePolygon.from_pairs(
                        [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
                    ),
                ),
            ),
        )
        result = VideoProcessor(
            FakeDetector([[detection_at(0.25)]]),
            FakeTracker(),
            make_config(tmp_path),
            zone_analyzer=ZoneAnalyzer(layout),
            behavior_analyzer=make_analyzer(),
        ).run(source(30), FakeSink())

        report = result.final_zone_report
        assert report is not None
        assert report.for_zone("left") is not None
        assert result.behavior_signals_emitted > 0


class TestIntegration:
    def test_signals_are_produced_and_returned(self, tmp_path: Path) -> None:
        result = VideoProcessor(
            FakeDetector(walking_script(40)),
            FakeTracker(),
            make_config(tmp_path),
            behavior_analyzer=make_analyzer(),
        ).run(source(40), FakeSink())

        frame = result.final_behavior_frame
        assert frame is not None
        assert frame.frame_index == 40
        assert result.behavior_signals_emitted > 0
        assert len(frame.of_type(SignalType.SPEED)) == 1

    def test_speed_uses_source_timestamps_not_wall_clock(self, tmp_path: Path) -> None:
        """0.008 frame-widths per frame at 25 FPS is 0.2 frame-widths per second."""
        result = VideoProcessor(
            FakeDetector(walking_script(60)),
            FakeTracker(),
            make_config(tmp_path),
            behavior_analyzer=make_analyzer(),
        ).run(source(60), FakeSink())

        frame = result.final_behavior_frame
        assert frame is not None
        speed = frame.value_for(SubjectRef(CAMERA, 0), SignalType.SPEED)
        assert speed == pytest.approx(0.2, rel=1e-6)

    def test_a_different_source_fps_reports_the_same_real_speed(self, tmp_path: Path) -> None:
        """Halve the frame rate and double the step: same person, same speed."""
        slow_source = FakeSource(
            frames(30),
            VideoMetadata(width=FRAME_W, height=FRAME_H, fps=12.5, frame_count=30),
        )
        result = VideoProcessor(
            FakeDetector(walking_script(30, speed_per_frame=0.016)),
            FakeTracker(),
            make_config(tmp_path),
            behavior_analyzer=make_analyzer(),
        ).run(slow_source, FakeSink())

        frame = result.final_behavior_frame
        assert frame is not None
        assert frame.value_for(SubjectRef(CAMERA, 0), SignalType.SPEED) == pytest.approx(
            0.2, rel=1e-6
        )

    def test_a_standing_person_accumulates_stationary_duration(self, tmp_path: Path) -> None:
        result = VideoProcessor(
            FakeDetector([[detection_at(0.5)]]),
            FakeTracker(),
            make_config(tmp_path),
            behavior_analyzer=make_analyzer(),
        ).run(source(75), FakeSink())  # 3 s at 25 FPS

        frame = result.final_behavior_frame
        assert frame is not None
        duration = frame.value_for(SubjectRef(CAMERA, 0), SignalType.STATIONARY_DURATION)
        assert duration == pytest.approx(2.96, abs=0.05)

    def test_signals_are_camera_stamped(self, tmp_path: Path) -> None:
        analyzer = BehaviorAnalyzer(
            camera_id="cam_backdoor",
            projector=NormalizedFrameProjector(FRAME_W, FRAME_H),
        )
        result = VideoProcessor(
            FakeDetector(walking_script(20)),
            FakeTracker(),
            make_config(tmp_path),
            behavior_analyzer=analyzer,
        ).run(source(20), FakeSink())

        frame = result.final_behavior_frame
        assert frame is not None
        assert all(signal.subject.camera_id == "cam_backdoor" for signal in frame)

    def test_no_detections_produces_no_signals(self, tmp_path: Path) -> None:
        result = VideoProcessor(
            FakeDetector([[]]),
            FakeTracker(),
            make_config(tmp_path),
            behavior_analyzer=make_analyzer(),
        ).run(source(10), FakeSink())

        assert result.behavior_signals_emitted == 0
        assert result.final_behavior_frame is not None
        assert len(result.final_behavior_frame) == 0

    def test_run_is_deterministic(self, tmp_path: Path) -> None:
        def records() -> list[dict[str, object]]:
            result = VideoProcessor(
                FakeDetector(walking_script(30)),
                FakeTracker(),
                make_config(tmp_path),
                clock=count(start=0.0, step=0.04).__next__,
                behavior_analyzer=make_analyzer(),
            ).run(source(30), FakeSink())
            assert result.final_behavior_frame is not None
            return result.final_behavior_frame.to_records()

        assert records() == records()


class TestBehaviorWriterIntegration:
    """The processor must actually drive the writer, and only when given one."""

    def test_writer_receives_one_call_per_frame(self, tmp_path: Path) -> None:
        writer = RecordingBehaviorWriter()
        VideoProcessor(
            FakeDetector(walking_script(12)),
            FakeTracker(),
            make_config(tmp_path),
            behavior_analyzer=make_analyzer(),
            behavior_writer=writer,
        ).run(source(12), FakeSink())

        assert len(writer.frames) == 12
        assert [frame.frame_index for frame in writer.frames] == list(range(1, 13))

    def test_written_frames_carry_positions(self, tmp_path: Path) -> None:
        writer = RecordingBehaviorWriter()
        VideoProcessor(
            FakeDetector(walking_script(10)),
            FakeTracker(),
            make_config(tmp_path),
            behavior_analyzer=make_analyzer(),
            behavior_writer=writer,
        ).run(source(10), FakeSink())

        snapshots = writer.frames[-1].snapshots
        assert len(snapshots) == 1
        assert snapshots[0].subject == SubjectRef(CAMERA, 0)
        assert snapshots[0].space is MotionSpace.NORMALIZED_IMAGE

    def test_no_writer_means_the_analyzer_still_runs(self, tmp_path: Path) -> None:
        analyzer = make_analyzer()
        result = VideoProcessor(
            FakeDetector(walking_script(10)),
            FakeTracker(),
            make_config(tmp_path),
            behavior_analyzer=analyzer,
        ).run(source(10), FakeSink())
        assert result.behavior_signals_emitted > 0

    def test_no_analyzer_means_the_writer_is_never_called(self, tmp_path: Path) -> None:
        """A writer without an analyzer has nothing to write, and must stay silent."""
        writer = RecordingBehaviorWriter()
        VideoProcessor(
            FakeDetector(walking_script(10)),
            FakeTracker(),
            make_config(tmp_path),
            behavior_writer=writer,
        ).run(source(10), FakeSink())
        assert writer.frames == []

    def test_writer_sees_the_same_frames_the_result_reports(self, tmp_path: Path) -> None:
        writer = RecordingBehaviorWriter()
        result = VideoProcessor(
            FakeDetector(walking_script(20)),
            FakeTracker(),
            make_config(tmp_path),
            behavior_analyzer=make_analyzer(),
            behavior_writer=writer,
        ).run(source(20), FakeSink())

        assert writer.frames[-1] is result.final_behavior_frame
        assert sum(len(frame) for frame in writer.frames) == result.behavior_signals_emitted


class TestImportOrder:
    """Regression: the visualization package must import standalone.

    ``visualization.overlay`` used to import the ``video`` package at runtime,
    whose ``__init__`` imports the processor, which imports ``overlay`` — a
    cycle that only failed when ``visualization`` was imported first. The
    annotation is now type-only.
    """

    def test_visualization_imports_before_video(self) -> None:
        script = (
            "import atreides_vision.visualization.calibration_overlay as m; "
            "assert m.render_calibration_overlay is not None"
        )
        assert subprocess.run([sys.executable, "-c", script], check=False).returncode == 0

    def test_video_imports_before_visualization(self) -> None:
        script = "import atreides_vision.video; import atreides_vision.visualization"
        assert subprocess.run([sys.executable, "-c", script], check=False).returncode == 0

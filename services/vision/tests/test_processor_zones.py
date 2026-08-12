"""Zone integration through VideoProcessor, driven by fake detector and tracker.

Includes the V0.1 regression guarantee: with no zone analyzer attached, the
pipeline must behave exactly as it did before V0.2.
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

import numpy as np
import pytest

from atreides_vision.config import (
    PipelineConfig,
    VisionConfig,
    VisualizationConfig,
    ZoneAnalysisConfig,
)
from atreides_vision.models import BoundingBox, Detection
from atreides_vision.video.processor import VideoProcessor
from atreides_vision.video.source import VideoMetadata
from atreides_vision.visualization.overlay import StatusOverlay
from atreides_vision.zones.analyzer import ZoneAnalyzer
from atreides_vision.zones.models import Zone, ZoneLayout, ZonePolygon
from atreides_vision.zones.reporting import JsonlMetricsWriter
from fakes import FakeDetector, FakeSink, FakeSource, FakeTracker

FRAME_W, FRAME_H = 400, 300
FPS = 25.0


def zone(zone_id: str, x0: float, y0: float, x1: float, y1: float) -> Zone:
    return Zone(
        id=zone_id,
        name=zone_id.title(),
        polygon=ZonePolygon.from_pairs([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]),
    )


LAYOUT = ZoneLayout(
    camera_id="cam_01",
    zones=(zone("left", 0.0, 0.0, 0.5, 1.0), zone("right", 0.5, 0.0, 1.0, 1.0)),
)


def frames(count: int) -> list[np.ndarray]:
    return [np.full((FRAME_H, FRAME_W, 3), 60, dtype=np.uint8) for _ in range(count)]


def source(count: int) -> FakeSource:
    return FakeSource(
        frames(count),
        VideoMetadata(width=FRAME_W, height=FRAME_H, fps=FPS, frame_count=count),
    )


def detection_at(x_norm: float) -> Detection:
    """A person whose feet land at ``x_norm`` horizontally, mid-frame vertically."""
    cx = x_norm * FRAME_W
    return Detection(
        bbox_xyxy=BoundingBox(cx - 10, 100.0, cx + 10, 150.0), confidence=0.9, class_id=1
    )


def make_config(tmp_path: Path, **zone_overrides: object) -> VisionConfig:
    return VisionConfig(
        pipeline=PipelineConfig(tmp_path / "in.mp4", tmp_path / "out.mp4"),
        visualization=VisualizationConfig(trail_length=5),
        zones=ZoneAnalysisConfig(**zone_overrides),  # type: ignore[arg-type]
    )


class TestV01Regression:
    """Without --zones, nothing about V0.1 behaviour may change."""

    def test_no_analyzer_means_no_zone_report(self, tmp_path: Path) -> None:
        result = VideoProcessor(
            FakeDetector([[detection_at(0.25)]]), FakeTracker(), make_config(tmp_path)
        ).run(source(5), FakeSink())

        assert result.frames_processed == 5
        assert result.final_zone_report is None
        assert result.metrics_records_written == 0

    def test_status_panel_keeps_the_v01_title(self) -> None:
        assert StatusOverlay(10, 6.4, 3).lines() == (
            "ATREIDES // SEE + TRACK",
            "Frame: 10",
            "Processing: 6.4 FPS",
            "Tracks: 3",
        )

    def test_every_frame_is_still_written(self, tmp_path: Path) -> None:
        sink = FakeSink()
        VideoProcessor(FakeDetector([[]]), FakeTracker(), make_config(tmp_path)).run(
            source(7), sink
        )
        assert len(sink.frames) == 7
        assert all(frame.shape == (FRAME_H, FRAME_W, 3) for frame in sink.frames)

    def test_rendering_is_a_pure_function_of_pipeline_state(self, tmp_path: Path) -> None:
        """With the clock pinned, two zone-free runs render byte-identical frames."""
        outputs = []
        for _ in range(2):
            sink = FakeSink()
            VideoProcessor(
                FakeDetector([[detection_at(0.3)]]),
                FakeTracker(),
                make_config(tmp_path),
                clock=count(start=0.0, step=0.04).__next__,
            ).run(source(4), sink)
            outputs.append(sink.frames)
        assert all(np.array_equal(a, b) for a, b in zip(*outputs, strict=True))


class TestZoneIntegration:
    def test_zone_report_is_produced_and_returned(self, tmp_path: Path) -> None:
        result = VideoProcessor(
            FakeDetector([[detection_at(0.25)]]),
            FakeTracker(),
            make_config(tmp_path),
            zone_analyzer=ZoneAnalyzer(LAYOUT),
        ).run(source(5), FakeSink())

        report = result.final_zone_report
        assert report is not None
        assert report.frame_index == 5
        assert report.for_zone("left").occupancy == 1  # type: ignore[union-attr]
        assert report.for_zone("right").occupancy == 0  # type: ignore[union-attr]

    def test_timestamps_come_from_the_source_frame_rate(self, tmp_path: Path) -> None:
        """Frame 51 of 25 FPS footage is at t=2.0s, not frame 51 of an assumed 30."""
        result = VideoProcessor(
            FakeDetector([[detection_at(0.25)]]),
            FakeTracker(),
            make_config(tmp_path),
            zone_analyzer=ZoneAnalyzer(LAYOUT),
        ).run(source(51), FakeSink())

        report = result.final_zone_report
        assert report is not None
        assert report.timestamp_seconds == pytest.approx(2.0)
        assert report.for_zone("left").max_dwell_seconds == pytest.approx(2.0)  # type: ignore[union-attr]

    def test_dwell_scales_with_a_different_source_fps(self, tmp_path: Path) -> None:
        slow = FakeSource(
            frames(51), VideoMetadata(width=FRAME_W, height=FRAME_H, fps=10.0, frame_count=51)
        )
        result = VideoProcessor(
            FakeDetector([[detection_at(0.25)]]),
            FakeTracker(),
            make_config(tmp_path),
            zone_analyzer=ZoneAnalyzer(LAYOUT),
        ).run(slow, FakeSink())

        report = result.final_zone_report
        assert report is not None
        assert report.for_zone("left").max_dwell_seconds == pytest.approx(5.0)  # type: ignore[union-attr]

    def test_person_walking_between_zones_moves_the_counters(self, tmp_path: Path) -> None:
        script = [[detection_at(0.05 + 0.03 * step)] for step in range(30)]
        result = VideoProcessor(
            FakeDetector(script),
            FakeTracker(),
            make_config(tmp_path, exit_grace_frames=2),
            zone_analyzer=ZoneAnalyzer(LAYOUT),
        ).run(source(30), FakeSink())

        report = result.final_zone_report
        assert report is not None
        left, right = report.for_zone("left"), report.for_zone("right")
        assert left is not None and right is not None
        assert left.entries == 1 and left.exits == 1
        assert right.entries == 1 and right.occupancy == 1

    def test_zones_are_drawn_onto_the_output(self, tmp_path: Path) -> None:
        with_zones = FakeSink()
        VideoProcessor(
            FakeDetector([[]]),
            FakeTracker(),
            make_config(tmp_path),
            zone_analyzer=ZoneAnalyzer(LAYOUT),
        ).run(source(2), with_zones)

        without_zones = FakeSink()
        VideoProcessor(FakeDetector([[]]), FakeTracker(), make_config(tmp_path)).run(
            source(2), without_zones
        )

        assert not np.array_equal(with_zones.frames[0], without_zones.frames[0])

    def test_frame_geometry_is_unchanged_by_zone_drawing(self, tmp_path: Path) -> None:
        sink = FakeSink()
        VideoProcessor(
            FakeDetector([[detection_at(0.5)]]),
            FakeTracker(),
            make_config(tmp_path),
            zone_analyzer=ZoneAnalyzer(LAYOUT),
        ).run(source(3), sink)
        assert all(frame.shape == (FRAME_H, FRAME_W, 3) for frame in sink.frames)

    def test_empty_zone_layout_runs_cleanly(self, tmp_path: Path) -> None:
        result = VideoProcessor(
            FakeDetector([[detection_at(0.5)]]),
            FakeTracker(),
            make_config(tmp_path),
            zone_analyzer=ZoneAnalyzer(ZoneLayout(camera_id="cam_01", zones=())),
        ).run(source(3), FakeSink())
        assert result.final_zone_report is not None
        assert result.final_zone_report.snapshots == ()


class TestMetricsStreamIntegration:
    def test_samples_once_per_second_of_source_time(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with JsonlMetricsWriter(path) as writer:
            result = VideoProcessor(
                FakeDetector([[detection_at(0.25)]]),
                FakeTracker(),
                make_config(tmp_path, metrics_sample_interval_seconds=1.0),
                zone_analyzer=ZoneAnalyzer(LAYOUT),
                metrics_writer=writer,
            ).run(source(100), FakeSink())  # 4 seconds at 25 FPS

        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 4
        assert result.metrics_records_written == 4
        assert [r["timestamp_seconds"] for r in records] == pytest.approx([0.0, 1.0, 2.0, 3.0])

    def test_interval_is_configurable(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        with JsonlMetricsWriter(path) as writer:
            VideoProcessor(
                FakeDetector([[detection_at(0.25)]]),
                FakeTracker(),
                make_config(tmp_path, metrics_sample_interval_seconds=0.5),
                zone_analyzer=ZoneAnalyzer(LAYOUT),
                metrics_writer=writer,
            ).run(source(50), FakeSink())  # 2 seconds

        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 4

    def test_metrics_change_as_the_scene_changes(self, tmp_path: Path) -> None:
        """The stream must actually track what is happening, not repeat itself."""
        script = [[detection_at(0.05 + 0.012 * step)] for step in range(80)]
        path = tmp_path / "metrics.jsonl"
        with JsonlMetricsWriter(path) as writer:
            VideoProcessor(
                FakeDetector(script),
                FakeTracker(),
                make_config(tmp_path, metrics_sample_interval_seconds=0.5),
                zone_analyzer=ZoneAnalyzer(LAYOUT),
                metrics_writer=writer,
            ).run(source(80), FakeSink())

        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        left = [r["zones"][0] for r in records]
        assert {row["occupancy"] for row in left} == {0, 1}, "occupancy should change"
        assert len({row["motion_level"] for row in left}) > 1, "motion should vary"

    def test_no_writer_means_no_records(self, tmp_path: Path) -> None:
        result = VideoProcessor(
            FakeDetector([[detection_at(0.25)]]),
            FakeTracker(),
            make_config(tmp_path),
            zone_analyzer=ZoneAnalyzer(LAYOUT),
        ).run(source(50), FakeSink())
        assert result.metrics_records_written == 0

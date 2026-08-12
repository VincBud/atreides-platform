"""The zone analyzer end to end, and the JSONL metrics stream."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atreides_vision.errors import MetricsOutputError
from atreides_vision.models import BoundingBox, Track
from atreides_vision.zones.analyzer import ZoneAnalyzer
from atreides_vision.zones.config import ZoneAnalysisConfig
from atreides_vision.zones.models import Zone, ZoneLayout, ZonePolygon
from atreides_vision.zones.reporting import JsonlMetricsWriter, MetricsSampler

FRAME_W, FRAME_H = 1000, 1000
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


def track_at(track_id: int, x: float, y: float = 0.5) -> Track:
    cx, cy = x * FRAME_W, y * FRAME_H
    return Track(track_id=track_id, bbox_xyxy=BoundingBox(cx - 20, cy - 100, cx + 20, cy))


def analyzer(**overrides: object) -> ZoneAnalyzer:
    return ZoneAnalyzer(LAYOUT, ZoneAnalysisConfig(**overrides))  # type: ignore[arg-type]


class TestAnalyzerBasics:
    def test_reports_every_zone_from_the_first_frame(self) -> None:
        report = analyzer().update([], 1, 0.0, FRAME_W, FRAME_H)
        assert [s.zone_id for s in report.snapshots] == ["left", "right"]
        assert all(s.occupancy == 0 for s in report.snapshots)

    def test_latest_report_is_none_before_the_first_frame(self) -> None:
        assert analyzer().latest_report is None

    def test_tracks_land_in_the_right_zone(self) -> None:
        report = analyzer().update(
            [track_at(1, 0.2), track_at(2, 0.8), track_at(3, 0.9)], 1, 0.0, FRAME_W, FRAME_H
        )
        assert report.for_zone("left").occupancy == 1  # type: ignore[union-attr]
        assert report.for_zone("right").occupancy == 2  # type: ignore[union-attr]


class TestWalkingAcrossAZoneBoundary:
    """A person crossing from one zone to the next must produce one exit and one entry."""

    def test_crossing_is_recorded_once(self) -> None:
        zones = analyzer(exit_grace_frames=2)
        report = None
        for frame in range(1, 41):
            # Walk steadily from x=0.1 to x=0.9 across the midline.
            x = 0.1 + 0.02 * (frame - 1)
            report = zones.update([track_at(1, x)], frame, (frame - 1) / FPS, FRAME_W, FRAME_H)

        assert report is not None
        left = report.for_zone("left")
        right = report.for_zone("right")
        assert left is not None and right is not None
        assert left.entries == 1 and left.exits == 1 and left.occupancy == 0
        assert right.entries == 1 and right.occupancy == 1

    def test_dwell_reflects_time_spent_in_each_zone(self) -> None:
        zones = analyzer(exit_grace_frames=0)
        report = None
        for frame in range(1, 41):
            x = 0.1 + 0.02 * (frame - 1)
            report = zones.update([track_at(1, x)], frame, (frame - 1) / FPS, FRAME_W, FRAME_H)

        assert report is not None
        right = report.for_zone("right")
        assert right is not None
        # Crossed the midline around frame 21, so ~0.76s of the 1.56s run.
        assert right.max_dwell_seconds == pytest.approx(0.76, abs=0.1)

    def test_movement_produces_a_non_zero_motion_level(self) -> None:
        zones = analyzer()
        report = None
        for frame in range(1, 41):
            x = 0.1 + 0.01 * (frame - 1)
            report = zones.update([track_at(1, x)], frame, (frame - 1) / FPS, FRAME_W, FRAME_H)
        assert report is not None
        left = report.for_zone("left")
        assert left is not None
        assert left.mean_normalized_speed > 0.0
        assert 0.0 < left.motion_level <= 1.0

    def test_a_still_crowd_reports_near_zero_motion(self) -> None:
        zones = analyzer()
        report = None
        for frame in range(1, 61):
            report = zones.update([track_at(1, 0.25)], frame, (frame - 1) / FPS, FRAME_W, FRAME_H)
        assert report is not None
        left = report.for_zone("left")
        assert left is not None
        assert left.mean_normalized_speed == pytest.approx(0.0, abs=1e-9)
        assert left.motion_level == pytest.approx(0.0, abs=1e-6)


class TestFrameRateIndependence:
    @pytest.mark.parametrize("fps", [10.0, 25.0, 50.0])
    def test_same_motion_yields_same_dwell_and_speed(self, fps: float) -> None:
        """Two seconds of the same walk must measure the same at any frame rate."""
        zones = analyzer(exit_grace_frames=0)
        frames = int(fps * 2)
        report = None
        for step in range(frames):
            t = step / fps
            x = 0.05 + 0.1 * t  # 0.1 frame-widths per second
            report = zones.update([track_at(1, x)], step + 1, t, FRAME_W, FRAME_H)

        assert report is not None
        left = report.for_zone("left")
        assert left is not None
        assert left.max_dwell_seconds == pytest.approx(2.0, abs=2.0 / fps)
        assert left.mean_normalized_speed == pytest.approx(0.1, rel=0.1)


class TestOverlappingZonesThroughTheAnalyzer:
    def test_one_person_counts_in_both_zones(self) -> None:
        overlapping = ZoneLayout(
            camera_id="cam_01",
            zones=(zone("wide", 0.0, 0.0, 1.0, 1.0), zone("vip", 0.4, 0.4, 0.6, 0.6)),
        )
        zones = ZoneAnalyzer(overlapping)
        report = zones.update([track_at(1, 0.5, 0.5)], 1, 0.0, FRAME_W, FRAME_H)
        assert report.for_zone("wide").occupancy == 1  # type: ignore[union-attr]
        assert report.for_zone("vip").occupancy == 1  # type: ignore[union-attr]


class TestAnalyzerMemory:
    def test_state_stays_bounded_with_churning_track_ids(self) -> None:
        zones = analyzer(exit_grace_frames=2, stale_track_ttl_frames=10, max_tracked_movements=64)
        for frame in range(1, 1001):
            zones.update([track_at(frame, 0.25)], frame, (frame - 1) / FPS, FRAME_W, FRAME_H)

        report = zones.latest_report
        assert report is not None
        left = report.for_zone("left")
        assert left is not None
        assert left.occupancy <= 3, "expired members must be released"
        assert left.entries == 1000, "every distinct track still counted once"


class TestEmptyLayout:
    def test_analyzer_with_no_zones_is_harmless(self) -> None:
        zones = ZoneAnalyzer(ZoneLayout(camera_id="cam_01", zones=()))
        report = zones.update([track_at(1, 0.5)], 1, 0.0, FRAME_W, FRAME_H)
        assert report.snapshots == ()

    def test_log_summary_without_frames_does_not_raise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        analyzer().log_summary()


class TestMetricsSampler:
    def test_first_frame_is_always_sampled(self) -> None:
        assert MetricsSampler(1.0).should_sample(0.0)

    def test_samples_once_per_interval(self) -> None:
        sampler = MetricsSampler(1.0)
        sampled = [t / 25.0 for t in range(0, 126) if sampler.should_sample(t / 25.0)]
        # 5 seconds of 25 FPS footage -> one record per second.
        assert len(sampled) == 6
        assert sampled == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    def test_interval_counted_in_source_time_not_frames(self) -> None:
        """A 50 FPS video yields the same record count as a 25 FPS one."""
        counts = []
        for fps in (25.0, 50.0):
            sampler = MetricsSampler(1.0)
            counts.append(sum(1 for f in range(int(fps * 4)) if sampler.should_sample(f / fps)))
        assert counts[0] == counts[1]

    def test_rejects_non_positive_interval(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            MetricsSampler(0.0)

    def test_reset_restarts_sampling(self) -> None:
        sampler = MetricsSampler(10.0)
        assert sampler.should_sample(0.0)
        assert not sampler.should_sample(1.0)
        sampler.reset()
        assert sampler.should_sample(1.0)


class TestJsonlMetricsWriter:
    def test_writes_one_json_object_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        zones = analyzer()
        with JsonlMetricsWriter(path) as writer:
            for frame in range(1, 4):
                writer.write(
                    zones.update([track_at(1, 0.25)], frame, frame / FPS, FRAME_W, FRAME_H)
                )
            assert writer.records_written == 3

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        records = [json.loads(line) for line in lines]
        assert records[0]["frame"] == 1
        assert {z["id"] for z in records[0]["zones"]} == {"left", "right"}

    def test_record_matches_the_documented_shape(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        zones = analyzer()
        with JsonlMetricsWriter(path) as writer:
            writer.write(zones.update([track_at(1, 0.25)], 120, 4.0, FRAME_W, FRAME_H))

        record = json.loads(path.read_text(encoding="utf-8").strip())
        assert record["frame"] == 120
        assert record["timestamp_seconds"] == 4.0
        zone_record = record["zones"][0]
        for key in (
            "id",
            "occupancy",
            "entries",
            "exits",
            "mean_normalized_speed",
            "motion_level",
            "average_dwell_seconds",
            "max_dwell_seconds",
        ):
            assert key in zone_record, key

    def test_creates_missing_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deep" / "metrics.jsonl"
        with JsonlMetricsWriter(path) as writer:
            writer.write(analyzer().update([], 1, 0.0, FRAME_W, FRAME_H))
        assert path.exists()

    def test_released_on_exception(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        writer = JsonlMetricsWriter(path)
        with pytest.raises(RuntimeError), writer:
            writer.write(analyzer().update([], 1, 0.0, FRAME_W, FRAME_H))
            raise RuntimeError("pipeline failed")

        assert path.read_text(encoding="utf-8").strip()
        with pytest.raises(MetricsOutputError, match="released"):
            writer.write(analyzer().update([], 2, 0.1, FRAME_W, FRAME_H))

    def test_unwritable_path_raises_a_domain_error(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        with pytest.raises(MetricsOutputError, match="Could not open metrics output"):
            JsonlMetricsWriter(blocker / "metrics.jsonl")

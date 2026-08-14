"""Tracking diagnostics: optional, observational, and valid JSONL.

The central guarantee under test is that enabling diagnostics changes nothing
about what the pipeline produces — it only copies state out to a file.
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

import numpy as np
import pytest

from atreides_vision.config import PipelineConfig, TrackerConfig, VisionConfig
from atreides_vision.errors import MetricsOutputError
from atreides_vision.models import BoundingBox, Detection, Track
from atreides_vision.tracking.diagnostics import (
    InternalTrackSnapshot,
    JsonlTrackingDebugWriter,
    TrackingDebugRecord,
    snapshot_internal_tracks,
)
from atreides_vision.tracking.ocsort import OcSortTracker
from atreides_vision.video.processor import VideoProcessor
from atreides_vision.video.source import VideoMetadata
from fakes import FakeDetector, FakeSink, FakeSource, FakeTracker

FRAME_W, FRAME_H = 400, 300


def frames(n: int) -> list[np.ndarray]:
    return [np.full((FRAME_H, FRAME_W, 3), 60, dtype=np.uint8) for _ in range(n)]


def source(n: int) -> FakeSource:
    return FakeSource(
        frames(n), VideoMetadata(width=FRAME_W, height=FRAME_H, fps=30.0, frame_count=n)
    )


def detection_at(x: float) -> Detection:
    return Detection(bbox_xyxy=BoundingBox(x, 100.0, x + 40, 200.0), confidence=0.9, class_id=1)


def make_config(tmp_path: Path, deduplicate_detections: bool = False) -> VisionConfig:
    return VisionConfig(
        pipeline=PipelineConfig(
            tmp_path / "in.mp4",
            tmp_path / "out.mp4",
            deduplicate_detections=deduplicate_detections,
        )
    )


def fixed_clock() -> object:
    return count(start=0.0, step=0.033).__next__


class TestSnapshotIsDefensive:
    """Diagnostics must never be able to break a run."""

    def test_tracker_without_a_backend_yields_nothing(self) -> None:
        assert snapshot_internal_tracks(FakeTracker()) == ()

    def test_backend_without_tracks_yields_nothing(self) -> None:
        class Backend:
            pass

        class Tracker:
            backend = Backend()

        assert snapshot_internal_tracks(Tracker()) == ()

    def test_unintrospectable_tracklet_is_skipped_not_fatal(self) -> None:
        class Exploding:
            @property
            def time_since_update(self) -> int:
                raise RuntimeError("boom")

        class Backend:
            def __init__(self) -> None:
                self.tracks = [Exploding()]

        class Tracker:
            def __init__(self) -> None:
                self.backend = Backend()

        assert snapshot_internal_tracks(Tracker()) == ()

    def test_plain_object_is_handled(self) -> None:
        assert snapshot_internal_tracks(object()) == ()


class TestSnapshotAgainstRealOcSort:
    """Reads real OC-SORT internals — no weights needed, no behaviour change."""

    def test_exposes_the_fields_needed_for_diagnosis(self) -> None:
        tracker = OcSortTracker(TrackerConfig(frame_rate=30.0, minimum_consecutive_frames=1))
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for step in range(4):
            tracker.update([detection_at(10.0 + step)], frame)

        snapshots = snapshot_internal_tracks(tracker)
        assert snapshots, "a real OC-SORT run should expose internal tracklets"

        snapshot = snapshots[0]
        assert snapshot.age >= 1
        assert snapshot.time_since_update == 0
        assert snapshot.state == "tracked"
        assert snapshot.last_observation_bbox is not None
        assert len(snapshot.last_observation_bbox) == 4
        assert snapshot.predicted_bbox is not None
        assert len(snapshot.predicted_bbox) == 4
        assert snapshot.consecutive_updates >= 1

    def test_lost_track_is_visible_with_a_growing_gap(self) -> None:
        """The exact signal needed for the occlusion case."""
        tracker = OcSortTracker(TrackerConfig(frame_rate=30.0, minimum_consecutive_frames=1))
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for step in range(4):
            tracker.update([detection_at(10.0 + step)], frame)

        for _ in range(3):
            tracker.update([], frame)  # the person is occluded

        snapshots = snapshot_internal_tracks(tracker)
        assert snapshots, "the track should still exist internally during a short gap"
        assert snapshots[0].state == "lost"
        assert snapshots[0].time_since_update >= 1

    def test_reading_a_snapshot_does_not_disturb_tracking(self) -> None:
        """Same inputs, with and without introspection, must track identically."""

        def run(introspect: bool) -> list[list[int]]:
            tracker = OcSortTracker(TrackerConfig(frame_rate=30.0, minimum_consecutive_frames=1))
            frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            observed: list[list[int]] = []
            for step in range(12):
                tracks = tracker.update([detection_at(10.0 + 2 * step)], frame)
                if introspect:
                    snapshot_internal_tracks(tracker)
                observed.append([t.track_id for t in tracks])
            return observed

        assert run(introspect=True) == run(introspect=False)


class TestRecordSerialisation:
    def test_record_shape_matches_the_documented_schema(self) -> None:
        record = TrackingDebugRecord(
            frame_index=132,
            detections=(detection_at(10.0),),
            reported_tracks=(Track(3, BoundingBox(10, 100, 50, 200), 0.9),),
            internal_tracks=(
                InternalTrackSnapshot(
                    track_id=3,
                    state="lost",
                    age=40,
                    time_since_update=5,
                    last_observation_bbox=[10.0, 100.0, 50.0, 200.0],
                    predicted_bbox=[12.0, 100.0, 52.0, 200.0],
                    consecutive_updates=0,
                    observed=False,
                    velocity=[0.0, 1.0],
                ),
            ),
        ).to_record()

        assert record["frame"] == 132
        assert record["detections"][0]["bbox"] == [10.0, 100.0, 50.0, 200.0]
        assert record["detections"][0]["confidence"] == 0.9
        assert record["reported_tracks"][0]["track_id"] == 3
        internal = record["internal_tracks"][0]
        for key in (
            "track_id",
            "state",
            "age",
            "time_since_update",
            "last_observation_bbox",
            "predicted_bbox",
        ):
            assert key in internal, key

    def test_record_is_json_serialisable(self) -> None:
        record = TrackingDebugRecord(1, (detection_at(5.0),), (), ()).to_record()
        assert json.loads(json.dumps(record))["frame"] == 1

    def test_track_without_confidence_serialises_as_null(self) -> None:
        record = TrackingDebugRecord(
            1, (), (Track(1, BoundingBox(0, 0, 10, 10), None),), ()
        ).to_record()
        assert record["reported_tracks"][0]["confidence"] is None


class TestJsonlWriter:
    def test_writes_one_valid_object_per_frame(self, tmp_path: Path) -> None:
        path = tmp_path / "debug.jsonl"
        with JsonlTrackingDebugWriter(path) as writer:
            for frame in range(1, 4):
                writer.write(TrackingDebugRecord(frame, (detection_at(1.0),), (), ()))
            assert writer.records_written == 3

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        assert [json.loads(line)["frame"] for line in lines] == [1, 2, 3]

    def test_creates_missing_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "debug.jsonl"
        with JsonlTrackingDebugWriter(path) as writer:
            writer.write(TrackingDebugRecord(1, (), (), ()))
        assert path.exists()

    def test_released_on_exception(self, tmp_path: Path) -> None:
        path = tmp_path / "debug.jsonl"
        writer = JsonlTrackingDebugWriter(path)
        with pytest.raises(RuntimeError), writer:
            writer.write(TrackingDebugRecord(1, (), (), ()))
            raise RuntimeError("pipeline failed")

        assert path.read_text(encoding="utf-8").strip()
        with pytest.raises(MetricsOutputError, match="released"):
            writer.write(TrackingDebugRecord(2, (), (), ()))

    def test_unwritable_path_raises_a_domain_error(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        with pytest.raises(MetricsOutputError, match="Could not open tracking debug output"):
            JsonlTrackingDebugWriter(blocker / "debug.jsonl")


class TestPipelineIntegration:
    def test_debug_output_is_optional(self, tmp_path: Path) -> None:
        result = VideoProcessor(
            FakeDetector([[detection_at(10.0)]]), FakeTracker(), make_config(tmp_path)
        ).run(source(5), FakeSink())
        assert result.frames_processed == 5

    def test_disabled_debug_leaves_output_byte_identical(self, tmp_path: Path) -> None:
        """The regression guarantee: diagnostics off means nothing changed."""

        def run(debug: bool) -> list[np.ndarray]:
            sink = FakeSink()
            writer = JsonlTrackingDebugWriter(tmp_path / "d.jsonl") if debug else None
            VideoProcessor(
                FakeDetector([[detection_at(10.0)]]),
                FakeTracker(),
                make_config(tmp_path),
                clock=fixed_clock(),  # type: ignore[arg-type]
                tracking_debug_writer=writer,
            ).run(source(4), sink)
            if writer is not None:
                writer.release()
            return sink.frames

        assert all(np.array_equal(a, b) for a, b in zip(run(False), run(True), strict=True))

    def test_one_record_per_processed_frame(self, tmp_path: Path) -> None:
        path = tmp_path / "debug.jsonl"
        with JsonlTrackingDebugWriter(path) as writer:
            VideoProcessor(
                FakeDetector([[detection_at(10.0)]]),
                FakeTracker(),
                make_config(tmp_path),
                tracking_debug_writer=writer,
            ).run(source(7), FakeSink())

        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 7, "diagnostics are not sampled"
        assert [r["frame"] for r in records] == list(range(1, 8))

    def test_records_carry_detections_and_reported_tracks(self, tmp_path: Path) -> None:
        path = tmp_path / "debug.jsonl"
        with JsonlTrackingDebugWriter(path) as writer:
            VideoProcessor(
                FakeDetector([[detection_at(10.0), detection_at(100.0)]]),
                FakeTracker(),
                make_config(tmp_path),
                tracking_debug_writer=writer,
            ).run(source(2), FakeSink())

        first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert len(first["detections"]) == 2
        assert len(first["reported_tracks"]) == 2
        assert first["detections"][0]["confidence"] == 0.9

    def test_dedup_disabled_leaves_the_record_shape_unchanged(self, tmp_path: Path) -> None:
        """A run without deduplication must produce exactly the old schema."""
        path = tmp_path / "debug.jsonl"
        with JsonlTrackingDebugWriter(path) as writer:
            VideoProcessor(
                FakeDetector([[detection_at(10.0)]]),
                FakeTracker(),
                make_config(tmp_path),
                tracking_debug_writer=writer,
            ).run(source(2), FakeSink())

        record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert set(record) == {"frame", "detections", "reported_tracks", "internal_tracks"}

    def test_dedup_disabled_passes_duplicates_straight_through(self, tmp_path: Path) -> None:
        """Off by default means off: a textbook duplicate pair must survive intact."""
        duplicate = Detection(
            bbox_xyxy=BoundingBox(10.0, 100.0, 50.0, 200.0), confidence=0.4, class_id=1
        )
        original = Detection(
            bbox_xyxy=BoundingBox(10.0, 100.0, 50.0, 200.0), confidence=0.9, class_id=1
        )
        path = tmp_path / "debug.jsonl"
        with JsonlTrackingDebugWriter(path) as writer:
            result = VideoProcessor(
                FakeDetector([[original, duplicate]]),
                FakeTracker(),
                make_config(tmp_path),
                tracking_debug_writer=writer,
            ).run(source(3), FakeSink())

        record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert len(record["detections"]) == 2, "both boxes reach the tracker"
        assert "raw_detections" not in record
        assert "detections_removed_by_deduplication" not in record
        assert result.detections_removed_by_deduplication == 0

    def test_dedup_enabled_records_before_and_after(self, tmp_path: Path) -> None:
        duplicate = Detection(
            bbox_xyxy=BoundingBox(10.0, 100.0, 50.0, 200.0), confidence=0.4, class_id=1
        )
        original = Detection(
            bbox_xyxy=BoundingBox(10.0, 100.0, 50.0, 200.0), confidence=0.9, class_id=1
        )
        path = tmp_path / "debug.jsonl"
        with JsonlTrackingDebugWriter(path) as writer:
            result = VideoProcessor(
                FakeDetector([[original, duplicate]]),
                FakeTracker(),
                make_config(tmp_path, deduplicate_detections=True),
                tracking_debug_writer=writer,
            ).run(source(3), FakeSink())

        record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert len(record["raw_detections"]) == 2
        assert len(record["tracker_input_detections"]) == 1
        assert len(record["detections"]) == 1, "detections still means tracker input"
        assert record["detections_removed_by_deduplication"] == 1
        assert record["removed_detections"][0]["confidence"] == 0.4
        assert result.detections_removed_by_deduplication == 3, "one per frame"

    def test_a_tracker_without_internals_still_produces_records(self, tmp_path: Path) -> None:
        """FakeTracker exposes no backend; the record is still written."""
        path = tmp_path / "debug.jsonl"
        with JsonlTrackingDebugWriter(path) as writer:
            VideoProcessor(
                FakeDetector([[detection_at(10.0)]]),
                FakeTracker(),
                make_config(tmp_path),
                tracking_debug_writer=writer,
            ).run(source(2), FakeSink())

        first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert first["internal_tracks"] == []

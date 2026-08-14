"""End-to-end: video → detector → tracker → projector → analyzer → JSONL.

Exercises the real composition path with fake detector and tracker so no model
weights are needed. The physical trajectory is known analytically, so the
metric values in the resulting file can be checked against arithmetic rather
than against whatever the code happened to produce.

:class:`TestMultiCameraBoundary` is the architectural test: two calibrated
cameras sharing a venue frame must keep their observations separate, even when
their track ids and venue coordinates coincide exactly.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from itertools import count
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from atreides_vision.behavior.analyzer import BehaviorAnalyzer
from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.behavior.coordinates import MotionSpace, NormalizedFrameProjector
from atreides_vision.behavior.reporting import (
    BEHAVIOR_SCHEMA_VERSION,
    BehaviorRunMetadata,
    CalibrationProvenance,
    JsonlBehaviorWriter,
    SourceProvenance,
)
from atreides_vision.calibration.fitting import CalibrationFit, fit_calibration
from atreides_vision.calibration.models import ReferenceFrame
from atreides_vision.calibration.projector import GroundPlaneProjector
from atreides_vision.config import PipelineConfig, VisionConfig, VisualizationConfig
from atreides_vision.models import BoundingBox, Detection
from atreides_vision.video.processor import VideoProcessor
from atreides_vision.video.source import VideoMetadata
from fakes import FakeSink, FakeSource, FakeTracker
from synthetic_cameras import (
    CAMERA_A,
    CAMERA_B,
    FRAME_A,
    FRAME_B,
    VENUE_ID,
    Matrix3x3,
    calibration_for,
    project_venue_to_image,
)

FPS = 25.0
DT = 1.0 / FPS

#: The floor region the synthetic reference points span.
CALIBRATION_REGION = [(0.0, 0.0), (12.0, 0.0), (12.0, 9.0), (0.0, 9.0)]

#: A known physical pace, in metres per second.
WALKING_SPEED = 1.4

VenuePath = Callable[[float], tuple[float, float]]


class ScriptedDetector:
    """Replays one detection per frame from a prepared list.

    A distinct fake from :class:`fakes.FakeDetector`: this one is indexed by
    frame so a trajectory can be laid out in advance, which is what makes the
    expected metric values computable by hand.
    """

    def __init__(self, per_frame: list[list[Detection]]) -> None:
        self._per_frame = per_frame
        self.calls = 0

    @property
    def name(self) -> str:
        return "Scripted Detector"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        index = min(self.calls, len(self._per_frame) - 1)
        self.calls += 1
        return list(self._per_frame[index])


def straight_walk(speed: float = WALKING_SPEED, y: float = 4.0) -> VenuePath:
    return lambda t: (1.0 + speed * t, y)


def detection_at_venue(matrix: Matrix3x3, x: float, y: float, scale: float = 1.0) -> Detection:
    """A detection whose bottom-centre anchor is the image of venue ``(x, y)``."""
    image_x, image_y = project_venue_to_image(matrix, x, y)
    image_x *= scale
    image_y *= scale
    half_width = 20.0 * scale
    height = 160.0 * scale
    return Detection(
        bbox_xyxy=BoundingBox(
            image_x - half_width, image_y - height, image_x + half_width, image_y
        ),
        confidence=0.9,
        class_id=1,
    )


def venue_script(
    path: VenuePath, frames: int, matrix: Matrix3x3 = CAMERA_A, scale: float = 1.0
) -> list[list[Detection]]:
    return [[detection_at_venue(matrix, *path(index * DT), scale=scale)] for index in range(frames)]


def source_for(frames: int, width: int, height: int) -> FakeSource:
    return FakeSource(
        [np.full((height, width, 3), 60, dtype=np.uint8) for _ in range(frames)],
        VideoMetadata(width=width, height=height, fps=FPS, frame_count=frames),
    )


def vision_config(tmp_path: Path) -> VisionConfig:
    return VisionConfig(
        pipeline=PipelineConfig(tmp_path / "in.mp4", tmp_path / "out.mp4"),
        visualization=VisualizationConfig(trail_length=5),
    )


def calibration_fit_for(
    matrix: Matrix3x3 = CAMERA_A,
    camera_id: str = "cam_a",
    reference_frame: ReferenceFrame = FRAME_A,
    venue_id: str = VENUE_ID,
) -> CalibrationFit:
    return fit_calibration(
        calibration_for(
            matrix,
            CALIBRATION_REGION,
            camera_id=camera_id,
            venue_id=venue_id,
            reference_frame=reference_frame,
        )
    )


def provenance_from(fit: CalibrationFit) -> CalibrationProvenance:
    """The same flattening the composition root performs."""
    quality = fit.quality
    return CalibrationProvenance(
        venue_id=fit.calibration.venue_id,
        surface_id=fit.calibration.surface_id,
        reference_width=fit.calibration.reference_frame.width,
        reference_height=fit.calibration.reference_frame.height,
        correspondence_count=quality.point_count,
        mean_venue_error_m=quality.mean_venue_error_m,
        max_venue_error_m=quality.max_venue_error_m,
        mean_image_error_px=quality.mean_image_error_px,
        max_image_error_px=quality.max_image_error_px,
        relative_error=quality.relative_error,
        venue_extent_m=quality.venue_extent_m,
    )


def run_pipeline(
    tmp_path: Path,
    output_name: str,
    script: list[list[Detection]],
    *,
    camera_id: str = "cam_a",
    fit: CalibrationFit | None = None,
    width: int = 1920,
    height: int = 1080,
    sink: FakeSink | None = None,
) -> list[dict[str, Any]]:
    """Drive the real VideoProcessor and return the parsed JSONL records."""
    if fit is None:
        analyzer = BehaviorAnalyzer(
            camera_id=camera_id,
            projector=NormalizedFrameProjector(width, height),
            config=BehaviorAnalysisConfig(),
        )
        metadata = BehaviorRunMetadata(
            camera_id=camera_id,
            space=MotionSpace.NORMALIZED_IMAGE,
            source=SourceProvenance(path="synthetic", width=width, height=height, fps=FPS),
        )
    else:
        analyzer = BehaviorAnalyzer(
            camera_id=camera_id,
            projector=GroundPlaneProjector(fit, frame_width=width, frame_height=height),
            config=BehaviorAnalysisConfig.for_ground_plane_metres(),
        )
        metadata = BehaviorRunMetadata(
            camera_id=camera_id,
            space=MotionSpace.GROUND_PLANE_METRES,
            source=SourceProvenance(path="synthetic", width=width, height=height, fps=FPS),
            calibration=provenance_from(fit),
        )

    path = tmp_path / output_name
    with JsonlBehaviorWriter(path, metadata) as writer:
        VideoProcessor(
            ScriptedDetector(script),
            FakeTracker(),
            vision_config(tmp_path),
            clock=count(start=0.0, step=0.04).__next__,
            behavior_analyzer=analyzer,
            behavior_writer=writer,
        ).run(source_for(len(script), width, height), sink or FakeSink())

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def observations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record["record_type"] == "observation"]


class TestMetricPipeline:
    """A known physical trajectory, all the way to the file."""

    def test_header_declares_metric_space_and_provenance(self, tmp_path: Path) -> None:
        records = run_pipeline(
            tmp_path, "b.jsonl", venue_script(straight_walk(), 50), fit=calibration_fit_for()
        )
        header = records[0]
        assert header["record_type"] == "run_metadata"
        assert header["schema_version"] == BEHAVIOR_SCHEMA_VERSION
        assert header["camera_id"] == "cam_a"
        assert header["space"] == "ground_plane_metres"
        assert header["position_unit"] == "m"
        assert header["calibration"]["venue_id"] == VENUE_ID

    def test_one_observation_per_frame_per_subject(self, tmp_path: Path) -> None:
        records = observations(
            run_pipeline(
                tmp_path, "b.jsonl", venue_script(straight_walk(), 50), fit=calibration_fit_for()
            )
        )
        assert len(records) == 50
        assert [record["frame_index"] for record in records] == list(range(1, 51))

    def test_projected_position_matches_the_known_venue_position(self, tmp_path: Path) -> None:
        path = straight_walk()
        records = observations(
            run_pipeline(tmp_path, "b.jsonl", venue_script(path, 50), fit=calibration_fit_for())
        )
        for record in records:
            expected = path((record["frame_index"] - 1) * DT)
            position = record["position"]
            assert (position["x"], position["y"]) == pytest.approx(expected, abs=1e-5)
            assert position["unit"] == "m"

    def test_timestamps_follow_the_source_frame_rate(self, tmp_path: Path) -> None:
        records = observations(
            run_pipeline(
                tmp_path, "b.jsonl", venue_script(straight_walk(), 51), fit=calibration_fit_for()
            )
        )
        assert records[0]["timestamp_seconds"] == pytest.approx(0.0)
        assert records[-1]["timestamp_seconds"] == pytest.approx(2.0)

    def test_speed_is_the_known_physical_speed(self, tmp_path: Path) -> None:
        """1.4 m/s walked, 1.4 m/s reported — end to end, through the file."""
        records = observations(
            run_pipeline(
                tmp_path,
                "b.jsonl",
                venue_script(straight_walk(WALKING_SPEED), 60),
                fit=calibration_fit_for(),
            )
        )
        final_signals = records[-1]["signals"]
        speed = next(entry for entry in final_signals if entry["type"] == "speed")
        assert speed["value"] == pytest.approx(WALKING_SPEED, rel=1e-4)
        assert speed["unit"] == "m/s"

    def test_subject_is_the_camera_local_track(self, tmp_path: Path) -> None:
        records = observations(
            run_pipeline(
                tmp_path, "b.jsonl", venue_script(straight_walk(), 20), fit=calibration_fit_for()
            )
        )
        assert all(record["camera_id"] == "cam_a" for record in records)
        assert all(record["subject"] == {"track_id": 0} for record in records)

    def test_runtime_resolution_change_preserves_metres(self, tmp_path: Path) -> None:
        """The V0.3.1 contract, now through the whole pipeline."""
        full = observations(
            run_pipeline(
                tmp_path, "full.jsonl", venue_script(straight_walk(), 50), fit=calibration_fit_for()
            )
        )
        half = observations(
            run_pipeline(
                tmp_path,
                "half.jsonl",
                venue_script(straight_walk(), 50, scale=0.5),
                fit=calibration_fit_for(),
                width=960,
                height=540,
            )
        )
        for full_record, half_record in zip(full, half, strict=True):
            assert half_record["position"]["x"] == pytest.approx(
                full_record["position"]["x"],
                abs=1e-6,
            )

    def test_output_is_deterministic(self, tmp_path: Path) -> None:
        def run(name: str) -> str:
            run_pipeline(
                tmp_path, name, venue_script(straight_walk(), 40), fit=calibration_fit_for()
            )
            return (tmp_path / name).read_text(encoding="utf-8")

        assert run("first.jsonl") == run("second.jsonl")


class TestNormalizedPipeline:
    def test_header_declares_normalized_space(self, tmp_path: Path) -> None:
        records = run_pipeline(tmp_path, "b.jsonl", venue_script(straight_walk(), 30))
        header = records[0]
        assert header["space"] == "normalized_image"
        assert header["position_unit"] == "normalized_frame_unit"
        assert header["is_physical"] is False
        assert header["calibration"] is None

    def test_positions_are_frame_fractions(self, tmp_path: Path) -> None:
        records = observations(run_pipeline(tmp_path, "b.jsonl", venue_script(straight_walk(), 30)))
        for record in records:
            position = record["position"]
            assert 0.0 <= position["x"] <= 1.0
            assert position["unit"] == "normalized_frame_unit"

    def test_signals_are_not_metric(self, tmp_path: Path) -> None:
        records = observations(run_pipeline(tmp_path, "b.jsonl", venue_script(straight_walk(), 50)))
        speed = next(entry for entry in records[-1]["signals"] if entry["type"] == "speed")
        assert speed["unit"] == "normalized_frame_unit/s"


class TestRenderingUnchanged:
    """Behaviour output is data only: it must not touch a single pixel."""

    def test_frames_are_byte_identical_with_and_without_behavior(self, tmp_path: Path) -> None:
        script = venue_script(straight_walk(), 20)

        with_behavior = FakeSink()
        run_pipeline(
            tmp_path,
            "b.jsonl",
            script,
            fit=calibration_fit_for(),
            sink=with_behavior,
        )

        without_behavior = FakeSink()
        VideoProcessor(
            ScriptedDetector(script),
            FakeTracker(),
            vision_config(tmp_path),
            clock=count(start=0.0, step=0.04).__next__,
        ).run(source_for(len(script), 1920, 1080), without_behavior)

        assert len(with_behavior.frames) == len(without_behavior.frames) == 20
        assert all(
            np.array_equal(a, b)
            for a, b in zip(with_behavior.frames, without_behavior.frames, strict=True)
        )

    def test_no_behavior_writer_produces_no_file(self, tmp_path: Path) -> None:
        script = venue_script(straight_walk(), 10)
        VideoProcessor(
            ScriptedDetector(script),
            FakeTracker(),
            vision_config(tmp_path),
        ).run(source_for(len(script), 1920, 1080), FakeSink())
        assert not list(tmp_path.glob("*.jsonl"))


class TestMultiCameraBoundary:
    """Two cameras, one venue frame — and no merging, ever."""

    @staticmethod
    def both_cameras(tmp_path: Path, frames: int = 50) -> tuple[list[dict[str, Any]], ...]:
        """The same physical walk, observed by two differently-placed cameras."""
        path = straight_walk()
        left = run_pipeline(
            tmp_path,
            "cam_a.jsonl",
            venue_script(path, frames, matrix=CAMERA_A),
            camera_id="cam_a",
            fit=calibration_fit_for(CAMERA_A, "cam_a", FRAME_A),
            width=FRAME_A.width,
            height=FRAME_A.height,
        )
        right = run_pipeline(
            tmp_path,
            "cam_b.jsonl",
            venue_script(path, frames, matrix=CAMERA_B),
            camera_id="cam_b",
            fit=calibration_fit_for(CAMERA_B, "cam_b", FRAME_B),
            width=FRAME_B.width,
            height=FRAME_B.height,
        )
        return (observations(left), observations(right))

    def test_both_cameras_recover_the_same_venue_positions(self, tmp_path: Path) -> None:
        left, right = self.both_cameras(tmp_path)
        for left_record, right_record in zip(left, right, strict=True):
            assert left_record["position"]["x"] == pytest.approx(
                right_record["position"]["x"],
                abs=1e-5,
            )
            assert left_record["position"]["y"] == pytest.approx(
                right_record["position"]["y"],
                abs=1e-5,
            )

    def test_both_cameras_report_the_same_speed(self, tmp_path: Path) -> None:
        left, right = self.both_cameras(tmp_path)

        def speed(record: dict[str, Any]) -> float:
            entry = next(item for item in record["signals"] if item["type"] == "speed")
            return float(entry["value"])

        assert speed(left[-1]) == pytest.approx(WALKING_SPEED, rel=1e-4)
        assert speed(right[-1]) == pytest.approx(speed(left[-1]), rel=1e-4)

    def test_identical_track_ids_stay_separate_records(self, tmp_path: Path) -> None:
        """The fake tracker gives both cameras track_id 0 — a routine collision."""
        left, right = self.both_cameras(tmp_path)
        assert all(record["subject"] == {"track_id": 0} for record in left)
        assert all(record["subject"] == {"track_id": 0} for record in right)

        # Identical track id, identical venue position, identical timestamp —
        # and still two independent observations, because the camera differs.
        assert {record["camera_id"] for record in left} == {"cam_a"}
        assert {record["camera_id"] for record in right} == {"cam_b"}

    def test_nothing_in_the_output_asserts_shared_identity(self, tmp_path: Path) -> None:
        """No venue-person id, no association, no merging."""
        left, right = self.both_cameras(tmp_path)
        for record in (*left, *right):
            assert set(record) <= {
                "record_type",
                "schema_version",
                "camera_id",
                "frame_index",
                "timestamp_seconds",
                "subject",
                "position",
                "signals",
                "venue_id",
                "surface_id",
                "zone_ids",
            }
            assert set(record["subject"]) == {"track_id"}

    def test_records_remain_distinguishable_when_merged(self, tmp_path: Path) -> None:
        """A consumer concatenating both files can still tell them apart."""
        left, right = self.both_cameras(tmp_path)
        merged = [*left, *right]
        keys = {
            (record["camera_id"], record["subject"]["track_id"], record["frame_index"])
            for record in merged
        }
        assert len(keys) == len(merged)

    def test_the_two_cameras_share_a_venue_frame(self, tmp_path: Path) -> None:
        left, right = self.both_cameras(tmp_path)
        assert {record["venue_id"] for record in left} == {VENUE_ID}
        assert {record["venue_id"] for record in right} == {VENUE_ID}

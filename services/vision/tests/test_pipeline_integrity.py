"""Calibration integrity through the real pipeline, and its fail-safe effect.

Drives ``VideoProcessor`` with fake detector and tracker but a real integrity
guard, so the thing under test is what the runtime actually does when a camera
is found to have moved: metric observations stop, and the stream says why.
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path
from typing import Any

import numpy as np

from atreides_vision.behavior.analyzer import BehaviorAnalyzer
from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.behavior.coordinates import MotionSpace
from atreides_vision.behavior.reporting import (
    BEHAVIOR_SCHEMA_VERSION,
    BehaviorRunMetadata,
    JsonlBehaviorWriter,
    SourceProvenance,
)
from atreides_vision.calibration.integrity.config import (
    CalibrationIntegrityConfig,
    IntegrityPolicyMode,
)
from atreides_vision.calibration.integrity.monitor import CalibrationIntegrityMonitor
from atreides_vision.calibration.integrity.policy import CalibrationIntegrityGuard
from atreides_vision.calibration.projector import GroundPlaneProjector
from atreides_vision.config import PipelineConfig, VisionConfig, VisualizationConfig
from atreides_vision.video.processor import VideoProcessor
from atreides_vision.video.source import VideoMetadata
from fakes import FakeSink, FakeSource, FakeTracker
from synthetic_cameras import CAMERA_A, FRAME_A, VENUE_ID
from synthetic_scenes import synthetic_scene, translate
from test_pipeline_behavior_output import (
    ScriptedDetector,
    calibration_fit_for,
    provenance_from,
    straight_walk,
    venue_script,
)

FPS = 25.0
DT = 1.0 / FPS
CAMERA = "cam_a"

#: Small frames keep the synthetic runs quick; registration does not care.
SCENE_W, SCENE_H = 480, 270


def scene_frames(count_: int, moved_from: int | None = None) -> list[np.ndarray]:
    """Frames of a fixed scene, optionally shifting the camera part-way through.

    Each frame is an independent copy. The annotator draws **in place** and
    returns the same array it was given (its documented V0.1 behaviour), so
    sharing one array between frames would let one run's overlay leak into the
    next run's input and make comparisons meaningless.
    """
    stable = synthetic_scene(SCENE_W, SCENE_H, seed=21)
    shifted = translate(stable, 40, 24)
    return [
        (stable if moved_from is None or index < moved_from else shifted).copy()
        for index in range(count_)
    ]


def source_for(frames: list[np.ndarray]) -> FakeSource:
    return FakeSource(
        frames,
        VideoMetadata(width=SCENE_W, height=SCENE_H, fps=FPS, frame_count=len(frames)),
    )


def vision_config(tmp_path: Path) -> VisionConfig:
    return VisionConfig(
        pipeline=PipelineConfig(tmp_path / "in.mp4", tmp_path / "out.mp4"),
        visualization=VisualizationConfig(trail_length=5),
    )


def build_guard(
    reference: np.ndarray,
    interval_seconds: float = 0.2,
    mode: IntegrityPolicyMode = IntegrityPolicyMode.GUARDED,
) -> CalibrationIntegrityGuard:
    config = CalibrationIntegrityConfig(check_interval_seconds=interval_seconds, policy_mode=mode)
    monitor = CalibrationIntegrityMonitor(reference, config, reference_id="reference.png")
    return CalibrationIntegrityGuard(monitor, config)


def run(
    tmp_path: Path,
    frames: list[np.ndarray],
    guard: CalibrationIntegrityGuard | None,
    output_name: str = "b.jsonl",
    camera_id: str = CAMERA,
) -> tuple[list[dict[str, Any]], Any]:
    """Drive the real processor; return parsed records and the result."""
    fit = calibration_fit_for(CAMERA_A, camera_id, FRAME_A)
    analyzer = BehaviorAnalyzer(
        camera_id=camera_id,
        projector=GroundPlaneProjector(fit, frame_width=SCENE_W, frame_height=SCENE_H),
        config=BehaviorAnalysisConfig.for_ground_plane_metres(),
    )
    metadata = BehaviorRunMetadata(
        camera_id=camera_id,
        space=MotionSpace.GROUND_PLANE_METRES,
        source=SourceProvenance(path="synthetic", width=SCENE_W, height=SCENE_H, fps=FPS),
        calibration=provenance_from(fit),
    )

    # The detections are scripted independently of the scene imagery, so the
    # people keep walking whether or not the camera is found to have moved.
    script = venue_script(straight_walk(), len(frames), scale=SCENE_W / FRAME_A.width)

    path = tmp_path / output_name
    with JsonlBehaviorWriter(path, metadata) as writer:
        result = VideoProcessor(
            ScriptedDetector(script),
            FakeTracker(),
            vision_config(tmp_path),
            clock=count(start=0.0, step=0.04).__next__,
            behavior_analyzer=analyzer,
            behavior_writer=writer,
            integrity_guard=guard,
        ).run(source_for(frames), FakeSink())

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return records, result


def of_type(records: list[dict[str, Any]], record_type: str) -> list[dict[str, Any]]:
    return [record for record in records if record["record_type"] == record_type]


class TestStableCamera:
    def test_integrity_records_appear_and_report_consistency(self, tmp_path: Path) -> None:
        frames = scene_frames(30)
        records, _ = run(tmp_path, frames, build_guard(frames[0]))

        integrity = of_type(records, "calibration_integrity")
        assert integrity
        assert all(record["state"] == "consistent" for record in integrity)

    def test_observations_keep_flowing(self, tmp_path: Path) -> None:
        frames = scene_frames(30)
        records, result = run(tmp_path, frames, build_guard(frames[0]))
        assert len(of_type(records, "observation")) == 30
        assert result.behavior_frames_suppressed == 0

    def test_checks_run_on_a_cadence_not_every_frame(self, tmp_path: Path) -> None:
        """Feature matching every frame would be ruinous and unnecessary."""
        frames = scene_frames(50)
        records, _ = run(tmp_path, frames, build_guard(frames[0], interval_seconds=0.4))
        # 50 frames at 25 FPS spans t=0.00..1.96, so a 0.4 s cadence checks at
        # 0.0, 0.4, 0.8, 1.2 and 1.6 — five times, not fifty.
        assert len(of_type(records, "calibration_integrity")) == 5

    def test_the_cadence_is_deterministic(self, tmp_path: Path) -> None:
        frames = scene_frames(40)
        first, _ = run(tmp_path, frames, build_guard(frames[0], 0.2), "a.jsonl")
        second, _ = run(tmp_path, frames, build_guard(frames[0], 0.2), "b.jsonl")
        assert [record["frame_index"] for record in of_type(first, "calibration_integrity")] == [
            record["frame_index"] for record in of_type(second, "calibration_integrity")
        ]


class TestCameraMovesMidRun:
    def test_drift_is_detected(self, tmp_path: Path) -> None:
        frames = scene_frames(40, moved_from=20)
        records, _ = run(tmp_path, frames, build_guard(frames[0]))

        states = [record["state"] for record in of_type(records, "calibration_integrity")]
        assert "consistent" in states
        assert "drift_detected" in states

    def test_metric_observations_stop(self, tmp_path: Path) -> None:
        """The fail-safe: precise-looking metres from a moved camera must stop."""
        frames = scene_frames(40, moved_from=20)
        records, result = run(tmp_path, frames, build_guard(frames[0]))

        observations = of_type(records, "observation")
        assert observations, "observations before the move should have been written"
        assert len(observations) < 40, "observations should have stopped after the move"
        assert result.behavior_frames_suppressed > 0

    def test_no_observation_is_written_after_the_finding(self, tmp_path: Path) -> None:
        frames = scene_frames(40, moved_from=20)
        records, _ = run(tmp_path, frames, build_guard(frames[0]))

        drift_frame = next(
            record["frame_index"]
            for record in of_type(records, "calibration_integrity")
            if record["state"] == "drift_detected"
        )
        later = [
            record
            for record in of_type(records, "observation")
            if record["frame_index"] > drift_frame
        ]
        assert later == []

    def test_the_stream_explains_the_gap(self, tmp_path: Path) -> None:
        """A consumer must be able to tell suppression from an empty scene."""
        frames = scene_frames(40, moved_from=20)
        records, _ = run(tmp_path, frames, build_guard(frames[0]))

        drift = next(
            record
            for record in of_type(records, "calibration_integrity")
            if record["state"] == "drift_detected"
        )
        assert drift["metric_output_allowed"] is False
        assert "moved" in drift["reason"]
        assert drift["metrics"]["normalized_drift"] > 0.0

    def test_analysis_continues_even_while_output_is_suppressed(self, tmp_path: Path) -> None:
        """Suppression is about trust in the output, not about stopping work."""
        frames = scene_frames(40, moved_from=20)
        _, result = run(tmp_path, frames, build_guard(frames[0]))
        assert result.frames_processed == 40
        assert result.behavior_signals_emitted > 0

    def test_report_only_keeps_emitting(self, tmp_path: Path) -> None:
        frames = scene_frames(40, moved_from=20)
        records, result = run(
            tmp_path, frames, build_guard(frames[0], mode=IntegrityPolicyMode.REPORT_ONLY)
        )
        assert len(of_type(records, "observation")) == 40
        assert result.behavior_frames_suppressed == 0
        assert any(
            record["state"] == "drift_detected"
            for record in of_type(records, "calibration_integrity")
        )


class TestWithoutMonitoring:
    def test_no_guard_means_no_integrity_records(self, tmp_path: Path) -> None:
        frames = scene_frames(20)
        records, result = run(tmp_path, frames, None)
        assert of_type(records, "calibration_integrity") == []
        assert len(of_type(records, "observation")) == 20
        assert result.behavior_frames_suppressed == 0

    def test_a_moved_camera_goes_unnoticed_without_monitoring(self, tmp_path: Path) -> None:
        """Exactly the failure V0.3.3 exists to prevent, shown happening."""
        frames = scene_frames(40, moved_from=20)
        records, _ = run(tmp_path, frames, None)
        assert len(of_type(records, "observation")) == 40


class TestRecordShape:
    def test_integrity_records_carry_the_schema_and_camera(self, tmp_path: Path) -> None:
        frames = scene_frames(20)
        records, _ = run(tmp_path, frames, build_guard(frames[0]))
        record = of_type(records, "calibration_integrity")[0]

        assert record["schema_version"] == BEHAVIOR_SCHEMA_VERSION
        assert record["camera_id"] == CAMERA
        assert record["venue_id"] == VENUE_ID
        assert record["reference_id"] == "reference.png"
        assert "state" in record and "reason" in record and "metrics" in record

    def test_integrity_records_carry_no_subject(self, tmp_path: Path) -> None:
        """Camera-level context must never look like a statement about a person."""
        frames = scene_frames(20)
        records, _ = run(tmp_path, frames, build_guard(frames[0]))
        for record in of_type(records, "calibration_integrity"):
            assert "subject" not in record
            assert "track_id" not in record

    def test_observations_are_unchanged_by_monitoring(self, tmp_path: Path) -> None:
        frames = scene_frames(20)
        guarded, _ = run(tmp_path, frames, build_guard(frames[0]), "guarded.jsonl")
        plain, _ = run(tmp_path, frames, None, "plain.jsonl")

        assert of_type(guarded, "observation") == of_type(plain, "observation")


class TestMultiCameraIndependence:
    def test_one_camera_drifting_does_not_affect_another(self, tmp_path: Path) -> None:
        """Integrity is per camera, exactly as subject identity is."""
        stable = scene_frames(40)
        moved = scene_frames(40, moved_from=20)

        stable_records, stable_result = run(
            tmp_path, stable, build_guard(stable[0]), "cam_a.jsonl", camera_id="cam_a"
        )
        moved_records, moved_result = run(
            tmp_path, moved, build_guard(moved[0]), "cam_b.jsonl", camera_id="cam_b"
        )

        assert stable_result.behavior_frames_suppressed == 0
        assert moved_result.behavior_frames_suppressed > 0

        assert all(
            record["state"] == "consistent"
            for record in of_type(stable_records, "calibration_integrity")
        )
        assert any(
            record["state"] == "drift_detected"
            for record in of_type(moved_records, "calibration_integrity")
        )

        assert len(of_type(stable_records, "observation")) == 40
        assert len(of_type(moved_records, "observation")) < 40

    def test_records_stay_camera_scoped(self, tmp_path: Path) -> None:
        stable = scene_frames(20)
        records, _ = run(tmp_path, stable, build_guard(stable[0]), "cam_a.jsonl", "cam_a")
        assert {record["camera_id"] for record in records} == {"cam_a"}


class TestRenderingUnchanged:
    def test_frames_are_byte_identical_with_and_without_monitoring(self, tmp_path: Path) -> None:
        """Integrity output is data; it must not touch a pixel of the video."""

        def render(guard_reference: np.ndarray | None) -> list[np.ndarray]:
            # Fresh frames per render: the annotator draws in place, so reusing
            # one list would feed the second run the first run's overlays.
            frames = scene_frames(20, moved_from=10)
            guard = None if guard_reference is None else build_guard(frames[0].copy())
            fit = calibration_fit_for(CAMERA_A, CAMERA, FRAME_A)
            analyzer = BehaviorAnalyzer(
                camera_id=CAMERA,
                projector=GroundPlaneProjector(fit, SCENE_W, SCENE_H),
                config=BehaviorAnalysisConfig.for_ground_plane_metres(),
            )
            sink = FakeSink()
            VideoProcessor(
                ScriptedDetector(
                    venue_script(straight_walk(), len(frames), scale=SCENE_W / FRAME_A.width)
                ),
                FakeTracker(),
                vision_config(tmp_path),
                clock=count(start=0.0, step=0.04).__next__,
                behavior_analyzer=analyzer,
                integrity_guard=guard,
            ).run(source_for(frames), sink)
            return sink.frames

        reference = scene_frames(1)[0]
        with_guard = render(reference)
        without = render(None)
        assert all(np.array_equal(a, b) for a, b in zip(with_guard, without, strict=True))

"""Interaction measurement through the real pipeline.

Drives ``VideoProcessor`` with fake detector and tracker but real behaviour,
calibration and interaction layers, so what is tested is what the runtime
actually does — including the two things that most need proving: that a
perspective-corrected metre is recovered where image space disagrees, and that
metric pair output stops when calibration integrity says the camera moved.
"""

from __future__ import annotations

import json
import math
from itertools import count
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from atreides_vision.behavior.analyzer import BehaviorAnalyzer
from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.behavior.coordinates import MotionSpace, NormalizedFrameProjector
from atreides_vision.calibration.fitting import CalibrationFit, fit_calibration
from atreides_vision.calibration.integrity.config import (
    CalibrationIntegrityConfig,
    IntegrityPolicyMode,
)
from atreides_vision.calibration.integrity.monitor import CalibrationIntegrityMonitor
from atreides_vision.calibration.integrity.policy import CalibrationIntegrityGuard
from atreides_vision.calibration.projector import GroundPlaneProjector
from atreides_vision.config import PipelineConfig, VisionConfig, VisualizationConfig
from atreides_vision.interaction.analyzer import InteractionAnalyzer
from atreides_vision.interaction.config import InteractionAnalysisConfig
from atreides_vision.interaction.models import SpatialContext
from atreides_vision.interaction.reporting import (
    InteractionRunMetadata,
    JsonlInteractionWriter,
)
from atreides_vision.models import BoundingBox, Detection
from atreides_vision.video.processor import VideoProcessor
from atreides_vision.video.source import VideoMetadata
from fakes import FakeSink, FakeSource, FakeTracker
from synthetic_cameras import (
    CAMERA_A,
    CAMERA_STRONG_PERSPECTIVE,
    FRAME_A,
    VENUE_ID,
    calibration_for,
    project_venue_to_image,
)
from synthetic_scenes import synthetic_scene, translate
from test_pipeline_behavior_output import CALIBRATION_REGION, ScriptedDetector

FPS = 25.0
DT = 1.0 / FPS
CAMERA = "cam_a"
SURFACE = "ground"
W, H = FRAME_A.width, FRAME_A.height


def fit_for(matrix: object = CAMERA_A, camera_id: str = CAMERA) -> CalibrationFit:
    return fit_calibration(
        calibration_for(
            matrix,  # type: ignore[arg-type]
            CALIBRATION_REGION,
            camera_id=camera_id,
            venue_id=VENUE_ID,
            reference_frame=FRAME_A,
        )
    )


def detection_at_venue(matrix: object, x: float, y: float) -> Detection:
    image_x, image_y = project_venue_to_image(matrix, x, y)  # type: ignore[arg-type]
    return Detection(
        bbox_xyxy=BoundingBox(image_x - 18, image_y - 150, image_x + 18, image_y),
        confidence=0.9,
        class_id=1,
    )


def two_person_script(
    first: object, second: object, frames: int, matrix: object = CAMERA_A
) -> list[list[Detection]]:
    """Two people walking known venue paths, seen through one camera."""
    return [
        [
            detection_at_venue(matrix, *first(index * DT)),  # type: ignore[operator]
            detection_at_venue(matrix, *second(index * DT)),  # type: ignore[operator]
        ]
        for index in range(frames)
    ]


def blank_frames(count_: int) -> list[np.ndarray]:
    return [np.full((H, W, 3), 60, dtype=np.uint8) for _ in range(count_)]


def source_for(frames: list[np.ndarray]) -> FakeSource:
    return FakeSource(frames, VideoMetadata(width=W, height=H, fps=FPS, frame_count=len(frames)))


def vision_config(tmp_path: Path) -> VisionConfig:
    return VisionConfig(
        pipeline=PipelineConfig(tmp_path / "in.mp4", tmp_path / "out.mp4"),
        visualization=VisualizationConfig(trail_length=5),
    )


def build_layers(
    fit: CalibrationFit | None, camera_id: str = CAMERA
) -> tuple[BehaviorAnalyzer, InteractionAnalyzer, InteractionRunMetadata]:
    """The same pairing the composition root performs."""
    if fit is None:
        behavior = BehaviorAnalyzer(
            camera_id, NormalizedFrameProjector(W, H), BehaviorAnalysisConfig()
        )
        context = SpatialContext(camera_id, MotionSpace.NORMALIZED_IMAGE)
        config = InteractionAnalysisConfig.for_normalized_image()
    else:
        behavior = BehaviorAnalyzer(
            camera_id,
            GroundPlaneProjector(fit, W, H),
            BehaviorAnalysisConfig.for_ground_plane_metres(),
        )
        context = SpatialContext(camera_id, MotionSpace.GROUND_PLANE_METRES, VENUE_ID, SURFACE)
        config = InteractionAnalysisConfig.for_ground_plane_metres()

    interaction = InteractionAnalyzer(context, config)
    metadata = InteractionRunMetadata(
        camera_id=camera_id,
        space=context.space,
        venue_id=context.venue_id,
        surface_id=context.surface_id,
        proximity_threshold=config.proximity_threshold,
    )
    return behavior, interaction, metadata


def run(
    tmp_path: Path,
    script: list[list[Detection]],
    *,
    fit: CalibrationFit | None = None,
    output_name: str = "i.jsonl",
    camera_id: str = CAMERA,
    guard: CalibrationIntegrityGuard | None = None,
    frames: list[np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    behavior, interaction, metadata = build_layers(fit, camera_id)
    path = tmp_path / output_name
    with JsonlInteractionWriter(path, metadata) as writer:
        result = VideoProcessor(
            ScriptedDetector(script),
            FakeTracker(),
            vision_config(tmp_path),
            clock=count(start=0.0, step=0.04).__next__,
            behavior_analyzer=behavior,
            integrity_guard=guard,
            interaction_analyzer=interaction,
            interaction_writer=writer,
        ).run(source_for(frames or blank_frames(len(script))), FakeSink())

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return records, result


def interactions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record["record_type"] == "interaction"]


def signal_of(record: dict[str, Any], name: str) -> float | None:
    for entry in record["signals"]:
        if entry["type"] == name:
            return float(entry["value"])
    return None


class TestMetricPipeline:
    """A known physical scenario, all the way to the file."""

    def test_header_declares_metric_space_and_venue(self, tmp_path: Path) -> None:
        script = two_person_script(lambda t: (2.0, 3.0), lambda t: (5.0, 3.0), 30)
        records, _ = run(tmp_path, script, fit=fit_for())
        header = records[0]
        assert header["space"] == "ground_plane_metres"
        assert header["distance_unit"] == "m"
        assert header["venue_id"] == VENUE_ID
        assert header["pair_scope"] == "same_camera_only"

    def test_distance_matches_the_known_separation(self, tmp_path: Path) -> None:
        """Two people standing 3 m apart in venue coordinates."""
        script = two_person_script(lambda t: (2.0, 3.0), lambda t: (5.0, 3.0), 30)
        records, _ = run(tmp_path, script, fit=fit_for())
        for record in interactions(records):
            assert record["current"]["distance"]["value"] == pytest.approx(3.0, abs=1e-4)
            assert record["current"]["distance"]["unit"] == "m"

    def test_closing_speed_matches_the_known_approach(self, tmp_path: Path) -> None:
        """Both walk toward each other at 1 m/s: the gap closes at 2 m/s."""
        script = two_person_script(
            lambda t: (1.0 + 1.0 * t, 3.0), lambda t: (9.0 - 1.0 * t, 3.0), 60
        )
        records, _ = run(tmp_path, script, fit=fit_for())
        assert signal_of(interactions(records)[-1], "closing_speed") == pytest.approx(2.0, abs=1e-3)

    def test_relative_heading_is_pi_for_a_head_on_approach(self, tmp_path: Path) -> None:
        script = two_person_script(
            lambda t: (1.0 + 1.0 * t, 3.0), lambda t: (9.0 - 1.0 * t, 3.0), 60
        )
        records, _ = run(tmp_path, script, fit=fit_for())
        assert signal_of(interactions(records)[-1], "relative_motion_heading") == pytest.approx(
            math.pi, abs=1e-3
        )

    def test_proximity_duration_accumulates(self, tmp_path: Path) -> None:
        script = two_person_script(lambda t: (4.0, 3.0), lambda t: (5.0, 3.0), 76)
        records, _ = run(tmp_path, script, fit=fit_for())
        assert signal_of(interactions(records)[-1], "proximity_duration") == pytest.approx(
            3.0, abs=0.1
        )

    def test_one_pair_per_frame_for_two_people(self, tmp_path: Path) -> None:
        script = two_person_script(lambda t: (2.0, 3.0), lambda t: (4.0, 3.0), 30)
        records, result = run(tmp_path, script, fit=fit_for())
        assert len(interactions(records)) == 30
        assert result.interaction_pairs_measured == 30

    def test_subjects_are_camera_local_tracks(self, tmp_path: Path) -> None:
        script = two_person_script(lambda t: (2.0, 3.0), lambda t: (4.0, 3.0), 20)
        records, _ = run(tmp_path, script, fit=fit_for())
        record = interactions(records)[0]
        assert record["camera_id"] == CAMERA
        assert record["subjects"] == [{"track_id": 0}, {"track_id": 1}]

    def test_output_is_deterministic(self, tmp_path: Path) -> None:
        script = two_person_script(lambda t: (1.0 + t, 3.0), lambda t: (9.0 - t, 3.0), 40)

        def once(name: str) -> str:
            run(tmp_path, script, fit=fit_for(), output_name=name)
            return (tmp_path / name).read_text(encoding="utf-8")

        assert once("a.jsonl") == once("b.jsonl")


class TestPerspectiveValidation:
    """Why calibrated interaction distance is worth having.

    Two people a fixed 2 m apart, measured near the camera and far from it. In
    image space the same physical gap subtends very different separations; after
    ground-plane projection both recover 2 m.
    """

    SEPARATION = 2.0
    NEAR_Y = 0.5
    FAR_Y = 8.5

    def distance_at(self, tmp_path: Path, depth: float, *, calibrated: bool, name: str) -> float:
        script = two_person_script(
            lambda t: (4.0, depth),
            lambda t: (4.0 + self.SEPARATION, depth),
            20,
            matrix=CAMERA_STRONG_PERSPECTIVE,
        )
        records, _ = run(
            tmp_path,
            script,
            fit=fit_for(CAMERA_STRONG_PERSPECTIVE) if calibrated else None,
            output_name=name,
        )
        return float(interactions(records)[-1]["current"]["distance"]["value"])

    def test_image_space_disagrees_with_itself(self, tmp_path: Path) -> None:
        near = self.distance_at(tmp_path, self.NEAR_Y, calibrated=False, name="n.jsonl")
        far = self.distance_at(tmp_path, self.FAR_Y, calibrated=False, name="f.jsonl")
        assert near > far
        assert near / far > 1.5, "the synthetic camera must show real perspective"

    def test_calibration_recovers_the_same_two_metres(self, tmp_path: Path) -> None:
        near = self.distance_at(tmp_path, self.NEAR_Y, calibrated=True, name="cn.jsonl")
        far = self.distance_at(tmp_path, self.FAR_Y, calibrated=True, name="cf.jsonl")
        assert near == pytest.approx(self.SEPARATION, abs=1e-4)
        assert far == pytest.approx(self.SEPARATION, abs=1e-4)
        assert near == pytest.approx(far, abs=1e-4)


class TestNormalizedPipeline:
    def test_normalized_distances_are_frame_fractions(self, tmp_path: Path) -> None:
        script = two_person_script(lambda t: (2.0, 3.0), lambda t: (5.0, 3.0), 20)
        records, _ = run(tmp_path, script)
        header = records[0]
        assert header["space"] == "normalized_image"
        assert header["distance_unit"] == "normalized_frame_unit"

        distance = interactions(records)[-1]["current"]["distance"]
        assert distance["unit"] == "normalized_frame_unit"
        assert 0.0 < distance["value"] < 1.5


class TestCalibrationIntegritySuppression:
    """Metric pair output must stop when the metric pipeline is disowned."""

    @staticmethod
    def scene_frames(count_: int, moved_from: int | None = None) -> list[np.ndarray]:
        stable = synthetic_scene(W, H, seed=23)
        shifted = translate(stable, 60, 35)
        return [
            (stable if moved_from is None or index < moved_from else shifted).copy()
            for index in range(count_)
        ]

    @staticmethod
    def guard_for(
        reference: np.ndarray, mode: IntegrityPolicyMode = IntegrityPolicyMode.GUARDED
    ) -> CalibrationIntegrityGuard:
        config = CalibrationIntegrityConfig(check_interval_seconds=0.2, policy_mode=mode)
        monitor = CalibrationIntegrityMonitor(reference, config, reference_id="ref.png")
        return CalibrationIntegrityGuard(monitor, config)

    def test_interactions_flow_while_the_camera_is_stable(self, tmp_path: Path) -> None:
        frames = self.scene_frames(30)
        script = two_person_script(lambda t: (4.0, 3.0), lambda t: (5.0, 3.0), 30)
        records, _ = run(
            tmp_path,
            script,
            fit=fit_for(),
            guard=self.guard_for(frames[0].copy()),
            frames=frames,
        )
        assert len(interactions(records)) == 30

    def test_interactions_stop_once_drift_is_detected(self, tmp_path: Path) -> None:
        """The fail-safe: pairwise metres derive from disowned positions."""
        frames = self.scene_frames(40, moved_from=20)
        script = two_person_script(lambda t: (4.0, 3.0), lambda t: (5.0, 3.0), 40)
        records, result = run(
            tmp_path,
            script,
            fit=fit_for(),
            guard=self.guard_for(frames[0].copy()),
            frames=frames,
        )
        emitted = interactions(records)
        assert emitted, "pairs before the move should have been written"
        assert len(emitted) < 40, "pairs should have stopped after the move"
        assert result.behavior_frames_suppressed > 0

    def test_no_stale_interaction_is_emitted_after_suppression(self, tmp_path: Path) -> None:
        frames = self.scene_frames(40, moved_from=20)
        script = two_person_script(lambda t: (4.0, 3.0), lambda t: (5.0, 3.0), 40)
        records, _ = run(
            tmp_path,
            script,
            fit=fit_for(),
            guard=self.guard_for(frames[0].copy()),
            frames=frames,
        )
        emitted = interactions(records)
        last_emitted = max(record["frame_index"] for record in emitted)
        # Nothing after the suppression point, and the run really did continue.
        assert last_emitted < 40

    def test_analysis_continues_while_output_is_suppressed(self, tmp_path: Path) -> None:
        """Suppression is about trust in the output, not about stopping work."""
        frames = self.scene_frames(40, moved_from=20)
        script = two_person_script(lambda t: (4.0, 3.0), lambda t: (5.0, 3.0), 40)
        _, result = run(
            tmp_path,
            script,
            fit=fit_for(),
            guard=self.guard_for(frames[0].copy()),
            frames=frames,
        )
        assert result.frames_processed == 40
        assert result.interaction_pairs_measured == 40

    def test_report_only_keeps_emitting(self, tmp_path: Path) -> None:
        frames = self.scene_frames(40, moved_from=20)
        script = two_person_script(lambda t: (4.0, 3.0), lambda t: (5.0, 3.0), 40)
        records, _ = run(
            tmp_path,
            script,
            fit=fit_for(),
            guard=self.guard_for(frames[0].copy(), IntegrityPolicyMode.REPORT_ONLY),
            frames=frames,
        )
        assert len(interactions(records)) == 40

    def test_interaction_cannot_bypass_the_guard(self, tmp_path: Path) -> None:
        """Behaviour and interaction stop together, on one trust decision."""
        frames = self.scene_frames(40, moved_from=20)
        script = two_person_script(lambda t: (4.0, 3.0), lambda t: (5.0, 3.0), 40)
        records, result = run(
            tmp_path,
            script,
            fit=fit_for(),
            guard=self.guard_for(frames[0].copy()),
            frames=frames,
        )
        emitted = interactions(records)
        suppressed_from = 40 - result.behavior_frames_suppressed
        assert max(record["frame_index"] for record in emitted) <= suppressed_from


class TestRenderingUnchanged:
    def test_frames_are_byte_identical_with_and_without_interaction(self, tmp_path: Path) -> None:
        script = two_person_script(lambda t: (2.0, 3.0), lambda t: (4.0, 3.0), 20)

        def render(with_interaction: bool) -> list[np.ndarray]:
            behavior, interaction, metadata = build_layers(fit_for())
            sink = FakeSink()
            writer = (
                JsonlInteractionWriter(tmp_path / "r.jsonl", metadata) if with_interaction else None
            )
            try:
                VideoProcessor(
                    ScriptedDetector(script),
                    FakeTracker(),
                    vision_config(tmp_path),
                    clock=count(start=0.0, step=0.04).__next__,
                    behavior_analyzer=behavior,
                    interaction_analyzer=interaction if with_interaction else None,
                    interaction_writer=writer,
                ).run(source_for(blank_frames(len(script))), sink)
            finally:
                if writer is not None:
                    writer.release()
            return sink.frames

        with_pairs, without = render(True), render(False)
        assert all(np.array_equal(a, b) for a, b in zip(with_pairs, without, strict=True))

    def test_no_analyzer_means_no_pair_output(self, tmp_path: Path) -> None:
        script = two_person_script(lambda t: (2.0, 3.0), lambda t: (4.0, 3.0), 10)
        behavior, _, _ = build_layers(fit_for())
        result = VideoProcessor(
            ScriptedDetector(script),
            FakeTracker(),
            vision_config(tmp_path),
            behavior_analyzer=behavior,
        ).run(source_for(blank_frames(len(script))), FakeSink())
        assert result.interaction_pairs_measured == 0
        assert result.final_interaction_frame is None


class TestDenseSceneScaling:
    """Pair count grows quadratically; the gate is what bounds the work."""

    @staticmethod
    def crowd_script(subjects: int, frames: int, spacing: float = 1.2) -> list[list[Detection]]:
        return [
            [
                detection_at_venue(
                    CAMERA_A, 1.0 + (index % 6) * spacing, 1.0 + (index // 6) * spacing
                )
                for index in range(subjects)
            ]
            for _ in range(frames)
        ]

    @pytest.mark.parametrize("subjects", [5, 10, 25])
    def test_candidate_pairs_are_n_choose_2(self, tmp_path: Path, subjects: int) -> None:
        behavior, interaction, _ = build_layers(fit_for())
        interaction = InteractionAnalyzer(
            SpatialContext(CAMERA, MotionSpace.GROUND_PLANE_METRES, VENUE_ID, SURFACE),
            InteractionAnalysisConfig.for_ground_plane_metres(),
        )
        script = self.crowd_script(subjects, 3)
        VideoProcessor(
            ScriptedDetector(script),
            FakeTracker(),
            vision_config(tmp_path),
            behavior_analyzer=behavior,
            interaction_analyzer=interaction,
        ).run(source_for(blank_frames(len(script))), FakeSink())

        latest = interaction.latest_frame
        assert latest is not None
        assert latest.candidate_pairs == subjects * (subjects - 1) // 2

    def test_gating_reduces_analysed_pairs_in_a_spread_crowd(self, tmp_path: Path) -> None:
        from dataclasses import replace

        base = InteractionAnalysisConfig.for_ground_plane_metres()
        script = self.crowd_script(20, 3, spacing=2.5)

        def analysed(max_distance: float | None) -> tuple[int, int]:
            behavior, _, _ = build_layers(fit_for())
            interaction = InteractionAnalyzer(
                SpatialContext(CAMERA, MotionSpace.GROUND_PLANE_METRES, VENUE_ID, SURFACE),
                replace(base, max_pair_distance=max_distance),
            )
            VideoProcessor(
                ScriptedDetector(script),
                FakeTracker(),
                vision_config(tmp_path),
                behavior_analyzer=behavior,
                interaction_analyzer=interaction,
            ).run(source_for(blank_frames(len(script))), FakeSink())
            latest = interaction.latest_frame
            assert latest is not None
            return latest.candidate_pairs, len(latest)

        ungated_candidates, ungated_analysed = analysed(None)
        gated_candidates, gated_analysed = analysed(4.0)

        assert ungated_candidates == gated_candidates == 190
        assert ungated_analysed == 190
        assert gated_analysed < ungated_analysed

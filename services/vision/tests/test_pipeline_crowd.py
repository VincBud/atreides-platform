"""Spatial-cluster measurement through the real pipeline.

Drives ``VideoProcessor`` with fake detector and tracker but real behaviour,
calibration and crowd layers. Two things most need proving here: that calibrated
geometry recovers the same cluster shape regardless of depth in the frame, and
that metric cluster output stops when calibration integrity says the camera
moved.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
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
from atreides_vision.crowd.analyzer import SpatialClusterAnalyzer
from atreides_vision.crowd.config import CrowdAnalysisConfig
from atreides_vision.crowd.reporting import CrowdRunMetadata, JsonlCrowdWriter
from atreides_vision.errors import CrowdError
from atreides_vision.interaction.models import SpatialContext
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

#: Venue positions of three people standing close together, relative to a depth.
TRIO = ((0.0, 0.0), (1.0, 0.0), (0.5, 0.9))


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


def static_script(
    venue_points: tuple[tuple[float, float], ...], frames: int, matrix: object = CAMERA_A
) -> list[list[Detection]]:
    """Several people holding fixed venue positions, seen through one camera."""
    return [[detection_at_venue(matrix, x, y) for x, y in venue_points] for _ in range(frames)]


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
    fit: CalibrationFit | None,
    crowd_config: CrowdAnalysisConfig | None = None,
    camera_id: str = CAMERA,
) -> tuple[BehaviorAnalyzer, SpatialClusterAnalyzer, CrowdRunMetadata]:
    """The same pairing the composition root performs."""
    if fit is None:
        behavior = BehaviorAnalyzer(
            camera_id, NormalizedFrameProjector(W, H), BehaviorAnalysisConfig()
        )
        context = SpatialContext(camera_id, MotionSpace.NORMALIZED_IMAGE)
        config = CrowdAnalysisConfig.for_normalized_image()
    else:
        behavior = BehaviorAnalyzer(
            camera_id,
            GroundPlaneProjector(fit, W, H),
            BehaviorAnalysisConfig.for_ground_plane_metres(),
        )
        context = SpatialContext(camera_id, MotionSpace.GROUND_PLANE_METRES, VENUE_ID, SURFACE)
        config = CrowdAnalysisConfig.for_ground_plane_metres()

    config = crowd_config or config
    crowd = SpatialClusterAnalyzer(context, config)
    metadata = CrowdRunMetadata(
        camera_id=camera_id,
        space=context.space,
        venue_id=context.venue_id,
        surface_id=context.surface_id,
        cluster_radius=config.cluster_radius,
        density_radius=config.density_radius,
        minimum_cluster_size=config.minimum_cluster_size,
    )
    return behavior, crowd, metadata


def run(
    tmp_path: Path,
    script: list[list[Detection]],
    *,
    fit: CalibrationFit | None = None,
    output_name: str = "c.jsonl",
    guard: CalibrationIntegrityGuard | None = None,
    frames: list[np.ndarray] | None = None,
    crowd_config: CrowdAnalysisConfig | None = None,
    camera_id: str = CAMERA,
) -> tuple[list[dict[str, Any]], Any]:
    behavior, crowd, metadata = build_layers(fit, crowd_config, camera_id)
    path = tmp_path / output_name
    with JsonlCrowdWriter(path, metadata) as writer:
        result = VideoProcessor(
            ScriptedDetector(script),
            FakeTracker(),
            vision_config(tmp_path),
            clock=count(start=0.0, step=0.04).__next__,
            behavior_analyzer=behavior,
            integrity_guard=guard,
            crowd_analyzer=crowd,
            crowd_writer=writer,
        ).run(source_for(frames or blank_frames(len(script))), FakeSink())

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return records, result


def of_type(records: list[dict[str, Any]], record_type: str) -> list[dict[str, Any]]:
    return [record for record in records if record["record_type"] == record_type]


class TestMetricPipeline:
    def test_header_declares_metric_space_and_venue(self, tmp_path: Path) -> None:
        records, _ = run(tmp_path, static_script(TRIO, 20), fit=fit_for())
        header = records[0]
        assert header["space"] == "ground_plane_metres"
        assert header["distance_unit"] == "m"
        assert header["density_unit"] == "subjects/m^2"
        assert header["venue_id"] == VENUE_ID
        assert header["cluster_scope"] == "same_camera_only"

    def test_three_close_people_form_one_cluster(self, tmp_path: Path) -> None:
        records, result = run(tmp_path, static_script(TRIO, 20), fit=fit_for())
        clusters = of_type(records, "spatial_cluster")
        assert len(clusters) == 20
        assert clusters[-1]["metrics"]["member_count"] == 3
        assert result.crowd_clusters_measured == 20

    def test_the_centroid_matches_the_known_venue_geometry(self, tmp_path: Path) -> None:
        records, _ = run(tmp_path, static_script(TRIO, 10), fit=fit_for())
        centroid = of_type(records, "spatial_cluster")[-1]["centroid"]
        expected_x = sum(x for x, _ in TRIO) / 3
        expected_y = sum(y for _, y in TRIO) / 3
        assert centroid["x"] == pytest.approx(expected_x, abs=1e-4)
        assert centroid["y"] == pytest.approx(expected_y, abs=1e-4)
        assert centroid["unit"] == "m"

    def test_a_distant_person_stays_isolated(self, tmp_path: Path) -> None:
        script = static_script((*TRIO, (9.0, 7.0)), 10)
        records, _ = run(tmp_path, script, fit=fit_for())
        summary = of_type(records, "crowd_frame")[-1]["summary"]
        assert summary["observed_subject_count"] == 4
        assert summary["clustered_subject_count"] == 3
        assert summary["isolated_subject_count"] == 1

    def test_persistence_accumulates_over_the_run(self, tmp_path: Path) -> None:
        records, _ = run(tmp_path, static_script(TRIO, 51), fit=fit_for())
        persistence = of_type(records, "spatial_cluster")[-1]["metrics"]["persistence_duration"]
        assert persistence["value"] == pytest.approx(2.0, abs=DT)
        assert persistence["unit"] == "s"

    def test_output_is_deterministic(self, tmp_path: Path) -> None:
        script = static_script(TRIO, 20)

        def once(name: str) -> str:
            run(tmp_path, script, fit=fit_for(), output_name=name)
            return (tmp_path / name).read_text(encoding="utf-8")

        assert once("a.jsonl") == once("b.jsonl")


class TestPerspectiveValidation:
    """Why calibrated cluster geometry is worth having.

    The same three people in the same physical arrangement, standing near the
    camera and far from it. In image space the arrangement shrinks with depth;
    after ground-plane projection both recover the same spread and separations.
    """

    NEAR_Y = 0.5
    FAR_Y = 8.0

    #: Wide enough to cluster the trio at both depths in normalized units, so
    #: image-space geometry can be compared at all. See the clustering-decision
    #: test below for what the default 0.12 radius does instead.
    WIDE_NORMALIZED = replace(CrowdAnalysisConfig.for_normalized_image(), cluster_radius=0.30)

    def measure_at(
        self,
        tmp_path: Path,
        depth: float,
        *,
        calibrated: bool,
        name: str,
        crowd_config: CrowdAnalysisConfig | None = None,
    ) -> dict[str, Any] | None:
        """Return the last cluster's metrics for the trio at a venue depth.

        ``crowd_config`` of ``None`` means the shipped default for the space.
        """
        venue_points = tuple((4.0 + dx, depth + dy) for dx, dy in TRIO)
        script = static_script(venue_points, 12, matrix=CAMERA_STRONG_PERSPECTIVE)
        records, _ = run(
            tmp_path,
            script,
            fit=fit_for(CAMERA_STRONG_PERSPECTIVE) if calibrated else None,
            output_name=name,
            crowd_config=crowd_config,
        )
        clusters = of_type(records, "spatial_cluster")
        return clusters[-1]["metrics"] if clusters else None

    def geometry_at(
        self, tmp_path: Path, depth: float, *, calibrated: bool, name: str
    ) -> tuple[float, float]:
        metrics = self.measure_at(
            tmp_path,
            depth,
            calibrated=calibrated,
            name=name,
            crowd_config=None if calibrated else self.WIDE_NORMALIZED,
        )
        assert metrics is not None, "expected a cluster to form"
        return (
            float(metrics["spatial_spread"]["value"]),
            float(metrics["mean_pair_distance"]["value"]),
        )

    def test_image_space_geometry_shrinks_with_depth(self, tmp_path: Path) -> None:
        near_spread, near_mean = self.geometry_at(
            tmp_path, self.NEAR_Y, calibrated=False, name="n.jsonl"
        )
        far_spread, far_mean = self.geometry_at(
            tmp_path, self.FAR_Y, calibrated=False, name="f.jsonl"
        )
        assert near_spread > far_spread
        assert near_mean / far_mean > 1.5, "the camera must show real perspective"

    def test_image_space_clustering_decision_depends_on_depth(self, tmp_path: Path) -> None:
        """The blunt cost of uncalibrated clustering, at the shipped default.

        One physical arrangement, one radius: three people who cluster when they
        stand at the back of the room are not a cluster at the front, because a
        fixed fraction of the frame covers less floor as they approach.
        """
        assert self.measure_at(tmp_path, self.NEAR_Y, calibrated=False, name="dn.jsonl") is None
        assert self.measure_at(tmp_path, self.FAR_Y, calibrated=False, name="df.jsonl") is not None

    def test_calibration_recovers_the_same_arrangement(self, tmp_path: Path) -> None:
        near_spread, near_mean = self.geometry_at(
            tmp_path, self.NEAR_Y, calibrated=True, name="cn.jsonl"
        )
        far_spread, far_mean = self.geometry_at(
            tmp_path, self.FAR_Y, calibrated=True, name="cf.jsonl"
        )
        assert near_spread == pytest.approx(far_spread, abs=1e-4)
        assert near_mean == pytest.approx(far_mean, abs=1e-4)

    def test_calibration_clusters_the_arrangement_at_both_depths(self, tmp_path: Path) -> None:
        """The same metric radius reaches the same decision at either depth."""
        for depth, name in ((self.NEAR_Y, "mn.jsonl"), (self.FAR_Y, "mf.jsonl")):
            metrics = self.measure_at(tmp_path, depth, calibrated=True, name=name)
            assert metrics is not None
            assert metrics["member_count"] == 3

    def test_the_recovered_geometry_matches_the_known_layout(self, tmp_path: Path) -> None:
        """The trio's true RMS radius, computed by hand from TRIO."""
        centre_x = sum(x for x, _ in TRIO) / 3
        centre_y = sum(y for _, y in TRIO) / 3
        expected = math.sqrt(sum((x - centre_x) ** 2 + (y - centre_y) ** 2 for x, y in TRIO) / 3)
        spread, _ = self.geometry_at(tmp_path, self.FAR_Y, calibrated=True, name="k.jsonl")
        assert spread == pytest.approx(expected, abs=1e-4)


class TestNormalizedPipeline:
    def test_normalized_density_is_not_labelled_as_square_metres(self, tmp_path: Path) -> None:
        records, _ = run(tmp_path, static_script(TRIO, 10))
        header = records[0]
        assert header["space"] == "normalized_image"
        assert header["density_unit"] == "subjects/normalized_area_unit"

        clusters = of_type(records, "spatial_cluster")
        if clusters:
            assert (
                clusters[-1]["metrics"]["mean_local_density"]["unit"]
                == "subjects/normalized_area_unit"
            )


class TestCalibrationIntegritySuppression:
    """Metric cluster output must stop when the metric pipeline is disowned."""

    @staticmethod
    def scene_frames(count_: int, moved_from: int | None = None) -> list[np.ndarray]:
        stable = synthetic_scene(W, H, seed=29)
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

    def test_clusters_flow_while_the_camera_is_stable(self, tmp_path: Path) -> None:
        frames = self.scene_frames(30)
        records, _ = run(
            tmp_path,
            static_script(TRIO, 30),
            fit=fit_for(),
            guard=self.guard_for(frames[0].copy()),
            frames=frames,
        )
        assert len(of_type(records, "spatial_cluster")) == 30

    def test_clusters_stop_once_drift_is_detected(self, tmp_path: Path) -> None:
        """The fail-safe: cluster metres derive from disowned positions."""
        frames = self.scene_frames(40, moved_from=20)
        records, result = run(
            tmp_path,
            static_script(TRIO, 40),
            fit=fit_for(),
            guard=self.guard_for(frames[0].copy()),
            frames=frames,
        )
        emitted = of_type(records, "spatial_cluster")
        assert emitted, "clusters before the move should have been written"
        assert len(emitted) < 40, "clusters should have stopped after the move"
        assert result.behavior_frames_suppressed > 0

    def test_no_stale_cluster_leaks_after_suppression(self, tmp_path: Path) -> None:
        frames = self.scene_frames(40, moved_from=20)
        records, result = run(
            tmp_path,
            static_script(TRIO, 40),
            fit=fit_for(),
            guard=self.guard_for(frames[0].copy()),
            frames=frames,
        )
        suppressed_from = 40 - result.behavior_frames_suppressed
        for record in of_type(records, "spatial_cluster") + of_type(records, "crowd_frame"):
            assert record["frame_index"] <= suppressed_from

    def test_analysis_continues_while_output_is_suppressed(self, tmp_path: Path) -> None:
        frames = self.scene_frames(40, moved_from=20)
        _, result = run(
            tmp_path,
            static_script(TRIO, 40),
            fit=fit_for(),
            guard=self.guard_for(frames[0].copy()),
            frames=frames,
        )
        assert result.frames_processed == 40
        assert result.crowd_clusters_measured == 40

    def test_report_only_keeps_emitting(self, tmp_path: Path) -> None:
        frames = self.scene_frames(40, moved_from=20)
        records, _ = run(
            tmp_path,
            static_script(TRIO, 40),
            fit=fit_for(),
            guard=self.guard_for(frames[0].copy(), IntegrityPolicyMode.REPORT_ONLY),
            frames=frames,
        )
        assert len(of_type(records, "spatial_cluster")) == 40

    def test_crowd_cannot_bypass_the_shared_trust_decision(self, tmp_path: Path) -> None:
        """One policy governs behaviour, interaction and crowd alike."""
        frames = self.scene_frames(40, moved_from=20)
        records, result = run(
            tmp_path,
            static_script(TRIO, 40),
            fit=fit_for(),
            guard=self.guard_for(frames[0].copy()),
            frames=frames,
        )
        emitted = of_type(records, "crowd_frame")
        assert max(record["frame_index"] for record in emitted) <= (
            40 - result.behavior_frames_suppressed
        )


class TestMultiCameraBoundary:
    """Clustering never crosses a camera, and the output cannot imply it did.

    Two cameras watching people standing at the same venue coordinates produce
    two independent streams. Nothing in either stream lets a reader join a
    member of one to a member of the other.
    """

    def two_camera_streams(
        self, tmp_path: Path
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        script = static_script(TRIO, 12)
        first, _ = run(
            tmp_path,
            script,
            fit=fit_for(camera_id="cam_a"),
            output_name="a.jsonl",
            camera_id="cam_a",
        )
        second, _ = run(
            tmp_path,
            script,
            fit=fit_for(camera_id="cam_b"),
            output_name="b.jsonl",
            camera_id="cam_b",
        )
        return first, second

    def test_each_stream_names_only_its_own_camera(self, tmp_path: Path) -> None:
        first, second = self.two_camera_streams(tmp_path)
        assert {record["camera_id"] for record in first} == {"cam_a"}
        assert {record["camera_id"] for record in second} == {"cam_b"}

    def test_no_cluster_mixes_cameras(self, tmp_path: Path) -> None:
        first, second = self.two_camera_streams(tmp_path)
        for records in (first, second):
            for cluster in of_type(records, "spatial_cluster"):
                # Members are camera-local track ids under the record's own
                # camera; there is no per-member camera field to cross.
                assert all(set(member) == {"track_id"} for member in cluster["members"])

    def test_the_streams_carry_no_shared_subject_identifier(self, tmp_path: Path) -> None:
        """Equal track ids across cameras are not evidence of the same person."""
        first, second = self.two_camera_streams(tmp_path)
        a_tracks = {
            member["track_id"]
            for cluster in of_type(first, "spatial_cluster")
            for member in cluster["members"]
        }
        b_tracks = {
            member["track_id"]
            for cluster in of_type(second, "spatial_cluster")
            for member in cluster["members"]
        }
        assert a_tracks == b_tracks, "the same local numbering, deliberately unjoinable"
        for records in (first, second):
            text = json.dumps(records)
            # "identity_scope" is a declaration of the boundary, not a crossing
            # of it, so the check names identifiers rather than the word itself.
            for banned in (
                "person_id",
                "global_id",
                "venue_person",
                "embedding",
                "appearance",
                "facial",
                "biometric",
            ):
                assert banned not in text

    def test_the_header_declares_the_scope(self, tmp_path: Path) -> None:
        first, second = self.two_camera_streams(tmp_path)
        for records in (first, second):
            assert records[0]["cluster_scope"] == "same_camera_only"
            assert records[0]["identity_scope"] == "camera_local_tracks"

    def test_an_analyzer_refuses_a_frame_from_another_camera(self, tmp_path: Path) -> None:
        """The pipeline cannot be wired into fusing two cameras by accident."""
        _, crowd_b, _ = build_layers(fit_for(camera_id="cam_b"), camera_id="cam_b")
        behavior_a, _, _ = build_layers(fit_for(camera_id="cam_a"), camera_id="cam_a")
        script = static_script(TRIO, 3)
        VideoProcessor(
            ScriptedDetector(script),
            FakeTracker(),
            vision_config(tmp_path),
            behavior_analyzer=behavior_a,
        ).run(source_for(blank_frames(len(script))), FakeSink())

        frame = behavior_a.latest_frame
        assert frame is not None
        with pytest.raises(CrowdError, match="camera"):
            crowd_b.update(frame)


class TestRenderingUnchanged:
    def test_frames_are_byte_identical_with_and_without_crowd(self, tmp_path: Path) -> None:
        script = static_script(TRIO, 20)

        def render(with_crowd: bool) -> list[np.ndarray]:
            behavior, crowd, metadata = build_layers(fit_for())
            sink = FakeSink()
            writer = JsonlCrowdWriter(tmp_path / "r.jsonl", metadata) if with_crowd else None
            try:
                VideoProcessor(
                    ScriptedDetector(script),
                    FakeTracker(),
                    vision_config(tmp_path),
                    clock=count(start=0.0, step=0.04).__next__,
                    behavior_analyzer=behavior,
                    crowd_analyzer=crowd if with_crowd else None,
                    crowd_writer=writer,
                ).run(source_for(blank_frames(len(script))), sink)
            finally:
                if writer is not None:
                    writer.release()
            return sink.frames

        with_clusters, without = render(True), render(False)
        assert all(np.array_equal(a, b) for a, b in zip(with_clusters, without, strict=True))

    def test_no_analyzer_means_no_crowd_output(self, tmp_path: Path) -> None:
        script = static_script(TRIO, 10)
        behavior, _, _ = build_layers(fit_for())
        result = VideoProcessor(
            ScriptedDetector(script),
            FakeTracker(),
            vision_config(tmp_path),
            behavior_analyzer=behavior,
        ).run(source_for(blank_frames(len(script))), FakeSink())
        assert result.crowd_clusters_measured == 0
        assert result.final_crowd_frame is None


class TestOneBehaviorFrameFansOut:
    def test_all_three_layers_share_one_behaviour_analysis(self, tmp_path: Path) -> None:
        """The frame is analysed once and consumed three times, never re-projected."""
        from atreides_vision.interaction.analyzer import InteractionAnalyzer
        from atreides_vision.interaction.config import InteractionAnalysisConfig

        fit = fit_for()
        behavior, crowd, _ = build_layers(fit)
        interaction = InteractionAnalyzer(
            SpatialContext(CAMERA, MotionSpace.GROUND_PLANE_METRES, VENUE_ID, SURFACE),
            InteractionAnalysisConfig.for_ground_plane_metres(),
        )
        script = static_script(TRIO, 15)

        result = VideoProcessor(
            ScriptedDetector(script),
            FakeTracker(),
            vision_config(tmp_path),
            behavior_analyzer=behavior,
            interaction_analyzer=interaction,
            crowd_analyzer=crowd,
        ).run(source_for(blank_frames(len(script))), FakeSink())

        # One behaviour pass drove all three: three subjects each frame, three
        # pairs each frame, one cluster each frame.
        assert result.frames_processed == 15
        assert result.behavior_signals_emitted > 0
        assert result.interaction_pairs_measured == 45
        assert result.crowd_clusters_measured == 15

        latest = behavior.latest_frame
        assert latest is not None
        assert len(latest.snapshots) == 3

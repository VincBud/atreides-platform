"""The V0.4.8 end-to-end venue validation runner.

Every test here drives the *real* runner over the *real* camera-local pipeline
— the actual ``VideoProcessor``, behaviour analyzer, projector, adapter,
``VenuePipeline``, occupancy and both report writers. Only the three seams the
runner already takes as parameters are faked: the detector, the tracker and the
video source. No weights, no network, no GPU, and no second implementation of
anything the runner orchestrates.

What these tests prove is *execution*: that the wiring runs, that ordering and
timing survive it, that the artifacts are byte-deterministic, and that the
conservative outcomes are reached where the evidence does not support a merge.
None of them is an accuracy measurement, because none of this footage exists.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from atreides_vision.association.models import AssociationOutcome
from atreides_vision.calibration.models import ReferenceFrame
from atreides_vision.errors import VideoSourceError
from atreides_vision.models import Detection
from atreides_vision.timing import FrameRateSource, TimeAlignmentBasis
from atreides_vision.venue.manifest import VenueCamera, VenueSessionManifest, VenueStreamEntry
from atreides_vision.venue_validation import (
    GROUND_TRUTH_STATUS,
    VenueValidationCameraConfig,
    VenueValidationConfig,
    VenueValidationSummary,
    run_venue_validation,
    summary_record,
)
from atreides_vision.video.source import VideoMetadata, VideoSource
from fakes import FakeSource, FakeTracker
from synthetic_cameras import (
    CAMERA_A,
    CAMERA_B,
    FRAME_A,
    FRAME_B,
    Matrix3x3,
    calibration_for,
    track_at_venue,
)
from synthetic_scenes import synthetic_scene, translate

FPS = 10.0

#: Two cameras of different resolutions, so a detector can tell them apart from
#: the frame alone. Both are the synthetic reference frames scaled down: small
#: frames keep the run quick and the projector folds the rescale in itself.
SCALE = 0.2
DIMS_A = (int(FRAME_A.width * SCALE), int(FRAME_A.height * SCALE))
DIMS_B = (int(FRAME_B.width * SCALE), int(FRAME_B.height * SCALE))

#: The floor area both synthetic cameras are calibrated over, in metres.
CALIBRATION_REGION = ((0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0), (5.0, 4.0), (2.0, 6.0))

#: Where the synthetic subject stands. Inside both calibrated regions.
STAND = (5.0, 4.0)

CAM_A = "cam_a"
CAM_B = "cam_b"


class ShapeDetector:
    """A detector that replays scripted detections keyed by frame shape.

    Detection is stateless and per-frame, so one detector instance is shared
    across cameras exactly as the runner shares the real one. Keying on the
    frame's own dimensions is what lets that single instance answer differently
    for two cameras without holding any per-camera state — a stateful script
    would make the result depend on the order cameras happened to be processed.
    """

    def __init__(self, by_shape: Mapping[tuple[int, int], Sequence[Detection]]) -> None:
        self._by_shape = {key: list(value) for key, value in by_shape.items()}
        self.calls = 0

    @property
    def name(self) -> str:
        return "Shape Detector"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        height, width = frame.shape[:2]
        return list(self._by_shape.get((height, width), ()))


class ExplodingSource:
    """A source that fails part-way through, as a dropped stream does."""

    def __init__(self, metadata: VideoMetadata, good_frames: int) -> None:
        self._metadata = metadata
        self._good_frames = good_frames
        self.released = False

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    def frames(self) -> Iterator[np.ndarray]:
        for _ in range(self._good_frames):
            yield blank(self._metadata.width, self._metadata.height)
        raise VideoSourceError("The stream stopped mid-session.")

    def release(self) -> None:
        self.released = True


def blank(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def detection_at(matrix: Matrix3x3, venue_point: tuple[float, float]) -> Detection:
    """A detection whose bottom-centre anchor is the image of ``venue_point``."""
    track = track_at_venue(matrix, venue_point[0], venue_point[1], scale=SCALE)
    return Detection(bbox_xyxy=track.bbox_xyxy, confidence=0.9, class_id=0)


def standing_detector() -> ShapeDetector:
    """One stationary person at :data:`STAND`, seen by both synthetic cameras."""
    return ShapeDetector(
        {
            (DIMS_A[1], DIMS_A[0]): [detection_at(CAMERA_A, STAND)],
            (DIMS_B[1], DIMS_B[0]): [detection_at(CAMERA_B, STAND)],
        }
    )


def source_for(
    width: int,
    height: int,
    frame_count: int,
    fps: float = FPS,
    fps_source: FrameRateSource = FrameRateSource.CONTAINER,
) -> FakeSource:
    return FakeSource(
        [blank(width, height) for _ in range(frame_count)],
        VideoMetadata(
            width=width, height=height, fps=fps, frame_count=frame_count, fps_source=fps_source
        ),
    )


def camera_entry(
    path: Path,
    *,
    offset_seconds: float | None = 0.0,
    accuracy_seconds: float | None = 0.05,
) -> VenueStreamEntry:
    return VenueStreamEntry(
        input_path=path,
        start_offset_seconds=offset_seconds,
        offset_accuracy_seconds=accuracy_seconds,
    )


def manifest_for(
    *cameras: VenueCamera, session_id: str = "validation_session"
) -> VenueSessionManifest:
    """A manifest built directly, so no test needs a decodable file on disk."""
    return VenueSessionManifest(session_id=session_id, cameras=tuple(cameras))


def write_calibration(
    tmp_path: Path,
    name: str,
    matrix: Matrix3x3,
    camera_id: str,
    reference_frame: ReferenceFrame,
) -> Path:
    calibration = calibration_for(
        matrix,
        CALIBRATION_REGION,
        camera_id=camera_id,
        reference_frame=reference_frame,
    )
    path = tmp_path / name
    path.write_text(json.dumps(calibration.to_record()), encoding="utf-8")
    return path


def run(
    config: VenueValidationConfig,
    sources: Mapping[Path, VideoSource],
    detector: ShapeDetector,
) -> VenueValidationSummary:
    return run_venue_validation(
        config,
        detector=detector,
        tracker_factory=lambda _config: FakeTracker(),
        source_factory=lambda entry: sources[entry.input_path],
    )


def records(path: Path) -> list[dict[str, object]]:
    """Every JSONL record in ``path``, metadata header included."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def two_camera_setup(
    tmp_path: Path,
    *,
    frames_a: int = 10,
    frames_b: int = 10,
    calibrate_b: bool = True,
    order: tuple[str, str] = (CAM_A, CAM_B),
    output_dir: Path | None = None,
) -> tuple[VenueValidationConfig, dict[Path, VideoSource], ShapeDetector]:
    """Two calibrated cameras watching one synthetic subject at :data:`STAND`."""
    path_a = tmp_path / "a.mp4"
    path_b = tmp_path / "b.mp4"
    cameras = {
        CAM_A: VenueCamera(camera_id=CAM_A, sources=(camera_entry(path_a),)),
        CAM_B: VenueCamera(camera_id=CAM_B, sources=(camera_entry(path_b),)),
    }
    camera_configs = [
        VenueValidationCameraConfig(
            camera_id=CAM_A,
            calibration_path=write_calibration(tmp_path, "cal_a.json", CAMERA_A, CAM_A, FRAME_A),
        )
    ]
    if calibrate_b:
        camera_configs.append(
            VenueValidationCameraConfig(
                camera_id=CAM_B,
                calibration_path=write_calibration(
                    tmp_path, "cal_b.json", CAMERA_B, CAM_B, FRAME_B
                ),
            )
        )
    config = VenueValidationConfig(
        manifest=manifest_for(*(cameras[camera_id] for camera_id in order)),
        output_dir=output_dir or tmp_path / "artifacts",
        cameras=tuple(camera_configs),
    )
    sources: dict[Path, VideoSource] = {
        path_a: source_for(*DIMS_A, frames_a),
        path_b: source_for(*DIMS_B, frames_b),
    }
    return config, sources, standing_detector()


class TestSingleCamera:
    """One camera is the degenerate multi-camera case, not a separate mode."""

    def test_a_single_camera_session_runs_end_to_end(self, tmp_path: Path) -> None:
        path = tmp_path / "a.mp4"
        config = VenueValidationConfig(
            manifest=manifest_for(VenueCamera(camera_id=CAM_A, sources=(camera_entry(path),))),
            output_dir=tmp_path / "artifacts",
        )
        summary = run(config, {path: source_for(*DIMS_A, 6)}, standing_detector())

        assert summary.venue_instants == 6
        assert summary.source_frames_processed == 6
        assert summary.observations_emitted == 6
        assert summary.contributing_camera_ids == (CAM_A,)
        assert config.association_report_path.exists()
        assert config.occupancy_report_path.exists()
        assert config.summary_path.exists()

    def test_one_camera_produces_no_cross_camera_decisions(self, tmp_path: Path) -> None:
        path = tmp_path / "a.mp4"
        config = VenueValidationConfig(
            manifest=manifest_for(VenueCamera(camera_id=CAM_A, sources=(camera_entry(path),))),
            output_dir=tmp_path / "artifacts",
        )
        summary = run(config, {path: source_for(*DIMS_A, 6)}, standing_detector())

        # Association is cross-camera by definition: with one camera there is
        # no pair to judge, which is not an error and not a merge.
        assert sum(summary.association_outcomes.values()) == 0
        assert summary.suppressed_duplicates == 0

    def test_single_camera_occupancy_is_its_own_active_tracks(self, tmp_path: Path) -> None:
        path = tmp_path / "a.mp4"
        config = VenueValidationConfig(
            manifest=manifest_for(VenueCamera(camera_id=CAM_A, sources=(camera_entry(path),))),
            output_dir=tmp_path / "artifacts",
        )
        summary = run(config, {path: source_for(*DIMS_A, 6)}, standing_detector())

        assert summary.min_occupancy == 1
        assert summary.max_occupancy == 1
        assert summary.mean_occupancy == 1.0

    def test_an_uncalibrated_camera_produces_no_venue_geometry(self, tmp_path: Path) -> None:
        path = tmp_path / "a.mp4"
        config = VenueValidationConfig(
            manifest=manifest_for(VenueCamera(camera_id=CAM_A, sources=(camera_entry(path),))),
            output_dir=tmp_path / "artifacts",
        )
        summary = run(config, {path: source_for(*DIMS_A, 4)}, standing_detector())

        assert summary.streams[0].calibrated is False
        assert summary.streams[0].integrity_monitored is False


class TestTwoCameras:
    """The ordinary V1 case: two cameras, one venue clock, one decision path."""

    def test_two_calibrated_cameras_associate_and_suppress_the_duplicate(
        self, tmp_path: Path
    ) -> None:
        config, sources, detector = two_camera_setup(tmp_path)
        summary = run(config, sources, detector)

        assert summary.observations_emitted == 20
        assert summary.association_outcomes.get(AssociationOutcome.ASSOCIATED, 0) > 0
        assert summary.venue_subjects_created >= 1
        # One person seen twice is counted once, and the suppression is
        # reported rather than silently folded away.
        assert summary.min_occupancy == 1
        assert summary.suppressed_duplicates > 0

    def test_an_uncalibrated_second_camera_is_never_merged(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, calibrate_b=False)
        summary = run(config, sources, detector)

        # No shared physical frame means nothing was measured, so nothing may
        # be concluded — and the two tracklets both keep counting.
        assert summary.association_outcomes.get(AssociationOutcome.NOT_COMPARABLE, 0) > 0
        assert summary.association_outcomes.get(AssociationOutcome.ASSOCIATED, 0) == 0
        assert summary.suppressed_duplicates == 0
        assert summary.max_occupancy == 2

    def test_untrusted_geometry_refuses_to_keep_merging(self, tmp_path: Path) -> None:
        """A camera that demonstrably moved stops being comparable."""
        width, height = 480, 270
        scene = synthetic_scene(width, height, seed=21)
        reference = tmp_path / "reference.png"
        assert cv2.imwrite(str(reference), scene)

        path_a = tmp_path / "a.mp4"
        path_b = tmp_path / "b.mp4"
        moved = FakeSource(
            [translate(scene, 40, 24).copy() for _ in range(10)],
            VideoMetadata(width=width, height=height, fps=FPS, frame_count=10),
        )
        config = VenueValidationConfig(
            manifest=manifest_for(
                VenueCamera(camera_id=CAM_A, sources=(camera_entry(path_a),)),
                VenueCamera(camera_id=CAM_B, sources=(camera_entry(path_b),)),
            ),
            output_dir=tmp_path / "artifacts",
            cameras=(
                VenueValidationCameraConfig(
                    camera_id=CAM_A,
                    calibration_path=write_calibration(
                        tmp_path, "cal_a.json", CAMERA_A, CAM_A, FRAME_A
                    ),
                    integrity_reference_path=reference,
                ),
                VenueValidationCameraConfig(
                    camera_id=CAM_B,
                    calibration_path=write_calibration(
                        tmp_path, "cal_b.json", CAMERA_B, CAM_B, FRAME_B
                    ),
                ),
            ),
        )
        detector = ShapeDetector(
            {
                (height, width): [detection_at(CAMERA_A, STAND)],
                (DIMS_B[1], DIMS_B[0]): [detection_at(CAMERA_B, STAND)],
            }
        )
        summary = run(config, {path_a: moved, path_b: source_for(*DIMS_B, 10)}, detector)

        assert summary.streams[0].integrity_monitored is True
        assert summary.association_outcomes.get(AssociationOutcome.ASSOCIATED, 0) == 0
        assert summary.association_outcomes.get(AssociationOutcome.NOT_COMPARABLE, 0) > 0
        assert summary.suppressed_duplicates == 0


class TestScheduling:
    """Venue time, stream identity and ordering come from V0.4.0, unchanged."""

    def test_differing_frame_rates_and_lengths_are_scheduled_not_resampled(
        self, tmp_path: Path
    ) -> None:
        path_a = tmp_path / "a.mp4"
        path_b = tmp_path / "b.mp4"
        config = VenueValidationConfig(
            manifest=manifest_for(
                VenueCamera(camera_id=CAM_A, sources=(camera_entry(path_a),)),
                VenueCamera(camera_id=CAM_B, sources=(camera_entry(path_b),)),
            ),
            output_dir=tmp_path / "artifacts",
        )
        sources: dict[Path, VideoSource] = {
            # 10 fps for 0.0-0.3 s, and 5 fps for 0.0-0.2 s. The 5 fps frames
            # land on venue instants the 10 fps stream also occupies.
            path_a: source_for(*DIMS_A, 4, fps=10.0),
            path_b: source_for(*DIMS_B, 2, fps=5.0),
        }
        summary = run(config, sources, standing_detector())

        assert summary.source_frames_processed == 6
        assert summary.venue_instants == 4
        assert summary.venue_span_seconds == pytest.approx(0.3)
        by_camera = {stream.stream.camera_id: stream for stream in summary.streams}
        assert by_camera[CAM_A].frames_processed == 4
        assert by_camera[CAM_A].fps == 10.0
        assert by_camera[CAM_B].frames_processed == 2
        assert by_camera[CAM_B].fps == 5.0

    def test_an_ended_stream_fabricates_nothing(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=8, frames_b=3)
        summary = run(config, sources, detector)

        assert summary.venue_instants == 8
        assert summary.source_frames_processed == 11
        assert summary.observations_emitted == 11
        # Once the short stream ends, the instants it does not reach carry one
        # camera's tracks only — the ended camera contributes nothing rather
        # than repeating its last frame.
        last = records(config.occupancy_report_path)[-1]
        assert last["raw_active_track_count"] == 1

    def test_a_zero_frame_camera_is_reported_not_fatal(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=5, frames_b=0)
        summary = run(config, sources, detector)

        assert summary.venue_instants == 5
        assert summary.contributing_camera_ids == (CAM_A,)
        assert [stream.camera_id for stream in summary.zero_frame_streams] == [CAM_B]
        assert summary.configured_camera_count == 2

    def test_stream_epochs_survive_into_the_summary(self, tmp_path: Path) -> None:
        first = tmp_path / "a0.mp4"
        second = tmp_path / "a1.mp4"
        config = VenueValidationConfig(
            manifest=manifest_for(
                VenueCamera(
                    camera_id=CAM_A,
                    sources=(
                        camera_entry(first, offset_seconds=0.0),
                        camera_entry(second, offset_seconds=1.0),
                    ),
                )
            ),
            output_dir=tmp_path / "artifacts",
        )
        sources: dict[Path, VideoSource] = {
            first: source_for(*DIMS_A, 3),
            second: source_for(*DIMS_A, 2),
        }
        summary = run(config, sources, standing_detector())

        assert [stream.stream.stream_epoch for stream in summary.streams] == [0, 1]
        assert summary.source_frames_processed == 5
        assert summary.venue_instants == 5

    def test_declared_offsets_establish_venue_time(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=6, frames_b=6)
        summary = run(config, sources, detector)

        assert summary.session_alignment.basis is TimeAlignmentBasis.OPERATOR_DECLARED
        assert summary.first_venue_seconds == pytest.approx(0.0)

    def test_undeclared_offsets_leave_venue_time_unestablished(self, tmp_path: Path) -> None:
        path_a = tmp_path / "a.mp4"
        path_b = tmp_path / "b.mp4"
        config = VenueValidationConfig(
            manifest=manifest_for(
                VenueCamera(
                    camera_id=CAM_A,
                    sources=(camera_entry(path_a, offset_seconds=None, accuracy_seconds=None),),
                ),
                VenueCamera(
                    camera_id=CAM_B,
                    sources=(camera_entry(path_b, offset_seconds=None, accuracy_seconds=None),),
                ),
            ),
            output_dir=tmp_path / "artifacts",
            cameras=(
                VenueValidationCameraConfig(
                    camera_id=CAM_A,
                    calibration_path=write_calibration(
                        tmp_path, "cal_a.json", CAMERA_A, CAM_A, FRAME_A
                    ),
                ),
                VenueValidationCameraConfig(
                    camera_id=CAM_B,
                    calibration_path=write_calibration(
                        tmp_path, "cal_b.json", CAMERA_B, CAM_B, FRAME_B
                    ),
                ),
            ),
        )
        sources: dict[Path, VideoSource] = {
            path_a: source_for(*DIMS_A, 8),
            path_b: source_for(*DIMS_B, 8),
        }
        summary = run(config, sources, standing_detector())

        # A shared tick is a scheduling unit, not evidence of simultaneity.
        assert summary.session_alignment.basis is TimeAlignmentBasis.NOT_ESTABLISHED
        assert summary.association_outcomes.get(AssociationOutcome.ASSOCIATED, 0) == 0
        assert summary.association_outcomes.get(AssociationOutcome.NOT_COMPARABLE, 0) > 0


class TestDeterminism:
    """Equivalent inputs, byte-equivalent artifacts."""

    def artifacts(self, config: VenueValidationConfig) -> dict[str, bytes]:
        return {
            "association": config.association_report_path.read_bytes(),
            "occupancy": config.occupancy_report_path.read_bytes(),
            "summary": config.summary_path.read_bytes(),
        }

    def test_a_replayed_run_is_byte_identical(self, tmp_path: Path) -> None:
        first_config, first_sources, first_detector = two_camera_setup(
            tmp_path, output_dir=tmp_path / "run_one"
        )
        run(first_config, first_sources, first_detector)
        second_config, second_sources, second_detector = two_camera_setup(
            tmp_path, output_dir=tmp_path / "run_two"
        )
        run(second_config, second_sources, second_detector)

        assert self.artifacts(first_config) == self.artifacts(second_config)

    def test_manifest_camera_order_does_not_change_the_artifacts(self, tmp_path: Path) -> None:
        forward_config, forward_sources, forward_detector = two_camera_setup(
            tmp_path, order=(CAM_A, CAM_B), output_dir=tmp_path / "forward"
        )
        run(forward_config, forward_sources, forward_detector)
        reverse_config, reverse_sources, reverse_detector = two_camera_setup(
            tmp_path, order=(CAM_B, CAM_A), output_dir=tmp_path / "reverse"
        )
        run(reverse_config, reverse_sources, reverse_detector)

        assert self.artifacts(forward_config) == self.artifacts(reverse_config)


class TestArtifacts:
    """Three files, each owned by the layer that defines its schema."""

    def test_the_association_report_carries_the_v0_4_5_schema(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=5, frames_b=5)
        summary = run(config, sources, detector)

        lines = records(config.association_report_path)
        assert lines[0]["schema_version"] == "0.4.5"
        assert lines[0]["session_id"] == config.session_id
        assert len(lines) - 1 == summary.venue_instants

    def test_the_occupancy_report_carries_the_v0_4_6_schema(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=5, frames_b=5)
        summary = run(config, sources, detector)

        lines = records(config.occupancy_report_path)
        assert lines[0]["schema_version"] == "0.4.6"
        assert len(lines) - 1 == summary.occupancy_snapshots

    def test_the_summary_counters_match_the_emitted_records(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=7, frames_b=4)
        summary = run(config, sources, detector)

        document = json.loads(config.summary_path.read_text(encoding="utf-8"))
        assert document == summary_record(summary)
        assert document["venue_instants"] == len(records(config.association_report_path)) - 1
        assert document["occupancy"]["snapshots"] == len(records(config.occupancy_report_path)) - 1
        assert document["source_frames_processed"] == 11
        assert (
            sum(stream["frames_processed"] for stream in document["streams"])
            == document["source_frames_processed"]
        )

    def test_the_summary_claims_no_ground_truth(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=4, frames_b=4)
        run(config, sources, detector)

        document = json.loads(config.summary_path.read_text(encoding="utf-8"))
        assert document["ground_truth"] == GROUND_TRUTH_STATUS == "NOT_ANNOTATED"
        assert "accuracy" in document["note"]
        # Execution statistics only: past the disclaimer, no accuracy figure may
        # appear anywhere in the record.
        del document["note"]
        flattened = json.dumps(document)
        for forbidden in ("precision", "recall", "false_merge", "accuracy", "id_switch"):
            assert forbidden not in flattened

    def test_the_summary_names_artifacts_without_absolute_paths(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=3, frames_b=3)
        run(config, sources, detector)

        document = json.loads(config.summary_path.read_text(encoding="utf-8"))
        assert document["artifacts"]["association_report"] == "venue_pipeline_report.jsonl"
        assert str(tmp_path) not in json.dumps(document)


class TestLifecycle:
    """Sources and writers are released on both paths."""

    def test_every_source_is_released_on_success(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=3, frames_b=3)
        run(config, sources, detector)

        assert all(source.released for source in sources.values())  # type: ignore[attr-defined]

    def test_a_failed_run_releases_its_sources_and_writes_no_summary(self, tmp_path: Path) -> None:
        path_a = tmp_path / "a.mp4"
        path_b = tmp_path / "b.mp4"
        config = VenueValidationConfig(
            manifest=manifest_for(
                VenueCamera(camera_id=CAM_A, sources=(camera_entry(path_a),)),
                VenueCamera(camera_id=CAM_B, sources=(camera_entry(path_b),)),
            ),
            output_dir=tmp_path / "artifacts",
        )
        healthy = source_for(*DIMS_B, 6)
        broken = ExplodingSource(
            VideoMetadata(width=DIMS_A[0], height=DIMS_A[1], fps=FPS, frame_count=6), good_frames=2
        )
        sources: dict[Path, VideoSource] = {path_a: broken, path_b: healthy}

        with pytest.raises(VideoSourceError):
            run(config, sources, standing_detector())

        assert broken.released
        assert healthy.released
        # A failure is not an empty successful run: no summary is written at all.
        assert not config.summary_path.exists()


class TestAdapterSeam:
    """The runner builds observations one way, and only one way."""

    def test_observations_carry_the_camera_local_subject_identity(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=4, frames_b=4)
        run(config, sources, detector)

        instants = records(config.association_report_path)[1:]
        observations = [
            observation
            for instant in instants
            for observation in instant["association"]["observations"]  # type: ignore[index]
        ]
        assert {observation["camera_id"] for observation in observations} == {CAM_A, CAM_B}
        # Camera-local track identity stays camera-scoped, and every observation
        # carries the stream it came from and the venue instant it was valid at.
        assert all(observation["stream_epoch"] == 0 for observation in observations)
        assert all(observation["space"] == "ground_plane_metres" for observation in observations)
        assert {observation["venue_seconds"] for observation in observations} == {
            instant["venue_seconds"] for instant in instants
        }

    def test_the_runner_never_constructs_an_observation_itself(self) -> None:
        source = Path("src/atreides_vision/venue_validation/runner.py")
        text = (Path(__file__).resolve().parents[1] / source).read_text(encoding="utf-8")
        assert "build_track_observation(" in text
        assert "TrackObservation(" not in text

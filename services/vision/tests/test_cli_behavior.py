"""CLI exposure of behaviour output and the calibration commands.

No model weights are loaded: the tests either parse arguments, or exercise
handlers whose failure happens before the detector is constructed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.behavior.coordinates import MotionSpace
from atreides_vision.calibration.fitting import fit_calibration
from atreides_vision.cli import (
    EXIT_ERROR,
    EXIT_OK,
    _build_behavior_analyzer,
    _build_run_metadata,
    build_parser,
    main,
)
from atreides_vision.video.source import VideoMetadata
from synthetic_cameras import CAMERA_A, FRAME_A, VENUE_ID, calibration_for

CALIBRATION_REGION = [(0.0, 0.0), (12.0, 0.0), (12.0, 9.0), (0.0, 9.0)]
METADATA = VideoMetadata(width=1920, height=1080, fps=25.0, frame_count=100)


def write_calibration(tmp_path: Path, camera_id: str = "cam_01", name: str = "cal.json") -> Path:
    path = tmp_path / name
    calibration = calibration_for(
        CAMERA_A, CALIBRATION_REGION, camera_id=camera_id, reference_frame=FRAME_A
    )
    path.write_text(json.dumps(calibration.to_record()), encoding="utf-8")
    return path


def read_image(path: Path) -> np.ndarray:
    """Read an image, failing the test rather than returning ``None``."""
    image = cv2.imread(str(path))
    assert image is not None, f"could not read {path}"
    return np.asarray(image)


def write_frame(tmp_path: Path, width: int = 1920, height: int = 1080) -> Path:
    path = tmp_path / "frame.png"
    frame = np.full((height, width, 3), 50, dtype=np.uint8)
    assert cv2.imwrite(str(path), frame)
    return path


class TestRunFlags:
    def test_behavior_output_defaults_to_off(self) -> None:
        args = build_parser().parse_args(["run", "--input", "in.mp4", "--output", "out.mp4"])
        assert args.behavior_output is None
        assert args.calibration is None
        assert args.camera_id is None

    def test_behavior_output_is_accepted(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--input",
                "in.mp4",
                "--output",
                "out.mp4",
                "--behavior-output",
                "artifacts/behavior.jsonl",
            ]
        )
        assert args.behavior_output == Path("artifacts/behavior.jsonl")

    def test_calibration_and_camera_id_are_accepted(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--input",
                "in.mp4",
                "--output",
                "out.mp4",
                "--behavior-output",
                "b.jsonl",
                "--calibration",
                "cal.json",
                "--camera-id",
                "cam_07",
            ]
        )
        assert args.calibration == Path("cal.json")
        assert args.camera_id == "cam_07"

    def test_run_help_lists_the_new_flags(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "--help"])
        out = capsys.readouterr().out
        for flag in ("--behavior-output", "--calibration", "--camera-id"):
            assert flag in out

    def test_calibration_without_behavior_output_is_rejected(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """It would otherwise be silently inert."""
        calibration = write_calibration(tmp_path)
        code = main(
            [
                "run",
                "--input",
                str(tmp_path / "in.mp4"),
                "--output",
                str(tmp_path / "out.mp4"),
                "--calibration",
                str(calibration),
            ]
        )
        assert code == EXIT_ERROR
        assert "--behavior-output" in caplog.text


class TestBehaviorFailsBeforeModelLoad:
    """Configuration problems must surface before weights are touched."""

    def test_missing_calibration_file_reports_clearly(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        code = main(
            [
                "run",
                "--input",
                str(tmp_path / "in.mp4"),
                "--output",
                str(tmp_path / "out.mp4"),
                "--behavior-output",
                str(tmp_path / "b.jsonl"),
                "--calibration",
                str(tmp_path / "absent.json"),
            ]
        )
        assert code == EXIT_ERROR
        assert "Calibration file not found" in caplog.text

    def test_malformed_calibration_reports_clearly(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        code = main(
            [
                "run",
                "--input",
                str(tmp_path / "in.mp4"),
                "--output",
                str(tmp_path / "out.mp4"),
                "--behavior-output",
                str(tmp_path / "b.jsonl"),
                "--calibration",
                str(broken),
            ]
        )
        assert code == EXIT_ERROR
        assert "not valid JSON" in caplog.text

    def test_camera_id_conflict_reports_clearly(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        calibration = write_calibration(tmp_path, camera_id="cam_01")
        code = main(
            [
                "run",
                "--input",
                str(tmp_path / "in.mp4"),
                "--output",
                str(tmp_path / "out.mp4"),
                "--behavior-output",
                str(tmp_path / "b.jsonl"),
                "--calibration",
                str(calibration),
                "--camera-id",
                "cam_02",
            ]
        )
        assert code == EXIT_ERROR
        assert "Conflicting camera ids" in caplog.text

    def test_missing_camera_id_reports_clearly(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Behaviour with no calibration and no zones has no camera namespace."""
        code = main(
            [
                "run",
                "--input",
                str(tmp_path / "in.mp4"),
                "--output",
                str(tmp_path / "out.mp4"),
                "--behavior-output",
                str(tmp_path / "b.jsonl"),
            ]
        )
        assert code == EXIT_ERROR
        assert "needs a camera id" in caplog.text

    def test_calibration_is_validated_before_the_video_is_opened(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The input path does not even exist, and the calibration error wins."""
        broken = tmp_path / "broken.json"
        broken.write_text("{}", encoding="utf-8")
        code = main(
            [
                "run",
                "--input",
                str(tmp_path / "nonexistent.mp4"),
                "--output",
                str(tmp_path / "out.mp4"),
                "--behavior-output",
                str(tmp_path / "b.jsonl"),
                "--calibration",
                str(broken),
            ]
        )
        assert code == EXIT_ERROR
        assert "missing required field" in caplog.text
        assert "Input video not found" not in caplog.text


class TestProjectorSelection:
    def test_no_calibration_selects_normalized(self) -> None:
        analyzer = _build_behavior_analyzer("cam_a", METADATA, None)
        assert analyzer.space is MotionSpace.NORMALIZED_IMAGE
        assert analyzer.config.intended_space is None

    def test_calibration_selects_ground_plane(self) -> None:
        fit = fit_calibration(calibration_for(CAMERA_A, CALIBRATION_REGION, camera_id="cam_a"))
        analyzer = _build_behavior_analyzer("cam_a", METADATA, fit)
        assert analyzer.space is MotionSpace.GROUND_PLANE_METRES

    def test_metric_mode_uses_metric_thresholds(self) -> None:
        """Normalized thresholds in metres would mean nobody is ever stationary."""
        fit = fit_calibration(calibration_for(CAMERA_A, CALIBRATION_REGION, camera_id="cam_a"))
        analyzer = _build_behavior_analyzer("cam_a", METADATA, fit)
        expected = BehaviorAnalysisConfig.for_ground_plane_metres()
        assert analyzer.config.stationary_speed_threshold == expected.stationary_speed_threshold
        assert analyzer.config.intended_space is MotionSpace.GROUND_PLANE_METRES

    def test_normalized_mode_keeps_v030_defaults(self) -> None:
        analyzer = _build_behavior_analyzer("cam_a", METADATA, None)
        assert (
            analyzer.config.stationary_speed_threshold
            == BehaviorAnalysisConfig().stationary_speed_threshold
        )

    def test_the_projector_follows_the_runtime_resolution(self) -> None:
        fit = fit_calibration(
            calibration_for(
                CAMERA_A, CALIBRATION_REGION, camera_id="cam_a", reference_frame=FRAME_A
            )
        )
        half = VideoMetadata(width=960, height=540, fps=25.0, frame_count=10)
        analyzer = _build_behavior_analyzer("cam_a", half, fit)
        assert analyzer.space is MotionSpace.GROUND_PLANE_METRES


class TestRunMetadataAssembly:
    def test_normalized_metadata_has_no_calibration(self) -> None:
        analyzer = _build_behavior_analyzer("cam_a", METADATA, None)
        metadata = _build_run_metadata("cam_a", analyzer, Path("in.mp4"), METADATA, None)
        assert metadata.calibration is None
        assert metadata.space is MotionSpace.NORMALIZED_IMAGE

    def test_metric_metadata_carries_calibration_provenance(self) -> None:
        fit = fit_calibration(
            calibration_for(CAMERA_A, CALIBRATION_REGION, camera_id="cam_a", venue_id=VENUE_ID)
        )
        analyzer = _build_behavior_analyzer("cam_a", METADATA, fit)
        metadata = _build_run_metadata("cam_a", analyzer, Path("in.mp4"), METADATA, fit)

        assert metadata.calibration is not None
        assert metadata.calibration.venue_id == VENUE_ID
        assert metadata.calibration.correspondence_count == 4
        assert metadata.calibration.reference_width == FRAME_A.width

    def test_source_provenance_is_recorded(self) -> None:
        analyzer = _build_behavior_analyzer("cam_a", METADATA, None)
        metadata = _build_run_metadata("cam_a", analyzer, Path("clip.mp4"), METADATA, None)
        assert metadata.source is not None
        assert metadata.source.path == "clip.mp4"
        assert metadata.source.width == 1920
        assert metadata.source.fps == pytest.approx(25.0)


class TestCalibrationValidateCommand:
    def test_reports_the_fit(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        calibration = write_calibration(tmp_path, camera_id="cam_01")
        with caplog.at_level(logging.INFO):
            assert main(["calibration", "validate", "--calibration", str(calibration)]) == EXIT_OK

        text = caplog.text
        assert "cam_01" in text
        assert VENUE_ID in text
        assert "ground" in text
        assert "1920x1080" in text
        assert "Homography:  solved" in text
        assert "Reprojection error" in text

    def test_states_that_error_is_not_accuracy(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No semantic verdict, and no implied real-world validation."""
        calibration = write_calibration(tmp_path)
        with caplog.at_level(logging.INFO):
            main(["calibration", "validate", "--calibration", str(calibration)])

        text = caplog.text
        assert "not a measurement of real-world accuracy" in text
        for verdict in ("GOOD", "BAD", "SAFE", "PRODUCTION READY", "PASS", "FAIL"):
            assert verdict not in text

    def test_warns_that_four_points_always_fit(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        calibration = write_calibration(tmp_path)
        with caplog.at_level(logging.INFO):
            main(["calibration", "validate", "--calibration", str(calibration)])
        assert "says nothing about whether the points were measured" in caplog.text

    def test_missing_file_exits_with_an_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        code = main(["calibration", "validate", "--calibration", str(tmp_path / "absent.json")])
        assert code == EXIT_ERROR
        assert "Calibration file not found" in caplog.text

    def test_degenerate_calibration_exits_with_an_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "collinear.json"
        path.write_text(
            json.dumps(
                {
                    "camera_id": "cam_01",
                    "venue_id": "v",
                    "reference_frame": {"width": 1920, "height": 1080},
                    "correspondences": [
                        {"image": [100.0, 900.0], "venue": [0.0, 0.0]},
                        {"image": [500.0, 900.0], "venue": [3.0, 0.0]},
                        {"image": [900.0, 900.0], "venue": [6.0, 0.0]},
                        {"image": [1300.0, 900.0], "venue": [9.0, 0.0]},
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert main(["calibration", "validate", "--calibration", str(path)]) == EXIT_ERROR
        assert "straight line" in caplog.text

    def test_requires_the_calibration_argument(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["calibration", "validate"])

    def test_requires_a_subcommand(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["calibration"])


class TestCalibrationRenderCommand:
    def test_writes_an_annotated_image(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        calibration = write_calibration(tmp_path)
        frame = write_frame(tmp_path)
        output = tmp_path / "diag.png"

        with caplog.at_level(logging.INFO):
            code = main(
                [
                    "calibration",
                    "render",
                    "--calibration",
                    str(calibration),
                    "--input",
                    str(frame),
                    "--output",
                    str(output),
                ]
            )
        assert code == EXIT_OK
        assert output.is_file()

        rendered = read_image(output)
        assert rendered.shape == (1080, 1920, 3)
        assert "Written" in caplog.text

    def test_the_rendering_actually_draws_something(self, tmp_path: Path) -> None:
        calibration = write_calibration(tmp_path)
        frame = write_frame(tmp_path)
        output = tmp_path / "diag.png"
        main(
            [
                "calibration",
                "render",
                "--calibration",
                str(calibration),
                "--input",
                str(frame),
                "--output",
                str(output),
            ]
        )
        assert not np.array_equal(read_image(frame), read_image(output))

    def test_works_at_a_different_runtime_resolution(self, tmp_path: Path) -> None:
        """The calibration was authored at 1920x1080; the still is half that."""
        calibration = write_calibration(tmp_path)
        frame = write_frame(tmp_path, width=960, height=540)
        output = tmp_path / "diag.png"

        assert (
            main(
                [
                    "calibration",
                    "render",
                    "--calibration",
                    str(calibration),
                    "--input",
                    str(frame),
                    "--output",
                    str(output),
                ]
            )
            == EXIT_OK
        )
        assert read_image(output).shape == (540, 960, 3)

    def test_grid_spacing_is_configurable(self, tmp_path: Path) -> None:
        calibration = write_calibration(tmp_path)
        frame = write_frame(tmp_path)

        def render(spacing: str, name: str) -> np.ndarray:
            output = tmp_path / name
            main(
                [
                    "calibration",
                    "render",
                    "--calibration",
                    str(calibration),
                    "--input",
                    str(frame),
                    "--output",
                    str(output),
                    "--grid-spacing",
                    spacing,
                ]
            )
            return read_image(output)

        assert not np.array_equal(render("1.0", "fine.png"), render("3.0", "coarse.png"))

    def test_grid_can_be_omitted(self, tmp_path: Path) -> None:
        calibration = write_calibration(tmp_path)
        frame = write_frame(tmp_path)

        def render(args: list[str], name: str) -> np.ndarray:
            output = tmp_path / name
            main(
                [
                    "calibration",
                    "render",
                    "--calibration",
                    str(calibration),
                    "--input",
                    str(frame),
                    "--output",
                    str(output),
                    *args,
                ]
            )
            return read_image(output)

        assert not np.array_equal(render([], "grid.png"), render(["--no-grid"], "plain.png"))

    def test_missing_input_frame_exits_with_an_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        calibration = write_calibration(tmp_path)
        code = main(
            [
                "calibration",
                "render",
                "--calibration",
                str(calibration),
                "--input",
                str(tmp_path / "absent.png"),
                "--output",
                str(tmp_path / "diag.png"),
            ]
        )
        assert code == EXIT_ERROR
        assert "Reference frame not found" in caplog.text

    def test_unreadable_input_exits_with_an_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        calibration = write_calibration(tmp_path)
        junk = tmp_path / "junk.png"
        junk.write_bytes(b"not an image at all")
        code = main(
            [
                "calibration",
                "render",
                "--calibration",
                str(calibration),
                "--input",
                str(junk),
                "--output",
                str(tmp_path / "diag.png"),
            ]
        )
        assert code == EXIT_ERROR
        # Whichever decoder path rejects it, the failure names the file and is
        # explicit — never a silently blank diagnostic image.
        assert "junk.png" in caplog.text
        assert not (tmp_path / "diag.png").exists()

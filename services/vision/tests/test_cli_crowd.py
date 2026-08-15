"""CLI exposure of spatial-cluster measurement.

No model weights are loaded: these parse arguments, or exercise builders and
failure paths that resolve before the detector is constructed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from atreides_vision.behavior.analyzer import BehaviorAnalyzer
from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.behavior.coordinates import MotionSpace, NormalizedFrameProjector
from atreides_vision.calibration.fitting import CalibrationFit, fit_calibration
from atreides_vision.calibration.projector import GroundPlaneProjector
from atreides_vision.cli import (
    EXIT_ERROR,
    _build_crowd_analyzer,
    _build_crowd_metadata,
    _log_crowd_startup,
    build_parser,
    main,
)
from atreides_vision.crowd.config import CrowdAnalysisConfig
from synthetic_cameras import CAMERA_A, FRAME_A, VENUE_ID, calibration_for

CALIBRATION_REGION = [(0.0, 0.0), (12.0, 0.0), (12.0, 9.0), (0.0, 9.0)]
W, H = FRAME_A.width, FRAME_A.height


def write_calibration(tmp_path: Path, camera_id: str = "cam_01") -> Path:
    path = tmp_path / "cal.json"
    calibration = calibration_for(
        CAMERA_A, CALIBRATION_REGION, camera_id=camera_id, reference_frame=FRAME_A
    )
    path.write_text(json.dumps(calibration.to_record()), encoding="utf-8")
    return path


def behavior_analyzer(calibrated: bool) -> BehaviorAnalyzer:
    if not calibrated:
        return BehaviorAnalyzer("cam_01", NormalizedFrameProjector(W, H), BehaviorAnalysisConfig())
    return BehaviorAnalyzer(
        "cam_01",
        GroundPlaneProjector(calibration_fit(), W, H),
        BehaviorAnalysisConfig.for_ground_plane_metres(),
    )


def calibration_fit() -> CalibrationFit:
    return fit_calibration(
        calibration_for(CAMERA_A, CALIBRATION_REGION, camera_id="cam_01", reference_frame=FRAME_A)
    )


def run_args(*extra: str) -> list[str]:
    return ["run", "--input", "i.mp4", "--output", "o.mp4", *extra]


class TestFlags:
    def test_crowd_output_defaults_to_off(self) -> None:
        args = build_parser().parse_args(run_args())
        assert args.crowd_output is None

    def test_it_parses(self) -> None:
        args = build_parser().parse_args(run_args("--crowd-output", "a/crowd.jsonl"))
        assert args.crowd_output == Path("a/crowd.jsonl")

    def test_run_help_lists_it(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "--help"])
        assert "--crowd-output" in capsys.readouterr().out

    def test_it_does_not_require_behaviour_output(self) -> None:
        """Nobody should have to produce one file to get another."""
        args = build_parser().parse_args(run_args("--crowd-output", "c.jsonl"))
        assert args.behavior_output is None
        assert args.crowd_output is not None

    def test_it_does_not_require_interaction_output(self) -> None:
        """Clusters and pairs are separate questions and separate streams."""
        args = build_parser().parse_args(run_args("--crowd-output", "c.jsonl"))
        assert args.interaction_output is None

    def test_all_three_streams_can_be_requested_together(self) -> None:
        args = build_parser().parse_args(
            run_args(
                "--behavior-output",
                "b.jsonl",
                "--interaction-output",
                "i.jsonl",
                "--crowd-output",
                "c.jsonl",
            )
        )
        assert args.behavior_output == Path("b.jsonl")
        assert args.interaction_output == Path("i.jsonl")
        assert args.crowd_output == Path("c.jsonl")


class TestValidation:
    def test_calibration_is_satisfied_by_crowd_output_alone(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """--calibration needs *a* measurement stream, not specifically behaviour."""
        code = main(
            [
                "run",
                "--input",
                str(tmp_path / "absent.mp4"),
                "--output",
                str(tmp_path / "out.mp4"),
                "--crowd-output",
                str(tmp_path / "c.jsonl"),
                "--calibration",
                str(write_calibration(tmp_path)),
            ]
        )
        assert code == EXIT_ERROR
        # It got past the flag check and failed on the missing video instead.
        assert "--behavior-output, --interaction-output or --crowd-output" not in caplog.text
        assert "Input video not found" in caplog.text

    def test_crowd_output_needs_a_camera_id(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Cluster records are namespaced by camera exactly as observations are."""
        code = main(
            [
                "run",
                "--input",
                str(tmp_path / "in.mp4"),
                "--output",
                str(tmp_path / "out.mp4"),
                "--crowd-output",
                str(tmp_path / "c.jsonl"),
            ]
        )
        assert code == EXIT_ERROR
        assert "need a camera id" in caplog.text


class TestAnalyzerBuilder:
    def test_normalized_context_when_uncalibrated(self) -> None:
        analyzer = _build_crowd_analyzer("cam_01", behavior_analyzer(False), None)
        assert analyzer.space is MotionSpace.NORMALIZED_IMAGE
        assert analyzer.context.venue_id is None
        assert (
            analyzer.config.cluster_radius
            == CrowdAnalysisConfig.for_normalized_image().cluster_radius
        )

    def test_metric_context_when_calibrated(self) -> None:
        analyzer = _build_crowd_analyzer("cam_01", behavior_analyzer(True), calibration_fit())
        assert analyzer.space is MotionSpace.GROUND_PLANE_METRES
        assert analyzer.context.venue_id == VENUE_ID
        assert analyzer.context.surface_id == "ground"

    def test_metric_radii_are_used_when_calibrated(self) -> None:
        """A normalized radius read as metres would connect people 12 cm apart."""
        analyzer = _build_crowd_analyzer("cam_01", behavior_analyzer(True), calibration_fit())
        expected = CrowdAnalysisConfig.for_ground_plane_metres()
        assert analyzer.config.cluster_radius == expected.cluster_radius
        assert analyzer.config.density_radius == expected.density_radius
        assert analyzer.config.intended_space is MotionSpace.GROUND_PLANE_METRES

    def test_the_context_follows_the_behaviour_layer(self) -> None:
        """The two layers cannot disagree about which space they measure in."""
        behavior = behavior_analyzer(True)
        analyzer = _build_crowd_analyzer("cam_01", behavior, calibration_fit())
        assert analyzer.space is behavior.space

    def test_the_camera_id_is_carried_through(self) -> None:
        analyzer = _build_crowd_analyzer("north_door", behavior_analyzer(False), None)
        assert analyzer.camera_id == "north_door"


class TestMetadataBuilder:
    def test_normalized_metadata(self) -> None:
        analyzer = _build_crowd_analyzer("cam_01", behavior_analyzer(False), None)
        metadata = _build_crowd_metadata(analyzer)
        assert metadata.camera_id == "cam_01"
        assert metadata.space is MotionSpace.NORMALIZED_IMAGE
        assert metadata.venue_id is None
        assert metadata.density_unit == "subjects/normalized_area_unit"

    def test_metric_metadata_carries_venue_and_radii(self) -> None:
        analyzer = _build_crowd_analyzer("cam_01", behavior_analyzer(True), calibration_fit())
        metadata = _build_crowd_metadata(analyzer)
        assert metadata.venue_id == VENUE_ID
        assert metadata.surface_id == "ground"
        assert metadata.cluster_radius == pytest.approx(1.5)
        assert metadata.density_radius == pytest.approx(2.0)
        assert metadata.minimum_cluster_size == 3
        assert metadata.density_unit == "subjects/m^2"

    def test_the_header_settings_match_the_analyzer(self) -> None:
        """A reader must be able to trust the header describes this run."""
        analyzer = _build_crowd_analyzer("cam_01", behavior_analyzer(True), calibration_fit())
        metadata = _build_crowd_metadata(analyzer)
        assert metadata.cluster_radius == analyzer.config.cluster_radius
        assert metadata.density_radius == analyzer.config.density_radius
        assert metadata.minimum_cluster_size == analyzer.config.minimum_cluster_size


class TestStartupLogging:
    def test_disabled_says_how_to_enable_it(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            _log_crowd_startup(None, None)
        assert "--crowd-output" in caplog.text

    def test_it_reports_the_units_it_will_measure_in(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        analyzer = _build_crowd_analyzer("cam_01", behavior_analyzer(True), calibration_fit())
        with caplog.at_level(logging.INFO):
            _log_crowd_startup(analyzer, Path("c.jsonl"))
        assert "1.50 m" in caplog.text
        assert "same-camera only" in caplog.text

    def test_it_states_that_a_cluster_is_not_a_group(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The operator reads this line before they read the output."""
        analyzer = _build_crowd_analyzer("cam_01", behavior_analyzer(False), None)
        with caplog.at_level(logging.INFO):
            _log_crowd_startup(analyzer, Path("c.jsonl"))
        assert "not a social group" in caplog.text

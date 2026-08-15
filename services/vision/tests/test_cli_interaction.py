"""CLI exposure of interaction measurement.

No model weights are loaded: these parse arguments, or exercise builders and
failure paths that resolve before the detector is constructed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atreides_vision.behavior.analyzer import BehaviorAnalyzer
from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.behavior.coordinates import MotionSpace, NormalizedFrameProjector
from atreides_vision.calibration.fitting import fit_calibration
from atreides_vision.calibration.projector import GroundPlaneProjector
from atreides_vision.cli import (
    EXIT_ERROR,
    _build_interaction_analyzer,
    _build_interaction_metadata,
    build_parser,
    main,
)
from atreides_vision.interaction.config import InteractionAnalysisConfig
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
    fit = fit_calibration(
        calibration_for(CAMERA_A, CALIBRATION_REGION, camera_id="cam_01", reference_frame=FRAME_A)
    )
    return BehaviorAnalyzer(
        "cam_01",
        GroundPlaneProjector(fit, W, H),
        BehaviorAnalysisConfig.for_ground_plane_metres(),
    )


def calibration_fit() -> object:
    return fit_calibration(
        calibration_for(CAMERA_A, CALIBRATION_REGION, camera_id="cam_01", reference_frame=FRAME_A)
    )


class TestFlags:
    def test_interaction_output_defaults_to_off(self) -> None:
        args = build_parser().parse_args(["run", "--input", "i.mp4", "--output", "o.mp4"])
        assert args.interaction_output is None

    def test_it_parses(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--input",
                "i.mp4",
                "--output",
                "o.mp4",
                "--interaction-output",
                "artifacts/interactions.jsonl",
            ]
        )
        assert args.interaction_output == Path("artifacts/interactions.jsonl")

    def test_run_help_lists_it(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "--help"])
        assert "--interaction-output" in capsys.readouterr().out

    def test_it_does_not_require_behaviour_output(self) -> None:
        """Nobody should have to produce one file to get another."""
        args = build_parser().parse_args(
            [
                "run",
                "--input",
                "i.mp4",
                "--output",
                "o.mp4",
                "--interaction-output",
                "x.jsonl",
            ]
        )
        assert args.behavior_output is None
        assert args.interaction_output is not None

    def test_calibration_is_satisfied_by_interaction_output_alone(
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
                "--interaction-output",
                str(tmp_path / "i.jsonl"),
                "--calibration",
                str(write_calibration(tmp_path)),
            ]
        )
        assert code == EXIT_ERROR
        # It got past the flag check and failed on the missing video instead.
        assert "--behavior-output, --interaction-output or --crowd-output" not in caplog.text
        assert "Input video not found" in caplog.text

    def test_calibration_without_any_measurement_stream_is_rejected(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        code = main(
            [
                "run",
                "--input",
                str(tmp_path / "in.mp4"),
                "--output",
                str(tmp_path / "out.mp4"),
                "--calibration",
                str(write_calibration(tmp_path)),
            ]
        )
        assert code == EXIT_ERROR
        assert "--behavior-output, --interaction-output or --crowd-output" in caplog.text

    def test_interaction_output_needs_a_camera_id(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Pair records are namespaced by camera exactly as observations are."""
        code = main(
            [
                "run",
                "--input",
                str(tmp_path / "in.mp4"),
                "--output",
                str(tmp_path / "out.mp4"),
                "--interaction-output",
                str(tmp_path / "i.jsonl"),
            ]
        )
        assert code == EXIT_ERROR
        assert "need a camera id" in caplog.text


class TestAnalyzerBuilder:
    def test_normalized_context_when_uncalibrated(self) -> None:
        analyzer = _build_interaction_analyzer("cam_01", behavior_analyzer(False), None)
        assert analyzer.space is MotionSpace.NORMALIZED_IMAGE
        assert analyzer.context.venue_id is None
        assert (
            analyzer.config.proximity_threshold
            == InteractionAnalysisConfig.for_normalized_image().proximity_threshold
        )

    def test_metric_context_when_calibrated(self) -> None:
        fit = calibration_fit()
        analyzer = _build_interaction_analyzer("cam_01", behavior_analyzer(True), fit)  # type: ignore[arg-type]
        assert analyzer.space is MotionSpace.GROUND_PLANE_METRES
        assert analyzer.context.venue_id == VENUE_ID
        assert analyzer.context.surface_id == "ground"

    def test_metric_thresholds_are_used_when_calibrated(self) -> None:
        """Normalized thresholds in metres would put proximity 15 cm away."""
        fit = calibration_fit()
        analyzer = _build_interaction_analyzer("cam_01", behavior_analyzer(True), fit)  # type: ignore[arg-type]
        expected = InteractionAnalysisConfig.for_ground_plane_metres()
        assert analyzer.config.proximity_threshold == expected.proximity_threshold
        assert analyzer.config.intended_space is MotionSpace.GROUND_PLANE_METRES

    def test_the_context_follows_the_behaviour_layer(self) -> None:
        """The two layers cannot disagree about which space they measure in."""
        behavior = behavior_analyzer(True)
        analyzer = _build_interaction_analyzer("cam_01", behavior, calibration_fit())  # type: ignore[arg-type]
        assert analyzer.space is behavior.space


class TestMetadataBuilder:
    def test_normalized_metadata(self) -> None:
        analyzer = _build_interaction_analyzer("cam_01", behavior_analyzer(False), None)
        metadata = _build_interaction_metadata(analyzer)
        assert metadata.camera_id == "cam_01"
        assert metadata.space is MotionSpace.NORMALIZED_IMAGE
        assert metadata.venue_id is None

    def test_metric_metadata_carries_venue_and_threshold(self) -> None:
        fit = calibration_fit()
        analyzer = _build_interaction_analyzer("cam_01", behavior_analyzer(True), fit)  # type: ignore[arg-type]
        metadata = _build_interaction_metadata(analyzer)
        assert metadata.venue_id == VENUE_ID
        assert metadata.surface_id == "ground"
        assert metadata.proximity_threshold == pytest.approx(1.5)

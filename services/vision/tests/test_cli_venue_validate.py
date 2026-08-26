"""CLI exposure of the V0.4.8 end-to-end venue validation runner.

The command itself loads detector weights, so these tests stop at the seam that
does not: argument parsing and the pure translation from parsed arguments to a
``VenueValidationConfig``. Nothing here downloads a model or opens a video.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from atreides_vision.cli import (
    _camera_path_arguments,
    _venue_camera_configs,
    build_parser,
)
from atreides_vision.errors import ConfigurationError


def parse(*arguments: str) -> argparse.Namespace:
    return build_parser().parse_args(["venue", "validate", *arguments])


def base_arguments(tmp_path: Path) -> list[str]:
    return [
        "--manifest",
        str(tmp_path / "session.json"),
        "--output-dir",
        str(tmp_path / "artifacts"),
    ]


class TestArgumentParsing:
    def test_the_command_routes_to_the_validation_handler(self, tmp_path: Path) -> None:
        args = parse(*base_arguments(tmp_path))
        assert args.command == "venue"
        assert args.venue_command == "validate"
        assert args.handler.__name__ == "_venue_validate_command"

    def test_the_manifest_and_output_directory_are_required(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            parse("--output-dir", str(tmp_path))
        with pytest.raises(SystemExit):
            parse("--manifest", str(tmp_path / "session.json"))

    def test_paths_arrive_as_paths(self, tmp_path: Path) -> None:
        args = parse(*base_arguments(tmp_path))
        assert args.manifest == tmp_path / "session.json"
        assert args.output_dir == tmp_path / "artifacts"

    def test_calibration_flags_accumulate_over_arbitrary_camera_counts(
        self, tmp_path: Path
    ) -> None:
        """No flag caps the camera count, and none names a camera in advance."""
        args = parse(
            *base_arguments(tmp_path),
            "--calibration",
            "cam_a=a.json",
            "--calibration",
            "cam_b=b.json",
            "--calibration",
            "cam_c=c.json",
        )
        assert args.calibration == ["cam_a=a.json", "cam_b=b.json", "cam_c=c.json"]

    def test_guarded_integrity_is_the_default(self, tmp_path: Path) -> None:
        assert parse(*base_arguments(tmp_path)).integrity_report_only is False

    def test_a_run_needs_no_calibration_at_all(self, tmp_path: Path) -> None:
        args = parse(*base_arguments(tmp_path))
        assert args.calibration == []
        assert args.calibration_reference == []
        assert _venue_camera_configs(args) == ()


class TestCameraPathArguments:
    def test_a_well_formed_entry_parses(self) -> None:
        parsed = _camera_path_arguments(["cam_a=/tmp/a.json"], "--calibration")
        assert parsed == {"cam_a": Path("/tmp/a.json")}

    @pytest.mark.parametrize("value", ["cam_a", "=/tmp/a.json", "cam_a=", "   =  "])
    def test_a_malformed_entry_is_refused(self, value: str) -> None:
        with pytest.raises(ConfigurationError, match="CAMERA=PATH"):
            _camera_path_arguments([value], "--calibration")

    def test_a_repeated_camera_is_refused(self) -> None:
        """Silently keeping the last path would hide a manifest mistake."""
        with pytest.raises(ConfigurationError, match="more than once"):
            _camera_path_arguments(["cam_a=a.json", "cam_a=b.json"], "--calibration")


class TestCameraConfiguration:
    def test_cameras_come_back_in_canonical_order(self, tmp_path: Path) -> None:
        """Flag order must not reach the runner: artifacts are order-independent."""
        forwards = parse(
            *base_arguments(tmp_path),
            "--calibration",
            "cam_b=b.json",
            "--calibration",
            "cam_a=a.json",
        )
        backwards = parse(
            *base_arguments(tmp_path),
            "--calibration",
            "cam_a=a.json",
            "--calibration",
            "cam_b=b.json",
        )
        assert _venue_camera_configs(forwards) == _venue_camera_configs(backwards)
        assert [camera.camera_id for camera in _venue_camera_configs(forwards)] == [
            "cam_a",
            "cam_b",
        ]

    def test_an_integrity_reference_is_carried_through(self, tmp_path: Path) -> None:
        args = parse(
            *base_arguments(tmp_path),
            "--calibration",
            "cam_a=a.json",
            "--calibration-reference",
            "cam_a=ref.png",
        )
        (camera,) = _venue_camera_configs(args)
        assert camera.calibration_path == Path("a.json")
        assert camera.integrity_reference_path == Path("ref.png")
        assert camera.integrity_report_only is False

    def test_report_only_applies_only_to_monitored_cameras(self, tmp_path: Path) -> None:
        args = parse(
            *base_arguments(tmp_path),
            "--calibration",
            "cam_a=a.json",
            "--calibration",
            "cam_b=b.json",
            "--calibration-reference",
            "cam_a=ref.png",
            "--integrity-report-only",
        )
        monitored, unmonitored = _venue_camera_configs(args)
        assert monitored.integrity_report_only is True
        assert unmonitored.integrity_reference_path is None
        assert unmonitored.integrity_report_only is False

    def test_a_reference_without_a_calibration_is_refused(self, tmp_path: Path) -> None:
        """There is no metric claim to invalidate on an uncalibrated camera."""
        args = parse(
            *base_arguments(tmp_path),
            "--calibration",
            "cam_a=a.json",
            "--calibration-reference",
            "cam_b=ref.png",
        )
        with pytest.raises(ConfigurationError, match="cam_b"):
            _venue_camera_configs(args)

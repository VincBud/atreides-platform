"""Loading calibration from disk, and the shipped example file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atreides_vision.calibration.loader import load_calibration, load_calibration_fit
from atreides_vision.errors import CalibrationError
from synthetic_cameras import CAMERA_A, calibration_for

RECTANGLE = [(0.0, 0.0), (12.0, 0.0), (12.0, 9.0), (0.0, 9.0)]
EXAMPLE = Path(__file__).resolve().parents[3] / "config" / "calibration.example.json"


def write(tmp_path: Path, payload: object, name: str = "calibration.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def valid_payload() -> dict[str, object]:
    return calibration_for(CAMERA_A, RECTANGLE, camera_id="cam_01").to_record()


class TestLoading:
    def test_loads_a_valid_file(self, tmp_path: Path) -> None:
        calibration = load_calibration(write(tmp_path, valid_payload()))
        assert calibration.camera_id == "cam_01"
        assert calibration.point_count == 4

    def test_round_trips_through_disk(self, tmp_path: Path) -> None:
        original = calibration_for(CAMERA_A, RECTANGLE, camera_id="cam_01")
        assert load_calibration(write(tmp_path, original.to_record())) == original

    def test_solves_the_homography(self, tmp_path: Path) -> None:
        fit = load_calibration_fit(write(tmp_path, valid_payload()))
        assert fit.camera_id == "cam_01"
        assert fit.quality.is_exact

    def test_accepts_a_string_path(self, tmp_path: Path) -> None:
        assert load_calibration(str(write(tmp_path, valid_payload()))).point_count == 4


class TestFailures:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(CalibrationError, match="Calibration file not found"):
            load_calibration(tmp_path / "absent.json")

    def test_directory_instead_of_file(self, tmp_path: Path) -> None:
        with pytest.raises(CalibrationError, match="is not a file"):
            load_calibration(tmp_path)

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(CalibrationError, match="is not valid JSON"):
            load_calibration(path)

    def test_invalid_content_names_the_file(self, tmp_path: Path) -> None:
        payload = valid_payload()
        del payload["camera_id"]
        path = write(tmp_path, payload)
        with pytest.raises(CalibrationError, match=f"Invalid calibration {path}"):
            load_calibration(path)

    def test_too_few_points(self, tmp_path: Path) -> None:
        payload = valid_payload()
        payload["correspondences"] = payload["correspondences"][:3]  # type: ignore[index]
        with pytest.raises(CalibrationError, match="at least 4 are required"):
            load_calibration(write(tmp_path, payload))

    def test_unsolvable_geometry_fails_at_load_not_at_runtime(self, tmp_path: Path) -> None:
        """A broken calibration must stop the run at startup."""
        payload = valid_payload()
        payload["correspondences"] = [
            {"image": [100.0, 900.0], "venue": [0.0, 0.0]},
            {"image": [500.0, 900.0], "venue": [3.0, 0.0]},
            {"image": [900.0, 900.0], "venue": [6.0, 0.0]},
            {"image": [1300.0, 900.0], "venue": [9.0, 0.0]},
        ]
        with pytest.raises(CalibrationError, match="straight line"):
            load_calibration_fit(write(tmp_path, payload))


class TestShippedExample:
    def test_the_example_file_exists(self) -> None:
        assert EXAMPLE.is_file(), f"expected an example calibration at {EXAMPLE}"

    def test_the_example_loads_and_solves(self) -> None:
        fit = load_calibration_fit(EXAMPLE)
        assert fit.quality.point_count >= 4
        assert fit.quality.relative_error < 0.001

    def test_the_example_is_labelled_synthetic(self) -> None:
        """It must not be mistaken for measured ground truth."""
        calibration = load_calibration(EXAMPLE)
        assert calibration.description is not None
        assert "SYNTHETIC" in calibration.description.upper()
        assert "NOT" in calibration.description.upper()

    def test_the_example_demonstrates_more_than_the_minimum(self) -> None:
        """Four points always fit exactly, so the example shows redundancy."""
        assert load_calibration(EXAMPLE).is_overdetermined

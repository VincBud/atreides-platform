"""The V0.4.9b-B additive ``CalibrationFit`` injection seam.

A validation adapter can build an Atreides :class:`CalibrationFit` upstream (for
WILDTRACK, from its OpenCV calibration) and hand it to the runner directly,
instead of pointing at a calibration document on disk. This must be exactly
equivalent to the path route: same single :class:`GroundPlaneProjector`, same
artifacts, same integrity behaviour. Synthetic geometry only — no real corpus.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import test_venue_validation_runner as base
from atreides_vision.association.models import AssociationOutcome
from atreides_vision.calibration.fitting import CalibrationFit, fit_calibration
from atreides_vision.calibration.models import ReferenceFrame
from atreides_vision.errors import ConfigurationError
from atreides_vision.venue_validation import (
    VenueValidationCameraConfig,
    VenueValidationConfig,
)
from synthetic_cameras import CAMERA_A, CAMERA_B, FRAME_A, FRAME_B, Matrix3x3, calibration_for

CAM_A = base.CAM_A
CAM_B = base.CAM_B


def fit_for(matrix: Matrix3x3, camera_id: str, reference_frame: ReferenceFrame) -> CalibrationFit:
    """A solved calibration fit for one synthetic camera, built in memory."""
    return fit_calibration(
        calibration_for(
            matrix,
            base.CALIBRATION_REGION,
            camera_id=camera_id,
            reference_frame=reference_frame,
        )
    )


def inject_fits(config: VenueValidationConfig) -> VenueValidationConfig:
    """Swap every camera's ``calibration_path`` for an equivalent in-memory fit."""
    fits = {
        CAM_A: fit_for(CAMERA_A, CAM_A, FRAME_A),
        CAM_B: fit_for(CAMERA_B, CAM_B, FRAME_B),
    }
    cameras = tuple(
        replace(camera, calibration_path=None, calibration_fit=fits[camera.camera_id])
        for camera in config.cameras
    )
    return replace(config, cameras=cameras)


def artifacts(config: VenueValidationConfig) -> dict[str, bytes]:
    return {
        "association": config.association_report_path.read_bytes(),
        "occupancy": config.occupancy_report_path.read_bytes(),
        "summary": config.summary_path.read_bytes(),
    }


class TestConfigContract:
    def test_no_calibration_source_is_accepted(self) -> None:
        camera = VenueValidationCameraConfig(camera_id=CAM_A)
        assert camera.calibration_path is None
        assert camera.calibration_fit is None

    def test_a_path_source_is_accepted(self, tmp_path: Path) -> None:
        camera = VenueValidationCameraConfig(
            camera_id=CAM_A, calibration_path=tmp_path / "cal.json"
        )
        assert camera.calibration_fit is None

    def test_a_fit_source_is_accepted(self) -> None:
        camera = VenueValidationCameraConfig(
            camera_id=CAM_A, calibration_fit=fit_for(CAMERA_A, CAM_A, FRAME_A)
        )
        assert camera.calibration_path is None

    def test_supplying_both_sources_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="both calibration_path and calibration_fit"):
            VenueValidationCameraConfig(
                camera_id=CAM_A,
                calibration_path=tmp_path / "cal.json",
                calibration_fit=fit_for(CAMERA_A, CAM_A, FRAME_A),
            )

    def test_a_fit_for_another_camera_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="built for camera"):
            VenueValidationCameraConfig(
                camera_id=CAM_A, calibration_fit=fit_for(CAMERA_B, CAM_B, FRAME_B)
            )


class TestRunnerIntegration:
    def test_a_directly_injected_fit_reaches_the_real_projector(self, tmp_path: Path) -> None:
        config, sources, detector = base.two_camera_setup(tmp_path)
        summary = base.run(inject_fits(config), sources, detector)

        assert summary.streams[0].calibrated is True
        observations = [
            observation
            for instant in base.records(config.association_report_path)[1:]
            # JSONL records are decoded as object; indexing the known shape here.
            for observation in instant["association"]["observations"]  # type: ignore[index]
        ]
        assert observations
        assert all(o["space"] == "ground_plane_metres" for o in observations)
        assert summary.association_outcomes.get(AssociationOutcome.ASSOCIATED, 0) > 0
        assert summary.suppressed_duplicates > 0

    def test_the_fit_route_is_byte_equivalent_to_the_path_route(self, tmp_path: Path) -> None:
        path_config, path_sources, path_detector = base.two_camera_setup(
            tmp_path, output_dir=tmp_path / "by_path"
        )
        base.run(path_config, path_sources, path_detector)

        fit_config, fit_sources, fit_detector = base.two_camera_setup(
            tmp_path, output_dir=tmp_path / "by_fit"
        )
        base.run(inject_fits(fit_config), fit_sources, fit_detector)

        assert artifacts(path_config) == artifacts(fit_config)

    def test_an_uncalibrated_camera_is_unaffected(self, tmp_path: Path) -> None:
        config, sources, detector = base.two_camera_setup(tmp_path, calibrate_b=False)
        summary = base.run(inject_fits(config), sources, detector)

        # Only cam_a carried a config; cam_b stays uncalibrated and unmerged.
        assert summary.association_outcomes.get(AssociationOutcome.ASSOCIATED, 0) == 0
        assert summary.association_outcomes.get(AssociationOutcome.NOT_COMPARABLE, 0) > 0
        assert summary.suppressed_duplicates == 0

    def test_no_absolute_path_enters_the_summary_by_the_fit_route(self, tmp_path: Path) -> None:
        config, sources, detector = base.two_camera_setup(tmp_path, frames_a=3, frames_b=3)
        base.run(inject_fits(config), sources, detector)

        document = json.loads(config.summary_path.read_text(encoding="utf-8"))
        assert str(tmp_path) not in json.dumps(document)


class TestIntegrityMonitoring:
    def test_a_fit_route_camera_still_gets_its_integrity_monitor(self, tmp_path: Path) -> None:
        import cv2

        from synthetic_scenes import synthetic_scene

        reference = tmp_path / "reference.png"
        assert cv2.imwrite(str(reference), synthetic_scene(120, 90, seed=7))

        config, sources, detector = base.two_camera_setup(tmp_path)
        injected = inject_fits(config)
        cameras = tuple(
            replace(camera, integrity_reference_path=reference)
            if camera.camera_id == CAM_A
            else camera
            for camera in injected.cameras
        )
        summary = base.run(replace(injected, cameras=cameras), sources, detector)

        # The monitor is wired and active — no silent bypass, no trust promotion.
        assert summary.streams[0].integrity_monitored is True
        assert summary.streams[0].calibrated is True


class TestBoundaries:
    ROOT = Path(__file__).resolve().parents[1] / "src" / "atreides_vision"

    def test_the_runner_does_not_import_the_wildtrack_adapter(self) -> None:
        text = (self.ROOT / "venue_validation" / "runner.py").read_text(encoding="utf-8")
        assert "wildtrack" not in text.lower()

    def test_calibration_core_does_not_import_venue_validation(self) -> None:
        for module in (self.ROOT / "calibration").glob("*.py"):
            assert "venue_validation" not in module.read_text(encoding="utf-8")

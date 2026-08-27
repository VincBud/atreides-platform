"""WILDTRACK's OpenCV pinhole calibration parsing, projection, and Atreides conversion."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from atreides_vision.calibration.fitting import CalibrationFit
from atreides_vision.errors import WildtrackDatasetError
from atreides_vision.venue_validation.datasets.wildtrack.calibration import (
    TVEC_CM_TO_M,
    VENUE_ID,
    WildtrackCameraCalibration,
    WildtrackExtrinsic,
    WildtrackIntrinsic,
    load_camera_calibration,
    parse_extrinsic_xml,
    parse_intrinsic_xml,
    project_world_point,
    rotation_matrix_from_rvec,
    to_ground_plane_calibration,
)
from wildtrack_fixtures import (
    DEFAULT_CAMERA_MATRIX,
    DEFAULT_DISTORTION,
    DEFAULT_RVEC,
    DEFAULT_TVEC_CM,
    build_layout,
    write_extrinsic_xml,
    write_intrinsic_xml,
)


def default_calibration(camera_id: str = "C1") -> WildtrackCameraCalibration:
    return WildtrackCameraCalibration(
        camera_id=camera_id,
        intrinsic=WildtrackIntrinsic(
            camera_matrix=DEFAULT_CAMERA_MATRIX, distortion_coefficients=DEFAULT_DISTORTION
        ),
        extrinsic=WildtrackExtrinsic(rvec=DEFAULT_RVEC, tvec_cm=DEFAULT_TVEC_CM),
    )


class TestWildtrackIntrinsic:
    def test_valid(self) -> None:
        intrinsic = WildtrackIntrinsic(
            camera_matrix=DEFAULT_CAMERA_MATRIX, distortion_coefficients=DEFAULT_DISTORTION
        )
        assert intrinsic.focal_length_x == 500.0
        assert intrinsic.focal_length_y == 500.0
        assert intrinsic.principal_point == (50.0, 40.0)

    def test_wrong_matrix_size_rejected(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            WildtrackIntrinsic(
                camera_matrix=(1.0, 2.0, 3.0), distortion_coefficients=DEFAULT_DISTORTION
            )

    def test_non_positive_focal_length_rejected(self) -> None:
        bad_matrix = (0.0, 0.0, 50.0, 0.0, 500.0, 40.0, 0.0, 0.0, 1.0)
        with pytest.raises(WildtrackDatasetError):
            WildtrackIntrinsic(camera_matrix=bad_matrix, distortion_coefficients=DEFAULT_DISTORTION)

    def test_non_finite_entry_rejected(self) -> None:
        bad_matrix = (float("nan"), 0.0, 50.0, 0.0, 500.0, 40.0, 0.0, 0.0, 1.0)
        with pytest.raises(WildtrackDatasetError):
            WildtrackIntrinsic(camera_matrix=bad_matrix, distortion_coefficients=DEFAULT_DISTORTION)


class TestWildtrackExtrinsic:
    def test_tvec_m_scales_centimetres_to_metres(self) -> None:
        extrinsic = WildtrackExtrinsic(rvec=(0.0, 0.0, 0.0), tvec_cm=(100.0, -200.0, 300.0))
        assert extrinsic.tvec_m == (1.0, -2.0, 3.0)
        assert extrinsic.tvec_m == tuple(v * TVEC_CM_TO_M for v in extrinsic.tvec_cm)

    def test_non_finite_entry_rejected(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            WildtrackExtrinsic(rvec=(0.0, 0.0, float("inf")), tvec_cm=(0.0, 0.0, 0.0))


class TestParseIntrinsicXml:
    def test_happy_path(self, tmp_path: Path) -> None:
        path = tmp_path / "intr_CVLab1.xml"
        write_intrinsic_xml(path)
        intrinsic = parse_intrinsic_xml(path)
        assert intrinsic.camera_matrix == DEFAULT_CAMERA_MATRIX
        assert intrinsic.distortion_coefficients == DEFAULT_DISTORTION

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(WildtrackDatasetError):
            parse_intrinsic_xml(tmp_path / "does_not_exist.xml")

    def test_malformed_xml(self, tmp_path: Path) -> None:
        path = tmp_path / "intr_CVLab1.xml"
        path.write_text("<not><valid", encoding="utf-8")
        with pytest.raises(WildtrackDatasetError):
            parse_intrinsic_xml(path)

    def test_missing_camera_matrix_tag(self, tmp_path: Path) -> None:
        path = tmp_path / "intr_CVLab1.xml"
        path.write_text(
            "<opencv_storage><distortion_coefficients><data>0 0 0 0 0</data>"
            "</distortion_coefficients></opencv_storage>",
            encoding="utf-8",
        )
        with pytest.raises(WildtrackDatasetError):
            parse_intrinsic_xml(path)

    def test_camera_matrix_missing_data_child(self, tmp_path: Path) -> None:
        path = tmp_path / "intr_CVLab1.xml"
        path.write_text(
            "<opencv_storage><camera_matrix>1 2 3 4 5 6 7 8 9</camera_matrix>"
            "<distortion_coefficients><data>0 0 0 0 0</data></distortion_coefficients>"
            "</opencv_storage>",
            encoding="utf-8",
        )
        with pytest.raises(WildtrackDatasetError):
            parse_intrinsic_xml(path)

    def test_wrong_element_count(self, tmp_path: Path) -> None:
        path = tmp_path / "intr_CVLab1.xml"
        write_intrinsic_xml(path, camera_matrix=(1.0, 2.0, 3.0))
        with pytest.raises(WildtrackDatasetError):
            parse_intrinsic_xml(path)

    def test_non_numeric_data(self, tmp_path: Path) -> None:
        path = tmp_path / "intr_CVLab1.xml"
        path.write_text(
            "<opencv_storage><camera_matrix><data>a b c d e f g h i</data></camera_matrix>"
            "<distortion_coefficients><data>0 0 0 0 0</data></distortion_coefficients>"
            "</opencv_storage>",
            encoding="utf-8",
        )
        with pytest.raises(WildtrackDatasetError):
            parse_intrinsic_xml(path)


class TestParseExtrinsicXml:
    def test_happy_path(self, tmp_path: Path) -> None:
        path = tmp_path / "extr_CVLab1.xml"
        write_extrinsic_xml(path)
        extrinsic = parse_extrinsic_xml(path)
        assert extrinsic.rvec == DEFAULT_RVEC
        assert extrinsic.tvec_cm == DEFAULT_TVEC_CM

    def test_missing_tvec_tag(self, tmp_path: Path) -> None:
        path = tmp_path / "extr_CVLab1.xml"
        path.write_text("<opencv_storage><rvec>0 0 0</rvec></opencv_storage>", encoding="utf-8")
        with pytest.raises(WildtrackDatasetError):
            parse_extrinsic_xml(path)

    def test_wrong_element_count(self, tmp_path: Path) -> None:
        path = tmp_path / "extr_CVLab1.xml"
        write_extrinsic_xml(path, tvec_cm=(1.0, 2.0, 3.0))
        path.write_text(
            "<opencv_storage><rvec>0 0 0</rvec><tvec>1 2</tvec></opencv_storage>",
            encoding="utf-8",
        )
        with pytest.raises(WildtrackDatasetError):
            parse_extrinsic_xml(path)


class TestLoadCameraCalibration:
    def test_unknown_camera_id_rejected(self, tmp_path: Path) -> None:
        layout = build_layout(tmp_path)
        with pytest.raises(WildtrackDatasetError):
            load_camera_calibration(layout, "C99")

    def test_happy_path(self, tmp_path: Path) -> None:
        layout = build_layout(tmp_path)
        calibration = load_camera_calibration(layout, "C1")
        assert calibration.camera_id == "C1"
        assert calibration.intrinsic.camera_matrix == DEFAULT_CAMERA_MATRIX


class TestRotationMatrixFromRvec:
    def test_zero_vector_is_identity(self) -> None:
        rotation = rotation_matrix_from_rvec((0.0, 0.0, 0.0))
        assert np.allclose(rotation, np.eye(3))

    def test_ninety_degrees_about_z(self) -> None:
        rotation = rotation_matrix_from_rvec((0.0, 0.0, math.pi / 2))
        rotated = rotation @ np.array([1.0, 0.0, 0.0])
        assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)


class TestProjectWorldPoint:
    def test_known_point_projects_to_principal_point(self) -> None:
        # World (0, 3, 0) is directly ahead of the default fixture camera
        # (identity rotation, tvec (0m, -3m, 5m)): camera_point = (0, 0, 5).
        calibration = default_calibration()
        pixel = project_world_point(calibration, (0.0, 3.0, 0.0))
        assert pixel is not None
        u, v = pixel
        assert u == pytest.approx(50.0)
        assert v == pytest.approx(40.0)

    def test_point_behind_camera_returns_none(self) -> None:
        calibration = default_calibration()
        pixel = project_world_point(calibration, (0.0, 3.0, -10.0))
        assert pixel is None


class TestToGroundPlaneCalibration:
    def test_too_few_points_inside_image_raises(self, tmp_path: Path) -> None:
        # The default fixture camera and its default 100x80 image only place a
        # single sampled ground point inside the image bounds.
        layout = build_layout(tmp_path, image_width=100, image_height=80)
        calibration = load_camera_calibration(layout, "C1")
        with pytest.raises(WildtrackDatasetError):
            to_ground_plane_calibration(calibration, layout)

    def test_enough_points_fits_a_calibration(self, tmp_path: Path) -> None:
        # A wide-field camera looking straight down at the plaza places many
        # sampled ground points inside a large enough image.
        image_width, image_height = 1000, 2000
        layout = build_layout(tmp_path, image_width=image_width, image_height=image_height)
        intrinsic_path = layout.intrinsic_path("C1")
        extrinsic_path = layout.extrinsic_path("C1")
        write_intrinsic_xml(
            intrinsic_path,
            camera_matrix=(400.0, 0.0, 500.0, 0.0, 400.0, 500.0, 0.0, 0.0, 1.0),
        )
        write_extrinsic_xml(extrinsic_path, rvec=(0.0, 0.0, 0.0), tvec_cm=(300.0, 900.0, 1000.0))
        calibration = load_camera_calibration(layout, "C1")

        fit = to_ground_plane_calibration(calibration, layout)

        assert isinstance(fit, CalibrationFit)
        assert fit.camera_id == "C1"
        assert fit.venue_id == VENUE_ID

    def test_determinism_of_sampled_correspondences(self, tmp_path: Path) -> None:
        image_width, image_height = 1000, 2000
        layout = build_layout(tmp_path, image_width=image_width, image_height=image_height)
        write_intrinsic_xml(
            layout.intrinsic_path("C1"),
            camera_matrix=(400.0, 0.0, 500.0, 0.0, 400.0, 500.0, 0.0, 0.0, 1.0),
        )
        write_extrinsic_xml(
            layout.extrinsic_path("C1"), rvec=(0.0, 0.0, 0.0), tvec_cm=(300.0, 900.0, 1000.0)
        )
        calibration = load_camera_calibration(layout, "C1")

        first = to_ground_plane_calibration(calibration, layout)
        second = to_ground_plane_calibration(calibration, layout)

        assert first.as_flat_tuple() == second.as_flat_tuple()

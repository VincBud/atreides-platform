"""Fitting a calibration and measuring how well it reproduces its own points."""

from __future__ import annotations

import numpy as np
import pytest

from atreides_vision.calibration.fitting import (
    DEFAULT_MAX_RELATIVE_REPROJECTION_ERROR,
    fit_calibration,
)
from atreides_vision.calibration.models import (
    GroundPlaneCalibration,
    PointCorrespondence,
    ReferenceFrame,
    VenuePoint,
)
from atreides_vision.errors import CalibrationError
from atreides_vision.models import Point
from synthetic_cameras import (
    CAMERA_A,
    CAMERA_B,
    VENUE_ID,
    calibration_for,
    image_points_for,
    project_venue_to_image,
)

RECTANGLE = [(0.0, 0.0), (12.0, 0.0), (12.0, 9.0), (0.0, 9.0)]
DENSE = [*RECTANGLE, (6.0, 4.5), (3.0, 2.0), (9.0, 7.0)]


class TestExactFits:
    def test_four_exact_points_reproduce_themselves(self) -> None:
        fit = fit_calibration(calibration_for(CAMERA_A, RECTANGLE))
        assert fit.quality.is_exact
        assert fit.quality.mean_venue_error_m == pytest.approx(0.0, abs=1e-9)
        assert fit.quality.mean_image_error_px == pytest.approx(0.0, abs=1e-6)
        assert fit.quality.point_count == 4

    def test_consistent_extra_points_stay_exact(self) -> None:
        fit = fit_calibration(calibration_for(CAMERA_A, DENSE))
        assert fit.quality.is_exact
        assert fit.quality.point_count == 7

    def test_records_the_calibrated_extent(self) -> None:
        fit = fit_calibration(calibration_for(CAMERA_A, RECTANGLE))
        assert fit.quality.venue_extent_m == pytest.approx(15.0)

    def test_carries_identity_through(self) -> None:
        fit = fit_calibration(
            calibration_for(CAMERA_A, RECTANGLE, camera_id="cam_north", venue_id="club_x")
        )
        assert fit.camera_id == "cam_north"
        assert fit.venue_id == "club_x"


class TestNoisyFits:
    def test_four_noisy_points_still_report_zero_error(self) -> None:
        """Four points always fit exactly, however badly they were measured.

        Eight equations, eight degrees of freedom: there is no redundancy for
        the fit to disagree with. This is the practical argument for supplying
        more than four reference points — at exactly four, a reprojection error
        of zero tells you nothing about whether the measurements were right.
        """
        noise = [(6.0, -6.0), (-5.0, 4.0), (7.0, 5.0), (-4.0, -7.0)]
        fit = fit_calibration(calibration_for(CAMERA_A, RECTANGLE, image_noise=noise))
        assert fit.quality.is_exact

    def test_redundant_points_expose_a_mismeasurement(self) -> None:
        """With more than four, a bad point has something to contradict."""
        noise = [(0.0, 0.0)] * 4 + [(20.0, 20.0), (0.0, 0.0), (0.0, 0.0)]
        fit = fit_calibration(calibration_for(CAMERA_A, DENSE, image_noise=noise))
        assert not fit.quality.is_exact
        assert fit.quality.mean_venue_error_m > 0.01
        assert fit.quality.max_venue_error_m >= fit.quality.mean_venue_error_m

    def test_larger_mismeasurement_produces_larger_error(self) -> None:
        def error_for(offset: float) -> float:
            noise = [(0.0, 0.0)] * 4 + [(offset, offset), (0.0, 0.0), (0.0, 0.0)]
            return fit_calibration(
                calibration_for(CAMERA_A, DENSE, image_noise=noise)
            ).quality.mean_venue_error_m

        assert error_for(40.0) > error_for(10.0) > error_for(2.0)

    def test_error_is_reported_in_both_units(self) -> None:
        noise = [(10.0, 10.0)] * 4 + [(0.0, 0.0)] * 3
        quality = fit_calibration(calibration_for(CAMERA_A, DENSE, image_noise=noise)).quality
        assert quality.mean_venue_error_m > 0.0
        assert quality.mean_image_error_px > 0.0
        assert quality.max_venue_error_m >= quality.mean_venue_error_m
        assert quality.max_image_error_px >= quality.mean_image_error_px


def mispaired(venue_points: list[tuple[float, float]]) -> GroundPlaneCalibration:
    """Correct image points paired with the wrong venue points."""
    image = image_points_for(CAMERA_A, DENSE)
    return GroundPlaneCalibration(
        camera_id="cam_01",
        venue_id=VENUE_ID,
        reference_frame=ReferenceFrame(1920, 1080),
        correspondences=tuple(
            PointCorrespondence(image=Point(ix, iy), venue=VenuePoint(vx, vy))
            for (ix, iy), (vx, vy) in zip(image, venue_points, strict=True)
        ),
    )


class TestSafeguard:
    @pytest.mark.parametrize(
        ("name", "venue_order"),
        [
            ("rotated by one", DENSE[1:] + DENSE[:1]),
            ("first two swapped", [DENSE[1], DENSE[0], *DENSE[2:]]),
        ],
    )
    def test_rejects_mispaired_correspondences(
        self, name: str, venue_order: list[tuple[float, float]]
    ) -> None:
        """Listing venue points in the wrong order must not silently "work".

        Caught by the horizon check rather than the reprojection limit: the
        contradiction is severe enough that no single planar surface explains
        the points at all, which is a more specific diagnosis.
        """
        with pytest.raises(CalibrationError, match="horizon line"):
            fit_calibration(mispaired(venue_order))

    def test_rejects_a_grossly_mismeasured_point(self) -> None:
        """A reference point hundreds of pixels out fails the safeguard."""
        noise = [(0.0, 0.0)] * 4 + [(600.0, 360.0), (0.0, 0.0), (0.0, 0.0)]
        with pytest.raises(CalibrationError, match="does not reproduce its own reference points"):
            fit_calibration(calibration_for(CAMERA_A, DENSE, image_noise=noise))

    def test_the_message_names_the_camera_and_the_numbers(self) -> None:
        noise = [(0.0, 0.0)] * 4 + [(600.0, 360.0), (0.0, 0.0), (0.0, 0.0)]
        model = calibration_for(CAMERA_A, DENSE, image_noise=noise, camera_id="cam_west")
        with pytest.raises(CalibrationError) as caught:
            fit_calibration(model)
        message = str(caught.value)
        assert "cam_west" in message
        assert "limit" in message

    def test_the_safeguard_is_relative_not_absolute(self) -> None:
        """The same relative error passes at any venue scale.

        The limit encodes no venue size: it is a fraction of whatever region
        the operator actually calibrated.
        """
        noise = [(0.0, 0.0)] * 4 + [(30.0, 30.0), (0.0, 0.0), (0.0, 0.0)]
        tiny_room = [
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 2.0),
            (0.0, 2.0),
            (1.5, 1.0),
            (1.0, 0.5),
            (2.0, 1.5),
        ]
        small = fit_calibration(calibration_for(CAMERA_A, tiny_room, image_noise=noise))
        assert small.quality.relative_error < DEFAULT_MAX_RELATIVE_REPROJECTION_ERROR

    def test_the_limit_is_configurable(self) -> None:
        noise = [(0.0, 0.0)] * 4 + [(25.0, 25.0), (0.0, 0.0), (0.0, 0.0)]
        model = calibration_for(CAMERA_A, DENSE, image_noise=noise)
        relative = fit_calibration(model).quality.relative_error

        with pytest.raises(CalibrationError, match="does not reproduce"):
            fit_calibration(model, max_relative_error=relative / 2.0)

    def test_rejects_a_non_positive_limit(self) -> None:
        with pytest.raises(CalibrationError, match="max_relative_error must be positive"):
            fit_calibration(calibration_for(CAMERA_A, RECTANGLE), max_relative_error=0.0)


class TestFitInterface:
    def test_exposes_both_directions(self) -> None:
        fit = fit_calibration(calibration_for(CAMERA_A, RECTANGLE))
        venue = np.array([[5.0, 6.0]])
        image = fit.project_image_points(venue)
        assert fit.project_venue_points(image) == pytest.approx(venue, abs=1e-8)

    def test_image_projection_matches_the_synthetic_camera(self) -> None:
        fit = fit_calibration(calibration_for(CAMERA_A, RECTANGLE))
        expected = project_venue_to_image(CAMERA_A, 7.0, 3.0)
        assert fit.project_image_points(np.array([[7.0, 3.0]]))[0] == pytest.approx(
            expected, abs=1e-6
        )

    def test_matrix_is_copied_not_shared(self) -> None:
        """A caller must not be able to corrupt the fit by mutating what it got."""
        fit = fit_calibration(calibration_for(CAMERA_A, RECTANGLE))
        borrowed = fit.matrix
        borrowed[0, 0] = 999.0
        assert fit.matrix[0, 0] != 999.0

    def test_inverse_is_copied_too(self) -> None:
        fit = fit_calibration(calibration_for(CAMERA_A, RECTANGLE))
        fit.inverse_matrix[0, 0] = 999.0
        assert fit.inverse_matrix[0, 0] != 999.0

    def test_flat_tuple_has_nine_floats(self) -> None:
        flat = fit_calibration(calibration_for(CAMERA_A, RECTANGLE)).as_flat_tuple()
        assert len(flat) == 9
        assert all(isinstance(value, float) for value in flat)

    def test_quality_record_is_serialisable(self) -> None:
        record = fit_calibration(calibration_for(CAMERA_B, DENSE)).quality.to_record()
        assert record["point_count"] == 7
        assert record["mean_venue_error_m"] == pytest.approx(0.0, abs=1e-6)


class TestDeterminism:
    def test_fitting_twice_gives_the_same_transform(self) -> None:
        model = calibration_for(CAMERA_B, DENSE)
        assert np.array_equal(fit_calibration(model).matrix, fit_calibration(model).matrix)

    def test_quality_is_reproducible(self) -> None:
        noise = [(3.0, -3.0)] * 7
        model = calibration_for(CAMERA_A, DENSE, image_noise=noise)
        assert fit_calibration(model).quality == fit_calibration(model).quality

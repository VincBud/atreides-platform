"""Homography estimation and application, against transforms we defined ourselves.

Every case here builds image points by pushing known venue points through an
explicit synthetic camera, then asks the estimator to recover a transform that
undoes it. The expected answer is therefore known independently of the code
under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from atreides_vision.calibration.homography import (
    MIN_CORRESPONDENCES,
    apply_homography,
    estimate_homography,
    invert_homography,
    project_point,
)
from atreides_vision.errors import CalibrationError, ProjectionError
from synthetic_cameras import (
    CAMERA_A,
    CAMERA_B,
    IDENTITY,
    Matrix3x3,
    image_points_for,
    project_venue_to_image,
)

RECTANGLE = [(0.0, 0.0), (12.0, 0.0), (12.0, 9.0), (0.0, 9.0)]

#: A pure affine camera — a nadir-mounted view, with no perspective row.
AFFINE: Matrix3x3 = ((60.0, 0.0, 300.0), (0.0, 60.0, 200.0), (0.0, 0.0, 1.0))


def recover(matrix: Matrix3x3, venue_points: list[tuple[float, float]]) -> np.ndarray:
    """Estimate the image-to-venue homography for a synthetic camera."""
    image = np.array(image_points_for(matrix, venue_points), dtype=np.float64)
    return estimate_homography(image, np.array(venue_points, dtype=np.float64))


class TestExactRecovery:
    @pytest.mark.parametrize(
        ("name", "matrix"),
        [
            ("identity", IDENTITY),
            ("affine", AFFINE),
            ("perspective_a", CAMERA_A),
            ("perspective_b", CAMERA_B),
        ],
    )
    def test_four_points_recover_the_transform_exactly(self, name: str, matrix: Matrix3x3) -> None:
        homography = recover(matrix, RECTANGLE)
        image = np.array(image_points_for(matrix, RECTANGLE), dtype=np.float64)
        recovered = apply_homography(homography, image)
        assert recovered == pytest.approx(np.array(RECTANGLE), abs=1e-9), name

    def test_recovers_points_not_used_for_calibration(self) -> None:
        """The real test of a transform: does it work away from its own points?"""
        homography = recover(CAMERA_A, RECTANGLE)
        probes = [(3.7, 1.2), (8.1, 6.4), (11.9, 8.9), (0.4, 4.0)]
        image = np.array(image_points_for(CAMERA_A, probes), dtype=np.float64)
        assert apply_homography(homography, image) == pytest.approx(np.array(probes), abs=1e-8)

    def test_extrapolates_beyond_the_calibrated_region(self) -> None:
        """A camera sees past the points used to calibrate it."""
        homography = recover(CAMERA_A, RECTANGLE)
        outside = [(-3.0, 2.0), (16.0, 5.0)]
        image = np.array(image_points_for(CAMERA_A, outside), dtype=np.float64)
        assert apply_homography(homography, image) == pytest.approx(np.array(outside), abs=1e-7)


class TestOverdetermined:
    def test_extra_points_are_used_not_discarded(self) -> None:
        """Eight consistent points must fit as exactly as four."""
        points = [*RECTANGLE, (6.0, 0.0), (12.0, 4.5), (6.0, 9.0), (0.0, 4.5)]
        homography = recover(CAMERA_A, points)
        image = np.array(image_points_for(CAMERA_A, points), dtype=np.float64)
        assert apply_homography(homography, image) == pytest.approx(np.array(points), abs=1e-8)

    def test_least_squares_spreads_error_from_one_bad_point(self) -> None:
        """One mismeasured point must not be reproduced exactly at the others' expense."""
        points = [*RECTANGLE, (6.0, 4.5)]
        image = np.array(image_points_for(CAMERA_A, points), dtype=np.float64)
        image[4] += np.array([25.0, 25.0])  # the centre point was measured badly

        homography = estimate_homography(image, np.array(points, dtype=np.float64))
        errors = np.linalg.norm(apply_homography(homography, image) - np.array(points), axis=1)
        assert errors.max() > 0.0, "a contradictory point must leave residual error"
        # The bad point is not honoured exactly; the error is shared out.
        assert errors[4] < 0.5
        assert np.count_nonzero(errors > 1e-6) > 1

    def test_more_points_reduce_the_effect_of_one_bad_measurement(self) -> None:
        sparse = [*RECTANGLE, (6.0, 4.5)]
        dense = [
            *RECTANGLE,
            (6.0, 4.5),
            (3.0, 2.0),
            (9.0, 2.0),
            (3.0, 7.0),
            (9.0, 7.0),
            (6.0, 0.0),
            (6.0, 9.0),
        ]

        def worst_error(points: list[tuple[float, float]]) -> float:
            image = np.array(image_points_for(CAMERA_A, points), dtype=np.float64)
            image[4] += np.array([25.0, 25.0])
            homography = estimate_homography(image, np.array(points, dtype=np.float64))
            probe = np.array(image_points_for(CAMERA_A, [(2.0, 8.0)]), dtype=np.float64)
            return float(
                np.linalg.norm(apply_homography(homography, probe) - np.array([[2.0, 8.0]]))
            )

        assert worst_error(dense) < worst_error(sparse)


class TestValidation:
    @pytest.mark.parametrize("count", [0, 1, 2, 3])
    def test_rejects_too_few_correspondences(self, count: int) -> None:
        points = RECTANGLE[:count]
        source = np.array(points, dtype=np.float64).reshape(-1, 2)
        with pytest.raises(CalibrationError, match=f"at least {MIN_CORRESPONDENCES}"):
            estimate_homography(source, source)

    def test_rejects_mismatched_counts(self) -> None:
        source = np.array(RECTANGLE, dtype=np.float64)
        target = np.array([*RECTANGLE[:3], (1.0, 1.0), (2.0, 2.0)], dtype=np.float64)
        with pytest.raises(CalibrationError, match="same number of points"):
            estimate_homography(source, target)

    @pytest.mark.parametrize("shape", [(4,), (4, 3), (4, 1)])
    def test_rejects_wrong_shapes(self, shape: tuple[int, ...]) -> None:
        bad = np.zeros(shape, dtype=np.float64)
        with pytest.raises(CalibrationError, match=r"shape \(N, 2\)"):
            estimate_homography(bad, np.array(RECTANGLE, dtype=np.float64))

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_values(self, bad_value: float) -> None:
        source = np.array(image_points_for(CAMERA_A, RECTANGLE), dtype=np.float64)
        source[2, 0] = bad_value
        with pytest.raises(CalibrationError, match="must all be finite"):
            estimate_homography(source, np.array(RECTANGLE, dtype=np.float64))

    def test_rejects_collinear_points(self) -> None:
        """Four points on a line determine no plane."""
        line = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        image = np.array(image_points_for(CAMERA_A, line), dtype=np.float64)
        with pytest.raises(CalibrationError, match="geometrically degenerate"):
            estimate_homography(image, np.array(line, dtype=np.float64))

    def test_rejects_three_of_four_collinear(self) -> None:
        almost = [(0.0, 0.0), (4.0, 0.0), (8.0, 0.0), (3.0, 6.0)]
        image = np.array(image_points_for(CAMERA_A, almost), dtype=np.float64)
        with pytest.raises(CalibrationError, match="geometrically degenerate"):
            estimate_homography(image, np.array(almost, dtype=np.float64))

    def test_rejects_coincident_points(self) -> None:
        same = np.full((4, 2), 5.0, dtype=np.float64)
        with pytest.raises(CalibrationError, match="same position"):
            estimate_homography(same, np.array(RECTANGLE, dtype=np.float64))

    def test_rejects_points_straddling_the_horizon(self) -> None:
        """Reference points on both sides of the horizon are not one surface."""
        source = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        # A target layout that forces opposite-sign w for the source points.
        target = np.array([(0.0, 0.0), (1.0, 0.0), (-40.0, -40.0), (0.0, 1.0)])
        with pytest.raises(CalibrationError, match=r"horizon|degenerate"):
            estimate_homography(source, target)


class TestOrientation:
    def test_calibrated_points_project_with_positive_w(self) -> None:
        """The sign convention that makes 'beyond the horizon' detectable."""
        homography = recover(CAMERA_A, RECTANGLE)
        image = np.array(image_points_for(CAMERA_A, RECTANGLE), dtype=np.float64)
        homogeneous = np.column_stack([image, np.ones(len(image))])
        assert np.all(homogeneous @ homography[2, :] > 0)


class TestApplyAndInvert:
    def test_inverse_round_trips(self) -> None:
        homography = recover(CAMERA_A, RECTANGLE)
        inverse = invert_homography(homography)
        venue = np.array([(4.0, 3.0), (9.5, 7.5)])
        image = apply_homography(inverse, venue)
        assert apply_homography(homography, image) == pytest.approx(venue, abs=1e-8)

    def test_inverse_matches_the_synthetic_camera(self) -> None:
        homography = recover(CAMERA_A, RECTANGLE)
        inverse = invert_homography(homography)
        expected = project_venue_to_image(CAMERA_A, 5.0, 6.0)
        assert apply_homography(inverse, np.array([[5.0, 6.0]]))[0] == pytest.approx(
            expected, abs=1e-6
        )

    def test_rejects_a_singular_matrix(self) -> None:
        singular = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]])
        with pytest.raises(CalibrationError, match="singular"):
            invert_homography(singular)

    def test_rejects_projection_onto_the_horizon(self) -> None:
        homography = recover(CAMERA_A, RECTANGLE)
        # Solve for the image row where w == 0 for this transform.
        _, h, i = homography[2, :]
        y_on_horizon = -i / h
        with pytest.raises(ProjectionError, match="horizon"):
            apply_homography(homography, np.array([[500.0, y_on_horizon]]))


class TestProjectPoint:
    def test_matches_the_array_path(self) -> None:
        """The scalar fast path must agree exactly with the array one."""
        homography = recover(CAMERA_A, RECTANGLE)
        flat = tuple(float(value) for value in homography.reshape(-1))
        for venue in [(1.0, 1.0), (7.5, 4.25), (11.0, 8.0)]:
            image = project_venue_to_image(CAMERA_A, *venue)
            scalar = project_point(flat, *image)
            array = apply_homography(homography, np.array([image]))[0]
            assert scalar == pytest.approx(tuple(array), abs=1e-9)
            assert scalar == pytest.approx(venue, abs=1e-8)

    def test_rejects_a_point_beyond_the_horizon(self) -> None:
        homography = recover(CAMERA_A, RECTANGLE)
        flat = tuple(float(value) for value in homography.reshape(-1))
        _, h, i = homography[2, :]
        above_horizon = -i / h - 50.0  # past the vanishing line
        with pytest.raises(ProjectionError, match="horizon"):
            project_point(flat, 900.0, above_horizon)

    def test_is_deterministic(self) -> None:
        homography = recover(CAMERA_B, RECTANGLE)
        flat = tuple(float(value) for value in homography.reshape(-1))
        results = {project_point(flat, 700.0, 480.0) for _ in range(10)}
        assert len(results) == 1


class TestDeterminism:
    def test_repeated_estimation_is_identical(self) -> None:
        first = recover(CAMERA_B, RECTANGLE)
        second = recover(CAMERA_B, RECTANGLE)
        assert np.array_equal(first, second)

    def test_point_order_does_not_change_the_result_materially(self) -> None:
        forward = recover(CAMERA_A, RECTANGLE)
        reversed_points = list(reversed(RECTANGLE))
        backward = recover(CAMERA_A, reversed_points)
        probe = np.array(image_points_for(CAMERA_A, [(5.0, 5.0)]), dtype=np.float64)
        assert apply_homography(forward, probe) == pytest.approx(
            apply_homography(backward, probe), abs=1e-8
        )

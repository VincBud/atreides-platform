"""Inverse projection and the visual calibration diagnostic.

The overlay exists because reprojection error cannot detect reference points
measured against the wrong spot on the floor. These tests check that what is
drawn is geometrically faithful — that a marker lands where the calibration
says it should — rather than merely that pixels changed.
"""

from __future__ import annotations

import numpy as np
import pytest

from atreides_vision.calibration.fitting import fit_calibration
from atreides_vision.calibration.models import VenuePoint
from atreides_vision.calibration.projector import GroundPlaneProjector
from atreides_vision.errors import ConfigurationError, ProjectionError
from atreides_vision.visualization.calibration_overlay import (
    CalibrationOverlayConfig,
    render_calibration_overlay,
)
from synthetic_cameras import CAMERA_A, CAMERA_B, FRAME_A, calibration_for, project_venue_to_image

RECTANGLE = [(0.0, 0.0), (12.0, 0.0), (12.0, 9.0), (0.0, 9.0)]


def projector(
    matrix: tuple[tuple[float, float, float], ...] = CAMERA_A,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> GroundPlaneProjector:
    fit = fit_calibration(
        calibration_for(matrix, RECTANGLE, camera_id="cam_a", reference_frame=FRAME_A)  # type: ignore[arg-type]
    )
    return GroundPlaneProjector(fit, frame_width, frame_height)


def blank(width: int = 1920, height: int = 1080) -> np.ndarray:
    return np.full((height, width, 3), 50, dtype=np.uint8)


class TestInverseProjection:
    @pytest.mark.parametrize("venue", [(0.0, 0.0), (12.0, 9.0), (6.0, 4.5), (2.5, 7.25)])
    def test_round_trips_through_the_forward_transform(self, venue: tuple[float, float]) -> None:
        instance = projector()
        pixel = instance.project_from_venue(VenuePoint(*venue))
        assert (instance.project_to_venue(pixel).x, instance.project_to_venue(pixel).y) == (
            pytest.approx(venue, abs=1e-7)
        )

    def test_matches_the_synthetic_camera(self) -> None:
        """The inverse must reproduce the camera we generated the points with."""
        expected = project_venue_to_image(CAMERA_A, 7.0, 3.0)
        pixel = projector().project_from_venue(VenuePoint(7.0, 3.0))
        assert (pixel.x, pixel.y) == pytest.approx(expected, abs=1e-6)

    def test_works_for_a_different_camera(self) -> None:
        expected = project_venue_to_image(CAMERA_B, 5.0, 6.0)
        pixel = projector(CAMERA_B).project_from_venue(VenuePoint(5.0, 6.0))
        assert (pixel.x, pixel.y) == pytest.approx(expected, abs=1e-6)

    def test_respects_the_runtime_resolution(self) -> None:
        """Grid overlays must land correctly on a rescaled frame."""
        full = projector().project_from_venue(VenuePoint(6.0, 4.5))
        half = projector(frame_width=960, frame_height=540).project_from_venue(VenuePoint(6.0, 4.5))
        assert (half.x, half.y) == pytest.approx((full.x / 2, full.y / 2), abs=1e-6)

    def test_round_trips_at_a_rescaled_resolution(self) -> None:
        instance = projector(frame_width=640, frame_height=360)
        pixel = instance.project_from_venue(VenuePoint(9.0, 2.0))
        recovered = instance.project_to_venue(pixel)
        assert (recovered.x, recovered.y) == pytest.approx((9.0, 2.0), abs=1e-7)

    def test_is_deterministic(self) -> None:
        instance = projector()
        results = {
            (point.x, point.y)
            for point in (instance.project_from_venue(VenuePoint(3.0, 3.0)) for _ in range(10))
        }
        assert len(results) == 1

    def test_is_cached_without_changing_results(self) -> None:
        instance = projector()
        first = instance.project_from_venue(VenuePoint(4.0, 4.0))
        second = instance.project_from_venue(VenuePoint(4.0, 4.0))
        assert (first.x, first.y) == (second.x, second.y)

    def test_a_venue_point_on_the_horizon_fails_explicitly(self) -> None:
        """Not every venue coordinate has an image position."""
        instance = projector()
        with pytest.raises(ProjectionError):
            # Far behind the camera plane, where the inverse divides by zero.
            for distance in (-1e6, -1e7, -1e8):
                instance.project_from_venue(VenuePoint(0.0, distance))


class TestOverlayRendering:
    def test_returns_a_new_frame_of_the_same_shape(self) -> None:
        frame = blank()
        rendered = render_calibration_overlay(frame, projector())
        assert rendered.shape == frame.shape
        assert rendered.dtype == frame.dtype

    def test_does_not_modify_the_input(self) -> None:
        frame = blank()
        original = frame.copy()
        render_calibration_overlay(frame, projector())
        assert np.array_equal(frame, original)

    def test_draws_something(self) -> None:
        frame = blank()
        assert not np.array_equal(render_calibration_overlay(frame, projector()), frame)

    def test_reference_points_are_drawn_where_the_calibration_says(self) -> None:
        """A marker in the wrong place is the failure this view exists to reveal."""
        instance = projector()
        rendered = render_calibration_overlay(
            blank(),
            instance,
            CalibrationOverlayConfig(draw_grid=False, draw_hull=False, draw_axes=False),
        )
        base = blank()
        for correspondence in instance.calibration.correspondences:
            x, y = round(correspondence.image.x), round(correspondence.image.y)
            patch = rendered[max(0, y - 6) : y + 7, max(0, x - 6) : x + 7]
            base_patch = base[max(0, y - 6) : y + 7, max(0, x - 6) : x + 7]
            assert not np.array_equal(patch, base_patch), (
                f"no marker drawn at reference point ({x}, {y})"
            )

    def test_reference_points_rescale_with_the_frame(self) -> None:
        instance = projector(frame_width=960, frame_height=540)
        rendered = render_calibration_overlay(
            blank(960, 540),
            instance,
            CalibrationOverlayConfig(draw_grid=False, draw_hull=False, draw_axes=False),
        )
        base = blank(960, 540)
        for correspondence in instance.calibration.correspondences:
            x, y = round(correspondence.image.x / 2), round(correspondence.image.y / 2)
            patch = rendered[max(0, y - 6) : y + 7, max(0, x - 6) : x + 7]
            assert not np.array_equal(patch, base[max(0, y - 6) : y + 7, max(0, x - 6) : x + 7])

    def test_each_element_changes_the_result(self) -> None:
        """Every toggle actually controls something."""
        frame = blank()
        everything = render_calibration_overlay(frame, projector())
        for disabled in (
            CalibrationOverlayConfig(draw_grid=False),
            CalibrationOverlayConfig(draw_hull=False),
            CalibrationOverlayConfig(draw_axes=False),
            CalibrationOverlayConfig(draw_points=False),
            CalibrationOverlayConfig(label_points=False),
        ):
            assert not np.array_equal(
                render_calibration_overlay(frame, projector(), disabled), everything
            )

    def test_grid_spacing_changes_the_drawing(self) -> None:
        frame = blank()
        fine = render_calibration_overlay(
            frame, projector(), CalibrationOverlayConfig(grid_spacing_metres=1.0)
        )
        coarse = render_calibration_overlay(
            frame, projector(), CalibrationOverlayConfig(grid_spacing_metres=4.0)
        )
        assert not np.array_equal(fine, coarse)

    def test_a_finer_grid_draws_more(self) -> None:
        frame = blank()

        def changed_pixels(spacing: float) -> int:
            rendered = render_calibration_overlay(
                frame,
                projector(),
                CalibrationOverlayConfig(
                    grid_spacing_metres=spacing,
                    draw_points=False,
                    draw_axes=False,
                    draw_hull=False,
                ),
            )
            return int(np.count_nonzero(np.any(rendered != frame, axis=2)))

        assert changed_pixels(1.0) > changed_pixels(3.0)

    def test_renders_at_several_resolutions(self) -> None:
        for width, height in ((1920, 1080), (960, 540), (640, 360), (384, 288)):
            instance = projector(frame_width=width, frame_height=height)
            rendered = render_calibration_overlay(blank(width, height), instance)
            assert rendered.shape == (height, width, 3)

    def test_works_for_a_different_camera_geometry(self) -> None:
        frame = blank()
        assert not np.array_equal(render_calibration_overlay(frame, projector(CAMERA_B)), frame)


class TestOverlayValidation:
    @pytest.mark.parametrize("spacing", [0.0, -1.0])
    def test_rejects_a_non_positive_grid_spacing(self, spacing: float) -> None:
        with pytest.raises(ConfigurationError, match="grid_spacing_metres must be positive"):
            CalibrationOverlayConfig(grid_spacing_metres=spacing)

    def test_rejects_a_negative_margin(self) -> None:
        with pytest.raises(ConfigurationError, match="grid_margin_fraction must be non-negative"):
            CalibrationOverlayConfig(grid_margin_fraction=-0.1)

    def test_rejects_an_absurdly_fine_grid(self) -> None:
        """A mistyped spacing must not ask for a million draw calls."""
        with pytest.raises(ConfigurationError, match="more than the"):
            render_calibration_overlay(
                blank(), projector(), CalibrationOverlayConfig(grid_spacing_metres=0.001)
            )

    def test_rejects_an_empty_frame(self) -> None:
        with pytest.raises(ConfigurationError, match="non-empty BGR frame"):
            render_calibration_overlay(np.zeros((0, 0, 3), dtype=np.uint8), projector())

    def test_rejects_a_non_bgr_frame(self) -> None:
        with pytest.raises(ConfigurationError, match="non-empty BGR frame"):
            render_calibration_overlay(np.zeros((10, 10), dtype=np.uint8), projector())


class TestDeterminism:
    def test_rendering_twice_is_identical(self) -> None:
        frame = blank()
        first = render_calibration_overlay(frame, projector())
        second = render_calibration_overlay(frame, projector())
        assert np.array_equal(first, second)

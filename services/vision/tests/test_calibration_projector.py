"""The calibrated projector: venue metres out, at any runtime resolution.

The load-bearing case is :class:`TestRuntimeResolution`. A calibration authored
against 1920x1080 must produce the same venue position when the same video is
decoded at 960x540, because a calibration that quietly became wrong on a
re-encode would be worse than no calibration at all.
"""

from __future__ import annotations

import logging

import pytest

from atreides_vision.behavior.coordinates import MotionProjector, MotionSpace
from atreides_vision.calibration.fitting import fit_calibration
from atreides_vision.calibration.models import ReferenceFrame, VenuePoint
from atreides_vision.calibration.projector import GroundPlaneProjector
from atreides_vision.errors import CalibrationError, ProjectionError
from atreides_vision.models import Point
from synthetic_cameras import (
    CAMERA_A,
    CAMERA_B,
    FRAME_A,
    FRAME_B,
    calibration_for,
    project_venue_to_image,
)

RECTANGLE = [(0.0, 0.0), (12.0, 0.0), (12.0, 9.0), (0.0, 9.0)]


def projector(
    matrix: tuple[tuple[float, float, float], ...] = CAMERA_A,
    reference: ReferenceFrame = FRAME_A,
    frame_width: int | None = None,
    frame_height: int | None = None,
    **calibration_kwargs: object,
) -> GroundPlaneProjector:
    calibration = calibration_for(
        matrix,  # type: ignore[arg-type]
        RECTANGLE,
        reference_frame=reference,
        **calibration_kwargs,  # type: ignore[arg-type]
    )
    return GroundPlaneProjector.from_calibration(calibration, frame_width, frame_height)


class TestProtocolConformance:
    def test_satisfies_the_motion_projector_protocol(self) -> None:
        """The whole point: it is interchangeable with the normalized projector."""
        assert isinstance(projector(), MotionProjector)

    def test_reports_the_metric_space(self) -> None:
        assert projector().space is MotionSpace.GROUND_PLANE_METRES

    def test_projected_points_carry_the_metric_space(self) -> None:
        point = projector().project(Point(*project_venue_to_image(CAMERA_A, 4.0, 3.0)))
        assert point.space is MotionSpace.GROUND_PLANE_METRES


class TestProjection:
    @pytest.mark.parametrize(
        "venue", [(0.0, 0.0), (12.0, 9.0), (6.0, 4.5), (1.25, 8.75), (11.5, 0.5)]
    )
    def test_recovers_known_venue_coordinates(self, venue: tuple[float, float]) -> None:
        image = project_venue_to_image(CAMERA_A, *venue)
        result = projector().project(Point(*image))
        assert (result.x, result.y) == pytest.approx(venue, abs=1e-8)

    def test_works_for_a_completely_different_camera(self) -> None:
        image = project_venue_to_image(CAMERA_B, 7.0, 2.5)
        result = projector(CAMERA_B, FRAME_B).project(Point(*image))
        assert (result.x, result.y) == pytest.approx((7.0, 2.5), abs=1e-8)

    def test_projects_outside_the_calibrated_rectangle(self) -> None:
        """A camera sees past the points used to calibrate it."""
        image = project_venue_to_image(CAMERA_A, 15.0, 11.0)
        result = projector().project(Point(*image))
        assert (result.x, result.y) == pytest.approx((15.0, 11.0), abs=1e-7)

    def test_handles_points_at_the_image_edges(self) -> None:
        for pixel in [Point(0.0, 1079.0), Point(1919.0, 1079.0), Point(0.0, 700.0)]:
            result = projector().project(pixel)
            assert result.space is MotionSpace.GROUND_PLANE_METRES

    def test_project_to_venue_returns_a_venue_point(self) -> None:
        image = project_venue_to_image(CAMERA_A, 5.0, 5.0)
        result = projector().project_to_venue(Point(*image))
        assert isinstance(result, VenuePoint)
        assert (result.x, result.y) == pytest.approx((5.0, 5.0), abs=1e-8)

    def test_is_deterministic(self) -> None:
        instance = projector()
        pixel = Point(1234.0, 890.0)
        results = {(p.x, p.y) for p in (instance.project(pixel) for _ in range(10))}
        assert len(results) == 1

    def test_rejects_points_beyond_the_horizon(self) -> None:
        """An anchor above the calibrated floor has no ground position."""
        with pytest.raises(ProjectionError, match="horizon"):
            projector().project(Point(960.0, -5000.0))


class TestRuntimeResolution:
    """A calibration must survive the same video being decoded differently."""

    @pytest.mark.parametrize(
        ("width", "height"),
        [(1920, 1080), (960, 540), (640, 360), (480, 270), (3840, 2160)],
    )
    def test_same_physical_point_projects_identically_at_any_resolution(
        self, width: int, height: int
    ) -> None:
        venue = (7.5, 6.25)
        reference_image = project_venue_to_image(CAMERA_A, *venue)
        scale = width / FRAME_A.width
        runtime_image = Point(reference_image[0] * scale, reference_image[1] * scale)

        result = projector(frame_width=width, frame_height=height).project(runtime_image)
        assert (result.x, result.y) == pytest.approx(venue, abs=1e-7)

    def test_half_resolution_matches_full_resolution(self) -> None:
        full = projector()
        half = projector(frame_width=960, frame_height=540)

        for venue in [(1.0, 1.0), (6.0, 4.5), (11.0, 8.0)]:
            image = project_venue_to_image(CAMERA_A, *venue)
            full_result = full.project(Point(*image))
            half_result = half.project(Point(image[0] / 2, image[1] / 2))
            assert (half_result.x, half_result.y) == pytest.approx(
                (full_result.x, full_result.y), abs=1e-8
            )

    def test_ignoring_the_rescale_would_be_badly_wrong(self) -> None:
        """Shows the size of the error this handling prevents."""
        venue = (6.0, 4.5)
        image = project_venue_to_image(CAMERA_A, *venue)
        half_image = Point(image[0] / 2, image[1] / 2)

        correct = projector(frame_width=960, frame_height=540).project(half_image)
        uncorrected = projector().project(half_image)

        assert (correct.x, correct.y) == pytest.approx(venue, abs=1e-8)
        error = ((uncorrected.x - venue[0]) ** 2 + (uncorrected.y - venue[1]) ** 2) ** 0.5
        assert error > 1.0, "an uncorrected rescale should be metres wrong"

    def test_defaults_to_the_calibration_reference_frame(self) -> None:
        instance = projector()
        assert instance.runtime_frame == FRAME_A
        assert instance.is_rescaling is False

    def test_reports_when_it_is_rescaling(self) -> None:
        assert projector(frame_width=960, frame_height=540).is_rescaling is True

    @pytest.mark.parametrize(("width", "height"), [(0, 100), (100, 0), (-4, 4)])
    def test_rejects_invalid_runtime_dimensions(self, width: int, height: int) -> None:
        with pytest.raises(CalibrationError, match="Runtime frame size must be positive"):
            projector(frame_width=width, frame_height=height)

    def test_warns_when_the_aspect_ratio_changes(self, caplog: pytest.LogCaptureFixture) -> None:
        """A cropped frame is not a rescaled one, and no rescale can fix it."""
        with caplog.at_level(logging.WARNING):
            projector(frame_width=1920, frame_height=1440)
        assert any("aspect ratio" in record.message for record in caplog.records)

    def test_does_not_warn_for_a_pure_rescale(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            projector(frame_width=960, frame_height=540)
        assert not caplog.records


class TestVenueIdentity:
    def test_exposes_camera_and_venue(self) -> None:
        instance = projector(camera_id="cam_north", venue_id="club_x")
        assert instance.camera_id == "cam_north"
        assert instance.venue_id == "club_x"

    def test_cameras_in_one_venue_share_a_frame(self) -> None:
        left = projector(CAMERA_A, FRAME_A, camera_id="cam_a")
        right = projector(CAMERA_B, FRAME_B, camera_id="cam_b")
        assert left.shares_venue_with(right)

    def test_cameras_in_different_venues_do_not(self) -> None:
        left = projector(camera_id="cam_a", venue_id="club_x")
        right = projector(camera_id="cam_b", venue_id="festival_y")
        assert not left.shares_venue_with(right)

    def test_exposes_calibration_quality(self) -> None:
        assert projector().quality.is_exact


class TestFailureIsExplicit:
    def test_an_unusable_calibration_raises_rather_than_falling_back(self) -> None:
        """Silently reverting to normalized units would change every signal's meaning."""
        noise = [(0.0, 0.0)] * 4 + [(600.0, 360.0), (0.0, 0.0), (0.0, 0.0)]
        broken = calibration_for(
            CAMERA_A,
            [*RECTANGLE, (6.0, 4.5), (3.0, 2.0), (9.0, 7.0)],
            image_noise=noise,
        )
        with pytest.raises(CalibrationError):
            GroundPlaneProjector.from_calibration(broken)

    def test_construction_from_a_prefitted_calibration(self) -> None:
        fit = fit_calibration(calibration_for(CAMERA_A, RECTANGLE))
        instance = GroundPlaneProjector(fit)
        image = project_venue_to_image(CAMERA_A, 3.0, 3.0)
        assert (instance.project(Point(*image)).x) == pytest.approx(3.0, abs=1e-8)

"""Coordinate spaces, the person anchor, and normalized projection."""

from __future__ import annotations

import pytest

from atreides_vision.behavior.coordinates import (
    MotionPoint,
    MotionProjector,
    MotionSpace,
    NormalizedFrameProjector,
    bottom_center_anchor,
)
from atreides_vision.models import BoundingBox, Point
from atreides_vision.zones.geometry import bottom_center

NORMALIZED = MotionSpace.NORMALIZED_IMAGE
GROUND = MotionSpace.GROUND_PLANE_METRES


class TestMotionSpace:
    def test_every_space_names_its_distance_unit(self) -> None:
        for space in MotionSpace:
            assert space.distance_unit, f"{space} has no distance unit"

    def test_normalized_image_is_not_physical(self) -> None:
        """The whole point: frame fractions are not metres."""
        assert NORMALIZED.is_physical is False
        assert NORMALIZED.distance_unit == "normalized_frame_unit"

    def test_ground_plane_is_physical(self) -> None:
        assert GROUND.is_physical is True
        assert GROUND.distance_unit == "m"


class TestMotionPoint:
    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            MotionPoint(float("nan"), 0.5, NORMALIZED)

    def test_distance(self) -> None:
        a = MotionPoint(0.0, 0.0, NORMALIZED)
        b = MotionPoint(0.3, 0.4, NORMALIZED)
        assert a.distance_to(b) == pytest.approx(0.5)

    def test_vector(self) -> None:
        a = MotionPoint(0.1, 0.2, NORMALIZED)
        b = MotionPoint(0.4, 0.7, NORMALIZED)
        assert a.vector_to(b) == pytest.approx((0.3, 0.5))

    def test_allows_positions_outside_the_frame(self) -> None:
        """Someone half out of shot must not be dragged back inside."""
        point = MotionPoint(-0.04, 1.03, NORMALIZED)
        assert (point.x, point.y) == (-0.04, 1.03)

    def test_refuses_to_mix_coordinate_spaces(self) -> None:
        """A normalized point minus a metric one is nonsense, not a number."""
        normalized = MotionPoint(0.5, 0.5, NORMALIZED)
        metric = MotionPoint(0.5, 0.5, GROUND)
        with pytest.raises(ValueError, match="different coordinate spaces"):
            normalized.distance_to(metric)
        with pytest.raises(ValueError, match="different coordinate spaces"):
            normalized.vector_to(metric)


class TestBottomCenterAnchor:
    def test_is_the_middle_of_the_bottom_edge(self) -> None:
        box = BoundingBox(100.0, 40.0, 200.0, 240.0)
        assert bottom_center_anchor(box) == Point(150.0, 240.0)
        assert bottom_center_anchor(box) != box.center

    def test_agrees_with_the_zone_layer(self) -> None:
        """One floor convention, shared — the layers cannot drift apart."""
        box = BoundingBox(12.5, 33.0, 91.5, 210.0)
        assert bottom_center_anchor(box) == bottom_center(box)

    def test_handles_a_degenerate_box(self) -> None:
        assert bottom_center_anchor(BoundingBox(5.0, 5.0, 5.0, 5.0)) == Point(5.0, 5.0)


class TestNormalizedFrameProjector:
    def test_divides_by_frame_size(self) -> None:
        projected = NormalizedFrameProjector(640, 360).project(Point(320.0, 90.0))
        assert (projected.x, projected.y) == pytest.approx((0.5, 0.25))

    def test_reports_its_space(self) -> None:
        projector = NormalizedFrameProjector(640, 360)
        assert projector.space is NORMALIZED
        assert projector.project(Point(1.0, 1.0)).space is NORMALIZED

    def test_satisfies_the_projector_protocol(self) -> None:
        assert isinstance(NormalizedFrameProjector(640, 360), MotionProjector)

    @pytest.mark.parametrize(
        ("width", "height"),
        [(320, 240), (384, 288), (640, 360), (1280, 720), (1920, 1080), (3840, 2160)],
    )
    def test_the_same_relative_position_projects_identically(self, width: int, height: int) -> None:
        """Resolution independence, across six resolutions."""
        pixel = Point(0.3 * width, 0.7 * height)
        projected = NormalizedFrameProjector(width, height).project(pixel)
        assert (projected.x, projected.y) == pytest.approx((0.3, 0.7), abs=1e-12)

    def test_is_deterministic(self) -> None:
        projector = NormalizedFrameProjector(1920, 1080)
        point = Point(613.0, 927.0)
        assert projector.project(point) == projector.project(point)

    def test_does_not_clamp_out_of_frame_points(self) -> None:
        projected = NormalizedFrameProjector(100, 100).project(Point(-5.0, 130.0))
        assert (projected.x, projected.y) == pytest.approx((-0.05, 1.3))

    @pytest.mark.parametrize(
        ("width", "height"), [(0, 100), (100, 0), (-10, 10), (10, -10), (0, 0)]
    )
    def test_rejects_invalid_frame_dimensions(self, width: int, height: int) -> None:
        with pytest.raises(ValueError, match="Frame size must be positive"):
            NormalizedFrameProjector(width, height)

    def test_anchor_and_projection_compose(self) -> None:
        """A box whose feet are dead centre projects to (0.5, 0.5)."""
        box = BoundingBox(280.0, 100.0, 360.0, 180.0)
        projected = NormalizedFrameProjector(640, 360).project(bottom_center_anchor(box))
        assert (projected.x, projected.y) == pytest.approx((0.5, 0.5))

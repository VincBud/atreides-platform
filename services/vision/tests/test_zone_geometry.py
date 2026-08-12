"""Zone geometry: floor position, normalization, and point-in-polygon."""

from __future__ import annotations

import pytest

from atreides_vision.models import BoundingBox, Point, Track
from atreides_vision.zones.geometry import (
    NormalizedPoint,
    bottom_center,
    point_in_polygon,
    to_normalized,
    track_floor_point,
)

# A unit square, used for most boundary tests.
UNIT_SQUARE = (
    NormalizedPoint(0.0, 0.0),
    NormalizedPoint(1.0, 0.0),
    NormalizedPoint(1.0, 1.0),
    NormalizedPoint(0.0, 1.0),
)


def square(x0: float, y0: float, x1: float, y1: float) -> tuple[NormalizedPoint, ...]:
    return (
        NormalizedPoint(x0, y0),
        NormalizedPoint(x1, y0),
        NormalizedPoint(x1, y1),
        NormalizedPoint(x0, y1),
    )


class TestNormalizedPoint:
    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            NormalizedPoint(float("nan"), 0.5)

    def test_allows_values_outside_the_frame(self) -> None:
        """A person half out of shot has a floor point past the frame edge."""
        point = NormalizedPoint(-0.05, 1.02)
        assert point.x == -0.05

    def test_round_trips_to_pixels(self) -> None:
        assert NormalizedPoint(0.5, 0.25).to_pixels(640, 360) == Point(320.0, 90.0)

    def test_distance(self) -> None:
        assert NormalizedPoint(0.0, 0.0).distance_to(NormalizedPoint(0.3, 0.4)) == pytest.approx(
            0.5
        )


class TestBottomCenter:
    def test_is_the_middle_of_the_bottom_edge(self) -> None:
        """Floor position, not box centre — the feet decide the zone."""
        box = BoundingBox(100.0, 40.0, 200.0, 240.0)
        assert bottom_center(box) == Point(150.0, 240.0)
        assert bottom_center(box) != box.center

    def test_matches_a_degenerate_box(self) -> None:
        assert bottom_center(BoundingBox(5.0, 5.0, 5.0, 5.0)) == Point(5.0, 5.0)


class TestToNormalized:
    def test_divides_by_frame_size(self) -> None:
        assert to_normalized(Point(320.0, 90.0), 640, 360) == NormalizedPoint(0.5, 0.25)

    def test_is_resolution_independent(self) -> None:
        """The same relative position yields the same normalized point."""
        small = to_normalized(Point(160.0, 90.0), 320, 180)
        large = to_normalized(Point(960.0, 540.0), 1920, 1080)
        assert small == large == NormalizedPoint(0.5, 0.5)

    @pytest.mark.parametrize(("width", "height"), [(0, 100), (100, 0), (-10, 10)])
    def test_rejects_invalid_frame_size(self, width: int, height: int) -> None:
        with pytest.raises(ValueError, match="Frame size must be positive"):
            to_normalized(Point(1.0, 1.0), width, height)


class TestTrackFloorPoint:
    def test_combines_bottom_center_and_normalization(self) -> None:
        track = Track(track_id=1, bbox_xyxy=BoundingBox(280.0, 100.0, 360.0, 180.0))
        assert track_floor_point(track, 640, 360) == NormalizedPoint(0.5, 0.5)

    def test_two_resolutions_agree(self) -> None:
        hd = Track(track_id=1, bbox_xyxy=BoundingBox(940.0, 500.0, 980.0, 540.0))
        sd = Track(track_id=1, bbox_xyxy=BoundingBox(313.33, 166.67, 326.67, 180.0))
        assert track_floor_point(hd, 1920, 1080).x == pytest.approx(
            track_floor_point(sd, 640, 360).x, abs=1e-3
        )


class TestPointInPolygonInside:
    def test_centre_of_a_square(self) -> None:
        assert point_in_polygon(NormalizedPoint(0.5, 0.5), UNIT_SQUARE)

    def test_just_inside_each_edge(self) -> None:
        for point in (
            NormalizedPoint(0.001, 0.5),
            NormalizedPoint(0.999, 0.5),
            NormalizedPoint(0.5, 0.001),
            NormalizedPoint(0.5, 0.999),
        ):
            assert point_in_polygon(point, UNIT_SQUARE), point

    def test_inside_a_trapezoid(self) -> None:
        """Perspective floor shapes are not rectangles."""
        trapezoid = (
            NormalizedPoint(0.3, 0.3),
            NormalizedPoint(0.7, 0.3),
            NormalizedPoint(0.9, 0.9),
            NormalizedPoint(0.1, 0.9),
        )
        assert point_in_polygon(NormalizedPoint(0.5, 0.8), trapezoid)
        assert not point_in_polygon(NormalizedPoint(0.15, 0.4), trapezoid)

    def test_concave_polygon_notch_is_outside(self) -> None:
        # An L-shape; the notch in the top-right must not count as inside.
        l_shape = (
            NormalizedPoint(0.0, 0.0),
            NormalizedPoint(0.5, 0.0),
            NormalizedPoint(0.5, 0.5),
            NormalizedPoint(1.0, 0.5),
            NormalizedPoint(1.0, 1.0),
            NormalizedPoint(0.0, 1.0),
        )
        assert point_in_polygon(NormalizedPoint(0.25, 0.25), l_shape)
        assert point_in_polygon(NormalizedPoint(0.75, 0.75), l_shape)
        assert not point_in_polygon(NormalizedPoint(0.75, 0.25), l_shape)


class TestPointInPolygonOutside:
    @pytest.mark.parametrize(
        "point",
        [
            NormalizedPoint(-0.1, 0.5),
            NormalizedPoint(1.1, 0.5),
            NormalizedPoint(0.5, -0.1),
            NormalizedPoint(0.5, 1.1),
            NormalizedPoint(2.0, 2.0),
        ],
    )
    def test_outside_the_square(self, point: NormalizedPoint) -> None:
        assert not point_in_polygon(point, UNIT_SQUARE)

    def test_outside_a_small_zone_but_inside_the_frame(self) -> None:
        zone = square(0.2, 0.2, 0.4, 0.4)
        assert not point_in_polygon(NormalizedPoint(0.5, 0.5), zone)


class TestPointInPolygonBoundary:
    """Boundary is inside — a documented, deliberate choice."""

    @pytest.mark.parametrize(
        "point",
        [
            NormalizedPoint(0.5, 0.0),  # bottom edge
            NormalizedPoint(0.5, 1.0),  # top edge
            NormalizedPoint(0.0, 0.5),  # left edge
            NormalizedPoint(1.0, 0.5),  # right edge
        ],
    )
    def test_point_on_edge_is_inside(self, point: NormalizedPoint) -> None:
        assert point_in_polygon(point, UNIT_SQUARE)

    @pytest.mark.parametrize("vertex", UNIT_SQUARE)
    def test_point_on_vertex_is_inside(self, vertex: NormalizedPoint) -> None:
        assert point_in_polygon(vertex, UNIT_SQUARE)

    def test_point_on_a_diagonal_edge_is_inside(self) -> None:
        triangle = (
            NormalizedPoint(0.0, 0.0),
            NormalizedPoint(1.0, 1.0),
            NormalizedPoint(0.0, 1.0),
        )
        assert point_in_polygon(NormalizedPoint(0.5, 0.5), triangle)

    def test_shared_edge_belongs_to_both_neighbours(self) -> None:
        """Adjacent zones both claim someone standing on the shared line."""
        left = square(0.0, 0.0, 0.5, 1.0)
        right = square(0.5, 0.0, 1.0, 1.0)
        on_the_line = NormalizedPoint(0.5, 0.5)
        assert point_in_polygon(on_the_line, left)
        assert point_in_polygon(on_the_line, right)


class TestPointInPolygonValidation:
    @pytest.mark.parametrize("count", [0, 1, 2])
    def test_rejects_degenerate_polygons(self, count: int) -> None:
        with pytest.raises(ValueError, match="at least 3 vertices"):
            point_in_polygon(NormalizedPoint(0.5, 0.5), UNIT_SQUARE[:count])

    def test_vertex_order_does_not_matter(self) -> None:
        clockwise = UNIT_SQUARE
        counter_clockwise = tuple(reversed(UNIT_SQUARE))
        point = NormalizedPoint(0.5, 0.5)
        assert point_in_polygon(point, clockwise) == point_in_polygon(point, counter_clockwise)

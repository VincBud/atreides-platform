"""Geometric primitives for zone analysis.

Two conventions matter here and are used consistently everywhere downstream.

**Floor position.** A person's position in a zone is the **bottom-centre** of
their bounding box, not the box centre. For a camera looking across a venue
floor, the bottom edge of the box is roughly where the person's feet are, and
that is what decides which part of the floor they are standing on. The box
centre sits around chest height and drifts into the zone behind them.

**Normalized coordinates.** Positions are divided by the frame size, so a zone
configuration written against one camera works at any resolution:
``x = pixel_x / frame_width``, ``y = pixel_y / frame_height``.

This module owns no state and imports nothing from the rest of the zone layer,
which keeps it trivially testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from atreides_vision.models import BoundingBox, Point, Track

__all__ = [
    "NormalizedPoint",
    "bottom_center",
    "point_in_polygon",
    "to_normalized",
    "track_floor_point",
]

#: Tolerance for "the point lies on this edge", in normalized units.
#: 1e-9 of a frame width is far below one pixel at any realistic resolution.
BOUNDARY_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class NormalizedPoint:
    """A point in frame-relative coordinates, origin top-left.

    Inside the frame both components lie in ``[0, 1]``. Values slightly outside
    that range are permitted and meaningful: a person standing half out of shot
    has a bounding box that extends past the frame edge, and clamping their
    position inward would silently place them in a zone they are not in.
    Zone polygon vertices are separately required to be within ``[0, 1]`` —
    that is a configuration constraint, not a geometric one.
    """

    x: float
    y: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.x) and np.isfinite(self.y)):
            raise ValueError(f"NormalizedPoint must be finite, got ({self.x!r}, {self.y!r})")

    def to_pixels(self, frame_width: int, frame_height: int) -> Point:
        """Project back into pixel coordinates for drawing."""
        _validate_frame_size(frame_width, frame_height)
        return Point(self.x * frame_width, self.y * frame_height)

    def distance_to(self, other: NormalizedPoint) -> float:
        """Euclidean distance in normalized units."""
        return float(np.hypot(self.x - other.x, self.y - other.y))


def bottom_center(bbox: BoundingBox) -> Point:
    """The bottom-centre of a box, in pixels — the approximate floor position.

    Delegates to :attr:`~atreides_vision.models.BoundingBox.bottom_center` so
    the floor-position convention has exactly one definition, shared with the
    behaviour layer.
    """
    return bbox.bottom_center


def to_normalized(point: Point, frame_width: int, frame_height: int) -> NormalizedPoint:
    """Convert a pixel position into frame-relative coordinates."""
    _validate_frame_size(frame_width, frame_height)
    return NormalizedPoint(point.x / frame_width, point.y / frame_height)


def track_floor_point(track: Track, frame_width: int, frame_height: int) -> NormalizedPoint:
    """The normalized floor position of a tracked person.

    This is the single function the rest of the zone layer uses to turn a
    :class:`~atreides_vision.models.Track` into a position, so the convention
    is defined in exactly one place.
    """
    return to_normalized(bottom_center(track.bbox_xyxy), frame_width, frame_height)


def point_in_polygon(
    point: NormalizedPoint,
    vertices: Sequence[NormalizedPoint],
    *,
    epsilon: float = BOUNDARY_EPSILON,
) -> bool:
    """Test whether ``point`` lies inside the polygon described by ``vertices``.

    Uses even-odd ray casting, with an explicit boundary test first.

    Boundary behaviour is defined, not incidental:

    * strictly inside -> ``True``
    * strictly outside -> ``False``
    * exactly on an edge -> ``True``
    * exactly on a vertex -> ``True``

    Treating the boundary as inside means adjacent zones sharing an edge both
    claim a person standing on it, which is preferable to that person briefly
    belonging to nothing.

    Implemented directly rather than via a geometry library: it is ~20 lines,
    the polygons are small, and it lets the boundary contract above be pinned
    down by tests instead of inherited from a third party.

    Args:
        point: The position to test.
        vertices: Polygon vertices in order; the polygon is implicitly closed.
        epsilon: Tolerance for the on-edge test, in normalized units.

    Raises:
        ValueError: if fewer than three vertices are supplied.
    """
    count = len(vertices)
    if count < 3:
        raise ValueError(f"A polygon needs at least 3 vertices, got {count}")

    px, py = point.x, point.y
    inside = False

    for index in range(count):
        current = vertices[index]
        previous = vertices[index - 1]  # index -1 wraps to the last vertex

        if _on_segment(px, py, previous, current, epsilon):
            return True

        # Even-odd rule: count edges crossed by a ray travelling in -x.
        # The guard ensures current.y != previous.y, so the division is safe.
        if (current.y > py) != (previous.y > py):
            x_at_py = (previous.x - current.x) * (py - current.y) / (
                previous.y - current.y
            ) + current.x
            if px < x_at_py:
                inside = not inside

    return inside


def _on_segment(
    px: float,
    py: float,
    start: NormalizedPoint,
    end: NormalizedPoint,
    epsilon: float,
) -> bool:
    """Whether ``(px, py)`` lies on the closed segment ``start -> end``."""
    cross = (end.x - start.x) * (py - start.y) - (end.y - start.y) * (px - start.x)
    if abs(cross) > epsilon:
        return False
    # Collinear with the edge; now check it falls within the segment's extent.
    return (
        min(start.x, end.x) - epsilon <= px <= max(start.x, end.x) + epsilon
        and min(start.y, end.y) - epsilon <= py <= max(start.y, end.y) + epsilon
    )


def _validate_frame_size(frame_width: int, frame_height: int) -> None:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"Frame size must be positive, got {frame_width}x{frame_height}")

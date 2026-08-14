"""The coordinate space behaviour signals are measured in.

Why this abstraction exists
---------------------------
Every kinematic signal in this package is a distance divided by a time. The
*time* is unambiguous — it comes from the source frame rate. The *distance* is
not: a person's box moves a different number of pixels per real step depending
on the camera's resolution, mounting height, lens and how far away they are.
Raw pixel velocity is therefore meaningless outside the one camera and one
resolution that produced it.

So no algorithm in this package ever sees a pixel. A :class:`MotionProjector`
converts a person's image-space anchor into a :class:`MotionPoint`, which
carries the space it belongs to, and the kinematics operate purely on those.
Swapping the projector changes what "distance" means without changing a single
line of :mod:`atreides_vision.behavior.kinematics`.

Two spaces are named. Only one is implemented.

``NORMALIZED_IMAGE``
    Implemented here by :class:`NormalizedFrameProjector`. Position divided by
    frame size, so the same physical scene gives the same numbers at 384x288
    and at 1920x1080. **This is camera-local image-plane motion.** It is not
    metres, it does not convert to metres, and two arbitrary cameras' values
    are not comparable: someone walking past a near camera crosses far more of
    the frame per step than the same person at the back of a wide shot.

``GROUND_PLANE_METRES``
    Reserved for a future calibrated projector (a homography from four floor
    reference points, per camera). Nothing in V0.3.0 produces it. It is named
    now because it is what makes the design claim testable: a calibrated
    projector is a new class implementing :class:`MotionProjector`, and every
    signal downstream then reports metres per second with no other change.

Venue independence
------------------
Nothing here encodes an assumed venue size, camera coverage area, or floor
layout. A camera covering 10 m² and one covering 200 m² are handled by the
same code; what differs is the projector (and, once calibrated, the numbers it
returns). Fixed-area assumptions must never be reintroduced.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from atreides_vision.models import BoundingBox, Point

__all__ = [
    "MotionPoint",
    "MotionProjector",
    "MotionSpace",
    "NormalizedFrameProjector",
    "PersonAnchor",
    "bottom_center_anchor",
]


class MotionSpace(StrEnum):
    """The coordinate space a :class:`MotionPoint` and its derived signals live in.

    A signal is only interpretable together with its space, which is why every
    :class:`~atreides_vision.behavior.models.BehaviorSignal` carries one.
    """

    #: Frame-relative coordinates: ``x / frame_width``, ``y / frame_height``.
    #: Camera-local and resolution-independent. Not physical distance.
    NORMALIZED_IMAGE = "normalized_image"

    #: Reserved. Calibrated floor-plane metres. No projector produces this in
    #: V0.3.0; see the module docstring.
    GROUND_PLANE_METRES = "ground_plane_metres"

    @property
    def distance_unit(self) -> str:
        """Name of one unit of distance in this space, used to label signals."""
        return _DISTANCE_UNITS[self]

    @property
    def is_physical(self) -> bool:
        """Whether distances in this space are real-world lengths.

        ``False`` for image-plane spaces. Callers that need physical units —
        a future risk layer comparing cameras, for instance — must check this
        rather than assuming.
        """
        return self is MotionSpace.GROUND_PLANE_METRES


_DISTANCE_UNITS: Final[Mapping[MotionSpace, str]] = {
    MotionSpace.NORMALIZED_IMAGE: "normalized_frame_unit",
    MotionSpace.GROUND_PLANE_METRES: "m",
}


@dataclass(frozen=True, slots=True)
class MotionPoint:
    """A position in a named motion-analysis coordinate space.

    Values are *not* clamped. A person standing half out of shot has an anchor
    past the frame edge, and moving them inside would fabricate motion.
    """

    x: float
    y: float
    space: MotionSpace

    def __post_init__(self) -> None:
        if not (math.isfinite(self.x) and math.isfinite(self.y)):
            raise ValueError(f"MotionPoint must be finite, got ({self.x!r}, {self.y!r})")

    def vector_to(self, other: MotionPoint) -> tuple[float, float]:
        """The displacement ``(dx, dy)`` from this point to ``other``."""
        self._require_same_space(other)
        return (other.x - self.x, other.y - self.y)

    def distance_to(self, other: MotionPoint) -> float:
        """Euclidean distance, in this space's distance unit."""
        self._require_same_space(other)
        return math.hypot(other.x - self.x, other.y - self.y)

    def _require_same_space(self, other: MotionPoint) -> None:
        """Guard against mixing spaces, which would silently produce nonsense.

        This is the check that makes a future calibrated projector safe to
        introduce alongside this one: normalized and metric points can coexist
        in a process, but they can never be subtracted from each other.
        """
        if self.space is not other.space:
            raise ValueError(
                "Cannot combine motion points from different coordinate spaces: "
                f"{self.space.value} and {other.space.value}"
            )


#: How a person's bounding box becomes the single image-space point that
#: represents them. Injectable so an overhead or fisheye camera can supply a
#: different convention later without touching the analyzer.
PersonAnchor = Callable[[BoundingBox], Point]


def bottom_center_anchor(bbox: BoundingBox) -> Point:
    """The default person anchor: the middle of the box's bottom edge.

    This is the same floor-position convention the zone layer uses, and for
    the same reason — it is roughly where the feet are, whereas the box centre
    sits at chest height and drifts backwards into the scene. Both call
    :attr:`~atreides_vision.models.BoundingBox.bottom_center`, so the two
    layers cannot drift apart.
    """
    return bbox.bottom_center


@runtime_checkable
class MotionProjector(Protocol):
    """Converts an image-space anchor into a motion-analysis coordinate.

    The whole point of the interface: the kinematics never learn where the
    numbers came from. Implementations must be deterministic and stateless
    with respect to time — the same input point must always give the same
    output — because signal reproducibility depends on it.
    """

    @property
    def space(self) -> MotionSpace:
        """The space this projector emits. Fixed for the projector's lifetime."""
        ...

    def project(self, point: Point) -> MotionPoint:
        """Project one image-space (pixel) point into :attr:`space`."""
        ...


@dataclass(frozen=True, slots=True)
class NormalizedFrameProjector:
    """Projects pixels into frame-relative coordinates. The V0.3.0 default.

    ``x_norm = x / frame_width``, ``y_norm = y / frame_height``, origin
    top-left. Deterministic, resolution-independent, and dependency-free — no
    OpenCV, no NumPy, no calibration data.

    Resolution independence is the property that matters and is tested
    directly: the same real trajectory filmed at 384x288 and at 1920x1080
    produces the same normalized positions, hence the same speed,
    acceleration and heading signals.

    What it does **not** give you is physical distance. Read
    :attr:`MotionSpace.NORMALIZED_IMAGE`'s caveats before comparing values
    across cameras, or before calling anything measured here a "speed" in the
    everyday sense.

    Raises:
        ValueError: if either frame dimension is not positive.
    """

    frame_width: int
    frame_height: int

    def __post_init__(self) -> None:
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError(
                f"Frame size must be positive, got {self.frame_width}x{self.frame_height}"
            )

    @property
    def space(self) -> MotionSpace:
        return MotionSpace.NORMALIZED_IMAGE

    def project(self, point: Point) -> MotionPoint:
        return MotionPoint(
            x=point.x / self.frame_width,
            y=point.y / self.frame_height,
            space=MotionSpace.NORMALIZED_IMAGE,
        )

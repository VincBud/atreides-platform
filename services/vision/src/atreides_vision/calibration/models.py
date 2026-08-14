"""Domain models for ground-plane calibration.

A calibration answers one question per camera: *where on the venue floor is
this pixel?* It is supplied by an operator as a handful of correspondences —
"this point in the image is that point on the floor" — and everything else is
derived from them.

The venue coordinate system
---------------------------
Venue coordinates are metres in a **venue-local frame shared by every camera**,
not a per-camera metric frame. The operator picks one origin and one axis
orientation for the venue and expresses every camera's reference points in it.
Two cameras calibrated against the same venue frame therefore project into the
same coordinates, and a person standing at the same spot appears at roughly the
same ``(x, y)`` in both — which is what makes any future spatial reasoning
possible.

:attr:`GroundPlaneCalibration.venue_id` records which frame was used, so two
calibrations can be checked for compatibility rather than merely assumed to
agree. Nothing here assumes a venue size, shape, or camera coverage area; a
camera covering a doorway and one covering a festival field differ only in the
numbers their correspondences contain.

**Spatial agreement is not identity.** Two cameras reporting a person at
``(12.4, 7.1)`` and ``(12.6, 7.0)`` are two independent observations that
happen to be near each other. V0.3.1 draws no conclusion from that.

The planar assumption
---------------------
One homography describes one plane. Every correspondence must lie on the same
approximately flat floor, and so must the people being measured. Stairs,
balconies, raised platforms, stages and multiple storeys all break it — a
person on a step is measured as though standing on the floor behind it.
:attr:`GroundPlaneCalibration.surface_id` names the surface a calibration
describes, so a future milestone can hold several per camera; V0.3.1 supports
exactly one.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from atreides_vision.calibration.homography import MIN_CORRESPONDENCES
from atreides_vision.errors import CalibrationError
from atreides_vision.models import Point

__all__ = [
    "CalibrationQuality",
    "GroundPlaneCalibration",
    "PointCorrespondence",
    "ReferenceFrame",
    "VenuePoint",
]

#: Two calibration points closer together than this, in their own units, are
#: treated as duplicates. Duplicated correspondences add no information and
#: push the estimate towards degeneracy.
DUPLICATE_TOLERANCE = 1e-6

#: Relative area below which three points count as collinear. Scaled by the
#: spread of the points themselves, so it means the same thing for a 2 m
#: doorway and a 60 m festival field.
COLLINEARITY_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class VenuePoint:
    """A position on the venue floor, in metres, in the venue-local frame.

    Not clamped to any extent: venues have no fixed size, and a camera may see
    past whatever region happened to be used for calibration. Coordinates may
    be negative — the origin is wherever the operator put it.
    """

    x: float
    y: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.x) and math.isfinite(self.y)):
            raise CalibrationError(
                f"Venue coordinates must be finite metres, got ({self.x!r}, {self.y!r})"
            )

    def distance_to(self, other: VenuePoint) -> float:
        """Euclidean distance in metres."""
        return math.hypot(other.x - self.x, other.y - self.y)

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class ReferenceFrame:
    """The image dimensions a calibration's pixel coordinates were authored at.

    Recorded because pixel coordinates are meaningless without it. The same
    camera decoded at half resolution puts the same physical point at half the
    pixel coordinates, and a calibration that did not know its own reference
    size would silently produce positions wrong by a factor of two. See
    :class:`~atreides_vision.calibration.projector.GroundPlaneProjector`, which
    rescales runtime frames back into these dimensions.
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise CalibrationError(
                f"Calibration reference frame must be positive, got {self.width}x{self.height}"
            )

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True, slots=True)
class PointCorrespondence:
    """One "this pixel is that floor position" pair.

    ``image`` is in the calibration's :class:`ReferenceFrame` pixels; ``venue``
    is in venue-local metres.
    """

    image: Point
    venue: VenuePoint

    def __post_init__(self) -> None:
        if not (math.isfinite(self.image.x) and math.isfinite(self.image.y)):
            raise CalibrationError(
                f"Image coordinates must be finite pixels, got ({self.image.x!r}, {self.image.y!r})"
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, position: int) -> Self:
        """Build one correspondence from a JSON object."""
        if not isinstance(data, Mapping):
            raise CalibrationError(
                f"correspondences[{position}] must be an object, got {type(data).__name__}"
            )
        for field in ("image", "venue"):
            if field not in data:
                raise CalibrationError(
                    f"correspondences[{position}] is missing required field '{field}'"
                )
        image = _coordinate_pair(data["image"], f"correspondences[{position}].image")
        venue = _coordinate_pair(data["venue"], f"correspondences[{position}].venue")
        return cls(image=Point(image[0], image[1]), venue=VenuePoint(venue[0], venue[1]))

    def to_record(self) -> dict[str, Any]:
        return {"image": [self.image.x, self.image.y], "venue": [self.venue.x, self.venue.y]}


@dataclass(frozen=True, slots=True)
class GroundPlaneCalibration:
    """One camera's mapping from image pixels onto the venue floor plane.

    Validated at construction for everything that can be checked from the
    points alone. Whether they actually yield a usable transform is decided by
    :func:`~atreides_vision.calibration.fitting.fit_calibration`, because that
    requires solving for the homography.

    Raises:
        CalibrationError: if identifiers are empty, there are fewer than four
            correspondences, any coordinate is non-finite, points are
            duplicated, or the layout is collinear.
    """

    camera_id: str
    #: Which venue coordinate frame the metre values are expressed in. Two
    #: calibrations sharing this string are directly comparable; two that do
    #: not are unrelated coordinate systems that happen to both use metres.
    venue_id: str
    reference_frame: ReferenceFrame
    correspondences: tuple[PointCorrespondence, ...]
    #: Names the planar surface this calibration describes. V0.3.1 supports one
    #: per camera; the field exists so a future multi-surface milestone does not
    #: have to redefine the config format.
    surface_id: str = "ground"
    #: Free-text note for whoever maintains the calibration — how the points
    #: were measured, when, by whom.
    description: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("camera_id", self.camera_id),
            ("venue_id", self.venue_id),
            ("surface_id", self.surface_id),
        ):
            if not value or not value.strip():
                raise CalibrationError(f"Calibration {name} must be a non-empty string")

        count = len(self.correspondences)
        if count < MIN_CORRESPONDENCES:
            raise CalibrationError(
                f"Camera '{self.camera_id}' has {count} calibration correspondence(s); "
                f"at least {MIN_CORRESPONDENCES} are required to solve a planar "
                "homography. Add more reference points spread across the floor."
            )

        self._reject_duplicates()
        self._reject_collinear()

    def _reject_duplicates(self) -> None:
        """Duplicate points on either side carry no information and destabilise the fit."""
        for label, points in (
            ("image", [(c.image.x, c.image.y) for c in self.correspondences]),
            ("venue", [(c.venue.x, c.venue.y) for c in self.correspondences]),
        ):
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    if math.dist(points[i], points[j]) <= DUPLICATE_TOLERANCE:
                        raise CalibrationError(
                            f"Camera '{self.camera_id}' correspondences {i} and {j} have the "
                            f"same {label} position {points[i]}. Each reference point must be "
                            "a distinct location."
                        )

    def _reject_collinear(self) -> None:
        """Reject layouts that cannot determine a homography.

        With exactly four points, any three on a line is fatal — the remaining
        freedom is not enough to pin down eight parameters. With more points a
        few collinear triples are harmless, so only a wholly collinear set is
        rejected and the finer conditioning check happens during estimation.
        """
        for label, points in (
            ("image", [(c.image.x, c.image.y) for c in self.correspondences]),
            ("venue", [(c.venue.x, c.venue.y) for c in self.correspondences]),
        ):
            scale = _point_spread(points)
            if scale <= 0:
                raise CalibrationError(
                    f"Camera '{self.camera_id}' {label} calibration points are all identical."
                )

            if len(points) == MIN_CORRESPONDENCES:
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        for k in range(j + 1, len(points)):
                            if _is_collinear(points[i], points[j], points[k], scale):
                                raise CalibrationError(
                                    f"Camera '{self.camera_id}' {label} calibration points "
                                    f"{i}, {j} and {k} lie on a straight line. With exactly "
                                    f"{MIN_CORRESPONDENCES} points, no three may be collinear "
                                    "— pick points spread over the floor area, such as the "
                                    "corners of a rectangle."
                                )
            elif _all_collinear(points, scale):
                raise CalibrationError(
                    f"Camera '{self.camera_id}' {label} calibration points all lie on a "
                    "straight line. They must span an area, not an edge."
                )

    @property
    def point_count(self) -> int:
        return len(self.correspondences)

    @property
    def is_overdetermined(self) -> bool:
        """Whether there are more points than the minimum, enabling a best fit."""
        return self.point_count > MIN_CORRESPONDENCES

    def image_points(self) -> np.ndarray:
        """``(N, 2)`` pixel coordinates in reference-frame dimensions."""
        return np.array([(c.image.x, c.image.y) for c in self.correspondences], dtype=np.float64)

    def venue_points(self) -> np.ndarray:
        """``(N, 2)`` venue coordinates in metres."""
        return np.array([(c.venue.x, c.venue.y) for c in self.correspondences], dtype=np.float64)

    def venue_extent_metres(self) -> float:
        """Diagonal of the calibrated region's bounding box, in metres.

        Used to express reprojection error relative to the area actually
        calibrated, so no check anywhere depends on an assumed venue size.
        """
        points = self.venue_points()
        span = points.max(axis=0) - points.min(axis=0)
        return float(math.hypot(span[0], span[1]))

    def covers_same_venue_as(self, other: GroundPlaneCalibration) -> bool:
        """Whether both calibrations express metres in the same venue frame.

        The precondition for comparing their outputs at all. It says nothing
        about whether the cameras' views overlap, and nothing whatsoever about
        the people in them.
        """
        return self.venue_id == other.venue_id and self.surface_id == other.surface_id

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Self:
        """Build a calibration from parsed JSON, naming the offending field on error."""
        if not isinstance(data, Mapping):
            raise CalibrationError(f"Calibration must be a JSON object, got {type(data).__name__}")
        for field in ("camera_id", "venue_id", "reference_frame", "correspondences"):
            if field not in data:
                raise CalibrationError(f"Calibration is missing required field '{field}'")

        camera_id = _require_string(data["camera_id"], "camera_id")
        venue_id = _require_string(data["venue_id"], "venue_id")
        surface_id = (
            "ground"
            if "surface_id" not in data
            else _require_string(data["surface_id"], "surface_id")
        )

        description = data.get("description")
        if description is not None and not isinstance(description, str):
            raise CalibrationError(f"description must be a string, got {description!r}")

        reference_frame = _reference_frame_from_mapping(data["reference_frame"])

        raw = data["correspondences"]
        if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
            raise CalibrationError("'correspondences' must be an array of objects")

        correspondences = tuple(
            PointCorrespondence.from_mapping(entry, position=position)
            for position, entry in enumerate(raw)
        )
        return cls(
            camera_id=camera_id,
            venue_id=venue_id,
            reference_frame=reference_frame,
            correspondences=correspondences,
            surface_id=surface_id,
            description=description,
        )

    def to_record(self) -> dict[str, Any]:
        """Round-trip back to the JSON representation."""
        record: dict[str, Any] = {
            "camera_id": self.camera_id,
            "venue_id": self.venue_id,
            "surface_id": self.surface_id,
            "reference_frame": {
                "width": self.reference_frame.width,
                "height": self.reference_frame.height,
            },
            "correspondences": [c.to_record() for c in self.correspondences],
        }
        if self.description is not None:
            record["description"] = self.description
        return record

    def __iter__(self) -> Iterator[PointCorrespondence]:
        return iter(self.correspondences)

    def __len__(self) -> int:
        return len(self.correspondences)


@dataclass(frozen=True, slots=True)
class CalibrationQuality:
    """How well a fitted homography reproduces the points it was built from.

    **The primary metric is venue-space error in metres**
    (:attr:`mean_venue_error_m`). It is the one that states the consequence
    directly: "positions from this camera are about this far off". Pixel error
    is reported alongside because it is the classical homography diagnostic,
    but it is resolution-dependent and so cannot be compared between cameras.

    Reprojection error measures **self-consistency, not accuracy**. It says how
    well the fit reproduces the operator's own points. If those points were
    measured wrongly on the floor, a perfect reprojection error simply means
    the calibration faithfully reproduces the wrong answer. Nothing here can
    detect that.

    There are deliberately no "good" and "bad" bands, and no acceptance
    threshold. What counts as acceptable depends on the camera, the venue and
    what the numbers are used for, and inventing a cutoff would make an
    engineering judgement look like a measured fact.
    """

    point_count: int
    #: Mean distance, in metres, between each supplied venue point and where
    #: the fitted homography projects its image point.
    mean_venue_error_m: float
    max_venue_error_m: float
    #: The same disagreement measured in reference-frame pixels, by projecting
    #: venue points back through the inverse homography.
    mean_image_error_px: float
    max_image_error_px: float
    #: Diagonal of the calibrated region, for scale-free interpretation of the
    #: errors above.
    venue_extent_m: float

    @property
    def relative_error(self) -> float:
        """Mean venue error as a fraction of the calibrated region's diagonal.

        Scale-free, so it means the same thing for a doorway and a field.
        ``0.0`` when the extent is degenerate.
        """
        if self.venue_extent_m <= 0:
            return 0.0
        return self.mean_venue_error_m / self.venue_extent_m

    @property
    def is_exact(self) -> bool:
        """Whether the fit reproduces its points to floating-point precision.

        True for an exactly-determined four-point calibration, and for
        overdetermined sets whose points are perfectly consistent.
        """
        return self.max_venue_error_m < 1e-9

    def to_record(self) -> dict[str, Any]:
        return {
            "point_count": self.point_count,
            "mean_venue_error_m": round(self.mean_venue_error_m, 6),
            "max_venue_error_m": round(self.max_venue_error_m, 6),
            "mean_image_error_px": round(self.mean_image_error_px, 4),
            "max_image_error_px": round(self.max_image_error_px, 4),
            "venue_extent_m": round(self.venue_extent_m, 4),
            "relative_error": round(self.relative_error, 6),
        }


def _point_spread(points: Sequence[tuple[float, float]]) -> float:
    """Largest pairwise distance, used to scale geometric tolerances."""
    return max(
        (math.dist(a, b) for index, a in enumerate(points) for b in points[index + 1 :]),
        default=0.0,
    )


def _is_collinear(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    scale: float,
) -> bool:
    """Whether three points lie on a line, relative to the set's own scale."""
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return abs(cross) <= COLLINEARITY_TOLERANCE * scale * scale


def _all_collinear(points: Sequence[tuple[float, float]], scale: float) -> bool:
    first, second = points[0], points[1]
    return all(_is_collinear(first, second, other, scale) for other in points[2:])


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise CalibrationError(f"{field} must be a string, got {value!r}")
    return value


def _reference_frame_from_mapping(data: object) -> ReferenceFrame:
    if not isinstance(data, Mapping):
        raise CalibrationError(
            f"reference_frame must be an object with width and height, got {data!r}"
        )
    for field in ("width", "height"):
        if field not in data:
            raise CalibrationError(f"reference_frame is missing required field '{field}'")
    width, height = data["width"], data["height"]
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise CalibrationError(
                f"reference_frame.{name} must be an integer number of pixels, got {value!r}"
            )
    return ReferenceFrame(width=int(width), height=int(height))


def _coordinate_pair(value: object, field: str) -> tuple[float, float]:
    """Parse a ``[x, y]`` JSON pair into floats."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise CalibrationError(f"{field} must be an [x, y] pair, got {value!r}")
    values = tuple(value)
    if len(values) != 2:
        raise CalibrationError(f"{field} must have exactly 2 values, got {len(values)}")
    parsed = []
    for entry in values:
        if isinstance(entry, bool) or not isinstance(entry, int | float):
            raise CalibrationError(f"{field} must contain two numbers, got {value!r}")
        number = float(entry)
        if not math.isfinite(number):
            raise CalibrationError(f"{field} must contain finite numbers, got {value!r}")
        parsed.append(number)
    return (parsed[0], parsed[1])

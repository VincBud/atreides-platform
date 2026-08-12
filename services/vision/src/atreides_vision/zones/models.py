"""Atreides-owned domain models for zones and their metrics.

Zone configuration arrives as JSON, but a raw ``dict`` never travels further
than :meth:`ZoneLayout.from_mapping`. Everything downstream works with these
validated, immutable types.

Privacy note
------------
Nothing here records who anyone is. A zone metric is a count, a duration or a
speed derived from anonymous box geometry. ``ZoneMembership`` holds ephemeral,
session-local track ids purely so that entries and exits can be distinguished
from continuous presence across consecutive frames.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from atreides_vision.errors import ZoneConfigurationError
from atreides_vision.zones.geometry import NormalizedPoint, point_in_polygon

__all__ = [
    "Zone",
    "ZoneLayout",
    "ZoneMembership",
    "ZonePolygon",
    "ZoneReport",
    "ZoneSnapshot",
]

MIN_POLYGON_VERTICES = 3


@dataclass(frozen=True, slots=True)
class ZonePolygon:
    """A closed polygon in normalized frame coordinates.

    Raises:
        ZoneConfigurationError: if there are fewer than three vertices or any
            coordinate falls outside ``[0, 1]``.
    """

    vertices: tuple[NormalizedPoint, ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < MIN_POLYGON_VERTICES:
            raise ZoneConfigurationError(
                f"A zone polygon needs at least {MIN_POLYGON_VERTICES} points, "
                f"got {len(self.vertices)}"
            )
        for index, vertex in enumerate(self.vertices):
            if not (0.0 <= vertex.x <= 1.0 and 0.0 <= vertex.y <= 1.0):
                raise ZoneConfigurationError(
                    f"Zone polygon point {index} is ({vertex.x}, {vertex.y}); coordinates "
                    "must be normalized to the frame, so both must be between 0.0 and 1.0"
                )

    @classmethod
    def from_pairs(cls, pairs: Sequence[Sequence[float]]) -> Self:
        """Build a polygon from ``[[x, y], ...]`` as it appears in JSON."""
        vertices: list[NormalizedPoint] = []
        for index, pair in enumerate(pairs):
            if isinstance(pair, str | bytes) or len(tuple(pair)) != 2:
                raise ZoneConfigurationError(
                    f"Zone polygon point {index} must be a [x, y] pair, got {pair!r}"
                )
            x, y = tuple(pair)
            if isinstance(x, bool) or isinstance(y, bool) or not _is_number(x) or not _is_number(y):
                raise ZoneConfigurationError(
                    f"Zone polygon point {index} must contain two numbers, got {pair!r}"
                )
            vertices.append(NormalizedPoint(float(x), float(y)))
        return cls(tuple(vertices))

    def contains(self, point: NormalizedPoint) -> bool:
        """Whether ``point`` is inside this zone. Boundary counts as inside."""
        return point_in_polygon(point, self.vertices)

    def to_pixel_array(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Vertices as an ``(N, 2)`` int32 pixel array, for OpenCV drawing."""
        return np.array(
            [
                (round(vertex.x * frame_width), round(vertex.y * frame_height))
                for vertex in self.vertices
            ],
            dtype=np.int32,
        )

    @property
    def top_left(self) -> NormalizedPoint:
        """Top-left corner of the bounding box — the label anchor."""
        return NormalizedPoint(
            min(vertex.x for vertex in self.vertices),
            min(vertex.y for vertex in self.vertices),
        )

    def as_pairs(self) -> tuple[tuple[float, float], ...]:
        """Round-trip back to the JSON representation."""
        return tuple((vertex.x, vertex.y) for vertex in self.vertices)


@dataclass(frozen=True, slots=True)
class Zone:
    """An operator-defined region of one camera's view."""

    id: str
    name: str
    polygon: ZonePolygon

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ZoneConfigurationError("Zone id must be a non-empty string")
        if not self.name or not self.name.strip():
            raise ZoneConfigurationError(f"Zone '{self.id}' must have a non-empty name")

    def contains(self, point: NormalizedPoint) -> bool:
        return self.polygon.contains(point)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, position: int) -> Self:
        """Build a zone from one entry of the ``zones`` array."""
        if not isinstance(data, Mapping):
            raise ZoneConfigurationError(
                f"zones[{position}] must be an object, got {type(data).__name__}"
            )
        for field in ("id", "name", "polygon"):
            if field not in data:
                raise ZoneConfigurationError(
                    f"zones[{position}] is missing required field '{field}'"
                )

        zone_id, name, polygon = data["id"], data["name"], data["polygon"]
        if not isinstance(zone_id, str):
            raise ZoneConfigurationError(f"zones[{position}].id must be a string, got {zone_id!r}")
        if not isinstance(name, str):
            raise ZoneConfigurationError(f"zones[{position}].name must be a string, got {name!r}")
        if not isinstance(polygon, Sequence) or isinstance(polygon, str | bytes):
            raise ZoneConfigurationError(
                f"zones[{position}].polygon must be an array of [x, y] pairs"
            )

        try:
            return cls(id=zone_id, name=name, polygon=ZonePolygon.from_pairs(polygon))
        except ZoneConfigurationError as exc:
            raise ZoneConfigurationError(f"zones[{position}] ('{zone_id}'): {exc}") from exc


@dataclass(frozen=True, slots=True)
class ZoneLayout:
    """All zones configured for one camera."""

    camera_id: str
    zones: tuple[Zone, ...]

    def __post_init__(self) -> None:
        if not self.camera_id or not self.camera_id.strip():
            raise ZoneConfigurationError("camera_id must be a non-empty string")
        seen: set[str] = set()
        for zone in self.zones:
            if zone.id in seen:
                raise ZoneConfigurationError(
                    f"Duplicate zone id '{zone.id}'. Zone ids must be unique within a camera."
                )
            seen.add(zone.id)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Self:
        """Build a layout from parsed JSON. Raises with a field path on error."""
        if not isinstance(data, Mapping):
            raise ZoneConfigurationError(
                f"Zone configuration must be a JSON object, got {type(data).__name__}"
            )
        if "camera_id" not in data:
            raise ZoneConfigurationError("Zone configuration is missing required field 'camera_id'")
        if "zones" not in data:
            raise ZoneConfigurationError("Zone configuration is missing required field 'zones'")

        camera_id = data["camera_id"]
        if not isinstance(camera_id, str):
            raise ZoneConfigurationError(f"camera_id must be a string, got {camera_id!r}")

        raw_zones = data["zones"]
        if not isinstance(raw_zones, Sequence) or isinstance(raw_zones, str | bytes):
            raise ZoneConfigurationError("'zones' must be an array of zone objects")

        zones = tuple(
            Zone.from_mapping(entry, position=position) for position, entry in enumerate(raw_zones)
        )
        return cls(camera_id=camera_id, zones=zones)

    def get(self, zone_id: str) -> Zone | None:
        """The zone with this id, or ``None``."""
        return next((zone for zone in self.zones if zone.id == zone_id), None)

    @property
    def zone_ids(self) -> tuple[str, ...]:
        return tuple(zone.id for zone in self.zones)

    def __iter__(self) -> Iterator[Zone]:
        return iter(self.zones)

    def __len__(self) -> int:
        return len(self.zones)


@dataclass(frozen=True, slots=True)
class ZoneMembership:
    """Which tracks were observed inside one zone on one frame."""

    zone_id: str
    track_ids: frozenset[int]

    @property
    def occupancy(self) -> int:
        return len(self.track_ids)


@dataclass(frozen=True, slots=True)
class ZoneSnapshot:
    """Metrics for one zone at one moment.

    ``entries``, ``exits`` and ``completed_visits`` are cumulative for the
    processing session. The rest describe the zone right now.
    """

    zone_id: str
    zone_name: str
    occupancy: int
    entries: int
    exits: int
    mean_normalized_speed: float
    motion_level: float
    average_dwell_seconds: float
    max_dwell_seconds: float
    completed_visits: int

    def to_record(self) -> dict[str, Any]:
        """A compact JSON-serialisable record for the metrics stream."""
        return {
            "id": self.zone_id,
            "name": self.zone_name,
            "occupancy": self.occupancy,
            "entries": self.entries,
            "exits": self.exits,
            "mean_normalized_speed": round(self.mean_normalized_speed, 5),
            "motion_level": round(self.motion_level, 3),
            "average_dwell_seconds": round(self.average_dwell_seconds, 2),
            "max_dwell_seconds": round(self.max_dwell_seconds, 2),
            "completed_visits": self.completed_visits,
        }


@dataclass(frozen=True, slots=True)
class ZoneReport:
    """Every zone's metrics for one frame."""

    frame_index: int
    timestamp_seconds: float
    snapshots: tuple[ZoneSnapshot, ...]

    def for_zone(self, zone_id: str) -> ZoneSnapshot | None:
        return next((snap for snap in self.snapshots if snap.zone_id == zone_id), None)

    def to_record(self) -> dict[str, Any]:
        return {
            "frame": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "zones": [snapshot.to_record() for snapshot in self.snapshots],
        }

    def __len__(self) -> int:
        return len(self.snapshots)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and np.isfinite(float(value))

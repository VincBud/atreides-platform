"""WILDTRACK ground-truth annotation schema: one frame, many people.

Each ``annotations_positions/<frame_id>.json`` file is a JSON array of person
records:

.. code-block:: json

    [
      {
        "personID": 122,
        "positionID": 456826,
        "views": [
          {"viewNum": 0, "xmin": 1510, "ymin": 139, "xmax": 1561, "ymax": 299},
          {"viewNum": 1, "xmin": -1,   "ymin": -1,  "xmax": -1,   "ymax": -1},
          ...
        ]
      },
      ...
    ]

confirmed by reading the shipped annotation files directly. ``views`` always
has exactly seven entries, one per camera (``viewNum`` 0-6, see
:mod:`atreides_vision.venue_validation.datasets.wildtrack.cameras`).

Visibility follows the official WILDTRACK annotations viewer's own rule
(``return -1 not in _box.values()``): a view is considered **visible only
when none of** ``xmin``/``ymin``/``xmax``/``ymax`` **equals** ``-1``. Any
occurrence of ``-1`` — whether on all four fields or just one — makes that
view non-visible, matching the WILDTRACK paper's description ("positions
which are not visible in a given view are assigned coordinates of -1")
without the stronger, unverified assumption that only the all-four-``-1``
shape occurs. The real shipped corpus does contain views with exactly one
field equal to ``-1`` and the other three real, positive pixel values
(observed in roughly a quarter of the released annotation files); these are
accepted here as ordinary non-visible views, per the official toolkit's
rule. This module does **not** interpret such records as an image-boundary
clipping convention — no authoritative WILDTRACK source establishes that
reading, so no clipped/partially-visible bbox is exposed for them.

``personID`` is WILDTRACK's own ground-truth identity: which annotated real
person this record describes, stable across the whole recording. It is
**evaluation truth about the dataset, never a runtime identity** — deliberately
named and typed apart from :class:`~atreides_vision.identity.models.VenueSubjectId`
and :class:`~atreides_vision.behavior.models.SubjectRef` so nothing can pass
one where the other is expected. Nothing in this module or its callers may
hand a :class:`WildtrackPersonId` to the association, fusion, identity or
occupancy layers.

``positionID`` is an index into WILDTRACK's fixed ground-plane grid — 480 by
1440 cells, 2.5 cm spacing, origin at ``(-3.0, -9.0)`` metres — documented
verbatim in the dataset's own ``README.txt``:

.. code-block:: text

    X = -3.0 + 0.025*ID%480
    Y = -9.0 + 0.025*ID/480

reproduced here as :func:`ground_position_from_position_id`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atreides_vision.errors import WildtrackDatasetError
from atreides_vision.venue_validation.datasets.wildtrack.cameras import (
    CAMERA_COUNT,
    CAMERA_IDS,
    camera_id_for_view_num,
)
from atreides_vision.venue_validation.datasets.wildtrack.time import WildtrackFrameId

__all__ = [
    "GRID_HEIGHT",
    "GRID_ORIGIN_X_M",
    "GRID_ORIGIN_Y_M",
    "GRID_SPACING_M",
    "GRID_WIDTH",
    "MAX_POSITION_ID",
    "WildtrackBoundingBox",
    "WildtrackFrameAnnotations",
    "WildtrackGroundPosition",
    "WildtrackPersonAnnotation",
    "WildtrackPersonId",
    "ground_position_from_position_id",
    "load_frame_annotations",
]

#: WILDTRACK's fixed ground-plane grid, verbatim from ``README.txt``.
GRID_WIDTH = 480
GRID_HEIGHT = 1440
GRID_SPACING_M = 0.025
GRID_ORIGIN_X_M = -3.0
GRID_ORIGIN_Y_M = -9.0
MAX_POSITION_ID = GRID_WIDTH * GRID_HEIGHT - 1

#: The non-visible marker used for every bbox field of an un-observed view.
_NOT_VISIBLE = -1


def ground_position_from_position_id(position_id: int) -> tuple[float, float]:
    """WILDTRACK's own ``positionID`` -> ground-plane metres formula.

    Raises:
        WildtrackDatasetError: if ``position_id`` is outside the fixed grid.
    """
    if not (0 <= position_id <= MAX_POSITION_ID):
        raise WildtrackDatasetError(
            f"WILDTRACK positionID must be in 0..{MAX_POSITION_ID}, got {position_id!r}"
        )
    x = GRID_ORIGIN_X_M + GRID_SPACING_M * (position_id % GRID_WIDTH)
    y = GRID_ORIGIN_Y_M + GRID_SPACING_M * (position_id // GRID_WIDTH)
    return (x, y)


@dataclass(frozen=True, slots=True, order=True)
class WildtrackPersonId:
    """WILDTRACK's own ground-truth person identity for one annotated frame set.

    This is **evaluation ground truth**, not a venue-local or camera-local
    runtime identity. It must never be passed to
    :mod:`atreides_vision.association`, :mod:`atreides_vision.fusion`,
    :mod:`atreides_vision.identity` or :mod:`atreides_vision.occupancy` — only
    an offline evaluator, comparing Atreides' own output against this truth,
    may ever hold both types at once.
    """

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise WildtrackDatasetError(f"WILDTRACK personID must be >= 0, got {self.value!r}")


@dataclass(frozen=True, slots=True)
class WildtrackGroundPosition:
    """A person's ground-truth position, both as WILDTRACK's grid cell and metres."""

    position_id: int
    x_m: float
    y_m: float

    def __post_init__(self) -> None:
        if not (0 <= self.position_id <= MAX_POSITION_ID):
            raise WildtrackDatasetError(
                f"WILDTRACK positionID must be in 0..{MAX_POSITION_ID}, got {self.position_id!r}"
            )
        if not (math.isfinite(self.x_m) and math.isfinite(self.y_m)):
            raise WildtrackDatasetError(
                f"WILDTRACK ground position must be finite, got ({self.x_m!r}, {self.y_m!r})"
            )

    @classmethod
    def from_position_id(cls, position_id: int) -> WildtrackGroundPosition:
        x, y = ground_position_from_position_id(position_id)
        return cls(position_id=position_id, x_m=x, y_m=y)


@dataclass(frozen=True, slots=True)
class WildtrackBoundingBox:
    """A visible-view bounding box, in the released (undistorted) image pixels."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        values = (self.xmin, self.ymin, self.xmax, self.ymax)
        if not all(math.isfinite(v) for v in values):
            raise WildtrackDatasetError(f"WILDTRACK bbox must be finite, got {values!r}")
        if self.xmax < self.xmin or self.ymax < self.ymin:
            raise WildtrackDatasetError(f"WILDTRACK bbox has max before min: {values!r}")

    def to_record(self) -> dict[str, float]:
        return {"xmin": self.xmin, "ymin": self.ymin, "xmax": self.xmax, "ymax": self.ymax}


@dataclass(frozen=True, slots=True)
class WildtrackPersonAnnotation:
    """One annotated person, at one frame, as seen (or not) by every camera.

    ``views`` holds exactly one entry per canonical camera id, in canonical
    order (:data:`~atreides_vision.venue_validation.datasets.wildtrack.cameras.CAMERA_IDS`),
    mapping to ``None`` where WILDTRACK's own marker records that camera did
    not see this person at this instant.
    """

    person_id: WildtrackPersonId
    position: WildtrackGroundPosition
    views: Mapping[str, WildtrackBoundingBox | None]

    def __post_init__(self) -> None:
        missing = set(CAMERA_IDS) - set(self.views)
        if missing:
            raise WildtrackDatasetError(
                f"Person {self.person_id.value} is missing view entries for "
                f"{sorted(missing)}; every WILDTRACK person record must carry all "
                f"{CAMERA_COUNT} camera views."
            )
        extra = set(self.views) - set(CAMERA_IDS)
        if extra:
            raise WildtrackDatasetError(
                f"Person {self.person_id.value} has view entries for unknown cameras "
                f"{sorted(extra)}"
            )

    def visible_in(self, camera_id: str) -> bool:
        return self.views.get(camera_id) is not None

    @property
    def visible_camera_ids(self) -> tuple[str, ...]:
        """Cameras that observed this person, in canonical order."""
        return tuple(camera_id for camera_id in CAMERA_IDS if self.visible_in(camera_id))

    def bbox_for(self, camera_id: str) -> WildtrackBoundingBox | None:
        if camera_id not in CAMERA_IDS:
            raise WildtrackDatasetError(f"Unknown WILDTRACK camera id {camera_id!r}")
        return self.views[camera_id]

    def to_record(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id.value,
            "position_id": self.position.position_id,
            "ground_position_m": [self.position.x_m, self.position.y_m],
            "views": {
                camera_id: (None if bbox is None else bbox.to_record())
                for camera_id in CAMERA_IDS
                for bbox in (self.views[camera_id],)
            },
        }


@dataclass(frozen=True, slots=True)
class WildtrackFrameAnnotations:
    """All annotated people at one WILDTRACK frame, in canonical order.

    ``persons`` is sorted by :class:`WildtrackPersonId`, independent of the
    order the source JSON array listed them in — two parses of the same file
    content produce byte-identical output regardless of upstream ordering.
    """

    frame_id: WildtrackFrameId
    persons: tuple[WildtrackPersonAnnotation, ...]

    def __post_init__(self) -> None:
        ids = [p.person_id.value for p in self.persons]
        if len(set(ids)) != len(ids):
            raise WildtrackDatasetError(
                f"Frame {self.frame_id.filename_stem} has a duplicate personID among {sorted(ids)}"
            )
        ordered = tuple(sorted(self.persons, key=lambda p: p.person_id.value))
        if ordered != self.persons:
            raise WildtrackDatasetError(
                f"Frame {self.frame_id.filename_stem} persons are not in canonical "
                "(personID-ascending) order; construct with persons already sorted"
            )

    @property
    def person_count(self) -> int:
        return len(self.persons)

    def person(self, person_id: int) -> WildtrackPersonAnnotation | None:
        for person in self.persons:
            if person.person_id.value == person_id:
                return person
        return None

    @classmethod
    def from_json(cls, frame_id: WildtrackFrameId, data: object) -> WildtrackFrameAnnotations:
        """Parse one ``annotations_positions/<frame_id>.json`` file's content.

        Raises:
            WildtrackDatasetError: if the top level is not an array, any
                record is malformed, a personID repeats, a viewNum is out of
                range or repeated, or a fully-visible (no ``-1``) bbox fails
                its own coordinate validation (e.g. max before min).
        """
        if not isinstance(data, Sequence) or isinstance(data, str | bytes):
            raise WildtrackDatasetError(
                f"Frame {frame_id.filename_stem} annotation must be a JSON array, got "
                f"{type(data).__name__}"
            )
        persons = tuple(
            _person_from_mapping(entry, frame_id=frame_id, position=index)
            for index, entry in enumerate(data)
        )
        ordered = tuple(sorted(persons, key=lambda p: p.person_id.value))
        return cls(frame_id=frame_id, persons=ordered)

    def to_record(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id.value,
            "venue_seconds": self.frame_id.venue_seconds,
            "persons": [p.to_record() for p in self.persons],
        }


def load_frame_annotations(path: Path, frame_id: WildtrackFrameId) -> WildtrackFrameAnnotations:
    """Read and parse one ``annotations_positions/<frame_id>.json`` file.

    Raises:
        WildtrackDatasetError: if the file is missing, is not valid JSON, or
            fails :meth:`WildtrackFrameAnnotations.from_json`'s validation.
    """
    if not path.is_file():
        raise WildtrackDatasetError(f"Missing WILDTRACK annotation file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WildtrackDatasetError(f"Could not read WILDTRACK annotation file: {path}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WildtrackDatasetError(f"Malformed WILDTRACK annotation JSON: {path}") from exc
    return WildtrackFrameAnnotations.from_json(frame_id, data)


def _person_from_mapping(
    entry: object, *, frame_id: WildtrackFrameId, position: int
) -> WildtrackPersonAnnotation:
    if not isinstance(entry, Mapping):
        raise WildtrackDatasetError(
            f"Frame {frame_id.filename_stem}: person record [{position}] must be an "
            f"object, got {type(entry).__name__}"
        )
    for field in ("personID", "positionID", "views"):
        if field not in entry:
            raise WildtrackDatasetError(
                f"Frame {frame_id.filename_stem}: person record [{position}] is "
                f"missing required field {field!r}"
            )
    person_id = _require_int(entry["personID"], f"person record [{position}].personID")
    position_id = _require_int(entry["positionID"], f"person record [{position}].positionID")
    raw_views = entry["views"]
    if not isinstance(raw_views, Sequence) or isinstance(raw_views, str | bytes):
        raise WildtrackDatasetError(
            f"Frame {frame_id.filename_stem}: person {person_id}'s 'views' must be an array"
        )
    if len(raw_views) != CAMERA_COUNT:
        raise WildtrackDatasetError(
            f"Frame {frame_id.filename_stem}: person {person_id} has {len(raw_views)} view "
            f"entries, expected exactly {CAMERA_COUNT}"
        )

    views: dict[str, WildtrackBoundingBox | None] = {}
    seen_view_nums: set[int] = set()
    for view_index, raw_view in enumerate(raw_views):
        if not isinstance(raw_view, Mapping):
            raise WildtrackDatasetError(
                f"Frame {frame_id.filename_stem}: person {person_id} view [{view_index}] "
                f"must be an object"
            )
        for field in ("viewNum", "xmin", "ymin", "xmax", "ymax"):
            if field not in raw_view:
                raise WildtrackDatasetError(
                    f"Frame {frame_id.filename_stem}: person {person_id} view [{view_index}] "
                    f"is missing required field {field!r}"
                )
        view_num = _require_int(
            raw_view["viewNum"], f"person {person_id} view [{view_index}].viewNum"
        )
        if view_num in seen_view_nums:
            raise WildtrackDatasetError(
                f"Frame {frame_id.filename_stem}: person {person_id} has more than one "
                f"view entry for viewNum {view_num}"
            )
        seen_view_nums.add(view_num)
        camera_id = camera_id_for_view_num(view_num)

        coords = tuple(
            _require_int(raw_view[field], f"person {person_id} view [{view_index}].{field}")
            for field in ("xmin", "ymin", "xmax", "ymax")
        )
        if _NOT_VISIBLE in coords:
            # Official WILDTRACK annotations viewer semantics: a view is
            # visible only when none of its four coordinates is -1. Any
            # occurrence of -1 — whether on all four fields or only one —
            # makes the view non-visible; no usable bbox is exposed for it.
            views[camera_id] = None
        else:
            xmin, ymin, xmax, ymax = coords
            views[camera_id] = WildtrackBoundingBox(
                xmin=float(xmin), ymin=float(ymin), xmax=float(xmax), ymax=float(ymax)
            )

    if seen_view_nums != set(range(CAMERA_COUNT)):
        raise WildtrackDatasetError(
            f"Frame {frame_id.filename_stem}: person {person_id} views do not cover "
            f"viewNum 0..{CAMERA_COUNT - 1} exactly; got {sorted(seen_view_nums)}"
        )

    return WildtrackPersonAnnotation(
        person_id=WildtrackPersonId(person_id),
        position=WildtrackGroundPosition.from_position_id(position_id),
        views=views,
    )


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WildtrackDatasetError(f"{field} must be an integer, got {value!r}")
    return value

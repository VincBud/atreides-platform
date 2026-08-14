"""Atreides-owned domain models for the SEE + TRACK pipeline.

These types are the *only* currency the pipeline passes around. Framework
objects (``supervision.Detections``, RF-DETR outputs, ``trackers`` state) never
cross an adapter boundary, which is what makes the detector and tracker
replaceable without touching :mod:`atreides_vision.video.processor`.

Coordinate conventions
----------------------
* All coordinates are **pixel coordinates in the source frame's resolution**.
* The origin is the **top-left** corner; ``x`` grows right, ``y`` grows down.
* Boxes are **xyxy**: ``(x1, y1, x2, y2)`` with ``x1 <= x2`` and ``y1 <= y2``.
* Coordinates are ``float`` throughout; rounding to ``int`` happens only at the
  drawing layer.

Privacy note
------------
``Track.track_id`` is an ephemeral, camera- and session-local integer. It is a
handle on a *box moving through this video*, not an identity. Nothing in this
module carries or derives biometric information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Self

import numpy as np

__all__ = [
    "XYXY",
    "BoundingBox",
    "Detection",
    "Point",
    "Track",
    "detections_to_arrays",
]

#: A bounding box as a plain 4-tuple in xyxy pixel coordinates.
XYXY = tuple[float, float, float, float]

_EPSILON: Final[float] = 1e-6


@dataclass(frozen=True, slots=True)
class Point:
    """A 2D point in source-frame pixel coordinates."""

    x: float
    y: float

    def as_int_tuple(self) -> tuple[int, int]:
        """Return the point rounded to integer pixels, for drawing."""
        return (round(self.x), round(self.y))


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An axis-aligned box in xyxy pixel coordinates.

    Raises:
        ValueError: if the box is degenerate (``x2 < x1`` or ``y2 < y1``) or any
            coordinate is not finite.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name, value in (("x1", self.x1), ("y1", self.y1), ("x2", self.x2), ("y2", self.y2)):
            if not np.isfinite(value):
                raise ValueError(f"BoundingBox.{name} must be finite, got {value!r}")
        if self.x2 < self.x1 - _EPSILON or self.y2 < self.y1 - _EPSILON:
            raise ValueError(
                "BoundingBox requires x1 <= x2 and y1 <= y2 (xyxy convention), "
                f"got ({self.x1}, {self.y1}, {self.x2}, {self.y2})"
            )

    @classmethod
    def from_xyxy(cls, xyxy: XYXY | np.ndarray) -> Self:
        """Build a box from a 4-element sequence or NumPy row in xyxy order."""
        values = np.asarray(xyxy, dtype=np.float64).reshape(-1)
        if values.size != 4:
            raise ValueError(f"Expected 4 xyxy values, got {values.size}")
        x1, y1, x2, y2 = (float(v) for v in values)
        return cls(x1, y1, x2, y2)

    def as_tuple(self) -> XYXY:
        """Return ``(x1, y1, x2, y2)``."""
        return (self.x1, self.y1, self.x2, self.y2)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        """Return the box rounded to integer pixels, for drawing."""
        return (round(self.x1), round(self.y1), round(self.x2), round(self.y2))

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        """The box centre, used as the trajectory sample point."""
        return Point((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Point:
        """The middle of the bottom edge — the canonical *floor position*.

        For a camera looking across a floor this is roughly where the person's
        feet are, and it is what decides which part of the floor they occupy.
        The box centre sits around chest height and drifts into whatever is
        behind them, so it is the wrong anchor for anything spatial.

        Every layer that needs a person's position on the ground uses this one
        definition: zone assignment
        (:func:`~atreides_vision.zones.geometry.bottom_center`) and behaviour
        analysis (:func:`~atreides_vision.behavior.coordinates.bottom_center_anchor`).
        """
        return Point((self.x1 + self.x2) / 2.0, self.y2)

    def clipped_to(self, width: int, height: int) -> BoundingBox:
        """Return a copy clamped to a ``width`` x ``height`` frame."""
        return BoundingBox(
            x1=min(max(self.x1, 0.0), float(width)),
            y1=min(max(self.y1, 0.0), float(height)),
            x2=min(max(self.x2, 0.0), float(width)),
            y2=min(max(self.y2, 0.0), float(height)),
        )


@dataclass(frozen=True, slots=True)
class Detection:
    """A single-frame detection produced by a :class:`PersonDetector`.

    A detection has no memory: it is what the detector saw in one frame, with no
    relationship to any other frame.
    """

    bbox_xyxy: BoundingBox
    confidence: float
    class_id: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Detection.confidence must be in [0, 1], got {self.confidence!r}")
        if self.class_id < 0:
            raise ValueError(f"Detection.class_id must be non-negative, got {self.class_id!r}")

    @classmethod
    def from_xyxy(cls, xyxy: XYXY | np.ndarray, confidence: float, class_id: int) -> Self:
        """Convenience constructor from a raw xyxy sequence."""
        return cls(
            bbox_xyxy=BoundingBox.from_xyxy(xyxy),
            confidence=float(confidence),
            class_id=int(class_id),
        )

    @property
    def center(self) -> Point:
        return self.bbox_xyxy.center


@dataclass(frozen=True, slots=True)
class Track:
    """A detection associated with a persistent, session-local track identity.

    ``track_id`` is assigned by the tracker, is unique only within this video
    session and this camera, and is discarded when the process exits. It is
    deliberately *not* an identity: see the module docstring.
    """

    track_id: int
    bbox_xyxy: BoundingBox
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError(f"Track.track_id must be non-negative, got {self.track_id!r}")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Track.confidence must be in [0, 1], got {self.confidence!r}")

    @property
    def center(self) -> Point:
        """The centre of the tracked box — the point appended to its trajectory."""
        return self.bbox_xyxy.center


def detections_to_arrays(
    detections: list[Detection],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten detections into ``(xyxy, confidence, class_id)`` NumPy arrays.

    Adapters that hand detections to array-oriented third-party trackers use
    this so the conversion lives in one tested place.

    Returns:
        ``xyxy`` with shape ``(N, 4)`` float32, ``confidence`` with shape ``(N,)``
        float32, and ``class_id`` with shape ``(N,)`` int32. For an empty input
        the shapes are ``(0, 4)``, ``(0,)`` and ``(0,)``.
    """
    if not detections:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    xyxy = np.array([d.bbox_xyxy.as_tuple() for d in detections], dtype=np.float32)
    confidence = np.array([d.confidence for d in detections], dtype=np.float32)
    class_id = np.array([d.class_id for d in detections], dtype=np.int32)
    return xyxy, confidence, class_id

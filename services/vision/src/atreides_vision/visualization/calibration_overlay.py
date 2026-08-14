"""Drawing a calibration onto a frame, so a human can check it.

Calibration is the one part of the pipeline whose correctness cannot be
established from its own numbers. A homography that reproduces its reference
points perfectly is still wrong if those points were measured against the wrong
spots on the floor, and reprojection error cannot detect that — it only ever
compares the fit against the operator's own claims.

The check that *can* catch it is visual: draw the venue grid back onto the
frame and look at whether the metre lines follow the floor. This module does
that and nothing else.

**Inspection only.** There is no click handling, no GUI, no automatic point
detection and no editing. It renders an image; a person looks at it.

Rendering lives here rather than in the calibration package so that package
stays free of OpenCV, and so all drawing code sits in one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from atreides_vision.calibration.models import VenuePoint
from atreides_vision.calibration.projector import GroundPlaneProjector
from atreides_vision.errors import ConfigurationError, ProjectionError
from atreides_vision.models import Point
from atreides_vision.visualization.layout import resolve_overlay_scale

logger = logging.getLogger(__name__)

__all__ = ["CalibrationOverlayConfig", "render_calibration_overlay"]

_FONT: Final[int] = cv2.FONT_HERSHEY_SIMPLEX

#: Muted BGR palette, consistent with the rest of the overlay: structure reads
#: as background, the reference points read as the subject.
_GRID_COLOR: Final[tuple[int, int, int]] = (150, 130, 90)
_HULL_COLOR: Final[tuple[int, int, int]] = (200, 170, 110)
_POINT_COLOR: Final[tuple[int, int, int]] = (90, 200, 230)
_X_AXIS_COLOR: Final[tuple[int, int, int]] = (110, 110, 235)
_Y_AXIS_COLOR: Final[tuple[int, int, int]] = (120, 210, 140)
_LABEL_TEXT: Final[tuple[int, int, int]] = (255, 255, 255)
_LABEL_BG: Final[tuple[int, int, int]] = (24, 24, 24)

#: Points further outside the frame than this multiple of its size are dropped
#: rather than drawn. Projection near the horizon produces coordinates in the
#: millions, which OpenCV cannot render and which would be meaningless anyway.
_MAX_OFFSCREEN_FACTOR: Final[float] = 6.0

#: Ceiling on grid lines per axis, so a mistyped spacing cannot ask for a
#: million draw calls. Expressed as a count, not a distance, so it holds for a
#: 3 m doorway and a 200 m field alike.
_MAX_GRID_LINES_PER_AXIS: Final[int] = 400

#: Samples per grid line. A homography maps a straight line to a straight line,
#: so two would suffice geometrically — but sampling lets a line that leaves the
#: usable region be drawn in the parts that remain.
_SAMPLES_PER_LINE: Final[int] = 12


@dataclass(frozen=True, slots=True)
class CalibrationOverlayConfig:
    """What to draw. Everything is optional so a cluttered view can be pared back.

    Raises:
        ConfigurationError: if the grid spacing is not positive.
    """

    #: Spacing of the venue grid, in metres. No default venue size is assumed;
    #: the grid covers whatever region the calibration's own points span.
    grid_spacing_metres: float = 1.0
    draw_grid: bool = True
    #: Outline of the region the reference points actually cover. Outside it the
    #: calibration is extrapolating, which is worth being able to see.
    draw_hull: bool = True
    #: Venue origin and the +x / +y directions.
    draw_axes: bool = True
    draw_points: bool = True
    #: Label each reference point with its index and venue coordinates.
    label_points: bool = True
    #: Extra margin drawn around the calibrated region, as a fraction of its
    #: extent, so the grid does not stop abruptly at the outermost point.
    grid_margin_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.grid_spacing_metres <= 0:
            raise ConfigurationError(
                f"grid_spacing_metres must be positive, got {self.grid_spacing_metres}"
            )
        if self.grid_margin_fraction < 0:
            raise ConfigurationError(
                f"grid_margin_fraction must be non-negative, got {self.grid_margin_fraction}"
            )


def render_calibration_overlay(
    frame: np.ndarray,
    projector: GroundPlaneProjector,
    config: CalibrationOverlayConfig | None = None,
) -> np.ndarray:
    """Draw a calibration's geometry onto a copy of ``frame``.

    Args:
        frame: A BGR frame, sized as the projector's runtime frame.
        projector: The calibrated projector to visualise.
        config: What to draw. Defaults to everything, at one-metre spacing.

    Returns:
        A new annotated frame. The input is not modified.

    Raises:
        ConfigurationError: if the frame is empty or the grid is unreasonable.
    """
    settings = config or CalibrationOverlayConfig()
    if frame.size == 0 or frame.ndim != 3:
        raise ConfigurationError(
            f"Calibration diagnostics need a non-empty BGR frame, got shape {frame.shape}"
        )

    canvas = frame.copy()
    height, width = canvas.shape[:2]
    scale = resolve_overlay_scale(width, height)

    if settings.draw_grid:
        _draw_grid(canvas, projector, settings, scale.outline_thickness)
    if settings.draw_hull:
        _draw_hull(canvas, projector, scale.outline_thickness + 1)
    if settings.draw_axes:
        _draw_axes(canvas, projector, settings, scale.box_thickness + 1, scale.font_scale)
    if settings.draw_points:
        _draw_reference_points(canvas, projector, settings, scale.font_scale, scale.padding)

    return canvas


def _venue_bounds(
    projector: GroundPlaneProjector, margin_fraction: float
) -> tuple[float, float, float, float]:
    """``(min_x, min_y, max_x, max_y)`` of the calibrated region, plus margin."""
    points = projector.calibration.venue_points()
    min_x, min_y = (float(value) for value in points.min(axis=0))
    max_x, max_y = (float(value) for value in points.max(axis=0))

    margin_x = max((max_x - min_x) * margin_fraction, 0.0)
    margin_y = max((max_y - min_y) * margin_fraction, 0.0)
    return (min_x - margin_x, min_y - margin_y, max_x + margin_x, max_y + margin_y)


def _to_pixel(projector: GroundPlaneProjector, x: float, y: float) -> Point | None:
    """Project a venue position into runtime pixels, or ``None`` if unusable."""
    try:
        return projector.project_from_venue(VenuePoint(x, y))
    except ProjectionError:
        return None


def _drawable(point: Point | None, width: int, height: int) -> tuple[int, int] | None:
    """Round to integer pixels, discarding anything absurdly far off-frame."""
    if point is None:
        return None
    limit_x = width * _MAX_OFFSCREEN_FACTOR
    limit_y = height * _MAX_OFFSCREEN_FACTOR
    if not (-limit_x <= point.x <= limit_x and -limit_y <= point.y <= limit_y):
        return None
    return (round(point.x), round(point.y))


def _draw_polyline(
    canvas: np.ndarray,
    projector: GroundPlaneProjector,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw a venue-space line segment, skipping any unprojectable stretch."""
    height, width = canvas.shape[:2]
    previous: tuple[int, int] | None = None
    for step in range(_SAMPLES_PER_LINE + 1):
        ratio = step / _SAMPLES_PER_LINE
        x = start[0] + (end[0] - start[0]) * ratio
        y = start[1] + (end[1] - start[1]) * ratio
        current = _drawable(_to_pixel(projector, x, y), width, height)
        if previous is not None and current is not None:
            cv2.line(canvas, previous, current, color, thickness, cv2.LINE_AA)
        previous = current


def _grid_positions(low: float, high: float, spacing: float) -> list[float]:
    """Multiples of ``spacing`` covering ``[low, high]``.

    Anchored on the venue origin rather than on the region's corner, so the
    lines land on whole metres — which is what makes them checkable against a
    real floor.

    Raises:
        ConfigurationError: if the spacing would produce an unreasonable count.
    """
    first = int(np.floor(low / spacing))
    last = int(np.ceil(high / spacing))
    count = last - first + 1
    if count > _MAX_GRID_LINES_PER_AXIS:
        raise ConfigurationError(
            f"A {spacing:g} m grid over a {high - low:.1f} m span needs {count} lines, "
            f"more than the {_MAX_GRID_LINES_PER_AXIS} allowed. Use a coarser "
            "--grid-spacing."
        )
    return [index * spacing for index in range(first, last + 1)]


def _draw_grid(
    canvas: np.ndarray,
    projector: GroundPlaneProjector,
    settings: CalibrationOverlayConfig,
    thickness: int,
) -> None:
    """Lines of constant venue x and constant venue y, one per grid step."""
    min_x, min_y, max_x, max_y = _venue_bounds(projector, settings.grid_margin_fraction)
    spacing = settings.grid_spacing_metres

    for x in _grid_positions(min_x, max_x, spacing):
        _draw_polyline(canvas, projector, (x, min_y), (x, max_y), _GRID_COLOR, thickness)
    for y in _grid_positions(min_y, max_y, spacing):
        _draw_polyline(canvas, projector, (min_x, y), (max_x, y), _GRID_COLOR, thickness)


def _draw_hull(canvas: np.ndarray, projector: GroundPlaneProjector, thickness: int) -> None:
    """Outline the region the reference points actually span."""
    points = projector.calibration.venue_points()
    min_x, min_y = (float(value) for value in points.min(axis=0))
    max_x, max_y = (float(value) for value in points.max(axis=0))

    corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    for index, corner in enumerate(corners):
        _draw_polyline(
            canvas, projector, corner, corners[(index + 1) % len(corners)], _HULL_COLOR, thickness
        )


def _draw_axes(
    canvas: np.ndarray,
    projector: GroundPlaneProjector,
    settings: CalibrationOverlayConfig,
    thickness: int,
    font_scale: float,
) -> None:
    """The venue origin and the direction of increasing x and y."""
    height, width = canvas.shape[:2]
    origin = _drawable(_to_pixel(projector, 0.0, 0.0), width, height)
    if origin is None:
        logger.debug("Venue origin does not project into this frame; skipping axes.")
        return

    length = max(settings.grid_spacing_metres, 1.0)
    for offset, color, label in (
        ((length, 0.0), _X_AXIS_COLOR, "+x"),
        ((0.0, length), _Y_AXIS_COLOR, "+y"),
    ):
        tip = _drawable(_to_pixel(projector, offset[0], offset[1]), width, height)
        if tip is None:
            continue
        cv2.arrowedLine(canvas, origin, tip, color, thickness, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(
            canvas, label, tip, _FONT, font_scale, color, max(1, thickness - 1), cv2.LINE_AA
        )

    cv2.circle(canvas, origin, max(3, thickness + 2), _LABEL_TEXT, -1, cv2.LINE_AA)
    _draw_label(canvas, "venue (0, 0)", origin, font_scale, padding=4)


def _draw_reference_points(
    canvas: np.ndarray,
    projector: GroundPlaneProjector,
    settings: CalibrationOverlayConfig,
    font_scale: float,
    padding: int,
) -> None:
    """Mark each correspondence at its *stated image position*.

    Drawn from the calibration's own image coordinates rather than by projecting
    its venue coordinates back, so that a marker landing in the wrong place is
    visible as exactly that: a point that was measured wrongly. Rescaled to the
    runtime frame, since the stated pixels are in reference-frame coordinates.
    """
    height, width = canvas.shape[:2]
    reference = projector.calibration.reference_frame
    runtime = projector.runtime_frame
    scale_x = runtime.width / reference.width
    scale_y = runtime.height / reference.height
    radius = max(4, round(min(width, height) * 0.006))

    for index, correspondence in enumerate(projector.calibration.correspondences):
        pixel = _drawable(
            Point(correspondence.image.x * scale_x, correspondence.image.y * scale_y),
            width,
            height,
        )
        if pixel is None:
            continue
        cv2.circle(canvas, pixel, radius, _POINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(canvas, pixel, radius + 2, _LABEL_BG, 1, cv2.LINE_AA)

        if settings.label_points:
            venue = correspondence.venue
            label = f"{index}: ({venue.x:g}, {venue.y:g}) m"
            _draw_label(canvas, label, (pixel[0] + radius + 3, pixel[1]), font_scale, padding)


def _draw_label(
    canvas: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    font_scale: float,
    padding: int,
) -> None:
    """Text on a filled chip, so it stays readable over any footage."""
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    x, y = anchor
    top = y - text_height - padding
    bottom = y + baseline

    cv2.rectangle(
        canvas, (x - padding, top), (x + text_width + padding, bottom), _LABEL_BG, -1, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        text,
        (x, y - padding // 2),
        _FONT,
        font_scale,
        _LABEL_TEXT,
        thickness,
        cv2.LINE_AA,
    )

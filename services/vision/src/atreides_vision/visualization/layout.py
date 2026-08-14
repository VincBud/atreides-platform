"""Frame-relative sizing for the overlay.

The overlay was authored against 1080p. Applied unchanged to a 384x288 CCTV
frame the status panel covered roughly a fifth of the image, hiding the part of
the scene furthest from the camera — exactly where the small, hard cases are.

Everything here is pure arithmetic over frame dimensions plus text measurement,
so it can be tested without rendering anything. Two rules do the work:

* sizes scale with frame height against a 1080p reference, clamped so they stay
  legible on small frames and do not become cartoonish on large ones;
* panels are additionally capped as a *fraction* of the frame, and the font
  shrinks until the panel fits inside that budget.

The second rule is what actually bounds the damage: without it, the legibility
floor on the first rule would still produce an oversized panel on a small frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import cv2

__all__ = [
    "COMPACT_BELOW_FACTOR",
    "MAX_PANEL_HEIGHT_FRACTION",
    "MAX_PANEL_WIDTH_FRACTION",
    "REFERENCE_HEIGHT",
    "OverlayScale",
    "PanelLayout",
    "panel_layout",
    "resolve_overlay_scale",
]

#: The frame height the overlay's authored sizes assume.
REFERENCE_HEIGHT: Final[int] = 1080

# Bounds on the raw height ratio. The floor stops a thumbnail collapsing every
# element to sub-pixel sizes; the ceiling stops 4K frames rendering a billboard.
_MIN_SCALE_FACTOR: Final[float] = 0.28
_MAX_SCALE_FACTOR: Final[float] = 1.60

# Legibility bounds on the font itself. 0.30 is about the smallest
# FONT_HERSHEY_SIMPLEX size that still reads on a 288-line frame.
_MIN_FONT_SCALE: Final[float] = 0.30
_MAX_FONT_SCALE: Final[float] = 1.10

#: Below this share of the reference height, panels use abbreviated labels.
#: Set where the full labels stop fitting their width budget even at the font
#: floor: around 320 lines. Expressed against the reference so it generalises
#: to any small frame rather than naming one resolution. 360-line and larger
#: frames keep the full wording, because there they still fit.
COMPACT_BELOW_FACTOR: Final[float] = 0.30

#: A panel may never exceed this share of the frame's width or height.
MAX_PANEL_WIDTH_FRACTION: Final[float] = 0.42
MAX_PANEL_HEIGHT_FRACTION: Final[float] = 0.30

_FONT: Final[int] = cv2.FONT_HERSHEY_SIMPLEX
#: Ascender-plus-descender sample, so line height does not depend on content.
_METRIC_SAMPLE: Final[str] = "Ag"
_SHRINK_STEP: Final[float] = 0.92
_MAX_SHRINK_ITERATIONS: Final[int] = 24


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True, slots=True)
class OverlayScale:
    """Sizes for one frame, derived from its dimensions.

    ``factor`` is the raw (clamped) height ratio; the rest are the authored
    sizes scaled by it and rounded to whole pixels where they must be integers.
    """

    factor: float
    font_scale: float
    padding: int
    line_gap: int
    box_thickness: int
    trail_thickness: int
    outline_thickness: int
    #: Panels should use abbreviated labels: there is not room for full ones.
    compact: bool

    @property
    def label_padding(self) -> int:
        """Padding inside a track's ``ID n`` label chip."""
        return max(1, self.padding // 2)


def resolve_overlay_scale(
    frame_width: int,
    frame_height: int,
    *,
    base_font_scale: float = 0.5,
    base_box_thickness: int = 2,
    base_trail_thickness: int = 2,
    base_outline_thickness: int = 2,
    base_padding: int = 8,
    base_line_gap: int = 5,
) -> OverlayScale:
    """Derive overlay sizes for a frame of the given dimensions.

    The base values are the authored 1080p sizes — normally
    :class:`~atreides_vision.config.VisualizationConfig` fields — so a 1080p
    frame reproduces the pre-V0.2.1 appearance exactly.

    Args:
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        base_font_scale: OpenCV font scale at the reference height.
        base_box_thickness: Bounding-box line width at the reference height.
        base_trail_thickness: Trajectory line width at the reference height.
        base_outline_thickness: Zone outline width at the reference height.
        base_padding: Panel inner padding at the reference height.
        base_line_gap: Extra space between text lines at the reference height.

    Returns:
        The sizes to draw this frame with.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"frame dimensions must be positive, got {frame_width}x{frame_height}")

    factor = _clamp(frame_height / REFERENCE_HEIGHT, _MIN_SCALE_FACTOR, _MAX_SCALE_FACTOR)
    return OverlayScale(
        factor=factor,
        font_scale=_clamp(base_font_scale * factor, _MIN_FONT_SCALE, _MAX_FONT_SCALE),
        padding=max(2, round(base_padding * factor)),
        line_gap=max(1, round(base_line_gap * factor)),
        box_thickness=max(1, round(base_box_thickness * factor)),
        trail_thickness=max(1, round(base_trail_thickness * factor)),
        outline_thickness=max(1, round(base_outline_thickness * factor)),
        compact=frame_height < REFERENCE_HEIGHT * COMPACT_BELOW_FACTOR,
    )


@dataclass(frozen=True, slots=True)
class PanelLayout:
    """Where a text panel sits and how its text is drawn."""

    font_scale: float
    line_height: int
    padding: int
    width: int
    height: int
    #: Baseline offset of the first line from the panel's top edge.
    first_baseline: int

    def baseline(self, index: int) -> int:
        """Baseline offset of the ``index``-th line from the panel's top edge."""
        return self.first_baseline + index * self.line_height


def _measure(font_scale: float, lines: Sequence[str]) -> tuple[int, int, int]:
    """``(widest_text, text_height, baseline)`` for ``lines`` at ``font_scale``."""
    widest = max((cv2.getTextSize(line, _FONT, font_scale, 1)[0][0] for line in lines), default=0)
    (_, text_height), baseline = cv2.getTextSize(_METRIC_SAMPLE, _FONT, font_scale, 1)
    return widest, text_height, baseline


def panel_layout(
    frame_width: int,
    frame_height: int,
    lines: Sequence[str],
    scale: OverlayScale,
    *,
    max_width_fraction: float = MAX_PANEL_WIDTH_FRACTION,
    max_height_fraction: float = MAX_PANEL_HEIGHT_FRACTION,
) -> PanelLayout:
    """Size a text panel so it stays inside its share of the frame.

    Starts at ``scale.font_scale`` and shrinks the font until the panel fits
    both fractional budgets, stopping at the legibility floor. If the panel
    still does not fit at that floor — a frame too small for the text at any
    readable size — the returned dimensions are clipped to the frame, which
    keeps drawing safe rather than raising.

    Args:
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        lines: The text lines the panel will contain.
        scale: Sizes for this frame from :func:`resolve_overlay_scale`.
        max_width_fraction: Largest share of frame width the panel may occupy.
        max_height_fraction: Largest share of frame height the panel may occupy.

    Returns:
        The panel's geometry and the font scale to draw it at.
    """
    if not lines:
        return PanelLayout(scale.font_scale, 0, scale.padding, 0, 0, 0)

    width_budget = max(1, int(frame_width * max_width_fraction))
    height_budget = max(1, int(frame_height * max_height_fraction))
    padding = scale.padding
    font_scale = scale.font_scale

    for _ in range(_MAX_SHRINK_ITERATIONS):
        widest, text_height, baseline = _measure(font_scale, lines)
        line_height = text_height + baseline + scale.line_gap
        width = widest + padding * 2
        height = line_height * len(lines) + padding
        if (width <= width_budget and height <= height_budget) or font_scale <= _MIN_FONT_SCALE:
            break
        font_scale = max(_MIN_FONT_SCALE, font_scale * _SHRINK_STEP)
        # Padding is part of the overflow; give it back proportionally too.
        padding = max(1, round(scale.padding * (font_scale / scale.font_scale)))

    widest, text_height, baseline = _measure(font_scale, lines)
    line_height = text_height + baseline + scale.line_gap
    return PanelLayout(
        font_scale=font_scale,
        line_height=line_height,
        padding=padding,
        width=min(widest + padding * 2, frame_width),
        height=min(line_height * len(lines) + padding, frame_height),
        first_baseline=padding + text_height,
    )

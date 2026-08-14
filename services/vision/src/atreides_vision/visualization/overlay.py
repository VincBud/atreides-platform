"""Frame annotation: zones, boxes, track ids, trails, and the status panel.

Deliberately plain. This is an operator diagnostic view, not a product UI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from atreides_vision.config import VisualizationConfig
from atreides_vision.models import Point, Track
from atreides_vision.visualization.layout import (
    OverlayScale,
    PanelLayout,
    panel_layout,
    resolve_overlay_scale,
)
from atreides_vision.zones.models import Zone, ZoneReport, ZoneSnapshot

if TYPE_CHECKING:
    # Type-only: importing this at runtime pulls in the ``video`` package, whose
    # ``__init__`` imports the processor, which imports this module — a cycle
    # that only bites when ``visualization`` happens to be imported first.
    # Annotations are strings here (``from __future__ import annotations``), so
    # nothing needs it at runtime.
    from atreides_vision.video.trajectories import TrajectoryStore

__all__ = [
    "Annotator",
    "StatusOverlay",
    "ZoneOverlay",
    "color_for_track_id",
    "color_for_zone_id",
    "color_for_zone_index",
]

_FONT: Final[int] = cv2.FONT_HERSHEY_SIMPLEX
_WHITE: Final[tuple[int, int, int]] = (255, 255, 255)
_PANEL_BG: Final[tuple[int, int, int]] = (24, 24, 24)

#: Muted BGR palette. Track ids cycle through it; colour carries no meaning
#: beyond "these boxes are different tracks".
_TRACK_PALETTE: Final[tuple[tuple[int, int, int], ...]] = (
    (180, 190, 90),
    (90, 170, 230),
    (140, 200, 140),
    (200, 150, 200),
    (90, 200, 230),
    (210, 160, 110),
    (160, 160, 235),
    (190, 190, 190),
)

#: Cooler, desaturated palette for zone boundaries, so zones read as background
#: structure and never compete with the track boxes in front of them.
_ZONE_PALETTE: Final[tuple[tuple[int, int, int], ...]] = (
    (200, 170, 110),
    (130, 190, 170),
    (170, 150, 210),
    (190, 190, 130),
    (150, 175, 200),
    (200, 140, 160),
)


def color_for_track_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR colour for a track id."""
    return _TRACK_PALETTE[track_id % len(_TRACK_PALETTE)]


def color_for_zone_index(index: int) -> tuple[int, int, int]:
    """BGR colour for the n-th configured zone.

    Preferred over :func:`color_for_zone_id`: assigning by position guarantees
    that the first ``len(_ZONE_PALETTE)`` zones are all visually distinct,
    where hashing ids can collide and render two zones identically.
    """
    return _ZONE_PALETTE[index % len(_ZONE_PALETTE)]


def color_for_zone_id(zone_id: str) -> tuple[int, int, int]:
    """Deterministic BGR colour for a zone id, stable across runs.

    Used when a zone is drawn outside the context of a full layout. Uses FNV-1a
    rather than :func:`hash`, whose string seed varies between processes.
    """
    digest = 2166136261
    for byte in zone_id.encode("utf-8"):
        digest = ((digest ^ byte) * 16777619) & 0xFFFFFFFF
    return _ZONE_PALETTE[digest % len(_ZONE_PALETTE)]


@dataclass(frozen=True, slots=True)
class ZoneOverlay:
    """The zones to draw and their current metrics."""

    zones: tuple[Zone, ...]
    snapshots: Mapping[str, ZoneSnapshot]

    @classmethod
    def from_report(cls, zones: Sequence[Zone], report: ZoneReport | None) -> ZoneOverlay:
        """Pair configured zones with the metrics from a report."""
        snapshots = {} if report is None else {s.zone_id: s for s in report.snapshots}
        return cls(zones=tuple(zones), snapshots=snapshots)


@dataclass(frozen=True, slots=True)
class StatusOverlay:
    """Numbers shown in the top-left processing panel.

    ``zone_count`` is ``None`` for a V0.1 run without zones, which keeps that
    panel exactly as it was.
    """

    frame_index: int
    processing_fps: float
    track_count: int
    zone_count: int | None = None

    def lines(self, compact: bool = False) -> tuple[str, ...]:
        """The panel's text.

        ``compact`` abbreviates the labels for small frames. The title carries
        the most width for the least information, so it is what gets cut first;
        the live numbers are the point of the panel and are always kept.
        """
        title = "ATREIDES" if compact else "ATREIDES // SEE + TRACK"
        if self.zone_count is None:
            if compact:
                return (
                    title,
                    f"F {self.frame_index}",
                    f"{self.processing_fps:.1f} FPS",
                    f"Trk {self.track_count}",
                )
            return (
                title,
                f"Frame: {self.frame_index}",
                f"Processing: {self.processing_fps:.1f} FPS",
                f"Tracks: {self.track_count}",
            )
        if compact:
            return (
                "ATREIDES",
                f"F {self.frame_index}",
                f"{self.processing_fps:.1f} FPS",
                f"Trk {self.track_count}",
                f"Zn {self.zone_count}",
            )
        return (
            "ATREIDES // ZONE INTELLIGENCE",
            f"Frame: {self.frame_index}",
            f"Processing: {self.processing_fps:.1f} FPS",
            f"Tracks: {self.track_count}",
            f"Zones: {self.zone_count}",
        )


class Annotator:
    """Draws Atreides tracks and status onto BGR frames.

    Drawing happens **in place** on the frame passed in — the array is also
    returned for convenience. The pipeline owns the frame at that point, and
    avoiding a full-frame copy per frame matters on CPU.
    """

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self._config = config or VisualizationConfig()

    @property
    def config(self) -> VisualizationConfig:
        return self._config

    def scale_for(self, frame: np.ndarray) -> OverlayScale:
        """Overlay sizes for this frame's dimensions.

        Every draw method resolves this itself when not given one, so each stays
        usable standalone; :meth:`annotate` resolves it once and passes it down.
        """
        height, width = frame.shape[:2]
        return resolve_overlay_scale(
            width,
            height,
            base_font_scale=self._config.font_scale,
            base_box_thickness=self._config.box_thickness,
            base_trail_thickness=self._config.trail_thickness,
            base_outline_thickness=self._config.zone_outline_thickness,
        )

    def annotate(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        trajectories: TrajectoryStore | None = None,
        status: StatusOverlay | None = None,
        zones: ZoneOverlay | None = None,
    ) -> np.ndarray:
        """Draw zones, then tracks, then the status panel. Returns the same array.

        Order matters: zones are venue structure and belong behind the people
        standing in them.
        """
        scale = self.scale_for(frame)
        if zones is not None and self._config.draw_zones:
            # The status panel is drawn last and would otherwise cover the label
            # of any zone anchored in the top-left corner, hiding its metrics.
            reserved = None if status is None else self._status_panel_size(frame, status, scale)
            self.draw_zones(frame, zones, reserved, scale)
        for track in tracks:
            color = color_for_track_id(track.track_id)
            if trajectories is not None:
                self.draw_trail(frame, trajectories.trail(track.track_id), color, scale)
            self.draw_track(frame, track, color, scale)
        if status is not None:
            self.draw_status(frame, status, scale)
        return frame

    def draw_zones(
        self,
        frame: np.ndarray,
        zones: ZoneOverlay,
        reserved: tuple[int, int] | None = None,
        scale: OverlayScale | None = None,
    ) -> None:
        """Draw every configured zone boundary and its metrics panel.

        Args:
            frame: Frame to draw on, in place.
            zones: Zones and their current metrics.
            reserved: ``(width, height)`` of the top-left region occupied by the
                status panel, which zone labels must not hide behind.
            scale: Overlay sizes for this frame; resolved from the frame if omitted.
        """
        scale = scale or self.scale_for(frame)
        for index, zone in enumerate(zones.zones):
            # Colour by position so configured zones are always distinguishable.
            self.draw_zone(
                frame,
                zone,
                zones.snapshots.get(zone.id),
                color=color_for_zone_index(index),
                reserved=reserved,
                scale=scale,
            )

    def draw_zone(
        self,
        frame: np.ndarray,
        zone: Zone,
        snapshot: ZoneSnapshot | None = None,
        color: tuple[int, int, int] | None = None,
        reserved: tuple[int, int] | None = None,
        scale: OverlayScale | None = None,
    ) -> None:
        """Draw one zone polygon, its name, and its current metrics."""
        scale = scale or self.scale_for(frame)
        height, width = frame.shape[:2]
        polygon = zone.polygon.to_pixel_array(width, height)
        color = color or color_for_zone_id(zone.id)

        if self._config.zone_fill_alpha > 0:
            self._fill_polygon(frame, polygon, color, self._config.zone_fill_alpha)

        cv2.polylines(
            frame,
            [polygon],
            isClosed=True,
            color=color,
            thickness=scale.outline_thickness,
            lineType=cv2.LINE_AA,
        )

        anchor = zone.polygon.top_left.to_pixels(width, height).as_int_tuple()
        self._draw_zone_panel(frame, zone, snapshot, anchor, color, reserved, scale)

    def _fill_polygon(
        self,
        frame: np.ndarray,
        polygon: np.ndarray,
        color: tuple[int, int, int],
        alpha: float,
    ) -> None:
        """Tint the polygon's interior.

        Blending is confined to the polygon's bounding rectangle so the cost
        stays proportional to zone size rather than frame size.
        """
        x, y, w, h = cv2.boundingRect(polygon)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            return

        region = frame[y0:y1, x0:x1]
        tinted = region.copy()
        cv2.fillPoly(tinted, [polygon - np.array([[x0, y0]], dtype=np.int32)], color)
        cv2.addWeighted(tinted, alpha, region, 1.0 - alpha, 0.0, dst=region)

    def _draw_zone_panel(
        self,
        frame: np.ndarray,
        zone: Zone,
        snapshot: ZoneSnapshot | None,
        anchor: tuple[int, int],
        color: tuple[int, int, int],
        reserved: tuple[int, int] | None = None,
        scale: OverlayScale | None = None,
    ) -> None:
        """Draw the zone's name and metrics just inside its top-left corner."""
        scale = scale or self.scale_for(frame)
        lines = [zone.name.upper()]
        if snapshot is not None:
            if scale.compact:
                lines.append(f"P {snapshot.occupancy}")
                lines.append(f"M {snapshot.motion_level:.2f}")
                lines.append(f"D {snapshot.average_dwell_seconds:.1f}s")
                if self._config.draw_zone_counters:
                    lines.append(f"{snapshot.entries}/{snapshot.exits}")
            else:
                lines.append(f"People: {snapshot.occupancy}")
                lines.append(f"Motion: {snapshot.motion_level:.2f}")
                lines.append(f"Avg dwell: {snapshot.average_dwell_seconds:.1f}s")
                if self._config.draw_zone_counters:
                    lines.append(f"In {snapshot.entries} / Out {snapshot.exits}")

        layout = panel_layout(frame.shape[1], frame.shape[0], lines, scale)
        panel_width, panel_height = layout.width, layout.height

        left = max(0, min(anchor[0], frame.shape[1] - panel_width))
        top = max(0, min(anchor[1], frame.shape[0] - panel_height))

        # Slide clear of the status panel rather than hiding underneath it.
        if reserved is not None:
            reserved_width, reserved_height = reserved
            overlaps = left < reserved_width and top < reserved_height
            if overlaps and reserved_height + panel_height <= frame.shape[0]:
                top = reserved_height
            elif overlaps:
                left = min(reserved_width, frame.shape[1] - panel_width)

        region = frame[top : top + panel_height, left : left + panel_width]
        backing = np.full(region.shape, _PANEL_BG, dtype=np.uint8)
        cv2.addWeighted(backing, 0.6, region, 0.4, 0.0, dst=region)

        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (left + layout.padding, top + layout.baseline(index)),
                _FONT,
                layout.font_scale,
                # The zone name takes the zone's colour; metrics stay neutral.
                color if index == 0 else _WHITE,
                1,
                lineType=cv2.LINE_AA,
            )

    def draw_track(
        self,
        frame: np.ndarray,
        track: Track,
        color: tuple[int, int, int] | None = None,
        scale: OverlayScale | None = None,
    ) -> None:
        """Draw one bounding box and its ``ID <n>`` label."""
        scale = scale or self.scale_for(frame)
        color = color or color_for_track_id(track.track_id)
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = track.bbox_xyxy.clipped_to(width, height).as_int_tuple()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, scale.box_thickness)

        label = f"ID {track.track_id}"
        if self._config.draw_confidence and track.confidence is not None:
            label = f"{label}  {track.confidence:.2f}"
        self._draw_label(frame, label, anchor=(x1, y1), color=color, scale=scale)

    def draw_trail(
        self,
        frame: np.ndarray,
        points: Sequence[Point],
        color: tuple[int, int, int],
        scale: OverlayScale | None = None,
    ) -> None:
        """Draw the recent path of a track as a polyline."""
        if len(points) < 2:
            return
        scale = scale or self.scale_for(frame)
        polyline = np.array([point.as_int_tuple() for point in points], dtype=np.int32)
        cv2.polylines(
            frame,
            [polyline],
            isClosed=False,
            color=color,
            thickness=scale.trail_thickness,
            lineType=cv2.LINE_AA,
        )

    def _status_layout(
        self, frame: np.ndarray, status: StatusOverlay, scale: OverlayScale
    ) -> PanelLayout:
        return panel_layout(frame.shape[1], frame.shape[0], status.lines(scale.compact), scale)

    def _status_panel_size(
        self, frame: np.ndarray, status: StatusOverlay, scale: OverlayScale | None = None
    ) -> tuple[int, int]:
        """The ``(width, height)`` the status panel will occupy, in pixels."""
        scale = scale or self.scale_for(frame)
        layout = self._status_layout(frame, status, scale)
        return layout.width, layout.height

    def draw_status(
        self, frame: np.ndarray, status: StatusOverlay, scale: OverlayScale | None = None
    ) -> None:
        """Draw the top-left processing panel."""
        scale = scale or self.scale_for(frame)
        lines = status.lines(scale.compact)
        layout = self._status_layout(frame, status, scale)
        panel_width, panel_height = layout.width, layout.height

        # Translucent backing so text stays readable over bright footage.
        region = frame[0:panel_height, 0:panel_width]
        backing = np.full(region.shape, _PANEL_BG, dtype=np.uint8)
        cv2.addWeighted(backing, 0.55, region, 0.45, 0.0, dst=region)

        rule_offset = max(1, round(5 * scale.factor))
        for index, line in enumerate(lines):
            y = layout.baseline(index)
            cv2.putText(
                frame,
                line,
                (layout.padding, y),
                _FONT,
                layout.font_scale,
                _WHITE,
                1,
                lineType=cv2.LINE_AA,
            )
            if index == 0 and y + rule_offset < panel_height:
                # Rule under the title, to separate branding from live numbers.
                cv2.line(
                    frame,
                    (layout.padding, y + rule_offset),
                    (panel_width - layout.padding, y + rule_offset),
                    (110, 110, 110),
                    1,
                )

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        anchor: tuple[int, int],
        color: tuple[int, int, int],
        scale: OverlayScale | None = None,
    ) -> None:
        scale = scale or self.scale_for(frame)
        font_scale = scale.font_scale
        pad = scale.label_padding
        (text_width, text_height), baseline = cv2.getTextSize(text, _FONT, font_scale, 1)
        x, y = anchor
        box_height = text_height + baseline + pad * 2
        box_width = text_width + pad * 2

        # Flip the label inside the box when the track is at the top edge.
        top = y - box_height if y - box_height >= 0 else y
        top = max(0, min(top, frame.shape[0] - box_height))
        left = max(0, min(x, frame.shape[1] - box_width))

        cv2.rectangle(
            frame,
            (left, top),
            (left + box_width, top + box_height),
            color,
            thickness=-1,
        )
        cv2.putText(
            frame,
            text,
            (left + pad, top + text_height + pad),
            _FONT,
            font_scale,
            (20, 20, 20),
            1,
            lineType=cv2.LINE_AA,
        )

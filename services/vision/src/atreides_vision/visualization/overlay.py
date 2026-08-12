"""Frame annotation: boxes, track ids, trails, and the status panel.

Deliberately plain. This is an operator diagnostic view, not a product UI.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from atreides_vision.config import VisualizationConfig
from atreides_vision.models import Point, Track
from atreides_vision.video.trajectories import TrajectoryStore

__all__ = ["Annotator", "StatusOverlay", "color_for_track_id"]

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


def color_for_track_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR colour for a track id."""
    return _TRACK_PALETTE[track_id % len(_TRACK_PALETTE)]


@dataclass(frozen=True, slots=True)
class StatusOverlay:
    """Numbers shown in the top-left processing panel."""

    frame_index: int
    processing_fps: float
    track_count: int

    def lines(self) -> tuple[str, ...]:
        return (
            "ATREIDES // SEE + TRACK",
            f"Frame: {self.frame_index}",
            f"Processing: {self.processing_fps:.1f} FPS",
            f"Tracks: {self.track_count}",
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

    def annotate(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        trajectories: TrajectoryStore | None = None,
        status: StatusOverlay | None = None,
    ) -> np.ndarray:
        """Draw every track, then the status panel. Returns the same array."""
        for track in tracks:
            color = color_for_track_id(track.track_id)
            if trajectories is not None:
                self.draw_trail(frame, trajectories.trail(track.track_id), color)
            self.draw_track(frame, track, color)
        if status is not None:
            self.draw_status(frame, status)
        return frame

    def draw_track(
        self, frame: np.ndarray, track: Track, color: tuple[int, int, int] | None = None
    ) -> None:
        """Draw one bounding box and its ``ID <n>`` label."""
        color = color or color_for_track_id(track.track_id)
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = track.bbox_xyxy.clipped_to(width, height).as_int_tuple()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self._config.box_thickness)

        label = f"ID {track.track_id}"
        if self._config.draw_confidence and track.confidence is not None:
            label = f"{label}  {track.confidence:.2f}"
        self._draw_label(frame, label, anchor=(x1, y1), color=color)

    def draw_trail(
        self,
        frame: np.ndarray,
        points: Sequence[Point],
        color: tuple[int, int, int],
    ) -> None:
        """Draw the recent path of a track as a polyline."""
        if len(points) < 2:
            return
        polyline = np.array([point.as_int_tuple() for point in points], dtype=np.int32)
        cv2.polylines(
            frame,
            [polyline],
            isClosed=False,
            color=color,
            thickness=self._config.trail_thickness,
            lineType=cv2.LINE_AA,
        )

    def draw_status(self, frame: np.ndarray, status: StatusOverlay) -> None:
        """Draw the top-left processing panel."""
        lines = status.lines()
        scale = self._config.font_scale
        line_height = int(22 * max(scale / 0.5, 1.0))
        padding = 8

        text_widths = [cv2.getTextSize(line, _FONT, scale, 1)[0][0] for line in lines]
        panel_width = max(text_widths) + padding * 2
        panel_height = line_height * len(lines) + padding

        panel_width = min(panel_width, frame.shape[1])
        panel_height = min(panel_height, frame.shape[0])

        # Translucent backing so text stays readable over bright footage.
        region = frame[0:panel_height, 0:panel_width]
        backing = np.full(region.shape, _PANEL_BG, dtype=np.uint8)
        cv2.addWeighted(backing, 0.55, region, 0.45, 0.0, dst=region)

        y = padding + int(line_height * 0.65)
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (padding, y),
                _FONT,
                scale,
                _WHITE,
                1,
                lineType=cv2.LINE_AA,
            )
            if index == 0:
                # Rule under the title, to separate branding from live numbers.
                cv2.line(
                    frame,
                    (padding, y + 5),
                    (panel_width - padding, y + 5),
                    (110, 110, 110),
                    1,
                )
            y += line_height

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        anchor: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        scale = self._config.font_scale
        (text_width, text_height), baseline = cv2.getTextSize(text, _FONT, scale, 1)
        x, y = anchor
        box_height = text_height + baseline + 4

        # Flip the label inside the box when the track is at the top edge.
        top = y - box_height if y - box_height >= 0 else y
        top = max(0, min(top, frame.shape[0] - box_height))
        left = max(0, min(x, frame.shape[1] - text_width - 6))

        cv2.rectangle(
            frame,
            (left, top),
            (left + text_width + 6, top + box_height),
            color,
            thickness=-1,
        )
        cv2.putText(
            frame,
            text,
            (left + 3, top + text_height + 2),
            _FONT,
            scale,
            (20, 20, 20),
            1,
            lineType=cv2.LINE_AA,
        )

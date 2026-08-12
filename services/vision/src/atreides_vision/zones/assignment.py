"""Assigning tracked people to zones, one frame at a time.

The step is stateless: given this frame's tracks, which zones is each one
standing in? Everything temporal (entries, dwell, speed) is built on top of a
sequence of these assignments in :mod:`atreides_vision.zones.state`.

Overlapping zones are supported deliberately. A track's floor point is tested
against every configured polygon independently, so one person may count toward
several zones at once — an "entrance" rectangle overlapping a wider "lounge"
region is a legitimate configuration, and each zone reports its own metrics.
No priority ordering or first-match rule is applied.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from atreides_vision.models import Track
from atreides_vision.zones.geometry import NormalizedPoint, track_floor_point
from atreides_vision.zones.models import ZoneLayout, ZoneMembership

__all__ = ["FrameAssignment", "ZoneAssigner"]


@dataclass(frozen=True, slots=True)
class FrameAssignment:
    """The result of assigning one frame's tracks to zones."""

    frame_index: int
    memberships: tuple[ZoneMembership, ...]
    #: Normalized floor position of every track seen this frame, by track id.
    floor_points: dict[int, NormalizedPoint] = field(default_factory=dict)

    def for_zone(self, zone_id: str) -> ZoneMembership | None:
        return next((m for m in self.memberships if m.zone_id == zone_id), None)

    def zones_for_track(self, track_id: int) -> tuple[str, ...]:
        """Every zone this track is currently inside. May be empty, or several."""
        return tuple(m.zone_id for m in self.memberships if track_id in m.track_ids)


class ZoneAssigner:
    """Maps tracks to the zones containing their floor position."""

    def __init__(self, layout: ZoneLayout) -> None:
        self._layout = layout

    @property
    def layout(self) -> ZoneLayout:
        return self._layout

    def assign(
        self,
        tracks: Sequence[Track],
        frame_width: int,
        frame_height: int,
        frame_index: int = 0,
    ) -> FrameAssignment:
        """Assign this frame's tracks to zones.

        Every configured zone appears in the result, including empty ones, so
        downstream code never has to distinguish "zone absent" from "zone empty".
        """
        floor_points = {
            track.track_id: track_floor_point(track, frame_width, frame_height) for track in tracks
        }

        memberships = tuple(
            ZoneMembership(
                zone_id=zone.id,
                track_ids=frozenset(
                    track_id for track_id, point in floor_points.items() if zone.contains(point)
                ),
            )
            for zone in self._layout
        )

        return FrameAssignment(
            frame_index=frame_index,
            memberships=memberships,
            floor_points=floor_points,
        )

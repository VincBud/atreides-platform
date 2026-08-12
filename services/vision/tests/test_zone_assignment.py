"""Per-frame assignment of tracks to zones, including overlapping zones."""

from __future__ import annotations

import pytest

from atreides_vision.models import BoundingBox, Track
from atreides_vision.zones.assignment import ZoneAssigner
from atreides_vision.zones.geometry import NormalizedPoint
from atreides_vision.zones.models import Zone, ZoneLayout, ZonePolygon

FRAME_W, FRAME_H = 1000, 1000


def zone(zone_id: str, x0: float, y0: float, x1: float, y1: float) -> Zone:
    return Zone(
        id=zone_id,
        name=zone_id.title(),
        polygon=ZonePolygon.from_pairs([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]),
    )


def layout(*zones: Zone) -> ZoneLayout:
    return ZoneLayout(camera_id="cam_01", zones=zones)


def track_at(track_id: int, floor_x: float, floor_y: float, box_height: float = 100.0) -> Track:
    """A track whose bottom-centre lands on ``(floor_x, floor_y)`` in normalized units."""
    cx, cy = floor_x * FRAME_W, floor_y * FRAME_H
    return Track(track_id=track_id, bbox_xyxy=BoundingBox(cx - 20, cy - box_height, cx + 20, cy))


class TestNoZones:
    def test_empty_layout_yields_no_memberships(self) -> None:
        assignment = ZoneAssigner(layout()).assign([track_at(1, 0.5, 0.5)], FRAME_W, FRAME_H)
        assert assignment.memberships == ()
        assert assignment.zones_for_track(1) == ()

    def test_floor_points_are_still_computed(self) -> None:
        """Movement tracking needs positions even with no zones configured."""
        assignment = ZoneAssigner(layout()).assign([track_at(1, 0.5, 0.5)], FRAME_W, FRAME_H)
        assert assignment.floor_points[1] == NormalizedPoint(0.5, 0.5)


class TestSingleZone:
    def test_track_inside(self) -> None:
        assigner = ZoneAssigner(layout(zone("bar", 0.0, 0.0, 0.5, 0.5)))
        assignment = assigner.assign([track_at(7, 0.25, 0.25)], FRAME_W, FRAME_H)
        assert assignment.for_zone("bar").track_ids == frozenset({7})  # type: ignore[union-attr]

    def test_track_outside(self) -> None:
        assigner = ZoneAssigner(layout(zone("bar", 0.0, 0.0, 0.5, 0.5)))
        assignment = assigner.assign([track_at(7, 0.9, 0.9)], FRAME_W, FRAME_H)
        assert assignment.for_zone("bar").track_ids == frozenset()  # type: ignore[union-attr]

    def test_empty_zone_is_still_reported(self) -> None:
        """Downstream never has to distinguish 'absent' from 'empty'."""
        assigner = ZoneAssigner(layout(zone("bar", 0.0, 0.0, 0.5, 0.5)))
        assignment = assigner.assign([], FRAME_W, FRAME_H)
        assert len(assignment.memberships) == 1
        assert assignment.for_zone("bar").occupancy == 0  # type: ignore[union-attr]

    def test_uses_feet_not_box_centre(self) -> None:
        """A tall person whose centre is above the zone still counts as inside it."""
        floor_zone = zone("floor", 0.0, 0.6, 1.0, 1.0)
        # Box spans y 0.3..0.7: centre at 0.5 (outside), feet at 0.7 (inside).
        track = Track(track_id=1, bbox_xyxy=BoundingBox(400, 300, 600, 700))
        assignment = ZoneAssigner(layout(floor_zone)).assign([track], FRAME_W, FRAME_H)
        assert assignment.zones_for_track(1) == ("floor",)


class TestMultipleZones:
    def test_disjoint_zones_each_get_their_own(self) -> None:
        assigner = ZoneAssigner(
            layout(zone("left", 0.0, 0.0, 0.4, 1.0), zone("right", 0.6, 0.0, 1.0, 1.0))
        )
        assignment = assigner.assign(
            [track_at(1, 0.2, 0.5), track_at(2, 0.8, 0.5), track_at(3, 0.5, 0.5)],
            FRAME_W,
            FRAME_H,
        )
        assert assignment.for_zone("left").track_ids == frozenset({1})  # type: ignore[union-attr]
        assert assignment.for_zone("right").track_ids == frozenset({2})  # type: ignore[union-attr]
        assert assignment.zones_for_track(3) == (), "track 3 is between the zones"

    def test_membership_order_follows_configuration(self) -> None:
        assigner = ZoneAssigner(
            layout(zone("a", 0.0, 0.0, 1.0, 1.0), zone("b", 0.0, 0.0, 1.0, 1.0))
        )
        assignment = assigner.assign([], FRAME_W, FRAME_H)
        assert [m.zone_id for m in assignment.memberships] == ["a", "b"]


class TestOverlappingZones:
    """Documented behaviour: overlaps are allowed and a track can be in several."""

    def test_track_in_the_overlap_belongs_to_both(self) -> None:
        assigner = ZoneAssigner(
            layout(zone("dancefloor", 0.0, 0.0, 0.6, 1.0), zone("entrance", 0.4, 0.0, 1.0, 1.0))
        )
        assignment = assigner.assign([track_at(5, 0.5, 0.5)], FRAME_W, FRAME_H)
        assert set(assignment.zones_for_track(5)) == {"dancefloor", "entrance"}

    def test_zones_outside_the_overlap_are_unaffected(self) -> None:
        assigner = ZoneAssigner(
            layout(zone("dancefloor", 0.0, 0.0, 0.6, 1.0), zone("entrance", 0.4, 0.0, 1.0, 1.0))
        )
        assignment = assigner.assign(
            [track_at(1, 0.1, 0.5), track_at(2, 0.5, 0.5), track_at(3, 0.9, 0.5)],
            FRAME_W,
            FRAME_H,
        )
        assert assignment.for_zone("dancefloor").track_ids == frozenset({1, 2})  # type: ignore[union-attr]
        assert assignment.for_zone("entrance").track_ids == frozenset({2, 3})  # type: ignore[union-attr]

    def test_a_fully_nested_zone_double_counts_deliberately(self) -> None:
        assigner = ZoneAssigner(
            layout(zone("venue", 0.0, 0.0, 1.0, 1.0), zone("vip", 0.4, 0.4, 0.6, 0.6))
        )
        assignment = assigner.assign([track_at(1, 0.5, 0.5)], FRAME_W, FRAME_H)
        assert assignment.for_zone("venue").occupancy == 1  # type: ignore[union-attr]
        assert assignment.for_zone("vip").occupancy == 1  # type: ignore[union-attr]


class TestFrameSize:
    def test_assignment_is_resolution_independent(self) -> None:
        assigner = ZoneAssigner(layout(zone("bar", 0.0, 0.0, 0.5, 0.5)))
        # The same person, filmed at two resolutions.
        sd = Track(track_id=1, bbox_xyxy=BoundingBox(100, 50, 140, 100))  # feet at (0.25, 0.25)
        hd = Track(track_id=1, bbox_xyxy=BoundingBox(400, 200, 560, 400))
        assert assigner.assign([sd], 480, 400).zones_for_track(1) == ("bar",)
        assert assigner.assign([hd], 1920, 1600).zones_for_track(1) == ("bar",)

    @pytest.mark.parametrize(("width", "height"), [(0, 100), (100, 0)])
    def test_rejects_invalid_frame_size(self, width: int, height: int) -> None:
        assigner = ZoneAssigner(layout(zone("bar", 0.0, 0.0, 0.5, 0.5)))
        with pytest.raises(ValueError, match="Frame size must be positive"):
            assigner.assign([track_at(1, 0.2, 0.2)], width, height)

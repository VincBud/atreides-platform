"""Zone domain model validation and JSON-shaped construction."""

from __future__ import annotations

from typing import Any

import pytest

from atreides_vision.errors import ZoneConfigurationError
from atreides_vision.zones.geometry import NormalizedPoint
from atreides_vision.zones.models import (
    Zone,
    ZoneLayout,
    ZoneMembership,
    ZonePolygon,
    ZoneReport,
    ZoneSnapshot,
)

SQUARE_PAIRS = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


def valid_zone(zone_id: str = "dancefloor", name: str = "Dancefloor") -> dict[str, Any]:
    return {"id": zone_id, "name": name, "polygon": list(SQUARE_PAIRS)}


def valid_config(*zones: dict[str, Any]) -> dict[str, Any]:
    return {"camera_id": "cam_01", "zones": list(zones) or [valid_zone()]}


class TestZonePolygon:
    def test_accepts_a_valid_polygon(self) -> None:
        polygon = ZonePolygon.from_pairs(SQUARE_PAIRS)
        assert len(polygon.vertices) == 4
        assert polygon.vertices[0] == NormalizedPoint(0.1, 0.1)

    def test_accepts_a_triangle(self) -> None:
        assert len(ZonePolygon.from_pairs([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]).vertices) == 3

    @pytest.mark.parametrize(
        "pairs",
        [
            [],
            [[0.1, 0.1]],
            [[0.1, 0.1], [0.9, 0.9]],
        ],
    )
    def test_rejects_fewer_than_three_points(self, pairs: list[list[float]]) -> None:
        with pytest.raises(ZoneConfigurationError, match="at least 3 points"):
            ZonePolygon.from_pairs(pairs)

    @pytest.mark.parametrize(
        "bad_point",
        [[1.5, 0.5], [-0.1, 0.5], [0.5, 1.2], [0.5, -2.0]],
    )
    def test_rejects_coordinates_outside_the_unit_range(self, bad_point: list[float]) -> None:
        with pytest.raises(ZoneConfigurationError, match=r"between 0\.0 and 1\.0"):
            ZonePolygon.from_pairs([[0.1, 0.1], [0.9, 0.1], bad_point])

    def test_accepts_coordinates_exactly_on_the_boundary(self) -> None:
        assert ZonePolygon.from_pairs([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    @pytest.mark.parametrize(
        "pairs",
        [
            [[0.1, 0.1], [0.9, 0.1], [0.9]],
            [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9, 0.5]],
        ],
    )
    def test_rejects_points_that_are_not_pairs(self, pairs: list[list[float]]) -> None:
        with pytest.raises(ZoneConfigurationError, match=r"must be a \[x, y\] pair"):
            ZonePolygon.from_pairs(pairs)

    @pytest.mark.parametrize("bad", [["a", 0.5], [None, 0.5], [True, 0.5]])
    def test_rejects_non_numeric_coordinates(self, bad: list[Any]) -> None:
        with pytest.raises(ZoneConfigurationError, match="must contain two numbers"):
            ZonePolygon.from_pairs([[0.1, 0.1], [0.9, 0.1], bad])

    def test_contains_delegates_to_geometry(self) -> None:
        polygon = ZonePolygon.from_pairs(SQUARE_PAIRS)
        assert polygon.contains(NormalizedPoint(0.5, 0.5))
        assert not polygon.contains(NormalizedPoint(0.95, 0.95))

    def test_to_pixel_array_scales_to_the_frame(self) -> None:
        polygon = ZonePolygon.from_pairs([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        pixels = polygon.to_pixel_array(640, 360)
        assert pixels.shape == (4, 2)
        assert pixels.tolist() == [[0, 0], [640, 0], [640, 360], [0, 360]]

    def test_top_left_is_the_bounding_box_corner(self) -> None:
        polygon = ZonePolygon.from_pairs([[0.3, 0.4], [0.8, 0.2], [0.9, 0.9]])
        assert polygon.top_left == NormalizedPoint(0.3, 0.2)

    def test_round_trips_through_pairs(self) -> None:
        polygon = ZonePolygon.from_pairs(SQUARE_PAIRS)
        assert ZonePolygon.from_pairs([list(p) for p in polygon.as_pairs()]) == polygon


class TestZone:
    def test_valid_zone(self) -> None:
        zone = Zone.from_mapping(valid_zone(), position=0)
        assert (zone.id, zone.name) == ("dancefloor", "Dancefloor")

    @pytest.mark.parametrize("name", ["", "   "])
    def test_rejects_empty_name(self, name: str) -> None:
        with pytest.raises(ZoneConfigurationError, match="non-empty name"):
            Zone.from_mapping(valid_zone(name=name), position=0)

    @pytest.mark.parametrize("zone_id", ["", "  "])
    def test_rejects_empty_id(self, zone_id: str) -> None:
        with pytest.raises(ZoneConfigurationError, match="non-empty string"):
            Zone.from_mapping(valid_zone(zone_id=zone_id), position=0)

    @pytest.mark.parametrize("field", ["id", "name", "polygon"])
    def test_rejects_missing_field(self, field: str) -> None:
        data = valid_zone()
        del data[field]
        with pytest.raises(ZoneConfigurationError, match=f"missing required field '{field}'"):
            Zone.from_mapping(data, position=2)

    def test_error_mentions_the_position_and_id(self) -> None:
        data = valid_zone()
        data["polygon"] = [[0.1, 0.1]]
        with pytest.raises(ZoneConfigurationError, match=r"zones\[3\] \('dancefloor'\)"):
            Zone.from_mapping(data, position=3)

    def test_rejects_non_string_id(self) -> None:
        data = valid_zone()
        data["id"] = 7
        with pytest.raises(ZoneConfigurationError, match="id must be a string"):
            Zone.from_mapping(data, position=0)

    def test_rejects_polygon_that_is_not_a_list(self) -> None:
        data = valid_zone()
        data["polygon"] = "0.1,0.1"
        with pytest.raises(ZoneConfigurationError, match="must be an array"):
            Zone.from_mapping(data, position=0)


class TestZoneLayout:
    def test_valid_layout(self) -> None:
        layout = ZoneLayout.from_mapping(
            valid_config(valid_zone("bar", "Bar"), valid_zone("dancefloor", "Dancefloor"))
        )
        assert layout.camera_id == "cam_01"
        assert layout.zone_ids == ("bar", "dancefloor")
        assert len(layout) == 2

    def test_rejects_duplicate_zone_ids(self) -> None:
        with pytest.raises(ZoneConfigurationError, match="Duplicate zone id 'bar'"):
            ZoneLayout.from_mapping(
                valid_config(valid_zone("bar", "Bar"), valid_zone("bar", "Bar 2"))
            )

    @pytest.mark.parametrize("field", ["camera_id", "zones"])
    def test_rejects_missing_top_level_field(self, field: str) -> None:
        data = valid_config()
        del data[field]
        with pytest.raises(ZoneConfigurationError, match=f"missing required field '{field}'"):
            ZoneLayout.from_mapping(data)

    def test_rejects_empty_camera_id(self) -> None:
        data = valid_config()
        data["camera_id"] = ""
        with pytest.raises(ZoneConfigurationError, match="camera_id must be a non-empty"):
            ZoneLayout.from_mapping(data)

    def test_rejects_non_object_root(self) -> None:
        with pytest.raises(ZoneConfigurationError, match="must be a JSON object"):
            ZoneLayout.from_mapping([])  # type: ignore[arg-type]

    def test_rejects_zones_that_are_not_a_list(self) -> None:
        data = valid_config()
        data["zones"] = {"id": "bar"}
        with pytest.raises(ZoneConfigurationError, match="'zones' must be an array"):
            ZoneLayout.from_mapping(data)

    def test_allows_an_empty_zone_list(self) -> None:
        layout = ZoneLayout.from_mapping({"camera_id": "cam_01", "zones": []})
        assert len(layout) == 0

    def test_get_finds_and_misses(self) -> None:
        layout = ZoneLayout.from_mapping(valid_config(valid_zone("bar", "Bar")))
        assert layout.get("bar") is not None
        assert layout.get("nope") is None

    def test_is_iterable(self) -> None:
        layout = ZoneLayout.from_mapping(valid_config(valid_zone("a", "A"), valid_zone("b", "B")))
        assert [zone.id for zone in layout] == ["a", "b"]


class TestZoneMembership:
    def test_occupancy_counts_tracks(self) -> None:
        assert ZoneMembership("bar", frozenset({1, 2, 3})).occupancy == 3

    def test_empty_membership(self) -> None:
        assert ZoneMembership("bar", frozenset()).occupancy == 0


class TestZoneSnapshotRecords:
    def snapshot(self, **overrides: Any) -> ZoneSnapshot:
        defaults: dict[str, Any] = {
            "zone_id": "dancefloor",
            "zone_name": "Dancefloor",
            "occupancy": 27,
            "entries": 8,
            "exits": 4,
            "mean_normalized_speed": 0.0831234,
            "motion_level": 0.5432,
            "average_dwell_seconds": 2.8412,
            "max_dwell_seconds": 4.0,
            "completed_visits": 4,
        }
        return ZoneSnapshot(**{**defaults, **overrides})

    def test_record_shape_and_rounding(self) -> None:
        record = self.snapshot().to_record()
        assert record["id"] == "dancefloor"
        assert record["occupancy"] == 27
        assert record["entries"] == 8
        assert record["exits"] == 4
        assert record["motion_level"] == 0.543
        assert record["average_dwell_seconds"] == 2.84
        assert record["mean_normalized_speed"] == 0.08312

    def test_report_record_shape(self) -> None:
        report = ZoneReport(frame_index=120, timestamp_seconds=4.0, snapshots=(self.snapshot(),))
        record = report.to_record()
        assert record["frame"] == 120
        assert record["timestamp_seconds"] == 4.0
        assert len(record["zones"]) == 1
        assert record["zones"][0]["id"] == "dancefloor"

    def test_report_lookup(self) -> None:
        report = ZoneReport(1, 0.0, (self.snapshot(),))
        assert report.for_zone("dancefloor") is not None
        assert report.for_zone("bar") is None
        assert len(report) == 1

    def test_record_carries_no_track_identifiers(self) -> None:
        """Metrics describe zones, never individuals."""
        record = ZoneReport(1, 0.0, (self.snapshot(),)).to_record()
        serialized = str(record)
        assert "track" not in serialized.lower()

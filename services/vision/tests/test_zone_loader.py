"""Loading zone configuration from disk, including the shipped example."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atreides_vision.errors import ZoneConfigurationError
from atreides_vision.zones.loader import load_zone_layout

VALID_CONFIG = {
    "camera_id": "cam_01",
    "zones": [
        {"id": "bar", "name": "Bar", "polygon": [[0.05, 0.05], [0.95, 0.05], [0.95, 0.22]]},
    ],
}


def write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class TestLoading:
    def test_loads_a_valid_file(self, tmp_path: Path) -> None:
        path = write(tmp_path / "zones.json", json.dumps(VALID_CONFIG))
        layout = load_zone_layout(path)
        assert layout.camera_id == "cam_01"
        assert layout.zone_ids == ("bar",)

    def test_accepts_a_string_path(self, tmp_path: Path) -> None:
        path = write(tmp_path / "zones.json", json.dumps(VALID_CONFIG))
        assert load_zone_layout(str(path)).zone_ids == ("bar",)


class TestFailureModes:
    def test_missing_file_points_at_the_example(self, tmp_path: Path) -> None:
        with pytest.raises(ZoneConfigurationError) as excinfo:
            load_zone_layout(tmp_path / "nope.json")
        message = str(excinfo.value)
        assert "not found" in message
        assert "zones.example.json" in message

    def test_directory_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ZoneConfigurationError, match="not a file"):
            load_zone_layout(tmp_path)

    def test_malformed_json_reports_the_location(self, tmp_path: Path) -> None:
        path = write(tmp_path / "zones.json", '{"camera_id": "cam_01", "zones": [ }')
        with pytest.raises(ZoneConfigurationError) as excinfo:
            load_zone_layout(path)
        message = str(excinfo.value)
        assert "not valid JSON" in message
        assert "line" in message

    def test_empty_file_is_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path / "zones.json", "")
        with pytest.raises(ZoneConfigurationError, match="not valid JSON"):
            load_zone_layout(path)

    def test_invalid_content_includes_the_file_name(self, tmp_path: Path) -> None:
        bad = {"camera_id": "cam_01", "zones": [{"id": "a", "name": "A", "polygon": [[0.1, 0.1]]}]}
        path = write(tmp_path / "broken.json", json.dumps(bad))
        with pytest.raises(ZoneConfigurationError) as excinfo:
            load_zone_layout(path)
        assert "broken.json" in str(excinfo.value)
        assert "at least 3 points" in str(excinfo.value)

    def test_duplicate_ids_surface_through_the_loader(self, tmp_path: Path) -> None:
        polygon = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]
        bad = {
            "camera_id": "cam_01",
            "zones": [
                {"id": "bar", "name": "Bar", "polygon": polygon},
                {"id": "bar", "name": "Bar Again", "polygon": polygon},
            ],
        }
        path = write(tmp_path / "zones.json", json.dumps(bad))
        with pytest.raises(ZoneConfigurationError, match="Duplicate zone id"):
            load_zone_layout(path)

    def test_empty_zone_list_warns_but_loads(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = write(tmp_path / "zones.json", json.dumps({"camera_id": "cam_01", "zones": []}))
        import logging

        with caplog.at_level(logging.WARNING, logger="atreides_vision.zones.loader"):
            layout = load_zone_layout(path)
        assert len(layout) == 0
        assert "no zones" in caplog.text


class TestShippedExample:
    """The example config must actually load — it is the documented starting point."""

    @property
    def example_path(self) -> Path:
        # tests/ -> vision/ -> services/ -> repo root
        return Path(__file__).resolve().parents[3] / "config" / "zones.example.json"

    def test_example_exists(self) -> None:
        assert self.example_path.is_file(), f"missing {self.example_path}"

    def test_example_loads_and_has_named_zones(self) -> None:
        layout = load_zone_layout(self.example_path)
        assert len(layout) >= 2
        assert all(zone.name for zone in layout)
        assert len(set(layout.zone_ids)) == len(layout)

    def test_example_polygons_are_normalized(self) -> None:
        for zone in load_zone_layout(self.example_path):
            for vertex in zone.polygon.vertices:
                assert 0.0 <= vertex.x <= 1.0
                assert 0.0 <= vertex.y <= 1.0

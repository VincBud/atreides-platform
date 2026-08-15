"""Camera identity resolution: precedence, and refusing to guess.

The camera id is half of every subject reference in the behaviour stream, so a
wrong one silently mislabels every record in the file.
"""

from __future__ import annotations

import pytest

from atreides_vision.config import resolve_camera_id
from atreides_vision.errors import ConfigurationError


class TestPrecedence:
    def test_explicit_wins_when_it_is_the_only_source(self) -> None:
        assert resolve_camera_id(explicit="cam_02") == "cam_02"

    def test_calibration_is_used_when_no_explicit_value(self) -> None:
        assert resolve_camera_id(calibration_camera_id="cam_01") == "cam_01"

    def test_zones_are_used_when_nothing_else_supplies_one(self) -> None:
        assert resolve_camera_id(zone_camera_id="cam_07") == "cam_07"

    def test_agreeing_sources_resolve(self) -> None:
        assert (
            resolve_camera_id(
                explicit="cam_01", calibration_camera_id="cam_01", zone_camera_id="cam_01"
            )
            == "cam_01"
        )

    @pytest.mark.parametrize(
        ("explicit", "calibration", "zones"),
        [
            ("cam_01", "cam_01", None),
            ("cam_01", None, "cam_01"),
            (None, "cam_01", "cam_01"),
        ],
    )
    def test_any_agreeing_pair_resolves(
        self, explicit: str | None, calibration: str | None, zones: str | None
    ) -> None:
        assert (
            resolve_camera_id(
                explicit=explicit, calibration_camera_id=calibration, zone_camera_id=zones
            )
            == "cam_01"
        )


class TestConflicts:
    """Disagreement is never resolved by precedence — it is always an error."""

    def test_explicit_conflicting_with_calibration(self) -> None:
        with pytest.raises(ConfigurationError, match="Conflicting camera ids"):
            resolve_camera_id(explicit="cam_02", calibration_camera_id="cam_01")

    def test_explicit_conflicting_with_zones(self) -> None:
        with pytest.raises(ConfigurationError, match="Conflicting camera ids"):
            resolve_camera_id(explicit="cam_02", zone_camera_id="cam_01")

    def test_calibration_conflicting_with_zones(self) -> None:
        with pytest.raises(ConfigurationError, match="Conflicting camera ids"):
            resolve_camera_id(calibration_camera_id="cam_01", zone_camera_id="cam_09")

    def test_all_three_conflicting(self) -> None:
        with pytest.raises(ConfigurationError, match="Conflicting camera ids"):
            resolve_camera_id(
                explicit="cam_a", calibration_camera_id="cam_b", zone_camera_id="cam_c"
            )

    def test_the_message_names_every_source_and_value(self) -> None:
        with pytest.raises(ConfigurationError) as caught:
            resolve_camera_id(explicit="cam_02", calibration_camera_id="cam_01")
        message = str(caught.value)
        assert "--camera-id says 'cam_02'" in message
        assert "the calibration says 'cam_01'" in message

    def test_precedence_does_not_silently_override(self) -> None:
        """The explicit value is highest precedence but still may not win a conflict."""
        with pytest.raises(ConfigurationError):
            resolve_camera_id(explicit="cam_02", calibration_camera_id="cam_01")


class TestMissing:
    def test_no_source_is_an_error_not_a_default(self) -> None:
        """A placeholder id would merge records from different cameras."""
        with pytest.raises(ConfigurationError, match="need a camera id"):
            resolve_camera_id()

    def test_the_message_says_how_to_supply_one(self) -> None:
        with pytest.raises(ConfigurationError) as caught:
            resolve_camera_id()
        message = str(caught.value)
        assert "--camera-id" in message
        assert "--calibration" in message
        assert "--zones" in message

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_explicit_value_is_rejected(self, blank: str) -> None:
        with pytest.raises(ConfigurationError, match="must not be blank"):
            resolve_camera_id(explicit=blank)

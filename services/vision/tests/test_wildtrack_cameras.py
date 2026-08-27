"""WILDTRACK's fixed seven-camera naming and viewNum mapping."""

from __future__ import annotations

import pytest

from atreides_vision.errors import WildtrackDatasetError
from atreides_vision.venue_validation.datasets.wildtrack.cameras import (
    CALIBRATION_NAME_BY_CAMERA_ID,
    CAMERA_COUNT,
    CAMERA_IDS,
    calibration_name_for,
    camera_id_for_view_num,
    view_num_for_camera_id,
)


class TestCanonicalOrder:
    def test_seven_cameras_in_view_num_order(self) -> None:
        assert CAMERA_IDS == ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
        assert CAMERA_COUNT == 7

    def test_every_camera_has_a_distinct_calibration_name(self) -> None:
        names = list(CALIBRATION_NAME_BY_CAMERA_ID.values())
        assert set(CALIBRATION_NAME_BY_CAMERA_ID) == set(CAMERA_IDS)
        assert len(set(names)) == len(names)


class TestCalibrationNameFor:
    def test_known_camera(self) -> None:
        assert calibration_name_for("C5") == "IDIAP1"

    def test_unknown_camera_raises(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            calibration_name_for("C8")


class TestViewNumRoundTrip:
    @pytest.mark.parametrize("view_num", range(7))
    def test_view_num_to_camera_id_and_back(self, view_num: int) -> None:
        camera_id = camera_id_for_view_num(view_num)
        assert view_num_for_camera_id(camera_id) == view_num

    @pytest.mark.parametrize("view_num", [-1, 7, 100])
    def test_out_of_range_view_num_raises(self, view_num: int) -> None:
        with pytest.raises(WildtrackDatasetError):
            camera_id_for_view_num(view_num)

    def test_unknown_camera_id_raises(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            view_num_for_camera_id("C0")

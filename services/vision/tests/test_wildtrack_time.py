"""WILDTRACK frame identifiers and their venue-relative time semantics."""

from __future__ import annotations

import pytest

from atreides_vision.errors import WildtrackDatasetError
from atreides_vision.venue_validation.datasets.wildtrack.time import (
    FRAME_RATE_HZ,
    FRAME_STEP,
    WildtrackFrameId,
)


class TestConstruction:
    def test_zero_is_valid(self) -> None:
        assert WildtrackFrameId(0).value == 0

    def test_multiple_of_step_is_valid(self) -> None:
        assert WildtrackFrameId(FRAME_STEP * 3).value == FRAME_STEP * 3

    def test_negative_value_rejected(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameId(-1)

    def test_non_multiple_of_step_rejected(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameId(FRAME_STEP - 1)


class TestFromFilename:
    def test_stem_only(self) -> None:
        assert WildtrackFrameId.from_filename("00000005") == WildtrackFrameId(5)

    def test_with_png_suffix(self) -> None:
        assert WildtrackFrameId.from_filename("00000010.png") == WildtrackFrameId(10)

    def test_non_digit_stem_rejected(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameId.from_filename("not_a_number.png")

    def test_negative_looking_stem_rejected(self) -> None:
        with pytest.raises(WildtrackDatasetError):
            WildtrackFrameId.from_filename("-00000005.png")


class TestDerivedProperties:
    def test_filename_stem_round_trips(self) -> None:
        frame = WildtrackFrameId(25)
        assert frame.filename_stem == "00000025"
        assert WildtrackFrameId.from_filename(frame.filename_stem) == frame

    def test_venue_seconds_uses_documented_frame_rate(self) -> None:
        frame = WildtrackFrameId(FRAME_STEP)
        assert frame.venue_seconds == pytest.approx(FRAME_STEP / FRAME_RATE_HZ)

    def test_annotation_cadence_is_two_hz(self) -> None:
        # FRAME_STEP frames at FRAME_RATE_HZ is the documented 2 fps annotation cadence.
        assert pytest.approx(2.0) == FRAME_RATE_HZ / FRAME_STEP

    def test_no_wall_clock_dependency(self) -> None:
        # Same frame id, called twice, must produce byte-identical results.
        first = WildtrackFrameId(15)
        second = WildtrackFrameId(15)
        assert first.venue_seconds == second.venue_seconds
        assert first.filename_stem == second.filename_stem

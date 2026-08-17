"""Venue time primitives: what is known, what is assumed, and what is refused.

These types are what stop a guess from being read later as a measurement, so
the rejections matter as much as the successes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atreides_vision.errors import VenueSessionError
from atreides_vision.timing import (
    MICROS_PER_SECOND,
    FrameRateSource,
    StreamFrame,
    StreamRef,
    TimeAlignmentBasis,
    TimeUncertainty,
)

STREAM = StreamRef(camera_id="cam_a", stream_epoch=0)


def image(value: int = 7) -> np.ndarray:
    return np.full((4, 6, 3), value, dtype=np.uint8)


def frame(
    frame_index: int = 1,
    source_seconds: float = 0.0,
    venue_seconds: float | None = None,
    stream: StreamRef = STREAM,
    timing: TimeUncertainty | None = None,
    value: int = 7,
) -> StreamFrame:
    return StreamFrame(
        stream=stream,
        frame_index=frame_index,
        source_seconds=source_seconds,
        venue_seconds=source_seconds if venue_seconds is None else venue_seconds,
        timing=TimeUncertainty.unknown() if timing is None else timing,
        image=image(value),
    )


class TestFrameRateSource:
    def test_an_assumed_rate_is_the_only_untrustworthy_one(self) -> None:
        assert FrameRateSource.CONTAINER.is_trustworthy
        assert FrameRateSource.DECLARED.is_trustworthy
        assert not FrameRateSource.ASSUMED.is_trustworthy

    def test_values_are_stable_strings(self) -> None:
        # They are written into a versioned record, so they are a contract.
        assert [source.value for source in FrameRateSource] == [
            "container",
            "assumed",
            "declared",
        ]


class TestTimeUncertainty:
    def test_unknown_is_not_established_and_carries_no_number(self) -> None:
        unknown = TimeUncertainty.unknown()
        assert unknown.basis is TimeAlignmentBasis.NOT_ESTABLISHED
        assert unknown.half_width_seconds is None
        assert not unknown.is_established

    def test_declared_carries_its_half_width(self) -> None:
        declared = TimeUncertainty.declared(0.25)
        assert declared.basis is TimeAlignmentBasis.OPERATOR_DECLARED
        assert declared.half_width_seconds == 0.25
        assert declared.is_established

    def test_not_established_with_a_half_width_is_refused(self) -> None:
        with pytest.raises(VenueSessionError, match="nobody established"):
            TimeUncertainty(basis=TimeAlignmentBasis.NOT_ESTABLISHED, half_width_seconds=0.5)

    def test_declared_without_a_half_width_is_refused(self) -> None:
        with pytest.raises(VenueSessionError, match="not a zero"):
            TimeUncertainty(basis=TimeAlignmentBasis.OPERATOR_DECLARED)

    def test_zero_half_width_is_refused(self) -> None:
        """Zero would claim an exactness no declared offset has."""
        with pytest.raises(VenueSessionError, match="strictly positive"):
            TimeUncertainty.declared(0.0)

    @pytest.mark.parametrize("value", [-0.5, -1e-9])
    def test_negative_half_width_is_refused(self, value: float) -> None:
        with pytest.raises(VenueSessionError, match="strictly positive"):
            TimeUncertainty.declared(value)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_half_width_is_refused(self, value: float) -> None:
        with pytest.raises(VenueSessionError, match="finite"):
            TimeUncertainty.declared(value)

    def test_not_established_holds_exactly_when_the_half_width_is_none(self) -> None:
        for uncertainty in (TimeUncertainty.unknown(), TimeUncertainty.declared(1.5)):
            assert (uncertainty.half_width_seconds is None) is (
                uncertainty.basis is TimeAlignmentBasis.NOT_ESTABLISHED
            )

    def test_worst_of_takes_the_widest_declared(self) -> None:
        worst = TimeUncertainty.declared(0.1).worst_of(
            TimeUncertainty.declared(0.9), TimeUncertainty.declared(0.4)
        )
        assert worst == TimeUncertainty.declared(0.9)

    def test_not_established_absorbs(self) -> None:
        worst = TimeUncertainty.declared(0.1).worst_of(
            TimeUncertainty.unknown(), TimeUncertainty.declared(5.0)
        )
        assert not worst.is_established

    def test_not_established_absorbs_from_the_left_too(self) -> None:
        worst = TimeUncertainty.unknown().worst_of(TimeUncertainty.declared(0.1))
        assert not worst.is_established

    def test_worst_of_nothing_is_itself(self) -> None:
        declared = TimeUncertainty.declared(0.2)
        assert declared.worst_of() == declared

    def test_there_is_no_threshold_or_alignment_verdict(self) -> None:
        """A boolean derived from a half-width is the first step to a merge rule."""
        names = dir(TimeUncertainty)
        assert not any("aligned" in name for name in names)
        assert not any("threshold" in name for name in names)
        for comparison in ("__lt__", "__le__", "__gt__", "__ge__"):
            assert getattr(TimeUncertainty, comparison) is getattr(object, comparison)


class TestStreamRef:
    def test_string_form_uses_a_separator_distinct_from_a_subject(self) -> None:
        assert str(StreamRef("cam_a", 2)) == "cam_a#2"

    @pytest.mark.parametrize("camera_id", ["", "   "])
    def test_blank_camera_id_is_refused(self, camera_id: str) -> None:
        with pytest.raises(VenueSessionError, match="non-empty"):
            StreamRef(camera_id=camera_id, stream_epoch=0)

    def test_negative_epoch_is_refused(self) -> None:
        with pytest.raises(VenueSessionError, match="non-negative"):
            StreamRef(camera_id="cam_a", stream_epoch=-1)

    def test_epochs_distinguish_two_recordings_of_one_camera(self) -> None:
        first, second = StreamRef("cam_a", 0), StreamRef("cam_a", 1)
        assert first != second
        assert len({first, second}) == 2

    def test_equal_refs_are_the_same_key(self) -> None:
        assert {StreamRef("cam_a", 0): "x"}[StreamRef("cam_a", 0)] == "x"


class TestStreamFrame:
    def test_frame_index_is_one_based(self) -> None:
        with pytest.raises(VenueSessionError, match="1-based"):
            frame(frame_index=0)

    @pytest.mark.parametrize("seconds", [-0.001, math.nan, math.inf])
    def test_impossible_source_seconds_are_refused(self, seconds: float) -> None:
        with pytest.raises(VenueSessionError, match="source_seconds"):
            frame(source_seconds=seconds, venue_seconds=0.0)

    @pytest.mark.parametrize("seconds", [-0.001, math.nan, math.inf])
    def test_impossible_venue_seconds_are_refused(self, seconds: float) -> None:
        with pytest.raises(VenueSessionError, match="venue_seconds"):
            frame(source_seconds=0.0, venue_seconds=seconds)

    def test_venue_micros_rounds_to_the_ordering_grid(self) -> None:
        assert frame(source_seconds=1.0).venue_micros == MICROS_PER_SECOND
        assert frame(source_seconds=3 / 25).venue_micros == 120_000

    def test_the_same_rational_instant_reached_two_ways_shares_the_key(self) -> None:
        """Exact integer equality is what makes simultaneity tolerance-free."""
        at_25 = frame(source_seconds=3 / 25)
        at_50 = frame(source_seconds=6 / 50)
        assert at_25.venue_micros == at_50.venue_micros

    def test_venue_seconds_is_not_derived_from_the_ordering_key(self) -> None:
        """The key is quantised; the measurement is not."""
        precise = frame(source_seconds=0.0, venue_seconds=1.000_000_4)
        assert precise.venue_micros == 1_000_000
        assert precise.venue_seconds == 1.000_000_4

    def test_comparing_two_frames_with_different_images_does_not_raise(self) -> None:
        """The trap: a frozen dataclass comparing ndarrays raises on ambiguity."""
        left = frame(value=1)
        right = frame(value=200)
        assert left == right  # same stream, index and times — the image is not identity
        assert not np.array_equal(left.image, right.image)

    def test_frames_are_hashable_despite_carrying_pixels(self) -> None:
        assert len({frame(value=1), frame(value=200)}) == 1

    def test_differing_metadata_still_compares_unequal(self) -> None:
        assert frame(frame_index=1) != frame(frame_index=2)
        assert frame(stream=StreamRef("cam_a", 0)) != frame(stream=StreamRef("cam_b", 0))

    def test_provenance_survives_construction(self) -> None:
        observed = frame(frame_index=51, source_seconds=2.0, venue_seconds=12.0)
        assert observed.stream.camera_id == "cam_a"
        assert observed.stream.stream_epoch == 0
        assert observed.frame_index == 51
        assert observed.source_seconds == 2.0
        assert observed.venue_seconds == 12.0

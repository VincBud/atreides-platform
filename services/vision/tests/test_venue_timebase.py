"""The venue timeline: what an offset means, and when one is refused."""

from __future__ import annotations

import logging
import math

import pytest

from atreides_vision.errors import VenueSessionError
from atreides_vision.timing import (
    FrameRateSource,
    StreamRef,
    TimeAlignmentBasis,
    TimeUncertainty,
)
from atreides_vision.venue.timebase import VenueTimebase

CAM_A = StreamRef("cam_a", 0)
CAM_B = StreamRef("cam_b", 0)


def timebase_with(*streams: StreamRef) -> VenueTimebase:
    timebase = VenueTimebase()
    for stream in streams:
        timebase.register(stream, fps_source=FrameRateSource.CONTAINER)
    return timebase


class TestRegistration:
    def test_a_stream_with_nothing_declared_sits_at_venue_zero(self) -> None:
        timebase = timebase_with(CAM_A)

        assert timebase.offset_seconds(CAM_A) == 0.0
        assert timebase.uncertainty(CAM_A).basis is TimeAlignmentBasis.NOT_ESTABLISHED

    def test_a_declared_offset_carries_its_accuracy(self) -> None:
        timebase = VenueTimebase()
        timebase.register(
            CAM_B,
            fps_source=FrameRateSource.CONTAINER,
            offset_seconds=12.5,
            offset_accuracy_seconds=0.25,
        )

        assert timebase.offset_seconds(CAM_B) == 12.5
        uncertainty = timebase.uncertainty(CAM_B)
        assert uncertainty.basis is TimeAlignmentBasis.OPERATOR_DECLARED
        assert uncertainty.half_width_seconds == 0.25

    def test_registering_a_stream_twice_is_refused(self) -> None:
        timebase = timebase_with(CAM_A)
        with pytest.raises(VenueSessionError, match="already registered"):
            timebase.register(CAM_A, fps_source=FrameRateSource.CONTAINER)

    def test_an_offset_without_an_accuracy_is_refused(self) -> None:
        timebase = VenueTimebase()
        with pytest.raises(VenueSessionError, match="needs an accuracy"):
            timebase.register(CAM_A, fps_source=FrameRateSource.CONTAINER, offset_seconds=3.0)

    def test_an_accuracy_without_an_offset_is_refused(self) -> None:
        timebase = VenueTimebase()
        with pytest.raises(VenueSessionError, match="no offset"):
            timebase.register(
                CAM_A, fps_source=FrameRateSource.CONTAINER, offset_accuracy_seconds=0.5
            )

    @pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
    def test_an_impossible_offset_is_refused(self, value: float) -> None:
        timebase = VenueTimebase()
        with pytest.raises(VenueSessionError, match="finite and non-negative"):
            timebase.register(
                CAM_A,
                fps_source=FrameRateSource.CONTAINER,
                offset_seconds=value,
                offset_accuracy_seconds=0.5,
            )


class TestAssumedFrameRates:
    """Placement and accuracy are separate concerns for a guessed frame rate.

    An ordinary NVR export — camera restarted mid-session, container reporting
    a useless rate — must remain representable. Refusing its offset outright
    would leave it unrepresentable and push an operator into inventing a
    ``declared_fps``, which *is* trusted: a guess laundered into a measurement.
    So the offset is honoured and the accuracy claim is not.
    """

    def test_an_assumed_rate_may_still_be_placed(self) -> None:
        timebase = VenueTimebase()
        timebase.register(
            CAM_A,
            fps_source=FrameRateSource.ASSUMED,
            offset_seconds=10.0,
            offset_accuracy_seconds=0.5,
        )

        assert timebase.offset_seconds(CAM_A) == 10.0
        assert timebase.venue_seconds(CAM_A, 2.5) == 12.5

    def test_but_its_accuracy_claim_is_discarded(self) -> None:
        """A fixed half-width cannot describe an error that grows with time."""
        timebase = VenueTimebase()
        timebase.register(
            CAM_A,
            fps_source=FrameRateSource.ASSUMED,
            offset_seconds=10.0,
            offset_accuracy_seconds=0.5,
        )

        uncertainty = timebase.uncertainty(CAM_A)
        assert uncertainty.basis is TimeAlignmentBasis.NOT_ESTABLISHED
        assert uncertainty.half_width_seconds is None

    def test_the_discard_is_never_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        timebase = VenueTimebase()

        with caplog.at_level(logging.WARNING, logger="atreides_vision.venue.timebase"):
            timebase.register(
                CAM_A,
                fps_source=FrameRateSource.ASSUMED,
                offset_seconds=10.0,
                offset_accuracy_seconds=0.5,
            )

        warnings = [record.getMessage() for record in caplog.records]
        assert any("discarding the declared offset accuracy" in message for message in warnings)
        assert any("cam_a#0" in message for message in warnings)
        assert any("NOT_ESTABLISHED" in message for message in warnings)

    def test_an_assumed_rate_without_an_offset_is_accepted_as_unaligned(self) -> None:
        timebase = VenueTimebase()
        timebase.register(CAM_A, fps_source=FrameRateSource.ASSUMED)

        assert timebase.offset_seconds(CAM_A) == 0.0
        assert not timebase.uncertainty(CAM_A).is_established

    def test_an_offset_without_an_accuracy_is_still_refused_for_an_assumed_rate(self) -> None:
        """The manifest rule does not relax; only the accuracy is discarded."""
        timebase = VenueTimebase()
        with pytest.raises(VenueSessionError, match="needs an accuracy"):
            timebase.register(CAM_A, fps_source=FrameRateSource.ASSUMED, offset_seconds=10.0)

    def test_a_declared_rate_keeps_its_accuracy(self) -> None:
        timebase = VenueTimebase()
        timebase.register(
            CAM_A,
            fps_source=FrameRateSource.DECLARED,
            offset_seconds=10.0,
            offset_accuracy_seconds=0.5,
        )

        assert timebase.uncertainty(CAM_A) == TimeUncertainty.declared(0.5)

    def test_a_container_rate_keeps_its_accuracy(self) -> None:
        timebase = VenueTimebase()
        timebase.register(
            CAM_B,
            fps_source=FrameRateSource.CONTAINER,
            offset_seconds=10.0,
            offset_accuracy_seconds=0.5,
        )

        assert timebase.uncertainty(CAM_B) == TimeUncertainty.declared(0.5)


class TestConversion:
    def test_venue_seconds_adds_the_offset(self) -> None:
        timebase = VenueTimebase()
        timebase.register(
            CAM_A,
            fps_source=FrameRateSource.CONTAINER,
            offset_seconds=10.0,
            offset_accuracy_seconds=0.5,
        )

        assert timebase.venue_seconds(CAM_A, 2.5) == 12.5

    @pytest.mark.parametrize("fps", [25.0, 30.0, 29.97, 23.976, 12.5])
    def test_a_zero_offset_is_bit_identical_to_source_time(self, fps: float) -> None:
        """One camera is the degenerate venue case, not an approximation of it."""
        timebase = timebase_with(CAM_A)

        for frame_index in range(1, 200):
            source_seconds = (frame_index - 1) / fps
            assert timebase.venue_seconds(CAM_A, source_seconds) == source_seconds

    def test_conversion_is_not_quantised_to_the_ordering_key(self) -> None:
        timebase = timebase_with(CAM_A)
        assert timebase.venue_seconds(CAM_A, 1.000_000_4) == 1.000_000_4

    @pytest.mark.parametrize("value", [-0.5, math.nan, math.inf])
    def test_impossible_source_seconds_are_refused(self, value: float) -> None:
        timebase = timebase_with(CAM_A)
        with pytest.raises(VenueSessionError, match="source_seconds"):
            timebase.venue_seconds(CAM_A, value)

    def test_an_unregistered_stream_has_no_venue_time(self) -> None:
        timebase = timebase_with(CAM_A)
        with pytest.raises(VenueSessionError, match="not registered"):
            timebase.venue_seconds(CAM_B, 1.0)

    def test_an_unregistered_stream_has_no_uncertainty_either(self) -> None:
        timebase = timebase_with(CAM_A)
        with pytest.raises(VenueSessionError, match="not registered"):
            timebase.uncertainty(CAM_B)


class TestOrdering:
    def test_streams_come_back_canonically_not_in_registration_order(self) -> None:
        timebase = VenueTimebase()
        for stream in (StreamRef("cam_z", 1), StreamRef("cam_a", 0), StreamRef("cam_z", 0)):
            timebase.register(stream, fps_source=FrameRateSource.CONTAINER)

        assert timebase.streams == (
            StreamRef("cam_a", 0),
            StreamRef("cam_z", 0),
            StreamRef("cam_z", 1),
        )

    def test_registration_is_visible_without_raising(self) -> None:
        timebase = timebase_with(CAM_A)
        assert timebase.is_registered(CAM_A)
        assert not timebase.is_registered(CAM_B)

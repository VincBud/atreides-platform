"""WILDTRACK's declared timing under the real Atreides timing domain.

Synthetic fixtures only — no real corpus, no person-level ground truth. These
tests pin the V0.4.9b decision: the WILDTRACK nominal (model) offset is exactly
``0.0`` s, and its published synchronization accuracy (50 ms, then manually
refined) is carried as a *positive* symmetric alignment half-width. The core
timing doctrine — a zero declared half-width stays invalid — is untouched.
"""

from __future__ import annotations

from association_builders import GATING, observation
from atreides_vision.association import (
    CandidatePair,
    GateReason,
    comparability_rejections,
)
from atreides_vision.association.geometry import timing_slack_seconds
from atreides_vision.timing import (
    FrameRateSource,
    StreamRef,
    TimeAlignmentBasis,
    TimeUncertainty,
)
from atreides_vision.venue.session import VenueSession, VideoSourceCursor
from atreides_vision.venue.timebase import VenueTimebase
from atreides_vision.venue_validation.datasets.wildtrack.runtime import (
    WILDTRACK_STREAM_OFFSET_ACCURACY_SECONDS,
    WILDTRACK_STREAM_OFFSET_SECONDS,
)
from venue_fakes import CountingSource

WILDTRACK_CAMERAS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")


def test_wildtrack_constants_are_zero_nominal_offset_and_positive_accuracy() -> None:
    assert WILDTRACK_STREAM_OFFSET_SECONDS == 0.0
    assert WILDTRACK_STREAM_OFFSET_ACCURACY_SECONDS == 0.05


def test_time_uncertainty_accepts_the_wildtrack_accuracy() -> None:
    uncertainty = TimeUncertainty.declared(WILDTRACK_STREAM_OFFSET_ACCURACY_SECONDS)

    assert uncertainty.basis is TimeAlignmentBasis.OPERATOR_DECLARED
    assert uncertainty.half_width_seconds == 0.05
    assert uncertainty.is_established is True


def _wildtrack_timebase(cameras: tuple[str, ...]) -> VenueTimebase:
    timebase = VenueTimebase()
    for camera_id in cameras:
        timebase.register(
            StreamRef(camera_id, 0),
            fps_source=FrameRateSource.DECLARED,
            offset_seconds=WILDTRACK_STREAM_OFFSET_SECONDS,
            offset_accuracy_seconds=WILDTRACK_STREAM_OFFSET_ACCURACY_SECONDS,
        )
    return timebase


def test_every_wildtrack_stream_has_established_half_width_alignment() -> None:
    timebase = _wildtrack_timebase(WILDTRACK_CAMERAS)

    for camera_id in WILDTRACK_CAMERAS:
        uncertainty = timebase.uncertainty(StreamRef(camera_id, 0))
        assert uncertainty.is_established
        assert uncertainty.half_width_seconds == 0.05


def test_synchronized_source_seconds_become_identical_venue_micros() -> None:
    timebase = _wildtrack_timebase(WILDTRACK_CAMERAS)

    for camera_id in WILDTRACK_CAMERAS:
        stream = StreamRef(camera_id, 0)
        assert timebase.venue_seconds(stream, 0.0) == 0.0
        assert timebase.venue_seconds(stream, 0.5) == 0.5
        assert timebase.venue_seconds(stream, 1.0) == 1.0


def test_two_hertz_streams_merge_into_one_tick_per_instant() -> None:
    cameras = ("C1", "C4", "C7")
    timebase = _wildtrack_timebase(cameras)
    cursors = [
        VideoSourceCursor(
            StreamRef(camera_id, 0),
            CountingSource(frame_count=3, fps=2.0, fps_source=FrameRateSource.DECLARED),
            timebase,
        )
        for camera_id in cameras
    ]
    session = VenueSession("wildtrack", timebase, cursors)

    ticks = list(session.ticks())

    assert [tick.venue_micros for tick in ticks] == [0, 500_000, 1_000_000]
    for tick in ticks:
        assert tick.absent == ()
        contributing = [contribution.stream.camera_id for contribution in tick.contributions]
        assert contributing == sorted(cameras)  # canonical, one per camera
        assert tick.alignment.half_width_seconds == 0.05


def test_all_seven_cameras_merge_at_exact_venue_micros() -> None:
    timebase = _wildtrack_timebase(WILDTRACK_CAMERAS)
    cursors = [
        VideoSourceCursor(
            StreamRef(camera_id, 0),
            CountingSource(frame_count=3, fps=2.0, fps_source=FrameRateSource.DECLARED),
            timebase,
        )
        for camera_id in WILDTRACK_CAMERAS
    ]
    session = VenueSession("wildtrack", timebase, cursors)

    ticks = list(session.ticks())

    assert [tick.venue_micros for tick in ticks] == [0, 500_000, 1_000_000]
    for tick in ticks:
        assert len(tick.contributions) == 7
        assert [c.stream.camera_id for c in tick.contributions] == list(WILDTRACK_CAMERAS)


def test_merge_is_identical_across_camera_ordering() -> None:
    def run(order: tuple[str, ...]) -> list[tuple[int, tuple[str, ...]]]:
        timebase = _wildtrack_timebase(WILDTRACK_CAMERAS)
        cursors = [
            VideoSourceCursor(
                StreamRef(camera_id, 0),
                CountingSource(frame_count=3, fps=2.0, fps_source=FrameRateSource.DECLARED),
                timebase,
            )
            for camera_id in order
        ]
        session = VenueSession("wildtrack", timebase, cursors)
        return [
            (tick.venue_micros, tuple(c.stream.camera_id for c in tick.contributions))
            for tick in session.ticks()
        ]

    assert run(WILDTRACK_CAMERAS) == run(tuple(reversed(WILDTRACK_CAMERAS)))


def test_same_instant_wildtrack_pair_clears_the_timing_gates() -> None:
    first = observation("cam_a", 1, venue_seconds=10.0, half_width_seconds=0.05)
    second = observation("cam_b", 4, x=0.5, venue_seconds=10.0, half_width_seconds=0.05)

    assert timing_slack_seconds(first.timing, second.timing) == 0.1

    reasons = {
        rejection.reason
        for rejection in comparability_rejections(CandidatePair(first, second), GATING)
    }
    assert GateReason.VENUE_TIME_NOT_ESTABLISHED not in reasons
    assert GateReason.NOT_SIMULTANEOUS not in reasons
    assert reasons == set()

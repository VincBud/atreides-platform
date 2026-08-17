"""Venue sessions: cursors, provenance, counters, and the single-camera case.

The last of these matters most architecturally: one camera must be the
degenerate case of the venue path, producing exactly what the single-source
pipeline has always produced — not a separate mode, and not an approximation.
"""

from __future__ import annotations

import logging
from itertools import count
from pathlib import Path

import numpy as np
import pytest

from atreides_vision.config import (
    DetectorConfig,
    PipelineConfig,
    TrackerConfig,
    VisionConfig,
    VisualizationConfig,
)
from atreides_vision.errors import VenueSessionError
from atreides_vision.models import Detection
from atreides_vision.timing import FrameRateSource, StreamRef, TimeAlignmentBasis, TimeUncertainty
from atreides_vision.venue.session import VenueSession, VideoSourceCursor
from atreides_vision.venue.timebase import VenueTimebase
from atreides_vision.video.processor import VideoProcessor
from atreides_vision.video.source import VideoMetadata
from fakes import FakeDetector, FakeSink, FakeSource, FakeTracker
from venue_fakes import FRAME_H, FRAME_W, CountingSource

CAM_A = StreamRef("cam_a", 0)
CAM_B = StreamRef("cam_b", 0)


def registered(*streams: StreamRef) -> VenueTimebase:
    timebase = VenueTimebase()
    for stream in streams:
        timebase.register(stream, fps_source=FrameRateSource.CONTAINER)
    return timebase


def build_cursor(
    stream: StreamRef,
    frame_count: int,
    fps: float = 25.0,
    timebase: VenueTimebase | None = None,
) -> tuple[VideoSourceCursor, CountingSource]:
    base = registered(stream) if timebase is None else timebase
    source = CountingSource(frame_count=frame_count, fps=fps)
    return VideoSourceCursor(stream, source, base), source


class TestVideoSourceCursor:
    def test_frames_carry_full_provenance(self) -> None:
        cursor, _ = build_cursor(CAM_A, frame_count=3, fps=25.0)

        seen = []
        while (head := cursor.peek()) is not None:
            seen.append(head)
            cursor.advance()

        assert [head.frame_index for head in seen] == [1, 2, 3]
        assert [head.stream for head in seen] == [CAM_A, CAM_A, CAM_A]
        assert [head.source_seconds for head in seen] == [0.0, 0.04, 0.08]
        assert [head.venue_seconds for head in seen] == [0.0, 0.04, 0.08]

    @pytest.mark.parametrize("fps", [25.0, 30.0, 29.97, 23.976, 12.5])
    def test_venue_seconds_is_bit_identical_to_the_source_formula(self, fps: float) -> None:
        """The exact expression the pipeline used before the venue seam existed."""
        cursor, _ = build_cursor(CAM_A, frame_count=60, fps=fps)

        frame_index = 0
        while (head := cursor.peek()) is not None:
            frame_index += 1
            assert head.venue_seconds == (frame_index - 1) / fps
            assert head.venue_seconds == head.source_seconds
            cursor.advance()

        assert frame_index == 60

    def test_a_declared_offset_shifts_venue_time_but_not_source_time(self) -> None:
        timebase = VenueTimebase()
        timebase.register(
            CAM_B,
            fps_source=FrameRateSource.CONTAINER,
            offset_seconds=30.0,
            offset_accuracy_seconds=0.5,
        )
        cursor, _ = build_cursor(CAM_B, frame_count=2, fps=25.0, timebase=timebase)

        head = cursor.peek()

        assert head is not None
        assert head.source_seconds == 0.0
        assert head.venue_seconds == 30.0
        assert head.timing == TimeUncertainty.declared(0.5)

    def test_peek_is_idempotent_and_holds_exactly_one_frame(self) -> None:
        cursor, source = build_cursor(CAM_A, frame_count=5)

        first, again = cursor.peek(), cursor.peek()

        assert first is again
        assert source.frames_pulled == 1
        cursor.advance()
        assert source.frames_pulled == 1  # nothing is decoded until it is peeked

    def test_a_source_with_no_frames_warns_and_contributes_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cursor, _ = build_cursor(CAM_A, frame_count=0)

        with caplog.at_level(logging.WARNING, logger="atreides_vision.venue.session"):
            assert cursor.peek() is None
            assert cursor.peek() is None

        assert sum("decoded no frames" in record.message for record in caplog.records) == 1

    def test_advancing_without_a_peeked_frame_is_refused(self) -> None:
        cursor, _ = build_cursor(CAM_A, frame_count=2)
        with pytest.raises(VenueSessionError, match="no peeked frame"):
            cursor.advance()

    def test_an_unregistered_stream_cannot_be_read(self) -> None:
        with pytest.raises(VenueSessionError, match="registered on the timebase"):
            VideoSourceCursor(CAM_B, CountingSource(frame_count=1), registered(CAM_A))

    def test_a_source_reporting_no_frame_rate_still_advances(self) -> None:
        """Mirrors the pipeline's own guard, so neither path can diverge."""
        source = CountingSource(frame_count=2, fps=0.0)
        cursor = VideoSourceCursor(CAM_A, source, registered(CAM_A))

        head = cursor.peek()
        assert head is not None
        cursor.advance()
        second = cursor.peek()
        assert second is not None
        assert second.source_seconds == 1.0


class TestVenueSession:
    def test_counts_ticks_and_frames_per_stream(self) -> None:
        timebase = registered(CAM_A, CAM_B)
        cursor_a, _ = build_cursor(CAM_A, frame_count=4, fps=25.0, timebase=timebase)
        cursor_b, _ = build_cursor(CAM_B, frame_count=4, fps=50.0, timebase=timebase)
        session = VenueSession("session_a", timebase, [cursor_a, cursor_b])

        ticks = list(session.ticks())

        assert session.tick_count == len(ticks)
        assert session.frames_by_stream == ((CAM_A, 4), (CAM_B, 4))
        assert session.first_venue_micros == 0
        assert session.last_venue_micros == max(tick.venue_micros for tick in ticks)

    def test_streams_are_canonical_regardless_of_construction_order(self) -> None:
        timebase = registered(CAM_A, CAM_B)
        cursor_a, _ = build_cursor(CAM_A, frame_count=1, timebase=timebase)
        cursor_b, _ = build_cursor(CAM_B, frame_count=1, timebase=timebase)

        session = VenueSession("session_a", timebase, [cursor_b, cursor_a])

        assert session.streams == (CAM_A, CAM_B)

    def test_an_all_empty_session_yields_nothing_and_returns_normally(self) -> None:
        timebase = registered(CAM_A, CAM_B)
        cursor_a, _ = build_cursor(CAM_A, frame_count=0, timebase=timebase)
        cursor_b, _ = build_cursor(CAM_B, frame_count=0, timebase=timebase)
        session = VenueSession("session_a", timebase, [cursor_a, cursor_b])

        assert list(session.ticks()) == []
        assert session.tick_count == 0
        assert session.first_venue_micros is None
        assert session.frames_by_stream == ((CAM_A, 0), (CAM_B, 0))

    def test_a_session_cannot_be_walked_twice(self) -> None:
        timebase = registered(CAM_A)
        cursor_a, _ = build_cursor(CAM_A, frame_count=2, timebase=timebase)
        session = VenueSession("session_a", timebase, [cursor_a])

        list(session.ticks())

        with pytest.raises(VenueSessionError, match="already been walked"):
            list(session.ticks())

    def test_session_alignment_is_the_worst_of_its_streams(self) -> None:
        timebase = VenueTimebase()
        timebase.register(
            CAM_A,
            fps_source=FrameRateSource.CONTAINER,
            offset_seconds=0.0,
            offset_accuracy_seconds=0.4,
        )
        timebase.register(CAM_B, fps_source=FrameRateSource.CONTAINER)
        cursor_a, _ = build_cursor(CAM_A, frame_count=1, timebase=timebase)
        cursor_b, _ = build_cursor(CAM_B, frame_count=1, timebase=timebase)

        session = VenueSession("session_a", timebase, [cursor_a, cursor_b])

        assert session.alignment.basis is TimeAlignmentBasis.NOT_ESTABLISHED

    def test_a_session_needs_a_name_and_a_stream(self) -> None:
        timebase = registered(CAM_A)
        cursor_a, _ = build_cursor(CAM_A, frame_count=1, timebase=timebase)

        with pytest.raises(VenueSessionError, match="non-empty session_id"):
            VenueSession(" ", timebase, [cursor_a])
        with pytest.raises(VenueSessionError, match="at least one stream"):
            VenueSession("session_a", timebase, [])

    def test_a_stream_unknown_to_the_timebase_is_refused(self) -> None:
        timebase = registered(CAM_A)
        cursor_a, _ = build_cursor(CAM_A, frame_count=1, timebase=timebase)
        other = VenueTimebase()
        other.register(CAM_B, fps_source=FrameRateSource.CONTAINER)
        cursor_b, _ = build_cursor(CAM_B, frame_count=1, timebase=other)

        with pytest.raises(VenueSessionError, match="not registered"):
            VenueSession("session_a", timebase, [cursor_a, cursor_b])

    def test_bounded_state_over_a_long_session(self) -> None:
        """Whatever the run length, one decoded frame per stream is in flight."""
        timebase = registered(CAM_A, CAM_B)
        cursor_a, source_a = build_cursor(CAM_A, frame_count=300, fps=25.0, timebase=timebase)
        cursor_b, source_b = build_cursor(CAM_B, frame_count=300, fps=30.0, timebase=timebase)
        session = VenueSession("session_a", timebase, [cursor_a, cursor_b])

        contributed: dict[StreamRef, int] = {CAM_A: 0, CAM_B: 0}
        for tick in session.ticks():
            for contribution in tick.contributions:
                contributed[contribution.stream] += 1
            assert source_a.frames_pulled - contributed[CAM_A] <= 1
            assert source_b.frames_pulled - contributed[CAM_B] <= 1

        assert contributed == {CAM_A: 300, CAM_B: 300}

    def test_every_contribution_resolves_back_to_its_source_frame(self) -> None:
        timebase = registered(CAM_A, CAM_B)
        cursor_a, _ = build_cursor(CAM_A, frame_count=5, fps=25.0, timebase=timebase)
        cursor_b, _ = build_cursor(CAM_B, frame_count=5, fps=50.0, timebase=timebase)
        session = VenueSession("session_a", timebase, [cursor_a, cursor_b])

        provenance: dict[StreamRef, list[tuple[int, float]]] = {CAM_A: [], CAM_B: []}
        for tick in session.ticks():
            for contribution in tick.contributions:
                provenance[contribution.stream].append(
                    (contribution.frame_index, contribution.source_seconds)
                )

        assert provenance[CAM_A] == [(index, (index - 1) / 25.0) for index in range(1, 6)]
        assert provenance[CAM_B] == [(index, (index - 1) / 50.0) for index in range(1, 6)]


def make_config(tmp_path: Path) -> VisionConfig:
    return VisionConfig(
        pipeline=PipelineConfig(input_path=tmp_path / "in.mp4", output_path=tmp_path / "out.mp4"),
        detector=DetectorConfig(),
        tracker=TrackerConfig(),
        visualization=VisualizationConfig(trail_length=5, trail_ttl_frames=3),
    )


def pipeline_frames(n: int) -> list[np.ndarray]:
    return [
        np.full((FRAME_H * 10, FRAME_W * 10, 3), 40 + index, dtype=np.uint8) for index in range(n)
    ]


class TestSingleCameraEquivalence:
    def test_the_venue_path_reproduces_run_exactly_for_one_clip(self, tmp_path: Path) -> None:
        # A fresh set of frames per run: the annotator draws onto the frame it
        # is given, so sharing one list would have the second run annotate the
        # first run's output.
        metadata = VideoMetadata(width=FRAME_W * 10, height=FRAME_H * 10, fps=30.0, frame_count=8)
        detections = [[Detection.from_xyxy((10, 10, 40, 60), 0.9, 1)]]

        direct_sink = FakeSink(tmp_path / "out.mp4")
        direct = VideoProcessor(
            FakeDetector(detections),
            FakeTracker(),
            make_config(tmp_path),
            clock=count(start=0.0, step=0.04).__next__,
        ).run(FakeSource(pipeline_frames(8), metadata), direct_sink)

        venue_sink = FakeSink(tmp_path / "out.mp4")
        processor = VideoProcessor(
            FakeDetector(detections),
            FakeTracker(),
            make_config(tmp_path),
            clock=count(start=0.0, step=0.04).__next__,
        )
        timebase = registered(CAM_A)
        source = FakeSource(pipeline_frames(8), metadata)
        cursor = VideoSourceCursor(CAM_A, source, timebase)
        session = VenueSession("session_a", timebase, [cursor])
        # The venue loop hands the processor the same VideoMetadata run() reads,
        # so both paths normalise geometry against the same dimensions.
        processor.begin(source.metadata, venue_sink)
        for tick in session.ticks():
            for contribution in tick.contributions:
                processor.process(contribution, venue_sink)
        venue = processor.finish()

        assert venue.frames_processed == direct.frames_processed == 8
        assert venue.unique_track_ids == direct.unique_track_ids
        assert venue.elapsed_seconds == direct.elapsed_seconds
        assert len(venue_sink.frames) == len(direct_sink.frames)
        for venue_frame, direct_frame in zip(venue_sink.frames, direct_sink.frames, strict=True):
            assert np.array_equal(venue_frame, direct_frame)

    def test_one_camera_needs_no_offset_and_no_special_mode(self, tmp_path: Path) -> None:
        frames = pipeline_frames(3)
        metadata = VideoMetadata(
            width=FRAME_W * 10, height=FRAME_H * 10, fps=25.0, frame_count=len(frames)
        )
        timebase = registered(CAM_A)
        session = VenueSession(
            "session_a",
            timebase,
            [VideoSourceCursor(CAM_A, FakeSource(frames, metadata), timebase)],
        )

        ticks = list(session.ticks())

        assert [tick.venue_micros for tick in ticks] == [0, 40_000, 80_000]
        assert all(tick.absent == () for tick in ticks)
        assert all(not tick.alignment.is_established for tick in ticks)

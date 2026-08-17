"""The process()/finish() seam, and that run() is unchanged by its existence.

``run()`` is now a thin wrapper: it supplies timed frames and calls the same two
methods a venue loop calls. These tests hold both halves to the behaviour the
single-source pipeline has always had.
"""

from __future__ import annotations

import logging
from itertools import count
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pytest

from atreides_vision.config import (
    DetectorConfig,
    PipelineConfig,
    TrackerConfig,
    VisionConfig,
    VisualizationConfig,
)
from atreides_vision.errors import ConfigurationError
from atreides_vision.models import Detection, Track
from atreides_vision.timing import StreamFrame, StreamRef, TimeUncertainty
from atreides_vision.video.processor import VideoProcessor
from atreides_vision.video.source import VideoMetadata
from atreides_vision.zones.analyzer import ZoneAnalyzer
from atreides_vision.zones.geometry import track_floor_point
from atreides_vision.zones.models import Zone, ZoneLayout, ZonePolygon
from fakes import FakeDetector, FakeSink, FakeSource, FakeTracker

FRAME_W, FRAME_H = 200, 120
STREAM = StreamRef("cam_venue", 3)

LAYOUT = ZoneLayout(
    camera_id="cam_venue",
    zones=(
        Zone(
            id="left",
            name="Left",
            polygon=ZonePolygon.from_pairs([[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]),
        ),
    ),
)


def frames(n: int) -> list[np.ndarray]:
    return [np.full((FRAME_H, FRAME_W, 3), 30 + index, dtype=np.uint8) for index in range(n)]


def make_config(tmp_path: Path, **pipeline_kwargs: object) -> VisionConfig:
    return VisionConfig(
        pipeline=PipelineConfig(
            input_path=tmp_path / "in.mp4",
            output_path=tmp_path / "out.mp4",
            **pipeline_kwargs,  # type: ignore[arg-type]
        ),
        detector=DetectorConfig(),
        tracker=TrackerConfig(),
        visualization=VisualizationConfig(trail_length=5, trail_ttl_frames=3),
    )


#: What the container declares for the seam's frames. ``begin()`` takes the
#: geometry from here, exactly as the composition root builds this camera's
#: projector from the same metadata.
SEAM_METADATA = VideoMetadata(width=FRAME_W, height=FRAME_H, fps=25.0, frame_count=None)


def stream_frame(frame_index: int, venue_seconds: float, image: np.ndarray) -> StreamFrame:
    return StreamFrame(
        stream=STREAM,
        frame_index=frame_index,
        source_seconds=venue_seconds,
        venue_seconds=venue_seconds,
        timing=TimeUncertainty.unknown(),
        image=image,
    )


def detection() -> list[list[Detection]]:
    return [[Detection.from_xyxy((10, 10, 40, 80), 0.9, 1)]]


class TestProcessAndFinish:
    def test_process_returns_none(self, tmp_path: Path) -> None:
        """A per-frame result type would be the hook association hangs on."""
        assert get_type_hints(VideoProcessor.process)["return"] is type(None)

        processor = VideoProcessor(FakeDetector(detection()), FakeTracker(), make_config(tmp_path))
        sink = FakeSink()

        processor.begin(SEAM_METADATA, sink)
        processor.process(stream_frame(1, 0.0, frames(1)[0]), sink)

        # The frame's effects are the output; there is nothing else to hand back.
        assert len(sink.frames) == 1

    def test_driving_the_seam_directly_processes_every_frame(self, tmp_path: Path) -> None:
        detector, tracker, sink = FakeDetector(detection()), FakeTracker(), FakeSink()
        processor = VideoProcessor(detector, tracker, make_config(tmp_path))

        processor.begin(SEAM_METADATA, sink)
        for index, image in enumerate(frames(4), start=1):
            processor.process(stream_frame(index, (index - 1) / 25.0, image), sink)
        result = processor.finish()

        assert detector.calls == tracker.calls == 4
        assert len(sink.frames) == 4
        assert result.frames_processed == 4

    def test_analyzers_are_driven_from_venue_seconds(self, tmp_path: Path) -> None:
        """A venue offset must reach the measurement layers, not source time."""
        processor = VideoProcessor(
            FakeDetector(detection()),
            FakeTracker(),
            make_config(tmp_path),
            zone_analyzer=ZoneAnalyzer(LAYOUT),
        )
        sink = FakeSink()

        processor.begin(SEAM_METADATA, sink)
        for index, image in enumerate(frames(3), start=1):
            processor.process(
                StreamFrame(
                    stream=STREAM,
                    frame_index=index,
                    source_seconds=(index - 1) / 25.0,
                    venue_seconds=(index - 1) / 25.0 + 600.0,
                    timing=TimeUncertainty.unknown(),
                    image=image,
                ),
                sink,
            )
        report = processor.finish().final_zone_report

        assert report is not None
        assert report.timestamp_seconds == 600.0 + 2 / 25.0

    def test_zone_geometry_uses_the_decoded_frame_dimensions(self, tmp_path: Path) -> None:
        processor = VideoProcessor(
            FakeDetector([[Detection.from_xyxy((10.0, 10.0, 30.0, 60.0), 0.9, 1)]]),
            FakeTracker(),
            make_config(tmp_path),
            zone_analyzer=ZoneAnalyzer(LAYOUT),
        )
        sink = FakeSink()

        processor.begin(SEAM_METADATA, sink)
        processor.process(stream_frame(1, 0.0, frames(1)[0]), sink)
        report = processor.finish().final_zone_report

        assert report is not None
        assert report.for_zone("left") is not None
        assert report.for_zone("left").occupancy == 1  # type: ignore[union-attr]

    def test_finish_with_no_frames_reports_an_empty_run(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        processor = VideoProcessor(FakeDetector([]), FakeTracker(), make_config(tmp_path))

        with caplog.at_level(logging.WARNING, logger="atreides_vision.video.processor"):
            result = processor.finish()

        assert result.frames_processed == 0
        assert result.output_path == tmp_path / "out.mp4"
        assert any("No frames" in record.message for record in caplog.records)

    def test_finish_discards_the_run_state_so_the_processor_is_reusable(
        self, tmp_path: Path
    ) -> None:
        processor = VideoProcessor(FakeDetector(detection()), FakeTracker(), make_config(tmp_path))
        sink = FakeSink()

        processor.begin(SEAM_METADATA, sink)
        for index, image in enumerate(frames(3), start=1):
            processor.process(stream_frame(index, (index - 1) / 25.0, image), sink)
        first = processor.finish()

        processor.begin(SEAM_METADATA, sink)
        processor.process(stream_frame(1, 0.0, frames(1)[0]), sink)
        second = processor.finish()

        assert first.frames_processed == 3
        assert second.frames_processed == 1

    def test_the_frame_cap_belongs_to_the_frame_supply_not_the_step(self, tmp_path: Path) -> None:
        """process() handles the frame it is given; run() decides how many there are."""
        processor = VideoProcessor(
            FakeDetector(detection()), FakeTracker(), make_config(tmp_path, max_frames=2)
        )
        sink = FakeSink()

        processor.begin(SEAM_METADATA, sink)
        for index, image in enumerate(frames(5), start=1):
            processor.process(stream_frame(index, (index - 1) / 25.0, image), sink)

        assert processor.finish().frames_processed == 5


class TestRunIsUnchanged:
    def test_run_still_caps_at_max_frames_and_says_so(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        sink = FakeSink()
        with caplog.at_level(logging.INFO, logger="atreides_vision.video.processor"):
            result = VideoProcessor(
                FakeDetector(detection()), FakeTracker(), make_config(tmp_path, max_frames=3)
            ).run(FakeSource(frames(20)), sink)

        assert result.frames_processed == 3
        assert len(sink.frames) == 3
        assert any("--max-frames" in record.message for record in caplog.records)

    def test_a_run_after_a_failed_one_starts_clean(self, tmp_path: Path) -> None:
        """A pass that raised belongs to itself; its counters must not leak."""

        class FlakyDetector(FakeDetector):
            """Fails once, part-way through the first pass, then recovers."""

            def __init__(self, per_frame: list[list[Detection]]) -> None:
                super().__init__(per_frame)
                self.will_fail = True

            def detect(self, frame: np.ndarray) -> list[Detection]:
                if self.will_fail and self.calls >= 2:
                    self.will_fail = False
                    raise RuntimeError("detector crashed")
                return super().detect(frame)

        processor = VideoProcessor(FlakyDetector(detection()), FakeTracker(), make_config(tmp_path))
        with pytest.raises(RuntimeError, match="detector crashed"):
            processor.run(FakeSource(frames(10)), FakeSink())

        second = processor.run(FakeSource(frames(3)), FakeSink())

        assert second.frames_processed == 3

    def test_run_reports_the_sink_path_even_for_an_empty_source(self, tmp_path: Path) -> None:
        sink = FakeSink(tmp_path / "annotated.mp4")

        result = VideoProcessor(FakeDetector([]), FakeTracker(), make_config(tmp_path)).run(
            FakeSource([]), sink
        )

        assert result.frames_processed == 0
        assert result.output_path == tmp_path / "annotated.mp4"

    def test_run_consumes_the_clock_in_the_same_order_as_the_seam(self, tmp_path: Path) -> None:
        """started_at is taken before the first decode, as it always was."""
        images = frames(4)
        metadata = VideoMetadata(width=FRAME_W, height=FRAME_H, fps=25.0, frame_count=4)

        direct = VideoProcessor(
            FakeDetector(detection()),
            FakeTracker(),
            make_config(tmp_path),
            clock=count(start=0.0, step=0.5).__next__,
        ).run(FakeSource(images, metadata), FakeSink())

        processor = VideoProcessor(
            FakeDetector(detection()),
            FakeTracker(),
            make_config(tmp_path),
            clock=count(start=0.0, step=0.5).__next__,
        )
        sink = FakeSink()
        processor.begin(SEAM_METADATA, sink)
        for index, image in enumerate(frames(4), start=1):
            processor.process(stream_frame(index, (index - 1) / 25.0, image), sink)
        seam = processor.finish()

        assert seam.elapsed_seconds == direct.elapsed_seconds

    @pytest.mark.parametrize("fps", [25.0, 30.0, 29.97, 12.5])
    def test_run_timestamps_stay_on_the_original_formula(self, tmp_path: Path, fps: float) -> None:
        metadata = VideoMetadata(width=FRAME_W, height=FRAME_H, fps=fps, frame_count=10)

        result = VideoProcessor(
            FakeDetector([[Detection.from_xyxy((10.0, 10.0, 30.0, 60.0), 0.9, 1)]]),
            FakeTracker(),
            make_config(tmp_path),
            zone_analyzer=ZoneAnalyzer(LAYOUT),
        ).run(FakeSource(frames(10), metadata), FakeSink())

        report = result.final_zone_report
        assert report is not None
        assert report.timestamp_seconds == (10 - 1) / fps

    def test_a_source_reporting_no_frame_rate_still_advances(self, tmp_path: Path) -> None:
        metadata = VideoMetadata(width=FRAME_W, height=FRAME_H, fps=0.0, frame_count=3)

        result = VideoProcessor(
            FakeDetector([[Detection.from_xyxy((10.0, 10.0, 30.0, 60.0), 0.9, 1)]]),
            FakeTracker(),
            make_config(tmp_path),
            zone_analyzer=ZoneAnalyzer(LAYOUT),
        ).run(FakeSource(frames(3), metadata), FakeSink())

        report = result.final_zone_report
        assert report is not None
        assert report.timestamp_seconds == 2.0


class TestDeclaredMetadataIsAuthoritativeForGeometry:
    """The container's declared resolution decides zone geometry, not the pixels.

    Not because the container is reliable — it is not. Rotation metadata is
    reported pre-rotation, so a portrait clip can announce its dimensions
    transposed, and then these dimensions are simply wrong.

    They are used anyway because they are the **same** dimensions the
    composition root builds this camera's projector from. Behaviour positions
    and the zone labels attached to them must be normalised in one space; a
    single site switching to the decoded array would compute zone membership in
    a different space from the metres it is attached to, which is worse than
    both being wrong together. Resolving it properly means changing the
    projector construction sites and this one in one bounded task.

    The probe below is built so the two answers differ, and pins the declared
    one. If this test ever has to be inverted, all four sites move with it.
    """

    #: Decoded 1000 wide by 100 high; the container claims the transpose.
    DECODED_WIDTH, DECODED_HEIGHT = 1000, 100
    DECLARED = VideoMetadata(width=100, height=1000, fps=25.0, frame_count=3)

    #: Feet at x=300, y=90. Against the container's claim that is (3.00, 0.09),
    #: outside the frame and therefore outside every zone. Against the decoded
    #: array it would be (0.30, 0.90) — inside the left-half zone.
    SUBJECT = Detection.from_xyxy((280.0, 40.0, 320.0, 90.0), 0.9, 1)

    def decoded_frames(self, count: int) -> list[np.ndarray]:
        return [
            np.full((self.DECODED_HEIGHT, self.DECODED_WIDTH, 3), 30 + index, dtype=np.uint8)
            for index in range(count)
        ]

    def test_run_normalises_against_the_container_not_the_decoded_frame(
        self, tmp_path: Path
    ) -> None:
        """V0.3.x behaviour, restored exactly."""
        result = VideoProcessor(
            FakeDetector([[self.SUBJECT]]),
            FakeTracker(),
            make_config(tmp_path),
            zone_analyzer=ZoneAnalyzer(LAYOUT),
        ).run(FakeSource(self.decoded_frames(3), self.DECLARED), FakeSink())

        report = result.final_zone_report
        assert report is not None
        left = report.for_zone("left")
        assert left is not None
        # The decoded array's dimensions would have placed this subject inside
        # the zone. The declared ones put it outside the frame.
        assert left.occupancy == 0

    def test_the_venue_seam_normalises_the_same_way(self, tmp_path: Path) -> None:
        """The seam takes the same metadata, so it cannot drift from run()."""
        processor = VideoProcessor(
            FakeDetector([[self.SUBJECT]]),
            FakeTracker(),
            make_config(tmp_path),
            zone_analyzer=ZoneAnalyzer(LAYOUT),
        )
        sink = FakeSink()

        processor.begin(self.DECLARED, sink)
        for index, image in enumerate(self.decoded_frames(3), start=1):
            processor.process(stream_frame(index, (index - 1) / 25.0, image), sink)
        report = processor.finish().final_zone_report

        assert report is not None
        left = report.for_zone("left")
        assert left is not None
        assert left.occupancy == 0

    def test_the_probe_really_does_distinguish_the_two_answers(self) -> None:
        """Guards the guard: with honest metadata the test would prove nothing."""
        track = Track(track_id=1, bbox_xyxy=self.SUBJECT.bbox_xyxy)
        zone = LAYOUT.zones[0]

        decoded = track_floor_point(track, self.DECODED_WIDTH, self.DECODED_HEIGHT)
        declared = track_floor_point(track, self.DECLARED.width, self.DECLARED.height)

        assert zone.contains(decoded)
        assert not zone.contains(declared)

    def test_process_refuses_to_guess_the_geometry(self, tmp_path: Path) -> None:
        """Without begin(), the decoded array is the only thing left to guess from."""
        processor = VideoProcessor(
            FakeDetector([[self.SUBJECT]]),
            FakeTracker(),
            make_config(tmp_path),
            zone_analyzer=ZoneAnalyzer(LAYOUT),
        )

        with pytest.raises(ConfigurationError, match="no frame geometry"):
            processor.process(stream_frame(1, 0.0, self.decoded_frames(1)[0]), FakeSink())

    def test_reopening_a_pass_at_another_resolution_is_refused(self, tmp_path: Path) -> None:
        processor = VideoProcessor(FakeDetector(detection()), FakeTracker(), make_config(tmp_path))
        sink = FakeSink()

        processor.begin(SEAM_METADATA, sink)
        processor.process(stream_frame(1, 0.0, frames(1)[0]), sink)

        with pytest.raises(ConfigurationError, match="re-opened"):
            processor.begin(self.DECLARED, sink)


class TestOneProcessorFollowsOneStream:
    """The provenance guard the scheduler applies, applied at this layer too.

    A pass accumulates one tracker, one trajectory store and one track-id set,
    all camera-local. Frames from a second camera would be tracked as a
    continuation of the first; frames from the next epoch of the same camera
    would be worse, because the index restarts at 1 and trail ageing reads it as
    monotonic.
    """

    OTHER_CAMERA = StreamRef("cam_other", 3)
    NEXT_EPOCH = StreamRef(STREAM.camera_id, STREAM.stream_epoch + 1)

    def opened(self, tmp_path: Path) -> tuple[VideoProcessor, FakeSink]:
        processor = VideoProcessor(FakeDetector(detection()), FakeTracker(), make_config(tmp_path))
        sink = FakeSink()
        processor.begin(SEAM_METADATA, sink)
        processor.process(stream_frame(1, 0.0, frames(1)[0]), sink)
        return processor, sink

    def foreign_frame(self, stream: StreamRef, venue_seconds: float) -> StreamFrame:
        return StreamFrame(
            stream=stream,
            frame_index=1,
            source_seconds=0.0,
            venue_seconds=venue_seconds,
            timing=TimeUncertainty.unknown(),
            image=frames(1)[0],
        )

    def test_a_frame_from_another_camera_is_refused(self, tmp_path: Path) -> None:
        processor, sink = self.opened(tmp_path)

        with pytest.raises(ConfigurationError, match="one processor per stream"):
            processor.process(self.foreign_frame(self.OTHER_CAMERA, 1.0), sink)

    def test_a_frame_from_the_next_epoch_of_the_same_camera_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Same camera id, different recording: unrelated tracklets."""
        processor, sink = self.opened(tmp_path)

        with pytest.raises(ConfigurationError, match="another recording of the same camera"):
            processor.process(self.foreign_frame(self.NEXT_EPOCH, 1.0), sink)

    def test_venue_time_that_does_not_advance_is_refused(self, tmp_path: Path) -> None:
        processor, sink = self.opened(tmp_path)

        with pytest.raises(ConfigurationError, match="not after the previous frame"):
            processor.process(stream_frame(2, 0.0, frames(1)[0]), sink)

    def test_the_binding_is_released_by_finish(self, tmp_path: Path) -> None:
        """A processor may follow another stream once this pass is closed."""
        processor, sink = self.opened(tmp_path)
        processor.finish()

        processor.begin(SEAM_METADATA, sink)
        processor.process(self.foreign_frame(self.OTHER_CAMERA, 0.0), sink)

        assert processor.finish().frames_processed == 1

    def test_the_happy_path_still_runs_the_whole_stream(self, tmp_path: Path) -> None:
        processor = VideoProcessor(FakeDetector(detection()), FakeTracker(), make_config(tmp_path))
        sink = FakeSink()

        processor.begin(SEAM_METADATA, sink)
        for index, image in enumerate(frames(6), start=1):
            processor.process(stream_frame(index, (index - 1) / 25.0, image), sink)

        assert processor.finish().frames_processed == 6

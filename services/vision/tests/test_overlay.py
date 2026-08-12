"""Annotation drawing: it must mark the frame, respect bounds, and not resize."""

from __future__ import annotations

import numpy as np
import pytest

from atreides_vision.config import VisualizationConfig
from atreides_vision.models import BoundingBox, Point, Track
from atreides_vision.video.trajectories import TrajectoryStore
from atreides_vision.visualization.overlay import Annotator, StatusOverlay, color_for_track_id


@pytest.fixture
def frame() -> np.ndarray:
    return np.full((360, 640, 3), 128, dtype=np.uint8)


class TestStatusOverlay:
    def test_lines_match_the_specified_panel(self) -> None:
        lines = StatusOverlay(frame_index=1240, processing_fps=6.42, track_count=18).lines()
        assert lines == (
            "ATREIDES // SEE + TRACK",
            "Frame: 1240",
            "Processing: 6.4 FPS",
            "Tracks: 18",
        )


class TestTrackColors:
    def test_is_deterministic(self) -> None:
        assert color_for_track_id(7) == color_for_track_id(7)

    def test_differs_between_neighbouring_ids(self) -> None:
        assert color_for_track_id(1) != color_for_track_id(2)

    def test_handles_large_ids(self) -> None:
        color = color_for_track_id(10_000)
        assert len(color) == 3
        assert all(0 <= channel <= 255 for channel in color)


class TestDrawing:
    def test_draw_track_marks_the_frame(self, frame: np.ndarray) -> None:
        before = frame.copy()
        Annotator().draw_track(frame, Track(1, BoundingBox(50, 50, 150, 250), 0.9))
        assert not np.array_equal(frame, before)

    def test_drawing_never_changes_frame_shape_or_dtype(self, frame: np.ndarray) -> None:
        annotator = Annotator()
        annotated = annotator.annotate(
            frame,
            [Track(1, BoundingBox(10, 10, 100, 200), 0.8)],
            status=StatusOverlay(1, 12.0, 1),
        )
        assert annotated.shape == (360, 640, 3)
        assert annotated.dtype == np.uint8
        assert annotated is frame, "annotation is in place by design"

    def test_boxes_outside_the_frame_do_not_raise(self, frame: np.ndarray) -> None:
        annotator = Annotator()
        for box in (
            BoundingBox(-100, -100, -10, -10),
            BoundingBox(600, 300, 900, 700),
            BoundingBox(-50, -50, 700, 400),
        ):
            annotator.draw_track(frame, Track(1, box, 0.5))

    def test_label_at_the_top_edge_stays_inside_the_frame(self, frame: np.ndarray) -> None:
        # A box flush with y=0 would push its label off-screen if not flipped.
        Annotator().draw_track(frame, Track(42, BoundingBox(10, 0, 90, 120), 0.9))
        assert frame[0:20, 10:90].std() > 0, "label should be drawn inside the frame"

    def test_confidence_can_be_suppressed(self, frame: np.ndarray) -> None:
        with_conf = frame.copy()
        without_conf = frame.copy()
        track = Track(1, BoundingBox(50, 50, 150, 250), 0.9)

        Annotator(VisualizationConfig(draw_confidence=True)).draw_track(with_conf, track)
        Annotator(VisualizationConfig(draw_confidence=False)).draw_track(without_conf, track)

        assert not np.array_equal(with_conf, without_conf), "labels should differ in width"

    def test_track_without_confidence_still_draws(self, frame: np.ndarray) -> None:
        before = frame.copy()
        Annotator().draw_track(frame, Track(1, BoundingBox(50, 50, 150, 250), None))
        assert not np.array_equal(frame, before)


class TestTrails:
    def test_single_point_trail_draws_nothing(self, frame: np.ndarray) -> None:
        before = frame.copy()
        Annotator().draw_trail(frame, [Point(10, 10)], (255, 0, 0))
        assert np.array_equal(frame, before)

    def test_multi_point_trail_is_drawn(self, frame: np.ndarray) -> None:
        before = frame.copy()
        Annotator().draw_trail(frame, [Point(10, 10), Point(100, 100)], (255, 0, 0))
        assert not np.array_equal(frame, before)

    def test_annotate_draws_trails_from_the_store(self, frame: np.ndarray) -> None:
        store = TrajectoryStore(max_length=10)
        for step in range(5):
            store.update([Track(1, BoundingBox(step * 10, 50, step * 10 + 40, 150))], step + 1)

        with_trail = frame.copy()
        Annotator().annotate(with_trail, [Track(1, BoundingBox(40, 50, 80, 150))], store)

        without_trail = frame.copy()
        Annotator().annotate(without_trail, [Track(1, BoundingBox(40, 50, 80, 150))], None)

        assert not np.array_equal(with_trail, without_trail)


class TestStatusPanel:
    def test_panel_is_drawn_top_left(self, frame: np.ndarray) -> None:
        before = frame.copy()
        Annotator().draw_status(frame, StatusOverlay(100, 6.4, 3))
        assert not np.array_equal(frame[0:100, 0:250], before[0:100, 0:250])
        assert np.array_equal(frame[300:, 400:], before[300:, 400:]), "rest of frame untouched"

    def test_panel_fits_a_tiny_frame(self) -> None:
        tiny = np.zeros((40, 60, 3), dtype=np.uint8)
        Annotator().draw_status(tiny, StatusOverlay(1, 1.0, 0))
        assert tiny.shape == (40, 60, 3)

    def test_annotate_without_status_leaves_the_corner_alone(self, frame: np.ndarray) -> None:
        before = frame.copy()
        Annotator().annotate(frame, [], None, None)
        assert np.array_equal(frame, before)

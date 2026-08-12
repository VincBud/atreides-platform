"""Annotation drawing: it must mark the frame, respect bounds, and not resize."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atreides_vision.config import VisualizationConfig
from atreides_vision.models import BoundingBox, Point, Track
from atreides_vision.video.trajectories import TrajectoryStore
from atreides_vision.visualization.overlay import (
    Annotator,
    StatusOverlay,
    ZoneOverlay,
    color_for_track_id,
    color_for_zone_id,
    color_for_zone_index,
)
from atreides_vision.zones.models import Zone, ZonePolygon, ZoneReport, ZoneSnapshot


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

    def test_title_evolves_when_zones_are_active(self) -> None:
        lines = StatusOverlay(
            frame_index=1240, processing_fps=6.42, track_count=18, zone_count=4
        ).lines()
        assert lines == (
            "ATREIDES // ZONE INTELLIGENCE",
            "Frame: 1240",
            "Processing: 6.4 FPS",
            "Tracks: 18",
            "Zones: 4",
        )

    def test_zero_zones_is_distinct_from_no_zones(self) -> None:
        assert StatusOverlay(1, 1.0, 0, zone_count=0).lines()[0] == "ATREIDES // ZONE INTELLIGENCE"
        assert StatusOverlay(1, 1.0, 0).lines()[0] == "ATREIDES // SEE + TRACK"


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


class TestZoneDrawing:
    def zone(self, zone_id: str = "dancefloor", name: str = "Dancefloor") -> Zone:
        return Zone(
            id=zone_id,
            name=name,
            polygon=ZonePolygon.from_pairs([[0.2, 0.3], [0.8, 0.3], [0.85, 0.8], [0.15, 0.8]]),
        )

    def snapshot(self, zone_id: str = "dancefloor", **overrides: object) -> ZoneSnapshot:
        defaults: dict[str, object] = {
            "zone_id": zone_id,
            "zone_name": "Dancefloor",
            "occupancy": 27,
            "entries": 8,
            "exits": 4,
            "mean_normalized_speed": 0.083,
            "motion_level": 0.54,
            "average_dwell_seconds": 18.2,
            "max_dwell_seconds": 42.0,
            "completed_visits": 4,
        }
        return ZoneSnapshot(**{**defaults, **overrides})  # type: ignore[arg-type]

    def test_zone_polygon_is_drawn(self, frame: np.ndarray) -> None:
        before = frame.copy()
        Annotator().draw_zone(frame, self.zone())
        assert not np.array_equal(frame, before)

    def test_metrics_panel_changes_the_output(self, frame: np.ndarray) -> None:
        without = frame.copy()
        Annotator().draw_zone(without, self.zone())
        with_metrics = frame.copy()
        Annotator().draw_zone(with_metrics, self.zone(), self.snapshot())
        assert not np.array_equal(without, with_metrics)

    def test_occupancy_change_is_visible(self, frame: np.ndarray) -> None:
        quiet = frame.copy()
        busy = frame.copy()
        Annotator().draw_zone(quiet, self.zone(), self.snapshot(occupancy=1))
        Annotator().draw_zone(busy, self.zone(), self.snapshot(occupancy=99))
        assert not np.array_equal(quiet, busy)

    def test_drawing_preserves_shape_and_dtype(self, frame: np.ndarray) -> None:
        Annotator().draw_zone(frame, self.zone(), self.snapshot())
        assert frame.shape == (360, 640, 3)
        assert frame.dtype == np.uint8

    def test_fill_can_be_disabled(self, frame: np.ndarray) -> None:
        outline_only = frame.copy()
        Annotator(VisualizationConfig(zone_fill_alpha=0.0)).draw_zone(outline_only, self.zone())
        filled = frame.copy()
        Annotator(VisualizationConfig(zone_fill_alpha=0.5)).draw_zone(filled, self.zone())
        assert not np.array_equal(outline_only, filled)

    def test_zone_touching_the_frame_edge_does_not_raise(self, frame: np.ndarray) -> None:
        edge = Zone(
            id="full",
            name="Full Frame",
            polygon=ZonePolygon.from_pairs([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        )
        Annotator().draw_zone(frame, edge, self.snapshot())

    def test_tiny_frame_does_not_raise(self) -> None:
        tiny = np.zeros((30, 40, 3), dtype=np.uint8)
        Annotator().draw_zone(tiny, self.zone(), self.snapshot())
        assert tiny.shape == (30, 40, 3)

    def test_zone_id_colors_are_deterministic(self) -> None:
        assert color_for_zone_id("bar") == color_for_zone_id("bar")

    def test_configured_zones_get_distinct_colors(self) -> None:
        """Colour by position, so no two configured zones can render alike."""
        colors = [color_for_zone_index(i) for i in range(6)]
        assert len(set(colors)) == 6

    def test_example_config_zones_are_all_visually_distinct(self) -> None:
        from atreides_vision.zones.loader import load_zone_layout

        example = Path(__file__).resolve().parents[3] / "config" / "zones.example.json"
        layout = load_zone_layout(example)
        colors = [color_for_zone_index(i) for i in range(len(layout))]
        assert len(set(colors)) == len(layout)

    def test_annotate_draws_zones_behind_tracks(self, frame: np.ndarray) -> None:
        """Zones are venue structure; people are drawn on top of them."""
        overlay = ZoneOverlay(zones=(self.zone(),), snapshots={"dancefloor": self.snapshot()})
        zones_only = frame.copy()
        Annotator().annotate(zones_only, [], None, None, overlay)

        with_track = frame.copy()
        Annotator().annotate(
            with_track, [Track(1, BoundingBox(250, 400, 350, 700), 0.9)], None, None, overlay
        )
        assert not np.array_equal(zones_only, with_track)

    def test_zone_drawing_can_be_switched_off(self, frame: np.ndarray) -> None:
        overlay = ZoneOverlay(zones=(self.zone(),), snapshots={})
        untouched = frame.copy()
        Annotator(VisualizationConfig(draw_zones=False)).annotate(frame, [], None, None, overlay)
        assert np.array_equal(frame, untouched)

    def test_overlay_from_report_pairs_snapshots(self) -> None:
        report = ZoneReport(frame_index=1, timestamp_seconds=0.0, snapshots=(self.snapshot(),))
        overlay = ZoneOverlay.from_report([self.zone()], report)
        assert overlay.snapshots["dancefloor"].occupancy == 27

    def test_top_left_zone_label_slides_clear_of_the_status_panel(self, frame: np.ndarray) -> None:
        """A zone anchored in the corner must not have its metrics hidden."""
        corner = Zone(
            id="bar",
            name="Bar",
            polygon=ZonePolygon.from_pairs([[0.0, 0.0], [0.9, 0.0], [0.9, 0.2], [0.0, 0.2]]),
        )
        overlay = ZoneOverlay(zones=(corner,), snapshots={"bar": self.snapshot("bar")})
        status = StatusOverlay(frame_index=1, processing_fps=6.0, track_count=2, zone_count=1)

        annotator = Annotator()
        panel_width, panel_height = annotator._status_panel_size(frame, status)

        annotated = frame.copy()
        annotator.annotate(annotated, [], None, status, overlay)

        # Below the status panel, inside the zone, the label must have marked pixels.
        strip = annotated[panel_height : panel_height + 60, 0:panel_width]
        assert strip.std() > 0, "zone metrics should be drawn below the status panel"

    def test_zone_label_is_unmoved_when_there_is_no_status_panel(self, frame: np.ndarray) -> None:
        overlay = ZoneOverlay(zones=(self.zone(),), snapshots={"dancefloor": self.snapshot()})
        without_status = frame.copy()
        Annotator().annotate(without_status, [], None, None, overlay)
        assert not np.array_equal(without_status, frame)

    def test_overlay_from_missing_report_still_draws_boundaries(self, frame: np.ndarray) -> None:
        overlay = ZoneOverlay.from_report([self.zone()], None)
        assert overlay.snapshots == {}
        before = frame.copy()
        Annotator().annotate(frame, [], None, None, overlay)
        assert not np.array_equal(frame, before)


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

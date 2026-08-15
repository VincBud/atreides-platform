"""Zone membership reaching behaviour output as neutral spatial context.

The boundary being tested is as important as the feature: zone ids travel to
the behaviour layer as **opaque strings**. Behaviour never learns what a zone
is, never sees a polygon, and never imports the zone package — so it keeps
working with no zones configured, and zones keep working with no behaviour.

Zones remain **camera-local image-space** regions. Attaching a zone label to a
metric position does not turn the polygon into venue geometry, and nothing here
pretends otherwise.
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import atreides_vision.behavior as behavior_package
from atreides_vision.behavior.analyzer import BehaviorAnalyzer
from atreides_vision.behavior.coordinates import MotionSpace, NormalizedFrameProjector
from atreides_vision.behavior.models import SignalType, SubjectRef
from atreides_vision.behavior.reporting import (
    BehaviorRunMetadata,
    JsonlBehaviorWriter,
    SourceProvenance,
)
from atreides_vision.config import PipelineConfig, VisionConfig, VisualizationConfig
from atreides_vision.models import BoundingBox, Detection, Track
from atreides_vision.video.processor import VideoProcessor
from atreides_vision.video.source import VideoMetadata
from atreides_vision.zones.analyzer import ZoneAnalyzer
from atreides_vision.zones.models import Zone, ZoneLayout, ZonePolygon
from fakes import FakeDetector, FakeSink, FakeSource, FakeTracker

FRAME_W, FRAME_H = 400, 300
FPS = 25.0
CAMERA = "cam_01"


def zone(zone_id: str, x0: float, y0: float, x1: float, y1: float) -> Zone:
    return Zone(
        id=zone_id,
        name=zone_id.title(),
        polygon=ZonePolygon.from_pairs([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]),
    )


#: Two separated regions plus a band overlapping both. Deliberately does not
#: tile the frame, so "in no zone at all" is a reachable state, and the band
#: overlaps so multi-zone membership is too — both are supported configurations.
LAYOUT = ZoneLayout(
    camera_id=CAMERA,
    zones=(
        zone("left", 0.0, 0.0, 0.4, 1.0),
        zone("right", 0.6, 0.0, 1.0, 1.0),
        zone("walkway", 0.2, 0.3, 0.8, 0.7),
    ),
)


def track_at(x_norm: float, y_norm: float, track_id: int = 1) -> Track:
    cx, feet = x_norm * FRAME_W, y_norm * FRAME_H
    return Track(track_id=track_id, bbox_xyxy=BoundingBox(cx - 10, feet - 50, cx + 10, feet))


def detection_at(x_norm: float, y_norm: float = 0.5) -> Detection:
    cx, feet = x_norm * FRAME_W, y_norm * FRAME_H
    return Detection(
        bbox_xyxy=BoundingBox(cx - 10, feet - 50, cx + 10, feet), confidence=0.9, class_id=1
    )


def frames(count_: int) -> list[np.ndarray]:
    return [np.full((FRAME_H, FRAME_W, 3), 60, dtype=np.uint8) for _ in range(count_)]


def source(count_: int) -> FakeSource:
    return FakeSource(
        frames(count_),
        VideoMetadata(width=FRAME_W, height=FRAME_H, fps=FPS, frame_count=count_),
    )


def make_config(tmp_path: Path) -> VisionConfig:
    return VisionConfig(
        pipeline=PipelineConfig(tmp_path / "in.mp4", tmp_path / "out.mp4"),
        visualization=VisualizationConfig(trail_length=5),
    )


def make_analyzer(camera_id: str = CAMERA) -> BehaviorAnalyzer:
    return BehaviorAnalyzer(
        camera_id=camera_id, projector=NormalizedFrameProjector(FRAME_W, FRAME_H)
    )


class TestZoneAnalyzerExposesMembership:
    def test_a_track_inside_one_zone(self) -> None:
        analyzer = ZoneAnalyzer(LAYOUT)
        analyzer.update([track_at(0.1, 0.9)], 1, 0.0, FRAME_W, FRAME_H)
        assert analyzer.zone_ids_by_track == {1: ("left",)}

    def test_a_track_in_overlapping_zones_keeps_both(self) -> None:
        """Dropping one would lose information the zone layer already has."""
        analyzer = ZoneAnalyzer(LAYOUT)
        analyzer.update([track_at(0.3, 0.5)], 1, 0.0, FRAME_W, FRAME_H)
        assert analyzer.zone_ids_by_track == {1: ("left", "walkway")}

    def test_a_track_outside_every_zone_is_absent(self) -> None:
        """Absence represents "no zone" — no sentinel label is invented."""
        analyzer = ZoneAnalyzer(LAYOUT)
        # x=0.5 falls in the gap between left and right; y=0.05 is above the walkway.
        analyzer.update([track_at(0.5, 0.05)], 1, 0.0, FRAME_W, FRAME_H)
        assert analyzer.zone_ids_by_track == {}

    def test_several_tracks_in_different_zones(self) -> None:
        analyzer = ZoneAnalyzer(LAYOUT)
        analyzer.update(
            [track_at(0.1, 0.9, 1), track_at(0.9, 0.9, 2), track_at(0.5, 0.05, 3)],
            1,
            0.0,
            FRAME_W,
            FRAME_H,
        )
        assert analyzer.zone_ids_by_track == {1: ("left",), 2: ("right",)}

    def test_ordering_follows_the_configuration(self) -> None:
        analyzer = ZoneAnalyzer(LAYOUT)
        analyzer.update([track_at(0.3, 0.5)], 1, 0.0, FRAME_W, FRAME_H)
        assert analyzer.zone_ids_by_track[1] == ("left", "walkway")

    def test_it_reflects_only_the_latest_frame(self) -> None:
        analyzer = ZoneAnalyzer(LAYOUT)
        analyzer.update([track_at(0.1, 0.9)], 1, 0.0, FRAME_W, FRAME_H)
        analyzer.update([track_at(0.9, 0.9)], 2, 0.04, FRAME_W, FRAME_H)
        assert analyzer.zone_ids_by_track == {1: ("right",)}

    def test_it_is_empty_before_any_frame(self) -> None:
        assert ZoneAnalyzer(LAYOUT).zone_ids_by_track == {}

    def test_zone_metrics_are_unaffected(self) -> None:
        """Exposing membership must not disturb what V0.2 already reports."""
        analyzer = ZoneAnalyzer(LAYOUT)
        report = analyzer.update([track_at(0.1, 0.9)], 1, 0.0, FRAME_W, FRAME_H)
        left = report.for_zone("left")
        assert left is not None
        assert left.occupancy == 1
        assert left.entries == 1


class TestLayerBoundary:
    def test_behaviour_receives_ids_as_opaque_strings(self) -> None:
        analyzer = make_analyzer()
        frame = analyzer.update(
            [track_at(0.3, 0.5)],
            frame_index=1,
            timestamp_seconds=0.0,
            zone_ids_by_track={1: ["anything", "at", "all"]},
        )
        snapshot = frame.snapshots[0]
        assert snapshot.zone_ids == ("anything", "at", "all")

    def test_no_behaviour_module_imports_the_zone_package(self) -> None:
        """The invariant that keeps each layer usable without the other.

        Checked by parsing the behaviour package's own imports rather than by
        inspecting ``sys.modules``: the top-level ``atreides_vision`` package
        pulls in the zone package through ``config``, which would mask the thing
        actually worth guarding — a behaviour module reaching for zone code.
        """
        import ast

        package = Path(behavior_package.__file__).parent
        offenders: list[str] = []

        for module_path in sorted(package.rglob("*.py")):
            tree = ast.parse(module_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "atreides_vision.zones"
                ):
                    offenders.append(f"{module_path.name}: from {node.module}")
                elif isinstance(node, ast.Import):
                    offenders.extend(
                        f"{module_path.name}: import {alias.name}"
                        for alias in node.names
                        if alias.name.startswith("atreides_vision.zones")
                    )

        assert offenders == [], f"behaviour must not import the zone layer: {offenders}"

    def test_behaviour_works_with_no_zone_context_at_all(self) -> None:
        analyzer = make_analyzer()
        frame = analyzer.update([track_at(0.3, 0.5)], frame_index=1, timestamp_seconds=0.0)
        assert frame.snapshots[0].zone_ids == ()


class TestProcessorWiring:
    @staticmethod
    def run(tmp_path: Path, with_zones: bool, camera_id: str = CAMERA) -> list[dict[str, Any]]:
        analyzer = make_analyzer(camera_id)
        metadata = BehaviorRunMetadata(
            camera_id=camera_id,
            space=MotionSpace.NORMALIZED_IMAGE,
            source=SourceProvenance(path="synthetic", width=FRAME_W, height=FRAME_H, fps=FPS),
        )
        path = tmp_path / f"{camera_id}.jsonl"
        with JsonlBehaviorWriter(path, metadata) as writer:
            VideoProcessor(
                FakeDetector([[detection_at(0.3, 0.5)]]),
                FakeTracker(),
                make_config(tmp_path),
                clock=count(start=0.0, step=0.04).__next__,
                zone_analyzer=ZoneAnalyzer(LAYOUT) if with_zones else None,
                behavior_analyzer=analyzer,
                behavior_writer=writer,
            ).run(source(10), FakeSink())
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_zone_ids_appear_in_the_output(self, tmp_path: Path) -> None:
        records = [
            r for r in self.run(tmp_path, with_zones=True) if r["record_type"] == "observation"
        ]
        assert records
        assert all(record["zone_ids"] == ["left", "walkway"] for record in records)

    def test_no_zone_analyzer_means_no_zone_ids(self, tmp_path: Path) -> None:
        records = [
            r for r in self.run(tmp_path, with_zones=False) if r["record_type"] == "observation"
        ]
        assert records
        assert all("zone_ids" not in record for record in records)

    def test_zone_context_does_not_change_positions_or_signals(self, tmp_path: Path) -> None:
        with_zones = [
            r
            for r in self.run(tmp_path, with_zones=True, camera_id="cam_a")
            if r["record_type"] == "observation"
        ]
        without = [
            r
            for r in self.run(tmp_path, with_zones=False, camera_id="cam_b")
            if r["record_type"] == "observation"
        ]
        for left, right in zip(with_zones, without, strict=True):
            assert left["position"] == right["position"]
            assert left["signals"] == right["signals"]

    def test_context_preserves_camera_local_identity(self, tmp_path: Path) -> None:
        """A shared zone is context, never grounds to merge subjects."""
        left = [r for r in self.run(tmp_path, True, "cam_a") if r["record_type"] == "observation"]
        right = [r for r in self.run(tmp_path, True, "cam_b") if r["record_type"] == "observation"]

        assert {record["camera_id"] for record in left} == {"cam_a"}
        assert {record["camera_id"] for record in right} == {"cam_b"}
        # Same zone, same track id, same timestamps — still two distinct subjects.
        assert left[0]["zone_ids"] == right[0]["zone_ids"]
        assert left[0]["subject"] == right[0]["subject"]
        assert left[0]["camera_id"] != right[0]["camera_id"]

    def test_signals_still_measure_correctly_with_zone_context(self, tmp_path: Path) -> None:
        analyzer = make_analyzer()
        script = [[detection_at(0.1 + 0.008 * step, 0.5)] for step in range(40)]
        VideoProcessor(
            FakeDetector(script),
            FakeTracker(),
            make_config(tmp_path),
            zone_analyzer=ZoneAnalyzer(LAYOUT),
            behavior_analyzer=analyzer,
        ).run(source(40), FakeSink())

        latest = analyzer.latest_frame
        assert latest is not None
        speed = latest.value_for(SubjectRef(CAMERA, 0), SignalType.SPEED)
        assert speed == pytest.approx(0.2, rel=1e-6)


class TestImageSpaceBoundary:
    def test_zones_remain_normalized_image_polygons(self) -> None:
        """V0.3.3 passes labels, it does not convert polygons to metres."""
        for configured in LAYOUT:
            for vertex in configured.polygon.vertices:
                assert 0.0 <= vertex.x <= 1.0
                assert 0.0 <= vertex.y <= 1.0

    def test_zone_ids_carry_no_coordinate_space(self) -> None:
        """The label says nothing about geometry, which is why it is safe to pass."""
        analyzer = ZoneAnalyzer(LAYOUT)
        analyzer.update([track_at(0.3, 0.5)], 1, 0.0, FRAME_W, FRAME_H)
        assert all(isinstance(value, str) for value in analyzer.zone_ids_by_track[1])

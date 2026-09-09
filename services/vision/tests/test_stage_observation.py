"""V0.4.10a — the camera-local perception/tracking observability seam.

These tests drive :mod:`atreides_vision.venue_validation.stage_observation`
directly, against synthetic :class:`TrackingDebugRecord` facts and a small
fake :class:`MotionProjector` — never a real detector, tracker or camera.
Nothing here is an accuracy measurement: it proves the module reshapes exactly
the facts it was handed, truthfully, and never invents or drops one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.errors import ProjectionError, ValidationError
from atreides_vision.models import BoundingBox, Detection, Point, Track
from atreides_vision.timing import StreamFrame, StreamRef, TimeUncertainty
from atreides_vision.tracking.diagnostics import InternalTrackSnapshot, TrackingDebugRecord
from atreides_vision.venue_validation.stage_observation import (
    LatestTrackingRecord,
    build_camera_stage_snapshot,
)

STREAM = StreamRef(camera_id="cam_a", stream_epoch=0)


@dataclass(frozen=True, slots=True)
class FakeProjector:
    """A :class:`MotionProjector` whose behaviour a test controls directly.

    ``unprojectable_x`` marks anchor x-coordinates that raise
    :class:`ProjectionError`, the same narrow failure a real calibrated
    projector raises beyond its horizon. Everything else maps to
    ``(x, y)`` unchanged, which is all a truthful test oracle needs.
    """

    declared_space: MotionSpace
    unprojectable_x: frozenset[float] = frozenset()

    @property
    def space(self) -> MotionSpace:
        return self.declared_space

    def project(self, point: Point) -> MotionPoint:
        if point.x in self.unprojectable_x:
            raise ProjectionError(f"Beyond the horizon: {point.x}")
        return MotionPoint(x=point.x, y=point.y, space=self.declared_space)


def box(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(x1, y1, x2, y2)


def frame_at(frame_index: int, venue_seconds: float = 1.0) -> StreamFrame:
    return StreamFrame(
        stream=STREAM,
        frame_index=frame_index,
        source_seconds=venue_seconds,
        venue_seconds=venue_seconds,
        timing=TimeUncertainty.unknown(),
        image=np.zeros((4, 4, 3), dtype=np.uint8),
    )


class TestLatestTrackingRecord:
    """A single synchronous slot, and nothing more."""

    def test_take_without_a_write_raises(self) -> None:
        capture = LatestTrackingRecord()
        with pytest.raises(ValidationError):
            capture.take()

    def test_write_then_take_returns_exactly_that_record(self) -> None:
        capture = LatestTrackingRecord()
        record = TrackingDebugRecord(
            frame_index=1, detections=(), reported_tracks=(), internal_tracks=()
        )
        capture.write(record)
        assert capture.take() is record

    def test_take_is_destructive(self) -> None:
        capture = LatestTrackingRecord()
        capture.write(
            TrackingDebugRecord(
                frame_index=1, detections=(), reported_tracks=(), internal_tracks=()
            )
        )
        capture.take()
        with pytest.raises(ValidationError):
            capture.take()

    def test_release_is_a_harmless_no_op(self) -> None:
        capture = LatestTrackingRecord()
        capture.release()  # must not raise, even with nothing written


class TestFrameRecordCrossCheck:
    def test_mismatched_frame_index_raises(self) -> None:
        record = TrackingDebugRecord(
            frame_index=2, detections=(), reported_tracks=(), internal_tracks=()
        )
        with pytest.raises(ValidationError):
            build_camera_stage_snapshot(
                record, frame_at(1), FakeProjector(MotionSpace.NORMALIZED_IMAGE)
            )


class TestDetectionSuppressionSemantics:
    """Kept and suppressed detections, exactly as deduplication reported them."""

    def test_without_deduplication_every_detection_is_reported_kept(self) -> None:
        detection = Detection(bbox_xyxy=box(0, 0, 10, 20), confidence=0.8, class_id=0)
        record = TrackingDebugRecord(
            frame_index=1,
            detections=(detection,),
            reported_tracks=(),
            internal_tracks=(),
            raw_detections=None,
            removed_detections=(),
        )
        snapshot = build_camera_stage_snapshot(
            record, frame_at(1), FakeProjector(MotionSpace.NORMALIZED_IMAGE)
        )

        assert len(snapshot.detections) == 1
        assert snapshot.detections[0].suppressed is False

    def test_a_kept_and_a_suppressed_detection_are_told_apart(self) -> None:
        kept = Detection(bbox_xyxy=box(0, 0, 10, 20), confidence=0.9, class_id=0)
        removed = Detection(bbox_xyxy=box(1, 1, 11, 21), confidence=0.5, class_id=0)
        record = TrackingDebugRecord(
            frame_index=1,
            detections=(kept,),
            reported_tracks=(),
            internal_tracks=(),
            raw_detections=(kept, removed),
            removed_detections=(removed,),
        )
        snapshot = build_camera_stage_snapshot(
            record, frame_at(1), FakeProjector(MotionSpace.NORMALIZED_IMAGE)
        )

        assert len(snapshot.detections) == 2
        by_suppressed = {d.suppressed: d for d in snapshot.detections}
        assert by_suppressed[False].bbox == kept.bbox_xyxy
        assert by_suppressed[True].bbox == removed.bbox_xyxy

    def test_suppression_is_matched_by_identity_not_equality(self) -> None:
        """Two coordinate-identical detections must still be told apart.

        Matching by ``==`` instead of ``id()`` would conflate a kept detection
        with a distinct-but-equal removed one, corrupting the semantic
        deduplication itself is defined by (object identity, not value).
        """
        kept = Detection(bbox_xyxy=box(0, 0, 10, 20), confidence=0.9, class_id=0)
        removed_twin = Detection(bbox_xyxy=box(0, 0, 10, 20), confidence=0.9, class_id=0)
        record = TrackingDebugRecord(
            frame_index=1,
            detections=(kept,),
            reported_tracks=(),
            internal_tracks=(),
            raw_detections=(kept, removed_twin),
            removed_detections=(removed_twin,),
        )
        snapshot = build_camera_stage_snapshot(
            record, frame_at(1), FakeProjector(MotionSpace.NORMALIZED_IMAGE)
        )

        assert sum(1 for d in snapshot.detections if d.suppressed) == 1
        assert sum(1 for d in snapshot.detections if not d.suppressed) == 1


class TestTrackSnapshots:
    def test_every_reported_track_is_carried_through(self) -> None:
        track = Track(track_id=3, bbox_xyxy=box(0, 0, 10, 20), confidence=0.7)
        record = TrackingDebugRecord(
            frame_index=1, detections=(), reported_tracks=(track,), internal_tracks=()
        )
        snapshot = build_camera_stage_snapshot(
            record, frame_at(1), FakeProjector(MotionSpace.NORMALIZED_IMAGE)
        )

        assert len(snapshot.tracks) == 1
        assert snapshot.tracks[0].track_id == 3
        assert snapshot.tracks[0].bbox == track.bbox_xyxy
        assert snapshot.tracks[0].confidence == 0.7


class TestTentativeTracklets:
    """A tracklet the tracker holds but has not yet made public."""

    def test_a_tentative_tracklet_appears_only_among_tracklets(self) -> None:
        tentative = InternalTrackSnapshot(
            track_id=-1,
            state="tracked",
            age=1,
            time_since_update=0,
            last_observation_bbox=[0.0, 0.0, 10.0, 20.0],
            predicted_bbox=[0.0, 0.0, 10.0, 20.0],
            consecutive_updates=1,
            observed=True,
            velocity=None,
        )
        record = TrackingDebugRecord(
            frame_index=1,
            detections=(),
            reported_tracks=(),  # not yet matured into a public Track
            internal_tracks=(tentative,),
        )
        snapshot = build_camera_stage_snapshot(
            record, frame_at(1), FakeProjector(MotionSpace.NORMALIZED_IMAGE)
        )

        assert snapshot.tracks == ()
        assert len(snapshot.tracklets) == 1
        assert snapshot.tracklets[0].track_id == -1
        assert snapshot.tracklets[0].last_observation_position == MotionPoint(
            x=5.0, y=20.0, space=MotionSpace.NORMALIZED_IMAGE
        )

    def test_a_tracklet_with_no_last_observation_projects_to_none(self) -> None:
        lost = InternalTrackSnapshot(
            track_id=-1,
            state="lost",
            age=5,
            time_since_update=3,
            last_observation_bbox=None,
            predicted_bbox=[0.0, 0.0, 10.0, 20.0],
            consecutive_updates=0,
            observed=False,
            velocity=None,
        )
        record = TrackingDebugRecord(
            frame_index=1, detections=(), reported_tracks=(), internal_tracks=(lost,)
        )
        snapshot = build_camera_stage_snapshot(
            record, frame_at(1), FakeProjector(MotionSpace.NORMALIZED_IMAGE)
        )

        assert snapshot.tracklets[0].last_observation_position is None
        assert snapshot.tracklets[0].predicted_position is not None


class TestProjectionFailure:
    """Beyond the calibration horizon, the answer is ``None`` — never a guess."""

    def test_an_unprojectable_detection_reports_position_none(self) -> None:
        detection = Detection(bbox_xyxy=box(0, 0, 10, 20), confidence=0.8, class_id=0)
        record = TrackingDebugRecord(
            frame_index=1, detections=(detection,), reported_tracks=(), internal_tracks=()
        )
        # bottom_center anchor x is (0 + 10) / 2 == 5.0
        projector = FakeProjector(MotionSpace.GROUND_PLANE_METRES, unprojectable_x=frozenset({5.0}))
        snapshot = build_camera_stage_snapshot(record, frame_at(1), projector)

        assert snapshot.detections[0].position is None

    def test_an_unprojectable_track_reports_position_none(self) -> None:
        track = Track(track_id=1, bbox_xyxy=box(0, 0, 10, 20), confidence=0.5)
        record = TrackingDebugRecord(
            frame_index=1, detections=(), reported_tracks=(track,), internal_tracks=()
        )
        projector = FakeProjector(MotionSpace.GROUND_PLANE_METRES, unprojectable_x=frozenset({5.0}))
        snapshot = build_camera_stage_snapshot(record, frame_at(1), projector)

        assert snapshot.tracks[0].position is None


class TestSpaceIsDeclaredNotInferred:
    def test_space_reflects_the_projector_even_when_every_position_is_none(self) -> None:
        detection = Detection(bbox_xyxy=box(0, 0, 10, 20), confidence=0.8, class_id=0)
        record = TrackingDebugRecord(
            frame_index=1, detections=(detection,), reported_tracks=(), internal_tracks=()
        )
        projector = FakeProjector(MotionSpace.GROUND_PLANE_METRES, unprojectable_x=frozenset({5.0}))
        snapshot = build_camera_stage_snapshot(record, frame_at(1), projector)

        assert snapshot.space is MotionSpace.GROUND_PLANE_METRES
        assert snapshot.detections[0].position is None

    def test_an_uncalibrated_projector_never_reports_ground_plane_metres(self) -> None:
        record = TrackingDebugRecord(
            frame_index=1, detections=(), reported_tracks=(), internal_tracks=()
        )
        snapshot = build_camera_stage_snapshot(
            record, frame_at(1), FakeProjector(MotionSpace.NORMALIZED_IMAGE)
        )

        assert snapshot.space is MotionSpace.NORMALIZED_IMAGE


class TestSnapshotIdentity:
    def test_the_snapshot_carries_the_frame_provenance(self) -> None:
        record = TrackingDebugRecord(
            frame_index=7, detections=(), reported_tracks=(), internal_tracks=()
        )
        frame = frame_at(7, venue_seconds=3.5)
        snapshot = build_camera_stage_snapshot(
            record, frame, FakeProjector(MotionSpace.NORMALIZED_IMAGE)
        )

        assert snapshot.stream == STREAM
        assert snapshot.frame_index == 7
        assert snapshot.venue_seconds == 3.5

    def test_the_result_types_are_immutable(self) -> None:
        record = TrackingDebugRecord(
            frame_index=1,
            detections=(Detection(bbox_xyxy=box(0, 0, 10, 20), confidence=0.8, class_id=0),),
            reported_tracks=(Track(track_id=1, bbox_xyxy=box(0, 0, 10, 20)),),
            internal_tracks=(),
        )
        snapshot = build_camera_stage_snapshot(
            record, frame_at(1), FakeProjector(MotionSpace.NORMALIZED_IMAGE)
        )

        with pytest.raises(AttributeError):
            snapshot.frame_index = 99  # type: ignore[misc]
        with pytest.raises(AttributeError):
            snapshot.detections[0].suppressed = True  # type: ignore[misc]

    def test_two_builds_from_identical_facts_are_equal(self) -> None:
        def make() -> TrackingDebugRecord:
            return TrackingDebugRecord(
                frame_index=1,
                detections=(Detection(bbox_xyxy=box(0, 0, 10, 20), confidence=0.8, class_id=0),),
                reported_tracks=(Track(track_id=1, bbox_xyxy=box(0, 0, 10, 20)),),
                internal_tracks=(),
            )

        projector = FakeProjector(MotionSpace.NORMALIZED_IMAGE)
        first = build_camera_stage_snapshot(make(), frame_at(1), projector)
        second = build_camera_stage_snapshot(make(), frame_at(1), projector)
        assert first == second


def _import_lines(module_path: str) -> list[str]:
    import ast
    import pathlib

    text = (
        pathlib.Path(__file__).resolve().parents[1] / "src/atreides_vision" / module_path
    ).read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            lines.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            lines.append(node.module)
    return lines


class TestBoundaries:
    """No model-library or lower-layer leakage across this module's surface."""

    def test_no_forbidden_library_is_imported(self) -> None:
        imports = _import_lines("venue_validation/stage_observation.py")
        forbidden = ("rfdetr", "trackers", "supervision", "cv2", "torch", "numpy", "np")
        assert not any(
            module == name or module.startswith(f"{name}.")
            for module in imports
            for name in forbidden
        )

    def test_no_wildtrack_or_dataset_module_is_imported(self) -> None:
        imports = _import_lines("venue_validation/stage_observation.py")
        assert not any("wildtrack" in module.lower() for module in imports)
        assert not any("dataset" in module.lower() for module in imports)

    def test_no_lower_layer_imports_the_observability_seam(self) -> None:
        """The observability module is consumed by the validation runner only —
        it must never be imported back into detection, tracking or behaviour."""
        import pathlib

        forbidden_dirs = ("video", "tracking", "behavior", "interaction", "crowd", "zones")
        src_root = pathlib.Path(__file__).resolve().parents[1] / "src/atreides_vision"
        offenders = []
        for directory in forbidden_dirs:
            for path in (src_root / directory).rglob("*.py"):
                if "stage_observation" in path.read_text(encoding="utf-8"):
                    offenders.append(str(path))
        assert offenders == []

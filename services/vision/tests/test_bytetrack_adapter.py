"""ByteTrack adapter conversions, exercised with a fake backend.

The ``supervision`` conversion is tested against the real library (it is a pure
data structure, no weights), while ByteTrack itself is faked so the tests stay
deterministic and fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from atreides_vision.config import TrackerConfig
from atreides_vision.errors import TrackerError
from atreides_vision.models import BoundingBox, Detection, Track
from atreides_vision.tracking.base import MultiObjectTracker
from atreides_vision.tracking.bytetrack import ByteTrackTracker


@dataclass
class FakeTrackedResult:
    """Stands in for the ``sv.Detections`` ByteTrack returns."""

    xyxy: np.ndarray
    tracker_id: np.ndarray | None = None
    confidence: np.ndarray | None = None


@dataclass
class FakeByteTrackBackend:
    result: FakeTrackedResult
    received: list[Any] = field(default_factory=list)
    reset_calls: int = 0

    def update(self, detections: Any) -> FakeTrackedResult:
        self.received.append(detections)
        return self.result

    def reset(self) -> None:
        self.reset_calls += 1


def build_tracker(
    result: FakeTrackedResult,
) -> tuple[ByteTrackTracker, FakeByteTrackBackend]:
    backend = FakeByteTrackBackend(result=result)
    tracker = ByteTrackTracker(TrackerConfig(), backend_factory=lambda _config: backend)
    return tracker, backend


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((120, 160, 3), dtype=np.uint8)


class TestInterfaceCompliance:
    def test_satisfies_the_atreides_protocol(self) -> None:
        tracker, _ = build_tracker(FakeTrackedResult(xyxy=np.empty((0, 4))))
        assert isinstance(tracker, MultiObjectTracker)
        assert tracker.name == "ByteTrack"

    def test_reset_delegates_to_the_backend(self) -> None:
        tracker, backend = build_tracker(FakeTrackedResult(xyxy=np.empty((0, 4))))
        tracker.reset()
        assert backend.reset_calls == 1


class TestDetectionsToSupervision:
    def test_converts_atreides_detections_into_sv_detections(self, frame: np.ndarray) -> None:
        import supervision as sv

        tracker, backend = build_tracker(FakeTrackedResult(xyxy=np.empty((0, 4))))
        detections = [
            Detection.from_xyxy((0, 0, 10, 20), 0.9, 1),
            Detection.from_xyxy((30, 30, 60, 90), 0.6, 1),
        ]
        tracker.update(detections, frame)

        passed = backend.received[0]
        assert isinstance(passed, sv.Detections)
        assert len(passed) == 2
        assert passed.confidence is not None
        np.testing.assert_allclose(passed.xyxy[1], [30.0, 30.0, 60.0, 90.0])
        np.testing.assert_allclose(passed.confidence, [0.9, 0.6], rtol=1e-6)

    def test_empty_detections_produce_an_empty_sv_detections(self, frame: np.ndarray) -> None:
        import supervision as sv

        tracker, backend = build_tracker(FakeTrackedResult(xyxy=np.empty((0, 4))))
        tracker.update([], frame)

        passed = backend.received[0]
        assert isinstance(passed, sv.Detections)
        assert len(passed) == 0

    def test_frame_is_not_forwarded_to_bytetrack(self, frame: np.ndarray) -> None:
        """ByteTrack does no motion compensation and warns if handed a frame."""

        class StrictBackend:
            def update(self, detections: Any) -> FakeTrackedResult:
                return FakeTrackedResult(xyxy=np.empty((0, 4)))

            def reset(self) -> None:
                pass

        tracker = ByteTrackTracker(TrackerConfig(), backend_factory=lambda _c: StrictBackend())
        # Would raise TypeError if the adapter passed frame= through.
        assert tracker.update([], frame) == []


class TestSupervisionToTracks:
    def test_builds_tracks_with_ids_boxes_and_confidence(self, frame: np.ndarray) -> None:
        tracker, _ = build_tracker(
            FakeTrackedResult(
                xyxy=np.array([[0, 0, 10, 20], [30, 30, 60, 90]], dtype=np.float32),
                tracker_id=np.array([4, 9], dtype=int),
                confidence=np.array([0.9, 0.6], dtype=np.float32),
            )
        )
        tracks = tracker.update([], frame)

        assert [t.track_id for t in tracks] == [4, 9]
        assert tracks[0].bbox_xyxy == BoundingBox(0.0, 0.0, 10.0, 20.0)
        assert tracks[1].confidence == pytest.approx(0.6)
        assert all(isinstance(t, Track) for t in tracks)

    def test_unconfirmed_detections_are_dropped(self, frame: np.ndarray) -> None:
        """ByteTrack marks unassigned detections with tracker_id -1."""
        tracker, _ = build_tracker(
            FakeTrackedResult(
                xyxy=np.array([[0, 0, 10, 20], [30, 30, 60, 90]], dtype=np.float32),
                tracker_id=np.array([-1, 9], dtype=int),
                confidence=np.array([0.9, 0.6], dtype=np.float32),
            )
        )
        tracks = tracker.update([], frame)
        assert [t.track_id for t in tracks] == [9]

    def test_missing_tracker_ids_yield_no_tracks(self, frame: np.ndarray) -> None:
        tracker, _ = build_tracker(
            FakeTrackedResult(
                xyxy=np.array([[0, 0, 10, 20]], dtype=np.float32),
                tracker_id=None,
            )
        )
        assert tracker.update([], frame) == []

    def test_missing_confidence_becomes_none(self, frame: np.ndarray) -> None:
        tracker, _ = build_tracker(
            FakeTrackedResult(
                xyxy=np.array([[0, 0, 10, 20]], dtype=np.float32),
                tracker_id=np.array([1], dtype=int),
                confidence=None,
            )
        )
        assert tracker.update([], frame)[0].confidence is None

    def test_nan_confidence_becomes_none(self, frame: np.ndarray) -> None:
        tracker, _ = build_tracker(
            FakeTrackedResult(
                xyxy=np.array([[0, 0, 10, 20]], dtype=np.float32),
                tracker_id=np.array([1], dtype=int),
                confidence=np.array([np.nan], dtype=np.float32),
            )
        )
        assert tracker.update([], frame)[0].confidence is None

    def test_malformed_box_is_skipped_not_fatal(self, frame: np.ndarray) -> None:
        tracker, _ = build_tracker(
            FakeTrackedResult(
                xyxy=np.array([[99, 0, 10, 20], [0, 0, 10, 20]], dtype=np.float32),
                tracker_id=np.array([1, 2], dtype=int),
            )
        )
        tracks = tracker.update([], frame)
        assert [t.track_id for t in tracks] == [2]

    def test_empty_result_yields_no_tracks(self, frame: np.ndarray) -> None:
        tracker, _ = build_tracker(
            FakeTrackedResult(xyxy=np.empty((0, 4)), tracker_id=np.empty((0,), dtype=int))
        )
        assert tracker.update([], frame) == []


class TestFailureHandling:
    def test_backend_failure_becomes_a_domain_error(self, frame: np.ndarray) -> None:
        class ExplodingBackend:
            def update(self, detections: Any) -> Any:
                raise RuntimeError("kalman filter diverged")

            def reset(self) -> None:
                pass

        tracker = ByteTrackTracker(TrackerConfig(), backend_factory=lambda _c: ExplodingBackend())
        with pytest.raises(TrackerError, match="ByteTrack update failed"):
            tracker.update([], frame)


class TestRealBackendWiring:
    """The real ByteTrack needs no weights, so the wiring itself is worth checking."""

    def test_default_factory_builds_a_working_tracker(self, frame: np.ndarray) -> None:
        tracker = ByteTrackTracker(TrackerConfig(frame_rate=20.0, minimum_consecutive_frames=1))
        detection = Detection.from_xyxy((10, 10, 40, 80), 0.9, 1)

        tracks: list[Track] = []
        for _ in range(5):
            tracks = tracker.update([detection], frame)

        assert tracks, "a stationary high-confidence person should produce a confirmed track"
        assert all(t.track_id >= 0 for t in tracks)

    def test_track_ids_persist_across_frames(self, frame: np.ndarray) -> None:
        tracker = ByteTrackTracker(TrackerConfig(frame_rate=20.0, minimum_consecutive_frames=1))
        observed: list[int] = []
        for step in range(6):
            x = 10.0 + step  # a slow, smoothly moving person
            tracks = tracker.update([Detection.from_xyxy((x, 10, x + 30, 80), 0.9, 1)], frame)
            observed.extend(t.track_id for t in tracks)

        assert observed, "expected at least one confirmed track"
        assert len(set(observed)) == 1, "the same person must keep one id across frames"

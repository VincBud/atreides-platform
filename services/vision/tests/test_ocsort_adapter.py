"""OC-SORT adapter conversions, exercised with a fake backend.

Mirrors the ByteTrack adapter tests: the two must behave identically at the
interface, otherwise an A/B comparison between them measures the adapters
rather than the trackers.
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
from atreides_vision.tracking.ocsort import OcSortTracker


@dataclass
class FakeTrackedResult:
    """Stands in for the ``sv.Detections`` OC-SORT returns."""

    xyxy: np.ndarray
    tracker_id: np.ndarray | None = None
    confidence: np.ndarray | None = None


@dataclass
class FakeOcSortBackend:
    result: FakeTrackedResult
    received: list[Any] = field(default_factory=list)
    reset_calls: int = 0

    def update(self, detections: Any) -> FakeTrackedResult:
        self.received.append(detections)
        return self.result

    def reset(self) -> None:
        self.reset_calls += 1


def build_tracker(
    result: FakeTrackedResult, config: TrackerConfig | None = None
) -> tuple[OcSortTracker, FakeOcSortBackend]:
    backend = FakeOcSortBackend(result=result)
    tracker = OcSortTracker(config or TrackerConfig(), backend_factory=lambda _config: backend)
    return tracker, backend


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((120, 160, 3), dtype=np.uint8)


class TestInitialization:
    def test_satisfies_the_atreides_protocol(self) -> None:
        tracker, _ = build_tracker(FakeTrackedResult(xyxy=np.empty((0, 4))))
        assert isinstance(tracker, MultiObjectTracker)
        assert tracker.name == "OC-SORT"

    def test_exposes_its_configuration(self) -> None:
        config = TrackerConfig(frame_rate=30.0, delta_t=3)
        tracker, _ = build_tracker(FakeTrackedResult(xyxy=np.empty((0, 4))), config)
        assert tracker.config is config

    def test_backend_is_built_from_the_config(self) -> None:
        received: list[TrackerConfig] = []

        def factory(config: TrackerConfig) -> Any:
            received.append(config)
            return FakeOcSortBackend(result=FakeTrackedResult(xyxy=np.empty((0, 4))))

        config = TrackerConfig(frame_rate=30.0)
        OcSortTracker(config, backend_factory=factory)
        assert received == [config]

    def test_default_factory_passes_every_ocsort_parameter(self) -> None:
        """Guards the A/B setup: a silently dropped parameter would change it.

        Reads the factory source rather than constructing a real tracker, so
        this stays a fast unit test that never imports the ``trackers`` package.
        """
        import inspect
        import re

        from atreides_vision.tracking import ocsort

        source = re.sub(r"\s+", "", inspect.getsource(ocsort._default_backend_factory))
        # OC-SORT constructor argument -> the TrackerConfig field feeding it.
        # IoU is the one that differs: OC-SORT reads its own field so it can be
        # tuned without moving ByteTrack's threshold.
        wiring = {
            "lost_track_buffer": "lost_track_buffer",
            "frame_rate": "frame_rate",
            "minimum_consecutive_frames": "minimum_consecutive_frames",
            "minimum_iou_threshold": "ocsort_minimum_iou_threshold",
            "direction_consistency_weight": "direction_consistency_weight",
            "high_conf_det_threshold": "high_conf_det_threshold",
            "delta_t": "delta_t",
        }
        for argument, config_field in wiring.items():
            assert f"{argument}=config.{config_field}" in source, argument


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

    def test_frame_is_not_forwarded_to_ocsort(self, frame: np.ndarray) -> None:
        """OC-SORT does no motion compensation and warns if handed a frame."""

        class StrictBackend:
            def update(self, detections: Any) -> FakeTrackedResult:
                return FakeTrackedResult(xyxy=np.empty((0, 4)))

            def reset(self) -> None:
                pass

        tracker = OcSortTracker(TrackerConfig(), backend_factory=lambda _c: StrictBackend())
        # Would raise TypeError if the adapter passed frame= through.
        assert tracker.update([], frame) == []

    def test_no_pixels_reach_the_backend(self, frame: np.ndarray) -> None:
        """No appearance or ReID features: identity comes from geometry alone."""
        tracker, backend = build_tracker(FakeTrackedResult(xyxy=np.empty((0, 4))))
        tracker.update([Detection.from_xyxy((0, 0, 10, 20), 0.9, 1)], frame)

        passed = backend.received[0]
        assert passed.mask is None
        assert not passed.data, "no per-detection payload should be attached"


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

    def test_unassigned_ids_are_dropped(self, frame: np.ndarray) -> None:
        """OC-SORT marks unmatched or immature tracks with tracker_id -1."""
        tracker, _ = build_tracker(
            FakeTrackedResult(
                xyxy=np.array([[0, 0, 10, 20], [30, 30, 60, 90]], dtype=np.float32),
                tracker_id=np.array([-1, 9], dtype=int),
                confidence=np.array([0.9, 0.6], dtype=np.float32),
            )
        )
        assert [t.track_id for t in tracker.update([], frame)] == [9]

    def test_all_unassigned_yields_no_tracks(self, frame: np.ndarray) -> None:
        tracker, _ = build_tracker(
            FakeTrackedResult(
                xyxy=np.array([[0, 0, 10, 20], [1, 1, 11, 21]], dtype=np.float32),
                tracker_id=np.array([-1, -1], dtype=int),
            )
        )
        assert tracker.update([], frame) == []

    def test_missing_tracker_ids_yield_no_tracks(self, frame: np.ndarray) -> None:
        tracker, _ = build_tracker(
            FakeTrackedResult(xyxy=np.array([[0, 0, 10, 20]], dtype=np.float32), tracker_id=None)
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

    def test_malformed_box_is_skipped_not_fatal(self, frame: np.ndarray) -> None:
        tracker, _ = build_tracker(
            FakeTrackedResult(
                xyxy=np.array([[99, 0, 10, 20], [0, 0, 10, 20]], dtype=np.float32),
                tracker_id=np.array([1, 2], dtype=int),
            )
        )
        assert [t.track_id for t in tracker.update([], frame)] == [2]

    def test_empty_result_yields_no_tracks(self, frame: np.ndarray) -> None:
        tracker, _ = build_tracker(
            FakeTrackedResult(xyxy=np.empty((0, 4)), tracker_id=np.empty((0,), dtype=int))
        )
        assert tracker.update([], frame) == []


class TestReset:
    def test_reset_delegates_to_the_backend(self) -> None:
        tracker, backend = build_tracker(FakeTrackedResult(xyxy=np.empty((0, 4))))
        tracker.reset()
        assert backend.reset_calls == 1

    def test_reset_is_repeatable(self) -> None:
        tracker, backend = build_tracker(FakeTrackedResult(xyxy=np.empty((0, 4))))
        tracker.reset()
        tracker.reset()
        assert backend.reset_calls == 2


class TestFailureHandling:
    def test_backend_failure_becomes_a_domain_error(self, frame: np.ndarray) -> None:
        class ExplodingBackend:
            def update(self, detections: Any) -> Any:
                raise RuntimeError("kalman filter diverged")

            def reset(self) -> None:
                pass

        tracker = OcSortTracker(TrackerConfig(), backend_factory=lambda _c: ExplodingBackend())
        with pytest.raises(TrackerError, match="OC-SORT update failed"):
            tracker.update([], frame)

    def test_initialisation_failure_becomes_a_domain_error(self) -> None:
        def exploding_factory(config: TrackerConfig) -> Any:
            raise TrackerError("Failed to initialise OC-SORT: bad parameter")

        with pytest.raises(TrackerError, match="Failed to initialise OC-SORT"):
            OcSortTracker(TrackerConfig(), backend_factory=exploding_factory)


class TestRealBackendWiring:
    """The real OC-SORT needs no weights, so the wiring itself is worth checking."""

    def test_default_factory_builds_a_working_tracker(self, frame: np.ndarray) -> None:
        tracker = OcSortTracker(TrackerConfig(frame_rate=30.0, minimum_consecutive_frames=1))
        detection = Detection.from_xyxy((10, 10, 40, 80), 0.9, 1)

        tracks: list[Track] = []
        for _ in range(5):
            tracks = tracker.update([detection], frame)

        assert tracks, "a stationary high-confidence person should produce a confirmed track"
        assert all(t.track_id >= 0 for t in tracks)

    def test_track_ids_persist_across_frames(self, frame: np.ndarray) -> None:
        tracker = OcSortTracker(TrackerConfig(frame_rate=30.0, minimum_consecutive_frames=1))
        observed: list[int] = []
        for step in range(6):
            x = 10.0 + step  # a slow, smoothly moving person
            tracks = tracker.update([Detection.from_xyxy((x, 10, x + 30, 80), 0.9, 1)], frame)
            observed.extend(t.track_id for t in tracks)

        assert observed, "expected at least one confirmed track"
        assert len(set(observed)) == 1, "the same person must keep one id across frames"

    def test_reset_clears_real_tracker_state(self, frame: np.ndarray) -> None:
        """A reset must return the tracker to its initial state, ids included."""
        tracker = OcSortTracker(TrackerConfig(frame_rate=30.0, minimum_consecutive_frames=1))
        detection = Detection.from_xyxy((10, 10, 40, 80), 0.9, 1)

        def confirm_a_track() -> int:
            """Feed frames until a track is confirmed, and return its id."""
            for _ in range(10):
                tracks = tracker.update([detection], frame)
                if tracks:
                    return tracks[0].track_id
            raise AssertionError("OC-SORT never confirmed a stationary person")

        first_id = confirm_a_track()
        tracker.reset()
        assert confirm_a_track() == first_id, "id allocation should start over after a reset"

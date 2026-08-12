"""Domain model validation and conversion."""

from __future__ import annotations

import numpy as np
import pytest

from atreides_vision.models import BoundingBox, Detection, Point, Track, detections_to_arrays


class TestBoundingBox:
    def test_geometry(self) -> None:
        box = BoundingBox(10.0, 20.0, 50.0, 120.0)
        assert box.width == 40.0
        assert box.height == 100.0
        assert box.area == 4000.0
        assert box.center == Point(30.0, 70.0)

    def test_from_xyxy_accepts_tuple_and_numpy(self) -> None:
        from_tuple = BoundingBox.from_xyxy((1.0, 2.0, 3.0, 4.0))
        from_array = BoundingBox.from_xyxy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        assert from_tuple == from_array == BoundingBox(1.0, 2.0, 3.0, 4.0)

    def test_from_xyxy_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="Expected 4 xyxy values"):
            BoundingBox.from_xyxy(np.array([1.0, 2.0, 3.0]))

    @pytest.mark.parametrize(
        "coords",
        [(50.0, 20.0, 10.0, 120.0), (10.0, 120.0, 50.0, 20.0)],
    )
    def test_rejects_inverted_box(self, coords: tuple[float, float, float, float]) -> None:
        with pytest.raises(ValueError, match="x1 <= x2"):
            BoundingBox(*coords)

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            BoundingBox(0.0, 0.0, float("nan"), 10.0)

    def test_zero_area_box_is_allowed(self) -> None:
        # A degenerate-but-ordered box is valid; detectors occasionally emit them.
        box = BoundingBox(5.0, 5.0, 5.0, 5.0)
        assert box.area == 0.0

    def test_clipped_to_frame(self) -> None:
        box = BoundingBox(-30.0, -10.0, 700.0, 400.0)
        clipped = box.clipped_to(width=640, height=360)
        assert clipped == BoundingBox(0.0, 0.0, 640.0, 360.0)

    def test_int_tuple_rounds(self) -> None:
        assert BoundingBox(1.4, 2.6, 3.5, 4.49).as_int_tuple() == (1, 3, 4, 4)

    def test_is_immutable(self) -> None:
        box = BoundingBox(0.0, 0.0, 1.0, 1.0)
        with pytest.raises(AttributeError):
            box.x1 = 5.0  # type: ignore[misc]


class TestDetection:
    def test_from_xyxy(self) -> None:
        detection = Detection.from_xyxy((1, 2, 3, 4), confidence=0.75, class_id=1)
        assert detection.bbox_xyxy == BoundingBox(1.0, 2.0, 3.0, 4.0)
        assert detection.confidence == 0.75
        assert detection.class_id == 1
        assert detection.center == Point(2.0, 3.0)

    @pytest.mark.parametrize("confidence", [-0.01, 1.01])
    def test_rejects_confidence_out_of_range(self, confidence: float) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            Detection(bbox_xyxy=BoundingBox(0, 0, 1, 1), confidence=confidence, class_id=1)

    def test_rejects_negative_class_id(self) -> None:
        with pytest.raises(ValueError, match="class_id must be non-negative"):
            Detection(bbox_xyxy=BoundingBox(0, 0, 1, 1), confidence=0.5, class_id=-1)


class TestTrack:
    def test_center_matches_box(self) -> None:
        track = Track(track_id=3, bbox_xyxy=BoundingBox(0.0, 0.0, 10.0, 20.0))
        assert track.center == Point(5.0, 10.0)

    def test_confidence_is_optional(self) -> None:
        assert Track(track_id=1, bbox_xyxy=BoundingBox(0, 0, 1, 1)).confidence is None

    def test_rejects_negative_track_id(self) -> None:
        with pytest.raises(ValueError, match="track_id must be non-negative"):
            Track(track_id=-1, bbox_xyxy=BoundingBox(0, 0, 1, 1))

    def test_rejects_confidence_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            Track(track_id=1, bbox_xyxy=BoundingBox(0, 0, 1, 1), confidence=1.5)


class TestDetectionsToArrays:
    def test_empty_input_has_correct_shapes(self) -> None:
        xyxy, confidence, class_id = detections_to_arrays([])
        assert xyxy.shape == (0, 4)
        assert confidence.shape == (0,)
        assert class_id.shape == (0,)

    def test_roundtrip_preserves_values_and_order(self) -> None:
        detections = [
            Detection.from_xyxy((0, 0, 10, 10), 0.9, 1),
            Detection.from_xyxy((5, 5, 25, 45), 0.6, 1),
        ]
        xyxy, confidence, class_id = detections_to_arrays(detections)
        assert xyxy.shape == (2, 4)
        np.testing.assert_allclose(xyxy[1], [5, 5, 25, 45])
        np.testing.assert_allclose(confidence, [0.9, 0.6], rtol=1e-6)
        np.testing.assert_array_equal(class_id, [1, 1])


class TestPoint:
    def test_as_int_tuple_rounds_half_away_from_zero(self) -> None:
        assert Point(1.4, 2.6).as_int_tuple() == (1, 3)

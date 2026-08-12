"""RF-DETR adapter conversion logic, exercised with a fake model.

No weights are downloaded and torch is never touched: the adapter takes a model
factory precisely so this layer can be tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pytest

from atreides_vision.config import DetectorConfig
from atreides_vision.detection.base import PersonDetector
from atreides_vision.detection.rfdetr import RFDetrPersonDetector
from atreides_vision.errors import DetectorError
from atreides_vision.models import Detection

PERSON = 1
CAR = 3


@dataclass
class FakeSvDetections:
    """Stands in for ``supervision.Detections`` with only what the adapter reads."""

    xyxy: np.ndarray
    confidence: np.ndarray | None = None
    class_id: np.ndarray | None = None


@dataclass
class FakeRFDetrModel:
    """Records how it was called and replays a canned result."""

    result: FakeSvDetections
    calls: list[dict[str, Any]] = field(default_factory=list)

    def predict(self, images: Any, threshold: float, **kwargs: Any) -> FakeSvDetections:
        self.calls.append({"images": images, "threshold": threshold, **kwargs})
        return self.result


def build_detector(
    result: FakeSvDetections, config: DetectorConfig | None = None
) -> tuple[RFDetrPersonDetector, FakeRFDetrModel]:
    model = FakeRFDetrModel(result=result)
    detector = RFDetrPersonDetector(
        config or DetectorConfig(confidence_threshold=0.5),
        model_factory=lambda _config: model,
    )
    return detector, model


class TestInterfaceCompliance:
    def test_satisfies_the_atreides_protocol(self) -> None:
        detector, _ = build_detector(FakeSvDetections(xyxy=np.empty((0, 4))))
        assert isinstance(detector, PersonDetector)
        assert detector.name == "RF-DETR Nano"

    def test_construction_is_lazy(self) -> None:
        """Constructing a detector must not build the model (which loads weights)."""
        built: list[int] = []

        def factory(config: DetectorConfig) -> Any:
            built.append(1)
            return FakeRFDetrModel(result=FakeSvDetections(xyxy=np.empty((0, 4))))

        detector = RFDetrPersonDetector(DetectorConfig(), model_factory=factory)
        assert built == []
        detector.load()
        assert built == [1]
        detector.load()
        assert built == [1], "load() must be idempotent"


class TestPersonFiltering:
    def test_keeps_only_the_person_class(self, blank_frame: np.ndarray) -> None:
        detector, _ = build_detector(
            FakeSvDetections(
                xyxy=np.array([[0, 0, 10, 20], [30, 30, 60, 90], [5, 5, 15, 25]], dtype=np.float32),
                confidence=np.array([0.9, 0.95, 0.8], dtype=np.float32),
                class_id=np.array([PERSON, CAR, PERSON], dtype=int),
            )
        )
        detections = detector.detect(blank_frame)

        assert len(detections) == 2
        assert {d.class_id for d in detections} == {PERSON}
        assert detections[0].bbox_xyxy.as_tuple() == (0.0, 0.0, 10.0, 20.0)
        assert detections[1].bbox_xyxy.as_tuple() == (5.0, 5.0, 15.0, 25.0)

    def test_returns_empty_list_when_nothing_is_detected(self, blank_frame: np.ndarray) -> None:
        detector, _ = build_detector(FakeSvDetections(xyxy=np.empty((0, 4), dtype=np.float32)))
        assert detector.detect(blank_frame) == []

    def test_returns_empty_list_when_only_non_people_are_detected(
        self, blank_frame: np.ndarray
    ) -> None:
        detector, _ = build_detector(
            FakeSvDetections(
                xyxy=np.array([[0, 0, 10, 20]], dtype=np.float32),
                confidence=np.array([0.99], dtype=np.float32),
                class_id=np.array([CAR], dtype=int),
            )
        )
        assert detector.detect(blank_frame) == []

    def test_person_class_id_is_configurable(self, blank_frame: np.ndarray) -> None:
        """A fine-tuned checkpoint may use a 0-based label space instead of COCO ids."""
        detector, _ = build_detector(
            FakeSvDetections(
                xyxy=np.array([[0, 0, 10, 20]], dtype=np.float32),
                confidence=np.array([0.9], dtype=np.float32),
                class_id=np.array([0], dtype=int),
            ),
            config=DetectorConfig(person_class_id=0),
        )
        assert len(detector.detect(blank_frame)) == 1


class TestConfidenceHandling:
    def test_threshold_is_passed_to_the_model(self, blank_frame: np.ndarray) -> None:
        detector, model = build_detector(
            FakeSvDetections(xyxy=np.empty((0, 4))),
            config=DetectorConfig(confidence_threshold=0.72),
        )
        detector.detect(blank_frame)
        assert model.calls[0]["threshold"] == pytest.approx(0.72)

    def test_threshold_is_re_applied_defensively(self, blank_frame: np.ndarray) -> None:
        """A backend that ignores the threshold must not leak weak detections."""
        detector, _ = build_detector(
            FakeSvDetections(
                xyxy=np.array([[0, 0, 10, 20], [1, 1, 11, 21]], dtype=np.float32),
                confidence=np.array([0.8, 0.2], dtype=np.float32),
                class_id=np.array([PERSON, PERSON], dtype=int),
            ),
            config=DetectorConfig(confidence_threshold=0.5),
        )
        detections = detector.detect(blank_frame)
        assert len(detections) == 1
        assert detections[0].confidence == pytest.approx(0.8)

    def test_missing_confidence_defaults_to_full_certainty(self, blank_frame: np.ndarray) -> None:
        detector, _ = build_detector(
            FakeSvDetections(
                xyxy=np.array([[0, 0, 10, 20]], dtype=np.float32),
                confidence=None,
                class_id=np.array([PERSON], dtype=int),
            )
        )
        detections = detector.detect(blank_frame)
        assert detections[0].confidence == 1.0

    def test_out_of_range_confidence_is_clamped(self, blank_frame: np.ndarray) -> None:
        detector, _ = build_detector(
            FakeSvDetections(
                xyxy=np.array([[0, 0, 10, 20]], dtype=np.float32),
                confidence=np.array([1.0000004], dtype=np.float32),
                class_id=np.array([PERSON], dtype=int),
            )
        )
        assert detector.detect(blank_frame)[0].confidence == 1.0


class TestFrameHandling:
    def test_frame_is_converted_from_bgr_to_rgb(self) -> None:
        """OpenCV decodes BGR; RF-DETR expects RGB. The adapter owns that conversion."""
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # pure blue in BGR

        detector, model = build_detector(FakeSvDetections(xyxy=np.empty((0, 4))))
        detector.detect(frame)

        passed = model.calls[0]["images"]
        np.testing.assert_array_equal(passed, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        assert passed[0, 0, 2] == 255, "blue must land in the R... B channel after swap"

    def test_source_image_is_not_attached(self, blank_frame: np.ndarray) -> None:
        detector, model = build_detector(FakeSvDetections(xyxy=np.empty((0, 4))))
        detector.detect(blank_frame)
        assert model.calls[0]["include_source_image"] is False

    @pytest.mark.parametrize("bad", [np.zeros((10, 10), dtype=np.uint8), np.zeros((4, 4, 1))])
    def test_rejects_non_bgr_input(self, bad: np.ndarray) -> None:
        detector, _ = build_detector(FakeSvDetections(xyxy=np.empty((0, 4))))
        with pytest.raises(DetectorError, match=r"Expected an \(H, W, 3\) BGR frame"):
            detector.detect(bad)


class TestFailureHandling:
    def test_inference_failure_becomes_a_domain_error(self, blank_frame: np.ndarray) -> None:
        class ExplodingModel:
            def predict(self, images: Any, threshold: float, **kwargs: Any) -> Any:
                raise RuntimeError("out of memory")

        detector = RFDetrPersonDetector(DetectorConfig(), model_factory=lambda _c: ExplodingModel())
        with pytest.raises(DetectorError, match="RF-DETR inference failed"):
            detector.detect(blank_frame)

    def test_malformed_box_is_skipped_not_fatal(self, blank_frame: np.ndarray) -> None:
        detector, _ = build_detector(
            FakeSvDetections(
                xyxy=np.array([[50, 0, 10, 20], [0, 0, 10, 20]], dtype=np.float32),
                confidence=np.array([0.9, 0.9], dtype=np.float32),
                class_id=np.array([PERSON, PERSON], dtype=int),
            )
        )
        detections = detector.detect(blank_frame)
        assert len(detections) == 1


class TestOutputContract:
    def test_returns_atreides_domain_objects_only(self, blank_frame: np.ndarray) -> None:
        detector, _ = build_detector(
            FakeSvDetections(
                xyxy=np.array([[0, 0, 10, 20]], dtype=np.float32),
                confidence=np.array([0.9], dtype=np.float32),
                class_id=np.array([PERSON], dtype=int),
            )
        )
        detections = detector.detect(blank_frame)
        assert all(isinstance(d, Detection) for d in detections)
        assert all(not hasattr(d, "xyxy") for d in detections), "no framework object leaked through"

"""RF-DETR Nano adapter — the initial :class:`PersonDetector` implementation.

This module is the **only** place in Atreides allowed to import ``rfdetr``.
Everything RF-DETR-specific lives here: weight loading, BGR->RGB conversion,
the COCO label space, and the ``supervision.Detections`` return type. The rest
of the platform sees Atreides :class:`~atreides_vision.models.Detection` objects.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

import cv2
import numpy as np

from atreides_vision.config import DetectorConfig
from atreides_vision.errors import DetectorError
from atreides_vision.models import BoundingBox, Detection

logger = logging.getLogger(__name__)

__all__ = ["RFDetrModel", "RFDetrPersonDetector"]


class RFDetrModel(Protocol):
    """The slice of the RF-DETR model surface this adapter uses.

    Declaring it lets tests inject a fake and keeps the dependency honest.
    """

    def predict(self, images: Any, threshold: float, **kwargs: Any) -> Any: ...


def _default_model_factory(config: DetectorConfig) -> RFDetrModel:
    """Construct a real RF-DETR Nano model.

    Imported inside the function so that merely importing this module — which
    the test-suite does — never pulls in torch or touches model weights.
    """
    try:
        from rfdetr import RFDETRNano
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise DetectorError(
            "The 'rfdetr' package is not installed. Run 'uv sync' from the repository root."
        ) from exc

    kwargs: dict[str, Any] = {"device": config.device}
    if config.resolution is not None:
        kwargs["resolution"] = config.resolution

    try:
        model = RFDETRNano(**kwargs)
    except Exception as exc:
        raise DetectorError(
            f"Failed to initialise RF-DETR Nano on device '{config.device}': {exc}. "
            "The first run downloads model weights and needs network access; "
            "afterwards it works offline. Re-run with --device cpu if an "
            "accelerator backend is the problem."
        ) from exc

    if config.optimize_for_inference:
        try:
            model.optimize_for_inference()
        except Exception as exc:
            logger.warning("Could not optimise RF-DETR for inference (%s); using eager model.", exc)

    return model


class RFDetrPersonDetector:
    """Detects people with RF-DETR Nano.

    Args:
        config: Detector settings (confidence threshold, device, person class id).
        model_factory: Builds the underlying model. Overridden in tests to avoid
            loading weights; defaults to a real RF-DETR Nano.
        lazy: When ``True`` (the default) the model is loaded on the first
            :meth:`detect` call, so constructing a detector is cheap. Call
            :meth:`load` to pay that cost up front.
    """

    def __init__(
        self,
        config: DetectorConfig,
        model_factory: Callable[[DetectorConfig], RFDetrModel] = _default_model_factory,
        *,
        lazy: bool = True,
    ) -> None:
        self._config = config
        self._model_factory = model_factory
        self._model: RFDetrModel | None = None
        if not lazy:
            self.load()

    @property
    def name(self) -> str:
        return "RF-DETR Nano"

    @property
    def config(self) -> DetectorConfig:
        return self._config

    def load(self) -> None:
        """Materialise the underlying model if it is not loaded yet."""
        if self._model is None:
            logger.debug("Loading RF-DETR Nano on device '%s'", self._config.device)
            self._model = self._model_factory(self._config)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect people in a BGR frame. See :class:`PersonDetector`."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise DetectorError(
                f"Expected an (H, W, 3) BGR frame, got array with shape {frame.shape}"
            )
        self.load()
        assert self._model is not None

        # OpenCV decodes BGR; RF-DETR expects RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            raw = self._model.predict(
                rgb_frame,
                threshold=self._config.confidence_threshold,
                include_source_image=False,
            )
        except TypeError:
            # Fakes and older RF-DETR builds may not accept include_source_image.
            raw = self._model.predict(rgb_frame, threshold=self._config.confidence_threshold)
        except Exception as exc:
            raise DetectorError(f"RF-DETR inference failed: {exc}") from exc

        return self._to_person_detections(raw)

    def _to_person_detections(self, raw: Any) -> list[Detection]:
        """Convert a ``supervision.Detections`` result into Atreides detections.

        Keeps only the person class and re-applies the confidence threshold
        defensively, so the contract holds even if a backend ignores it.
        """
        xyxy = np.asarray(getattr(raw, "xyxy", np.empty((0, 4))), dtype=np.float64)
        if xyxy.size == 0:
            return []
        xyxy = xyxy.reshape(-1, 4)
        count = xyxy.shape[0]

        confidence = _as_float_array(getattr(raw, "confidence", None), count, default=1.0)
        class_id = _as_int_array(getattr(raw, "class_id", None), count, default=-1)

        threshold = self._config.confidence_threshold
        person_class_id = self._config.person_class_id

        detections: list[Detection] = []
        for index in range(count):
            if int(class_id[index]) != person_class_id:
                continue
            score = float(confidence[index])
            if score < threshold:
                continue
            try:
                box = BoundingBox.from_xyxy(xyxy[index])
            except ValueError:
                logger.debug("Skipping malformed RF-DETR box %s", xyxy[index])
                continue
            detections.append(
                Detection(
                    bbox_xyxy=box,
                    confidence=min(max(score, 0.0), 1.0),
                    class_id=person_class_id,
                )
            )
        return detections


def _as_float_array(values: Any, count: int, *, default: float) -> np.ndarray:
    if values is None:
        return np.full(count, default, dtype=np.float64)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _as_int_array(values: Any, count: int, *, default: int) -> np.ndarray:
    if values is None:
        return np.full(count, default, dtype=np.int64)
    return np.asarray(values, dtype=np.int64).reshape(-1)

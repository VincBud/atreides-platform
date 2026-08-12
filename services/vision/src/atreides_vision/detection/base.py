"""The Atreides person-detector interface.

The pipeline depends on this Protocol, never on a concrete model. Swapping
RF-DETR for another detector means writing a new adapter that satisfies
:class:`PersonDetector` — no pipeline change.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from atreides_vision.models import Detection

__all__ = ["PersonDetector"]


@runtime_checkable
class PersonDetector(Protocol):
    """Detects people in a single video frame.

    Implementations must return **only** people. Class filtering is the
    adapter's responsibility so the pipeline never has to know a model's label
    space.
    """

    @property
    def name(self) -> str:
        """Human-readable detector name, used in startup logs."""
        ...

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect people in ``frame``.

        Args:
            frame: An ``(H, W, 3)`` uint8 image in **BGR** channel order, as
                produced by OpenCV. Adapters convert to whatever their model
                expects.

        Returns:
            Detections of people only, in source-frame pixel coordinates.
        """
        ...

"""Person detection: the Atreides interface and its adapters.

Import :class:`PersonDetector` from here. Concrete adapters
(:mod:`atreides_vision.detection.rfdetr`) are imported explicitly by the
composition root (the CLI) and nowhere else, so no framework import leaks into
the pipeline.
"""

from __future__ import annotations

from atreides_vision.detection.base import PersonDetector

__all__ = ["PersonDetector"]

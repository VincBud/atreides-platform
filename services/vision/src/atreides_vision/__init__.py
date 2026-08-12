"""Atreides Vision — V0.1 milestone: SEE + TRACK.

A local pipeline that turns an MP4 into an annotated MP4 with person boxes,
persistent track ids and short trajectories.

Scope boundaries are part of the design, not an omission. This package does not
identify people, does not perform face or biometric recognition, does not infer
intent or aggression, does not match identities across cameras, does not persist
tracks, and does not send video anywhere. Track ids are ephemeral and local to
one camera session.
"""

from __future__ import annotations

__version__ = "0.1.0"

from atreides_vision.config import (
    DetectorConfig,
    Device,
    PipelineConfig,
    TrackerConfig,
    VisionConfig,
    VisualizationConfig,
)
from atreides_vision.errors import AtreidesVisionError
from atreides_vision.models import BoundingBox, Detection, Point, Track

__all__ = [
    "AtreidesVisionError",
    "BoundingBox",
    "Detection",
    "DetectorConfig",
    "Device",
    "PipelineConfig",
    "Point",
    "Track",
    "TrackerConfig",
    "VisionConfig",
    "VisualizationConfig",
    "__version__",
]

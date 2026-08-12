"""Shared fixtures.

Nothing here loads a neural network or downloads weights. The whole suite runs
offline on CPU. Reusable fake implementations live in :mod:`fakes`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from atreides_vision.models import BoundingBox, Detection, Track


@pytest.fixture
def blank_frame() -> np.ndarray:
    """A 640x360 BGR frame of mid-grey."""
    return np.full((360, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def sample_detection() -> Detection:
    return Detection(bbox_xyxy=BoundingBox(10.0, 20.0, 50.0, 120.0), confidence=0.9, class_id=1)


@pytest.fixture
def sample_track() -> Track:
    return Track(track_id=7, bbox_xyxy=BoundingBox(10.0, 20.0, 50.0, 120.0), confidence=0.9)


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """A short synthetic MP4 written with OpenCV.

    Generated locally on purpose: this milestone never downloads footage.
    """
    path = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 20.0, (160, 120))
    assert writer.isOpened(), "OpenCV could not open an mp4v writer in this environment"
    for index in range(10):
        writer.write(np.full((120, 160, 3), index * 20, dtype=np.uint8))
    writer.release()
    return path

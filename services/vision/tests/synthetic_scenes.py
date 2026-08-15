"""Deterministic synthetic scenes and geometry changes, for integrity tests.

Feature-based registration is only testable against imagery with real, findable
structure. These helpers generate that structure from a **seeded** generator, so
every run sees identical pixels and a verdict never depends on chance.

The pattern is deliberately non-repeating. A checkerboard would look like ideal
test imagery and is in fact the worst case: every corner looks like every other
corner, so descriptor matching becomes ambiguous and the failures would be the
pattern's fault rather than the code's. Scattered shapes of varied size and
colour give locally distinctive neighbourhoods, which is what a real venue —
bar edges, door frames, fixtures — also provides.

The transforms mirror the ways a camera's geometry actually changes: someone
knocks it (translation), it rotates on its mount, the zoom is altered, or the
stream is re-cropped.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "crop_and_rescale",
    "low_texture_scene",
    "moving_foreground",
    "perspective_warp",
    "rotate",
    "scale_about_centre",
    "synthetic_scene",
    "translate",
    "with_noise",
]


def synthetic_scene(width: int = 960, height: int = 540, seed: int = 17) -> np.ndarray:
    """A richly textured BGR scene, identical for a given seed and size."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), 38, dtype=np.uint8)

    shapes = max(80, (width * height) // 4000)
    for _ in range(shapes):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        size = int(rng.integers(max(6, height // 60), max(12, height // 12)))
        colour = tuple(int(value) for value in rng.integers(60, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(image, (x, y), (x + size, y + size), colour, -1)
        else:
            cv2.circle(image, (x, y), max(2, size // 2), colour, -1)
    return image


def low_texture_scene(width: int = 960, height: int = 540) -> np.ndarray:
    """A near-featureless frame: a dark room, or a camera facing a blank wall."""
    return np.full((height, width, 3), 30, dtype=np.uint8)


def translate(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift the view — the camera was knocked sideways."""
    height, width = image.shape[:2]
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(image, matrix, (width, height))


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate about the centre — the camera turned on its mount."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (width, height))


def scale_about_centre(image: np.ndarray, factor: float) -> np.ndarray:
    """Zoom in or out about the centre, keeping the frame size — a zoom change.

    Distinct from a decode resize: the frame stays the same size while the
    *scene* changes scale within it, which is a genuine geometry change.
    """
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 0.0, factor)
    return cv2.warpAffine(image, matrix, (width, height))


def crop_and_rescale(image: np.ndarray, margin_fraction: float = 0.12) -> np.ndarray:
    """Crop the edges away and stretch back to full size — a re-framed stream."""
    height, width = image.shape[:2]
    margin_x = int(width * margin_fraction)
    margin_y = int(height * margin_fraction)
    cropped = image[margin_y : height - margin_y, margin_x : width - margin_x]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def perspective_warp(image: np.ndarray, strength: float = 0.06) -> np.ndarray:
    """Tilt the view — the camera pitched, changing the viewing angle."""
    height, width = image.shape[:2]
    shift = width * strength
    source = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    destination = np.array(
        [[shift, 0], [width - shift, 0], [width, height], [0, height]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def with_noise(image: np.ndarray, amplitude: int = 8, seed: int = 3) -> np.ndarray:
    """Add seeded sensor noise. The camera has not moved; the pixels differ."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(-amplitude, amplitude + 1, image.shape, dtype=np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def moving_foreground(
    image: np.ndarray, count: int = 6, offset: int = 0, seed: int = 5
) -> np.ndarray:
    """Paint opaque blocks over the scene — people standing in front of it.

    ``offset`` slides them, so two frames differ in their foreground while the
    background is untouched. That is the everyday case for venue footage, and it
    must not read as camera movement.
    """
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    result = image.copy()
    block_width = max(20, width // 22)
    block_height = max(50, height // 4)

    for _ in range(count):
        x = int(rng.integers(0, max(1, width - block_width))) + offset
        y = int(rng.integers(height // 3, max(height // 3 + 1, height - block_height)))
        cv2.rectangle(
            result,
            (x, y),
            (min(x + block_width, width - 1), min(y + block_height, height - 1)),
            (25, 25, 30),
            -1,
        )
    return result

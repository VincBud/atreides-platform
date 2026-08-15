"""Geometric registration of a current frame against a reference frame.

The only module in Atreides that touches OpenCV's feature APIs. Everything
above it consumes :class:`~atreides_vision.calibration.integrity.models.RegistrationEvidence`,
a plain value object, which is what lets the threshold and state logic be tested
with hand-written evidence instead of against ORB's behaviour on a given image.

What is being matched
---------------------
The **static scene** — door frames, bar edges, floor markings, fixtures. Not
people. No face, body or person feature is computed anywhere here; the detector
is a general-purpose corner/blob detector that has no concept of a person.
Moving people are simply a minority of the frame that fails to match
consistently, which is exactly how a robust fit is supposed to treat them.

Why ORB
-------
It ships with OpenCV, which is already a dependency, so no new package is
added. It is not a neural model, needs no weights, and runs in milliseconds.
Its rotation and scale invariance is what lets a proportional resize register
cleanly while a real rotation does not.

Determinism
-----------
``findHomography`` with RANSAC draws random samples. The backend seeds
OpenCV's RNG immediately before each call so the same pair of frames always
produces the same evidence. That seeding is process-global state, which this
codebase otherwise avoids — it is accepted here because a non-reproducible
integrity verdict would be worse, and it is confined to this one call site.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from atreides_vision.calibration.integrity.models import RegistrationEvidence
from atreides_vision.errors import CalibrationError

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_HOMOGRAPHY_MATCHES",
    "FrameRegistrationBackend",
    "OrbRegistrationBackend",
    "to_grayscale",
]

#: A homography needs four correspondences. Below this no fit is attempted at
#: all, and the evidence comes back with no transform.
MIN_HOMOGRAPHY_MATCHES = 4


class FrameRegistrationBackend(Protocol):
    """Estimates how a current frame is positioned relative to a reference.

    Implementations must be deterministic: the same frame pair must always
    produce the same evidence, or an integrity verdict becomes a coin toss.
    """

    def register(self, reference: np.ndarray, current: np.ndarray) -> RegistrationEvidence:
        """Compare two frames and report what was found.

        Must not raise for ordinary difficulty — a dark or featureless frame is
        a normal outcome reported as low counts, not an error.
        """
        ...


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR or already-grey frame to single-channel uint8.

    Raises:
        CalibrationError: if the array is not a usable image.
    """
    if frame is None or frame.size == 0:
        raise CalibrationError("Cannot register an empty frame.")
    if frame.ndim == 2:
        grey = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        raise CalibrationError(
            f"Expected a BGR or grayscale image for registration, got shape {frame.shape}"
        )
    return np.ascontiguousarray(grey, dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class OrbRegistrationBackend:
    """Default backend: ORB keypoints, Hamming matching, RANSAC homography.

    Args:
        max_features: Ceiling on keypoints per frame. Bounds cost and memory;
            a venue frame has far fewer distinctive corners than this anyway.
        match_ratio: Lowe ratio for the two-nearest-neighbour test. Lower is
            stricter, discarding ambiguous matches — which repeated venue
            texture such as floor tiles produces plenty of.
        ransac_reprojection_threshold_px: Residual within which a match counts
            as an inlier, in reference pixels.
        rng_seed: Seed applied before RANSAC so results are reproducible.
    """

    max_features: int = 2000
    match_ratio: float = 0.75
    ransac_reprojection_threshold_px: float = 3.0
    rng_seed: int = 0

    def __post_init__(self) -> None:
        if self.max_features < MIN_HOMOGRAPHY_MATCHES:
            raise CalibrationError(f"max_features must be >= {MIN_HOMOGRAPHY_MATCHES}")
        if not 0.0 < self.match_ratio <= 1.0:
            raise CalibrationError(f"match_ratio must be in (0, 1], got {self.match_ratio}")
        if self.ransac_reprojection_threshold_px <= 0:
            raise CalibrationError("ransac_reprojection_threshold_px must be positive")

    def register(self, reference: np.ndarray, current: np.ndarray) -> RegistrationEvidence:
        reference_grey = to_grayscale(reference)
        current_grey = to_grayscale(current)

        detector = cv2.ORB.create(nfeatures=self.max_features)
        reference_keypoints, reference_descriptors = detector.detectAndCompute(reference_grey, None)
        current_keypoints, current_descriptors = detector.detectAndCompute(current_grey, None)

        reference_count = len(reference_keypoints)
        current_count = len(current_keypoints)

        if reference_descriptors is None or current_descriptors is None:
            # A frame with no texture at all. Not an error: the honest report is
            # that there was nothing to match.
            return RegistrationEvidence(
                reference_keypoints=reference_count,
                current_keypoints=current_count,
                matches=0,
                inliers=0,
            )

        matches = self._match(reference_descriptors, current_descriptors)
        if len(matches) < MIN_HOMOGRAPHY_MATCHES:
            return RegistrationEvidence(
                reference_keypoints=reference_count,
                current_keypoints=current_count,
                matches=len(matches),
                inliers=0,
            )

        source = np.array(
            [reference_keypoints[match.queryIdx].pt for match in matches], dtype=np.float32
        ).reshape(-1, 1, 2)
        destination = np.array(
            [current_keypoints[match.trainIdx].pt for match in matches], dtype=np.float32
        ).reshape(-1, 1, 2)

        # Seeded so the same frames always give the same verdict; see the module
        # docstring for why this global is tolerated here.
        cv2.setRNGSeed(self.rng_seed)
        matrix, mask = cv2.findHomography(
            source, destination, cv2.RANSAC, self.ransac_reprojection_threshold_px
        )

        if matrix is None or mask is None:
            return RegistrationEvidence(
                reference_keypoints=reference_count,
                current_keypoints=current_count,
                matches=len(matches),
                inliers=0,
                failure_reason="no homography could be fitted to the matched points",
            )

        inlier_mask = mask.ravel().astype(bool)
        inlier_count = int(inlier_mask.sum())
        error = _mean_inlier_error(matrix, source, destination, inlier_mask)

        return RegistrationEvidence(
            reference_keypoints=reference_count,
            current_keypoints=current_count,
            matches=len(matches),
            inliers=inlier_count,
            mean_reprojection_error_px=error,
            transform=tuple(float(value) for value in np.asarray(matrix).reshape(-1)),
        )

    def _match(
        self, reference_descriptors: np.ndarray, current_descriptors: np.ndarray
    ) -> list[cv2.DMatch]:
        """Ratio-tested nearest-neighbour matches, ordered deterministically."""
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        if len(current_descriptors) < 2:
            return []

        candidates = matcher.knnMatch(reference_descriptors, current_descriptors, k=2)
        accepted = [
            pair[0]
            for pair in candidates
            if len(pair) == 2 and pair[0].distance < self.match_ratio * pair[1].distance
        ]
        # Sorting makes the point order fed to RANSAC independent of the
        # matcher's internal iteration order.
        accepted.sort(key=lambda match: (match.distance, match.queryIdx, match.trainIdx))
        return accepted


def _mean_inlier_error(
    matrix: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
    inlier_mask: np.ndarray,
) -> float | None:
    """Mean residual of the inliers, in pixels. ``None`` if there are none."""
    if not inlier_mask.any():
        return None
    projected = cv2.perspectiveTransform(source[inlier_mask], matrix)
    residuals = np.linalg.norm(
        projected.reshape(-1, 2) - destination[inlier_mask].reshape(-1, 2), axis=1
    )
    return float(residuals.mean())

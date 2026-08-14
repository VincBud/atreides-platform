"""Turning a calibration's correspondences into a usable transform.

Separated from both the maths (:mod:`~atreides_vision.calibration.homography`,
which only knows arrays) and the data (:mod:`~atreides_vision.calibration.models`,
which only knows points). This module is where the two meet: solve for the
homography, measure how well it reproduces the points it came from, and refuse
to hand back a fit that is grossly wrong.

Fitting is deliberately eager and done once. A :class:`CalibrationFit` is
computed when a calibration is loaded, not lazily on first use, so a broken
calibration fails at startup rather than a thousand frames into a run.
"""

from __future__ import annotations

import logging

import numpy as np

from atreides_vision.calibration.homography import (
    apply_homography,
    estimate_homography,
    invert_homography,
)
from atreides_vision.calibration.models import CalibrationQuality, GroundPlaneCalibration
from atreides_vision.errors import CalibrationError

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_MAX_RELATIVE_REPROJECTION_ERROR", "CalibrationFit", "fit_calibration"]

#: Development safeguard, **not** an accuracy standard. A fit whose mean
#: reprojection error exceeds this fraction of the calibrated region's own
#: diagonal has not converged on anything resembling the supplied points —
#: typically transposed coordinates, mixed-up axes, or metres entered where
#: pixels belong. It is expressed as a ratio precisely so that it encodes no
#: venue size: 10% of a 3 m doorway and 10% of a 60 m field are both "this is
#: not a working calibration".
#:
#: It is not a quality bar. A calibration well inside this limit may still be
#: far too coarse for the use it is put to, and nothing here can tell you that.
DEFAULT_MAX_RELATIVE_REPROJECTION_ERROR = 0.10


class CalibrationFit:
    """A solved homography for one camera, with its quality measurement.

    Holds both directions: image to venue for projecting people, and its
    inverse for measuring pixel-space reprojection error and for drawing venue
    geometry back onto a frame later.

    Not a dataclass because it owns NumPy arrays, for which structural equality
    is ambiguous. Compare :attr:`quality` or the underlying calibration
    instead.
    """

    def __init__(
        self,
        calibration: GroundPlaneCalibration,
        matrix: np.ndarray,
        quality: CalibrationQuality,
    ) -> None:
        self._calibration = calibration
        self._matrix = matrix
        self._quality = quality
        self._inverse: np.ndarray | None = None

    @property
    def calibration(self) -> GroundPlaneCalibration:
        return self._calibration

    @property
    def camera_id(self) -> str:
        return self._calibration.camera_id

    @property
    def venue_id(self) -> str:
        return self._calibration.venue_id

    @property
    def quality(self) -> CalibrationQuality:
        return self._quality

    @property
    def matrix(self) -> np.ndarray:
        """The image-to-venue homography, in reference-frame pixels.

        Returns a copy: the fit is immutable from the caller's point of view,
        and handing out the live array would let a caller silently corrupt
        every subsequent projection.
        """
        return self._matrix.copy()

    @property
    def inverse_matrix(self) -> np.ndarray:
        """The venue-to-image homography. Computed once, on first use."""
        if self._inverse is None:
            self._inverse = invert_homography(self._matrix)
        return self._inverse.copy()

    def as_flat_tuple(self) -> tuple[float, ...]:
        """The nine matrix entries, row-major.

        The form :func:`~atreides_vision.calibration.homography.project_point`
        consumes for its per-point scalar path.
        """
        return tuple(float(value) for value in self._matrix.reshape(-1))

    def project_venue_points(self, image_points: np.ndarray) -> np.ndarray:
        """Project ``(N, 2)`` reference-frame pixels onto the venue plane."""
        return apply_homography(self._matrix, image_points)

    def project_image_points(self, venue_points: np.ndarray) -> np.ndarray:
        """Project ``(N, 2)`` venue metres back into reference-frame pixels."""
        return apply_homography(self.inverse_matrix, venue_points)


def fit_calibration(
    calibration: GroundPlaneCalibration,
    max_relative_error: float = DEFAULT_MAX_RELATIVE_REPROJECTION_ERROR,
) -> CalibrationFit:
    """Solve and validate the homography for one camera.

    With exactly four correspondences the solution is exact. With more, every
    point contributes to a least-squares fit rather than being discarded, so
    additional reference points make the result better instead of being wasted.

    Args:
        calibration: The validated correspondences to solve.
        max_relative_error: Development safeguard, as a fraction of the
            calibrated region's diagonal. See
            :data:`DEFAULT_MAX_RELATIVE_REPROJECTION_ERROR`.

    Returns:
        The solved transform and its quality measurement.

    Raises:
        CalibrationError: if the points are degenerate, the transform is
            singular, or the fit does not reproduce its own points to within
            ``max_relative_error``.
    """
    if max_relative_error <= 0:
        raise CalibrationError(f"max_relative_error must be positive, got {max_relative_error}")

    image_points = calibration.image_points()
    venue_points = calibration.venue_points()

    matrix = estimate_homography(image_points, venue_points)
    # Fails loudly here rather than at the first frame that needs the inverse.
    invert_homography(matrix)

    quality = _measure_quality(calibration, matrix)

    if quality.relative_error > max_relative_error:
        raise CalibrationError(
            f"Camera '{calibration.camera_id}': the fitted homography does not reproduce its "
            f"own reference points. Mean reprojection error is "
            f"{quality.mean_venue_error_m:.3f} m across a calibrated region "
            f"{quality.venue_extent_m:.2f} m across "
            f"({quality.relative_error:.1%}, limit {max_relative_error:.1%}). "
            "Check that image and venue points are listed in the same order, that venue "
            "coordinates are metres in one consistent frame, and that every point lies on "
            "the same flat floor."
        )

    if calibration.is_overdetermined:
        logger.debug(
            "Camera '%s': fitted %d correspondences, mean error %.4f m (%.2f%% of a %.2f m region)",
            calibration.camera_id,
            quality.point_count,
            quality.mean_venue_error_m,
            quality.relative_error * 100.0,
            quality.venue_extent_m,
        )

    return CalibrationFit(calibration=calibration, matrix=matrix, quality=quality)


def _measure_quality(calibration: GroundPlaneCalibration, matrix: np.ndarray) -> CalibrationQuality:
    """Reprojection error in both directions, from the calibration's own points."""
    image_points = calibration.image_points()
    venue_points = calibration.venue_points()

    projected_venue = apply_homography(matrix, image_points)
    venue_errors = np.linalg.norm(projected_venue - venue_points, axis=1)

    projected_image = apply_homography(invert_homography(matrix), venue_points)
    image_errors = np.linalg.norm(projected_image - image_points, axis=1)

    return CalibrationQuality(
        point_count=calibration.point_count,
        mean_venue_error_m=float(venue_errors.mean()),
        max_venue_error_m=float(venue_errors.max()),
        mean_image_error_px=float(image_errors.mean()),
        max_image_error_px=float(image_errors.max()),
        venue_extent_m=calibration.venue_extent_metres(),
    )

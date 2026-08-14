"""Planar homography estimation and application — pure array maths.

This module knows nothing about cameras, venues or Atreides domain types. It
takes arrays of source and target points and returns a 3x3 matrix, which makes
every property below testable against configurations with known analytic
answers.

Why the maths is owned here
---------------------------
OpenCV is already a project dependency and ``cv2.findHomography`` would work.
It is not used, for the same reason point-in-polygon is ~20 lines of ray
casting rather than a geometry library: the contract matters more than the
implementation. Owning it means the degeneracy rules, the orientation
convention and the failure modes are ours to define and pin down with tests,
rather than inherited from a third party and discovered later. It also keeps
the whole calibration package free of OpenCV, so nothing in the behaviour path
depends on a native imaging library.

The method is the normalized Direct Linear Transform:

1. **Hartley normalization.** Both point sets are translated to their centroid
   and scaled so the mean distance from the origin is ``sqrt(2)``. Skipping
   this is the classic way to get a numerically poor homography: pixel
   coordinates around 1000 and metre coordinates around 10 differ by two orders
   of magnitude, and the resulting matrix is badly conditioned.
2. **Linear system.** Each correspondence contributes two rows to a ``2N x 9``
   matrix.
3. **SVD.** The solution is the right singular vector for the smallest singular
   value. With exactly four correspondences this is the exact solution; with
   more it is the least-squares best fit over all of them, which is why extra
   points improve the result instead of being discarded.
4. **Denormalization**, then scaling to the canonical ``H[2][2] == 1`` form.

Orientation convention
----------------------
``H`` and ``-H`` describe the same projective transform. :func:`estimate_homography`
resolves the ambiguity by orienting the result so the calibration's own source
points have a positive homogeneous ``w``. That turns the sign of ``w`` into
usable information: after estimation, ``w <= 0`` means a point lies at or
beyond the horizon relative to the calibrated surface, which is a real
condition worth reporting rather than a numerical artefact.
"""

from __future__ import annotations

import math

import numpy as np

from atreides_vision.errors import CalibrationError, ProjectionError

__all__ = [
    "MIN_CORRESPONDENCES",
    "apply_homography",
    "estimate_homography",
    "invert_homography",
    "project_point",
]

#: Four correspondences are the mathematical minimum for a planar projective
#: transform: eight degrees of freedom, two equations per point.
MIN_CORRESPONDENCES = 4

#: Relative floor for the eighth singular value. Below this the linear system
#: has a null space of more than one dimension, so the homography is not
#: uniquely determined — the signature of a degenerate point layout.
_RANK_TOLERANCE = 1e-8

#: Smallest homogeneous divisor treated as usable. Below it the projected point
#: is at or beyond the horizon and has no finite ground position.
_MIN_HOMOGENEOUS_W = 1e-9

#: Floor for the determinant of a normalized homography, relative to its scale.
_MIN_DETERMINANT = 1e-12

_COINCIDENT_TOLERANCE = 1e-12


def estimate_homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Best-fit homography mapping ``source`` points onto ``target`` points.

    Args:
        source: ``(N, 2)`` source coordinates, ``N >= 4``.
        target: ``(N, 2)`` target coordinates, same length as ``source``.

    Returns:
        A ``(3, 3)`` matrix ``H`` with ``H[2][2] == 1`` where possible, oriented
        so the source points project with positive homogeneous ``w``.

    Raises:
        CalibrationError: if the inputs are the wrong shape or length, contain
            non-finite values, or describe a degenerate configuration from
            which no unique homography can be recovered.
    """
    src = _as_point_array(source, "source")
    dst = _as_point_array(target, "target")

    if len(src) != len(dst):
        raise CalibrationError(
            f"Source and target must have the same number of points, got {len(src)} and {len(dst)}"
        )
    if len(src) < MIN_CORRESPONDENCES:
        raise CalibrationError(
            f"A planar homography needs at least {MIN_CORRESPONDENCES} point "
            f"correspondences, got {len(src)}"
        )

    src_transform = _normalization_matrix(src, "source")
    dst_transform = _normalization_matrix(dst, "target")
    src_n = _transform(src, src_transform)
    dst_n = _transform(dst, dst_transform)

    rows = []
    for (x, y), (u, v) in zip(src_n, dst_n, strict=True):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    design = np.asarray(rows, dtype=np.float64)

    _, singular_values, right_vectors = np.linalg.svd(design)

    # Index 7 is the eighth singular value in both shapes this can take: with
    # four points the design matrix is 8x9 and s has eight entries, so this is
    # the smallest; with more points s has nine and this is the second
    # smallest. Either way it is the one that must stay clear of zero for the
    # null space — and therefore the homography — to be unique.
    if singular_values[7] <= _RANK_TOLERANCE * singular_values[0]:
        raise CalibrationError(
            "The calibration points are geometrically degenerate: no unique homography "
            "exists for this layout. The usual cause is three or more points lying on a "
            "straight line, or points that are nearly coincident. Choose reference points "
            "spread across the floor area, not along one edge."
        )

    matrix = right_vectors[-1].reshape(3, 3)
    matrix = _denormalize(matrix, src_transform, dst_transform)
    matrix = _canonicalize(matrix)
    return _orient_towards(matrix, src)


def apply_homography(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Project ``(N, 2)`` points through ``matrix``.

    Raises:
        ProjectionError: if any point has a homogeneous divisor too close to
            zero, or projects to a non-finite coordinate.
    """
    pts = _as_point_array(points, "points")
    homogeneous = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    projected = homogeneous @ np.asarray(matrix, dtype=np.float64).T

    w = projected[:, 2]
    if np.any(np.abs(w) < _MIN_HOMOGENEOUS_W):
        raise ProjectionError(
            "At least one point lies on the horizon line of this calibration and has no "
            "finite ground-plane position."
        )

    result = projected[:, :2] / w[:, np.newaxis]
    if not np.all(np.isfinite(result)):
        raise ProjectionError("Projection produced a non-finite ground-plane coordinate.")
    return np.asarray(result, dtype=np.float64)


def project_point(matrix: tuple[float, ...], x: float, y: float) -> tuple[float, float]:
    """Project a single point through a flattened 3x3 ``matrix``.

    A scalar path deliberately free of NumPy: this runs once per tracked person
    per frame, and the array machinery costs more than the nine multiplications
    it would wrap.

    Args:
        matrix: The nine matrix entries in row-major order.
        x: Source x coordinate.
        y: Source y coordinate.

    Raises:
        ProjectionError: if the point is at or beyond the horizon of the
            calibrated plane, or projects to a non-finite coordinate.
    """
    m0, m1, m2, m3, m4, m5, m6, m7, m8 = matrix

    w = m6 * x + m7 * y + m8
    if w <= _MIN_HOMOGENEOUS_W:
        # Not merely small: estimate_homography orients the matrix so that the
        # calibrated surface has positive w, which makes a non-positive value
        # here mean "at or past the horizon", not "numerically awkward".
        raise ProjectionError(
            f"Image point ({x:.1f}, {y:.1f}) is at or beyond the horizon of this "
            "calibration and has no ground-plane position. This usually means the "
            "anchor sits above the calibrated floor area in the frame."
        )

    projected_x = (m0 * x + m1 * y + m2) / w
    projected_y = (m3 * x + m4 * y + m5) / w
    if not (math.isfinite(projected_x) and math.isfinite(projected_y)):
        raise ProjectionError(
            f"Image point ({x:.1f}, {y:.1f}) projected to a non-finite ground position."
        )
    return (projected_x, projected_y)


def invert_homography(matrix: np.ndarray) -> np.ndarray:
    """Invert a homography, for projecting venue coordinates back into pixels.

    Raises:
        CalibrationError: if the matrix is singular.
    """
    array = np.asarray(matrix, dtype=np.float64)
    determinant = float(np.linalg.det(array))
    scale = float(np.abs(array).max())
    if scale <= 0 or abs(determinant) < _MIN_DETERMINANT * scale**3:
        raise CalibrationError(
            "The calibration produced a singular transform that cannot be inverted. "
            "The reference points are almost certainly degenerate."
        )
    try:
        return np.linalg.inv(array)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded by the check above
        raise CalibrationError(f"Could not invert the calibration transform: {exc}") from exc


def _as_point_array(points: np.ndarray, name: str) -> np.ndarray:
    """Coerce to a validated ``(N, 2)`` float array."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise CalibrationError(f"{name} points must have shape (N, 2), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise CalibrationError(f"{name} points must all be finite")
    return array


def _normalization_matrix(points: np.ndarray, name: str) -> np.ndarray:
    """Hartley normalization: centroid to the origin, mean distance ``sqrt(2)``."""
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_distance = float(np.sqrt((shifted**2).sum(axis=1)).mean())
    if mean_distance <= _COINCIDENT_TOLERANCE:
        raise CalibrationError(
            f"All {name} calibration points are at the same position; "
            "they must describe an area, not a point."
        )
    scale = math.sqrt(2.0) / mean_distance
    return np.array(
        [
            [scale, 0.0, -scale * float(centroid[0])],
            [0.0, scale, -scale * float(centroid[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply an affine normalization matrix to ``(N, 2)`` points."""
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    return np.asarray((homogeneous @ matrix.T)[:, :2], dtype=np.float64)


def _denormalize(
    matrix: np.ndarray, src_transform: np.ndarray, dst_transform: np.ndarray
) -> np.ndarray:
    """Undo Hartley normalization on both sides of the estimate."""
    return np.asarray(np.linalg.inv(dst_transform) @ matrix @ src_transform, dtype=np.float64)


def _canonicalize(matrix: np.ndarray) -> np.ndarray:
    """Scale to ``H[2][2] == 1`` where that entry is usable.

    An affine calibration — a nadir-mounted camera, for instance — has a
    perfectly good homography whose bottom-right entry is the only non-zero
    term in its row, so this is a normalization, not a requirement.
    """
    scale = float(matrix[2, 2])
    largest = float(np.abs(matrix).max())
    if largest <= 0:
        raise CalibrationError("Estimation produced an all-zero transform.")
    if abs(scale) > _MIN_DETERMINANT * largest:
        return matrix / scale
    return matrix / largest


def _orient_towards(matrix: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Fix the sign so the calibrated points have positive homogeneous ``w``.

    Raises:
        CalibrationError: if the calibration points do not all fall on the same
            side of the horizon, which means the supplied correspondences
            cannot describe one coherent planar surface seen by one camera.
    """
    homogeneous = np.column_stack([source, np.ones(len(source), dtype=np.float64)])
    w = homogeneous @ matrix[2, :]

    if np.all(w > 0):
        return matrix
    if np.all(w < 0):
        return -matrix
    raise CalibrationError(
        "The calibration points do not lie on one side of the transform's horizon line. "
        "This means they cannot describe a single planar surface viewed from one camera "
        "position — check that every reference point is on the same physical floor."
    )

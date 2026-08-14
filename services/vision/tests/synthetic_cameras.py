"""Synthetic cameras with known venue-to-image geometry, for calibration tests.

Every calibration test needs image points whose true venue position is known
exactly. These helpers generate them from explicit venue-to-image homographies
— a *forward* model of a camera — so the tests check that estimation recovers a
transform we independently defined, rather than checking the estimator against
itself.

:func:`project_venue_to_image` is therefore written in plain arithmetic and does
not call anything from :mod:`atreides_vision.calibration`. If the production
maths were wrong, these helpers would not be wrong in the same way.

The matrices carry real perspective: the third row makes distant parts of the
floor compress towards the horizon, exactly as an obliquely mounted camera does.
That is what makes the perspective and multi-camera tests meaningful rather
than affine exercises.

No physical claim is made about any real venue. These are invented geometries
chosen to be well-conditioned and easy to reason about.
"""

from __future__ import annotations

from collections.abc import Sequence

from atreides_vision.calibration.models import (
    GroundPlaneCalibration,
    PointCorrespondence,
    ReferenceFrame,
    VenuePoint,
)
from atreides_vision.models import BoundingBox, Point, Track

__all__ = [
    "CAMERA_A",
    "CAMERA_B",
    "CAMERA_STRONG_PERSPECTIVE",
    "FRAME_A",
    "FRAME_B",
    "IDENTITY",
    "VENUE_ID",
    "Matrix3x3",
    "calibration_for",
    "image_points_for",
    "project_venue_to_image",
    "track_at_venue",
]

#: A 3x3 homography as nested tuples, row-major.
Matrix3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]

#: The shared venue coordinate frame every synthetic camera is calibrated
#: against. Two cameras agreeing on this string is what makes their outputs
#: comparable at all.
VENUE_ID = "synthetic_venue"

#: Camera A: mounted along the venue's -y side, looking across the floor.
#: 80 px per metre at the near edge, compressing with depth.
CAMERA_A: Matrix3x3 = (
    (80.0, 0.0, 960.0),
    (0.0, -70.0, 900.0),
    (0.0, 0.045, 1.0),
)
FRAME_A = ReferenceFrame(1920, 1080)

#: Camera B: a different position, orientation and focal length entirely — the
#: axes are mixed, so venue x and y both contribute to both image axes. Shares
#: only the venue coordinate frame with camera A.
CAMERA_B: Matrix3x3 = (
    (-34.0, 46.0, 900.0),
    (-16.0, -40.0, 620.0),
    (-0.004, 0.030, 1.0),
)
FRAME_B = ReferenceFrame(1280, 720)

#: A deliberately steep oblique view, where the far end of the floor is
#: compressed to a fraction of the near end. Used to show how badly image-space
#: displacement misrepresents physical displacement.
CAMERA_STRONG_PERSPECTIVE: Matrix3x3 = (
    (90.0, 0.0, 960.0),
    (0.0, -80.0, 1000.0),
    (0.0, 0.150, 1.0),
)

#: The trivial transform: image pixels *are* venue metres. Useful for isolating
#: behaviour that should not depend on the transform at all.
IDENTITY: Matrix3x3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def project_venue_to_image(matrix: Matrix3x3, x: float, y: float) -> tuple[float, float]:
    """Project a venue point into image pixels through ``matrix``.

    Plain arithmetic on purpose: this is the test oracle, so it must not share
    an implementation with the code under test.
    """
    (a, b, c), (d, e, f), (g, h, i) = matrix
    w = g * x + h * y + i
    if abs(w) < 1e-12:
        raise ValueError(f"Venue point ({x}, {y}) is on the horizon of this synthetic camera")
    return ((a * x + b * y + c) / w, (d * x + e * y + f) / w)


def image_points_for(
    matrix: Matrix3x3, venue_points: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Project several venue points into image pixels."""
    return [project_venue_to_image(matrix, x, y) for x, y in venue_points]


def calibration_for(
    matrix: Matrix3x3,
    venue_points: Sequence[tuple[float, float]],
    *,
    camera_id: str = "cam_synthetic",
    venue_id: str = VENUE_ID,
    reference_frame: ReferenceFrame | None = None,
    surface_id: str = "ground",
    image_noise: Sequence[tuple[float, float]] | None = None,
) -> GroundPlaneCalibration:
    """Build a calibration whose correspondences are exact by construction.

    Args:
        matrix: The synthetic camera's venue-to-image transform.
        venue_points: Floor positions to use as reference points, in metres.
        camera_id: Identifier for the resulting calibration.
        venue_id: Shared venue frame identifier.
        reference_frame: Image dimensions the pixel values belong to.
        surface_id: Named planar surface.
        image_noise: Optional per-point ``(dx, dy)`` pixel offsets, for
            simulating imperfectly measured reference points.
    """
    image_points = image_points_for(matrix, venue_points)
    if image_noise is not None:
        if len(image_noise) != len(image_points):
            raise ValueError("image_noise must have one offset per venue point")
        image_points = [
            (x + dx, y + dy) for (x, y), (dx, dy) in zip(image_points, image_noise, strict=True)
        ]

    return GroundPlaneCalibration(
        camera_id=camera_id,
        venue_id=venue_id,
        reference_frame=reference_frame or FRAME_A,
        correspondences=tuple(
            PointCorrespondence(image=Point(image_x, image_y), venue=VenuePoint(venue_x, venue_y))
            for (image_x, image_y), (venue_x, venue_y) in zip(
                image_points, venue_points, strict=True
            )
        ),
        surface_id=surface_id,
    )


def track_at_venue(
    matrix: Matrix3x3,
    x: float,
    y: float,
    *,
    track_id: int = 1,
    box_width: float = 40.0,
    box_height: float = 160.0,
    scale: float = 1.0,
) -> Track:
    """A track whose bottom-centre anchor is the image of venue point ``(x, y)``.

    Only the anchor matters to behaviour analysis, so the box dimensions are
    arbitrary; ``scale`` lets a test render the same scene at another
    resolution.
    """
    image_x, image_y = project_venue_to_image(matrix, x, y)
    image_x *= scale
    image_y *= scale
    half_width = box_width * scale / 2.0
    height = box_height * scale
    return Track(
        track_id=track_id,
        bbox_xyxy=BoundingBox(
            image_x - half_width, image_y - height, image_x + half_width, image_y
        ),
        confidence=0.9,
    )

"""Ground-plane calibration — V0.3.1 SPATIAL CALIBRATION.

Converts camera-local image positions into a **venue-local 2D coordinate
system measured in metres**, shared by every camera in a venue.

Why this exists
---------------
V0.3.0 measured behaviour in normalized image coordinates: resolution
independent, but camera-local. A frame fraction is not a distance. The same
person walking the same speed produces different numbers on a doorway camera
and a wide floor shot, and different numbers again near the lens versus at the
back of the room. Every threshold built on such a value silently encodes one
installation.

A calibrated ground plane fixes that at the source. Four or more operator-
supplied correspondences — "this pixel is that spot on the floor" — determine a
planar homography, and positions come out in metres.

The V0.3.0 abstraction is what makes this cheap. ``GroundPlaneProjector``
implements the same
:class:`~atreides_vision.behavior.coordinates.MotionProjector` protocol as
``NormalizedFrameProjector``, so no history, kinematics or signal code changes:
speed becomes ``m/s`` and acceleration ``m/s²`` because units are derived from
the coordinate space, not hard-coded.

Boundaries
----------
This package transforms **positions, not identities**. It contains no
recognition of any kind, and calibration operates purely on geometry.

Two cameras calibrated against the same venue frame will place a person at
approximately the same coordinates. That spatial agreement is a *property of
the coordinate system*, and V0.3.1 draws no conclusion from it: a camera-local
track id remains not a unique venue person, observations from two cameras stay
separate observations, and there is no re-identification, appearance embedding,
biometric identifier, persistent identity or cross-camera association anywhere
in this milestone.

Assumptions
-----------
One homography describes **one plane**. Every reference point, and every person
measured, must be on the same approximately flat floor. Stairs, balconies,
raised platforms, stages and multiple storeys are not supported — see
:mod:`atreides_vision.calibration.models`.
"""

from __future__ import annotations

from atreides_vision.calibration.fitting import (
    DEFAULT_MAX_RELATIVE_REPROJECTION_ERROR,
    CalibrationFit,
    fit_calibration,
)
from atreides_vision.calibration.homography import (
    MIN_CORRESPONDENCES,
    apply_homography,
    estimate_homography,
    invert_homography,
    project_point,
)
from atreides_vision.calibration.loader import load_calibration, load_calibration_fit
from atreides_vision.calibration.models import (
    CalibrationQuality,
    GroundPlaneCalibration,
    PointCorrespondence,
    ReferenceFrame,
    VenuePoint,
)
from atreides_vision.calibration.projector import GroundPlaneProjector

__all__ = [
    "DEFAULT_MAX_RELATIVE_REPROJECTION_ERROR",
    "MIN_CORRESPONDENCES",
    "CalibrationFit",
    "CalibrationQuality",
    "GroundPlaneCalibration",
    "GroundPlaneProjector",
    "PointCorrespondence",
    "ReferenceFrame",
    "VenuePoint",
    "apply_homography",
    "estimate_homography",
    "fit_calibration",
    "invert_homography",
    "load_calibration",
    "load_calibration_fit",
    "project_point",
]

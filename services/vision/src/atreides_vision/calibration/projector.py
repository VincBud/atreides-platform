"""The calibrated projector: image pixels in, venue metres out.

This is the payoff of the V0.3.0 coordinate abstraction. ``GroundPlaneProjector``
implements the same
:class:`~atreides_vision.behavior.coordinates.MotionProjector` protocol as
``NormalizedFrameProjector``, so swapping it in changes what "distance" means
and nothing else. No history, kinematics or signal code is aware of the
difference — speed simply starts arriving in ``m/s`` because
:class:`~atreides_vision.behavior.models.BehaviorSignal` derives its unit from
the coordinate space it was told about.

Direction of dependency
-----------------------
Calibration depends on ``behavior.coordinates`` — a leaf module holding the
shared coordinate vocabulary — and never the other way round. The behaviour
package has no import of this one. That is what keeps behaviour analysis usable
with no calibration at all, and keeps NumPy out of the per-frame behaviour path.

Runtime resolution
------------------
A calibration is authored against one specific frame size, recorded as its
:class:`~atreides_vision.calibration.models.ReferenceFrame`. Runtime frames need
not match: the projector folds the runtime-to-reference scale directly into the
homography at construction, so decoding the same camera at a different
resolution yields the same venue coordinates. See :meth:`GroundPlaneProjector.__init__`.
"""

from __future__ import annotations

import logging
from typing import Self

import numpy as np

from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.calibration.fitting import CalibrationFit, fit_calibration
from atreides_vision.calibration.homography import invert_homography, project_point
from atreides_vision.calibration.models import (
    CalibrationQuality,
    GroundPlaneCalibration,
    ReferenceFrame,
    VenuePoint,
)
from atreides_vision.errors import CalibrationError
from atreides_vision.models import Point

logger = logging.getLogger(__name__)

__all__ = ["ASPECT_RATIO_TOLERANCE", "GroundPlaneProjector"]

#: Relative aspect-ratio difference tolerated between a calibration's reference
#: frame and the runtime frame before a warning is logged. A pure rescale keeps
#: the ratio; a change to it means the runtime frame is cropped, padded or
#: stretched relative to what was calibrated, which no rescale can undo.
ASPECT_RATIO_TOLERANCE = 0.01


class GroundPlaneProjector:
    """Projects a person's image anchor onto the venue floor, in metres.

    Satisfies :class:`~atreides_vision.behavior.coordinates.MotionProjector`,
    emitting :attr:`~atreides_vision.behavior.coordinates.MotionSpace.GROUND_PLANE_METRES`.

    Args:
        fit: A solved calibration. Use :meth:`from_calibration` to build one
            from correspondences.
        frame_width: Runtime frame width. Defaults to the calibration's own
            reference width.
        frame_height: Runtime frame height. Defaults to the reference height.

    Raises:
        CalibrationError: if the runtime frame dimensions are not positive.
    """

    def __init__(
        self,
        fit: CalibrationFit,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> None:
        reference = fit.calibration.reference_frame
        width = reference.width if frame_width is None else frame_width
        height = reference.height if frame_height is None else frame_height

        if width <= 0 or height <= 0:
            raise CalibrationError(f"Runtime frame size must be positive, got {width}x{height}")

        self._fit = fit
        self._runtime_frame = ReferenceFrame(width=width, height=height)
        self._warn_on_aspect_change(reference, self._runtime_frame)

        # Fold the runtime-to-reference rescale into the homography once, rather
        # than rescaling every point on every frame. Composing it here also means
        # there is exactly one matrix to reason about at projection time.
        scale = np.array(
            [
                [reference.width / width, 0.0, 0.0],
                [0.0, reference.height / height, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._effective = fit.matrix @ scale
        self._matrix = tuple(float(value) for value in self._effective.reshape(-1))
        self._inverse: tuple[float, ...] | None = None

    @classmethod
    def from_calibration(
        cls,
        calibration: GroundPlaneCalibration,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> Self:
        """Solve ``calibration`` and build a projector from it.

        Raises:
            CalibrationError: if the calibration cannot be solved. There is
                deliberately no fallback to normalized coordinates — that would
                silently change every downstream signal's unit from metres to
                frame fractions.
        """
        return cls(
            fit=fit_calibration(calibration),
            frame_width=frame_width,
            frame_height=frame_height,
        )

    @property
    def space(self) -> MotionSpace:
        return MotionSpace.GROUND_PLANE_METRES

    @property
    def camera_id(self) -> str:
        return self._fit.camera_id

    @property
    def venue_id(self) -> str:
        """The venue coordinate frame this projector's output is expressed in."""
        return self._fit.venue_id

    @property
    def calibration(self) -> GroundPlaneCalibration:
        return self._fit.calibration

    @property
    def quality(self) -> CalibrationQuality:
        return self._fit.quality

    @property
    def runtime_frame(self) -> ReferenceFrame:
        """The frame size this projector expects to receive points in."""
        return self._runtime_frame

    @property
    def is_rescaling(self) -> bool:
        """Whether runtime frames differ in size from the calibration reference."""
        reference = self._fit.calibration.reference_frame
        return (
            reference.width != self._runtime_frame.width
            or reference.height != self._runtime_frame.height
        )

    def project(self, point: Point) -> MotionPoint:
        """Project one image-space anchor onto the venue floor plane.

        Args:
            point: Pixel position in **runtime** frame coordinates.

        Returns:
            The venue-local position, in metres.

        Raises:
            ProjectionError: if the anchor is at or beyond the calibration's
                horizon line and therefore has no ground position. Callers
                running over many points per frame should catch this and skip
                the point.
        """
        x, y = project_point(self._matrix, point.x, point.y)
        return MotionPoint(x=x, y=y, space=MotionSpace.GROUND_PLANE_METRES)

    def project_to_venue(self, point: Point) -> VenuePoint:
        """The same projection, as a :class:`VenuePoint`.

        For callers working in calibration terms rather than motion terms.
        """
        x, y = project_point(self._matrix, point.x, point.y)
        return VenuePoint(x=x, y=y)

    def project_from_venue(self, venue: VenuePoint) -> Point:
        """Project a venue position back into **runtime** image pixels.

        The inverse direction, needed to draw venue geometry — a metre grid, the
        axes, the calibrated hull — onto a frame for inspection. It inverts the
        *effective* transform, so the result is in runtime pixels and the
        overlay lands correctly on a rescaled frame.

        Scope is deliberately narrow. This exists for calibration diagnostics
        and drawing; behaviour algorithms never see it, and no raw matrix is
        exposed. Measurement runs image-to-venue only.

        Raises:
            CalibrationError: if the transform cannot be inverted.
            ProjectionError: if this venue position has no image position — it
                lies on the image plane's horizon under the inverse transform.
        """
        x, y = project_point(self._inverse_matrix(), venue.x, venue.y)
        return Point(x=x, y=y)

    def _inverse_matrix(self) -> tuple[float, ...]:
        """The venue-to-runtime-pixel transform. Computed once, on first use."""
        if self._inverse is None:
            inverse = invert_homography(self._effective)
            # Orient so venue points inside the calibrated region invert with a
            # positive homogeneous w, matching the forward convention and making
            # "behind the horizon" mean the same thing in both directions.
            venue_points = self._fit.calibration.venue_points()
            homogeneous = np.column_stack(
                [venue_points, np.ones(len(venue_points), dtype=np.float64)]
            )
            if np.all(homogeneous @ inverse[2, :] < 0):
                inverse = -inverse
            self._inverse = tuple(float(value) for value in inverse.reshape(-1))
        return self._inverse

    def shares_venue_with(self, other: GroundPlaneProjector) -> bool:
        """Whether another camera's projector emits into the same venue frame.

        The precondition for comparing two cameras' positions at all. It is a
        statement about coordinate systems, **not** about people: two cameras
        agreeing that somebody is at ``(12.4, 7.1)`` does not make them the same
        person, and V0.3.1 draws no such conclusion.
        """
        return self.calibration.covers_same_venue_as(other.calibration)

    @staticmethod
    def _warn_on_aspect_change(reference: ReferenceFrame, runtime: ReferenceFrame) -> None:
        """Warn when the runtime frame is not a pure rescale of the reference.

        A resolution change preserving aspect ratio is handled exactly. A change
        to the ratio itself means the frame was cropped, padded or stretched, so
        the same pixel no longer shows the same part of the scene — and no
        rescale can recover that. Warned rather than rejected, because a
        deliberately anamorphic encode is legitimate and the operator may know
        it is fine.
        """
        if reference.width == runtime.width and reference.height == runtime.height:
            return

        difference = abs(runtime.aspect_ratio - reference.aspect_ratio)
        if difference > ASPECT_RATIO_TOLERANCE * reference.aspect_ratio:
            logger.warning(
                "Calibration reference frame %s has aspect ratio %.4f but runtime frames are "
                "%s with ratio %.4f. Rescaling assumes the runtime frame shows the same scene; "
                "if it is cropped or letterboxed instead, projected positions will be wrong.",
                reference,
                reference.aspect_ratio,
                runtime,
                runtime.aspect_ratio,
            )
        else:
            logger.debug(
                "Rescaling calibration for camera frames %s -> %s (same aspect ratio).",
                reference,
                runtime,
            )

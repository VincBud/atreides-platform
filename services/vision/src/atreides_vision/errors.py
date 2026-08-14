"""Atreides Vision error hierarchy.

Every failure the operator can plausibly cause (bad path, unreadable video,
unavailable accelerator) surfaces as an :class:`AtreidesVisionError` with a
message written for a human, not a stack trace written for a library author.
The CLI catches this base class and prints the message without a traceback.
"""

from __future__ import annotations


class AtreidesVisionError(Exception):
    """Base class for all Atreides Vision domain errors."""


class ConfigurationError(AtreidesVisionError):
    """Raised when the supplied configuration is internally inconsistent."""


class DeviceUnavailableError(AtreidesVisionError):
    """Raised when a requested compute device cannot be used on this machine."""


class VideoSourceError(AtreidesVisionError):
    """Raised when a video source cannot be opened or read."""


class VideoSinkError(AtreidesVisionError):
    """Raised when a video sink cannot be created or written to."""


class ZoneConfigurationError(AtreidesVisionError):
    """Raised when a zone configuration file is missing, malformed or invalid."""


class MetricsOutputError(AtreidesVisionError):
    """Raised when the zone-metrics output stream cannot be created or written."""


class BehaviorOutputError(AtreidesVisionError):
    """Raised when the behaviour signal stream cannot be created or written."""


class CalibrationError(AtreidesVisionError):
    """Raised when a ground-plane calibration is missing, malformed or unusable.

    Covers configuration problems (too few correspondences, mismatched counts,
    non-finite values) and geometric ones (collinear or duplicate points, a
    singular transform, a fit whose reprojection error is grossly wrong).

    A calibration that cannot be trusted is always an error. Falling back to
    normalized image coordinates would silently change the unit of every
    downstream signal from metres to frame fractions, which is far worse than
    stopping.
    """


class ProjectionError(CalibrationError):
    """Raised when a specific point cannot be projected onto the ground plane.

    Distinct from a bad calibration: the calibration may be perfectly good and
    the *point* still have no ground position — anchors at or beyond the
    horizon line project to infinity or to the mirrored side of it. Callers
    that process many points per frame catch this narrowly and skip the point
    rather than aborting.
    """


class DetectorError(AtreidesVisionError):
    """Raised when the detector cannot be initialised or fails at inference."""


class TrackerError(AtreidesVisionError):
    """Raised when the tracker cannot be initialised or fails during update."""

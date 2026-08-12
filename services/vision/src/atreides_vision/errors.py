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


class DetectorError(AtreidesVisionError):
    """Raised when the detector cannot be initialised or fails at inference."""


class TrackerError(AtreidesVisionError):
    """Raised when the tracker cannot be initialised or fails during update."""

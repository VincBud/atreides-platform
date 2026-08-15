"""Thresholds for calibration-integrity checking.

Every number that turns evidence into a state lives here, with its unit and the
reasoning behind it. No algorithm in this package contains a bare constant.

**These are DEVELOPMENT defaults.** No measurement backs them: they have not
been validated against real venue footage, and no false-positive or
false-negative rate is known. They were chosen to be defensible and to make the
states reachable in tests.

**No threshold encodes a physical size.** Drift is a fraction of the frame
diagonal, never a distance in metres or pixels, so the same configuration
applies to a camera covering a doorway and one covering a festival field. A
venue's dimensions must never appear in a camera-drift threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

from atreides_vision.errors import ConfigurationError

__all__ = ["CalibrationIntegrityConfig", "IntegrityPolicyMode"]

from enum import StrEnum


class IntegrityPolicyMode(StrEnum):
    """What the runtime does when drift is detected."""

    #: Stop emitting metric observations once drift is detected. The default,
    #: because precise-looking metres from a moved camera are worse than no
    #: metres at all.
    GUARDED = "guarded"

    #: Record the state but keep emitting. For diagnosing the monitor itself,
    #: or for a run where the operator knows the drift and wants the data
    #: anyway. Never silent: the stream still carries the integrity records.
    REPORT_ONLY = "report_only"


@dataclass(frozen=True, slots=True)
class CalibrationIntegrityConfig:
    """Evidence requirements, drift limits and check cadence."""

    #: Minimum keypoints the backend must find in *each* frame before the check
    #: is considered to have anything to work with. Below this the scene is too
    #: dark or too flat — a blank wall at night — and the honest answer is
    #: "cannot tell", not "unchanged".
    minimum_keypoints: int = 40

    #: Minimum candidate descriptor matches between the two frames. Four is the
    #: mathematical floor for a homography; this sits well above it so a fit is
    #: not built from the bare minimum.
    minimum_matches: int = 20

    #: Minimum matches that must agree with the fitted transform. The count that
    #: actually decides whether the geometry is pinned down.
    minimum_inliers: int = 12

    #: Minimum share of candidate matches that must be inliers. A low ratio with
    #: a high count means the matcher found lots of coincidences and the fit is
    #: explaining only a fraction of them — an unreliable transform even if the
    #: absolute inlier count looks healthy.
    minimum_inlier_ratio: float = 0.30

    #: Largest mean inlier residual, in reference-frame pixels, for the fit to
    #: count as stable. This one is in pixels because it measures the *fit*, not
    #: the scene geometry; it is compared against the frame the fit was made in.
    maximum_reprojection_error_px: float = 6.0

    #: Largest mean corner displacement, as a fraction of the frame diagonal,
    #: still called consistent. 0.02 is roughly 44 px on a 1080p frame and
    #: roughly 9 px on a 384x288 one — the same *relative* movement on both,
    #: which is the point. Small enough to catch a real nudge, large enough not
    #: to fire on matcher noise.
    maximum_normalized_drift: float = 0.02

    #: The worst single corner may move further than the mean before the view is
    #: called changed, since rotation and keystone shifts load onto one corner.
    maximum_max_normalized_drift: float = 0.035

    #: Seconds of source time between checks. Feature matching is far too
    #: expensive to run every frame, and camera movement is an event that
    #: happens on a scale of seconds, not frames. Time-based rather than
    #: frame-based so the cadence does not change with source frame rate.
    check_interval_seconds: float = 5.0

    #: What happens when drift is found. See :class:`IntegrityPolicyMode`.
    policy_mode: IntegrityPolicyMode = IntegrityPolicyMode.GUARDED

    def __post_init__(self) -> None:
        for name, count in (
            ("minimum_keypoints", self.minimum_keypoints),
            ("minimum_matches", self.minimum_matches),
            ("minimum_inliers", self.minimum_inliers),
        ):
            if count < 1:
                raise ConfigurationError(f"{name} must be >= 1, got {count}")

        if not 0.0 < self.minimum_inlier_ratio <= 1.0:
            raise ConfigurationError(
                f"minimum_inlier_ratio must be in (0, 1], got {self.minimum_inlier_ratio}"
            )
        if self.maximum_reprojection_error_px <= 0:
            raise ConfigurationError(
                "maximum_reprojection_error_px must be positive, got "
                f"{self.maximum_reprojection_error_px}"
            )
        for name, value in (
            ("maximum_normalized_drift", self.maximum_normalized_drift),
            ("maximum_max_normalized_drift", self.maximum_max_normalized_drift),
        ):
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive, got {value}")
        if self.check_interval_seconds <= 0:
            raise ConfigurationError(
                f"check_interval_seconds must be positive, got {self.check_interval_seconds}"
            )
        if self.minimum_inliers > self.minimum_matches:
            raise ConfigurationError(
                f"minimum_inliers ({self.minimum_inliers}) cannot exceed minimum_matches "
                f"({self.minimum_matches}): the requirement would be unsatisfiable."
            )
        if self.maximum_max_normalized_drift < self.maximum_normalized_drift:
            raise ConfigurationError(
                f"maximum_max_normalized_drift ({self.maximum_max_normalized_drift}) must be at "
                f"least maximum_normalized_drift ({self.maximum_normalized_drift}); the worst "
                "corner cannot be held to a tighter limit than the mean."
            )

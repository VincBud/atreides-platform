"""Typed results of a calibration-integrity check.

What this layer answers
-----------------------
**Does the current frame still look like it was taken from the same camera
viewpoint as the reference frame the calibration was authored against?**

That is a question about *image geometry*, and it is emphatically **not** the
question "is this calibration accurate to within X centimetres". The two are
separate and must stay separate:

* A calibration whose reference points were measured badly can be perfectly
  ``CONSISTENT`` — the camera has not moved, it is simply wrong.
* A calibration that was measured perfectly becomes untrustworthy the moment
  the camera is nudged, which is what ``DRIFT_DETECTED`` reports.

Only physical measurement can establish accuracy. This layer can establish that
the *precondition* for accuracy — an unchanged viewpoint — still holds.

Everything here is plain data. No OpenCV type crosses this boundary, which is
what lets the decision logic be tested with injected evidence rather than
against a feature detector's behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "CalibrationIntegrityMetrics",
    "IntegrityAssessment",
    "IntegrityState",
    "RegistrationEvidence",
]


class IntegrityState(StrEnum):
    """The outcome of one integrity check.

    Names describe *what was observed about the image*, never a judgement about
    physical accuracy or fitness for use. There is deliberately no ``ACCURATE``,
    ``SAFE`` or ``PRODUCTION_READY``: nothing here can support such a claim.
    """

    #: No check has run yet — before the first check, or with monitoring off.
    UNCHECKED = "unchecked"

    #: The current frame's geometry matches the reference under the implemented
    #: check. This means the camera viewpoint appears unchanged. It does **not**
    #: mean the calibration's metres have been verified against anything real.
    CONSISTENT = "consistent"

    #: The frame registers against the reference, but the fitted geometry moved
    #: further than the configured limit. The camera view has changed, so metric
    #: output from this calibration can no longer be relied upon.
    DRIFT_DETECTED = "drift_detected"

    #: The check could not reach a conclusion: too little texture, too few
    #: matches, or an unstable fit. **Not** a synonym for ``CONSISTENT`` — "we
    #: could not tell" and "nothing has changed" are different facts, and
    #: collapsing them would be the most dangerous thing this module could do.
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

    @property
    def is_conclusive(self) -> bool:
        """Whether the check reached a finding either way."""
        return self in (IntegrityState.CONSISTENT, IntegrityState.DRIFT_DETECTED)


@dataclass(frozen=True, slots=True)
class RegistrationEvidence:
    """Raw output of a registration backend, before any thresholds apply.

    Deliberately dumb: counts, an error and an optional transform. Whether any
    of it is *good enough* is a policy question answered elsewhere, which is
    what lets thresholds be re-tuned without touching image code — and lets the
    threshold logic be tested with hand-written evidence.
    """

    reference_keypoints: int
    current_keypoints: int
    #: Candidate descriptor matches before geometric verification.
    matches: int
    #: Matches consistent with the fitted transform.
    inliers: int
    #: Mean residual of the inliers, in reference-frame pixels. ``None`` when no
    #: transform could be fitted.
    mean_reprojection_error_px: float | None = None
    #: Row-major 3x3 mapping reference pixels to current pixels, or ``None``.
    #: A tuple rather than an array so this stays a plain value object.
    transform: tuple[float, ...] | None = None
    #: Set when the backend itself failed rather than merely finding too little.
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("reference_keypoints", self.reference_keypoints),
            ("current_keypoints", self.current_keypoints),
            ("matches", self.matches),
            ("inliers", self.inliers),
        ):
            if value < 0:
                raise ValueError(f"RegistrationEvidence.{name} must be non-negative, got {value}")
        if self.inliers > self.matches:
            raise ValueError(
                f"RegistrationEvidence has {self.inliers} inliers among only {self.matches} matches"
            )
        if self.transform is not None and len(self.transform) != 9:
            raise ValueError(
                f"RegistrationEvidence.transform must have 9 entries, got {len(self.transform)}"
            )

    @property
    def inlier_ratio(self) -> float:
        """Inliers as a fraction of candidate matches. ``0.0`` with no matches."""
        return self.inliers / self.matches if self.matches else 0.0


@dataclass(frozen=True, slots=True)
class CalibrationIntegrityMetrics:
    """Neutral diagnostics from one check, resolution-independent where it matters.

    Geometric drift is expressed as a **fraction of the frame diagonal**, never
    in pixels. A 40-pixel shift means something entirely different on a 384x288
    camera than on a 4K one, and Atreides must behave the same on both. It also
    makes a proportional decode resize register as *zero* drift, which is the
    V0.3.1 resolution contract holding at this layer too.
    """

    reference_keypoints: int
    current_keypoints: int
    matches: int
    inliers: int
    inlier_ratio: float
    mean_reprojection_error_px: float | None
    #: Mean corner displacement under the fitted transform, as a fraction of the
    #: frame diagonal. ``None`` when no transform was fitted.
    normalized_drift: float | None
    #: The worst single corner, same units. Catches a rotation or a keystone
    #: change that barely moves the centre but swings the edges.
    max_normalized_drift: float | None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "reference_keypoints": self.reference_keypoints,
            "current_keypoints": self.current_keypoints,
            "matches": self.matches,
            "inliers": self.inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
        }
        if self.mean_reprojection_error_px is not None:
            record["mean_reprojection_error_px"] = round(self.mean_reprojection_error_px, 3)
        if self.normalized_drift is not None:
            record["normalized_drift"] = round(self.normalized_drift, 6)
        if self.max_normalized_drift is not None:
            record["max_normalized_drift"] = round(self.max_normalized_drift, 6)
        return record

    @classmethod
    def from_evidence(
        cls,
        evidence: RegistrationEvidence,
        reference_width: int,
        reference_height: int,
        current_width: int,
        current_height: int,
    ) -> CalibrationIntegrityMetrics:
        """Derive metrics, computing normalized drift from the fitted transform."""
        drift = _corner_drift(
            evidence.transform,
            reference_width,
            reference_height,
            current_width,
            current_height,
        )
        mean_drift, max_drift = drift if drift is not None else (None, None)
        return cls(
            reference_keypoints=evidence.reference_keypoints,
            current_keypoints=evidence.current_keypoints,
            matches=evidence.matches,
            inliers=evidence.inliers,
            inlier_ratio=evidence.inlier_ratio,
            mean_reprojection_error_px=evidence.mean_reprojection_error_px,
            normalized_drift=mean_drift,
            max_normalized_drift=max_drift,
        )


@dataclass(frozen=True, slots=True)
class IntegrityAssessment:
    """One check's state, metrics and the reason behind the state."""

    state: IntegrityState
    frame_index: int
    timestamp_seconds: float
    metrics: CalibrationIntegrityMetrics | None = None
    #: Plain-language explanation of why this state was reached. Written for an
    #: operator reading a log, not for a parser.
    reason: str = ""
    #: Identifies the reference frame the check was made against.
    reference_id: str | None = None

    @property
    def allows_metric_output(self) -> bool:
        """Whether this state alone is evidence *against* trusting the metres.

        Only positively detected drift is. An inconclusive check is not
        permission, but neither is it grounds to discard measurements — the
        stream records the inconclusive state so a consumer can judge.
        """
        return self.state is not IntegrityState.DRIFT_DETECTED

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "state": self.state.value,
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "reason": self.reason,
        }
        if self.reference_id is not None:
            record["reference_id"] = self.reference_id
        if self.metrics is not None:
            record["metrics"] = self.metrics.to_record()
        return record


def _corner_drift(
    transform: tuple[float, ...] | None,
    reference_width: int,
    reference_height: int,
    current_width: int,
    current_height: int,
) -> tuple[float, float] | None:
    """Mean and worst corner displacement, as fractions of the frame diagonal.

    The reference frame's four corners are pushed through the fitted transform
    into current-frame pixels, then **both sides are normalized by their own
    frame size** before being compared. That normalization is what makes a pure
    proportional resize score zero: the corners land at the same relative
    positions even though every pixel coordinate doubled.

    Returns ``None`` if there is no transform or the result is not finite.
    """
    if transform is None:
        return None
    if min(reference_width, reference_height, current_width, current_height) <= 0:
        return None

    corners = (
        (0.0, 0.0),
        (float(reference_width), 0.0),
        (float(reference_width), float(reference_height)),
        (0.0, float(reference_height)),
    )
    m0, m1, m2, m3, m4, m5, m6, m7, m8 = transform

    displacements = []
    for x, y in corners:
        w = m6 * x + m7 * y + m8
        if abs(w) < 1e-12:
            return None
        moved_x = (m0 * x + m1 * y + m2) / w
        moved_y = (m3 * x + m4 * y + m5) / w
        if not (math.isfinite(moved_x) and math.isfinite(moved_y)):
            return None

        # Normalize each side by its own frame, so scale alone contributes zero.
        expected_x, expected_y = x / reference_width, y / reference_height
        actual_x, actual_y = moved_x / current_width, moved_y / current_height
        displacements.append(math.hypot(actual_x - expected_x, actual_y - expected_y))

    # Normalized coordinates span a unit square, whose diagonal is sqrt(2).
    diagonal = math.sqrt(2.0)
    return (
        sum(displacements) / len(displacements) / diagonal,
        max(displacements) / diagonal,
    )

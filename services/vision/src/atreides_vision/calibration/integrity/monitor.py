"""Turning registration evidence into an integrity state, on a schedule.

The monitor **measures**. It decides whether the evidence is strong enough to
conclude anything, and if so whether the geometry moved. It does not decide what
the pipeline should do about that — see
:mod:`atreides_vision.calibration.integrity.policy`.

That separation is the point. Thresholds and states are pure functions of
evidence, so they can be tested with hand-written numbers and no images at all;
the consequences are tested separately with no thresholds involved.

Order of judgement
------------------
Evidence sufficiency is checked **before** drift. A frame that produced twelve
matches can technically yield a homography, but a "the camera moved" verdict
built on twelve matches is a guess. Asking "can I tell?" before "what do I see?"
is what keeps ``INSUFFICIENT_EVIDENCE`` from silently becoming either of the
other answers.
"""

from __future__ import annotations

import logging

import numpy as np

from atreides_vision.calibration.integrity.config import CalibrationIntegrityConfig
from atreides_vision.calibration.integrity.models import (
    CalibrationIntegrityMetrics,
    IntegrityAssessment,
    IntegrityState,
    RegistrationEvidence,
)
from atreides_vision.calibration.integrity.registration import (
    FrameRegistrationBackend,
    OrbRegistrationBackend,
)
from atreides_vision.errors import CalibrationError
from atreides_vision.sampling import IntervalSampler

logger = logging.getLogger(__name__)

__all__ = ["CalibrationIntegrityMonitor", "assess_evidence"]


def assess_evidence(
    evidence: RegistrationEvidence,
    config: CalibrationIntegrityConfig,
    reference_width: int,
    reference_height: int,
    current_width: int,
    current_height: int,
    frame_index: int = 0,
    timestamp_seconds: float = 0.0,
    reference_id: str | None = None,
) -> IntegrityAssessment:
    """Classify one piece of registration evidence.

    A pure function: same evidence and config, same assessment, no images and no
    state involved.
    """
    metrics = CalibrationIntegrityMetrics.from_evidence(
        evidence, reference_width, reference_height, current_width, current_height
    )

    def assessment(state: IntegrityState, reason: str) -> IntegrityAssessment:
        return IntegrityAssessment(
            state=state,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            metrics=metrics,
            reason=reason,
            reference_id=reference_id,
        )

    if evidence.failure_reason is not None:
        return assessment(
            IntegrityState.INSUFFICIENT_EVIDENCE,
            f"registration failed: {evidence.failure_reason}",
        )

    # --- Can we tell anything at all? ---------------------------------------
    weakest_keypoints = min(evidence.reference_keypoints, evidence.current_keypoints)
    if weakest_keypoints < config.minimum_keypoints:
        return assessment(
            IntegrityState.INSUFFICIENT_EVIDENCE,
            f"only {weakest_keypoints} scene features found "
            f"(need {config.minimum_keypoints}); the view may be too dark or too flat to check",
        )
    if evidence.matches < config.minimum_matches:
        return assessment(
            IntegrityState.INSUFFICIENT_EVIDENCE,
            f"only {evidence.matches} candidate matches (need {config.minimum_matches})",
        )
    if evidence.transform is None or metrics.normalized_drift is None:
        return assessment(
            IntegrityState.INSUFFICIENT_EVIDENCE,
            "no usable geometric transform could be fitted between the frames",
        )
    if evidence.inliers < config.minimum_inliers:
        return assessment(
            IntegrityState.INSUFFICIENT_EVIDENCE,
            f"only {evidence.inliers} matches agree with the fitted geometry "
            f"(need {config.minimum_inliers})",
        )
    if evidence.inlier_ratio < config.minimum_inlier_ratio:
        return assessment(
            IntegrityState.INSUFFICIENT_EVIDENCE,
            f"only {evidence.inlier_ratio:.0%} of matches agree with the fitted geometry "
            f"(need {config.minimum_inlier_ratio:.0%}); the fit explains too little to trust",
        )
    if (
        evidence.mean_reprojection_error_px is not None
        and evidence.mean_reprojection_error_px > config.maximum_reprojection_error_px
    ):
        return assessment(
            IntegrityState.INSUFFICIENT_EVIDENCE,
            f"the fit is loose: mean residual {evidence.mean_reprojection_error_px:.1f} px "
            f"exceeds {config.maximum_reprojection_error_px:.1f} px",
        )

    # --- The evidence is good enough. What does it say? ---------------------
    mean_drift = metrics.normalized_drift
    max_drift = metrics.max_normalized_drift or mean_drift

    if mean_drift > config.maximum_normalized_drift:
        return assessment(
            IntegrityState.DRIFT_DETECTED,
            f"the view has moved: mean corner displacement {mean_drift:.2%} of the frame "
            f"diagonal exceeds {config.maximum_normalized_drift:.2%}",
        )
    if max_drift > config.maximum_max_normalized_drift:
        return assessment(
            IntegrityState.DRIFT_DETECTED,
            f"the view has moved: worst corner displacement {max_drift:.2%} of the frame "
            f"diagonal exceeds {config.maximum_max_normalized_drift:.2%}",
        )

    return assessment(
        IntegrityState.CONSISTENT,
        f"geometry matches the reference: mean corner displacement {mean_drift:.2%} of the "
        f"frame diagonal, {evidence.inliers} inliers",
    )


class CalibrationIntegrityMonitor:
    """Holds the reference frame and checks current frames against it on a cadence.

    One monitor per camera. State is per-instance, so two cameras in one venue
    reach their own verdicts independently and one camera being knocked never
    marks another as drifted.

    Args:
        reference_frame: The frame captured from the camera geometry the
            calibration was authored against.
        config: Thresholds and cadence.
        backend: Registration implementation. Injectable so tests can supply a
            deterministic stand-in.
        reference_id: Identifies the reference in output records — normally its
            filename.

    Raises:
        CalibrationError: if the reference frame is not a usable image.
    """

    def __init__(
        self,
        reference_frame: np.ndarray,
        config: CalibrationIntegrityConfig | None = None,
        backend: FrameRegistrationBackend | None = None,
        reference_id: str | None = None,
    ) -> None:
        if reference_frame is None or reference_frame.size == 0 or reference_frame.ndim < 2:
            raise CalibrationError("The calibration integrity reference must be a non-empty image.")
        self._reference = reference_frame
        self._config = config or CalibrationIntegrityConfig()
        self._backend = backend or OrbRegistrationBackend()
        self._reference_id = reference_id
        self._sampler = IntervalSampler(self._config.check_interval_seconds)
        self._latest: IntegrityAssessment | None = None
        self._latest_evidence: RegistrationEvidence | None = None
        self._checks_run = 0

    @property
    def config(self) -> CalibrationIntegrityConfig:
        return self._config

    @property
    def reference_id(self) -> str | None:
        return self._reference_id

    @property
    def reference_size(self) -> tuple[int, int]:
        """``(width, height)`` of the reference frame."""
        height, width = self._reference.shape[:2]
        return (int(width), int(height))

    @property
    def latest(self) -> IntegrityAssessment | None:
        """The most recent assessment, or ``None`` before the first check."""
        return self._latest

    @property
    def latest_evidence(self) -> RegistrationEvidence | None:
        """Raw evidence behind the most recent assessment.

        Retained for diagnostics — the fitted transform is what lets the overlay
        draw where the reference frame now lands. Deliberately not part of the
        assessment, and so never serialised into the behaviour stream, where a
        nine-float matrix on every record would be noise.
        """
        return self._latest_evidence

    @property
    def state(self) -> IntegrityState:
        """Current state, ``UNCHECKED`` until the first check completes."""
        return IntegrityState.UNCHECKED if self._latest is None else self._latest.state

    @property
    def checks_run(self) -> int:
        return self._checks_run

    def check_if_due(
        self, frame: np.ndarray, frame_index: int, timestamp_seconds: float
    ) -> IntegrityAssessment | None:
        """Run a check if the cadence says one is due.

        Returns:
            The assessment, or ``None`` when no check was due this frame — which
            is the common case, since checks are seconds apart.
        """
        if not self._sampler.should_sample(timestamp_seconds):
            return None
        return self.check(frame, frame_index, timestamp_seconds)

    def check(
        self, frame: np.ndarray, frame_index: int, timestamp_seconds: float
    ) -> IntegrityAssessment:
        """Register ``frame`` against the reference and classify the result.

        Backend failures become ``INSUFFICIENT_EVIDENCE`` rather than
        propagating: a monitor that crashes a long run because one frame was
        awkward is worse than one that reports it could not tell. A genuinely
        broken configuration has already failed at construction.
        """
        current_height, current_width = frame.shape[:2]
        reference_width, reference_height = self.reference_size

        try:
            evidence = self._backend.register(self._reference, frame)
        except CalibrationError as exc:
            logger.debug("Integrity registration failed at frame %d: %s", frame_index, exc)
            evidence = RegistrationEvidence(
                reference_keypoints=0,
                current_keypoints=0,
                matches=0,
                inliers=0,
                failure_reason=str(exc),
            )

        assessment = assess_evidence(
            evidence,
            self._config,
            reference_width=reference_width,
            reference_height=reference_height,
            current_width=int(current_width),
            current_height=int(current_height),
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            reference_id=self._reference_id,
        )
        self._latest = assessment
        self._latest_evidence = evidence
        self._checks_run += 1
        return assessment

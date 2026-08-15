"""What the pipeline does about an integrity finding.

Kept apart from measurement on purpose. The monitor answers "what does the image
show?"; this answers "may metric output continue?". Burying the second question
inside image-registration code would make the safety rule impossible to read,
impossible to test without images, and easy to change by accident.

The rule
--------
**Positively detected drift stops metric observations.** A moved camera still
produces confident-looking metre values — the arithmetic is unchanged, only its
premise is false — and a downstream consumer has no way to notice. Silence is
recoverable; a stream of precise wrong positions is not.

An inconclusive check does **not** stop output, because "could not tell" is not
evidence of drift. It is never silent either: the integrity record goes into the
stream, so a consumer can see exactly which stretch of the run was unverified
and decide for itself.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from atreides_vision.calibration.integrity.config import (
    CalibrationIntegrityConfig,
    IntegrityPolicyMode,
)
from atreides_vision.calibration.integrity.models import IntegrityAssessment, IntegrityState

logger = logging.getLogger(__name__)

__all__ = ["CalibrationIntegrityGuard", "IntegrityMonitor"]


class IntegrityMonitor(Protocol):
    """What the guard needs from a monitor.

    Narrow on purpose: the guard is about consequences, so it depends on the
    smallest possible surface and can be tested against a stub that returns
    scripted assessments with no images involved.
    """

    @property
    def config(self) -> CalibrationIntegrityConfig: ...

    @property
    def state(self) -> IntegrityState: ...

    def check_if_due(
        self, frame: np.ndarray, frame_index: int, timestamp_seconds: float
    ) -> IntegrityAssessment | None: ...


class CalibrationIntegrityGuard:
    """Runs checks on cadence and decides whether metric output may continue.

    The single collaborator the pipeline holds: it owns the monitor, applies the
    policy, and exposes one boolean the processor can act on. One guard per
    camera, so integrity never leaks between cameras in a venue.

    Args:
        monitor: Measures. Anything with ``check_if_due`` and ``state``.
        config: Supplies :attr:`~CalibrationIntegrityConfig.policy_mode`.
            Defaults to the monitor's own configuration.
    """

    def __init__(
        self,
        monitor: IntegrityMonitor,
        config: CalibrationIntegrityConfig | None = None,
    ) -> None:
        self._monitor = monitor
        self._config = config or monitor.config
        self._suppressed = False
        self._suppressed_frames = 0

    @property
    def monitor(self) -> IntegrityMonitor:
        return self._monitor

    @property
    def mode(self) -> IntegrityPolicyMode:
        return self._config.policy_mode

    @property
    def state(self) -> IntegrityState:
        return self._monitor.state

    @property
    def metric_output_allowed(self) -> bool:
        """Whether metric observations may still be serialised.

        Once drift has been detected in guarded mode this stays ``False`` for
        the rest of the run, even if a later check comes back consistent. A
        camera that moved, was checked, and appears to match again has still
        been through an unobserved change; treating the intervening
        measurements as fine because the last check passed would be exactly the
        silent trust this guard exists to prevent. Re-establishing trust means
        re-calibrating, which is a deliberate operator action.
        """
        return not self._suppressed

    @property
    def suppressed_frames(self) -> int:
        """Frames whose observations were withheld because of detected drift."""
        return self._suppressed_frames

    def evaluate(
        self, frame: np.ndarray, frame_index: int, timestamp_seconds: float
    ) -> IntegrityAssessment | None:
        """Check if due, apply the policy, and report the assessment.

        Returns:
            The assessment when a check ran, otherwise ``None``. Callers write
            the returned assessment to the behaviour stream; a ``None`` means
            nothing happened this frame.
        """
        assessment = self._monitor.check_if_due(frame, frame_index, timestamp_seconds)
        if assessment is None:
            return None

        if assessment.state is IntegrityState.DRIFT_DETECTED and not self._suppressed:
            if self._config.policy_mode is IntegrityPolicyMode.GUARDED:
                self._suppressed = True
                logger.warning(
                    "Calibration integrity: camera view drift detected at frame %d (%s). "
                    "Metric behaviour observations are being withheld from here on — the "
                    "calibration's geometry no longer matches this camera. Re-calibrate "
                    "before trusting metres from this camera again.",
                    frame_index,
                    assessment.reason,
                )
            else:
                logger.warning(
                    "Calibration integrity: camera view drift detected at frame %d (%s). "
                    "Report-only mode: metric output continues and may be wrong.",
                    frame_index,
                    assessment.reason,
                )
        elif assessment.state is IntegrityState.INSUFFICIENT_EVIDENCE:
            logger.info(
                "Calibration integrity: could not verify camera geometry at frame %d (%s). "
                "Output continues; this stretch of the run is unverified, not confirmed.",
                frame_index,
                assessment.reason,
            )

        return assessment

    def note_suppressed_frame(self) -> None:
        """Record that one frame's observations were withheld."""
        self._suppressed_frames += 1

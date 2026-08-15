"""Calibration integrity — V0.3.3.

Answers one question, repeatedly, while a calibrated run is in progress:

> Does the current frame still look like it came from the same camera viewpoint
> the calibration was authored against?

A ground-plane calibration is only as valid as the camera pose it was measured
in. Nudge the camera, change the zoom, re-crop the stream, and the homography
keeps producing confident metre values from a premise that is no longer true.
Nothing downstream can notice. This package exists so that failure is detected
and acted on instead of silently propagating.

Integrity is not accuracy
-------------------------
``CONSISTENT`` means the *view has not changed*. It does **not** mean the
calibration's metres were ever correct — reference points measured against the
wrong spots on the floor produce a perfectly consistent, perfectly wrong
calibration. Only physical measurement establishes accuracy; this package
establishes that its precondition still holds. The two claims are kept
rigorously separate throughout.

Layers
------
``reference frame + current frame``
→ :class:`~atreides_vision.calibration.integrity.registration.FrameRegistrationBackend`
→ :class:`~atreides_vision.calibration.integrity.models.RegistrationEvidence`
→ :class:`~atreides_vision.calibration.integrity.monitor.CalibrationIntegrityMonitor`
→ :class:`~atreides_vision.calibration.integrity.models.IntegrityAssessment`
→ :class:`~atreides_vision.calibration.integrity.policy.CalibrationIntegrityGuard`

Measurement and consequence are deliberately separate: the monitor decides what
the image shows, the guard decides what the pipeline does about it.

OpenCV is confined to
:mod:`~atreides_vision.calibration.integrity.registration`. The rest of the
calibration package, and all of the behaviour package, remain free of it.

Boundaries
----------
Registration matches the **static scene**, never people. No face, body,
appearance or biometric feature is computed. Integrity is camera- and run-level
context: it says nothing about any individual, and never appears in a subject
observation. Detection is also not recalibration — drift is reported, never
silently corrected.
"""

from __future__ import annotations

from atreides_vision.calibration.integrity.config import (
    CalibrationIntegrityConfig,
    IntegrityPolicyMode,
)
from atreides_vision.calibration.integrity.models import (
    CalibrationIntegrityMetrics,
    IntegrityAssessment,
    IntegrityState,
    RegistrationEvidence,
)
from atreides_vision.calibration.integrity.monitor import (
    CalibrationIntegrityMonitor,
    assess_evidence,
)
from atreides_vision.calibration.integrity.policy import (
    CalibrationIntegrityGuard,
    IntegrityMonitor,
)
from atreides_vision.calibration.integrity.registration import (
    FrameRegistrationBackend,
    OrbRegistrationBackend,
    to_grayscale,
)

__all__ = [
    "CalibrationIntegrityConfig",
    "CalibrationIntegrityGuard",
    "CalibrationIntegrityMetrics",
    "CalibrationIntegrityMonitor",
    "FrameRegistrationBackend",
    "IntegrityAssessment",
    "IntegrityMonitor",
    "IntegrityPolicyMode",
    "IntegrityState",
    "OrbRegistrationBackend",
    "RegistrationEvidence",
    "assess_evidence",
    "to_grayscale",
]

"""Integrity thresholds, states and drift maths — tested without any images.

The decision logic is a pure function of evidence, so every case here supplies
hand-written numbers. That is the point of the split: a threshold test that
depended on ORB's behaviour would be testing the feature detector, and would
change meaning whenever OpenCV did.
"""

from __future__ import annotations

import math

import pytest

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
from atreides_vision.calibration.integrity.monitor import assess_evidence
from atreides_vision.errors import ConfigurationError

IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
FRAME = (960, 540)


def translation(dx: float, dy: float) -> tuple[float, ...]:
    return (1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0)


def scaling(factor: float) -> tuple[float, ...]:
    return (factor, 0.0, 0.0, 0.0, factor, 0.0, 0.0, 0.0, 1.0)


def evidence(
    *,
    reference_keypoints: int = 500,
    current_keypoints: int = 500,
    matches: int = 200,
    inliers: int = 160,
    error: float | None = 1.0,
    transform: tuple[float, ...] | None = IDENTITY,
    failure_reason: str | None = None,
) -> RegistrationEvidence:
    return RegistrationEvidence(
        reference_keypoints=reference_keypoints,
        current_keypoints=current_keypoints,
        matches=matches,
        inliers=inliers,
        mean_reprojection_error_px=error,
        transform=transform,
        failure_reason=failure_reason,
    )


def assess(
    item: RegistrationEvidence,
    config: CalibrationIntegrityConfig | None = None,
    current: tuple[int, int] = FRAME,
) -> IntegrityAssessment:
    return assess_evidence(
        item,
        config or CalibrationIntegrityConfig(),
        reference_width=FRAME[0],
        reference_height=FRAME[1],
        current_width=current[0],
        current_height=current[1],
    )


class TestStates:
    def test_the_four_states_exist_and_nothing_else(self) -> None:
        assert {state.value for state in IntegrityState} == {
            "unchecked",
            "consistent",
            "drift_detected",
            "insufficient_evidence",
        }

    def test_no_state_claims_physical_accuracy(self) -> None:
        """These names describe the image, never fitness for use."""
        names = {state.value.lower() for state in IntegrityState}
        for forbidden in ("accurate", "safe", "production", "valid", "verified", "trusted"):
            assert not any(forbidden in name for name in names)

    def test_only_findings_are_conclusive(self) -> None:
        assert IntegrityState.CONSISTENT.is_conclusive
        assert IntegrityState.DRIFT_DETECTED.is_conclusive
        assert not IntegrityState.INSUFFICIENT_EVIDENCE.is_conclusive
        assert not IntegrityState.UNCHECKED.is_conclusive


class TestEvidenceValidation:
    @pytest.mark.parametrize(
        "field", ["reference_keypoints", "current_keypoints", "matches", "inliers"]
    )
    def test_rejects_negative_counts(self, field: str) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            evidence(**{field: -1})  # type: ignore[arg-type]

    def test_rejects_more_inliers_than_matches(self) -> None:
        with pytest.raises(ValueError, match="inliers among only"):
            evidence(matches=10, inliers=11)

    def test_rejects_a_malformed_transform(self) -> None:
        with pytest.raises(ValueError, match="must have 9 entries"):
            evidence(transform=(1.0, 0.0, 0.0))

    def test_inlier_ratio_is_zero_without_matches(self) -> None:
        assert evidence(matches=0, inliers=0).inlier_ratio == 0.0

    def test_inlier_ratio(self) -> None:
        assert evidence(matches=200, inliers=50).inlier_ratio == pytest.approx(0.25)


class TestNormalizedDrift:
    """Drift is a fraction of the frame diagonal, never a pixel count."""

    def test_identity_is_zero_drift(self) -> None:
        metrics = CalibrationIntegrityMetrics.from_evidence(evidence(), *FRAME, *FRAME)
        assert metrics.normalized_drift == pytest.approx(0.0)

    def test_a_proportional_resize_is_zero_drift(self) -> None:
        """The V0.3.1 resolution contract, at the integrity layer.

        Half-resolution decode means the fitted transform is a 0.5 scale, and
        every pixel coordinate halves — yet nothing about the view changed.
        """
        half = CalibrationIntegrityMetrics.from_evidence(
            evidence(transform=scaling(0.5)), *FRAME, FRAME[0] // 2, FRAME[1] // 2
        )
        assert half.normalized_drift == pytest.approx(0.0, abs=1e-9)

        double = CalibrationIntegrityMetrics.from_evidence(
            evidence(transform=scaling(2.0)), *FRAME, FRAME[0] * 2, FRAME[1] * 2
        )
        assert double.normalized_drift == pytest.approx(0.0, abs=1e-9)

    def test_translation_scales_with_the_frame_not_the_pixels(self) -> None:
        """The same relative shift reads the same on any resolution."""
        small = CalibrationIntegrityMetrics.from_evidence(
            evidence(transform=translation(9.6, 5.4)), 960, 540, 960, 540
        )
        large = CalibrationIntegrityMetrics.from_evidence(
            evidence(transform=translation(19.2, 10.8)), 1920, 1080, 1920, 1080
        )
        assert small.normalized_drift == pytest.approx(large.normalized_drift)

    def test_translation_magnitude_is_computable_by_hand(self) -> None:
        # Every corner moves by (0.01, 0.01) of the frame; the diagonal is sqrt(2).
        metrics = CalibrationIntegrityMetrics.from_evidence(
            evidence(transform=translation(9.6, 5.4)), 960, 540, 960, 540
        )
        expected = math.hypot(0.01, 0.01) / math.sqrt(2.0)
        assert metrics.normalized_drift == pytest.approx(expected)

    def test_max_drift_is_at_least_the_mean(self) -> None:
        metrics = CalibrationIntegrityMetrics.from_evidence(
            evidence(transform=scaling(1.1)), *FRAME, *FRAME
        )
        assert metrics.max_normalized_drift is not None
        assert metrics.normalized_drift is not None
        assert metrics.max_normalized_drift >= metrics.normalized_drift

    def test_no_transform_means_no_drift_value(self) -> None:
        metrics = CalibrationIntegrityMetrics.from_evidence(
            evidence(transform=None), *FRAME, *FRAME
        )
        assert metrics.normalized_drift is None

    def test_a_degenerate_transform_yields_no_drift(self) -> None:
        metrics = CalibrationIntegrityMetrics.from_evidence(
            evidence(transform=(0.0,) * 9), *FRAME, *FRAME
        )
        assert metrics.normalized_drift is None


class TestConsistent:
    def test_strong_identical_evidence(self) -> None:
        assessment = assess(evidence())
        assert assessment.state is IntegrityState.CONSISTENT
        assert assessment.allows_metric_output

    def test_small_drift_below_the_limit(self) -> None:
        assessment = assess(evidence(transform=translation(4.0, 2.0)))
        assert assessment.state is IntegrityState.CONSISTENT

    def test_a_proportional_resize_is_consistent(self) -> None:
        assessment = assess(
            evidence(transform=scaling(0.5)), current=(FRAME[0] // 2, FRAME[1] // 2)
        )
        assert assessment.state is IntegrityState.CONSISTENT


class TestDriftDetected:
    def test_large_translation(self) -> None:
        assessment = assess(evidence(transform=translation(40.0, 30.0)))
        assert assessment.state is IntegrityState.DRIFT_DETECTED
        assert not assessment.allows_metric_output
        assert "moved" in assessment.reason

    def test_zoom_change(self) -> None:
        assert assess(evidence(transform=scaling(1.1))).state is IntegrityState.DRIFT_DETECTED

    def test_the_worst_corner_alone_can_trigger_it(self) -> None:
        """A change can sit inside the mean limit while one corner swings past it.

        A 2% zoom leaves the near corner still and pushes the far one furthest:
        mean drift ~1.21% of the diagonal, worst corner ~2.00%. With the mean
        limit at 1.5% only the corner check can catch it.
        """
        zoomed = evidence(transform=scaling(1.02))
        metrics = CalibrationIntegrityMetrics.from_evidence(zoomed, *FRAME, *FRAME)
        assert metrics.normalized_drift == pytest.approx(0.0121, abs=5e-4)
        assert metrics.max_normalized_drift == pytest.approx(0.0200, abs=5e-4)

        caught = assess(
            zoomed,
            CalibrationIntegrityConfig(
                maximum_normalized_drift=0.015, maximum_max_normalized_drift=0.018
            ),
        )
        assert caught.state is IntegrityState.DRIFT_DETECTED
        assert "worst corner" in caught.reason

        tolerated = assess(
            zoomed,
            CalibrationIntegrityConfig(
                maximum_normalized_drift=0.5, maximum_max_normalized_drift=0.5
            ),
        )
        assert tolerated.state is IntegrityState.CONSISTENT

    def test_the_threshold_is_configurable(self) -> None:
        moved = evidence(transform=translation(40.0, 30.0))
        tolerant = CalibrationIntegrityConfig(
            maximum_normalized_drift=0.5, maximum_max_normalized_drift=0.5
        )
        assert assess(moved, tolerant).state is IntegrityState.CONSISTENT


class TestInsufficientEvidence:
    """ "Could not tell" must never collapse into "nothing changed"."""

    def test_too_few_keypoints(self) -> None:
        assessment = assess(evidence(reference_keypoints=5, current_keypoints=5))
        assert assessment.state is IntegrityState.INSUFFICIENT_EVIDENCE
        assert "scene features" in assessment.reason

    def test_too_few_keypoints_in_only_one_frame(self) -> None:
        """The weaker of the two decides; a dark current frame is still unknowable."""
        assessment = assess(evidence(reference_keypoints=500, current_keypoints=3))
        assert assessment.state is IntegrityState.INSUFFICIENT_EVIDENCE

    def test_too_few_matches(self) -> None:
        assessment = assess(evidence(matches=5, inliers=5))
        assert assessment.state is IntegrityState.INSUFFICIENT_EVIDENCE
        assert "candidate matches" in assessment.reason

    def test_too_few_inliers(self) -> None:
        assessment = assess(evidence(matches=200, inliers=8))
        assert assessment.state is IntegrityState.INSUFFICIENT_EVIDENCE

    def test_low_inlier_ratio(self) -> None:
        """Plenty of inliers in absolute terms, but the fit explains little."""
        assessment = assess(evidence(matches=1000, inliers=100))
        assert assessment.state is IntegrityState.INSUFFICIENT_EVIDENCE
        assert "agree with the fitted geometry" in assessment.reason

    def test_no_transform(self) -> None:
        assessment = assess(evidence(transform=None))
        assert assessment.state is IntegrityState.INSUFFICIENT_EVIDENCE

    def test_loose_fit(self) -> None:
        assessment = assess(evidence(error=50.0))
        assert assessment.state is IntegrityState.INSUFFICIENT_EVIDENCE
        assert "loose" in assessment.reason

    def test_backend_failure(self) -> None:
        assessment = assess(evidence(failure_reason="camera on fire"))
        assert assessment.state is IntegrityState.INSUFFICIENT_EVIDENCE
        assert "camera on fire" in assessment.reason

    def test_inconclusive_does_not_stop_output(self) -> None:
        """Not evidence of drift — but the stream still records the state."""
        assessment = assess(evidence(matches=5, inliers=5))
        assert assessment.allows_metric_output

    def test_weak_evidence_is_judged_before_drift(self) -> None:
        """A huge apparent shift from six matches is a guess, not a finding."""
        assessment = assess(evidence(matches=6, inliers=6, transform=translation(400.0, 300.0)))
        assert assessment.state is IntegrityState.INSUFFICIENT_EVIDENCE


class TestConfigValidation:
    @pytest.mark.parametrize("field", ["minimum_keypoints", "minimum_matches", "minimum_inliers"])
    def test_rejects_non_positive_counts(self, field: str) -> None:
        with pytest.raises(ConfigurationError, match=f"{field} must be >= 1"):
            CalibrationIntegrityConfig(**{field: 0})  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [0.0, 1.5, -0.2])
    def test_rejects_invalid_inlier_ratio(self, value: float) -> None:
        with pytest.raises(ConfigurationError, match="minimum_inlier_ratio"):
            CalibrationIntegrityConfig(minimum_inlier_ratio=value)

    @pytest.mark.parametrize(
        "field",
        [
            "maximum_reprojection_error_px",
            "maximum_normalized_drift",
            "maximum_max_normalized_drift",
            "check_interval_seconds",
        ],
    )
    def test_rejects_non_positive_limits(self, field: str) -> None:
        with pytest.raises(ConfigurationError, match=field):
            CalibrationIntegrityConfig(**{field: 0.0})  # type: ignore[arg-type]

    def test_rejects_unsatisfiable_inlier_requirement(self) -> None:
        with pytest.raises(ConfigurationError, match="unsatisfiable"):
            CalibrationIntegrityConfig(minimum_matches=10, minimum_inliers=20)

    def test_rejects_a_tighter_max_corner_limit_than_the_mean(self) -> None:
        with pytest.raises(ConfigurationError, match="worst corner cannot be held"):
            CalibrationIntegrityConfig(
                maximum_normalized_drift=0.1, maximum_max_normalized_drift=0.05
            )

    def test_defaults_are_valid_and_guarded(self) -> None:
        config = CalibrationIntegrityConfig()
        assert config.policy_mode is IntegrityPolicyMode.GUARDED
        assert config.check_interval_seconds == 5.0

    def test_no_threshold_encodes_a_physical_size(self) -> None:
        """Drift limits are frame fractions, so venues of any size share them."""
        config = CalibrationIntegrityConfig()
        assert 0.0 < config.maximum_normalized_drift < 1.0
        assert 0.0 < config.maximum_max_normalized_drift < 1.0


class TestAssessmentRecord:
    def test_serialises_state_and_metrics(self) -> None:
        record = assess(evidence()).to_record()
        assert record["state"] == "consistent"
        assert record["metrics"]["inliers"] == 160
        assert "normalized_drift" in record["metrics"]

    def test_carries_the_reference_identifier(self) -> None:
        assessment = assess_evidence(
            evidence(),
            CalibrationIntegrityConfig(),
            reference_width=FRAME[0],
            reference_height=FRAME[1],
            current_width=FRAME[0],
            current_height=FRAME[1],
            reference_id="reference.png",
        )
        assert assessment.to_record()["reference_id"] == "reference.png"

    def test_record_contains_no_person_fields(self) -> None:
        """Integrity is camera-level context and must never look person-level."""
        serialised = str(assess(evidence()).to_record()).lower()
        for forbidden in ("track", "subject", "person", "identity"):
            assert forbidden not in serialised


class TestDeterminism:
    def test_the_same_evidence_always_gives_the_same_state(self) -> None:
        item = evidence(transform=translation(12.0, 9.0))
        states = {assess(item).state for _ in range(10)}
        assert len(states) == 1

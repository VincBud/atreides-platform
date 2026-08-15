"""The fail-safe policy: what happens to metric output after a finding.

No images and no thresholds here. The guard is handed scripted assessments, so
these tests describe the safety rule itself rather than the conditions that
happen to trigger it.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from atreides_vision.calibration.integrity.config import (
    CalibrationIntegrityConfig,
    IntegrityPolicyMode,
)
from atreides_vision.calibration.integrity.models import IntegrityAssessment, IntegrityState
from atreides_vision.calibration.integrity.policy import CalibrationIntegrityGuard

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


class ScriptedMonitor:
    """Returns pre-arranged assessments, one per check, ignoring the image."""

    def __init__(
        self,
        states: list[IntegrityState | None],
        config: CalibrationIntegrityConfig | None = None,
    ) -> None:
        self._states = list(states)
        self.config = config or CalibrationIntegrityConfig()
        self.state = IntegrityState.UNCHECKED
        self.calls = 0

    def check_if_due(
        self, frame: np.ndarray, frame_index: int, timestamp_seconds: float
    ) -> IntegrityAssessment | None:
        self.calls += 1
        state = self._states.pop(0) if self._states else None
        if state is None:
            return None
        self.state = state
        return IntegrityAssessment(
            state=state,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            reason=f"scripted {state.value}",
        )


def guard(
    states: list[IntegrityState | None], mode: IntegrityPolicyMode = IntegrityPolicyMode.GUARDED
) -> tuple[CalibrationIntegrityGuard, ScriptedMonitor]:
    config = CalibrationIntegrityConfig(policy_mode=mode)
    monitor = ScriptedMonitor(states, config)
    return CalibrationIntegrityGuard(monitor, config), monitor


class TestBeforeAnyCheck:
    def test_output_is_allowed_while_unchecked(self) -> None:
        """A run must not be blocked simply because the first check is not due."""
        instance, _ = guard([None])
        assert instance.state is IntegrityState.UNCHECKED
        assert instance.metric_output_allowed

    def test_no_assessment_is_returned_when_none_is_due(self) -> None:
        instance, _ = guard([None])
        assert instance.evaluate(FRAME, 1, 0.0) is None


class TestConsistent:
    def test_output_continues(self) -> None:
        instance, _ = guard([IntegrityState.CONSISTENT])
        assessment = instance.evaluate(FRAME, 1, 0.0)
        assert assessment is not None
        assert assessment.state is IntegrityState.CONSISTENT
        assert instance.metric_output_allowed

    def test_repeated_consistency_keeps_output_flowing(self) -> None:
        instance, _ = guard([IntegrityState.CONSISTENT] * 5)
        for index in range(5):
            instance.evaluate(FRAME, index, float(index))
        assert instance.metric_output_allowed


class TestDriftDetected:
    def test_output_stops(self) -> None:
        instance, _ = guard([IntegrityState.DRIFT_DETECTED])
        instance.evaluate(FRAME, 1, 0.0)
        assert not instance.metric_output_allowed

    def test_suppression_persists_after_a_later_consistent_check(self) -> None:
        """Trust does not come back on its own.

        The camera moved at some point; every measurement since the previous
        check is suspect, and a later passing check cannot retroactively vouch
        for them. Only re-calibration should clear this.
        """
        instance, _ = guard([IntegrityState.DRIFT_DETECTED, IntegrityState.CONSISTENT])
        instance.evaluate(FRAME, 1, 0.0)
        instance.evaluate(FRAME, 2, 5.0)
        assert not instance.metric_output_allowed

    def test_it_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        instance, _ = guard([IntegrityState.DRIFT_DETECTED])
        with caplog.at_level(logging.WARNING):
            instance.evaluate(FRAME, 7, 1.0)
        assert "drift detected" in caplog.text
        assert "withheld" in caplog.text

    def test_it_warns_once_not_on_every_frame(self, caplog: pytest.LogCaptureFixture) -> None:
        instance, _ = guard([IntegrityState.DRIFT_DETECTED] * 4)
        with caplog.at_level(logging.WARNING):
            for index in range(4):
                instance.evaluate(FRAME, index, float(index))
        assert caplog.text.count("withheld") == 1

    def test_suppressed_frames_are_counted(self) -> None:
        instance, _ = guard([IntegrityState.DRIFT_DETECTED])
        instance.evaluate(FRAME, 1, 0.0)
        for _ in range(3):
            instance.note_suppressed_frame()
        assert instance.suppressed_frames == 3


class TestReportOnly:
    def test_output_continues_despite_drift(self) -> None:
        instance, _ = guard([IntegrityState.DRIFT_DETECTED], IntegrityPolicyMode.REPORT_ONLY)
        instance.evaluate(FRAME, 1, 0.0)
        assert instance.metric_output_allowed

    def test_it_still_warns_loudly(self, caplog: pytest.LogCaptureFixture) -> None:
        """Never silent: report-only must not look like everything is fine."""
        instance, _ = guard([IntegrityState.DRIFT_DETECTED], IntegrityPolicyMode.REPORT_ONLY)
        with caplog.at_level(logging.WARNING):
            instance.evaluate(FRAME, 1, 0.0)
        assert "may be wrong" in caplog.text

    def test_the_assessment_is_still_returned_for_the_stream(self) -> None:
        instance, _ = guard([IntegrityState.DRIFT_DETECTED], IntegrityPolicyMode.REPORT_ONLY)
        assessment = instance.evaluate(FRAME, 1, 0.0)
        assert assessment is not None
        assert assessment.state is IntegrityState.DRIFT_DETECTED


class TestInsufficientEvidence:
    def test_output_continues(self) -> None:
        """Not evidence of drift, so not grounds to discard measurements."""
        instance, _ = guard([IntegrityState.INSUFFICIENT_EVIDENCE])
        instance.evaluate(FRAME, 1, 0.0)
        assert instance.metric_output_allowed

    def test_it_is_recorded_rather_than_passed_over(self, caplog: pytest.LogCaptureFixture) -> None:
        instance, _ = guard([IntegrityState.INSUFFICIENT_EVIDENCE])
        with caplog.at_level(logging.INFO):
            assessment = instance.evaluate(FRAME, 1, 0.0)
        assert assessment is not None
        assert "could not verify" in caplog.text
        assert "unverified, not confirmed" in caplog.text

    def test_it_does_not_clear_an_earlier_suppression(self) -> None:
        instance, _ = guard([IntegrityState.DRIFT_DETECTED, IntegrityState.INSUFFICIENT_EVIDENCE])
        instance.evaluate(FRAME, 1, 0.0)
        instance.evaluate(FRAME, 2, 5.0)
        assert not instance.metric_output_allowed


class TestPerCameraIsolation:
    def test_two_guards_reach_verdicts_independently(self) -> None:
        """One camera being knocked must never mark another as drifted."""
        left, _ = guard([IntegrityState.DRIFT_DETECTED])
        right, _ = guard([IntegrityState.CONSISTENT])

        left.evaluate(FRAME, 1, 0.0)
        right.evaluate(FRAME, 1, 0.0)

        assert not left.metric_output_allowed
        assert right.metric_output_allowed
        assert left.state is IntegrityState.DRIFT_DETECTED
        assert right.state is IntegrityState.CONSISTENT

    def test_suppression_counts_are_separate(self) -> None:
        left, _ = guard([IntegrityState.DRIFT_DETECTED])
        right, _ = guard([IntegrityState.CONSISTENT])
        left.evaluate(FRAME, 1, 0.0)
        left.note_suppressed_frame()
        assert left.suppressed_frames == 1
        assert right.suppressed_frames == 0


class TestCadence:
    """Checks run on a schedule, not every frame."""

    def test_the_monitor_decides_when_to_check(self) -> None:
        instance, monitor = guard([None, None, IntegrityState.CONSISTENT])
        assert instance.evaluate(FRAME, 1, 0.0) is None
        assert instance.evaluate(FRAME, 2, 0.04) is None
        assert instance.evaluate(FRAME, 3, 5.0) is not None
        assert monitor.calls == 3

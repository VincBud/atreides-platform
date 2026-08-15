"""The ORB registration backend, against controlled synthetic geometry changes.

Separate from the threshold tests on purpose: these exercise the real feature
detector on real pixels, so they are the only place a change in OpenCV's
behaviour should show up.

Every scene is seeded, so a verdict never depends on chance.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from atreides_vision.calibration.integrity.config import CalibrationIntegrityConfig
from atreides_vision.calibration.integrity.models import IntegrityState
from atreides_vision.calibration.integrity.monitor import CalibrationIntegrityMonitor
from atreides_vision.calibration.integrity.registration import (
    OrbRegistrationBackend,
    to_grayscale,
)
from atreides_vision.errors import CalibrationError
from synthetic_scenes import (
    crop_and_rescale,
    low_texture_scene,
    moving_foreground,
    perspective_warp,
    rotate,
    scale_about_centre,
    synthetic_scene,
    translate,
    with_noise,
)

REFERENCE = synthetic_scene()


def state_for(
    current: np.ndarray, config: CalibrationIntegrityConfig | None = None
) -> IntegrityState:
    monitor = CalibrationIntegrityMonitor(REFERENCE, config or CalibrationIntegrityConfig())
    return monitor.check(current, frame_index=1, timestamp_seconds=0.0).state


def drift_for(current: np.ndarray) -> float | None:
    monitor = CalibrationIntegrityMonitor(REFERENCE, CalibrationIntegrityConfig())
    metrics = monitor.check(current, 1, 0.0).metrics
    return None if metrics is None else metrics.normalized_drift


class TestStableGeometry:
    """The camera has not moved. These must not read as drift."""

    def test_the_identical_frame(self) -> None:
        assert state_for(REFERENCE) is IntegrityState.CONSISTENT
        assert drift_for(REFERENCE) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("factor", [0.5, 0.75, 2.0])
    def test_a_proportional_resize(self, factor: float) -> None:
        """The V0.3.1 resolution contract: a decode resize is not a camera move."""
        height, width = REFERENCE.shape[:2]
        resized = cv2.resize(REFERENCE, (int(width * factor), int(height * factor)))
        assert state_for(resized) is IntegrityState.CONSISTENT

        drift = drift_for(resized)
        assert drift is not None and drift < 0.005

    def test_sensor_noise(self) -> None:
        assert state_for(with_noise(REFERENCE)) is IntegrityState.CONSISTENT

    def test_a_brightness_change(self) -> None:
        """ORB works on intensity gradients, so uniform lighting shifts survive."""
        brighter = np.clip(REFERENCE.astype(np.int16) + 30, 0, 255).astype(np.uint8)
        assert state_for(brighter) is IntegrityState.CONSISTENT

    def test_moving_people_in_front_of_a_stable_background(self) -> None:
        """The everyday venue case: foreground moves, camera does not.

        The robust fit is supposed to treat the moving blocks as the minority
        that fails to agree, and hold onto the static scene behind them.
        """
        before = moving_foreground(REFERENCE, offset=0)
        after = moving_foreground(REFERENCE, offset=60)

        monitor = CalibrationIntegrityMonitor(before, CalibrationIntegrityConfig())
        assessment = monitor.check(after, 1, 0.0)
        assert assessment.state is IntegrityState.CONSISTENT

    def test_heavier_foreground_still_leaves_enough_background(self) -> None:
        before = moving_foreground(REFERENCE, count=10, offset=0)
        after = moving_foreground(REFERENCE, count=10, offset=45)
        monitor = CalibrationIntegrityMonitor(before, CalibrationIntegrityConfig())
        assert monitor.check(after, 1, 0.0).state is IntegrityState.CONSISTENT


class TestChangedGeometry:
    """The camera view has changed. These must be caught."""

    @pytest.mark.parametrize("shift", [25, 60, 120])
    def test_translation(self, shift: int) -> None:
        assert state_for(translate(REFERENCE, shift, shift * 0.6)) is (
            IntegrityState.DRIFT_DETECTED
        )

    @pytest.mark.parametrize("degrees", [3.0, 8.0])
    def test_rotation(self, degrees: float) -> None:
        assert state_for(rotate(REFERENCE, degrees)) is IntegrityState.DRIFT_DETECTED

    @pytest.mark.parametrize("factor", [1.12, 0.85])
    def test_zoom(self, factor: float) -> None:
        assert state_for(scale_about_centre(REFERENCE, factor)) is IntegrityState.DRIFT_DETECTED

    def test_crop_and_rescale(self) -> None:
        """A re-framed stream keeps the resolution but changes the field of view."""
        assert state_for(crop_and_rescale(REFERENCE)) is IntegrityState.DRIFT_DETECTED

    def test_perspective_change(self) -> None:
        assert state_for(perspective_warp(REFERENCE)) is IntegrityState.DRIFT_DETECTED

    def test_drift_grows_with_the_change(self) -> None:
        small = drift_for(translate(REFERENCE, 20, 12))
        large = drift_for(translate(REFERENCE, 90, 54))
        assert small is not None and large is not None
        assert large > small > 0.0


class TestInsufficientEvidence:
    def test_a_blank_frame(self) -> None:
        assert state_for(low_texture_scene()) is IntegrityState.INSUFFICIENT_EVIDENCE

    def test_a_blank_reference(self) -> None:
        monitor = CalibrationIntegrityMonitor(low_texture_scene(), CalibrationIntegrityConfig())
        assert monitor.check(REFERENCE, 1, 0.0).state is IntegrityState.INSUFFICIENT_EVIDENCE

    def test_an_unrelated_scene(self) -> None:
        """A completely different view yields no coherent geometry, not a drift value."""
        other = synthetic_scene(seed=999)
        assert state_for(other) in {
            IntegrityState.INSUFFICIENT_EVIDENCE,
            IntegrityState.DRIFT_DETECTED,
        }

    def test_blank_is_never_reported_as_consistent(self) -> None:
        """The single most dangerous confusion this module could make."""
        assert state_for(low_texture_scene()) is not IntegrityState.CONSISTENT


class TestBackendMechanics:
    def test_grayscale_accepts_bgr_and_grey(self) -> None:
        assert to_grayscale(REFERENCE).ndim == 2
        grey = cv2.cvtColor(REFERENCE, cv2.COLOR_BGR2GRAY)
        assert np.array_equal(to_grayscale(grey), grey)

    def test_grayscale_rejects_an_empty_frame(self) -> None:
        with pytest.raises(CalibrationError, match="empty frame"):
            to_grayscale(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_grayscale_rejects_an_odd_shape(self) -> None:
        with pytest.raises(CalibrationError, match="BGR or grayscale"):
            to_grayscale(np.zeros((10, 10, 5), dtype=np.uint8))

    @pytest.mark.parametrize(
        ("field", "value"),
        [("max_features", 2), ("match_ratio", 0.0), ("ransac_reprojection_threshold_px", 0.0)],
    )
    def test_rejects_invalid_settings(self, field: str, value: float) -> None:
        with pytest.raises(CalibrationError):
            OrbRegistrationBackend(**{field: value})  # type: ignore[arg-type]

    def test_evidence_reports_counts(self) -> None:
        evidence = OrbRegistrationBackend().register(REFERENCE, REFERENCE)
        assert evidence.reference_keypoints > 0
        assert evidence.inliers > 0
        assert evidence.transform is not None
        assert evidence.mean_reprojection_error_px is not None

    def test_a_blank_pair_reports_nothing_rather_than_raising(self) -> None:
        blank = low_texture_scene()
        evidence = OrbRegistrationBackend().register(blank, blank)
        assert evidence.matches == 0
        assert evidence.transform is None


class TestDeterminism:
    def test_repeated_registration_is_identical(self) -> None:
        backend = OrbRegistrationBackend()
        moved = translate(REFERENCE, 30, 18)
        first = backend.register(REFERENCE, moved)
        for _ in range(4):
            assert backend.register(REFERENCE, moved) == first

    def test_repeated_states_are_identical(self) -> None:
        moved = rotate(REFERENCE, 5.0)
        assert len({state_for(moved) for _ in range(3)}) == 1

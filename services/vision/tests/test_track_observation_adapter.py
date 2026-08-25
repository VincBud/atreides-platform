"""V0.4.7: converting real camera-local runtime state into a TrackObservation.

Every test below builds its inputs from the actual runtime types this adapter
reads — :class:`BehaviorSubjectSnapshot`, :class:`StreamFrame`,
:class:`GroundPlaneProjector`, :class:`CalibrationIntegrityGuard` — rather than
hand-building a :class:`TrackObservation` and checking it round-trips, so a
regression in the adapter's field mapping cannot hide behind a fixture that
already assumes the mapping is correct.
"""

from __future__ import annotations

import numpy as np
import pytest

from atreides_vision.association import (
    AssociationEngine,
    AssociationGatingConfig,
    GeometryTrust,
)
from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.behavior.models import BehaviorSubjectSnapshot, SubjectRef
from atreides_vision.calibration.integrity.config import (
    CalibrationIntegrityConfig,
    IntegrityPolicyMode,
)
from atreides_vision.calibration.integrity.models import IntegrityAssessment, IntegrityState
from atreides_vision.calibration.integrity.policy import CalibrationIntegrityGuard
from atreides_vision.calibration.projector import GroundPlaneProjector
from atreides_vision.errors import AssociationError
from atreides_vision.timing import StreamFrame, TimeUncertainty
from atreides_vision.timing import StreamRef as StreamRefType
from atreides_vision.track_observation_adapter import build_track_observation
from synthetic_cameras import CAMERA_A, FRAME_A, VENUE_ID, calibration_for

NORMALIZED = MotionSpace.NORMALIZED_IMAGE
METRIC = MotionSpace.GROUND_PLANE_METRES
FRAME_IMAGE = np.zeros((4, 4, 3), dtype=np.uint8)


def stream_ref(camera_id: str = "cam_a", stream_epoch: int = 0) -> StreamRefType:
    return StreamRefType(camera_id, stream_epoch)


def stream_frame(
    *,
    camera_id: str = "cam_a",
    stream_epoch: int = 0,
    frame_index: int = 1,
    venue_seconds: float = 10.0,
    half_width_seconds: float | None = 0.25,
) -> StreamFrame:
    timing = (
        TimeUncertainty.unknown()
        if half_width_seconds is None
        else TimeUncertainty.declared(half_width_seconds)
    )
    return StreamFrame(
        stream=stream_ref(camera_id, stream_epoch),
        frame_index=frame_index,
        source_seconds=venue_seconds,
        venue_seconds=venue_seconds,
        timing=timing,
        image=FRAME_IMAGE,
    )


def snapshot_normalized(
    *,
    camera_id: str = "cam_a",
    track_id: int = 7,
    frame_index: int = 1,
    timestamp_seconds: float = 10.0,
    x: float = 0.25,
    y: float = 0.5,
) -> BehaviorSubjectSnapshot:
    return BehaviorSubjectSnapshot(
        subject=SubjectRef(camera_id, track_id),
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        position=MotionPoint(x=x, y=y, space=NORMALIZED),
    )


def metric_projector(camera_id: str = "cam_a") -> GroundPlaneProjector:
    calibration = calibration_for(
        CAMERA_A,
        [(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)],
        camera_id=camera_id,
        venue_id=VENUE_ID,
        reference_frame=FRAME_A,
    )
    return GroundPlaneProjector.from_calibration(calibration)


def snapshot_metric(
    projector: GroundPlaneProjector,
    *,
    track_id: int = 7,
    frame_index: int = 1,
    timestamp_seconds: float = 10.0,
    venue_x: float = 3.0,
    venue_y: float = 2.0,
) -> BehaviorSubjectSnapshot:
    return BehaviorSubjectSnapshot(
        subject=SubjectRef(projector.camera_id, track_id),
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        position=MotionPoint(x=venue_x, y=venue_y, space=METRIC),
    )


class ScriptedMonitor:
    """Same fixture idiom as ``test_calibration_integrity_guard.py``."""

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
    states: list[IntegrityState | None],
    mode: IntegrityPolicyMode = IntegrityPolicyMode.GUARDED,
) -> CalibrationIntegrityGuard:
    config = CalibrationIntegrityConfig(policy_mode=mode)
    monitor = ScriptedMonitor(states, config)
    instance = CalibrationIntegrityGuard(monitor, config)
    for _ in states:
        instance.evaluate(FRAME_IMAGE, 1, 0.0)
    return instance


class TestNormalizedSingleCamera:
    """The uncalibrated, single-camera path — no special mode, no crash."""

    def test_fields_are_preserved_verbatim(self) -> None:
        snapshot = snapshot_normalized()
        frame = stream_frame()
        observation = build_track_observation(snapshot, frame)

        assert observation.subject == snapshot.subject
        assert observation.stream == frame.stream
        assert observation.frame_index == frame.frame_index
        assert observation.venue_seconds == frame.venue_seconds
        assert observation.timing == frame.timing
        assert observation.position == snapshot.position

    def test_spatial_context_has_no_venue_or_calibration(self) -> None:
        observation = build_track_observation(snapshot_normalized(), stream_frame())
        assert observation.spatial.space is NORMALIZED
        assert observation.spatial.venue_id is None
        assert observation.spatial.surface_id is None
        assert observation.spatial.calibrated_region is None

    def test_no_integrity_guard_is_unverified_not_untrusted(self) -> None:
        observation = build_track_observation(snapshot_normalized(), stream_frame())
        assert observation.spatial.trust is GeometryTrust.UNVERIFIED

    def test_unknown_timing_reaches_the_observation_unknown(self) -> None:
        frame = stream_frame(half_width_seconds=None)
        observation = build_track_observation(snapshot_normalized(), frame)
        assert not observation.timing.is_established
        assert observation.timing == TimeUncertainty.unknown()

    def test_inputs_are_not_mutated(self) -> None:
        snapshot = snapshot_normalized()
        frame = stream_frame()
        build_track_observation(snapshot, frame)
        assert snapshot.position == MotionPoint(x=0.25, y=0.5, space=NORMALIZED)
        assert frame.venue_seconds == 10.0

    def test_determinism_same_inputs_same_output_twice(self) -> None:
        snapshot = snapshot_normalized()
        frame = stream_frame()
        first = build_track_observation(snapshot, frame)
        second = build_track_observation(snapshot, frame)
        assert first == second


class TestMetricCalibratedCamera:
    def test_position_and_units_are_preserved(self) -> None:
        projector = metric_projector()
        snapshot = snapshot_metric(projector, venue_x=3.0, venue_y=2.0)
        frame = stream_frame(camera_id=projector.camera_id)

        observation = build_track_observation(snapshot, frame, calibration=projector)

        assert observation.position == snapshot.position
        assert observation.position.space is METRIC
        assert observation.spatial.is_metric

    def test_calibrated_region_covers_the_calibration_points(self) -> None:
        projector = metric_projector()
        snapshot = snapshot_metric(projector)
        frame = stream_frame(camera_id=projector.camera_id)

        observation = build_track_observation(snapshot, frame, calibration=projector)

        region = observation.spatial.calibrated_region
        assert region is not None
        assert region.min_x <= 0.0 and region.max_x >= 10.0
        assert region.min_y <= 0.0 and region.max_y >= 8.0

    def test_venue_and_surface_id_come_from_calibration(self) -> None:
        projector = metric_projector()
        snapshot = snapshot_metric(projector)
        frame = stream_frame(camera_id=projector.camera_id)

        observation = build_track_observation(snapshot, frame, calibration=projector)

        assert observation.spatial.venue_id == projector.venue_id
        assert observation.spatial.surface_id == projector.calibration.surface_id


class TestGeometryTrustDerivation:
    def test_consistent_state_yields_verified(self) -> None:
        active_guard = guard([IntegrityState.CONSISTENT])
        observation = build_track_observation(
            snapshot_normalized(), stream_frame(), integrity_guard=active_guard
        )
        assert observation.spatial.trust is GeometryTrust.VERIFIED

    def test_unchecked_state_yields_unverified(self) -> None:
        active_guard = guard([None])
        observation = build_track_observation(
            snapshot_normalized(), stream_frame(), integrity_guard=active_guard
        )
        assert observation.spatial.trust is GeometryTrust.UNVERIFIED

    def test_insufficient_evidence_yields_unverified(self) -> None:
        active_guard = guard([IntegrityState.INSUFFICIENT_EVIDENCE])
        observation = build_track_observation(
            snapshot_normalized(), stream_frame(), integrity_guard=active_guard
        )
        assert observation.spatial.trust is GeometryTrust.UNVERIFIED

    def test_drift_detected_yields_untrusted(self) -> None:
        active_guard = guard([IntegrityState.DRIFT_DETECTED])
        observation = build_track_observation(
            snapshot_normalized(), stream_frame(), integrity_guard=active_guard
        )
        assert observation.spatial.trust is GeometryTrust.UNTRUSTED

    def test_guarded_suppression_stays_untrusted_even_after_a_later_consistent_read(
        self,
    ) -> None:
        """The guard's own sticky invariant must survive translation to GeometryTrust.

        In GUARDED mode, once drift is detected metric output stays withheld
        forever, even if a later individual check happens to read consistent
        again. GeometryTrust must not silently re-extend trust the guard
        itself refuses to.
        """
        active_guard = guard(
            [IntegrityState.DRIFT_DETECTED, IntegrityState.CONSISTENT],
            mode=IntegrityPolicyMode.GUARDED,
        )
        assert active_guard.state is IntegrityState.CONSISTENT
        assert not active_guard.metric_output_allowed

        observation = build_track_observation(
            snapshot_normalized(), stream_frame(), integrity_guard=active_guard
        )
        assert observation.spatial.trust is GeometryTrust.UNTRUSTED

    def test_report_only_mode_does_not_suppress_but_still_reports_untrusted_on_drift(
        self,
    ) -> None:
        active_guard = guard([IntegrityState.DRIFT_DETECTED], mode=IntegrityPolicyMode.REPORT_ONLY)
        assert active_guard.metric_output_allowed

        observation = build_track_observation(
            snapshot_normalized(), stream_frame(), integrity_guard=active_guard
        )
        assert observation.spatial.trust is GeometryTrust.UNTRUSTED


class TestStreamEpochProvenance:
    def test_same_track_id_different_epoch_remains_distinct(self) -> None:
        first = build_track_observation(
            snapshot_normalized(track_id=7),
            stream_frame(stream_epoch=0),
        )
        second = build_track_observation(
            snapshot_normalized(track_id=7),
            stream_frame(stream_epoch=1),
        )
        assert first.stream != second.stream
        assert first.stream.stream_epoch == 0
        assert second.stream.stream_epoch == 1
        assert first.subject == second.subject  # same camera-local track id
        assert first != second

    def test_stream_epoch_is_never_derived_from_frame_index_or_track_id(self) -> None:
        observation = build_track_observation(
            snapshot_normalized(track_id=99, frame_index=42),
            stream_frame(stream_epoch=3, frame_index=42),
        )
        assert observation.stream.stream_epoch == 3


class TestStructuralImpossibilities:
    def test_rejects_camera_mismatch_between_subject_and_stream(self) -> None:
        snapshot = snapshot_normalized(camera_id="cam_a")
        frame = stream_frame(camera_id="cam_b")
        with pytest.raises(AssociationError):
            build_track_observation(snapshot, frame)

    def test_rejects_frame_index_mismatch(self) -> None:
        snapshot = snapshot_normalized(frame_index=1)
        frame = stream_frame(frame_index=2)
        with pytest.raises(AssociationError):
            build_track_observation(snapshot, frame)

    def test_rejects_timestamp_mismatch(self) -> None:
        snapshot = snapshot_normalized(timestamp_seconds=10.0)
        frame = stream_frame(venue_seconds=10.5)
        with pytest.raises(AssociationError):
            build_track_observation(snapshot, frame)

    def test_rejects_metric_position_without_calibration(self) -> None:
        projector = metric_projector()
        snapshot = snapshot_metric(projector)
        frame = stream_frame(camera_id=projector.camera_id)
        with pytest.raises(AssociationError):
            build_track_observation(snapshot, frame)

    def test_rejects_calibration_for_normalized_position(self) -> None:
        projector = metric_projector()
        snapshot = snapshot_normalized(camera_id=projector.camera_id)
        frame = stream_frame(camera_id=projector.camera_id)
        with pytest.raises(AssociationError):
            build_track_observation(snapshot, frame, calibration=projector)

    def test_rejects_calibration_naming_a_different_camera(self) -> None:
        projector = metric_projector(camera_id="cam_other")
        snapshot = snapshot_metric(projector, track_id=1)
        # Force a frame for a different camera than the projector.
        mismatched_frame = stream_frame(camera_id="cam_a")
        with pytest.raises(AssociationError):
            build_track_observation(
                BehaviorSubjectSnapshot(
                    subject=SubjectRef("cam_a", 1),
                    frame_index=1,
                    timestamp_seconds=10.0,
                    position=snapshot.position,
                ),
                mismatched_frame,
                calibration=projector,
            )


class TestDownstreamCompatibility:
    """The output must actually be accepted by AssociationEngine's own gates."""

    def test_two_untrusted_observations_are_not_silently_compared(self) -> None:
        active_guard = guard([IntegrityState.DRIFT_DETECTED])
        projector_a = metric_projector(camera_id="cam_a")
        projector_b = metric_projector(camera_id="cam_b")

        observation_a = build_track_observation(
            snapshot_metric(projector_a, track_id=1, venue_x=3.0, venue_y=2.0),
            stream_frame(camera_id="cam_a"),
            calibration=projector_a,
            integrity_guard=active_guard,
        )
        observation_b = build_track_observation(
            snapshot_metric(projector_b, track_id=2, venue_x=3.05, venue_y=2.0),
            stream_frame(camera_id="cam_b"),
            calibration=projector_b,
            integrity_guard=active_guard,
        )

        engine = AssociationEngine(
            gating=AssociationGatingConfig(
                max_pairing_gap_seconds=0.5,
                max_subject_speed_m_per_s=2.0,
                position_tolerance_m=1.0,
                ambiguity_margin_m=0.5,
            )
        )
        round_ = engine.observe(10.0, [observation_a, observation_b])
        assert len(round_.decisions) == 1
        decision = round_.decisions[0]
        assert decision.outcome.value == "not_comparable"

    def test_verified_close_observations_are_comparable(self) -> None:
        projector_a = metric_projector(camera_id="cam_a")
        projector_b = metric_projector(camera_id="cam_b")

        observation_a = build_track_observation(
            snapshot_metric(projector_a, track_id=1, venue_x=3.0, venue_y=2.0),
            stream_frame(camera_id="cam_a"),
            calibration=projector_a,
        )
        observation_b = build_track_observation(
            snapshot_metric(projector_b, track_id=2, venue_x=3.05, venue_y=2.0),
            stream_frame(camera_id="cam_b"),
            calibration=projector_b,
        )

        engine = AssociationEngine(
            gating=AssociationGatingConfig(
                max_pairing_gap_seconds=0.5,
                max_subject_speed_m_per_s=2.0,
                position_tolerance_m=1.0,
                ambiguity_margin_m=0.5,
            )
        )
        round_ = engine.observe(10.0, [observation_a, observation_b])
        assert len(round_.decisions) == 1
        assert round_.decisions[0].outcome.value != "not_comparable"

    def test_single_camera_path_produces_not_comparable_for_self(self) -> None:
        """One camera is the same path: no special mode, no crash."""
        projector = metric_projector()
        observation_a = build_track_observation(
            snapshot_metric(projector, track_id=1, venue_x=3.0, venue_y=2.0),
            stream_frame(camera_id=projector.camera_id),
            calibration=projector,
        )
        observation_b = build_track_observation(
            snapshot_metric(projector, track_id=2, venue_x=3.05, venue_y=2.0),
            stream_frame(camera_id=projector.camera_id),
            calibration=projector,
        )
        engine = AssociationEngine(
            gating=AssociationGatingConfig(
                max_pairing_gap_seconds=0.5,
                max_subject_speed_m_per_s=2.0,
                position_tolerance_m=1.0,
                ambiguity_margin_m=0.5,
            )
        )
        round_ = engine.observe(10.0, [observation_a, observation_b])
        # Same-camera candidates are always NOT_COMPARABLE; association never
        # compares two tracklets from the same camera to each other.
        assert len(round_.decisions) == 1
        assert round_.decisions[0].outcome.value == "not_comparable"

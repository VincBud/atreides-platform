"""Behaviour signals through a calibrated ground plane: the V0.3.1 payoff.

Three groups matter most.

:class:`TestMetricSignals` proves the V0.3.0 seam held — swapping the projector
turns every signal metric with no change to history, kinematics or the analyzer.

:class:`TestMultiCamera` proves the architecture scales past one camera: two
cameras with entirely different image geometry, calibrated against one venue
frame, recover the same trajectory and the same signals — while their
observations stay strictly separate, including when they use the same numeric
track id.

:class:`TestPerspective` proves why any of this was necessary: equal physical
movement produces unequal image movement, and only calibration recovers the
physical truth.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

import pytest

from atreides_vision.behavior.analyzer import BehaviorAnalyzer
from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.behavior.coordinates import MotionSpace, NormalizedFrameProjector
from atreides_vision.behavior.models import BehaviorFrame, SignalType, SubjectRef
from atreides_vision.calibration.projector import GroundPlaneProjector
from atreides_vision.errors import ConfigurationError
from atreides_vision.models import BoundingBox, Point, Track
from synthetic_cameras import (
    CAMERA_A,
    CAMERA_B,
    CAMERA_STRONG_PERSPECTIVE,
    FRAME_A,
    FRAME_B,
    VENUE_ID,
    Matrix3x3,
    calibration_for,
    project_venue_to_image,
    track_at_venue,
)

FPS = 25.0
DT = 1.0 / FPS

#: Reference points spanning the floor area the synthetic people walk across.
CALIBRATION_REGION = [(0.0, 0.0), (12.0, 0.0), (12.0, 9.0), (0.0, 9.0)]

#: A brisk but ordinary walking pace, in metres per second.
WALKING_SPEED = 1.4

VenuePath = Callable[[float], tuple[float, float]]


def ground_plane_analyzer(
    matrix: Matrix3x3 = CAMERA_A,
    camera_id: str = "cam_a",
    reference_frame: object = FRAME_A,
    frame_width: int | None = None,
    frame_height: int | None = None,
    venue_id: str = VENUE_ID,
    **config_overrides: object,
) -> BehaviorAnalyzer:
    calibration = calibration_for(
        matrix,
        CALIBRATION_REGION,
        camera_id=camera_id,
        venue_id=venue_id,
        reference_frame=reference_frame,  # type: ignore[arg-type]
    )
    projector = GroundPlaneProjector.from_calibration(calibration, frame_width, frame_height)
    config = BehaviorAnalysisConfig.for_ground_plane_metres()
    if config_overrides:
        config = replace(config, **config_overrides)  # type: ignore[arg-type]
    return BehaviorAnalyzer(camera_id=camera_id, projector=projector, config=config)


def walk_venue(
    analyzer: BehaviorAnalyzer,
    path: VenuePath,
    frames: int,
    matrix: Matrix3x3 = CAMERA_A,
    track_id: int = 1,
    scale: float = 1.0,
) -> list[BehaviorFrame]:
    """Move a synthetic person along a venue-metre path, seen through ``matrix``."""
    results = []
    for index in range(frames):
        timestamp = index * DT
        x, y = path(timestamp)
        results.append(
            analyzer.update(
                [track_at_venue(matrix, x, y, track_id=track_id, scale=scale)],
                frame_index=index + 1,
                timestamp_seconds=timestamp,
            )
        )
    return results


def straight(speed: float = WALKING_SPEED, y: float = 4.0) -> VenuePath:
    """Walk along +x at a constant metres per second."""
    return lambda t: (1.0 + speed * t, y)


def accelerating(rate: float = 0.5, y: float = 4.0) -> VenuePath:
    """Start at rest and accelerate along +x at ``rate`` m/s²."""
    return lambda t: (1.0 + 0.5 * rate * t * t, y)


def standing(x: float = 6.0, y: float = 4.0) -> VenuePath:
    return lambda _t: (x, y)


def corner(turn_time: float = 1.0, speed: float = WALKING_SPEED) -> VenuePath:
    """Walk along +x, then turn 90 degrees onto +y."""

    def path(t: float) -> tuple[float, float]:
        if t < turn_time:
            return (1.0 + speed * t, 2.0)
        return (1.0 + speed * turn_time, 2.0 + speed * (t - turn_time))

    return path


def final(frames: list[BehaviorFrame], signal: SignalType, camera: str = "cam_a") -> float | None:
    return frames[-1].value_for(SubjectRef(camera, 1), signal)


class TestMetricSignals:
    """Every signal arrives in physical units, with no downstream change."""

    def test_speed_is_metres_per_second(self) -> None:
        frames = walk_venue(ground_plane_analyzer(), straight(WALKING_SPEED), 50)
        assert final(frames, SignalType.SPEED) == pytest.approx(WALKING_SPEED, rel=1e-6)

    @pytest.mark.parametrize("speed", [0.5, 1.4, 3.0])
    def test_speed_recovers_any_pace(self, speed: float) -> None:
        frames = walk_venue(ground_plane_analyzer(), straight(speed), 50)
        assert final(frames, SignalType.SPEED) == pytest.approx(speed, rel=1e-6)

    def test_units_are_reported_as_metric(self) -> None:
        frames = walk_venue(ground_plane_analyzer(), straight(), 50)
        units = {
            signal.signal_type: signal.unit
            for signal in frames[-1].for_subject(SubjectRef("cam_a", 1))
        }
        assert units[SignalType.SPEED] == "m/s"
        assert units[SignalType.ACCELERATION] == "m/s^2"
        assert all(signal.space is MotionSpace.GROUND_PLANE_METRES for signal in frames[-1])

    @pytest.mark.parametrize("rate", [0.3, 0.8])
    def test_acceleration_is_metres_per_second_squared(self, rate: float) -> None:
        frames = walk_venue(ground_plane_analyzer(), accelerating(rate), 50)
        assert final(frames, SignalType.ACCELERATION) == pytest.approx(rate, rel=1e-4)

    def test_constant_speed_has_no_acceleration(self) -> None:
        frames = walk_venue(ground_plane_analyzer(), straight(), 50)
        assert final(frames, SignalType.ACCELERATION) == pytest.approx(0.0, abs=1e-6)

    def test_stationary_duration_still_accumulates(self) -> None:
        frames = walk_venue(ground_plane_analyzer(), standing(), 76)  # 3 s
        duration = final(frames, SignalType.STATIONARY_DURATION)
        assert duration == pytest.approx(3.0, abs=DT)

    def test_a_walking_person_is_not_stationary(self) -> None:
        frames = walk_venue(ground_plane_analyzer(), straight(), 50)
        assert final(frames, SignalType.STATIONARY_DURATION) is None

    def test_heading_is_still_radians_per_second(self) -> None:
        frames = walk_venue(ground_plane_analyzer(), corner(1.0), 60)
        rates = [
            abs(value)
            for frame in frames
            if (value := frame.value_for(SubjectRef("cam_a", 1), SignalType.HEADING_CHANGE_RATE))
            is not None
        ]
        assert rates
        assert max(rates) == pytest.approx(math.pi, rel=0.05)
        heading_signals = frames[-1].of_type(SignalType.HEADING_CHANGE_RATE)
        assert all(signal.unit == "rad/s" for signal in heading_signals)

    def test_a_straight_walk_does_not_turn(self) -> None:
        frames = walk_venue(ground_plane_analyzer(), straight(), 50)
        assert final(frames, SignalType.HEADING_CHANGE_RATE) == pytest.approx(0.0, abs=1e-6)

    def test_speed_is_independent_of_runtime_resolution(self) -> None:
        """Half-resolution decode, same physical speed."""
        full = walk_venue(ground_plane_analyzer(), straight(), 50)
        half = walk_venue(
            ground_plane_analyzer(frame_width=960, frame_height=540),
            straight(),
            50,
            scale=0.5,
        )
        assert final(half, SignalType.SPEED) == pytest.approx(
            final(full, SignalType.SPEED), rel=1e-6
        )

    def test_the_analyzer_reports_the_metric_space(self) -> None:
        assert ground_plane_analyzer().space is MotionSpace.GROUND_PLANE_METRES


class TestMultiCamera:
    """Two cameras, one venue frame — the architecture's scalability claim."""

    @staticmethod
    def observe_from_both(path: VenuePath, frames: int = 50) -> tuple[list[BehaviorFrame], ...]:
        left = ground_plane_analyzer(CAMERA_A, camera_id="cam_a", reference_frame=FRAME_A)
        right = ground_plane_analyzer(CAMERA_B, camera_id="cam_b", reference_frame=FRAME_B)
        # The same physical person, deliberately given the same numeric track id
        # on both cameras — a routine collision that must not merge anything.
        return (
            walk_venue(left, path, frames, matrix=CAMERA_A, track_id=1),
            walk_venue(right, path, frames, matrix=CAMERA_B, track_id=1),
        )

    def test_the_two_cameras_have_genuinely_different_geometry(self) -> None:
        """Guards the test itself: agreement is meaningless if they are alike."""
        a = project_venue_to_image(CAMERA_A, 6.0, 4.0)
        b = project_venue_to_image(CAMERA_B, 6.0, 4.0)
        assert math.dist(a, b) > 200.0, "the synthetic cameras must not be near-identical"

    def test_both_cameras_recover_the_same_venue_position(self) -> None:
        left = GroundPlaneProjector.from_calibration(
            calibration_for(
                CAMERA_A, CALIBRATION_REGION, camera_id="cam_a", reference_frame=FRAME_A
            )
        )
        right = GroundPlaneProjector.from_calibration(
            calibration_for(
                CAMERA_B, CALIBRATION_REGION, camera_id="cam_b", reference_frame=FRAME_B
            )
        )
        for venue in [(3.0, 2.0), (7.5, 5.5), (11.0, 8.0)]:
            left_result = left.project(Point(*project_venue_to_image(CAMERA_A, *venue)))
            right_result = right.project(Point(*project_venue_to_image(CAMERA_B, *venue)))
            assert (left_result.x, left_result.y) == pytest.approx(venue, abs=1e-7)
            assert (right_result.x, right_result.y) == pytest.approx(venue, abs=1e-7)
            assert (left_result.x, left_result.y) == pytest.approx(
                (right_result.x, right_result.y), abs=1e-6
            )

    def test_both_cameras_report_the_same_speed(self) -> None:
        left, right = self.observe_from_both(straight(WALKING_SPEED))
        left_speed = final(left, SignalType.SPEED, "cam_a")
        right_speed = final(right, SignalType.SPEED, "cam_b")
        assert left_speed == pytest.approx(WALKING_SPEED, rel=1e-6)
        assert right_speed == pytest.approx(WALKING_SPEED, rel=1e-6)
        assert left_speed == pytest.approx(right_speed, rel=1e-6)

    def test_both_cameras_report_the_same_acceleration(self) -> None:
        left, right = self.observe_from_both(accelerating(0.6))
        assert final(left, SignalType.ACCELERATION, "cam_a") == pytest.approx(
            final(right, SignalType.ACCELERATION, "cam_b"), rel=1e-4
        )

    def test_both_cameras_agree_a_person_is_stationary(self) -> None:
        left, right = self.observe_from_both(standing(), frames=76)
        assert final(left, SignalType.STATIONARY_DURATION, "cam_a") == pytest.approx(
            final(right, SignalType.STATIONARY_DURATION, "cam_b"), abs=1e-6
        )

    def test_the_same_track_id_on_two_cameras_stays_two_subjects(self) -> None:
        """Spatial agreement is not identity, and must never be treated as it."""
        left, right = self.observe_from_both(straight())

        left_subjects = left[-1].subjects
        right_subjects = right[-1].subjects
        assert left_subjects == (SubjectRef("cam_a", 1),)
        assert right_subjects == (SubjectRef("cam_b", 1),)
        assert left_subjects != right_subjects
        assert set(left_subjects).isdisjoint(right_subjects)

        # Neither analyzer knows anything about the other camera's subject.
        assert left[-1].value_for(SubjectRef("cam_b", 1), SignalType.SPEED) is None
        assert right[-1].value_for(SubjectRef("cam_a", 1), SignalType.SPEED) is None

    def test_cameras_declare_a_shared_venue_frame(self) -> None:
        left = GroundPlaneProjector.from_calibration(
            calibration_for(CAMERA_A, CALIBRATION_REGION, camera_id="cam_a")
        )
        right = GroundPlaneProjector.from_calibration(
            calibration_for(CAMERA_B, CALIBRATION_REGION, camera_id="cam_b")
        )
        assert left.shares_venue_with(right)
        assert left.venue_id == right.venue_id

    def test_different_venues_do_not_claim_comparability(self) -> None:
        left = GroundPlaneProjector.from_calibration(
            calibration_for(CAMERA_A, CALIBRATION_REGION, camera_id="cam_a", venue_id="club")
        )
        right = GroundPlaneProjector.from_calibration(
            calibration_for(CAMERA_B, CALIBRATION_REGION, camera_id="cam_b", venue_id="tent")
        )
        assert not left.shares_venue_with(right)

    def test_cameras_run_at_their_own_resolutions(self) -> None:
        """Nothing requires two cameras to share a resolution or a lens."""
        left, right = self.observe_from_both(straight())
        assert FRAME_A.width != FRAME_B.width
        assert final(left, SignalType.SPEED, "cam_a") == pytest.approx(
            final(right, SignalType.SPEED, "cam_b"), rel=1e-6
        )


class TestPerspective:
    """Why V0.3.1 exists: image displacement is not physical displacement."""

    NEAR_Y = 0.5
    FAR_Y = 8.5

    @staticmethod
    def one_metre_walk(y: float) -> VenuePath:
        """Walk exactly one metre along +x, in one second, at depth ``y``."""
        return lambda t: (5.0 + 1.0 * min(t, 1.0), y)

    def normalized_speed_at(self, y: float) -> float:
        analyzer = BehaviorAnalyzer(
            camera_id="cam_a",
            projector=NormalizedFrameProjector(FRAME_A.width, FRAME_A.height),
        )
        frames = walk_venue(analyzer, self.one_metre_walk(y), 26, matrix=CAMERA_STRONG_PERSPECTIVE)
        speed = final(frames, SignalType.SPEED)
        assert speed is not None
        return speed

    def metric_speed_at(self, y: float) -> float:
        analyzer = ground_plane_analyzer(CAMERA_STRONG_PERSPECTIVE, camera_id="cam_a")
        frames = walk_venue(analyzer, self.one_metre_walk(y), 26, matrix=CAMERA_STRONG_PERSPECTIVE)
        speed = final(frames, SignalType.SPEED)
        assert speed is not None
        return speed

    def test_image_space_misreports_equal_physical_movement(self) -> None:
        """The same metre, at two depths, measured very differently."""
        near = self.normalized_speed_at(self.NEAR_Y)
        far = self.normalized_speed_at(self.FAR_Y)
        assert near > far
        assert near / far > 1.5, (
            "the synthetic camera must show real perspective for this test to mean anything"
        )

    def test_calibration_recovers_the_same_metre(self) -> None:
        near = self.metric_speed_at(self.NEAR_Y)
        far = self.metric_speed_at(self.FAR_Y)
        assert near == pytest.approx(1.0, rel=1e-6)
        assert far == pytest.approx(1.0, rel=1e-6)
        assert near == pytest.approx(far, rel=1e-6)

    def test_the_correction_is_large(self) -> None:
        """Quantifies what image-space measurement would have got wrong."""
        image_ratio = self.normalized_speed_at(self.NEAR_Y) / self.normalized_speed_at(self.FAR_Y)
        metric_ratio = self.metric_speed_at(self.NEAR_Y) / self.metric_speed_at(self.FAR_Y)
        assert image_ratio > 1.5
        assert metric_ratio == pytest.approx(1.0, rel=1e-6)


class TestNormalizedBehaviourUnchanged:
    """V0.3.0 must behave exactly as before when no calibration is used."""

    def test_normalized_projector_still_reports_frame_units(self) -> None:
        analyzer = BehaviorAnalyzer(
            camera_id="cam_a",
            projector=NormalizedFrameProjector(FRAME_A.width, FRAME_A.height),
        )
        frames = walk_venue(analyzer, straight(), 50)
        signals = frames[-1].of_type(SignalType.SPEED)
        assert signals[0].unit == "normalized_frame_unit/s"
        assert signals[0].space is MotionSpace.NORMALIZED_IMAGE

    def test_default_config_declares_no_space_and_is_unrestricted(self) -> None:
        assert BehaviorAnalysisConfig().intended_space is None

    def test_default_config_still_works_with_a_normalized_projector(self) -> None:
        analyzer = BehaviorAnalyzer(
            camera_id="cam_a",
            projector=NormalizedFrameProjector(640, 360),
            config=BehaviorAnalysisConfig(),
        )
        assert analyzer.space is MotionSpace.NORMALIZED_IMAGE


class TestUnitSafety:
    """Mixing spaces and thresholds is silent and severe, so it is blocked."""

    def test_metric_config_rejects_a_normalized_projector(self) -> None:
        with pytest.raises(ConfigurationError, match="written for ground_plane_metres"):
            BehaviorAnalyzer(
                camera_id="cam_a",
                projector=NormalizedFrameProjector(1920, 1080),
                config=BehaviorAnalysisConfig.for_ground_plane_metres(),
            )

    def test_normalized_config_rejects_a_calibrated_projector(self) -> None:
        projector = GroundPlaneProjector.from_calibration(
            calibration_for(CAMERA_A, CALIBRATION_REGION, camera_id="cam_a")
        )
        declared = BehaviorAnalysisConfig(intended_space=MotionSpace.NORMALIZED_IMAGE)
        with pytest.raises(ConfigurationError, match="written for normalized_image"):
            BehaviorAnalyzer(camera_id="cam_a", projector=projector, config=declared)

    def test_an_undeclared_config_is_allowed_anywhere(self) -> None:
        """V0.3.0 configs keep working; the check is opt-in."""
        projector = GroundPlaneProjector.from_calibration(
            calibration_for(CAMERA_A, CALIBRATION_REGION, camera_id="cam_a")
        )
        analyzer = BehaviorAnalyzer(
            camera_id="cam_a", projector=projector, config=BehaviorAnalysisConfig()
        )
        assert analyzer.space is MotionSpace.GROUND_PLANE_METRES

    def test_metric_thresholds_differ_from_normalized_ones(self) -> None:
        metric = BehaviorAnalysisConfig.for_ground_plane_metres()
        normalized = BehaviorAnalysisConfig()
        assert metric.stationary_speed_threshold > normalized.stationary_speed_threshold
        assert metric.intended_space is MotionSpace.GROUND_PLANE_METRES

    def test_metric_defaults_separate_standing_from_walking(self) -> None:
        """A sanity check that the development values are at least the right size."""
        threshold = BehaviorAnalysisConfig.for_ground_plane_metres().stationary_speed_threshold
        assert threshold < WALKING_SPEED
        assert threshold > 0.05


class TestUnprojectableAnchors:
    def test_a_track_above_the_horizon_is_skipped_not_fatal(self) -> None:
        """One impossible detection must not end a run of thousands of frames."""
        analyzer = ground_plane_analyzer()
        # A box whose feet sit far above the calibrated floor, past the horizon.
        impossible = Track(track_id=99, bbox_xyxy=BoundingBox(900.0, -6000.0, 940.0, -5000.0))
        frame = analyzer.update([impossible], frame_index=1, timestamp_seconds=0.0)

        assert len(frame) == 0
        assert analyzer.unprojectable_observations == 1
        assert analyzer.tracked_subjects == 0

    def test_other_tracks_in_the_same_frame_still_measure(self) -> None:
        analyzer = ground_plane_analyzer()
        impossible = Track(track_id=99, bbox_xyxy=BoundingBox(900.0, -6000.0, 940.0, -5000.0))
        last: BehaviorFrame | None = None
        for index in range(50):
            timestamp = index * DT
            x, y = straight()(timestamp)
            last = analyzer.update(
                [track_at_venue(CAMERA_A, x, y, track_id=1), impossible],
                frame_index=index + 1,
                timestamp_seconds=timestamp,
            )
        assert last is not None
        assert last.value_for(SubjectRef("cam_a", 1), SignalType.SPEED) == pytest.approx(
            WALKING_SPEED, rel=1e-6
        )
        assert analyzer.unprojectable_observations == 50

    def test_normalized_projection_never_fails(self) -> None:
        analyzer = BehaviorAnalyzer(
            camera_id="cam_a", projector=NormalizedFrameProjector(1920, 1080)
        )
        walk_venue(analyzer, straight(), 10)
        assert analyzer.unprojectable_observations == 0


class TestDeterminism:
    def test_metric_runs_are_reproducible(self) -> None:
        def records() -> list[dict[str, object]]:
            frames = walk_venue(ground_plane_analyzer(), accelerating(0.4), 50)
            return [record for frame in frames for record in frame.to_records()]

        assert records() == records()

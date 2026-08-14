"""The behaviour analyzer end to end, driven by synthetic tracks.

The centrepiece is :class:`TestResolutionInvariance`: the same real trajectory
filmed at 384x288 and at 1920x1080 must produce the same signals. That is the
property the coordinate abstraction exists to guarantee, and it is what stops
behaviour logic from silently depending on pixels.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import pytest

from atreides_vision.behavior.analyzer import BehaviorAnalyzer
from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.behavior.coordinates import (
    MotionPoint,
    MotionSpace,
    NormalizedFrameProjector,
)
from atreides_vision.behavior.models import BehaviorFrame, SignalType, SubjectRef
from atreides_vision.models import BoundingBox, Point, Track

FPS = 25.0
DT = 1.0 / FPS
CAMERA = "cam_01"

#: A person's apparent size as a fraction of the frame. Held constant across
#: resolutions so the two renderings really are the same scene.
BOX_W_NORM = 0.05
BOX_H_NORM = 0.20


def config(**overrides: object) -> BehaviorAnalysisConfig:
    return BehaviorAnalysisConfig(**overrides)  # type: ignore[arg-type]


def analyzer(
    width: int = 640,
    height: int = 360,
    camera_id: str = CAMERA,
    **overrides: object,
) -> BehaviorAnalyzer:
    return BehaviorAnalyzer(
        camera_id=camera_id,
        projector=NormalizedFrameProjector(width, height),
        config=config(**overrides),
    )


def track_at(
    x_norm: float,
    y_norm: float,
    width: int,
    height: int,
    track_id: int = 1,
) -> Track:
    """A track whose bottom-centre lands exactly on ``(x_norm, y_norm)``."""
    half_w = BOX_W_NORM * width / 2.0
    box_h = BOX_H_NORM * height
    cx, feet = x_norm * width, y_norm * height
    return Track(
        track_id=track_id,
        bbox_xyxy=BoundingBox(cx - half_w, feet - box_h, cx + half_w, feet),
        confidence=0.9,
    )


def run(
    subject_analyzer: BehaviorAnalyzer,
    path: Callable[[float], tuple[float, float]],
    frames: int,
    width: int = 640,
    height: int = 360,
    track_id: int = 1,
) -> list[BehaviorFrame]:
    """Feed a normalized trajectory through the analyzer, one frame at a time."""
    results = []
    for index in range(frames):
        frame_index = index + 1
        timestamp = index * DT
        x_norm, y_norm = path(timestamp)
        results.append(
            subject_analyzer.update(
                [track_at(x_norm, y_norm, width, height, track_id)],
                frame_index=frame_index,
                timestamp_seconds=timestamp,
            )
        )
    return results


def still(x: float = 0.5, y: float = 0.5) -> Callable[[float], tuple[float, float]]:
    return lambda _t: (x, y)


def walking(speed: float = 0.2) -> Callable[[float], tuple[float, float]]:
    return lambda t: (0.1 + speed * t, 0.5)


def accelerating(rate: float = 0.2) -> Callable[[float], tuple[float, float]]:
    return lambda t: (0.1 + 0.5 * rate * t * t, 0.5)


def corner_at(turn_time: float, speed: float = 0.2) -> Callable[[float], tuple[float, float]]:
    """Walk along +x, then turn 90 degrees onto +y at ``turn_time``."""

    def path(t: float) -> tuple[float, float]:
        if t < turn_time:
            return (0.2 + speed * t, 0.5)
        return (0.2 + speed * turn_time, 0.5 + speed * (t - turn_time))

    return path


def final_value(frames: Sequence[BehaviorFrame], signal_type: SignalType) -> float | None:
    return frames[-1].value_for(SubjectRef(CAMERA, 1), signal_type)


class TestConstruction:
    @pytest.mark.parametrize("camera_id", ["", "  "])
    def test_requires_a_camera_id(self, camera_id: str) -> None:
        with pytest.raises(ValueError, match="non-empty camera_id"):
            BehaviorAnalyzer(camera_id, NormalizedFrameProjector(640, 360))

    def test_reports_its_coordinate_space(self) -> None:
        assert analyzer().space is MotionSpace.NORMALIZED_IMAGE

    def test_starts_empty(self) -> None:
        subject_analyzer = analyzer()
        assert subject_analyzer.latest_frame is None
        assert subject_analyzer.signals_emitted == 0
        assert subject_analyzer.tracked_subjects == 0

    def test_accepts_a_custom_anchor(self) -> None:
        """The anchor is injectable for cameras where feet are the wrong point."""
        subject_analyzer = BehaviorAnalyzer(
            CAMERA,
            NormalizedFrameProjector(640, 360),
            anchor=lambda bbox: Point(bbox.x1, bbox.y1),
        )
        frame = subject_analyzer.update(
            [Track(1, BoundingBox(64.0, 36.0, 128.0, 72.0))],
            frame_index=1,
            timestamp_seconds=0.0,
        )
        assert frame.frame_index == 1


class TestEmptyAndBasicFrames:
    def test_no_tracks_yields_no_signals(self) -> None:
        frame = analyzer().update([], frame_index=1, timestamp_seconds=0.0)
        assert len(frame) == 0
        assert frame.frame_index == 1

    def test_first_frame_yields_no_signals(self) -> None:
        """One observation is not motion."""
        frames = run(analyzer(), walking(), 1)
        assert len(frames[0]) == 0

    def test_speed_appears_once_there_is_history(self) -> None:
        frames = run(analyzer(), walking(0.2), 2)
        assert final_value(frames, SignalType.SPEED) == pytest.approx(0.2)

    def test_signals_carry_camera_local_subject(self) -> None:
        frames = run(analyzer(), walking(), 10)
        for signal in frames[-1]:
            assert signal.subject == SubjectRef(CAMERA, 1)
            assert signal.space is MotionSpace.NORMALIZED_IMAGE
            assert signal.frame_index == 10

    def test_units_are_normalized_not_metric(self) -> None:
        frames = run(analyzer(), walking(), 30)
        speed = frames[-1].for_subject(SubjectRef(CAMERA, 1))
        units = {signal.signal_type: signal.unit for signal in speed}
        assert units[SignalType.SPEED] == "normalized_frame_unit/s"


class TestSpeedSignal:
    def test_zero_movement_reports_zero_speed(self) -> None:
        frames = run(analyzer(), still(), 30)
        assert final_value(frames, SignalType.SPEED) == pytest.approx(0.0)

    @pytest.mark.parametrize("speed", [0.05, 0.2, 0.6])
    def test_constant_velocity_is_recovered(self, speed: float) -> None:
        frames = run(analyzer(), walking(speed), 40)
        assert final_value(frames, SignalType.SPEED) == pytest.approx(speed, rel=1e-9)

    def test_quality_grows_as_history_accumulates(self) -> None:
        """A freshly acquired track's estimate rests on very little."""
        frames = run(analyzer(speed_window_seconds=1.0), walking(), 40)

        def quality(frame: BehaviorFrame) -> float | None:
            for signal in frame.of_type(SignalType.SPEED):
                return signal.quality
            return None

        early, late = quality(frames[2]), quality(frames[-1])
        assert early is not None and late is not None
        assert early < late == pytest.approx(1.0)

    def test_speed_is_frame_rate_independent(self) -> None:
        """The same motion at 10 and 50 FPS reports the same speed."""

        def measure(fps: float) -> float | None:
            subject_analyzer = analyzer()
            frames = int(2 * fps)
            last: BehaviorFrame | None = None
            for index in range(frames):
                timestamp = index / fps
                last = subject_analyzer.update(
                    [track_at(0.1 + 0.2 * timestamp, 0.5, 640, 360)],
                    frame_index=index + 1,
                    timestamp_seconds=timestamp,
                )
            assert last is not None
            return last.value_for(SubjectRef(CAMERA, 1), SignalType.SPEED)

        assert measure(10.0) == pytest.approx(measure(50.0), rel=1e-9)


class TestAccelerationSignal:
    def test_constant_velocity_has_no_acceleration(self) -> None:
        frames = run(analyzer(), walking(0.2), 40)
        assert final_value(frames, SignalType.ACCELERATION) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("rate", [0.1, 0.4])
    def test_constant_acceleration_is_recovered(self, rate: float) -> None:
        frames = run(analyzer(), accelerating(rate), 40)
        assert final_value(frames, SignalType.ACCELERATION) == pytest.approx(rate, rel=1e-6)

    def test_not_emitted_before_both_halves_have_data(self) -> None:
        frames = run(analyzer(), walking(), 3)
        assert final_value(frames, SignalType.ACCELERATION) is None


class TestHeadingSignal:
    def test_straight_line_has_no_turn(self) -> None:
        frames = run(analyzer(), walking(0.2), 40)
        assert final_value(frames, SignalType.HEADING_CHANGE_RATE) == pytest.approx(0.0, abs=1e-9)

    def test_a_standing_person_produces_no_heading_signal(self) -> None:
        """Direction is undefined at rest — jitter must not become a turn."""
        frames = run(analyzer(), still(), 40)
        assert final_value(frames, SignalType.HEADING_CHANGE_RATE) is None

    def test_a_right_angle_turn_is_measured(self) -> None:
        """Walk +x for a second, then +y for a second.

        The peak turn rate occurs when the corner sits at the centre of the
        window: each half then holds one clean leg, giving a 90 deg turn across
        the 0.5 s between the half centres — pi rad/s.
        """
        frames = run(analyzer(), corner_at(1.0), 60)
        rates = [
            abs(value)
            for frame in frames
            if (value := final_value([frame], SignalType.HEADING_CHANGE_RATE)) is not None
        ]
        assert rates, "a right-angle turn must produce a heading signal"
        assert max(rates) == pytest.approx(math.pi, rel=0.05)

    def test_the_turn_rate_decays_once_the_corner_leaves_the_window(self) -> None:
        """The signal is a rate over a window, not a permanent flag."""
        frames = run(analyzer(), corner_at(1.0), 60)
        assert final_value(frames, SignalType.HEADING_CHANGE_RATE) == pytest.approx(
            0.0, abs=math.radians(5)
        )


class TestStationaryDuration:
    def test_accumulates_while_still(self) -> None:
        frames = run(analyzer(), still(), 76)  # 3 s at 25 FPS
        duration = final_value(frames, SignalType.STATIONARY_DURATION)
        assert duration == pytest.approx(3.0, abs=DT)

    def test_dates_from_the_start_of_the_evidence(self) -> None:
        """Not from the frame the spell was confirmed — the window counts."""
        frames = run(analyzer(), still(), 26)  # 1 s
        assert final_value(frames, SignalType.STATIONARY_DURATION) == pytest.approx(1.0)

    def test_a_moving_person_is_not_stationary(self) -> None:
        frames = run(analyzer(), walking(0.2), 40)
        assert final_value(frames, SignalType.STATIONARY_DURATION) is None

    def test_movement_resets_the_spell(self) -> None:
        def stand_then_walk(t: float) -> tuple[float, float]:
            return (0.3, 0.5) if t < 2.0 else (0.3 + 0.3 * (t - 2.0), 0.5)

        frames = run(analyzer(), stand_then_walk, 100)  # 4 s
        assert final_value(frames, SignalType.STATIONARY_DURATION) is None

    def test_outlives_the_retained_history_window(self) -> None:
        """A person can stand still far longer than the history window."""
        frames = run(analyzer(history_window_seconds=2.0), still(), 251)  # 10 s
        duration = final_value(frames, SignalType.STATIONARY_DURATION)
        assert duration is not None
        assert duration > 5.0

    def test_threshold_is_configurable(self) -> None:
        """A slow drift counts as stationary only under a loose threshold."""
        strict = run(analyzer(stationary_speed_threshold=0.001), walking(0.01), 40)
        loose = run(analyzer(stationary_speed_threshold=0.1), walking(0.01), 40)
        assert final_value(strict, SignalType.STATIONARY_DURATION) is None
        assert final_value(loose, SignalType.STATIONARY_DURATION) is not None


class TestResolutionInvariance:
    """The property the whole coordinate abstraction exists to provide."""

    LOW = (384, 288)
    HIGH = (1920, 1080)

    @staticmethod
    def signals_at(
        width: int,
        height: int,
        path: Callable[[float], tuple[float, float]],
        frames: int = 50,
    ) -> dict[SignalType, float]:
        results = run(analyzer(width, height), path, frames, width=width, height=height)
        subject = SubjectRef(CAMERA, 1)
        return {signal.signal_type: signal.value for signal in results[-1].for_subject(subject)}

    @pytest.mark.parametrize(
        ("name", "path"),
        [
            ("walking", walking(0.2)),
            ("accelerating", accelerating(0.3)),
            ("still", still()),
        ],
    )
    def test_same_trajectory_gives_same_signals_at_both_resolutions(
        self, name: str, path: Callable[[float], tuple[float, float]]
    ) -> None:
        low = self.signals_at(*self.LOW, path)
        high = self.signals_at(*self.HIGH, path)

        assert low.keys() == high.keys(), f"{name}: different signals emitted"
        assert low, f"{name}: no signals to compare"
        for signal_type, value in low.items():
            assert value == pytest.approx(high[signal_type], rel=1e-9, abs=1e-12), (
                f"{name}: {signal_type.value} differs between resolutions"
            )

    def test_a_turn_measures_the_same_at_both_resolutions(self) -> None:
        low = self.signals_at(*self.LOW, corner_at(1.6), frames=45)
        high = self.signals_at(*self.HIGH, corner_at(1.6), frames=45)
        assert low[SignalType.HEADING_CHANGE_RATE] == pytest.approx(
            high[SignalType.HEADING_CHANGE_RATE], rel=1e-9
        )
        assert abs(low[SignalType.HEADING_CHANGE_RATE]) > 0.1, "the turn must be visible"

    def test_raw_pixel_speed_would_have_differed_enormously(self) -> None:
        """Why normalization is not optional.

        The same walk covers 5x as many pixels at 1920 wide as at 384 wide.
        A pixel-based speed would report the same person as five times faster.
        """
        low_px = 0.2 * self.LOW[0]
        high_px = 0.2 * self.HIGH[0]
        assert high_px == pytest.approx(5.0 * low_px)

        low = self.signals_at(*self.LOW, walking(0.2))
        high = self.signals_at(*self.HIGH, walking(0.2))
        assert low[SignalType.SPEED] == pytest.approx(high[SignalType.SPEED], rel=1e-9)


class TestMissingObservations:
    def test_a_short_gap_is_bridged(self) -> None:
        """A dropped detection or two must not destroy the estimate."""
        subject_analyzer = analyzer()
        last: BehaviorFrame | None = None
        for index in range(40):
            if index in (12, 13):  # detector misses two frames
                continue
            timestamp = index * DT
            last = subject_analyzer.update(
                [track_at(0.1 + 0.2 * timestamp, 0.5, 640, 360)],
                frame_index=index + 1,
                timestamp_seconds=timestamp,
            )
        assert last is not None
        assert last.value_for(SubjectRef(CAMERA, 1), SignalType.SPEED) == pytest.approx(
            0.2, rel=1e-9
        )

    def test_a_long_gap_restarts_measurement(self) -> None:
        """Motion across an unobserved second is unknown, not slow."""
        subject_analyzer = analyzer(max_observation_gap_seconds=0.5)
        subject_analyzer.update(
            [track_at(0.1, 0.5, 640, 360)], frame_index=1, timestamp_seconds=0.0
        )
        subject_analyzer.update(
            [track_at(0.15, 0.5, 640, 360)], frame_index=2, timestamp_seconds=DT
        )
        # Reappears far away, three seconds later.
        frame = subject_analyzer.update(
            [track_at(0.9, 0.5, 640, 360)], frame_index=80, timestamp_seconds=3.16
        )
        assert len(frame) == 0, "no signal may be invented across an unobserved gap"

    def test_a_track_that_vanishes_expires(self) -> None:
        subject_analyzer = analyzer(stale_subject_ttl_frames=10)
        run(subject_analyzer, walking(), 5)
        assert subject_analyzer.tracked_subjects == 1

        for index in range(5, 40):
            subject_analyzer.update([], frame_index=index + 1, timestamp_seconds=index * DT)
        assert subject_analyzer.tracked_subjects == 0

    def test_long_runs_stay_memory_bounded(self) -> None:
        """A thousand distinct ids must not accumulate."""
        subject_analyzer = analyzer(stale_subject_ttl_frames=5)
        for track_id in range(1, 1001):
            subject_analyzer.update(
                [track_at(0.5, 0.5, 640, 360, track_id=track_id)],
                frame_index=track_id,
                timestamp_seconds=track_id * DT,
            )
        assert subject_analyzer.tracked_subjects <= 5


class TestSubjectIsolation:
    def test_two_tracks_do_not_leak_state(self) -> None:
        """One walking, one standing, in the same frames."""
        subject_analyzer = analyzer()
        last: BehaviorFrame | None = None
        for index in range(40):
            timestamp = index * DT
            last = subject_analyzer.update(
                [
                    track_at(0.1 + 0.2 * timestamp, 0.5, 640, 360, track_id=1),
                    track_at(0.8, 0.7, 640, 360, track_id=2),
                ],
                frame_index=index + 1,
                timestamp_seconds=timestamp,
            )
        assert last is not None
        walker, stander = SubjectRef(CAMERA, 1), SubjectRef(CAMERA, 2)
        assert last.value_for(walker, SignalType.SPEED) == pytest.approx(0.2, rel=1e-9)
        assert last.value_for(stander, SignalType.SPEED) == pytest.approx(0.0, abs=1e-12)
        assert last.value_for(stander, SignalType.STATIONARY_DURATION) is not None
        assert last.value_for(walker, SignalType.STATIONARY_DURATION) is None

    def test_two_cameras_with_the_same_track_id_stay_independent(self) -> None:
        """The failure this design exists to prevent."""
        left = analyzer(camera_id="cam_left")
        right = analyzer(camera_id="cam_right")

        left_frames = run(left, walking(0.3), 40, track_id=7)
        right_frames = run(right, still(), 40, track_id=7)

        left_speed = left_frames[-1].value_for(SubjectRef("cam_left", 7), SignalType.SPEED)
        right_speed = right_frames[-1].value_for(SubjectRef("cam_right", 7), SignalType.SPEED)
        assert left_speed == pytest.approx(0.3, rel=1e-9)
        assert right_speed == pytest.approx(0.0)

        # Neither analyzer knows anything about the other camera's subject.
        assert left_frames[-1].value_for(SubjectRef("cam_right", 7), SignalType.SPEED) is None
        assert right_frames[-1].value_for(SubjectRef("cam_left", 7), SignalType.SPEED) is None


class TestDeterminism:
    def test_two_identical_runs_produce_identical_signals(self) -> None:
        def records() -> list[dict[str, object]]:
            frames = run(analyzer(), accelerating(0.3), 40)
            return [record for frame in frames for record in frame.to_records()]

        assert records() == records()

    def test_signal_order_is_stable(self) -> None:
        frames = run(analyzer(), accelerating(0.3), 40)
        types = [signal.signal_type for signal in frames[-1]]
        assert types == sorted(types, key=types.index)  # no reordering between runs
        assert len(set(types)) == len(types)


class TestSignalCounting:
    def test_counts_accumulate(self) -> None:
        subject_analyzer = analyzer()
        frames = run(subject_analyzer, walking(), 20)
        assert subject_analyzer.signals_emitted == sum(len(frame) for frame in frames)

    def test_latest_frame_is_exposed(self) -> None:
        subject_analyzer = analyzer()
        frames = run(subject_analyzer, walking(), 10)
        assert subject_analyzer.latest_frame is frames[-1]


class TestZoneContext:
    def test_zone_ids_are_copied_onto_signals(self) -> None:
        subject_analyzer = analyzer()
        last: BehaviorFrame | None = None
        for index in range(10):
            timestamp = index * DT
            last = subject_analyzer.update(
                [track_at(0.1 + 0.2 * timestamp, 0.5, 640, 360)],
                frame_index=index + 1,
                timestamp_seconds=timestamp,
                zone_ids_by_track={1: ["dancefloor", "bar"]},
            )
        assert last is not None
        assert all(signal.zone_ids == ("dancefloor", "bar") for signal in last)

    def test_zone_context_is_optional(self) -> None:
        """Behaviour analysis works with no zones configured at all."""
        frames = run(analyzer(), walking(), 10)
        assert all(signal.zone_ids == () for signal in frames[-1])

    def test_unlisted_tracks_get_no_zone_context(self) -> None:
        subject_analyzer = analyzer()
        subject_analyzer.update(
            [track_at(0.1, 0.5, 640, 360)], frame_index=1, timestamp_seconds=0.0
        )
        frame = subject_analyzer.update(
            [track_at(0.2, 0.5, 640, 360)],
            frame_index=2,
            timestamp_seconds=DT,
            zone_ids_by_track={99: ["elsewhere"]},
        )
        assert all(signal.zone_ids == () for signal in frame)


class TestProjectorSubstitution:
    def test_a_different_projector_changes_units_not_logic(self) -> None:
        """The claim the abstraction makes: swap the projector, keep the maths.

        A stand-in metric projector (a stub, not calibration) proves the
        signals report metres with no change to any kinematics code.
        """

        class MetreProjector:
            """Pretends 1 pixel is 1 cm. Not calibration — a test double."""

            @property
            def space(self) -> MotionSpace:
                return MotionSpace.GROUND_PLANE_METRES

            def project(self, point: Point) -> MotionPoint:
                return MotionPoint(
                    point.x / 100.0, point.y / 100.0, MotionSpace.GROUND_PLANE_METRES
                )

        subject_analyzer = BehaviorAnalyzer(CAMERA, MetreProjector(), config())
        last: BehaviorFrame | None = None
        for index in range(30):
            timestamp = index * DT
            last = subject_analyzer.update(
                [
                    Track(
                        1,
                        BoundingBox(100.0 * timestamp, 0.0, 100.0 * timestamp + 20.0, 180.0),
                    )
                ],
                frame_index=index + 1,
                timestamp_seconds=timestamp,
            )

        assert last is not None
        speed = last.of_type(SignalType.SPEED)[0]
        assert speed.space is MotionSpace.GROUND_PLANE_METRES
        assert speed.unit == "m/s"
        assert speed.value == pytest.approx(1.0)  # 100 px/s at 1 cm/px = 1 m/s

"""The interaction analyzer end to end, driven by synthetic behaviour frames.

The centrepiece is :class:`TestClosingSpeedScenarios`: five physically-stated
situations whose closing speed is known by arithmetic. If the pairwise maths is
wrong, those numbers move.

:class:`TestMultiCameraBoundary` is the architectural test — cross-camera
subjects must never become a pair.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.behavior.models import (
    BehaviorFrame,
    BehaviorSubjectSnapshot,
    SubjectMotion,
    SubjectRef,
)
from atreides_vision.errors import ConfigurationError, InteractionError
from atreides_vision.interaction.analyzer import InteractionAnalyzer
from atreides_vision.interaction.config import InteractionAnalysisConfig
from atreides_vision.interaction.models import (
    InteractionFrame,
    InteractionSignalType,
    SpatialContext,
    SubjectPairRef,
)

METRIC = MotionSpace.GROUND_PLANE_METRES
NORMALIZED = MotionSpace.NORMALIZED_IMAGE
FPS = 25.0
DT = 1.0 / FPS
CAMERA = "cam_01"

#: ``(x, y, dx, dy)`` at a moment: position plus recent motion.
Path = Callable[[float], tuple[float, float, float, float]]


def metric_context(camera_id: str = CAMERA) -> SpatialContext:
    return SpatialContext(camera_id, METRIC, "venue_01", "main_floor")


def analyzer(context: SpatialContext | None = None, **overrides: object) -> InteractionAnalyzer:
    config = InteractionAnalysisConfig.for_ground_plane_metres()
    if overrides:
        from dataclasses import replace

        config = replace(config, **overrides)  # type: ignore[arg-type]
    return InteractionAnalyzer(context or metric_context(), config)


def snapshot(
    track_id: int,
    x: float,
    y: float,
    dx: float = 0.0,
    dy: float = 0.0,
    frame_index: int = 1,
    timestamp: float = 0.0,
    camera_id: str = CAMERA,
    space: MotionSpace = METRIC,
    zone_ids: tuple[str, ...] = (),
) -> BehaviorSubjectSnapshot:
    return BehaviorSubjectSnapshot(
        subject=SubjectRef(camera_id, track_id),
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        position=MotionPoint(x, y, space),
        zone_ids=zone_ids,
        motion=SubjectMotion(dx, dy, 1.0, space),
    )


def run(
    instance: InteractionAnalyzer,
    paths: dict[int, Path],
    frames: int = 40,
) -> list[InteractionFrame]:
    """Drive several subjects along known paths and collect every frame.

    Subjects are stamped with the analyzer's own camera, since it rejects
    anything else — which is the point of the boundary, not an inconvenience.
    """
    results = []
    for index in range(frames):
        timestamp = index * DT
        snapshots = [
            snapshot(
                track_id,
                *path(timestamp),
                frame_index=index + 1,
                timestamp=timestamp,
                camera_id=instance.camera_id,
            )
            for track_id, path in paths.items()
        ]
        results.append(
            instance.update(BehaviorFrame.from_snapshots(index + 1, timestamp, snapshots))
        )
    return results


def still(x: float, y: float = 0.0) -> Path:
    return lambda _t: (x, y, 0.0, 0.0)


def moving(start: float, velocity: float, y: float = 0.0) -> Path:
    """Walks along x at a constant metres per second."""
    return lambda t: (start + velocity * t, y, velocity, 0.0)


def pair_of(first: int, second: int, camera_id: str = CAMERA) -> SubjectPairRef:
    return SubjectPairRef(SubjectRef(camera_id, first), SubjectRef(camera_id, second))


def final_value(
    frames: list[InteractionFrame], signal: InteractionSignalType, first: int = 1, second: int = 2
) -> float | None:
    snapshot_ = frames[-1].for_pair(pair_of(first, second))
    return None if snapshot_ is None else snapshot_.value_for(signal)


class TestConstruction:
    def test_reports_its_context(self) -> None:
        instance = analyzer()
        assert instance.camera_id == CAMERA
        assert instance.space is METRIC

    def test_rejects_a_config_written_for_another_space(self) -> None:
        """Normalized thresholds in metres put proximity 15 cm away."""
        with pytest.raises(ConfigurationError, match="written for normalized_image"):
            InteractionAnalyzer(metric_context(), InteractionAnalysisConfig.for_normalized_image())

    def test_rejects_metric_config_in_normalized_space(self) -> None:
        with pytest.raises(ConfigurationError, match="written for ground_plane_metres"):
            InteractionAnalyzer(
                SpatialContext(CAMERA, NORMALIZED),
                InteractionAnalysisConfig.for_ground_plane_metres(),
            )

    def test_an_undeclared_config_is_allowed_anywhere(self) -> None:
        instance = InteractionAnalyzer(metric_context(), InteractionAnalysisConfig())
        assert instance.space is METRIC

    def test_starts_empty(self) -> None:
        instance = analyzer()
        assert instance.latest_frame is None
        assert instance.tracked_pairs == 0
        assert instance.pairs_analysed == 0


class TestPairGeneration:
    def test_no_pairs_from_a_single_subject(self) -> None:
        frames = run(analyzer(), {1: still(0.0)}, frames=5)
        assert len(frames[-1]) == 0

    def test_no_pairs_from_an_empty_frame(self) -> None:
        instance = analyzer()
        result = instance.update(BehaviorFrame.from_snapshots(1, 0.0, []))
        assert len(result) == 0

    def test_two_subjects_make_one_pair(self) -> None:
        frames = run(analyzer(), {1: still(0.0), 2: still(1.0)}, frames=5)
        assert len(frames[-1]) == 1

    @pytest.mark.parametrize(("subjects", "expected"), [(3, 3), (4, 6), (5, 10), (8, 28)])
    def test_pair_count_is_n_choose_2(self, subjects: int, expected: int) -> None:
        paths = {index: still(index * 0.5) for index in range(1, subjects + 1)}
        frames = run(analyzer(max_pair_distance=None), paths, frames=3)
        assert len(frames[-1]) == expected
        assert frames[-1].candidate_pairs == expected

    def test_no_self_pairs(self) -> None:
        paths = {index: still(index * 0.5) for index in range(1, 5)}
        frames = run(analyzer(max_pair_distance=None), paths, frames=3)
        for snapshot_ in frames[-1]:
            assert snapshot_.pair.first != snapshot_.pair.second

    def test_no_duplicate_pairs(self) -> None:
        paths = {index: still(index * 0.5) for index in range(1, 6)}
        frames = run(analyzer(max_pair_distance=None), paths, frames=3)
        pairs = [snapshot_.pair for snapshot_ in frames[-1]]
        assert len(pairs) == len(set(pairs))

    def test_ordering_is_deterministic(self) -> None:
        paths = {index: still(index * 0.5) for index in range(1, 6)}

        def order() -> list[tuple[int, int]]:
            frames = run(analyzer(max_pair_distance=None), paths, frames=3)
            return [snapshot_.pair.track_ids for snapshot_ in frames[-1]]

        assert order() == order()


class TestDistance:
    def test_known_separation(self) -> None:
        frames = run(analyzer(), {1: still(0.0, 0.0), 2: still(3.0, 4.0)}, frames=5)
        snapshot_ = frames[-1].for_pair(pair_of(1, 2))
        assert snapshot_ is not None
        assert snapshot_.distance == pytest.approx(5.0)

    def test_zero_distance_does_not_merge_subjects(self) -> None:
        """Two tracks at one point stay two subjects."""
        frames = run(analyzer(), {1: still(2.0), 2: still(2.0)}, frames=5)
        snapshot_ = frames[-1].for_pair(pair_of(1, 2))
        assert snapshot_ is not None
        assert snapshot_.distance == 0.0
        assert snapshot_.pair.track_ids == (1, 2)

    def test_distance_is_in_metres(self) -> None:
        frames = run(analyzer(), {1: still(0.0), 2: still(2.5)}, frames=5)
        snapshot_ = frames[-1].for_pair(pair_of(1, 2))
        assert snapshot_ is not None
        assert snapshot_.geometry.distance_unit == "m"


class TestClosingSpeedScenarios:
    """Five physically-stated situations with arithmetic answers."""

    def test_stationary_b_while_a_approaches_at_1_ms(self) -> None:
        frames = run(analyzer(), {1: moving(6.0, -1.0), 2: still(0.0)})
        assert final_value(frames, InteractionSignalType.CLOSING_SPEED) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_stationary_b_while_a_leaves_at_1_ms(self) -> None:
        frames = run(analyzer(), {1: moving(1.0, 1.0), 2: still(0.0)})
        assert final_value(frames, InteractionSignalType.CLOSING_SPEED) == pytest.approx(
            -1.0, abs=1e-6
        )

    def test_both_moving_the_same_way_at_the_same_speed(self) -> None:
        frames = run(analyzer(), {1: moving(0.0, 1.0), 2: moving(3.0, 1.0)})
        assert final_value(frames, InteractionSignalType.CLOSING_SPEED) == pytest.approx(
            0.0, abs=1e-6
        )

    def test_head_on_at_1_ms_each_closes_at_2_ms(self) -> None:
        frames = run(analyzer(), {1: moving(0.0, 1.0), 2: moving(7.0, -1.0)})
        assert final_value(frames, InteractionSignalType.CLOSING_SPEED) == pytest.approx(
            2.0, abs=1e-6
        )

    def test_separating_at_1_ms_each_gives_minus_2_ms(self) -> None:
        frames = run(analyzer(), {1: moving(2.0, -1.0), 2: moving(3.0, 1.0)})
        assert final_value(frames, InteractionSignalType.CLOSING_SPEED) == pytest.approx(
            -2.0, abs=1e-6
        )

    def test_the_unit_is_metres_per_second(self) -> None:
        frames = run(analyzer(), {1: moving(6.0, -1.0), 2: still(0.0)})
        snapshot_ = frames[-1].for_pair(pair_of(1, 2))
        assert snapshot_ is not None
        speed = next(
            signal
            for signal in snapshot_.signals
            if signal.signal_type is InteractionSignalType.CLOSING_SPEED
        )
        assert speed.unit == "m/s"

    def test_it_is_not_the_sum_of_individual_speeds(self) -> None:
        """Two people moving fast in parallel are not closing at all."""
        frames = run(analyzer(), {1: moving(0.0, 3.0), 2: moving(2.0, 3.0)})
        assert final_value(frames, InteractionSignalType.CLOSING_SPEED) == pytest.approx(
            0.0, abs=1e-6
        )


class TestRelativeHeading:
    def test_same_direction(self) -> None:
        frames = run(analyzer(), {1: moving(0.0, 1.0), 2: moving(3.0, 1.0)})
        assert final_value(frames, InteractionSignalType.RELATIVE_MOTION_HEADING) == pytest.approx(
            0.0
        )

    def test_opposite_directions(self) -> None:
        frames = run(analyzer(), {1: moving(0.0, 1.0), 2: moving(7.0, -1.0)})
        assert final_value(frames, InteractionSignalType.RELATIVE_MOTION_HEADING) == pytest.approx(
            math.pi
        )

    def test_perpendicular(self) -> None:
        def northward(start_y: float) -> Path:
            return lambda t: (3.0, start_y + 1.0 * t, 0.0, 1.0)

        frames = run(analyzer(), {1: moving(0.0, 1.0), 2: northward(0.0)})
        assert final_value(frames, InteractionSignalType.RELATIVE_MOTION_HEADING) == pytest.approx(
            math.pi / 2
        )

    def test_a_stationary_subject_omits_the_signal(self) -> None:
        frames = run(analyzer(), {1: moving(6.0, -1.0), 2: still(0.0)})
        assert final_value(frames, InteractionSignalType.RELATIVE_MOTION_HEADING) is None

    def test_the_unit_is_radians(self) -> None:
        frames = run(analyzer(), {1: moving(0.0, 1.0), 2: moving(3.0, 1.0)})
        snapshot_ = frames[-1].for_pair(pair_of(1, 2))
        assert snapshot_ is not None
        heading = next(
            signal
            for signal in snapshot_.signals
            if signal.signal_type is InteractionSignalType.RELATIVE_MOTION_HEADING
        )
        assert heading.unit == "rad"


class TestProximityDuration:
    def test_accumulates_while_close(self) -> None:
        frames = run(analyzer(), {1: still(0.0), 2: still(1.0)}, frames=76)  # 3 s
        assert final_value(frames, InteractionSignalType.PROXIMITY_DURATION) == pytest.approx(
            3.0, abs=DT
        )

    def test_omitted_when_apart(self) -> None:
        frames = run(analyzer(), {1: still(0.0), 2: still(5.0)}, frames=40)
        assert final_value(frames, InteractionSignalType.PROXIMITY_DURATION) is None

    def test_starts_when_they_come_together(self) -> None:
        """Approaching from 5 m, the spell begins only once inside 1.5 m."""
        frames = run(analyzer(), {1: moving(5.0, -1.0), 2: still(0.0)}, frames=125)
        duration = final_value(frames, InteractionSignalType.PROXIMITY_DURATION)
        assert duration is not None
        assert 0.0 < duration < 4.0

    def test_separating_resets_it(self) -> None:
        def together_then_apart(t: float) -> tuple[float, float, float, float]:
            return (1.0, 0.0, 0.0, 0.0) if t < 1.5 else (1.0 + 3.0 * (t - 1.5), 0.0, 3.0, 0.0)

        frames = run(analyzer(), {1: together_then_apart, 2: still(0.0)}, frames=100)
        assert final_value(frames, InteractionSignalType.PROXIMITY_DURATION) is None

    def test_the_unit_is_seconds(self) -> None:
        frames = run(analyzer(), {1: still(0.0), 2: still(1.0)}, frames=30)
        snapshot_ = frames[-1].for_pair(pair_of(1, 2))
        assert snapshot_ is not None
        duration = next(
            signal
            for signal in snapshot_.signals
            if signal.signal_type is InteractionSignalType.PROXIMITY_DURATION
        )
        assert duration.unit == "s"

    def test_it_is_independent_of_frame_rate(self) -> None:
        def measure(fps: float) -> float:
            instance = analyzer()
            frames = int(3 * fps)
            last = None
            for index in range(frames):
                timestamp = index / fps
                last = instance.update(
                    BehaviorFrame.from_snapshots(
                        index + 1,
                        timestamp,
                        [
                            snapshot(1, 0.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                            snapshot(2, 1.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                        ],
                    )
                )
            assert last is not None
            value = last.for_pair(pair_of(1, 2))
            assert value is not None
            result = value.value_for(InteractionSignalType.PROXIMITY_DURATION)
            assert result is not None
            return result

        assert measure(10.0) == pytest.approx(measure(50.0), abs=0.15)

    def test_it_outlives_the_retained_window(self) -> None:
        """Two people can stand together far longer than the history window."""
        frames = run(analyzer(), {1: still(0.0), 2: still(1.0)}, frames=251)  # 10 s
        duration = final_value(frames, InteractionSignalType.PROXIMITY_DURATION)
        assert duration is not None
        assert duration > 8.0


class TestObservationGaps:
    def test_a_short_gap_preserves_the_spell(self) -> None:
        instance = analyzer()
        last = None
        for index in range(60):
            if index in (20, 21):  # both boxes dropped briefly
                continue
            timestamp = index * DT
            last = instance.update(
                BehaviorFrame.from_snapshots(
                    index + 1,
                    timestamp,
                    [
                        snapshot(1, 0.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                        snapshot(2, 1.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    ],
                )
            )
        assert last is not None
        snapshot_ = last.for_pair(pair_of(1, 2))
        assert snapshot_ is not None
        duration = snapshot_.value_for(InteractionSignalType.PROXIMITY_DURATION)
        assert duration is not None and duration > 2.0

    def test_a_long_gap_resets_the_spell(self) -> None:
        """Where they went while unobserved is unknown."""
        instance = analyzer()
        for index in range(40):
            timestamp = index * DT
            instance.update(
                BehaviorFrame.from_snapshots(
                    index + 1,
                    timestamp,
                    [
                        snapshot(1, 0.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                        snapshot(2, 1.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    ],
                )
            )
        # Reappear together six seconds later. The first frame back begins a new
        # motion segment, so no spell can be claimed yet — "we could not see them"
        # is not "they stayed together".
        first_back = instance.update(
            BehaviorFrame.from_snapshots(
                200,
                8.0,
                [
                    snapshot(1, 0.0, 0.0, frame_index=200, timestamp=8.0),
                    snapshot(2, 1.0, 0.0, frame_index=200, timestamp=8.0),
                ],
            )
        )
        reappeared = first_back.for_pair(pair_of(1, 2))
        assert reappeared is not None
        assert reappeared.value_for(InteractionSignalType.PROXIMITY_DURATION) is None

        # It then rebuilds from zero rather than resuming the pre-gap spell.
        last = None
        for index in range(201, 226):
            timestamp = 8.0 + (index - 200) * DT
            last = instance.update(
                BehaviorFrame.from_snapshots(
                    index,
                    timestamp,
                    [
                        snapshot(1, 0.0, 0.0, frame_index=index, timestamp=timestamp),
                        snapshot(2, 1.0, 0.0, frame_index=index, timestamp=timestamp),
                    ],
                )
            )
        assert last is not None
        rebuilt = last.for_pair(pair_of(1, 2))
        assert rebuilt is not None
        duration = rebuilt.value_for(InteractionSignalType.PROXIMITY_DURATION)
        assert duration is not None
        assert duration < 1.1, "a fresh spell, not one resumed across the gap"


class TestPairExpiry:
    def test_a_pair_expires_when_a_subject_leaves(self) -> None:
        instance = analyzer()
        run(instance, {1: still(0.0), 2: still(1.0)}, frames=20)
        assert instance.tracked_pairs == 1

        for index in range(20, 40):
            timestamp = index * DT
            instance.update(
                BehaviorFrame.from_snapshots(
                    index + 1,
                    timestamp,
                    [snapshot(1, 0.0, 0.0, frame_index=index + 1, timestamp=timestamp)],
                )
            )
        assert instance.tracked_pairs == 0

    def test_track_id_churn_ends_the_old_pair(self) -> None:
        """A re-acquired person is a new subject; the old pair simply ends."""
        instance = analyzer()
        run(instance, {1: still(0.0), 2: still(1.0)}, frames=20)
        assert pair_of(1, 2) in {p for p in instance._history}

        last = None
        for index in range(20, 60):
            timestamp = index * DT
            last = instance.update(
                BehaviorFrame.from_snapshots(
                    index + 1,
                    timestamp,
                    [
                        snapshot(1, 0.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                        snapshot(31, 1.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    ],
                )
            )
        assert last is not None
        assert last.for_pair(pair_of(1, 2)) is None
        assert last.for_pair(pair_of(1, 31)) is not None

        # The new pair's proximity started fresh, not inherited from the old one.
        new_snapshot = last.for_pair(pair_of(1, 31))
        assert new_snapshot is not None
        duration = new_snapshot.value_for(InteractionSignalType.PROXIMITY_DURATION)
        assert duration is not None and duration < 1.7

    def test_memory_stays_bounded_under_churn(self) -> None:
        instance = analyzer(stale_pair_ttl_seconds=0.5)
        for index in range(200):
            timestamp = index * DT
            instance.update(
                BehaviorFrame.from_snapshots(
                    index + 1,
                    timestamp,
                    [
                        snapshot(index * 2, 0.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                        snapshot(
                            index * 2 + 1, 1.0, 0.0, frame_index=index + 1, timestamp=timestamp
                        ),
                    ],
                )
            )
        assert instance.tracked_pairs <= 2

    def test_two_pairs_do_not_leak_state(self) -> None:
        """One pair together, one pair apart, in the same frames."""
        frames = run(
            analyzer(max_pair_distance=None),
            {1: still(0.0), 2: still(1.0), 3: still(20.0)},
            frames=60,
        )
        close = final_value(frames, InteractionSignalType.PROXIMITY_DURATION, 1, 2)
        far = final_value(frames, InteractionSignalType.PROXIMITY_DURATION, 1, 3)
        assert close is not None and close > 1.0
        assert far is None


class TestCandidateGating:
    def test_distant_pairs_are_not_analysed(self) -> None:
        frames = run(
            analyzer(max_pair_distance=5.0),
            {1: still(0.0), 2: still(1.0), 3: still(50.0)},
            frames=5,
        )
        assert frames[-1].candidate_pairs == 3
        assert len(frames[-1]) == 1
        assert frames[-1].for_pair(pair_of(1, 2)) is not None

    def test_gating_off_analyses_everything(self) -> None:
        frames = run(
            analyzer(max_pair_distance=None),
            {1: still(0.0), 2: still(1.0), 3: still(50.0)},
            frames=5,
        )
        assert len(frames[-1]) == 3

    def test_a_pair_drifting_into_range_is_picked_up(self) -> None:
        frames = run(
            analyzer(max_pair_distance=5.0), {1: moving(20.0, -5.0), 2: still(0.0)}, frames=100
        )
        assert frames[0].for_pair(pair_of(1, 2)) is None
        assert frames[-1].for_pair(pair_of(1, 2)) is not None

    def test_counters_separate_candidates_from_analysed(self) -> None:
        instance = analyzer(max_pair_distance=5.0)
        run(instance, {1: still(0.0), 2: still(1.0), 3: still(50.0)}, frames=10)
        assert instance.candidate_pairs_seen == 30
        assert instance.pairs_analysed == 10


class TestCompatibilityEnforcement:
    def test_a_foreign_camera_snapshot_is_rejected(self) -> None:
        instance = analyzer()
        frame = BehaviorFrame.from_snapshots(
            1, 0.0, [snapshot(1, 0.0, 0.0), snapshot(2, 1.0, 0.0, camera_id="cam_other")]
        )
        with pytest.raises(InteractionError, match="does not belong to camera"):
            instance.update(frame)

    def test_a_foreign_space_snapshot_is_rejected(self) -> None:
        instance = analyzer()
        frame = BehaviorFrame.from_snapshots(
            1,
            0.0,
            [snapshot(1, 0.0, 0.0), snapshot(2, 0.5, 0.5, space=NORMALIZED)],
        )
        with pytest.raises(InteractionError, match="measures in ground_plane_metres"):
            instance.update(frame)


class TestMultiCameraBoundary:
    """Cross-camera subjects must never become a pair."""

    def test_two_cameras_use_two_analyzers_with_separate_state(self) -> None:
        left = analyzer(metric_context("cam_a"))
        right = analyzer(metric_context("cam_b"))

        left_frames = run(left, {1: still(0.0), 2: still(1.0)}, frames=20)
        right_frames = run(right, {1: still(0.0), 2: still(1.0)}, frames=20)

        # Same numeric track ids on both cameras, and still four distinct subjects.
        assert left_frames[-1].snapshots[0].pair != right_frames[-1].snapshots[0].pair

        assert left_frames[-1].snapshots[0].pair.camera_id == "cam_a"
        assert right_frames[-1].snapshots[0].pair.camera_id == "cam_b"
        assert left.tracked_pairs == right.tracked_pairs == 1

    def test_an_analyzer_refuses_the_other_camera_s_subjects(self) -> None:
        """Even at identical venue coordinates, the pair is refused."""
        instance = analyzer(metric_context("cam_a"))
        mixed = BehaviorFrame.from_snapshots(
            1,
            0.0,
            [
                snapshot(1, 4.0, 4.0, camera_id="cam_a"),
                snapshot(9, 4.0, 4.0, camera_id="cam_b"),
            ],
        )
        with pytest.raises(InteractionError, match="camera-local"):
            instance.update(mixed)

    def test_the_pair_reference_itself_refuses_it(self) -> None:
        with pytest.raises(InteractionError, match="Cross-camera"):
            SubjectPairRef(SubjectRef("cam_a", 1), SubjectRef("cam_b", 9))


class TestZoneContext:
    def test_shared_zones_are_reported(self) -> None:
        instance = analyzer()
        frame = BehaviorFrame.from_snapshots(
            1,
            0.0,
            [
                snapshot(1, 0.0, 0.0, zone_ids=("dancefloor", "bar")),
                snapshot(2, 1.0, 0.0, zone_ids=("dancefloor",)),
            ],
        )
        result = instance.update(frame)
        assert result.snapshots[0].shared_zone_ids == ("dancefloor",)

    def test_different_zones_share_nothing(self) -> None:
        instance = analyzer()
        frame = BehaviorFrame.from_snapshots(
            1,
            0.0,
            [
                snapshot(1, 0.0, 0.0, zone_ids=("bar",)),
                snapshot(2, 1.0, 0.0, zone_ids=("entrance",)),
            ],
        )
        assert instance.update(frame).snapshots[0].shared_zone_ids == ()

    def test_one_subject_without_zones(self) -> None:
        instance = analyzer()
        frame = BehaviorFrame.from_snapshots(
            1, 0.0, [snapshot(1, 0.0, 0.0, zone_ids=("bar",)), snapshot(2, 1.0, 0.0)]
        )
        assert instance.update(frame).snapshots[0].shared_zone_ids == ()

    def test_ordering_follows_the_first_subject(self) -> None:
        instance = analyzer()
        frame = BehaviorFrame.from_snapshots(
            1,
            0.0,
            [
                snapshot(1, 0.0, 0.0, zone_ids=("bar", "dancefloor", "walkway")),
                snapshot(2, 1.0, 0.0, zone_ids=("walkway", "dancefloor")),
            ],
        )
        assert instance.update(frame).snapshots[0].shared_zone_ids == ("dancefloor", "walkway")

    def test_zones_do_not_gate_pair_creation(self) -> None:
        """People either side of a zone boundary can still be close."""
        instance = analyzer()
        frame = BehaviorFrame.from_snapshots(
            1,
            0.0,
            [
                snapshot(1, 0.0, 0.0, zone_ids=("left",)),
                snapshot(2, 0.4, 0.0, zone_ids=("right",)),
            ],
        )
        result = instance.update(frame)
        assert len(result) == 1
        assert result.snapshots[0].distance == pytest.approx(0.4)


class TestNormalizedMode:
    def test_normalized_analysis_works(self) -> None:
        instance = InteractionAnalyzer(
            SpatialContext(CAMERA, NORMALIZED), InteractionAnalysisConfig.for_normalized_image()
        )
        frame = BehaviorFrame.from_snapshots(
            1,
            0.0,
            [
                snapshot(1, 0.1, 0.1, space=NORMALIZED),
                snapshot(2, 0.4, 0.5, space=NORMALIZED),
            ],
        )
        result = instance.update(frame)
        assert result.snapshots[0].distance == pytest.approx(0.5)
        assert result.snapshots[0].geometry.distance_unit == "normalized_frame_unit"


class TestDeterminism:
    def test_two_identical_runs_agree(self) -> None:
        def measure() -> list[tuple[tuple[int, int], float]]:
            frames = run(
                analyzer(max_pair_distance=None),
                {1: moving(0.0, 1.0), 2: moving(6.0, -1.0), 3: still(3.0)},
                frames=40,
            )
            return [(s.pair.track_ids, s.distance) for s in frames[-1]]

        assert measure() == measure()

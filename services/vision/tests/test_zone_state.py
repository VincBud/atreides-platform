"""Zone state: entries, exits, the grace period, dwell timing, and cleanup."""

from __future__ import annotations

import pytest

from atreides_vision.zones.config import ZoneAnalysisConfig
from atreides_vision.zones.geometry import NormalizedPoint
from atreides_vision.zones.models import ZoneMembership
from atreides_vision.zones.state import TrackMovementStore, ZoneStateStore

ZONE = "dancefloor"


def members(*track_ids: int) -> tuple[ZoneMembership, ...]:
    return (ZoneMembership(ZONE, frozenset(track_ids)),)


def store(grace: int = 0, **overrides: object) -> ZoneStateStore:
    config = ZoneAnalysisConfig(exit_grace_frames=grace, **overrides)  # type: ignore[arg-type]
    return ZoneStateStore([ZONE], config)


class TestTransitions:
    def test_outside_to_inside_is_an_entry(self) -> None:
        state = store()
        state.update(members(1), frame_index=1, timestamp_seconds=0.0)
        assert state.entries(ZONE) == 1
        assert state.occupancy(ZONE) == 1
        assert state.exits(ZONE) == 0

    def test_inside_to_inside_adds_no_further_entry(self) -> None:
        """A continuously present person is counted once, not once per frame."""
        state = store()
        for frame in range(1, 51):
            state.update(members(1), frame_index=frame, timestamp_seconds=frame / 25.0)
        assert state.entries(ZONE) == 1
        assert state.occupancy(ZONE) == 1

    def test_inside_to_outside_is_an_exit(self) -> None:
        state = store()
        state.update(members(1), frame_index=1, timestamp_seconds=0.0)
        state.update(members(), frame_index=2, timestamp_seconds=0.04)
        assert state.exits(ZONE) == 1
        assert state.occupancy(ZONE) == 0

    def test_re_entry_counts_again(self) -> None:
        state = store()
        state.update(members(1), 1, 0.0)
        state.update(members(), 2, 0.04)
        state.update(members(1), 3, 0.08)
        assert state.entries(ZONE) == 2
        assert state.exits(ZONE) == 1
        assert state.occupancy(ZONE) == 1

    def test_several_tracks_are_counted_independently(self) -> None:
        state = store()
        state.update(members(1, 2, 3), 1, 0.0)
        state.update(members(2, 3), 2, 0.04)
        assert state.entries(ZONE) == 3
        assert state.exits(ZONE) == 1
        assert state.occupancy(ZONE) == 2

    def test_occupancy_always_equals_entries_minus_exits(self) -> None:
        """An invariant worth holding: the counters can never drift apart."""
        state = store(grace=3)
        script = [(1,), (1, 2), (1, 2, 3), (2, 3), (3,), (), (), (), (), (4,), (4, 5)]
        for frame, present in enumerate(script, start=1):
            state.update(members(*present), frame, frame / 25.0)
            assert state.occupancy(ZONE) == state.entries(ZONE) - state.exits(ZONE)


class TestGracePeriod:
    def test_brief_gap_does_not_produce_an_exit(self) -> None:
        """A dropped detection must not look like the person leaving."""
        state = store(grace=5)
        state.update(members(1), 1, 0.00)
        for frame in range(2, 6):  # four frames with no detection
            state.update(members(), frame, frame / 25.0)
        assert state.exits(ZONE) == 0
        assert state.occupancy(ZONE) == 1

    def test_gap_longer_than_grace_produces_an_exit(self) -> None:
        state = store(grace=5)
        state.update(members(1), 1, 0.0)
        for frame in range(2, 9):
            state.update(members(), frame, frame / 25.0)
        assert state.exits(ZONE) == 1
        assert state.occupancy(ZONE) == 0

    def test_exit_lands_exactly_one_frame_past_the_grace_window(self) -> None:
        state = store(grace=3)
        state.update(members(1), 10, 0.4)
        for frame in (11, 12, 13):
            state.update(members(), frame, frame / 25.0)
            assert state.occupancy(ZONE) == 1, f"still within grace at frame {frame}"
        state.update(members(), 14, 14 / 25.0)
        assert state.occupancy(ZONE) == 0

    def test_reappearing_within_grace_keeps_one_continuous_visit(self) -> None:
        state = store(grace=5)
        state.update(members(1), 1, 0.0)
        state.update(members(), 2, 0.04)
        state.update(members(1), 3, 0.08)
        assert state.entries(ZONE) == 1, "no phantom re-entry"
        assert state.exits(ZONE) == 0

    def test_zero_grace_exits_immediately(self) -> None:
        state = store(grace=0)
        state.update(members(1), 1, 0.0)
        state.update(members(), 2, 0.04)
        assert state.exits(ZONE) == 1


class TestDwell:
    def test_dwell_grows_with_source_time(self) -> None:
        state = store()
        state.update(members(1), 1, 0.0)
        state.update(members(1), 26, 1.0)
        average, maximum = state.dwell_stats(ZONE, now_seconds=1.0)
        assert average == pytest.approx(1.0)
        assert maximum == pytest.approx(1.0)

    @pytest.mark.parametrize("fps", [10.0, 25.0, 30.0, 59.94])
    def test_dwell_is_correct_at_any_frame_rate(self, fps: float) -> None:
        """Two seconds of footage is two seconds of dwell, whatever the frame rate."""
        state = store()
        # N frames span N-1 intervals, so 2 seconds needs one extra frame.
        frames = round(fps * 2) + 1
        for frame in range(1, frames + 1):
            state.update(members(1), frame, (frame - 1) / fps)
        now = (frames - 1) / fps
        _, maximum = state.dwell_stats(ZONE, now)
        assert maximum == pytest.approx(2.0, abs=1.0 / fps)

    def test_average_and_max_differ_with_staggered_arrivals(self) -> None:
        state = store()
        state.update(members(1), 1, 0.0)  # arrives at t=0
        state.update(members(1, 2), 26, 1.0)  # track 2 arrives at t=1
        average, maximum = state.dwell_stats(ZONE, now_seconds=3.0)
        assert maximum == pytest.approx(3.0)
        assert average == pytest.approx(2.5)

    def test_empty_zone_reports_zero(self) -> None:
        assert store().dwell_stats(ZONE, now_seconds=10.0) == (0.0, 0.0)

    def test_dwell_is_never_negative(self) -> None:
        state = store()
        state.update(members(1), 5, 2.0)
        assert state.dwell_stats(ZONE, now_seconds=1.0) == (0.0, 0.0)

    def test_completed_visit_is_timestamped_at_the_last_sighting(self) -> None:
        """Not at the moment the grace window lapses — that would inflate dwell."""
        state = store(grace=5)
        state.update(members(1), 1, 0.0)
        state.update(members(1), 26, 1.0)  # last seen at t=1.0
        for frame in range(27, 40):
            state.update(members(), frame, (frame - 1) / 25.0)

        visits = state.recent_visits(ZONE)
        assert len(visits) == 1
        assert visits[0].duration_seconds == pytest.approx(1.0)
        assert state.completed_visits(ZONE) == 1


class TestCleanup:
    def test_departed_tracks_are_released(self) -> None:
        state = store(grace=2)
        for frame in range(1, 501):
            # A brand new track every frame: worst case for retained state.
            state.update(members(frame), frame, frame / 25.0)
        assert state.occupancy(ZONE) <= 3, "expired members must be dropped"

    def test_completed_visit_history_is_bounded(self) -> None:
        state = store(grace=0, max_completed_visits_per_zone=10)
        for frame in range(1, 201):
            state.update(members(frame), frame, frame / 25.0)
        assert len(state.recent_visits(ZONE)) <= 10
        assert state.completed_visits(ZONE) > 10, "the total count is still accurate"

    def test_unknown_zone_reads_as_empty(self) -> None:
        state = store()
        assert state.occupancy("nope") == 0
        assert state.entries("nope") == 0
        assert state.dwell_stats("nope", 1.0) == (0.0, 0.0)

    def test_zone_seen_only_at_runtime_is_registered(self) -> None:
        state = ZoneStateStore([], ZoneAnalysisConfig(exit_grace_frames=0))
        state.update((ZoneMembership("late", frozenset({1})),), 1, 0.0)
        assert state.occupancy("late") == 1


class TestTrackMovementStore:
    def test_speed_needs_two_samples(self) -> None:
        movement = TrackMovementStore()
        assert movement.speed(1) is None
        movement.observe(1, NormalizedPoint(0.1, 0.1), 1, 0.0)
        assert movement.speed(1) is None

    def test_straight_line_speed(self) -> None:
        """0.2 of a frame in 1 second is 0.2 normalized units per second."""
        movement = TrackMovementStore()
        movement.observe(1, NormalizedPoint(0.1, 0.5), 1, 0.0)
        movement.observe(1, NormalizedPoint(0.3, 0.5), 26, 1.0)
        assert movement.speed(1) == pytest.approx(0.2)

    @pytest.mark.parametrize("fps", [10.0, 25.0, 50.0])
    def test_speed_is_independent_of_frame_rate(self, fps: float) -> None:
        """The same physical motion yields the same speed at any frame rate."""
        movement = TrackMovementStore(ZoneAnalysisConfig(speed_window_seconds=1.0))
        frames = int(fps)
        for step in range(frames + 1):
            t = step / fps
            movement.observe(1, NormalizedPoint(0.1 + 0.2 * t, 0.5), step + 1, t)
        assert movement.speed(1) == pytest.approx(0.2, rel=0.05)

    def test_zero_speed_for_a_stationary_track(self) -> None:
        movement = TrackMovementStore()
        for step in range(10):
            movement.observe(1, NormalizedPoint(0.4, 0.4), step + 1, step / 25.0)
        assert movement.speed(1) == pytest.approx(0.0)

    def test_movement_in_place_registers_activity(self) -> None:
        """Path length, not displacement: dancing on the spot is movement."""
        movement = TrackMovementStore()
        positions = [0.40, 0.45, 0.40, 0.45, 0.40, 0.45]
        for step, x in enumerate(positions):
            movement.observe(1, NormalizedPoint(x, 0.5), step + 1, step / 25.0)
        speed = movement.speed(1)
        assert speed is not None and speed > 0.5

    def test_only_the_recent_window_counts(self) -> None:
        movement = TrackMovementStore(ZoneAnalysisConfig(speed_window_seconds=0.5))
        # Fast early movement, then stationary for well over the window.
        movement.observe(1, NormalizedPoint(0.0, 0.5), 1, 0.0)
        movement.observe(1, NormalizedPoint(0.9, 0.5), 2, 0.1)
        for step in range(2, 40):
            movement.observe(1, NormalizedPoint(0.9, 0.5), step + 1, step / 10.0)
        assert movement.speed(1) == pytest.approx(0.0, abs=1e-6)

    def test_simultaneous_samples_do_not_divide_by_zero(self) -> None:
        movement = TrackMovementStore()
        movement.observe(1, NormalizedPoint(0.1, 0.1), 1, 5.0)
        movement.observe(1, NormalizedPoint(0.2, 0.2), 2, 5.0)
        assert movement.speed(1) is None

    def test_mean_speed_skips_tracks_without_history(self) -> None:
        movement = TrackMovementStore()
        movement.observe(1, NormalizedPoint(0.0, 0.0), 1, 0.0)
        movement.observe(1, NormalizedPoint(0.1, 0.0), 2, 1.0)
        movement.observe(2, NormalizedPoint(0.5, 0.5), 2, 1.0)  # only one sample
        assert movement.mean_speed(frozenset({1, 2})) == pytest.approx(0.1)

    def test_mean_speed_of_nothing_is_zero(self) -> None:
        assert TrackMovementStore().mean_speed(frozenset()) == 0.0

    def test_stale_tracks_are_pruned(self) -> None:
        movement = TrackMovementStore(ZoneAnalysisConfig(stale_track_ttl_frames=10))
        movement.observe(1, NormalizedPoint(0.1, 0.1), 1, 0.0)
        movement.prune(frame_index=5)
        assert 1 in movement
        movement.prune(frame_index=11)
        assert 1 not in movement

    def test_memory_stays_bounded_over_a_long_run(self) -> None:
        movement = TrackMovementStore(
            ZoneAnalysisConfig(stale_track_ttl_frames=10, max_tracked_movements=50)
        )
        for frame in range(1, 2001):
            movement.observe(frame, NormalizedPoint(0.5, 0.5), frame, frame / 25.0)
            movement.prune(frame)
        assert len(movement) <= 50

    def test_hard_cap_evicts_least_recently_seen(self) -> None:
        movement = TrackMovementStore(
            ZoneAnalysisConfig(stale_track_ttl_frames=10_000, max_tracked_movements=3)
        )
        for track_id in range(1, 6):
            movement.observe(track_id, NormalizedPoint(0.5, 0.5), track_id, track_id / 25.0)
        movement.prune(frame_index=6)
        assert len(movement) == 3
        assert 1 not in movement and 5 in movement

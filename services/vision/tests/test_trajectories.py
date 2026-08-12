"""Trajectory history: bounded length, TTL cleanup, hard ceiling."""

from __future__ import annotations

import pytest

from atreides_vision.models import BoundingBox, Point, Track
from atreides_vision.video.trajectories import TrajectoryStore


def track_at(track_id: int, x: float, y: float) -> Track:
    """A 10x10 track box centred on ``(x, y)``."""
    return Track(track_id=track_id, bbox_xyxy=BoundingBox(x - 5, y - 5, x + 5, y + 5))


class TestConstruction:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"max_length": 1}, "max_length"),
            ({"ttl_frames": 0}, "ttl_frames"),
            ({"max_trails": 0}, "max_trails"),
        ],
    )
    def test_rejects_invalid_bounds(self, kwargs: dict[str, int], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            TrajectoryStore(**kwargs)


class TestAccumulation:
    def test_appends_track_centres_in_order(self) -> None:
        store = TrajectoryStore(max_length=5)
        for frame_index in range(1, 4):
            store.update([track_at(1, float(frame_index * 10), 20.0)], frame_index)

        assert store.trail(1) == (Point(10.0, 20.0), Point(20.0, 20.0), Point(30.0, 20.0))

    def test_trail_is_capped_at_max_length(self) -> None:
        store = TrajectoryStore(max_length=3, ttl_frames=100)
        for frame_index in range(1, 21):
            store.update([track_at(1, float(frame_index), 0.0)], frame_index)

        trail = store.trail(1)
        assert len(trail) == 3
        # Oldest points are dropped, newest kept.
        assert [point.x for point in trail] == [18.0, 19.0, 20.0]

    def test_tracks_are_independent(self) -> None:
        store = TrajectoryStore(max_length=5)
        store.update([track_at(1, 10, 10), track_at(2, 50, 50)], 1)
        store.update([track_at(1, 12, 10)], 2)

        assert len(store.trail(1)) == 2
        assert len(store.trail(2)) == 1

    def test_unknown_track_has_empty_trail(self) -> None:
        assert TrajectoryStore().trail(999) == ()


class TestCleanup:
    def test_stale_track_is_forgotten_after_ttl(self) -> None:
        store = TrajectoryStore(max_length=5, ttl_frames=3)
        store.update([track_at(1, 10, 10)], frame_index=1)
        assert 1 in store

        # Frames 2 and 3 arrive without track 1.
        store.update([track_at(2, 20, 20)], frame_index=2)
        store.update([track_at(2, 21, 20)], frame_index=3)
        assert 1 in store, "should survive until the TTL actually elapses"

        store.update([track_at(2, 22, 20)], frame_index=4)
        assert 1 not in store
        assert store.trail(1) == ()
        assert 2 in store

    def test_reappearing_track_keeps_its_trail(self) -> None:
        store = TrajectoryStore(max_length=5, ttl_frames=10)
        store.update([track_at(1, 10, 10)], 1)
        store.update([], 2)
        store.update([track_at(1, 12, 10)], 3)

        assert len(store.trail(1)) == 2

    def test_memory_does_not_grow_without_bound(self) -> None:
        """The core guarantee: a long run with churning ids stays bounded."""
        store = TrajectoryStore(max_length=25, ttl_frames=5, max_trails=512)
        for frame_index in range(1, 1001):
            # A brand new track id every frame — worst case for the store.
            store.update([track_at(frame_index, 1.0, 1.0)], frame_index)

        assert len(store) <= 5, "stale trails must be released as the run proceeds"

    def test_hard_ceiling_evicts_least_recently_seen(self) -> None:
        store = TrajectoryStore(max_length=5, ttl_frames=10_000, max_trails=3)
        for track_id in range(1, 6):
            store.update([track_at(track_id, 1.0, 1.0)], frame_index=track_id)

        assert len(store) == 3
        assert store.track_ids == {3, 4, 5}

    def test_prune_can_be_called_without_updates(self) -> None:
        store = TrajectoryStore(max_length=5, ttl_frames=2)
        store.update([track_at(1, 10, 10)], 1)
        store.prune(frame_index=50)
        assert len(store) == 0

    def test_clear_forgets_everything(self) -> None:
        store = TrajectoryStore()
        store.update([track_at(1, 1, 1), track_at(2, 2, 2)], 1)
        store.clear()
        assert len(store) == 0
        assert store.track_ids == frozenset()

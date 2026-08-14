"""Per-subject history: motion segments, gaps, bounds and state isolation."""

from __future__ import annotations

import pytest

from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.behavior.history import SubjectHistory, SubjectHistoryStore
from atreides_vision.behavior.models import BehaviorObservation, SubjectRef

NORMALIZED = MotionSpace.NORMALIZED_IMAGE
FPS = 25.0
DT = 1.0 / FPS

ONE = SubjectRef("cam_01", 1)
TWO = SubjectRef("cam_01", 2)


def observation(
    subject: SubjectRef,
    frame_index: int,
    *,
    timestamp: float | None = None,
    x: float = 0.5,
    y: float = 0.5,
) -> BehaviorObservation:
    return BehaviorObservation(
        subject=subject,
        frame_index=frame_index,
        timestamp_seconds=(frame_index - 1) * DT if timestamp is None else timestamp,
        position=MotionPoint(x, y, NORMALIZED),
    )


def config(**overrides: object) -> BehaviorAnalysisConfig:
    return BehaviorAnalysisConfig(**overrides)  # type: ignore[arg-type]


class TestSubjectHistoryAppend:
    def test_records_observations_in_order(self) -> None:
        history = SubjectHistory(ONE, config())
        for frame in range(1, 6):
            history.append(observation(ONE, frame))
        assert len(history) == 5
        assert [record.frame_index for record in history.observations] == [1, 2, 3, 4, 5]
        assert history.latest is not None and history.latest.frame_index == 5

    def test_first_observation_is_not_continuous(self) -> None:
        history = SubjectHistory(ONE, config())
        history.append(observation(ONE, 1))
        assert history.is_continuous is False
        assert history.segment_started_at_frame == 1

    def test_second_observation_continues_the_segment(self) -> None:
        history = SubjectHistory(ONE, config())
        history.append(observation(ONE, 1))
        history.append(observation(ONE, 2))
        assert history.is_continuous is True
        assert history.segment_started_at_frame == 1

    def test_rejects_an_observation_for_a_different_subject(self) -> None:
        history = SubjectHistory(ONE, config())
        with pytest.raises(ValueError, match="cannot be appended to the history"):
            history.append(observation(TWO, 1))

    @pytest.mark.parametrize("second_frame", [3, 4])
    def test_rejects_out_of_order_frames(self, second_frame: int) -> None:
        """A caller with a broken frame loop must fail loudly, not silently."""
        history = SubjectHistory(ONE, config())
        history.append(observation(ONE, 4))
        with pytest.raises(ValueError, match="strictly increasing frame order"):
            history.append(observation(ONE, second_frame))

    def test_empty_history_has_no_latest(self) -> None:
        assert SubjectHistory(ONE, config()).latest is None
        assert SubjectHistory(ONE, config()).observations == ()


class TestMotionSegments:
    def test_a_short_gap_is_bridged(self) -> None:
        """A dropped detection or two must not throw away good history."""
        history = SubjectHistory(ONE, config(max_observation_gap_seconds=0.5))
        history.append(observation(ONE, 1))
        history.append(observation(ONE, 2))
        history.append(observation(ONE, 8))  # ~0.24 s later at 25 FPS

        assert history.is_continuous is True
        assert len(history) == 3
        assert history.segment_started_at_frame == 1

    def test_a_long_gap_starts_a_new_segment(self) -> None:
        """Motion across an unobserved interval is unknown, not slow."""
        history = SubjectHistory(ONE, config(max_observation_gap_seconds=0.5))
        history.append(observation(ONE, 1))
        history.append(observation(ONE, 2))
        history.append(observation(ONE, 60))  # ~2.4 s later

        assert history.is_continuous is False
        assert len(history) == 1
        assert history.segment_started_at_frame == 60
        assert history.observations[0].frame_index == 60

    def test_gap_boundary_is_inclusive(self) -> None:
        history = SubjectHistory(ONE, config(max_observation_gap_seconds=0.5))
        history.append(observation(ONE, 1, timestamp=0.0))
        history.append(observation(ONE, 2, timestamp=0.5))
        assert history.is_continuous is True

    def test_a_long_gap_clears_the_stationary_spell(self) -> None:
        history = SubjectHistory(ONE, config(max_observation_gap_seconds=0.5))
        history.append(observation(ONE, 1))
        history.set_stationary_since(0.0)
        history.append(observation(ONE, 60))
        assert history.stationary_since is None

    def test_a_short_gap_keeps_the_stationary_spell(self) -> None:
        history = SubjectHistory(ONE, config(max_observation_gap_seconds=0.5))
        history.append(observation(ONE, 1))
        history.set_stationary_since(0.0)
        history.append(observation(ONE, 3))
        assert history.stationary_since == 0.0


class TestRecentWindow:
    def test_returns_only_the_window(self) -> None:
        history = SubjectHistory(ONE, config(history_window_seconds=2.0))
        for frame in range(1, 51):  # 2 s at 25 FPS
            history.append(observation(ONE, frame))

        recent = history.recent(1.0)
        assert len(recent) == 26  # inclusive of the boundary sample
        assert recent[-1].frame_index == 50

    def test_window_wider_than_history_returns_everything(self) -> None:
        history = SubjectHistory(ONE, config())
        for frame in range(1, 6):
            history.append(observation(ONE, frame))
        assert len(history.recent(10.0)) == 5

    def test_empty_history_returns_nothing(self) -> None:
        assert SubjectHistory(ONE, config()).recent(1.0) == ()

    def test_rejects_a_non_positive_window(self) -> None:
        history = SubjectHistory(ONE, config())
        history.append(observation(ONE, 1))
        with pytest.raises(ValueError, match="window_seconds must be positive"):
            history.recent(0.0)


class TestBounds:
    def test_history_is_trimmed_to_the_retention_window(self) -> None:
        history = SubjectHistory(ONE, config(history_window_seconds=1.0))
        for frame in range(1, 251):  # 10 s at 25 FPS
            history.append(observation(ONE, frame))
        assert len(history) <= 26
        assert history.observations[-1].frame_index == 250

    def test_absolute_sample_cap_is_enforced(self) -> None:
        """A second, independent bound against pathological timestamps."""
        history = SubjectHistory(
            ONE, config(max_samples_per_subject=10, history_window_seconds=10_000.0)
        )
        for frame in range(1, 200):
            history.append(observation(ONE, frame))
        assert len(history) == 10

    def test_at_least_two_samples_survive_a_short_window(self) -> None:
        """Speed stays computable when samples are sparser than the window.

        Low-frame-rate footage: samples arrive 5 s apart but are still one
        continuous segment, and the 1 s retention window would otherwise leave
        a single sample and no measurable motion at all.
        """
        history = SubjectHistory(
            ONE, config(history_window_seconds=1.0, max_observation_gap_seconds=10.0)
        )
        for frame in range(1, 20):
            history.append(observation(ONE, frame, timestamp=(frame - 1) * 5.0))
        assert history.is_continuous is True
        assert len(history) == 2


class TestStoreIsolation:
    def test_two_track_ids_do_not_leak_into_each_other(self) -> None:
        store = SubjectHistoryStore(config())
        for frame in range(1, 6):
            store.observe(observation(ONE, frame, x=0.1 * frame))
            store.observe(observation(TWO, frame, x=0.9))

        first, second = store.get(ONE), store.get(TWO)
        assert first is not None and second is not None
        assert len(first) == len(second) == 5
        assert [record.position.x for record in second.observations] == [0.9] * 5

    def test_two_cameras_sharing_a_track_id_stay_distinct(self) -> None:
        """cam_a/7 and cam_b/7 are different people and must not share state."""
        store = SubjectHistoryStore(config())
        a, b = SubjectRef("cam_a", 7), SubjectRef("cam_b", 7)
        for frame in range(1, 4):
            store.observe(observation(a, frame, x=0.2))
        store.observe(observation(b, 1, x=0.8))

        history_a, history_b = store.get(a), store.get(b)
        assert history_a is not None and history_b is not None
        assert len(history_a) == 3
        assert len(history_b) == 1
        assert history_b.observations[0].position.x == 0.8
        assert len(store) == 2

    def test_two_stores_share_nothing(self) -> None:
        """No module-level state: two analyzers cannot see each other."""
        first, second = SubjectHistoryStore(config()), SubjectHistoryStore(config())
        first.observe(observation(ONE, 1))
        assert ONE in first
        assert ONE not in second
        assert len(second) == 0

    def test_unknown_subject_is_none(self) -> None:
        assert SubjectHistoryStore(config()).get(ONE) is None


class TestStorePruning:
    def test_stale_subjects_expire(self) -> None:
        store = SubjectHistoryStore(config(stale_subject_ttl_frames=10))
        store.observe(observation(ONE, 1))
        store.observe(observation(TWO, 1))

        store.prune(frame_index=5)
        assert len(store) == 2

        store.prune(frame_index=11)
        assert len(store) == 0

    def test_a_still_visible_subject_survives_pruning(self) -> None:
        store = SubjectHistoryStore(config(stale_subject_ttl_frames=10))
        for frame in range(1, 40):
            store.observe(observation(ONE, frame))
            store.prune(frame)
        assert ONE in store

    def test_track_churn_cannot_grow_memory_without_limit(self) -> None:
        store = SubjectHistoryStore(config(stale_subject_ttl_frames=5))
        for track_id in range(1, 1001):
            store.observe(observation(SubjectRef("cam_01", track_id), track_id))
            store.prune(track_id)
        assert len(store) <= 5

    def test_hard_ceiling_evicts_least_recently_seen(self) -> None:
        store = SubjectHistoryStore(config(max_tracked_subjects=3, stale_subject_ttl_frames=10_000))
        for track_id in range(1, 11):
            store.observe(observation(SubjectRef("cam_01", track_id), track_id))
        store.prune(frame_index=10)

        assert len(store) == 3
        assert set(store.subjects()) == {SubjectRef("cam_01", tid) for tid in (8, 9, 10)}

    def test_pruned_subject_starts_clean_if_it_returns(self) -> None:
        """Expiry is real: a returning id gets a fresh history, not a resurrected one."""
        store = SubjectHistoryStore(config(stale_subject_ttl_frames=5))
        store.observe(observation(ONE, 1))
        store.prune(frame_index=100)
        assert ONE not in store

        history = store.observe(observation(ONE, 101))
        assert len(history) == 1
        assert history.is_continuous is False


class TestDeterminism:
    def test_identical_input_gives_identical_history(self) -> None:
        def run() -> list[tuple[int, float]]:
            store = SubjectHistoryStore(config())
            for frame in range(1, 30):
                store.observe(observation(ONE, frame, x=0.01 * frame))
                store.prune(frame)
            history = store.get(ONE)
            assert history is not None
            return [(rec.frame_index, rec.position.x) for rec in history.observations]

        assert run() == run()

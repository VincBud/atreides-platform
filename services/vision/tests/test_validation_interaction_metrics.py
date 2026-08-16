"""Pair episodes, pair churn, and track churn propagating into pair history."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pytest

from atreides_vision.validation.interaction_metrics import InteractionMetricsAccumulator
from atreides_vision.validation.track_metrics import (
    TrackPresence,
    TrackPresenceAccumulator,
    TrackRef,
)
from validation_records import behavior_stream, interaction_header, interaction_record

FPS = 10.0


def presence_for(
    tracks: Iterable[tuple[int, Sequence[int]]], frames_processed: int
) -> Mapping[TrackRef, TrackPresence]:
    accumulator = TrackPresenceAccumulator(frames_processed, FPS)
    accumulator.consume(behavior_stream(tracks, fps=FPS))
    accumulator.finalize()
    return accumulator.presence


def records(
    pairs: Iterable[tuple[int, int, Sequence[int]]], signals: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """``pairs`` is ``(first, second, frame_indices)``, serialised frame by frame."""
    by_frame: dict[int, list[tuple[int, int]]] = {}
    for first, second, frames in pairs:
        for frame in frames:
            by_frame.setdefault(frame, []).append((first, second))
    out: list[dict[str, Any]] = [interaction_header()]
    for frame in sorted(by_frame):
        for first, second in sorted(by_frame[frame]):
            out.append(interaction_record(first, second, frame, (frame - 1) / FPS, signals=signals))
    return out


def test_a_contiguous_run_of_frames_is_one_episode() -> None:
    accumulator = InteractionMetricsAccumulator(frames_processed=50, source_fps=FPS)
    accumulator.consume(records([(1, 2, range(1, 21))]))

    result = accumulator.finalize()

    assert result.unique_pair_references == 1
    assert result.pair_episodes == 1
    # Frames 1..20 at 10 FPS spans t=0.0 to t=1.9.
    assert result.episode_seconds.median == pytest.approx(1.9)


def test_a_pair_that_separates_and_returns_is_two_episodes_of_one_reference() -> None:
    accumulator = InteractionMetricsAccumulator(frames_processed=60, source_fps=FPS)
    accumulator.consume(records([(1, 2, [*range(1, 11), *range(31, 41)])]))

    result = accumulator.finalize()

    assert result.unique_pair_references == 1
    assert result.pair_episodes == 2


def test_pair_order_is_canonical() -> None:
    """(7, 3) and (3, 7) are the same pair reference."""
    accumulator = InteractionMetricsAccumulator(frames_processed=10, source_fps=FPS)
    accumulator.consume(
        [
            interaction_header(),
            interaction_record(7, 3, 1, 0.0),
            interaction_record(3, 7, 2, 0.1),
        ]
    )

    result = accumulator.finalize()

    assert result.unique_pair_references == 1
    assert result.pair_episodes == 1


def test_pair_births_per_minute_counts_episodes() -> None:
    """Six episodes over 600 frames at 10 FPS -> 60 s -> 6.0 per minute."""
    accumulator = InteractionMetricsAccumulator(frames_processed=600, source_fps=FPS)
    accumulator.consume(records([(1, 2, range(start, start + 5)) for start in range(1, 600, 100)]))

    result = accumulator.finalize()

    assert result.pair_episodes == 6
    assert result.pair_births_per_minute == pytest.approx(6.0)


def test_pairs_per_frame_counts_frames_with_no_pair_as_zero() -> None:
    accumulator = InteractionMetricsAccumulator(frames_processed=20, source_fps=FPS)
    accumulator.consume(records([(1, 2, range(1, 11)), (3, 4, range(1, 11))]))

    result = accumulator.finalize()

    # 10 frames with 2 pairs, 10 frames with none.
    assert result.pairs_per_frame.mean == pytest.approx(1.0)
    assert result.pairs_per_frame.maximum == pytest.approx(2.0)


def test_short_episode_fraction_uses_the_documented_threshold() -> None:
    accumulator = InteractionMetricsAccumulator(frames_processed=200, source_fps=FPS)
    # One 0.2 s episode (3 frames) and one 1.9 s episode (20 frames).
    accumulator.consume(records([(1, 2, range(1, 4)), (3, 4, range(50, 70))]))

    result = accumulator.finalize()

    assert result.pair_episodes == 2
    assert result.short_episode_fraction == pytest.approx(0.5)


def test_episode_ending_with_a_member_track_disappearing_is_attributed() -> None:
    """Track 2's last observation is frame 20; the pair ends there because of it."""
    presence = presence_for([(1, range(1, 41)), (2, range(1, 21))], frames_processed=40)
    accumulator = InteractionMetricsAccumulator(
        frames_processed=40, source_fps=FPS, track_presence=presence, final_frame_index=40
    )
    accumulator.consume(records([(1, 2, range(1, 21))]))

    result = accumulator.finalize()

    assert result.episodes_ended_before_clip_end == 1
    assert result.episodes_ending_with_member_track == 1
    assert result.member_track_end_fraction == pytest.approx(1.0)


def test_episode_ending_because_the_pair_separated_is_not_attributed_to_churn() -> None:
    """Both tracks survive well past the episode, so the pair simply moved apart."""
    presence = presence_for([(1, range(1, 41)), (2, range(1, 41))], frames_processed=40)
    accumulator = InteractionMetricsAccumulator(
        frames_processed=40, source_fps=FPS, track_presence=presence, final_frame_index=40
    )
    accumulator.consume(records([(1, 2, range(1, 21))]))

    result = accumulator.finalize()

    assert result.episodes_ended_before_clip_end == 1
    assert result.episodes_ending_with_member_track == 0
    assert result.member_track_end_fraction == pytest.approx(0.0)


def test_episodes_still_open_at_the_clip_end_are_excluded_from_attribution() -> None:
    """An episode running when the footage stopped has no ending to explain."""
    presence = presence_for([(1, range(1, 21)), (2, range(1, 21))], frames_processed=20)
    accumulator = InteractionMetricsAccumulator(
        frames_processed=20, source_fps=FPS, track_presence=presence, final_frame_index=20
    )
    accumulator.consume(records([(1, 2, range(1, 21))]))

    result = accumulator.finalize()

    assert result.pair_episodes == 1
    assert result.episodes_ended_before_clip_end == 0
    assert result.member_track_end_fraction is None


def test_attribution_is_omitted_when_no_track_presence_is_supplied() -> None:
    accumulator = InteractionMetricsAccumulator(
        frames_processed=40, source_fps=FPS, final_frame_index=40
    )
    accumulator.consume(records([(1, 2, range(1, 21))]))

    result = accumulator.finalize()

    assert result.episodes_ended_before_clip_end == 1
    assert result.episodes_ending_with_member_track == 0


def test_signal_availability_is_counted_per_record() -> None:
    accumulator = InteractionMetricsAccumulator(frames_processed=20, source_fps=FPS)
    accumulator.consume(
        [
            interaction_header(),
            interaction_record(1, 2, 1, 0.0),
            interaction_record(1, 2, 2, 0.1, signals=("closing_speed",)),
            interaction_record(1, 2, 3, 0.2, signals=("closing_speed", "proximity_duration")),
        ]
    )

    result = accumulator.finalize()

    assert result.total_pair_records == 3
    assert result.signal_coverage("closing_speed") == pytest.approx(2 / 3)
    assert result.signal_coverage("proximity_duration") == pytest.approx(1 / 3)


def test_empty_stream_reports_nothing_rather_than_zeros() -> None:
    accumulator = InteractionMetricsAccumulator(frames_processed=100, source_fps=FPS)
    accumulator.consume([interaction_header()])

    result = accumulator.finalize()

    assert result.pair_episodes == 0
    assert result.pair_births_per_minute == pytest.approx(0.0)
    assert result.episode_seconds.median is None
    assert result.short_episode_fraction is None
    assert result.member_track_end_fraction is None


def test_record_states_that_a_pair_is_not_a_social_interaction() -> None:
    accumulator = InteractionMetricsAccumulator(frames_processed=10, source_fps=FPS)
    accumulator.consume(records([(1, 2, range(1, 5))]))

    record = accumulator.finalize().to_record()

    assert record["measurement_class"] == "automated_proxy_system_metrics"
    assert record["identity_scope"] == "camera_local_track_pair"
    assert "not social interaction" in record["interpretation_boundary"]
    assert "NOT a rate of social encounters" in record["definitions"]["pair_births_per_minute"]

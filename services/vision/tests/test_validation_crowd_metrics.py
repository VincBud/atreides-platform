"""Cluster occupancy, composition episodes and persistence, on analytic inputs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pytest

from atreides_vision.validation.crowd_metrics import (
    PERSISTENCE_SURVIVAL_THRESHOLDS_SECONDS,
    CrowdMetricsAccumulator,
)
from validation_records import crowd_frame_record, crowd_header, spatial_cluster_record

FPS = 10.0


Frames = Sequence[tuple[int, int, Sequence[tuple[Sequence[int], float]]]]


def stream(frames: Frames) -> list[dict[str, Any]]:
    """``frames`` is ``(frame_index, observed, [(members, persistence), ...])``."""
    records: list[dict[str, Any]] = [crowd_header()]
    for frame_index, observed, clusters in frames:
        clustered = sum(len(members) for members, _ in clusters)
        largest = max((len(members) for members, _ in clusters), default=0)
        timestamp = (frame_index - 1) / FPS
        records.append(
            crowd_frame_record(
                frame_index=frame_index,
                timestamp_seconds=timestamp,
                observed=observed,
                cluster_count=len(clusters),
                clustered=clustered,
                largest=largest,
            )
        )
        for members, persistence in clusters:
            records.append(spatial_cluster_record(members, frame_index, timestamp, persistence))
    return records


def held(members: Sequence[int], frames: Iterable[int], fps: float = FPS) -> Frames:
    """A composition held continuously across ``frames``, persistence accumulating."""
    return [(frame, len(members), [(members, (index) / fps)]) for index, frame in enumerate(frames)]


def test_frame_counts_use_processed_frames_as_the_denominator() -> None:
    accumulator = CrowdMetricsAccumulator(frames_processed=20, source_fps=FPS)
    accumulator.consume(stream([(frame, 4, [((1, 2, 3), 0.0)]) for frame in range(1, 11)]))

    result = accumulator.finalize()

    # 10 frames with one cluster, 10 frames the stream did not cover.
    assert result.frames_with_summary == 10
    assert result.clusters_per_frame.mean == pytest.approx(0.5)
    assert result.clusters_per_frame.maximum == pytest.approx(1.0)
    assert result.clustered_subjects_per_frame.mean == pytest.approx(1.5)
    assert result.unclustered_subjects_per_frame.mean == pytest.approx(0.5)


def test_an_exact_member_set_held_across_frames_is_one_episode() -> None:
    accumulator = CrowdMetricsAccumulator(frames_processed=30, source_fps=FPS)
    accumulator.consume(stream(held((1, 2, 3), range(1, 21))))

    result = accumulator.finalize()

    assert result.unique_compositions == 1
    assert result.composition_episodes == 1
    assert result.total_cluster_records == 20
    # Persistence is the analyser's own value at the end of the episode.
    assert result.persistence_seconds.maximum == pytest.approx(1.9)


def test_a_membership_change_starts_a_new_composition() -> None:
    """Adding a fourth member is a different composition, not a bigger one."""
    frames: Frames = [
        *[(frame, 4, [((1, 2, 3), (frame - 1) / FPS)]) for frame in range(1, 11)],
        *[(frame, 4, [((1, 2, 3, 4), (frame - 11) / FPS)]) for frame in range(11, 21)],
    ]
    accumulator = CrowdMetricsAccumulator(frames_processed=20, source_fps=FPS)
    accumulator.consume(stream(frames))

    result = accumulator.finalize()

    assert result.unique_compositions == 2
    assert result.composition_episodes == 2
    assert result.cluster_size_histogram == {3: 10, 4: 10}
    assert result.largest_cluster_size == 4


def test_the_same_composition_returning_later_is_a_second_episode() -> None:
    frames: Frames = [
        *[(frame, 3, [((1, 2, 3), (frame - 1) / FPS)]) for frame in range(1, 6)],
        *[(frame, 3, []) for frame in range(6, 11)],
        *[(frame, 3, [((1, 2, 3), (frame - 11) / FPS)]) for frame in range(11, 16)],
    ]
    accumulator = CrowdMetricsAccumulator(frames_processed=15, source_fps=FPS)
    accumulator.consume(stream(frames))

    result = accumulator.finalize()

    assert result.unique_compositions == 1
    assert result.composition_episodes == 2


def test_composition_births_per_minute() -> None:
    """Six one-frame compositions over 600 frames at 10 FPS -> 6.0 per minute."""
    frames: Frames = [(frame, 3, [((1, 2, frame), 0.0)]) for frame in range(1, 600, 100)]
    accumulator = CrowdMetricsAccumulator(frames_processed=600, source_fps=FPS)
    accumulator.consume(stream(frames))

    result = accumulator.finalize()

    assert result.composition_episodes == 6
    assert result.composition_births_per_minute == pytest.approx(6.0)


def test_survival_fractions_use_reported_persistence() -> None:
    """Two episodes: one held 1.9 s, one held 0.2 s, over a 6 s clip."""
    frames: Frames = [
        *held((1, 2, 3), range(1, 21)),
        *[(frame, 3, [((4, 5, 6), (frame - 31) / FPS)]) for frame in range(31, 34)],
    ]
    accumulator = CrowdMetricsAccumulator(
        frames_processed=60, source_fps=FPS, clip_duration_seconds=6.0
    )
    accumulator.consume(stream(frames))

    result = accumulator.finalize()

    assert result.composition_episodes == 2
    assert result.survival_fractions[1.0] == pytest.approx(0.5)
    assert result.survival_fractions[2.0] == pytest.approx(0.0)
    # A 6 s clip can answer the 5 s question.
    assert result.survival_fractions[5.0] == pytest.approx(0.0)


def test_a_clip_too_short_for_a_threshold_reports_none_not_zero() -> None:
    accumulator = CrowdMetricsAccumulator(
        frames_processed=20, source_fps=FPS, clip_duration_seconds=2.0
    )
    accumulator.consume(stream(held((1, 2, 3), range(1, 21))))

    result = accumulator.finalize()

    assert set(result.survival_fractions) == set(PERSISTENCE_SURVIVAL_THRESHOLDS_SECONDS)
    assert result.survival_fractions[1.0] is not None
    # A 2.0 s clip cannot answer "did it hold for more than 5 s".
    assert result.survival_fractions[5.0] is None


def test_frames_with_no_cluster_still_contribute_a_summary() -> None:
    accumulator = CrowdMetricsAccumulator(frames_processed=10, source_fps=FPS)
    accumulator.consume(stream([(frame, 2, []) for frame in range(1, 11)]))

    result = accumulator.finalize()

    assert result.frames_with_summary == 10
    assert result.clusters_per_frame.mean == pytest.approx(0.0)
    assert result.observed_subjects_per_frame.mean == pytest.approx(2.0)
    assert result.unclustered_subjects_per_frame.mean == pytest.approx(2.0)
    assert result.composition_episodes == 0
    assert result.persistence_seconds.median is None


def test_empty_stream_reports_zeros_without_dividing_by_zero() -> None:
    accumulator = CrowdMetricsAccumulator(frames_processed=100, source_fps=FPS)
    accumulator.consume([crowd_header()])

    result = accumulator.finalize()

    assert result.composition_episodes == 0
    assert result.composition_births_per_minute == pytest.approx(0.0)
    assert result.largest_cluster_size == 0
    assert result.cluster_size_histogram == {}
    assert all(value is None or value == 0.0 for value in result.survival_fractions.values())


def test_record_states_that_a_cluster_is_not_a_social_group() -> None:
    accumulator = CrowdMetricsAccumulator(frames_processed=20, source_fps=FPS)
    accumulator.consume(stream(held((1, 2, 3), range(1, 11))))

    record = accumulator.finalize().to_record()

    assert record["measurement_class"] == "automated_proxy_system_metrics"
    assert record["identity_scope"] == "camera_local_tracks"
    assert "not social group stability" in record["interpretation_boundary"]
    assert (
        "NOT automatically a real-world group event"
        in record["definitions"]["cluster_composition_births_per_minute"]
    )

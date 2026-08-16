"""Track presence, lifetime, occupancy and churn proxies, on analytic inputs.

Each case states the arithmetic it expects: a track observed on frames 1-31 at
10 FPS spans 3.0 s, so its lifetime is 3.0 s and it is not a "short track" under
any of the reported thresholds.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pytest

from atreides_vision.validation.distributions import Distribution
from atreides_vision.validation.track_metrics import (
    SHORT_TRACK_THRESHOLDS_SECONDS,
    TrackMetrics,
    TrackPresenceAccumulator,
)
from validation_records import behavior_stream, observation

FPS = 10.0


def metrics(
    tracks: Iterable[tuple[int, Sequence[int]]],
    frames_processed: int,
    fps: float = FPS,
) -> TrackMetrics:
    accumulator = TrackPresenceAccumulator(frames_processed, fps)
    accumulator.consume(behavior_stream(tracks, fps=fps))
    return accumulator.finalize()


def test_lifetime_is_the_span_from_first_to_last_observation() -> None:
    # Frames 1..31 at 10 FPS: timestamps 0.0 .. 3.0, so a 3.0 s span.
    result = metrics([(7, range(1, 32))], frames_processed=31)

    assert result.unique_track_count == 1
    assert result.lifetime_seconds.median == pytest.approx(3.0)
    assert result.observed_frames_per_track.median == pytest.approx(31.0)


def test_a_single_frame_track_has_zero_lifetime() -> None:
    """Not padded to a frame period: padding would understate short-track counts."""
    result = metrics([(1, [5])], frames_processed=10)

    assert result.lifetime_seconds.median == pytest.approx(0.0)
    assert result.short_track_fractions[0.5] == pytest.approx(1.0)


def test_active_tracks_counts_empty_frames_as_zero() -> None:
    """20 frames processed, two tracks present on only the first 10."""
    result = metrics([(1, range(1, 11)), (2, range(1, 11))], frames_processed=20)

    # 10 frames with 2 active, 10 frames with 0 active.
    assert result.active_tracks.mean == pytest.approx(1.0)
    assert result.max_active_tracks == 2
    assert result.active_track_histogram == {2: 10, 0: 10}


def test_median_active_tracks_uses_the_full_frame_distribution() -> None:
    result = metrics([(1, range(1, 8))], frames_processed=10)

    # 7 frames with 1 active, 3 with 0 -> sorted [0,0,0,1,1,1,1,1,1,1], median 1.
    assert result.active_tracks.median == pytest.approx(1.0)


def test_births_per_minute_counts_first_appearances() -> None:
    """Six tracks over 600 frames at 10 FPS -> 60 s -> 6.0 births/min."""
    result = metrics(
        [(track_id, range(track_id * 10, track_id * 10 + 5)) for track_id in range(1, 7)],
        frames_processed=600,
    )

    assert result.unique_track_count == 6
    assert result.track_births_per_minute == pytest.approx(6.0)
    assert result.unique_track_ids_per_minute == pytest.approx(6.0)


def test_short_track_fractions_use_the_documented_thresholds() -> None:
    # One 0.2 s track (frames 1-3), one 0.7 s (1-8), one 3.0 s (1-31).
    result = metrics([(1, range(1, 4)), (2, range(1, 9)), (3, range(1, 32))], frames_processed=31)

    assert set(result.short_track_fractions) == set(SHORT_TRACK_THRESHOLDS_SECONDS)
    assert result.short_track_fractions[0.5] == pytest.approx(1 / 3)
    assert result.short_track_fractions[1.0] == pytest.approx(2 / 3)
    assert result.short_track_fractions[2.0] == pytest.approx(2 / 3)


def test_gaps_are_detected_and_measured() -> None:
    """Observed on 1-5 and 11-15: one gap of 0.6 s (t=0.4 -> t=1.0)."""
    result = metrics([(1, [*range(1, 6), *range(11, 16)])], frames_processed=15)

    assert result.tracks_with_gaps == 1
    assert result.gap_count == 1
    assert result.gap_seconds.maximum == pytest.approx(0.6)
    assert result.gap_seconds.median == pytest.approx(0.6)


def test_multiple_gaps_are_each_recorded() -> None:
    result = metrics([(1, [1, 2, 6, 7, 20])], frames_processed=20)

    assert result.gap_count == 2
    # t=0.1 -> 0.5 is 0.4 s; t=0.6 -> 1.9 is 1.3 s.
    assert result.gap_seconds.minimum == pytest.approx(0.4)
    assert result.gap_seconds.maximum == pytest.approx(1.3)


def test_contiguous_observations_produce_no_gap() -> None:
    result = metrics([(1, range(1, 21))], frames_processed=20)

    assert result.tracks_with_gaps == 0
    assert result.gap_count == 0
    assert result.gap_seconds.count == 0
    assert result.gap_seconds.median is None


def test_tracks_on_different_cameras_are_never_merged() -> None:
    accumulator = TrackPresenceAccumulator(frames_processed=4, source_fps=FPS)
    accumulator.observe("cam_a", 1, 1, 0.0)
    accumulator.observe("cam_b", 1, 1, 0.0)

    result = accumulator.finalize()

    assert result.unique_track_count == 2


def test_non_observation_records_are_ignored() -> None:
    accumulator = TrackPresenceAccumulator(frames_processed=2, source_fps=FPS)
    accumulator.consume(
        [
            {"record_type": "run_metadata", "camera_id": "cam_test"},
            {"record_type": "calibration_integrity", "camera_id": "cam_test"},
            observation(1, 1, 0.0),
            observation(1, 2, 0.1),
        ]
    )

    result = accumulator.finalize()

    assert result.unique_track_count == 1
    assert result.total_observations == 2


def test_empty_stream_reports_zeros_without_dividing_by_zero() -> None:
    accumulator = TrackPresenceAccumulator(frames_processed=100, source_fps=FPS)

    result = accumulator.finalize()

    assert result.unique_track_count == 0
    assert result.total_observations == 0
    assert result.track_births_per_minute == pytest.approx(0.0)
    assert result.lifetime_seconds.count == 0
    assert result.lifetime_seconds.median is None
    assert result.active_tracks.mean == pytest.approx(0.0)
    assert all(value == 0.0 for value in result.short_track_fractions.values())


def test_zero_frame_rate_reports_no_rate_rather_than_infinity() -> None:
    accumulator = TrackPresenceAccumulator(frames_processed=10, source_fps=0.0)
    accumulator.observe("cam_test", 1, 1, 0.0)

    result = accumulator.finalize()

    assert result.track_births_per_minute is None
    assert result.unique_track_ids_per_minute is None


def test_record_states_its_limitations_next_to_the_numbers() -> None:
    result = metrics([(1, range(1, 11))], frames_processed=10)

    record = result.to_record()

    assert record["measurement_class"] == "automated_proxy_system_metrics"
    assert record["identity_scope"] == "camera_local_track"
    assert "NOT an ID-switch rate" in record["limitations"]["track_births_per_minute"]
    assert "Not proof of an ID switch" in record["limitations"]["short_track_fractions"]
    assert "upper bound" in record["limitations"]["unique_track_ids"]


def test_distribution_percentiles_interpolate() -> None:
    distribution = Distribution.from_values([0.0, 1.0, 2.0, 3.0, 4.0])

    assert distribution.count == 5
    assert distribution.median == pytest.approx(2.0)
    assert distribution.p10 == pytest.approx(0.4)
    assert distribution.p90 == pytest.approx(3.6)
    assert distribution.minimum == pytest.approx(0.0)
    assert distribution.maximum == pytest.approx(4.0)
    assert distribution.mean == pytest.approx(2.0)


def test_empty_distribution_reports_no_statistics() -> None:
    distribution = Distribution.from_values([])

    assert distribution.count == 0
    assert distribution.median is None
    assert distribution.to_record()["median"] is None


def test_one_accumulator_per_clip_keeps_identical_track_refs_independent() -> None:
    """`camera_01/track_7` in two clips is two tracklets, not one.

    The lowest-level statement of the isolation guarantee: a validation run
    builds one accumulator per clip, so an identical (camera_id, track_id) in
    two clips never meets itself.
    """
    clip_a = TrackPresenceAccumulator(frames_processed=10, source_fps=FPS)
    clip_b = TrackPresenceAccumulator(frames_processed=40, source_fps=FPS)

    for frame in range(1, 11):
        clip_a.observe("camera_01", 7, frame, (frame - 1) / FPS)
    for frame in range(1, 41):
        clip_b.observe("camera_01", 7, frame, (frame - 1) / FPS)

    result_a = clip_a.finalize()
    result_b = clip_b.finalize()

    # Each sees exactly one track, and only its own observations of it.
    assert result_a.unique_track_count == 1
    assert result_b.unique_track_count == 1
    assert result_a.total_observations == 10
    assert result_b.total_observations == 40
    assert result_a.lifetime_seconds.median == pytest.approx(0.9)
    assert result_b.lifetime_seconds.median == pytest.approx(3.9)
    # Nothing is shared between them.
    assert clip_a.presence[("camera_01", 7)] is not clip_b.presence[("camera_01", 7)]

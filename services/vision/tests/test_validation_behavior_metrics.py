"""Behaviour signal coverage and warm-up, on analytic inputs.

The property under test throughout: an omitted signal is counted as *not
measurable*, never as a zero value.
"""

from __future__ import annotations

import pytest

from atreides_vision.behavior.models import SignalType
from atreides_vision.validation.behavior_metrics import (
    WARM_UP_SIGNALS,
    BehaviorMetricsAccumulator,
)
from validation_records import behavior_header, behavior_stream, observation

FPS = 10.0


def test_coverage_is_records_with_the_signal_over_all_observations() -> None:
    # 20 observations, the last 15 of which carry speed.
    accumulator = BehaviorMetricsAccumulator()
    accumulator.consume(
        behavior_stream([(1, range(1, 21))], fps=FPS, signals_from_frame=5, signals=("speed",))
    )

    result = accumulator.finalize()

    assert result.total_observations == 20
    assert result.coverage("speed") == pytest.approx(15 / 20)
    assert result.observations_with_any_signal == 15
    assert result.any_signal_coverage == pytest.approx(0.75)


def test_a_signal_never_emitted_has_zero_coverage_but_is_still_listed() -> None:
    accumulator = BehaviorMetricsAccumulator()
    accumulator.consume(
        behavior_stream([(1, range(1, 11))], fps=FPS, signals_from_frame=0, signals=("speed",))
    )

    result = accumulator.finalize()

    assert result.coverage("speed") == pytest.approx(1.0)
    assert result.coverage(SignalType.ACCELERATION.value) == pytest.approx(0.0)
    # Every signal type appears in the record, so a reader can tell "never
    # measured" from "signal type does not exist".
    assert set(result.signal_observation_counts) >= {signal.value for signal in SignalType}


def test_coverage_of_an_empty_stream_is_none_not_zero() -> None:
    accumulator = BehaviorMetricsAccumulator()
    accumulator.consume([behavior_header()])

    result = accumulator.finalize()

    assert result.total_observations == 0
    assert result.coverage("speed") is None
    assert result.any_signal_coverage is None


def test_warm_up_is_measured_from_each_track_s_own_first_observation() -> None:
    """Track 2 appears at frame 51; its warm-up is measured from there, not from 0."""
    accumulator = BehaviorMetricsAccumulator()
    accumulator.consume(
        behavior_stream(
            [(1, range(1, 41)), (2, range(51, 91))],
            fps=FPS,
            signals_from_frame=10,
            signals=("speed",),
        )
    )

    result = accumulator.finalize()

    warm_up = result.warm_up_seconds["speed"]
    assert warm_up.count == 2
    # 10 frames at 10 FPS = 1.0 s for both tracks.
    assert warm_up.median == pytest.approx(1.0)
    assert warm_up.minimum == pytest.approx(1.0)
    assert warm_up.maximum == pytest.approx(1.0)


def test_tracks_that_never_produce_a_signal_are_counted_separately() -> None:
    """A track too short to fill its window contributes to 'never', not to the median."""
    accumulator = BehaviorMetricsAccumulator()
    accumulator.consume(
        [
            behavior_header(),
            # Track 1 warms up after 5 frames.
            *[observation(1, frame, (frame - 1) / FPS) for frame in range(1, 6)],
            *[
                observation(1, frame, (frame - 1) / FPS, signals=("speed",))
                for frame in range(6, 11)
            ],
            # Track 2 is observed twice and never produces anything.
            observation(2, 11, 1.0),
            observation(2, 12, 1.1),
        ]
    )

    result = accumulator.finalize()

    assert result.tracks_observed == 2
    assert result.tracks_never_producing["speed"] == 1
    assert result.warm_up_seconds["speed"].count == 1
    assert result.warm_up_seconds["speed"].median == pytest.approx(0.5)


def test_warm_up_is_reported_for_the_documented_signals_only() -> None:
    accumulator = BehaviorMetricsAccumulator()
    accumulator.consume(behavior_stream([(1, range(1, 5))], fps=FPS))

    result = accumulator.finalize()

    assert set(result.warm_up_seconds) == {signal.value for signal in WARM_UP_SIGNALS}
    # Stationary duration is excluded on purpose — see the module docstring.
    assert SignalType.STATIONARY_DURATION.value not in result.warm_up_seconds


def test_space_is_read_from_the_stream_header() -> None:
    accumulator = BehaviorMetricsAccumulator()
    accumulator.consume(behavior_stream([(1, [1])], fps=FPS))

    assert accumulator.finalize().space == "normalized_image"


def test_record_documents_the_absent_subject_motion_coverage() -> None:
    accumulator = BehaviorMetricsAccumulator()
    accumulator.consume(behavior_stream([(1, range(1, 5))], fps=FPS))

    record = accumulator.finalize().to_record()

    assert record["measurement_class"] == "automated_proxy_system_metrics"
    assert "not currently measurable, never zero" in record["definitions"]["coverage"]
    assert "not serialised" in record["definitions"]["subject_motion"]


def test_signals_from_several_tracks_are_attributed_independently() -> None:
    accumulator = BehaviorMetricsAccumulator()
    accumulator.consume(
        [
            behavior_header(),
            observation(1, 1, 0.0, signals=("speed",)),
            observation(2, 1, 0.0),
            observation(1, 2, 0.1, signals=("speed", "acceleration")),
            observation(2, 2, 0.1),
        ]
    )

    result = accumulator.finalize()

    assert result.total_observations == 4
    assert result.coverage("speed") == pytest.approx(0.5)
    assert result.coverage("acceleration") == pytest.approx(0.25)
    assert result.tracks_never_producing["speed"] == 1

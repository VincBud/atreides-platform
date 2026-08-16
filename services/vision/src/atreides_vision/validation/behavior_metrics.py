"""How often the behaviour layer could actually produce each signal.

Coverage answers one question: given that a subject was observed, how often did
the stream carry a given measurement for it?

    coverage(signal) = observations containing that signal / total observations

The denominator is every ``observation`` record, including those with no signals
at all — a freshly acquired track has a position and no history, which is
exactly the case coverage exists to quantify.

**An omitted signal means "not currently measurable", never "zero".** V0.3
emits a signal only when its window holds enough history; a subject who is
standing still still reports a speed of 0.0, so an absent speed is genuinely an
absence of measurement rather than an absence of motion. Coverage counts
presence in the record and never imputes a value.

Warm-up measures the other half of the same fact: how long after a track's first
observation its first speed, acceleration or heading-change measurement arrived.
Track fragmentation resets that clock, which is why this sits next to the
tracking proxies rather than in a section of its own.

A note on what is not here
--------------------------
``SubjectMotion`` — the net displacement the interaction and crowd layers use for
relative heading — is **not serialised** in the behaviour output contract
(schema 0.3.3). It exists in memory during a run and is consumed in process.
There is therefore no honest per-subject motion-heading coverage to report from
the streams, and this module does not invent one; the nearest available
observables are ``heading_change_rate`` coverage here and the pairwise
``relative_motion_heading`` availability in
:mod:`~atreides_vision.validation.interaction_metrics`.

Memory
------
Coverage is a handful of counters. Warm-up keeps one small record per unique
track id, so it scales the same way
:mod:`~atreides_vision.validation.track_metrics` does.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from atreides_vision.behavior.models import SignalType
from atreides_vision.validation.distributions import Distribution
from atreides_vision.validation.track_metrics import TrackRef

__all__ = [
    "WARM_UP_SIGNALS",
    "BehaviorCoverageMetrics",
    "BehaviorMetricsAccumulator",
]

#: Signals whose time-to-first-measurement is reported. Stationary duration is
#: excluded on purpose: it is emitted as soon as a subject qualifies as
#: stationary, so "time until first stationary duration" measures how long
#: someone moved for, not how long the layer took to warm up.
WARM_UP_SIGNALS: tuple[SignalType, ...] = (
    SignalType.SPEED,
    SignalType.ACCELERATION,
    SignalType.HEADING_CHANGE_RATE,
)

_PRECISION = 4


@dataclass(frozen=True, slots=True)
class BehaviorCoverageMetrics:
    """Signal availability across one clip's behaviour stream."""

    total_observations: int
    observations_with_any_signal: int
    #: Records containing each signal type, keyed by the signal's stream name.
    signal_observation_counts: Mapping[str, int]
    #: Seconds from a track's first observation to its first measurement of each
    #: signal, over the tracks that ever produced one.
    warm_up_seconds: Mapping[str, Distribution]
    #: Tracks that were observed but never produced each signal at all.
    tracks_observed: int
    tracks_never_producing: Mapping[str, int]
    #: Coordinate space the stream declared, so a reader cannot mistake
    #: normalized coverage for metric coverage.
    space: str | None = None

    def coverage(self, signal: str) -> float | None:
        """Fraction of observations carrying ``signal``, or ``None`` if none exist."""
        if self.total_observations == 0:
            return None
        return self.signal_observation_counts.get(signal, 0) / self.total_observations

    @property
    def any_signal_coverage(self) -> float | None:
        if self.total_observations == 0:
            return None
        return self.observations_with_any_signal / self.total_observations

    def to_record(self) -> dict[str, Any]:
        return {
            "measurement_class": "automated_proxy_system_metrics",
            "space": self.space,
            "total_observations": self.total_observations,
            "observations_with_any_signal": self.observations_with_any_signal,
            "any_signal_coverage": _round(self.any_signal_coverage),
            "coverage": {
                name: _round(self.coverage(name)) for name in sorted(self.signal_observation_counts)
            },
            "observation_counts": dict(sorted(self.signal_observation_counts.items())),
            "tracks_observed": self.tracks_observed,
            "tracks_never_producing_signal": dict(sorted(self.tracks_never_producing.items())),
            "warm_up_seconds": {
                name: distribution.to_record()
                for name, distribution in sorted(self.warm_up_seconds.items())
            },
            "definitions": {
                "coverage": (
                    "observations containing the signal / total subject observations. An "
                    "omitted signal means not currently measurable, never zero."
                ),
                "warm_up_seconds": (
                    "seconds from a track's first observation to its first record carrying "
                    "that signal, over tracks that ever produced it. Track fragmentation "
                    "restarts this clock, so it is coupled to the tracking proxies."
                ),
                "subject_motion": (
                    "SubjectMotion / usable motion heading is not serialised by behaviour "
                    "schema 0.3.3, so no per-subject motion-heading coverage is reported. "
                    "heading_change_rate coverage is the nearest available observable."
                ),
            },
        }


def _round(value: float | None, digits: int = _PRECISION) -> float | None:
    return None if value is None else round(value, digits)


@dataclass(slots=True)
class _TrackWarmUp:
    """First-observation time for a track, and its first time per signal."""

    first_timestamp_seconds: float
    first_signal_timestamp: dict[str, float]


class BehaviorMetricsAccumulator:
    """Streams a behaviour output file into coverage and warm-up metrics.

    Records are consumed one at a time; nothing but the per-track warm-up table
    is retained.
    """

    def __init__(self, warm_up_signals: Iterable[SignalType] = WARM_UP_SIGNALS) -> None:
        self._warm_up_signals = tuple(signal.value for signal in warm_up_signals)
        self._total_observations = 0
        self._observations_with_any_signal = 0
        self._signal_counts: dict[str, int] = {signal.value: 0 for signal in SignalType}
        self._tracks: dict[TrackRef, _TrackWarmUp] = {}
        self._space: str | None = None

    def observe(
        self,
        camera_id: str,
        track_id: int,
        timestamp_seconds: float,
        signal_types: Iterable[str],
    ) -> None:
        """Record one subject observation and which signals it carried."""
        self._total_observations += 1
        present = tuple(signal_types)
        if present:
            self._observations_with_any_signal += 1

        ref: TrackRef = (camera_id, track_id)
        state = self._tracks.get(ref)
        if state is None:
            state = _TrackWarmUp(
                first_timestamp_seconds=timestamp_seconds, first_signal_timestamp={}
            )
            self._tracks[ref] = state

        for name in present:
            self._signal_counts[name] = self._signal_counts.get(name, 0) + 1
            if name in self._warm_up_signals and name not in state.first_signal_timestamp:
                state.first_signal_timestamp[name] = timestamp_seconds

    def consume(self, records: Iterable[Mapping[str, Any]]) -> BehaviorMetricsAccumulator:
        """Feed a behaviour stream, reading the header for its coordinate space."""
        for record in records:
            record_type = record.get("record_type")
            if record_type == "run_metadata":
                space = record.get("space")
                self._space = None if space is None else str(space)
                continue
            if record_type != "observation":
                continue
            subject = record.get("subject")
            if not isinstance(subject, dict):
                continue
            signals = record.get("signals")
            names = (
                tuple(str(signal["type"]) for signal in signals if isinstance(signal, dict))
                if isinstance(signals, list)
                else ()
            )
            self.observe(
                camera_id=str(record["camera_id"]),
                track_id=int(subject["track_id"]),
                timestamp_seconds=float(record["timestamp_seconds"]),
                signal_types=names,
            )
        return self

    def finalize(self) -> BehaviorCoverageMetrics:
        warm_up: dict[str, Distribution] = {}
        never: dict[str, int] = {}
        for name in self._warm_up_signals:
            delays: list[float] = []
            missing = 0
            for state in self._tracks.values():
                first = state.first_signal_timestamp.get(name)
                if first is None:
                    missing += 1
                else:
                    delays.append(first - state.first_timestamp_seconds)
            warm_up[name] = Distribution.from_values(delays)
            never[name] = missing

        return BehaviorCoverageMetrics(
            total_observations=self._total_observations,
            observations_with_any_signal=self._observations_with_any_signal,
            signal_observation_counts=dict(self._signal_counts),
            warm_up_seconds=warm_up,
            tracks_observed=len(self._tracks),
            tracks_never_producing=never,
            space=self._space,
        )

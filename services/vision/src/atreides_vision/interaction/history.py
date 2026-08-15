"""Bounded per-pair separation history.

One :class:`PairHistory` per canonical
:class:`~atreides_vision.interaction.models.SubjectPairRef`, all owned by one
:class:`PairHistoryStore` belonging to one analyzer. No module-level state, so
two cameras — or two tests — cannot see each other's pairs.

Pair state is quadratic in subject count, which makes bounding it more pressing
than it is for individual subjects. Three independent bounds apply: per pair by
time window and by absolute sample count, and store-wide by a time-to-live on
unseen pairs plus a hard ceiling that evicts the least recently seen.

Observation gaps
----------------
Exactly as the behaviour layer treats a single subject's motion, a pair's
history is the current unbroken run of observations. A short gap — one of the
two boxes dropped for a frame — is bridged, because both people were still
there. A longer gap starts a new run and discards what came before: two people
who were together, disappeared for four seconds, and are together again may
have parted and returned, and a proximity duration spanning that gap would be a
claim about time nobody observed.

Track-id churn
--------------
If a tracker loses one subject and re-acquires it under a new id, the old pair
simply stops being observed and expires, while a *different* pair begins. That
is a direct consequence of camera-local identity and is not papered over here.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

from atreides_vision.interaction.config import InteractionAnalysisConfig
from atreides_vision.interaction.models import SubjectPairRef

__all__ = ["PairHistory", "PairHistoryStore", "PairSample"]


@dataclass(frozen=True, slots=True)
class PairSample:
    """The separation between two subjects at one moment."""

    frame_index: int
    timestamp_seconds: float
    distance: float


class PairHistory:
    """The current unbroken run of separations for one pair, plus its proximity spell."""

    def __init__(self, pair: SubjectPairRef, config: InteractionAnalysisConfig) -> None:
        self._pair = pair
        self._config = config
        self._samples: deque[PairSample] = deque(maxlen=config.max_samples_per_pair)
        self._is_continuous = False
        self._last_seen_seconds = 0.0
        self._proximity_since: float | None = None

    @property
    def pair(self) -> SubjectPairRef:
        return self._pair

    @property
    def is_continuous(self) -> bool:
        """Whether the last observation extended the run rather than starting one.

        ``False`` on the first observation and after any gap longer than the
        configured maximum.
        """
        return self._is_continuous

    @property
    def samples(self) -> tuple[PairSample, ...]:
        """The retained run, oldest first."""
        return tuple(self._samples)

    @property
    def latest(self) -> PairSample | None:
        return self._samples[-1] if self._samples else None

    @property
    def last_seen_seconds(self) -> float:
        return self._last_seen_seconds

    @property
    def proximity_since(self) -> float | None:
        """Start of the proximity spell in progress, or ``None``."""
        return self._proximity_since

    def observe(self, sample: PairSample) -> None:
        """Record one separation, closing or extending the run.

        Raises:
            ValueError: if the sample does not come strictly after the previous
                one. Out-of-order input means the caller's frame loop is wrong,
                and every rate derived afterwards would be quietly incorrect.
        """
        previous = self.latest
        if previous is not None:
            if sample.frame_index <= previous.frame_index:
                raise ValueError(
                    "Pair samples must arrive in strictly increasing frame order for "
                    f"{self._pair}: got frame {sample.frame_index} after {previous.frame_index}"
                )
            gap = sample.timestamp_seconds - previous.timestamp_seconds
            self._is_continuous = gap <= self._config.max_observation_gap_seconds
        else:
            self._is_continuous = False

        if not self._is_continuous:
            # Nothing before the gap can be differenced against anything after
            # it, and the proximity spell cannot be vouched for across it.
            self._samples.clear()
            self._proximity_since = None

        self._samples.append(sample)
        self._last_seen_seconds = sample.timestamp_seconds
        self._trim()

    def recent(self, window_seconds: float) -> tuple[PairSample, ...]:
        """Samples within ``window_seconds`` of the latest one."""
        if not self._samples:
            return ()
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        cutoff = self._samples[-1].timestamp_seconds - window_seconds
        return tuple(sample for sample in self._samples if sample.timestamp_seconds >= cutoff)

    def set_proximity_since(self, value: float | None) -> None:
        """Record the proximity spell computed for this frame.

        The value is decided by
        :func:`~atreides_vision.interaction.geometry.proximity_since`; the
        history only stores it, because it has to outlive the retained window.
        """
        self._proximity_since = value

    def seconds_since_seen(self, timestamp_seconds: float) -> float:
        return timestamp_seconds - self._last_seen_seconds

    def _trim(self) -> None:
        """Drop samples older than the retained window, keeping at least two."""
        cutoff = self._samples[-1].timestamp_seconds - self._config.history_window_seconds
        while len(self._samples) > 2 and self._samples[0].timestamp_seconds < cutoff:
            self._samples.popleft()

    def __len__(self) -> int:
        return len(self._samples)


class PairHistoryStore:
    """Every pair's history for one analyzer, with bounded lifetime.

    Keyed by canonical :class:`SubjectPairRef`, so the same two subjects always
    reach the same entry regardless of the order the tracker returned them in.
    """

    def __init__(self, config: InteractionAnalysisConfig | None = None) -> None:
        self._config = config or InteractionAnalysisConfig()
        self._histories: dict[SubjectPairRef, PairHistory] = {}

    def observe(self, pair: SubjectPairRef, sample: PairSample) -> PairHistory:
        """Append a separation, creating the pair's history if needed."""
        history = self._histories.get(pair)
        if history is None:
            history = PairHistory(pair, self._config)
            self._histories[pair] = history
        history.observe(sample)
        return history

    def get(self, pair: SubjectPairRef) -> PairHistory | None:
        return self._histories.get(pair)

    def prune(self, timestamp_seconds: float) -> None:
        """Forget pairs unseen for a while, then enforce the hard ceiling.

        Time-based first, because that is the meaningful rule: a pair nobody has
        seen for five seconds is over regardless of frame rate. The count
        ceiling is a second, cruder bound for a scene dense enough that even
        live pairs would exhaust memory.
        """
        ttl = self._config.stale_pair_ttl_seconds
        stale = [
            pair
            for pair, history in self._histories.items()
            if history.seconds_since_seen(timestamp_seconds) >= ttl
        ]
        for pair in stale:
            del self._histories[pair]

        excess = len(self._histories) - self._config.max_tracked_pairs
        if excess > 0:
            oldest = sorted(self._histories.items(), key=lambda item: item[1].last_seen_seconds)
            for pair, _ in oldest[:excess]:
                del self._histories[pair]

    def forget_subjects_absent_from(self, present: frozenset[object]) -> None:
        """Drop pairs whose subjects are no longer being tracked at all.

        Time-based expiry would eventually collect these, but dropping them as
        soon as a subject disappears keeps the store proportional to what is
        actually on screen rather than to what was on screen five seconds ago.
        """
        gone = [
            pair
            for pair in self._histories
            if pair.first not in present or pair.second not in present
        ]
        for pair in gone:
            del self._histories[pair]

    def pairs(self) -> tuple[SubjectPairRef, ...]:
        return tuple(self._histories)

    def __len__(self) -> int:
        return len(self._histories)

    def __contains__(self, pair: object) -> bool:
        return pair in self._histories

    def __iter__(self) -> Iterator[SubjectPairRef]:
        return iter(self._histories)

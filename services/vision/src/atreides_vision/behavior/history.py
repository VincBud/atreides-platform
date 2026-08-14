"""Bounded per-subject observation history.

One :class:`SubjectHistory` per :class:`~atreides_vision.behavior.models.SubjectRef`,
all of them owned by one :class:`SubjectHistoryStore` that belongs to one
analyzer instance. There is no module-level state, so two analyzers — two
cameras, or two tests — cannot see each other's data.

Motion segments
---------------
Trackers lose boxes. A history is therefore not simply "the last N
observations": it is the current **motion segment**, meaning the unbroken run
of observations in which no gap exceeded
:attr:`~atreides_vision.behavior.config.BehaviorAnalysisConfig.max_observation_gap_seconds`.

* A short gap (a dropped detection or two) is bridged. The person kept moving
  and the samples either side are still comparable.
* A long gap starts a new segment and discards what came before. Where somebody
  went while unobserved is unknown, and averaging across the gap would invent
  a straight line through it — usually a slow one, since the elapsed time is
  large. That is worse than reporting nothing.

Callers can tell the two apart: :attr:`SubjectHistory.is_continuous` reports
whether the most recent append extended the segment or began a new one.

Memory
------
Three independent bounds, so no input can grow memory without limit:
per-subject by time window and by absolute sample count, and store-wide by TTL
on unseen subjects plus a hard ceiling that evicts the least recently seen.
Nothing is written to disk; all of it is discarded when the process exits.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.behavior.models import BehaviorObservation, SubjectRef

__all__ = ["SubjectHistory", "SubjectHistoryStore"]


class SubjectHistory:
    """The current motion segment of one subject, plus its stationary spell.

    Observations must arrive in strictly increasing frame order. Out-of-order
    appends raise rather than being silently reordered or dropped: they mean
    the caller's frame loop is wrong, and every signal derived afterwards would
    be quietly incorrect.
    """

    def __init__(self, subject: SubjectRef, config: BehaviorAnalysisConfig) -> None:
        self._subject = subject
        self._config = config
        self._observations: deque[BehaviorObservation] = deque(
            maxlen=config.max_samples_per_subject
        )
        self._is_continuous = False
        self._segment_started_at_frame = 0
        self._last_seen_frame = -1
        self._stationary_since: float | None = None

    @property
    def subject(self) -> SubjectRef:
        return self._subject

    @property
    def is_continuous(self) -> bool:
        """Whether the last append extended the segment rather than starting one.

        ``False`` on the very first observation and after any gap longer than
        the configured maximum.
        """
        return self._is_continuous

    @property
    def segment_started_at_frame(self) -> int:
        """Frame index at which the current motion segment began."""
        return self._segment_started_at_frame

    @property
    def last_seen_frame(self) -> int:
        return self._last_seen_frame

    @property
    def stationary_since(self) -> float | None:
        """Start of the stationary spell in progress, or ``None``."""
        return self._stationary_since

    @property
    def observations(self) -> tuple[BehaviorObservation, ...]:
        """The retained segment, oldest first."""
        return tuple(self._observations)

    @property
    def latest(self) -> BehaviorObservation | None:
        return self._observations[-1] if self._observations else None

    def append(self, observation: BehaviorObservation) -> None:
        """Record one observation, closing or extending the motion segment.

        Raises:
            ValueError: if ``observation`` belongs to a different subject, or
                does not come strictly after the previous one.
        """
        if observation.subject != self._subject:
            raise ValueError(
                f"Observation for {observation.subject} cannot be appended to the "
                f"history of {self._subject}"
            )

        previous = self.latest
        if previous is not None:
            if observation.frame_index <= previous.frame_index:
                raise ValueError(
                    "Observations must be appended in strictly increasing frame order for "
                    f"{self._subject}: got frame {observation.frame_index} after "
                    f"{previous.frame_index}"
                )
            gap = observation.timestamp_seconds - previous.timestamp_seconds
            self._is_continuous = gap <= self._config.max_observation_gap_seconds
        else:
            self._is_continuous = False

        if not self._is_continuous:
            # A new segment: nothing before the gap can be compared with
            # anything after it, and the stationary spell cannot be vouched for.
            self._observations.clear()
            self._segment_started_at_frame = observation.frame_index
            self._stationary_since = None

        self._observations.append(observation)
        self._last_seen_frame = observation.frame_index
        self._trim()

    def recent(self, window_seconds: float) -> tuple[BehaviorObservation, ...]:
        """Observations within ``window_seconds`` of the latest one.

        Always within the current segment, because the segment is all that is
        retained. Empty if there is no history at all.
        """
        if not self._observations:
            return ()
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        cutoff = self._observations[-1].timestamp_seconds - window_seconds
        return tuple(
            observation
            for observation in self._observations
            if observation.timestamp_seconds >= cutoff
        )

    def set_stationary_since(self, value: float | None) -> None:
        """Record the start of the stationary spell computed for this frame.

        The value itself is decided by
        :func:`~atreides_vision.behavior.kinematics.stationary_since`; the
        history only stores it, because it has to outlive the retained window.
        """
        self._stationary_since = value

    def frames_since_seen(self, frame_index: int) -> int:
        return frame_index - self._last_seen_frame

    def _trim(self) -> None:
        """Drop observations older than the retained history window.

        At least two are always kept so a speed estimate remains possible even
        if the configured window is short relative to the frame interval.
        """
        cutoff = self._observations[-1].timestamp_seconds - self._config.history_window_seconds
        while len(self._observations) > 2 and self._observations[0].timestamp_seconds < cutoff:
            self._observations.popleft()

    def __len__(self) -> int:
        return len(self._observations)


class SubjectHistoryStore:
    """Every subject's history for one analyzer, with bounded lifetime.

    Keyed by :class:`~atreides_vision.behavior.models.SubjectRef`, so two
    cameras that both hand out track id ``7`` keep entirely separate state.
    """

    def __init__(self, config: BehaviorAnalysisConfig | None = None) -> None:
        self._config = config or BehaviorAnalysisConfig()
        self._histories: dict[SubjectRef, SubjectHistory] = {}

    def observe(self, observation: BehaviorObservation) -> SubjectHistory:
        """Append an observation, creating the subject's history if needed."""
        history = self._histories.get(observation.subject)
        if history is None:
            history = SubjectHistory(observation.subject, self._config)
            self._histories[observation.subject] = history
        history.append(observation)
        return history

    def get(self, subject: SubjectRef) -> SubjectHistory | None:
        return self._histories.get(subject)

    def prune(self, frame_index: int) -> None:
        """Forget subjects unseen for a while, then enforce the hard ceiling."""
        ttl = self._config.stale_subject_ttl_frames
        stale = [
            subject
            for subject, history in self._histories.items()
            if history.frames_since_seen(frame_index) >= ttl
        ]
        for subject in stale:
            del self._histories[subject]

        excess = len(self._histories) - self._config.max_tracked_subjects
        if excess > 0:
            oldest = sorted(self._histories.items(), key=lambda item: item[1].last_seen_frame)
            for subject, _ in oldest[:excess]:
                del self._histories[subject]

    def subjects(self) -> tuple[SubjectRef, ...]:
        return tuple(self._histories)

    def __len__(self) -> int:
        return len(self._histories)

    def __contains__(self, subject: object) -> bool:
        return subject in self._histories

    def __iter__(self) -> Iterator[SubjectRef]:
        return iter(self._histories)

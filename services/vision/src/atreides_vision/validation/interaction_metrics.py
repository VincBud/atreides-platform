"""Stability of pairwise interaction records, and what track churn does to them.

Two vocabulary decisions, stated before the numbers because they are what makes
the numbers readable:

**A pair reference** is a canonical ``(camera_id, low_track_id, high_track_id)``.
It identifies two tracklets, not two people.

**A pair episode** is a maximal run of consecutive frames in which that pair
reference was measured. If the two subjects separate beyond the analyser's
candidate gate and later come back together, that is two episodes of one pair
reference — the same way the interaction layer itself treats it, because its
proximity duration restarts.

``pair_births_per_minute`` counts **episodes started per minute of processed
source time**. It is not a rate of social encounters. A pair episode can begin
because two people approached each other, because one of them was re-acquired
under a new track id after an occlusion, or because a third track fragmented
into two that are trivially close together.

Nothing here interprets an interaction. These are geometric co-presence records.

Track-churn propagation
-----------------------
The metric this module exists for: when a pair episode ends, *why* did it end?

An episode can end for exactly three reasons, and which are possible depends on
the analyser's ``max_pair_distance``:

1. **A member track stopped existing** for the rest of the clip. Measurable
   exactly, and what this module reports: a member track whose final observation
   in the behaviour stream is the episode's final frame did not walk away, it
   vanished.
2. **A member was momentarily unobserved** and later returned under the same
   track id. That is an observation gap, one layer downstream.
3. **The two separated past the distance gate** — *only when one is configured*.

With ``max_pair_distance = None`` — the normalized-image default, and the mode
every current validation clip runs in — every co-present pair is measured at any
separation, so reason 3 **cannot occur**. Pair-episode churn there measures
observation continuity and nothing else, and the report says so rather than
leaving a reader to assume people drifted apart.

No identity stitching is attempted, and none should be inferred: knowing that a
pair ended because a track disappeared says nothing about which later track, if
any, is the same person.

Memory
------
One entry per *currently open* pair episode, plus one float per *finished*
episode for the percentiles. Episode count scales with clip length and density,
which for a dense 50 s clip is tens of thousands of floats — bounded, but worth
knowing before pointing this at an hour of footage.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from atreides_vision.validation.distributions import Distribution
from atreides_vision.validation.track_metrics import TrackPresence, TrackRef

__all__ = [
    "SHORT_EPISODE_THRESHOLD_SECONDS",
    "InteractionMetrics",
    "InteractionMetricsAccumulator",
    "PairRef",
]

#: A canonical camera-local pair: ``(camera_id, low_track_id, high_track_id)``.
PairRef = tuple[str, int, int]

#: Episodes shorter than this are reported as a separate fraction. Descriptive
#: only — it is roughly the point below which no pairwise rate signal can have
#: filled its window, and it carries no acceptance meaning.
SHORT_EPISODE_THRESHOLD_SECONDS = 0.5

_PRECISION = 4


def _round(value: float | None, digits: int = _PRECISION) -> float | None:
    return None if value is None else round(value, digits)


@dataclass(slots=True)
class _OpenEpisode:
    start_timestamp: float
    last_timestamp: float
    last_frame_index: int


@dataclass(frozen=True, slots=True)
class InteractionMetrics:
    """Pair occupancy, churn and lifetime for one clip."""

    total_pair_records: int
    frames_processed: int
    pairs_per_frame: Distribution
    unique_pair_references: int
    pair_episodes: int
    pair_births_per_minute: float | None
    episode_seconds: Distribution
    short_episode_fraction: float | None
    #: Episodes whose last frame is also the last frame one of its member tracks
    #: was ever observed on — upstream track churn propagating into pair history.
    episodes_ending_with_member_track: int
    #: Episodes that ended before the clip did (the rest were still open at the
    #: final processed frame and have no ending to attribute).
    episodes_ended_before_clip_end: int
    #: Records carrying each pairwise signal type, for availability.
    signal_record_counts: Mapping[str, int]
    space: str | None = None

    @property
    def member_track_end_fraction(self) -> float | None:
        if self.episodes_ended_before_clip_end == 0:
            return None
        return self.episodes_ending_with_member_track / self.episodes_ended_before_clip_end

    def signal_coverage(self, signal: str) -> float | None:
        if self.total_pair_records == 0:
            return None
        return self.signal_record_counts.get(signal, 0) / self.total_pair_records

    def to_record(self) -> dict[str, Any]:
        return {
            "measurement_class": "automated_proxy_system_metrics",
            "identity_scope": "camera_local_track_pair",
            "space": self.space,
            "total_pair_records": self.total_pair_records,
            "pairs_per_frame": self.pairs_per_frame.to_record(),
            "unique_pair_references": self.unique_pair_references,
            "pair_episodes": self.pair_episodes,
            "pair_births_per_minute": _round(self.pair_births_per_minute),
            "pair_episode_seconds": self.episode_seconds.to_record(),
            "short_episode_fraction": _round(self.short_episode_fraction),
            "short_episode_threshold_seconds": SHORT_EPISODE_THRESHOLD_SECONDS,
            "signal_coverage": {
                name: _round(self.signal_coverage(name))
                for name in sorted(self.signal_record_counts)
            },
            "signal_record_counts": dict(sorted(self.signal_record_counts.items())),
            "track_churn_propagation": {
                "episodes_ended_before_clip_end": self.episodes_ended_before_clip_end,
                "episodes_ending_with_member_track_end": self.episodes_ending_with_member_track,
                "fraction": _round(self.member_track_end_fraction),
                "definition": (
                    "pair episodes whose final frame is also the final frame one of their two "
                    "member track ids was ever observed on: a member track stopped existing "
                    "for the rest of the clip. The remaining ended episodes ended because a "
                    "member was momentarily unobserved, or - only when max_pair_distance is "
                    "set - because the two separated past the gate. No identity stitching is "
                    "attempted."
                ),
            },
            "definitions": {
                "pair_reference": (
                    "a canonical (camera_id, low_track_id, high_track_id). Two tracklets, "
                    "not two people."
                ),
                "pair_episode": (
                    "a maximal run of consecutive frames in which a pair reference was "
                    "measured. One pair reference can have several episodes."
                ),
                "pair_births_per_minute": (
                    "pair episodes started per minute of processed source time. NOT a rate of "
                    "social encounters: an episode can begin because a track was re-acquired "
                    "under a new id, or because one fragmented into two nearby tracks."
                ),
            },
            "interpretation_boundary": (
                "these are geometric co-presence records. Interaction frequency is not social "
                "interaction, and no record here describes a relationship."
            ),
        }


class InteractionMetricsAccumulator:
    """Streams an interaction output file into pair-stability metrics.

    Args:
        frames_processed: Frames the pipeline processed, used as the denominator
            for pairs-per-frame so that frames with no measured pair count as
            zero rather than being excluded.
        source_fps: Source frame rate, for the per-minute rates.
        track_presence: Per-track presence from the behaviour stream, used to
            attribute episode endings. Optional: without it the attribution is
            simply not reported rather than guessed at.
        final_frame_index: The last frame index the interaction stream covers.
            Episodes still open at that frame have no ending to attribute and
            are excluded from the churn denominator.
    """

    def __init__(
        self,
        frames_processed: int,
        source_fps: float,
        track_presence: Mapping[TrackRef, TrackPresence] | None = None,
        final_frame_index: int | None = None,
    ) -> None:
        self._frames_processed = frames_processed
        self._source_fps = source_fps
        self._presence = track_presence or {}
        self._final_frame_index = final_frame_index

        self._open: dict[PairRef, _OpenEpisode] = {}
        self._episode_durations: list[float] = []
        self._episode_count = 0
        self._short_episodes = 0
        self._ended_before_clip_end = 0
        self._ended_with_member_track = 0
        self._unique_pairs: set[PairRef] = set()
        self._total_records = 0
        self._signal_counts: dict[str, int] = {}
        self._space: str | None = None

        self._current_frame: int | None = None
        self._current_frame_pairs: set[PairRef] = set()
        self._pairs_per_frame_counts: list[int] = []
        self._frames_with_records = 0
        self._last_seen_frame: int | None = None

    def observe(
        self,
        pair: PairRef,
        frame_index: int,
        timestamp_seconds: float,
        signal_types: Iterable[str] = (),
    ) -> None:
        """Record one measured pair on one frame."""
        if frame_index != self._current_frame:
            self._close_frame()
            self._current_frame = frame_index
            self._current_frame_pairs = set()

        self._total_records += 1
        self._unique_pairs.add(pair)
        self._current_frame_pairs.add(pair)
        self._last_seen_frame = frame_index
        for name in signal_types:
            self._signal_counts[name] = self._signal_counts.get(name, 0) + 1

        episode = self._open.get(pair)
        if episode is None or frame_index > episode.last_frame_index + 1:
            if episode is not None:
                self._finish(pair, episode)
            self._open[pair] = _OpenEpisode(
                start_timestamp=timestamp_seconds,
                last_timestamp=timestamp_seconds,
                last_frame_index=frame_index,
            )
            return
        episode.last_timestamp = timestamp_seconds
        episode.last_frame_index = frame_index

    def consume(self, records: Iterable[Mapping[str, Any]]) -> InteractionMetricsAccumulator:
        """Feed an interaction stream, reading the header for its space."""
        for record in records:
            record_type = record.get("record_type")
            if record_type == "run_metadata":
                space = record.get("space")
                self._space = None if space is None else str(space)
                continue
            if record_type != "interaction":
                continue
            subjects = record.get("subjects")
            if not isinstance(subjects, list) or len(subjects) != 2:
                continue
            camera_id = str(record["camera_id"])
            first = int(subjects[0]["track_id"])
            second = int(subjects[1]["track_id"])
            low, high = (first, second) if first <= second else (second, first)
            signals = record.get("signals")
            names = (
                tuple(str(signal["type"]) for signal in signals if isinstance(signal, dict))
                if isinstance(signals, list)
                else ()
            )
            self.observe(
                pair=(camera_id, low, high),
                frame_index=int(record["frame_index"]),
                timestamp_seconds=float(record["timestamp_seconds"]),
                signal_types=names,
            )
        return self

    def _close_frame(self) -> None:
        if self._current_frame is None:
            return
        self._pairs_per_frame_counts.append(len(self._current_frame_pairs))
        self._frames_with_records += 1
        # An episode not seen on this frame has ended. Closing here rather than
        # lazily is what makes an episode's end frame exact, which the
        # track-churn attribution below depends on.
        stale = [
            pair
            for pair, episode in self._open.items()
            if episode.last_frame_index < self._current_frame
        ]
        for pair in stale:
            self._finish(pair, self._open.pop(pair))

    def _finish(self, pair: PairRef, episode: _OpenEpisode) -> None:
        duration = episode.last_timestamp - episode.start_timestamp
        self._episode_durations.append(duration)
        self._episode_count += 1
        if duration < SHORT_EPISODE_THRESHOLD_SECONDS:
            self._short_episodes += 1

        # An episode still running when the clip ended has no ending to explain.
        if self._final_frame_index is not None and episode.last_frame_index >= (
            self._final_frame_index
        ):
            return
        self._ended_before_clip_end += 1
        if self._presence and self._ended_because_member_track_ended(pair, episode):
            self._ended_with_member_track += 1

    def _ended_because_member_track_ended(self, pair: PairRef, episode: _OpenEpisode) -> bool:
        camera_id, low, high = pair
        for track_id in (low, high):
            presence = self._presence.get((camera_id, track_id))
            if presence is not None and presence.last_frame_index <= episode.last_frame_index:
                return True
        return False

    def finalize(self) -> InteractionMetrics:
        """Close the final frame and every still-open episode, then summarise."""
        self._close_frame()
        self._current_frame = None
        for pair in list(self._open):
            self._finish(pair, self._open.pop(pair))

        counts = list(self._pairs_per_frame_counts)
        empty_frames = max(self._frames_processed - self._frames_with_records, 0)
        counts.extend([0] * empty_frames)

        duration_seconds = (
            self._frames_processed / self._source_fps if self._source_fps > 0 else 0.0
        )
        minutes = duration_seconds / 60.0

        return InteractionMetrics(
            total_pair_records=self._total_records,
            frames_processed=self._frames_processed,
            pairs_per_frame=Distribution.from_values(counts),
            unique_pair_references=len(self._unique_pairs),
            pair_episodes=self._episode_count,
            pair_births_per_minute=(self._episode_count / minutes) if minutes > 0 else None,
            episode_seconds=Distribution.from_values(self._episode_durations),
            short_episode_fraction=(
                self._short_episodes / self._episode_count if self._episode_count else None
            ),
            episodes_ending_with_member_track=self._ended_with_member_track,
            episodes_ended_before_clip_end=self._ended_before_clip_end,
            signal_record_counts=dict(self._signal_counts),
            space=self._space,
        )

    @property
    def last_frame_index(self) -> int | None:
        """Highest frame index seen, for callers that need it after the pass."""
        return self._last_seen_frame

"""Camera-local track presence and stability, reconstructed from behaviour output.

Everything in this module is an **automated proxy / system metric**. It measures
what the tracker did, not what was true in the room. The distinction is not
pedantry — it is the difference between a number that can be checked and one
that cannot:

* A **track birth** is the first time a ``(camera_id, track_id)`` pair appears.
  It may be a person entering the view, the same person re-acquired after an
  occlusion under a new id, a fragmentation of a continuously visible person, or
  a tracker lifecycle event. Nothing in the output stream distinguishes these,
  so ``track_births_per_minute`` **is not an ID-switch rate**.
* A **short track** is a track whose observations span less than some duration.
  It is consistent with fragmentation and also with someone genuinely crossing a
  corner of the frame. A short-track fraction is a fragmentation *proxy*, never
  proof that an identity was broken.
* An **observation gap** is a track that was reported, absent for one or more
  frames, then reported again under the same id. It is a fact about the stream.
  A track that vanished and came back as a *different* id leaves no gap here at
  all — it leaves two tracks — and this module never stitches those together.

Vocabulary is held deliberately: *track*, *tracklet*, *camera-local subject*.
Never *person*, and never *identity*.

Memory
------
Per-frame state is bounded, but per-track state is not: one presence record is
kept per unique ``(camera_id, track_id)``, so peak memory scales with the number
of unique track ids in a clip, not with clip length. For the validation footage
this is thousands of small records. A stream with millions of unique ids would
need a different design, and that is stated rather than discovered later.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from atreides_vision.validation.distributions import Distribution

__all__ = [
    "SHORT_TRACK_THRESHOLDS_SECONDS",
    "TrackMetrics",
    "TrackPresence",
    "TrackPresenceAccumulator",
    "TrackRef",
]

#: A camera-local track: ``(camera_id, track_id)``. Never a person.
TrackRef = tuple[str, int]

#: Durations the short-track fractions are reported against. Chosen to bracket
#: the region where a track is too brief for the behaviour layer's windows to
#: fill — they are *descriptive*, not acceptance thresholds, and no verdict
#: anywhere in this package is derived from them.
SHORT_TRACK_THRESHOLDS_SECONDS: tuple[float, ...] = (0.5, 1.0, 2.0)

_PRECISION = 4


@dataclass(slots=True)
class TrackPresence:
    """When one camera-local track was observed, and where it went missing.

    Mutable and accumulated in place: one of these exists per unique track for
    the length of the pass, and rebuilding an immutable copy per observation
    would cost more than the whole rest of the analysis.
    """

    ref: TrackRef
    first_frame_index: int
    last_frame_index: int
    first_timestamp_seconds: float
    last_timestamp_seconds: float
    observed_frames: int = 1
    gap_count: int = 0
    total_gap_seconds: float = 0.0
    max_gap_seconds: float = 0.0

    @property
    def lifetime_seconds(self) -> float:
        """Span from first to last observation.

        A track observed on exactly one frame has a lifetime of ``0.0``: the
        span between one observation and itself really is zero. It is not padded
        out to a frame period, because a padded value would understate the
        short-track fractions this feeds.
        """
        return self.last_timestamp_seconds - self.first_timestamp_seconds

    @property
    def observed_frame_span(self) -> int:
        """Frames from first to last observation inclusive, gaps included."""
        return self.last_frame_index - self.first_frame_index + 1

    @property
    def has_gap(self) -> bool:
        return self.gap_count > 0

    def observe(self, frame_index: int, timestamp_seconds: float) -> float | None:
        """Record one further observation, returning any gap it just closed.

        A frame index that does not advance is ignored for gap purposes: two
        records for one track on one frame cannot happen in a well-formed
        stream, and treating it as a negative gap would corrupt the totals
        rather than surface the problem.

        Returns:
            The duration of the gap this observation closed, or ``None`` when
            the track was observed on the immediately preceding frame.
        """
        self.observed_frames += 1
        if frame_index <= self.last_frame_index:
            return None

        gap: float | None = None
        if frame_index > self.last_frame_index + 1:
            gap = timestamp_seconds - self.last_timestamp_seconds
            self.gap_count += 1
            self.total_gap_seconds += gap
            self.max_gap_seconds = max(self.max_gap_seconds, gap)
        self.last_frame_index = frame_index
        self.last_timestamp_seconds = timestamp_seconds
        return gap


@dataclass(frozen=True, slots=True)
class TrackMetrics:
    """Track lifetime, occupancy and churn proxies for one clip.

    Every field is a measurement of the output stream. None is a statement about
    people, and the serialised form repeats that next to the numbers.
    """

    unique_track_count: int
    total_observations: int
    frames_processed: int
    processed_duration_seconds: float
    #: Distinct tracks observed per processed frame. Frames with no observations
    #: count as zero rather than being dropped, so the mean is over the clip.
    active_tracks: Distribution
    active_track_histogram: Mapping[int, int]
    lifetime_seconds: Distribution
    #: Fraction of unique tracks whose lifetime fell below each threshold.
    short_track_fractions: Mapping[float, float]
    track_births_per_minute: float | None
    unique_track_ids_per_minute: float | None
    tracks_with_gaps: int
    gap_count: int
    gap_seconds: Distribution
    observed_frames_per_track: Distribution

    @property
    def max_active_tracks(self) -> int:
        return int(self.active_tracks.maximum or 0)

    def to_record(self) -> dict[str, Any]:
        return {
            "measurement_class": "automated_proxy_system_metrics",
            "identity_scope": "camera_local_track",
            "unique_track_ids": self.unique_track_count,
            "total_observations": self.total_observations,
            "frames_processed": self.frames_processed,
            "active_tracks_per_frame": self.active_tracks.to_record(),
            "active_track_histogram": {
                str(count): frames for count, frames in sorted(self.active_track_histogram.items())
            },
            "track_lifetime_seconds": self.lifetime_seconds.to_record(),
            "observed_frames_per_track": self.observed_frames_per_track.to_record(),
            "short_track_fractions": {
                f"under_{threshold:g}s": round(fraction, _PRECISION)
                for threshold, fraction in sorted(self.short_track_fractions.items())
            },
            "track_births_per_minute": _round(self.track_births_per_minute),
            "unique_track_ids_per_minute": _round(self.unique_track_ids_per_minute),
            "observation_gaps": {
                "tracks_with_at_least_one_gap": self.tracks_with_gaps,
                "gap_count": self.gap_count,
                "gap_duration_seconds": self.gap_seconds.to_record(),
                "definition": (
                    "the same track id was reported, absent for one or more frames, then "
                    "reported again. A track lost and re-acquired under a new id produces two "
                    "tracks and no gap; nothing here stitches those together."
                ),
            },
            "limitations": {
                "track_births_per_minute": (
                    "a track birth is the first appearance of a (camera_id, track_id). It may be "
                    "a new person entering, a re-detection after occlusion, fragmentation, or a "
                    "tracker lifecycle event. This is NOT an ID-switch rate."
                ),
                "short_track_fractions": (
                    "a fragmentation proxy, consistent with both fragmentation and someone "
                    "genuinely crossing a corner of the view. Not proof of an ID switch."
                ),
                "unique_track_ids": (
                    "an upper bound on unique people, never a census: one person may hold "
                    "several track ids across a clip."
                ),
                "track_lifetime_seconds": (
                    "span from a track's first to its last observation. A track seen on one "
                    "frame has a lifetime of 0.0 s."
                ),
            },
        }


def _round(value: float | None, digits: int = _PRECISION) -> float | None:
    return None if value is None else round(value, digits)


class TrackPresenceAccumulator:
    """Reconstructs camera-local track presence from a behaviour observation stream.

    Consumes records one at a time so a clip's whole stream is never resident.
    Frame boundaries are detected from ``frame_index`` changing, which the
    writers guarantee: a frame's observations are serialised together, in track
    order, before the next frame's.

    Args:
        frames_processed: Frames the pipeline actually processed. Used as the
            denominator for per-frame occupancy so that frames in which nobody
            was observed count as zero active tracks rather than vanishing.
        source_fps: Source frame rate, used to convert processed frames into
            source seconds for the per-minute rates.
    """

    def __init__(self, frames_processed: int, source_fps: float) -> None:
        if frames_processed < 0:
            raise ValueError(f"frames_processed must be non-negative, got {frames_processed}")
        self._frames_processed = frames_processed
        self._source_fps = source_fps
        self._presence: dict[TrackRef, TrackPresence] = {}
        self._active_histogram: Counter[int] = Counter()
        self._current_frame: int | None = None
        self._current_frame_tracks: set[TrackRef] = set()
        self._observed_frames = 0
        self._total_observations = 0
        # One float per gap that actually occurred, not per track. Gaps are the
        # exception rather than the rule, so this stays far smaller than the
        # per-track table above.
        self._gap_durations: list[float] = []

    def observe(
        self, camera_id: str, track_id: int, frame_index: int, timestamp_seconds: float
    ) -> None:
        """Record one subject observation."""
        if not math.isfinite(timestamp_seconds):
            raise ValueError(f"timestamp_seconds must be finite, got {timestamp_seconds!r}")
        if frame_index != self._current_frame:
            self._close_frame()
            self._current_frame = frame_index
            self._current_frame_tracks = set()

        ref: TrackRef = (camera_id, track_id)
        self._current_frame_tracks.add(ref)
        self._total_observations += 1

        existing = self._presence.get(ref)
        if existing is None:
            self._presence[ref] = TrackPresence(
                ref=ref,
                first_frame_index=frame_index,
                last_frame_index=frame_index,
                first_timestamp_seconds=timestamp_seconds,
                last_timestamp_seconds=timestamp_seconds,
            )
        else:
            gap = existing.observe(frame_index, timestamp_seconds)
            if gap is not None:
                self._gap_durations.append(gap)

    def consume(self, records: Iterable[Mapping[str, Any]]) -> TrackPresenceAccumulator:
        """Feed every ``observation`` record from a behaviour stream.

        Other record types — the run-metadata header, calibration-integrity
        findings — are skipped rather than rejected, so a stream gaining a
        record type does not break validation of the ones it already had.
        """
        for record in records:
            if record.get("record_type") != "observation":
                continue
            subject = record.get("subject")
            if not isinstance(subject, dict):
                continue
            self.observe(
                camera_id=str(record["camera_id"]),
                track_id=int(subject["track_id"]),
                frame_index=int(record["frame_index"]),
                timestamp_seconds=float(record["timestamp_seconds"]),
            )
        return self

    def _close_frame(self) -> None:
        if self._current_frame is None:
            return
        self._active_histogram[len(self._current_frame_tracks)] += 1
        self._observed_frames += 1

    @property
    def presence(self) -> Mapping[TrackRef, TrackPresence]:
        """Per-track presence, for cross-stream analysis. O(unique tracks)."""
        return self._presence

    def finalize(self) -> TrackMetrics:
        """Close the final frame and summarise. Safe to call once."""
        self._close_frame()
        self._current_frame = None

        histogram = Counter(self._active_histogram)
        # Frames the pipeline processed but which produced no observation at all
        # are frames with zero active tracks. Dropping them would inflate mean
        # occupancy by conditioning on somebody being present.
        empty_frames = max(self._frames_processed - self._observed_frames, 0)
        if empty_frames:
            histogram[0] += empty_frames

        active_counts = [count for count, frames in histogram.items() for _ in range(frames)]
        lifetimes = [presence.lifetime_seconds for presence in self._presence.values()]
        observed_frames_per_track = [
            float(presence.observed_frames) for presence in self._presence.values()
        ]

        duration_seconds = (
            self._frames_processed / self._source_fps if self._source_fps > 0 else 0.0
        )
        minutes = duration_seconds / 60.0
        unique = len(self._presence)

        return TrackMetrics(
            unique_track_count=unique,
            total_observations=self._total_observations,
            frames_processed=self._frames_processed,
            processed_duration_seconds=duration_seconds,
            active_tracks=Distribution.from_values(active_counts),
            active_track_histogram=dict(histogram),
            lifetime_seconds=Distribution.from_values(lifetimes),
            short_track_fractions={
                threshold: (
                    sum(1 for lifetime in lifetimes if lifetime < threshold) / unique
                    if unique
                    else 0.0
                )
                for threshold in SHORT_TRACK_THRESHOLDS_SECONDS
            },
            # A birth is the first observation of a track id, so the number of
            # births over a clip is exactly the number of unique track ids. Both
            # are reported because they answer different questions — rate of
            # new ids appearing, and total ids seen — and a reader comparing
            # them should be able to see they are the same measurement.
            track_births_per_minute=(unique / minutes) if minutes > 0 else None,
            unique_track_ids_per_minute=(unique / minutes) if minutes > 0 else None,
            tracks_with_gaps=sum(1 for presence in self._presence.values() if presence.has_gap),
            gap_count=sum(presence.gap_count for presence in self._presence.values()),
            gap_seconds=Distribution.from_values(self._gap_durations),
            observed_frames_per_track=Distribution.from_values(observed_frames_per_track),
        )

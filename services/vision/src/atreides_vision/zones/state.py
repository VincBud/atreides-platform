"""Temporal zone state: who is in a zone, since when, and how they are moving.

Two stores, both in-memory and both bounded:

* :class:`ZoneStateStore` — membership per zone, with entry/exit counting and
  dwell timing.
* :class:`TrackMovementStore` — a short window of recent floor positions per
  track, used to estimate image-plane speed.

Nothing here is written to disk. All state is keyed by ephemeral,
session-local track ids and is discarded when the process exits.

Membership and the grace period
-------------------------------
A track counts as being in a zone from the first frame it is observed inside
until it has gone ``exit_grace_frames`` consecutive frames without being
observed inside. That single rule covers both ways a track stops being seen:
the tracker briefly loses the box, or the person genuinely walks out. The
consequences are worth stating plainly:

* A dropped detection does not produce a phantom exit-then-re-entry pair.
* A person loitering on a zone boundary does not churn the counters.
* A real exit is recorded ``exit_grace_frames`` frames after the person's last
  observation inside — and is timestamped at that last observation, so dwell
  durations stay truthful even though the count arrives late.
* Occupancy therefore includes tracks inside their grace window, which keeps
  the invariant ``occupancy == entries - exits`` true at every frame.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from atreides_vision.zones.config import ZoneAnalysisConfig
from atreides_vision.zones.geometry import NormalizedPoint
from atreides_vision.zones.models import ZoneMembership

__all__ = ["TrackMovementStore", "TrackZoneState", "ZoneStateStore", "ZoneVisit"]

#: Absolute cap on samples kept per track, as a guard against pathological
#: timestamps. Time-based trimming is the real bound.
_MAX_SAMPLES_PER_TRACK = 4096


@dataclass(slots=True)
class TrackZoneState:
    """One track's current presence in one zone."""

    track_id: int
    entered_at_frame: int
    entered_at_seconds: float
    last_inside_frame: int
    last_inside_seconds: float

    def dwell_seconds(self, now_seconds: float) -> float:
        """How long this track has been in the zone, as of ``now_seconds``."""
        return max(0.0, now_seconds - self.entered_at_seconds)


@dataclass(frozen=True, slots=True)
class ZoneVisit:
    """A completed stay in a zone, retained for session statistics only."""

    track_id: int
    entered_at_seconds: float
    exited_at_seconds: float

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.exited_at_seconds - self.entered_at_seconds)


class ZoneStateStore:
    """Tracks membership, entries, exits and dwell for every zone."""

    def __init__(self, zone_ids: Sequence[str], config: ZoneAnalysisConfig | None = None) -> None:
        self._config = config or ZoneAnalysisConfig()
        self._members: dict[str, dict[int, TrackZoneState]] = {zid: {} for zid in zone_ids}
        self._entries: dict[str, int] = dict.fromkeys(zone_ids, 0)
        self._exits: dict[str, int] = dict.fromkeys(zone_ids, 0)
        self._completed: dict[str, deque[ZoneVisit]] = {
            zid: deque(maxlen=self._config.max_completed_visits_per_zone) for zid in zone_ids
        }
        self._completed_counts: dict[str, int] = dict.fromkeys(zone_ids, 0)

    def update(
        self,
        memberships: Sequence[ZoneMembership],
        frame_index: int,
        timestamp_seconds: float,
    ) -> None:
        """Fold one frame's memberships into the running state."""
        for membership in memberships:
            self._ensure_zone(membership.zone_id)
            members = self._members[membership.zone_id]
            for track_id in membership.track_ids:
                state = members.get(track_id)
                if state is None:
                    members[track_id] = TrackZoneState(
                        track_id=track_id,
                        entered_at_frame=frame_index,
                        entered_at_seconds=timestamp_seconds,
                        last_inside_frame=frame_index,
                        last_inside_seconds=timestamp_seconds,
                    )
                    self._entries[membership.zone_id] += 1
                else:
                    state.last_inside_frame = frame_index
                    state.last_inside_seconds = timestamp_seconds

        self._expire(frame_index)

    def _expire(self, frame_index: int) -> None:
        """Retire members whose grace window has elapsed."""
        grace = self._config.exit_grace_frames
        for zone_id, members in self._members.items():
            expired = [
                track_id
                for track_id, state in members.items()
                if frame_index - state.last_inside_frame > grace
            ]
            for track_id in expired:
                state = members.pop(track_id)
                self._exits[zone_id] += 1
                self._completed_counts[zone_id] += 1
                self._completed[zone_id].append(
                    ZoneVisit(
                        track_id=track_id,
                        entered_at_seconds=state.entered_at_seconds,
                        # Timestamped at the last real observation, not at the
                        # moment the grace window ran out.
                        exited_at_seconds=state.last_inside_seconds,
                    )
                )

    def _ensure_zone(self, zone_id: str) -> None:
        if zone_id not in self._members:
            self._members[zone_id] = {}
            self._entries[zone_id] = 0
            self._exits[zone_id] = 0
            self._completed[zone_id] = deque(maxlen=self._config.max_completed_visits_per_zone)
            self._completed_counts[zone_id] = 0

    def occupancy(self, zone_id: str) -> int:
        """Tracks currently considered present, including those in grace."""
        return len(self._members.get(zone_id, {}))

    def member_ids(self, zone_id: str) -> frozenset[int]:
        return frozenset(self._members.get(zone_id, {}))

    def entries(self, zone_id: str) -> int:
        return self._entries.get(zone_id, 0)

    def exits(self, zone_id: str) -> int:
        return self._exits.get(zone_id, 0)

    def completed_visits(self, zone_id: str) -> int:
        """Total completed stays this session (may exceed the retained sample)."""
        return self._completed_counts.get(zone_id, 0)

    def recent_visits(self, zone_id: str) -> tuple[ZoneVisit, ...]:
        return tuple(self._completed.get(zone_id, ()))

    def dwell_stats(self, zone_id: str, now_seconds: float) -> tuple[float, float]:
        """``(average, maximum)`` dwell in seconds for tracks currently inside.

        Returns ``(0.0, 0.0)`` for an empty zone.
        """
        members = self._members.get(zone_id, {})
        if not members:
            return (0.0, 0.0)
        durations = [state.dwell_seconds(now_seconds) for state in members.values()]
        return (sum(durations) / len(durations), max(durations))

    def state_for(self, zone_id: str, track_id: int) -> TrackZoneState | None:
        return self._members.get(zone_id, {}).get(track_id)


class TrackMovementStore:
    """A short, time-bounded history of each track's normalized floor position.

    Speed is reported in **normalized frame units per second**: 0.1 means the
    person's floor point moved a tenth of the frame's width-equivalent in one
    second. This is image-plane movement. It is not metres per second and
    cannot be converted to one without camera calibration — a person near the
    camera covers far more pixels per step than one at the back of the room.
    """

    def __init__(self, config: ZoneAnalysisConfig | None = None) -> None:
        self._config = config or ZoneAnalysisConfig()
        self._samples: dict[int, deque[tuple[float, NormalizedPoint]]] = {}
        self._last_seen: dict[int, int] = {}

    def observe(
        self,
        track_id: int,
        point: NormalizedPoint,
        frame_index: int,
        timestamp_seconds: float,
    ) -> None:
        """Record where a track was at this moment."""
        samples = self._samples.get(track_id)
        if samples is None:
            samples = deque(maxlen=_MAX_SAMPLES_PER_TRACK)
            self._samples[track_id] = samples
        samples.append((timestamp_seconds, point))
        self._last_seen[track_id] = frame_index
        self._trim(samples)

    def _trim(self, samples: deque[tuple[float, NormalizedPoint]]) -> None:
        """Drop samples older than the speed window."""
        cutoff = samples[-1][0] - self._config.speed_window_seconds
        # Keep at least two samples so a speed can still be estimated.
        while len(samples) > 2 and samples[0][0] < cutoff:
            samples.popleft()

    def speed(self, track_id: int) -> float | None:
        """Mean speed over the window, or ``None`` without enough history.

        Computed as total path length divided by elapsed time, which reflects
        movement in place — someone dancing on the spot registers activity,
        where straight-line displacement over the window would report zero.
        """
        samples = self._samples.get(track_id)
        if samples is None or len(samples) < 2:
            return None

        elapsed = samples[-1][0] - samples[0][0]
        if elapsed <= 0:
            return None

        distance = sum(
            previous[1].distance_to(current[1])
            for previous, current in zip(samples, list(samples)[1:], strict=False)
        )
        return distance / elapsed

    def mean_speed(self, track_ids: frozenset[int] | set[int]) -> float:
        """Mean speed across the tracks that have enough history. ``0.0`` if none do."""
        speeds = [speed for tid in track_ids if (speed := self.speed(tid)) is not None]
        if not speeds:
            return 0.0
        return sum(speeds) / len(speeds)

    def prune(self, frame_index: int) -> None:
        """Forget tracks that have not been seen recently, then enforce the cap."""
        ttl = self._config.stale_track_ttl_frames
        stale = [
            track_id
            for track_id, last_seen in self._last_seen.items()
            if frame_index - last_seen >= ttl
        ]
        for track_id in stale:
            self._forget(track_id)

        excess = len(self._samples) - self._config.max_tracked_movements
        if excess > 0:
            oldest = sorted(self._last_seen.items(), key=lambda item: item[1])
            for track_id, _ in oldest[:excess]:
                self._forget(track_id)

    def _forget(self, track_id: int) -> None:
        self._samples.pop(track_id, None)
        self._last_seen.pop(track_id, None)

    def __len__(self) -> int:
        return len(self._samples)

    def __contains__(self, track_id: object) -> bool:
        return track_id in self._samples

"""Short-term trajectory history for tracked people.

Bounded on both axes, deliberately:

* **Per track** — a fixed-length ring of recent centre points, so one long-lived
  track cannot grow without limit.
* **Across tracks** — trails whose track has not been seen for ``ttl_frames``
  are discarded, and a hard ceiling evicts the least-recently-seen trail if the
  tracker somehow produces an unbounded number of ids.

Trails exist only in memory for the duration of a run and are never persisted.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

from atreides_vision.models import Point, Track

__all__ = ["TrajectoryStore"]


class TrajectoryStore:
    """Keeps the last N centre points for each currently interesting track."""

    def __init__(self, max_length: int = 25, ttl_frames: int = 45, max_trails: int = 512) -> None:
        """
        Args:
            max_length: Centre points retained per track.
            ttl_frames: Frames a track may be absent before its trail is dropped.
            max_trails: Hard ceiling on retained trails.
        """
        if max_length < 2:
            raise ValueError(f"max_length must be >= 2, got {max_length}")
        if ttl_frames < 1:
            raise ValueError(f"ttl_frames must be >= 1, got {ttl_frames}")
        if max_trails < 1:
            raise ValueError(f"max_trails must be >= 1, got {max_trails}")

        self._max_length = max_length
        self._ttl_frames = ttl_frames
        self._max_trails = max_trails
        self._points: dict[int, deque[Point]] = {}
        self._last_seen: dict[int, int] = {}

    def update(self, tracks: Iterable[Track], frame_index: int) -> None:
        """Append the centre of each track and prune anything stale."""
        for track in tracks:
            trail = self._points.get(track.track_id)
            if trail is None:
                trail = deque(maxlen=self._max_length)
                self._points[track.track_id] = trail
            trail.append(track.center)
            self._last_seen[track.track_id] = frame_index
        self.prune(frame_index)

    def prune(self, frame_index: int) -> None:
        """Drop trails past their TTL, then enforce the hard ceiling."""
        expired = [
            track_id
            for track_id, last_seen in self._last_seen.items()
            if frame_index - last_seen >= self._ttl_frames
        ]
        for track_id in expired:
            self._forget(track_id)

        if len(self._points) > self._max_trails:
            # Evict least-recently-seen first.
            ordered = sorted(self._last_seen.items(), key=lambda item: item[1])
            excess = len(self._points) - self._max_trails
            for track_id, _ in ordered[:excess]:
                self._forget(track_id)

    def _forget(self, track_id: int) -> None:
        self._points.pop(track_id, None)
        self._last_seen.pop(track_id, None)

    def trail(self, track_id: int) -> Sequence[Point]:
        """Recent centre points for ``track_id``, oldest first. Empty if unknown."""
        trail = self._points.get(track_id)
        return () if trail is None else tuple(trail)

    def clear(self) -> None:
        """Forget every trail."""
        self._points.clear()
        self._last_seen.clear()

    @property
    def track_ids(self) -> frozenset[int]:
        return frozenset(self._points)

    def __len__(self) -> int:
        return len(self._points)

    def __contains__(self, track_id: object) -> bool:
        return track_id in self._points

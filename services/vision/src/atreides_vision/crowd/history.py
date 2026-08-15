"""Bounded persistence tracking for spatial-cluster compositions.

Persistence is keyed by the **exact canonical member set**. If ``{1, 3, 7}``
remains a cluster frame after frame, its duration grows. The moment it becomes
``{1, 3, 7, 12}`` that is a different composition, and the new one starts at
zero.

Why exact membership
--------------------
The alternative — following a cluster through changing membership — requires
deciding that a cluster which gained and lost people is "still the same cluster".
That is a claim about a persistent entity, and from geometry alone there is no
basis for it. Two people leaving and two others arriving would produce a
"cluster" that has persisted for minutes while sharing no members with itself.

Exact membership refuses to make that claim. The cost is that persistence
resets on every join or leave, which makes it a conservative, short-lived
measure and **not** a measure of how long a group has existed. That trade is
deliberate: under-reporting duration is recoverable, inventing group continuity
is not.

Track-id churn has the same effect. A person re-acquired under a new id changes
the member set, so persistence resets — nothing here stitches ids together.

Memory
------
Compositions churn quickly in a busy scene, so state is bounded three ways:
a time-to-live on unseen compositions, a hard ceiling evicting the least
recently seen, and immediate disposal of compositions whose members are no
longer being tracked.
"""

from __future__ import annotations

from collections.abc import Iterator

from atreides_vision.behavior.models import SubjectRef
from atreides_vision.crowd.config import CrowdAnalysisConfig
from atreides_vision.crowd.models import ClusterMembers

__all__ = ["ClusterEpisode", "ClusterHistoryStore"]


class ClusterEpisode:
    """One continuous run of a single member composition.

    Deliberately not called a cluster *identity*: it is a run-local analytical
    episode, meaningful only within this process, and it never implies a
    persistent real-world group.
    """

    def __init__(self, members: ClusterMembers, timestamp_seconds: float) -> None:
        self._members = members
        self._started_at = timestamp_seconds
        self._last_seen = timestamp_seconds

    @property
    def members(self) -> ClusterMembers:
        return self._members

    @property
    def started_at_seconds(self) -> float:
        return self._started_at

    @property
    def last_seen_seconds(self) -> float:
        return self._last_seen

    def observe(self, timestamp_seconds: float, max_gap_seconds: float) -> float:
        """Record a sighting and return the persistence duration in seconds.

        A gap longer than ``max_gap_seconds`` restarts the episode: the
        composition was unobserved long enough that claiming it held together
        throughout would be a statement about time nobody watched.
        """
        gap = timestamp_seconds - self._last_seen
        if gap > max_gap_seconds or gap < 0:
            self._started_at = timestamp_seconds
        self._last_seen = timestamp_seconds
        return max(0.0, timestamp_seconds - self._started_at)

    def seconds_since_seen(self, timestamp_seconds: float) -> float:
        return timestamp_seconds - self._last_seen


class ClusterHistoryStore:
    """Every live cluster composition for one camera, with bounded lifetime."""

    def __init__(self, config: CrowdAnalysisConfig | None = None) -> None:
        self._config = config or CrowdAnalysisConfig()
        self._episodes: dict[ClusterMembers, ClusterEpisode] = {}

    def observe(self, members: ClusterMembers, timestamp_seconds: float) -> float:
        """Record that this composition exists now; return its persistence duration.

        A composition seen for the first time reports ``0.0`` — it has existed
        for no measured time yet, which is different from having existed for one
        frame's worth of it.
        """
        episode = self._episodes.get(members)
        if episode is None:
            episode = ClusterEpisode(members, timestamp_seconds)
            self._episodes[members] = episode
            return 0.0
        return episode.observe(timestamp_seconds, self._config.max_observation_gap_seconds)

    def get(self, members: ClusterMembers) -> ClusterEpisode | None:
        return self._episodes.get(members)

    def prune(self, timestamp_seconds: float, present: frozenset[SubjectRef]) -> None:
        """Forget compositions that are stale or whose members have left.

        Membership-based disposal runs first because it is the sharper rule:
        once a member is no longer tracked at all, the composition cannot recur,
        so holding it for the full TTL would only waste memory.
        """
        gone = [
            members
            for members in self._episodes
            if any(member not in present for member in members)
        ]
        for members in gone:
            del self._episodes[members]

        ttl = self._config.stale_cluster_ttl_seconds
        stale = [
            members
            for members, episode in self._episodes.items()
            if episode.seconds_since_seen(timestamp_seconds) >= ttl
        ]
        for members in stale:
            del self._episodes[members]

        excess = len(self._episodes) - self._config.max_tracked_clusters
        if excess > 0:
            oldest = sorted(self._episodes.items(), key=lambda item: item[1].last_seen_seconds)
            for members, _ in oldest[:excess]:
                del self._episodes[members]

    def compositions(self) -> tuple[ClusterMembers, ...]:
        return tuple(self._episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def __contains__(self, members: object) -> bool:
        return members in self._episodes

    def __iter__(self) -> Iterator[ClusterMembers]:
        return iter(self._episodes)

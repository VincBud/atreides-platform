"""Stability of spatial-cluster composition, measured from the crowd stream.

V0.3.5 addresses a cluster by its **exact member set**. Adding or losing one
member makes a different cluster, and its persistence restarts. That is a
deliberate property of the measurement layer, and this module quantifies what it
costs in practice.

Two things are counted, and they must not be confused:

**A composition** is a canonical member set — the tuple of ``(camera_id,
track_id)`` members. **A composition episode** is a maximal run of consecutive
frames in which that exact set was reported as a cluster.

``cluster_composition_births_per_minute`` counts composition episodes started
per minute of processed source time. A composition change is **not** a
real-world group event. It happens when someone joins, when someone leaves, when
a member's track id fragments and is re-acquired under a new one, and when two
members drift a few centimetres across the connectivity radius and back.

Persistence is read from the analyser's own ``persistence_duration`` field
rather than recomputed here, so the reported figure is the one the production
layer would hand a downstream consumer, not a validation-only reinterpretation
of it.

Nothing here says anything about social groups, and the survival fractions below
are descriptive — no threshold in this module carries an acceptance meaning.

Memory
------
One entry per *currently open* composition, one float per *finished* composition
episode, plus a set of every distinct member set seen. The last of those scales
with unique compositions rather than clip length, which in a dense scene where
membership churns every few frames is the largest structure this package builds.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from atreides_vision.validation.distributions import Distribution

__all__ = [
    "PERSISTENCE_SURVIVAL_THRESHOLDS_SECONDS",
    "CompositionRef",
    "CrowdMetrics",
    "CrowdMetricsAccumulator",
]

#: A cluster composition: camera id plus its members' track ids, canonically
#: ordered. Members are tracklets, so this is not a group of people.
CompositionRef = tuple[str, tuple[int, ...]]

#: Durations the composition-survival fractions are reported against. A clip
#: shorter than one of these simply reports it as not applicable rather than as
#: a zero, which would read as a failure rather than as an unanswerable question.
PERSISTENCE_SURVIVAL_THRESHOLDS_SECONDS: tuple[float, ...] = (1.0, 2.0, 5.0)

_PRECISION = 4


def _round(value: float | None, digits: int = _PRECISION) -> float | None:
    return None if value is None else round(value, digits)


@dataclass(slots=True)
class _OpenComposition:
    last_frame_index: int
    max_persistence_seconds: float


@dataclass(frozen=True, slots=True)
class CrowdMetrics:
    """Cluster occupancy, size and composition stability for one clip."""

    frames_with_summary: int
    frames_processed: int
    clusters_per_frame: Distribution
    observed_subjects_per_frame: Distribution
    clustered_subjects_per_frame: Distribution
    unclustered_subjects_per_frame: Distribution
    largest_cluster_size: int
    cluster_size_histogram: Mapping[int, int]
    total_cluster_records: int
    unique_compositions: int
    composition_episodes: int
    composition_births_per_minute: float | None
    persistence_seconds: Distribution
    #: Fraction of composition episodes whose reported persistence exceeded each
    #: threshold. ``None`` where the clip is too short for the question.
    survival_fractions: Mapping[float, float | None]
    space: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "measurement_class": "automated_proxy_system_metrics",
            "identity_scope": "camera_local_tracks",
            "space": self.space,
            "frames_with_crowd_summary": self.frames_with_summary,
            "clusters_per_frame": self.clusters_per_frame.to_record(),
            "observed_subjects_per_frame": self.observed_subjects_per_frame.to_record(),
            "clustered_subjects_per_frame": self.clustered_subjects_per_frame.to_record(),
            "unclustered_subjects_per_frame": self.unclustered_subjects_per_frame.to_record(),
            "largest_cluster_size": self.largest_cluster_size,
            "cluster_size_histogram": {
                str(size): count for size, count in sorted(self.cluster_size_histogram.items())
            },
            "total_cluster_records": self.total_cluster_records,
            "unique_compositions": self.unique_compositions,
            "composition_episodes": self.composition_episodes,
            "cluster_composition_births_per_minute": _round(self.composition_births_per_minute),
            "composition_persistence_seconds": self.persistence_seconds.to_record(),
            "composition_survival_fractions": {
                f"over_{threshold:g}s": _round(fraction)
                for threshold, fraction in sorted(self.survival_fractions.items())
            },
            "definitions": {
                "composition": (
                    "a canonical exact member set of camera-local track ids. Members are "
                    "tracklets, so member_count is not a count of unique people."
                ),
                "composition_episode": (
                    "a maximal run of consecutive frames in which that exact member set was "
                    "reported as a cluster."
                ),
                "cluster_composition_births_per_minute": (
                    "composition episodes started per minute of processed source time. A "
                    "composition change is NOT automatically a real-world group event: it "
                    "occurs on a join, a leave, a member track fragmenting, or two members "
                    "drifting across the connectivity radius and back."
                ),
                "composition_persistence_seconds": (
                    "the analyser's own persistence_duration at the end of each episode, not "
                    "a validation-only recomputation."
                ),
            },
            "interpretation_boundary": (
                "a spatial cluster is a geometric arrangement at an instant. Composition "
                "stability is not social group stability."
            ),
        }


class CrowdMetricsAccumulator:
    """Streams a crowd output file into cluster-stability metrics.

    Args:
        frames_processed: Frames the pipeline processed, so frames the crowd
            stream did not cover count as zero clusters rather than vanishing.
        source_fps: Source frame rate, for the per-minute rate.
        clip_duration_seconds: Processed source duration, used to decide which
            survival thresholds the clip is long enough to answer.
    """

    def __init__(
        self,
        frames_processed: int,
        source_fps: float,
        clip_duration_seconds: float | None = None,
    ) -> None:
        self._frames_processed = frames_processed
        self._source_fps = source_fps
        self._clip_duration = (
            clip_duration_seconds
            if clip_duration_seconds is not None
            else (frames_processed / source_fps if source_fps > 0 else 0.0)
        )

        self._cluster_counts: list[int] = []
        self._observed_counts: list[int] = []
        self._clustered_counts: list[int] = []
        self._unclustered_counts: list[int] = []
        self._largest_cluster = 0
        self._size_histogram: Counter[int] = Counter()
        self._total_cluster_records = 0

        self._open: dict[CompositionRef, _OpenComposition] = {}
        self._compositions_seen: set[CompositionRef] = set()
        self._episode_persistences: list[float] = []
        self._episode_count = 0

        self._current_frame: int | None = None
        self._space: str | None = None

    def observe_frame_summary(
        self,
        frame_index: int,
        observed_subject_count: int,
        cluster_count: int,
        clustered_subject_count: int,
        isolated_subject_count: int,
    ) -> None:
        """Record one ``crowd_frame`` summary."""
        self._advance_to(frame_index)
        self._cluster_counts.append(cluster_count)
        self._observed_counts.append(observed_subject_count)
        self._clustered_counts.append(clustered_subject_count)
        self._unclustered_counts.append(isolated_subject_count)

    def observe_cluster(
        self,
        camera_id: str,
        member_track_ids: Sequence[int],
        frame_index: int,
        persistence_seconds: float,
    ) -> None:
        """Record one ``spatial_cluster``."""
        self._advance_to(frame_index)
        self._total_cluster_records += 1
        size = len(member_track_ids)
        self._size_histogram[size] += 1
        self._largest_cluster = max(self._largest_cluster, size)

        composition: CompositionRef = (camera_id, tuple(sorted(member_track_ids)))
        self._compositions_seen.add(composition)

        episode = self._open.get(composition)
        if episode is None or frame_index > episode.last_frame_index + 1:
            if episode is not None:
                self._finish(episode)
            self._open[composition] = _OpenComposition(
                last_frame_index=frame_index, max_persistence_seconds=persistence_seconds
            )
            return
        episode.last_frame_index = frame_index
        # Persistence is monotonic within an episode, but taking the max rather
        # than the last value keeps the metric correct even if a stream is ever
        # written out of order.
        episode.max_persistence_seconds = max(episode.max_persistence_seconds, persistence_seconds)

    def consume(self, records: Iterable[Mapping[str, Any]]) -> CrowdMetricsAccumulator:
        """Feed a crowd stream, reading the header for its space."""
        for record in records:
            record_type = record.get("record_type")
            if record_type == "run_metadata":
                space = record.get("space")
                self._space = None if space is None else str(space)
                continue
            if record_type == "crowd_frame":
                summary = record.get("summary")
                if not isinstance(summary, dict):
                    continue
                self.observe_frame_summary(
                    frame_index=int(record["frame_index"]),
                    observed_subject_count=int(summary.get("observed_subject_count", 0)),
                    cluster_count=int(summary.get("cluster_count", 0)),
                    clustered_subject_count=int(summary.get("clustered_subject_count", 0)),
                    isolated_subject_count=int(summary.get("isolated_subject_count", 0)),
                )
                continue
            if record_type != "spatial_cluster":
                continue
            members = record.get("members")
            if not isinstance(members, list) or not members:
                continue
            metrics = record.get("metrics")
            persistence = 0.0
            if isinstance(metrics, dict):
                measured = metrics.get("persistence_duration")
                if isinstance(measured, dict) and measured.get("value") is not None:
                    persistence = float(measured["value"])
            self.observe_cluster(
                camera_id=str(record["camera_id"]),
                member_track_ids=[int(member["track_id"]) for member in members],
                frame_index=int(record["frame_index"]),
                persistence_seconds=persistence,
            )
        return self

    def _advance_to(self, frame_index: int) -> None:
        """Close compositions absent from the frame that just ended."""
        if frame_index == self._current_frame:
            return
        if self._current_frame is not None:
            stale = [
                composition
                for composition, episode in self._open.items()
                if episode.last_frame_index < self._current_frame
            ]
            for composition in stale:
                self._finish(self._open.pop(composition))
        self._current_frame = frame_index

    def _finish(self, episode: _OpenComposition) -> None:
        self._episode_persistences.append(episode.max_persistence_seconds)
        self._episode_count += 1

    def finalize(self) -> CrowdMetrics:
        """Close every still-open composition, then summarise."""
        for composition in list(self._open):
            self._finish(self._open.pop(composition))

        frames_with_summary = len(self._cluster_counts)
        empty = max(self._frames_processed - frames_with_summary, 0)
        cluster_counts = [*self._cluster_counts, *([0] * empty)]
        observed = [*self._observed_counts, *([0] * empty)]
        clustered = [*self._clustered_counts, *([0] * empty)]
        unclustered = [*self._unclustered_counts, *([0] * empty)]

        duration_seconds = (
            self._frames_processed / self._source_fps if self._source_fps > 0 else 0.0
        )
        minutes = duration_seconds / 60.0

        survival: dict[float, float | None] = {}
        for threshold in PERSISTENCE_SURVIVAL_THRESHOLDS_SECONDS:
            if self._clip_duration < threshold or self._episode_count == 0:
                # A 17 s clip can answer "how many held for 5 s"; a 3 s clip
                # cannot, and reporting 0.0 would read as an answer.
                survival[threshold] = None
                continue
            survivors = sum(1 for value in self._episode_persistences if value > threshold)
            survival[threshold] = survivors / self._episode_count

        return CrowdMetrics(
            frames_with_summary=frames_with_summary,
            frames_processed=self._frames_processed,
            clusters_per_frame=Distribution.from_values(cluster_counts),
            observed_subjects_per_frame=Distribution.from_values(observed),
            clustered_subjects_per_frame=Distribution.from_values(clustered),
            unclustered_subjects_per_frame=Distribution.from_values(unclustered),
            largest_cluster_size=self._largest_cluster,
            cluster_size_histogram=dict(self._size_histogram),
            total_cluster_records=self._total_cluster_records,
            unique_compositions=len(self._compositions_seen),
            composition_episodes=self._episode_count,
            composition_births_per_minute=(
                (self._episode_count / minutes) if minutes > 0 else None
            ),
            persistence_seconds=Distribution.from_values(self._episode_persistences),
            survival_fractions=survival,
            space=self._space,
        )

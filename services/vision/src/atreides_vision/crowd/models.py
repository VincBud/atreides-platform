"""Domain models for spatial-cluster measurement.

A **spatial cluster** means exactly one thing:

> Several camera-local observations were spatially connected under a configured
> geometric rule, at one moment in time.

It does **not** mean those people know each other, arrived together, are
socially associated, or form any persistent real-world group. A queue at a bar,
four strangers waiting for the same door, and a family are all identical to this
layer, because geometry is all it measures. Nothing here is named ``group`` for
that reason.

Identity boundary
-----------------
Members are **camera-local** subjects. A camera-local track id is not a unique
real-world person, so ``member_count`` counts *tracklets*, not people: an
occlusion that re-ids someone inflates it, and two cameras seeing the same
person are never merged. Clusters never span cameras.

There is no cluster identity. A cluster is addressed by its canonical member
set, which is a fact about the current frame rather than a claim about a
persistent entity.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.errors import CrowdError

__all__ = [
    "ClusterMembers",
    "CrowdFrame",
    "CrowdFrameSummary",
    "SpatialClusterSnapshot",
    "canonical_members",
]

#: A cluster's members, in canonical order. A tuple rather than a set so it is
#: both hashable — usable as a history key — and deterministically ordered for
#: output. Set iteration order must never reach a serialised record.
ClusterMembers = tuple[SubjectRef, ...]


def canonical_members(subjects: Sequence[SubjectRef]) -> ClusterMembers:
    """Order subjects canonically and reject an impossible member set.

    Raises:
        CrowdError: for duplicates, for fewer than two members, or for members
            spanning cameras. Cross-camera clustering would let one real person
            visible in two cameras be counted twice and clustered with
            themselves, inflating size and distorting every geometric measure.
    """
    if len(subjects) < 2:
        raise CrowdError(f"A spatial cluster needs at least two members, got {len(subjects)}")
    if len(set(subjects)) != len(subjects):
        raise CrowdError(f"A subject cannot appear twice in one cluster: {subjects}")

    cameras = {subject.camera_id for subject in subjects}
    if len(cameras) > 1:
        raise CrowdError(
            f"Spatial clusters are same-camera only, got members from {sorted(cameras)}. "
            "Without multi-camera association, one real person seen by two cameras would "
            "be clustered with themselves and counted twice."
        )
    return tuple(sorted(subjects, key=lambda ref: (ref.camera_id, ref.track_id)))


@dataclass(frozen=True, slots=True)
class SpatialClusterSnapshot:
    """One spatial cluster, at one frame: its geometry and derived measures.

    Every field is a measurement. There is deliberately nothing here describing
    what the arrangement might mean.
    """

    members: ClusterMembers
    frame_index: int
    timestamp_seconds: float
    #: Arithmetic mean of the members' projected anchors.
    centroid: MotionPoint
    #: RMS distance from the centroid — how spread out the cluster is.
    spatial_spread: float
    #: Mean separation over all member pairs. ``None`` below two members.
    mean_pair_distance: float | None = None
    #: Widest member pair. Routinely exceeds the cluster radius, because
    #: connectivity chains — see :mod:`atreides_vision.crowd.clustering`.
    max_pair_distance: float | None = None
    #: Mean and peak of the members' individual local densities.
    mean_local_density: float | None = None
    max_local_density: float | None = None
    #: Mean resultant length of the moving members' headings, in ``[0, 1]``.
    #: ``None`` when too few members are moving to compare directions.
    motion_heading_coherence: float | None = None
    #: Continuous seconds this exact member set has been a cluster.
    persistence_duration: float = 0.0
    #: Zones every member is inside, in the first member's configuration order.
    common_zone_ids: tuple[str, ...] = ()
    #: Every zone any member is inside.
    member_zone_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise CrowdError(
                f"SpatialClusterSnapshot.frame_index must be non-negative, got {self.frame_index!r}"
            )
        if not math.isfinite(self.spatial_spread) or self.spatial_spread < 0:
            raise CrowdError(
                f"spatial_spread must be finite and non-negative, got {self.spatial_spread!r}"
            )
        if self.persistence_duration < 0:
            raise CrowdError(
                f"persistence_duration must be non-negative, got {self.persistence_duration!r}"
            )
        if self.motion_heading_coherence is not None and not (
            0.0 <= self.motion_heading_coherence <= 1.0
        ):
            raise CrowdError(
                f"motion_heading_coherence must be in [0, 1], got {self.motion_heading_coherence!r}"
            )

    @property
    def member_count(self) -> int:
        """Camera-local tracked subjects in this cluster.

        **Not guaranteed to equal unique real-world people.** Occlusion
        fragmentation can split one person into several tracks and inflate this,
        and overlapping cameras are never merged.
        """
        return len(self.members)

    @property
    def space(self) -> MotionSpace:
        return self.centroid.space

    @property
    def camera_id(self) -> str:
        return self.members[0].camera_id


@dataclass(frozen=True, slots=True)
class CrowdFrameSummary:
    """Neutral frame-level counts.

    Counts only. There is no crowding status, no density verdict and no
    safe/unsafe label, because none of those can be derived from geometry alone.
    """

    observed_subject_count: int
    cluster_count: int
    clustered_subject_count: int
    largest_cluster_size: int = 0

    @property
    def isolated_subject_count(self) -> int:
        """Observed subjects in no cluster of the configured minimum size."""
        return self.observed_subject_count - self.clustered_subject_count

    def to_record(self) -> dict[str, Any]:
        return {
            "observed_subject_count": self.observed_subject_count,
            "cluster_count": self.cluster_count,
            "clustered_subject_count": self.clustered_subject_count,
            "isolated_subject_count": self.isolated_subject_count,
            "largest_cluster_size": self.largest_cluster_size,
        }


@dataclass(frozen=True, slots=True)
class CrowdFrame:
    """Every spatial cluster measured on one frame, in deterministic order."""

    frame_index: int
    timestamp_seconds: float
    clusters: tuple[SpatialClusterSnapshot, ...] = field(default_factory=tuple)
    summary: CrowdFrameSummary = field(default_factory=lambda: CrowdFrameSummary(0, 0, 0, 0))
    #: Pairs the radius graph examined this frame — ``N(N-1)/2``.
    pair_comparisons: int = 0
    #: Pairs found within the cluster radius.
    edge_count: int = 0

    @classmethod
    def from_clusters(
        cls,
        frame_index: int,
        timestamp_seconds: float,
        clusters: Sequence[SpatialClusterSnapshot],
        observed_subject_count: int,
        pair_comparisons: int = 0,
        edge_count: int = 0,
    ) -> Self:
        ordered = tuple(clusters)
        clustered = sum(cluster.member_count for cluster in ordered)
        return cls(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            clusters=ordered,
            summary=CrowdFrameSummary(
                observed_subject_count=observed_subject_count,
                cluster_count=len(ordered),
                clustered_subject_count=clustered,
                largest_cluster_size=max((cluster.member_count for cluster in ordered), default=0),
            ),
            pair_comparisons=pair_comparisons,
            edge_count=edge_count,
        )

    def for_members(self, members: Sequence[SubjectRef]) -> SpatialClusterSnapshot | None:
        """The cluster with exactly this member set, whatever order it is given in."""
        key = canonical_members(members)
        for cluster in self.clusters:
            if cluster.members == key:
                return cluster
        return None

    def containing(self, subject: SubjectRef) -> SpatialClusterSnapshot | None:
        """The cluster this subject belongs to, or ``None``.

        At most one: connected components partition the frame's subjects.
        """
        for cluster in self.clusters:
            if subject in cluster.members:
                return cluster
        return None

    def __len__(self) -> int:
        return len(self.clusters)

    def __iter__(self) -> Iterator[SpatialClusterSnapshot]:
        return iter(self.clusters)

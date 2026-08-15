"""The spatial-cluster analyzer — the single component a pipeline talks to.

It owns the whole crowd path for one camera session::

    BehaviorFrame -> radius graph -> connected components -> cluster measures

and returns a :class:`~atreides_vision.crowd.models.CrowdFrame` per frame,
exactly as the behaviour, zone and interaction analyzers do for their layers.

Why it consumes behaviour snapshots
-----------------------------------
Its input is the **already-projected** output of behaviour analysis. There is
one projection in the process, one motion history, and no path by which crowd
code could acquire an opinion about RF-DETR, OC-SORT or pixels.

Relationship to the interaction layer
-------------------------------------
Both consume ``BehaviorFrame`` and both need pairwise distances, but crowd
analysis does **not** consume ``InteractionFrame``. Interaction measures only
pairs that survive its own candidate gate, so depending on it would silently
tie cluster connectivity to an unrelated setting and make crowd output require
interaction to be enabled. Distances are one ``hypot`` each; recomputing them is
cheaper than the coupling. Both layers bottom out in the same
``MotionPoint.distance_to`` primitive, so there is no second distance
implementation.

Temporal pair state is likewise not duplicated: closing speed and proximity
duration stay in V0.3.4. This layer keeps only cluster-composition persistence.

Same camera only
----------------
One analyzer serves one camera, and every cluster is drawn from that camera's
own frame. Cross-camera clustering is rejected rather than merely absent.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.behavior.models import BehaviorFrame, BehaviorSubjectSnapshot, SubjectRef
from atreides_vision.crowd.clustering import build_cluster_graph
from atreides_vision.crowd.config import CrowdAnalysisConfig
from atreides_vision.crowd.geometry import (
    centroid,
    local_density,
    max_pair_distance,
    mean_pair_distance,
    motion_heading_coherence,
    neighbours_within,
    rms_radius,
)
from atreides_vision.crowd.history import ClusterHistoryStore
from atreides_vision.crowd.models import (
    ClusterMembers,
    CrowdFrame,
    SpatialClusterSnapshot,
    canonical_members,
)
from atreides_vision.errors import ConfigurationError, CrowdError
from atreides_vision.interaction.models import SpatialContext

logger = logging.getLogger(__name__)

__all__ = ["SpatialClusterAnalyzer"]


class SpatialClusterAnalyzer:
    """Per-frame spatial-cluster measurement for one camera session.

    Args:
        context: The coordinate frame these measurements are made in. Fixes the
            camera, the space, and — for metric analysis — the venue and
            surface. Reused from the interaction layer rather than redefined, so
            there is one compatibility rule in the codebase, not two.
        config: Radii, thresholds and memory bounds. Development defaults.

    Raises:
        ConfigurationError: if ``config`` declares an ``intended_space`` the
            context does not provide. A normalized radius read as metres would
            connect only people already touching; a metric one read as frame
            fractions would connect the entire image into a single cluster.
    """

    def __init__(
        self,
        context: SpatialContext,
        config: CrowdAnalysisConfig | None = None,
    ) -> None:
        self._context = context
        self._config = config or CrowdAnalysisConfig()
        self._require_matching_space()

        self._history = ClusterHistoryStore(self._config)
        self._latest: CrowdFrame | None = None
        self._clusters_measured = 0
        self._pair_comparisons = 0
        self._edges_found = 0

    def _require_matching_space(self) -> None:
        intended = self._config.intended_space
        actual = self._context.space
        if intended is not None and intended is not actual:
            raise ConfigurationError(
                f"Crowd configuration is written for {intended.value} coordinates but the "
                f"spatial context is {actual.value}. Cluster and density radii would be "
                "interpreted in the wrong unit. Use "
                "CrowdAnalysisConfig.for_ground_plane_metres() with a calibrated context, "
                "or for_normalized_image() with a normalized one."
            )

    @property
    def context(self) -> SpatialContext:
        return self._context

    @property
    def config(self) -> CrowdAnalysisConfig:
        return self._config

    @property
    def camera_id(self) -> str:
        return self._context.camera_id

    @property
    def space(self) -> MotionSpace:
        return self._context.space

    @property
    def latest_frame(self) -> CrowdFrame | None:
        return self._latest

    @property
    def clusters_measured(self) -> int:
        """Cluster measurements produced this session."""
        return self._clusters_measured

    @property
    def pair_comparisons(self) -> int:
        """Pairs examined by the radius graph across the run."""
        return self._pair_comparisons

    @property
    def edges_found(self) -> int:
        """Pairs found within the cluster radius across the run."""
        return self._edges_found

    @property
    def tracked_compositions(self) -> int:
        """Cluster compositions currently holding persistence state."""
        return len(self._history)

    def update(self, frame: BehaviorFrame) -> CrowdFrame:
        """Measure every spatial cluster in one behaviour frame.

        Args:
            frame: The behaviour analyzer's output. Its snapshots carry
                projected positions, motion vectors and zone labels — everything
                cluster measurement needs.

        Returns:
            Clusters meeting the configured minimum size, in deterministic
            order, plus neutral frame-level counts.

        Raises:
            CrowdError: if a snapshot belongs to a different camera or a
                different coordinate space than this analyzer's context. Both
                mean the caller wired something wrong, and measuring anyway
                would produce plausible numbers for an invalid arrangement.
        """
        subjects = self._validated_snapshots(frame.snapshots)
        positions = {snapshot.subject: snapshot.position for snapshot in subjects}
        by_subject = {snapshot.subject: snapshot for snapshot in subjects}

        graph = build_cluster_graph(positions, self._config.cluster_radius)
        components = graph.components(self._config.minimum_cluster_size)

        clusters = [
            self._measure(
                canonical_members(component),
                by_subject,
                positions,
                frame.frame_index,
                frame.timestamp_seconds,
            )
            for component in components
        ]

        self._history.prune(frame.timestamp_seconds, frozenset(positions))

        result = CrowdFrame.from_clusters(
            frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds,
            clusters=clusters,
            observed_subject_count=len(subjects),
            pair_comparisons=graph.comparisons,
            edge_count=graph.edge_count,
        )
        self._clusters_measured += len(clusters)
        self._pair_comparisons += graph.comparisons
        self._edges_found += graph.edge_count
        self._latest = result
        return result

    def _validated_snapshots(
        self, snapshots: Sequence[BehaviorSubjectSnapshot]
    ) -> list[BehaviorSubjectSnapshot]:
        """Reject anything that cannot legitimately be clustered in this context."""
        for snapshot in snapshots:
            if snapshot.subject.camera_id != self._context.camera_id:
                raise CrowdError(
                    f"{snapshot.subject} does not belong to camera "
                    f"'{self._context.camera_id}'. Spatial clustering is camera-local: "
                    "clustering across cameras would count one real person twice and "
                    "cluster them with themselves."
                )
            if snapshot.space is not self._context.space:
                raise CrowdError(
                    f"{snapshot.subject} is in {snapshot.space.value} but this analyzer "
                    f"measures in {self._context.space.value}."
                )
        return list(snapshots)

    def _measure(
        self,
        members: ClusterMembers,
        by_subject: dict[SubjectRef, BehaviorSubjectSnapshot],
        all_positions: dict[SubjectRef, MotionPoint],
        frame_index: int,
        timestamp_seconds: float,
    ) -> SpatialClusterSnapshot:
        """Everything measurable about one cluster at this frame."""
        member_snapshots = [by_subject[member] for member in members]
        positions = [snapshot.position for snapshot in member_snapshots]
        middle = centroid(positions)

        densities = [self._local_density_for(subject, all_positions) for subject in members]
        motions = [snapshot.motion for snapshot in member_snapshots if snapshot.motion is not None]
        coherence = (
            motion_heading_coherence(motions, self._config.min_motion_displacement)
            if len(motions) >= self._config.minimum_moving_members
            else None
        )

        return SpatialClusterSnapshot(
            members=members,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            centroid=middle,
            spatial_spread=rms_radius(positions, middle),
            mean_pair_distance=mean_pair_distance(positions),
            max_pair_distance=max_pair_distance(positions),
            mean_local_density=sum(densities) / len(densities),
            max_local_density=max(densities),
            motion_heading_coherence=coherence,
            persistence_duration=self._history.observe(members, timestamp_seconds),
            common_zone_ids=_common_zones(member_snapshots),
            member_zone_ids=_union_zones(member_snapshots),
        )

    def _local_density_for(
        self, subject: SubjectRef, all_positions: dict[SubjectRef, MotionPoint]
    ) -> float:
        """One subject's local density, sampled against every observed subject.

        Neighbours are counted across the whole frame, not only within the
        cluster: density describes how concentrated the neighbourhood *is*, and
        restricting it to fellow members would make it a restatement of cluster
        membership rather than an independent measure.
        """
        others = [position for other, position in all_positions.items() if other != subject]
        count = neighbours_within(all_positions[subject], others, self._config.density_radius)
        return local_density(count, self._config.density_radius, self._context.space)

    def log_summary(self) -> None:
        """Log a final summary. Called once at the end of a run."""
        logger.info(
            "Spatial clusters:    %d measured from %d pair comparisons in %s space",
            self._clusters_measured,
            self._pair_comparisons,
            self.space.value,
        )


def _common_zones(snapshots: Sequence[BehaviorSubjectSnapshot]) -> tuple[str, ...]:
    """Zones every member is inside, in the first member's order.

    Context only. Members sharing a zone are in the same operator-drawn region,
    which can be an entire dancefloor — it says nothing about how close they are,
    and it is never what put them in a cluster.
    """
    if not snapshots or any(not snapshot.zone_ids for snapshot in snapshots):
        return ()
    shared = set(snapshots[0].zone_ids)
    for snapshot in snapshots[1:]:
        shared &= set(snapshot.zone_ids)
    return tuple(zone for zone in snapshots[0].zone_ids if zone in shared)


def _union_zones(snapshots: Sequence[BehaviorSubjectSnapshot]) -> tuple[str, ...]:
    """Every zone any member is inside, in first-seen order."""
    seen: dict[str, None] = {}
    for snapshot in snapshots:
        for zone in snapshot.zone_ids:
            seen.setdefault(zone, None)
    return tuple(seen)

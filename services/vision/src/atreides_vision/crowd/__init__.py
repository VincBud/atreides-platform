"""Spatial clusters and crowd dynamics — V0.3.5.

Turns the behaviour layer's per-subject snapshots into **neutral spatial
aggregates**: which camera-local subjects are spatially connected, where the
cluster's centre is, how spread out it is, how concentrated the local
neighbourhood is, how similarly its members are travelling, and how long that
exact composition has persisted.

A spatial cluster is not a group
--------------------------------
It means only that **several camera-local observations were spatially connected
under a configured geometric rule at one moment**. It does not mean those people
know each other, arrived together, are socially associated, or form a persistent
real-world group. A bar queue, four strangers at a door and a family are
indistinguishable here, because geometry is all that is measured — which is why
nothing in this package is named ``group``.

There is deliberately no crowd risk, danger, panic, unrest, escalation,
overcrowding or safety output. Those are interpretations requiring evidence
Atreides does not have, and a package that measured *and* judged would make both
jobs impossible to audit.

Identity boundary
-----------------
Members are camera-local subjects, and **clusters never span cameras**: without
multi-camera association, one real person visible in two cameras would be
counted twice and clustered with themselves, inflating size and distorting every
geometric measure. A camera-local track id is also not a unique person, so
member counts are counts of tracklets.

No cluster identity exists. A cluster is addressed by its canonical member set,
and persistence belongs to that exact set — see
:mod:`atreides_vision.crowd.history` for why following a cluster through
changing membership would be a claim this layer cannot support.

Layering
--------
``tracks → BehaviorAnalyzer → BehaviorFrame → SpatialClusterAnalyzer``

Cluster state lives here, never in the behaviour analyzer. This package imports
no detector, no tracker and no zone engine, and performs no pixel arithmetic. It
reuses :class:`~atreides_vision.interaction.models.SpatialContext` rather than
defining a second compatibility rule, and shares the JSONL writer and interval
sampler with the other streams.
"""

from __future__ import annotations

from atreides_vision.crowd.analyzer import SpatialClusterAnalyzer
from atreides_vision.crowd.clustering import (
    ClusterGraph,
    build_cluster_graph,
    connected_components,
)
from atreides_vision.crowd.config import CrowdAnalysisConfig
from atreides_vision.crowd.geometry import (
    centroid,
    local_density,
    max_pair_distance,
    mean_pair_distance,
    motion_heading_coherence,
    neighbours_within,
    rms_radius,
    separation,
)
from atreides_vision.crowd.history import ClusterEpisode, ClusterHistoryStore
from atreides_vision.crowd.models import (
    ClusterMembers,
    CrowdFrame,
    CrowdFrameSummary,
    SpatialClusterSnapshot,
    canonical_members,
)
from atreides_vision.crowd.reporting import (
    CROWD_SCHEMA_VERSION,
    CrowdRunMetadata,
    CrowdWriter,
    JsonlCrowdWriter,
    crowd_records,
)

__all__ = [
    "CROWD_SCHEMA_VERSION",
    "ClusterEpisode",
    "ClusterGraph",
    "ClusterHistoryStore",
    "ClusterMembers",
    "CrowdAnalysisConfig",
    "CrowdFrame",
    "CrowdFrameSummary",
    "CrowdRunMetadata",
    "CrowdWriter",
    "JsonlCrowdWriter",
    "SpatialClusterAnalyzer",
    "SpatialClusterSnapshot",
    "build_cluster_graph",
    "canonical_members",
    "centroid",
    "connected_components",
    "crowd_records",
    "local_density",
    "max_pair_distance",
    "mean_pair_distance",
    "motion_heading_coherence",
    "neighbours_within",
    "rms_radius",
    "separation",
]

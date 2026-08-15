"""Radius-connectivity clustering: which subjects are spatially connected.

The rule, in full
-----------------
Build an undirected graph over the compatible subjects in one frame:

* one node per subject;
* an edge between two subjects when their separation is **at or below** the
  cluster radius.

A spatial cluster is a **connected component** of that graph with at least
``minimum_cluster_size`` members.

Chaining is intended, not a defect
----------------------------------
Connectivity is transitive. If A is within the radius of B, and B of C, then
A, B and C are one cluster **even when A and C are far further apart than the
radius**. A queue of people standing a metre apart is one cluster however long
the queue is.

This is the mathematical meaning of the chosen rule, so the parameter is called
``cluster_radius`` and never a maximum cluster diameter or size. The actual
extent of a cluster is reported separately as ``max_pair_distance``, which is
routinely larger than the radius.

Each subject belongs to **at most one cluster per frame** — connected components
partition the graph. Overlapping or community-style clustering, where a person
can belong to several groups at once, is a different model and is not
implemented.

Cost
----
Edge construction compares every pair, so it is O(N²) in the subjects on one
frame. At realistic per-camera counts that is a few hundred hypot calls per
frame; see the milestone document for the measured cost and for where a spatial
index would become worthwhile.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

from atreides_vision.behavior.coordinates import MotionPoint
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.crowd.geometry import separation
from atreides_vision.errors import CrowdError

__all__ = ["ClusterGraph", "build_cluster_graph", "connected_components"]


class ClusterGraph:
    """The radius graph for one frame, and the components derived from it.

    Holds the edge count as well as the components because it is the honest
    measure of how much work the frame cost, and because a caller reporting
    scaling needs it.
    """

    def __init__(
        self,
        subjects: Sequence[SubjectRef],
        adjacency: Mapping[SubjectRef, set[SubjectRef]],
        edge_count: int,
        comparisons: int,
    ) -> None:
        self._subjects = tuple(subjects)
        self._adjacency = adjacency
        self._edge_count = edge_count
        self._comparisons = comparisons

    @property
    def subjects(self) -> tuple[SubjectRef, ...]:
        return self._subjects

    @property
    def edge_count(self) -> int:
        """Pairs found within the cluster radius."""
        return self._edge_count

    @property
    def comparisons(self) -> int:
        """Pairs examined — ``N(N-1)/2`` for this frame."""
        return self._comparisons

    def neighbours(self, subject: SubjectRef) -> frozenset[SubjectRef]:
        return frozenset(self._adjacency.get(subject, frozenset()))

    def components(self, minimum_size: int = 1) -> list[tuple[SubjectRef, ...]]:
        """Connected components with at least ``minimum_size`` members.

        Members within a component, and components themselves, come back in a
        deterministic order derived from the subjects' own ordering — never from
        set iteration, which would let the same frame produce different output
        between runs.
        """
        if minimum_size < 1:
            raise CrowdError(f"minimum_size must be >= 1, got {minimum_size}")

        seen: set[SubjectRef] = set()
        components: list[tuple[SubjectRef, ...]] = []

        # Iterating the ordered subject tuple, not the adjacency mapping, is what
        # makes the traversal order stable.
        for subject in self._subjects:
            if subject in seen:
                continue
            component: list[SubjectRef] = []
            stack = [subject]
            seen.add(subject)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbour in sorted(
                    self._adjacency.get(current, set()),
                    key=_ordering_key,
                    reverse=True,
                ):
                    if neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
            if len(component) >= minimum_size:
                components.append(tuple(sorted(component, key=_ordering_key)))

        components.sort(key=lambda members: [_ordering_key(member) for member in members])
        return components


def build_cluster_graph(positions: Mapping[SubjectRef, MotionPoint], radius: float) -> ClusterGraph:
    """Connect every pair of subjects separated by at most ``radius``.

    Args:
        positions: One projected anchor per subject, all in the same space.
        radius: Connectivity distance in that space's units. The boundary counts
            as connected, so ``separation == radius`` forms an edge.

    Raises:
        CrowdError: if the radius is not positive, or positions span coordinate
            spaces.
    """
    if radius <= 0:
        raise CrowdError(f"cluster_radius must be positive, got {radius}")

    subjects = sorted(positions, key=_ordering_key)
    adjacency: dict[SubjectRef, set[SubjectRef]] = {subject: set() for subject in subjects}

    edges = 0
    comparisons = 0
    for first, second in combinations(subjects, 2):
        comparisons += 1
        if separation(positions[first], positions[second]) <= radius:
            adjacency[first].add(second)
            adjacency[second].add(first)
            edges += 1

    return ClusterGraph(subjects, adjacency, edges, comparisons)


def connected_components(
    positions: Mapping[SubjectRef, MotionPoint], radius: float, minimum_size: int = 1
) -> list[tuple[SubjectRef, ...]]:
    """Convenience wrapper: build the graph and return its components."""
    return build_cluster_graph(positions, radius).components(minimum_size)


def _ordering_key(subject: SubjectRef) -> tuple[str, int]:
    """Total order over subjects, so every traversal and output is reproducible."""
    return (subject.camera_id, subject.track_id)

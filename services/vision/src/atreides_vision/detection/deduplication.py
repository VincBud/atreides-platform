"""Geometric suppression of duplicate person detections within a single frame.

RF-DETR occasionally emits two near-identical boxes on one person, most often
when both are weakly scored. Left alone these can each seed a track, so one
person is reported twice. This module removes the clearly-redundant box before
anything downstream sees it.

Scope, deliberately narrow:

* It looks only at the **current frame's** anonymous
  :class:`~atreides_vision.models.Detection` geometry.
* No track ids, no appearance descriptors, no re-identification, no biometrics,
  no temporal state of any kind. The function is pure and stateless.
* It is detector-agnostic — nothing here knows RF-DETR exists.

The policy is deliberately not exposed as tuning knobs. :data:`RULE_B3` is one
named, conservative, experimentally-validated setting; offering its three
constants as parameters would invite tuning that the evidence does not yet
support.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from atreides_vision.models import Detection

__all__ = [
    "RULE_B3",
    "DeduplicationPolicy",
    "DeduplicationResult",
    "deduplicate_detections",
    "deduplicate_with_report",
]


@dataclass(frozen=True, slots=True)
class DeduplicationPolicy:
    """Geometry a pair must satisfy on *all* three axes to be a duplicate."""

    #: Intersection over union of the two boxes.
    min_iou: float
    #: ``min(area) / max(area)`` — 1.0 means identically sized.
    min_area_similarity: float
    #: Centre separation divided by the mean of the two box diagonals.
    max_center_distance: float


#: The validated conservative policy, variant B3 of the Rule-B family.
#:
#: Offline cross-clip simulation covering 1500 frames of dense daylight CCTV
#: (34,750 detections) and 1050 frames of artificially-lit night CCTV (19,232
#: detections) flagged 54 detections in total — 29 and 25 respectively, roughly
#: 0.08% and 0.13%. Every one of the 7 candidates that B3 adds over the earlier
#: B0 constants (IoU >= 0.90, centre <= 0.02) was inspected at full resolution
#: and confirmed to be two boxes on one person; none was two separate people,
#: and none was judged ambiguous.
#:
#: The relaxed axis is IoU, not centre distance. Centre distance was never the
#: binding term: at IoU >= 0.90 with area similarity >= 0.95, observed pairs
#: never exceeded 0.016, so widening that gate alone changed nothing on either
#: clip. Centre distance is nonetheless the term that keeps genuinely adjacent
#: pedestrians apart — confirmed two-person overlaps sit at 0.112 and above,
#: more than 3.7x outside this gate — so it stays tight deliberately.
RULE_B3 = DeduplicationPolicy(
    min_iou=0.85,
    min_area_similarity=0.95,
    max_center_distance=0.03,
)


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    """What survived and what was suppressed, for diagnostics."""

    kept: tuple[Detection, ...]
    removed: tuple[Detection, ...]

    @property
    def removed_count(self) -> int:
        return len(self.removed)


def _is_duplicate_pair(a: Detection, b: Detection, policy: DeduplicationPolicy) -> bool:
    """Whether two detections satisfy every axis of ``policy``."""
    box_a, box_b = a.bbox_xyxy, b.bbox_xyxy

    ix1 = max(box_a.x1, box_b.x1)
    iy1 = max(box_a.y1, box_b.y1)
    ix2 = min(box_a.x2, box_b.x2)
    iy2 = min(box_a.y2, box_b.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return False

    area_a, area_b = box_a.area, box_b.area
    union = area_a + area_b - intersection
    if union <= 0.0:
        return False
    if intersection / union < policy.min_iou:
        return False

    larger = max(area_a, area_b)
    if larger <= 0.0:
        return False
    if min(area_a, area_b) / larger < policy.min_area_similarity:
        return False

    centre_a, centre_b = box_a.center, box_b.center
    separation = math.hypot(centre_a.x - centre_b.x, centre_a.y - centre_b.y)
    diagonal_a = math.hypot(box_a.width, box_a.height)
    diagonal_b = math.hypot(box_b.width, box_b.height)
    mean_diagonal = (diagonal_a + diagonal_b) / 2.0
    if mean_diagonal <= 0.0:
        return False
    return separation / mean_diagonal <= policy.max_center_distance


def deduplicate_with_report(
    detections: list[Detection], policy: DeduplicationPolicy = RULE_B3
) -> DeduplicationResult:
    """Suppress duplicate detections, reporting what was removed.

    Detections that satisfy ``policy`` pairwise are gathered into connected
    groups, so three mutually-overlapping boxes collapse to one representative
    rather than being handled as three separate pairs. Each group keeps its
    highest-confidence member; ties are broken by earliest input position, so
    the result is fully deterministic. Retained detections are the original
    objects, unmodified, in their original relative order.

    Args:
        detections: This frame's detections. Not mutated.
        policy: Geometry thresholds. Defaults to :data:`RULE_B3`.

    Returns:
        The kept and removed detections.
    """
    count = len(detections)
    if count < 2:
        return DeduplicationResult(kept=tuple(detections), removed=())

    adjacency: dict[int, set[int]] = defaultdict(set)
    for i in range(count):
        for j in range(i + 1, count):
            if _is_duplicate_pair(detections[i], detections[j], policy):
                adjacency[i].add(j)
                adjacency[j].add(i)

    if not adjacency:
        return DeduplicationResult(kept=tuple(detections), removed=())

    removed_indices: set[int] = set()
    visited: set[int] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        # Connected component: one duplicate group, however many boxes it spans.
        stack = [start]
        visited.add(start)
        group: list[int] = []
        while stack:
            node = stack.pop()
            group.append(node)
            for neighbour in sorted(adjacency[node]):
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)

        # Highest confidence wins; earliest index breaks ties deterministically.
        survivor = min(group, key=lambda index: (-detections[index].confidence, index))
        removed_indices.update(index for index in group if index != survivor)

    kept = tuple(d for i, d in enumerate(detections) if i not in removed_indices)
    removed = tuple(d for i, d in enumerate(detections) if i in removed_indices)
    return DeduplicationResult(kept=kept, removed=removed)


def deduplicate_detections(
    detections: list[Detection], policy: DeduplicationPolicy = RULE_B3
) -> list[Detection]:
    """Return ``detections`` with duplicate boxes suppressed.

    Pure and stateless. See :func:`deduplicate_with_report` for the algorithm.
    """
    return list(deduplicate_with_report(detections, policy).kept)

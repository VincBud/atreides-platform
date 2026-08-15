"""The spatial aggregate maths: centroid, spread, separation, density, coherence.

Pure functions over plain values. No cluster state, no configuration objects, no
knowledge of cameras or frames — so every formula can be checked against a
layout whose answer is known by arithmetic.

Distances bottom out in :meth:`~atreides_vision.behavior.coordinates.MotionPoint.distance_to`,
the same primitive the interaction layer uses. There is deliberately no second
distance implementation anywhere in Atreides.

Units follow the active coordinate space throughout: metres and ``subjects/m²``
under a calibrated projector, frame fractions and ``subjects/normalized_area_unit``
without one. Nothing here divides by a venue size, so the same code serves a
doorway camera and a festival field.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations

from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.behavior.models import SubjectMotion
from atreides_vision.errors import CrowdError

__all__ = [
    "centroid",
    "local_density",
    "max_pair_distance",
    "mean_pair_distance",
    "motion_heading_coherence",
    "neighbours_within",
    "rms_radius",
    "separation",
]


def separation(first: MotionPoint, second: MotionPoint) -> float:
    """Euclidean distance between two positions, in their space's units.

    Raises:
        CrowdError: if the two points are in different coordinate spaces.
            ``MotionPoint`` refuses the subtraction; this restates it as a crowd
            failure so the caller learns which comparison was invalid.
    """
    try:
        return first.distance_to(second)
    except ValueError as exc:
        raise CrowdError(f"Cannot measure separation: {exc}") from exc


def centroid(positions: Sequence[MotionPoint]) -> MotionPoint:
    """Arithmetic mean position, in the shared coordinate space.

    ``cx = mean(x_i)``, ``cy = mean(y_i)`` over the projected person anchors —
    never bounding-box centres.

    Raises:
        CrowdError: if there are no positions, or they are not all in one space.
    """
    if not positions:
        raise CrowdError("A centroid needs at least one position.")
    space = positions[0].space
    for position in positions:
        if position.space is not space:
            raise CrowdError(
                "Cannot average positions from different coordinate spaces: "
                f"{space.value} and {position.space.value}"
            )
    count = len(positions)
    return MotionPoint(
        x=sum(position.x for position in positions) / count,
        y=sum(position.y for position in positions) / count,
        space=space,
    )


def rms_radius(positions: Sequence[MotionPoint], center: MotionPoint | None = None) -> float:
    """Root-mean-square distance from the centroid — the cluster's spatial spread.

    ``sqrt(mean(||p_i - centroid||²))``, in the space's distance units.

    RMS rather than a hull area because it is stable at any member count: a
    three-person hull is a triangle whose area collapses to zero the moment they
    line up, which would make "spread" jump around for reasons that have nothing
    to do with how far apart people are.

    A single position has a spread of ``0.0``.
    """
    if not positions:
        raise CrowdError("A spatial spread needs at least one position.")
    middle = center if center is not None else centroid(positions)
    squares = [separation(position, middle) ** 2 for position in positions]
    return math.sqrt(sum(squares) / len(squares))


def mean_pair_distance(positions: Sequence[MotionPoint]) -> float | None:
    """Mean separation over all ``N(N-1)/2`` unique member pairs.

    Self-distances are excluded, and each unordered pair is counted once.

    Returns:
        ``None`` for fewer than two positions, where no pair exists — never a
        fabricated zero.
    """
    if len(positions) < 2:
        return None
    distances = [separation(a, b) for a, b in combinations(positions, 2)]
    return sum(distances) / len(distances)


def max_pair_distance(positions: Sequence[MotionPoint]) -> float | None:
    """Largest separation between any two members — the cluster's extent.

    Distinct from the cluster radius, and routinely larger than it: connectivity
    clustering chains, so a cluster's widest pair can be several radii apart. See
    :mod:`atreides_vision.crowd.clustering`.

    Returns:
        ``None`` for fewer than two positions.
    """
    if len(positions) < 2:
        return None
    return max(separation(a, b) for a, b in combinations(positions, 2))


def neighbours_within(position: MotionPoint, others: Sequence[MotionPoint], radius: float) -> int:
    """How many of ``others`` lie within ``radius`` of ``position``, boundary included.

    ``others`` should exclude ``position`` itself; the caller owns that, because
    only the caller knows which entry is the subject being sampled.
    """
    if radius <= 0:
        raise CrowdError(f"A density radius must be positive, got {radius}")
    return sum(1 for other in others if separation(position, other) <= radius)


def local_density(neighbour_count: int, radius: float, space: MotionSpace) -> float:
    """Subjects per unit area in the disk around one subject.

    ``local_density = (1 + neighbour_count) / (π r²)``

    **The subject itself is counted**, which is why the numerator starts at one:
    the person occupies the neighbourhood being sampled, so an entirely alone
    subject has a real, non-zero local density rather than a zero that would be
    indistinguishable from "no measurement".

    Units are ``subjects/m²`` under a calibrated projector and
    ``subjects/normalized_area_unit`` otherwise — the latter is a
    perspective-dependent camera-local number and **not** a physical density.

    **Boundary bias, uncorrected.** Near the edge of the image or of the
    calibrated floor, part of the disk is not observable, so fewer neighbours can
    be seen and density is under-reported. Correcting for that needs a venue
    boundary model Atreides does not have, and inventing one would be worse than
    the documented bias.
    """
    if radius <= 0:
        raise CrowdError(f"A density radius must be positive, got {radius}")
    if neighbour_count < 0:
        raise CrowdError(f"neighbour_count must be non-negative, got {neighbour_count}")
    del space  # Present so callers state the space explicitly; area is unitless here.
    return (1 + neighbour_count) / (math.pi * radius * radius)


def motion_heading_coherence(
    motions: Sequence[SubjectMotion], min_displacement: float
) -> float | None:
    """How similarly the moving members are travelling, in ``[0, 1]``.

    The **mean resultant length** of their unit heading vectors::

        u_i = unit vector of member i's net displacement
        R   = ||Σ u_i|| / N_moving

    Near ``1`` the headings point the same way; near ``0`` they are spread out or
    opposed. Two members heading in exactly opposite directions give ``0``.

    This is a geometric summary of *directions of travel* and nothing else. It is
    not coordination, not organisation, not panic, and not a group acting
    together — Atreides has no signal for any of those, and members walking the
    same way down a corridor score high for no reason but the corridor.

    Members below ``min_displacement`` are excluded: a stationary person has no
    direction of travel, and including a zero vector would drag the resultant
    towards ``0`` as though they were actively disagreeing.

    Returns:
        ``None`` when fewer than two members are moving. That is not ``0`` —
        "not enough evidence to say" and "headings cancel out" are different
        facts, and reporting the latter for the former would invent disagreement.

    Raises:
        CrowdError: if the motions are not all in one coordinate space.
    """
    if min_displacement <= 0:
        raise CrowdError(f"min_displacement must be positive, got {min_displacement}")

    moving = [motion for motion in motions if motion.magnitude >= min_displacement]
    if len(moving) < 2:
        return None

    space = moving[0].space
    for motion in moving:
        if motion.space is not space:
            raise CrowdError(
                "Cannot compare motion across coordinate spaces: "
                f"{space.value} and {motion.space.value}"
            )

    sum_x = sum(motion.dx / motion.magnitude for motion in moving)
    sum_y = sum(motion.dy / motion.magnitude for motion in moving)
    # Clamped because floating-point summation can nudge a perfectly aligned set
    # a hair above 1.0, and a "coherence" of 1.0000000002 is indefensible.
    return min(1.0, math.hypot(sum_x, sum_y) / len(moving))

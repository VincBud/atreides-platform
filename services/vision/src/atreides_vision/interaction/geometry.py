"""The pairwise maths: separation, closing speed and relative heading.

Pure functions over plain values. No pair state, no configuration objects, no
knowledge of cameras or frames — so every formula can be checked against a
synthetic scenario whose answer is known by arithmetic.

Sign conventions are fixed here and stated once:

* **closing speed is positive when the gap is shrinking.** Two people walking
  towards each other at 1 m/s each close at ``+2 m/s``; the same two walking
  apart give ``-2 m/s``; a constant separation gives ``0``.
* **relative motion heading is unsigned**, in ``[0, π]``. ``0`` means the two
  are travelling in the same direction, ``π`` in opposite directions, ``π/2``
  perpendicular. The sign of the turn between them carries no pairwise meaning,
  so it is not reported.

What relative heading is not
----------------------------
It compares **directions of travel**, derived from where each subject moved. It
is *not* body orientation, gaze, or which way anyone is facing — Atreides has no
signal for those. Two people standing still facing each other produce no heading
at all, because neither is moving.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from atreides_vision.behavior.coordinates import MotionPoint
from atreides_vision.behavior.kinematics import wrap_angle
from atreides_vision.behavior.models import SubjectMotion
from atreides_vision.errors import InteractionError

__all__ = [
    "MIN_SAMPLES",
    "TimedDistance",
    "closing_speed",
    "pair_distance",
    "proximity_since",
    "relative_motion_heading",
]

#: Two separation samples are the minimum from which a rate can be derived.
MIN_SAMPLES = 2


class TimedDistance(Protocol):
    """A separation measured at a moment.

    :class:`~atreides_vision.interaction.history.PairSample` satisfies this
    structurally. Depending on the narrow shape keeps these formulas from
    quietly starting to use zone ids or subject references.
    """

    @property
    def timestamp_seconds(self) -> float: ...

    @property
    def distance(self) -> float: ...


def pair_distance(first: MotionPoint, second: MotionPoint) -> float:
    """Euclidean separation between two positions, in their space's units.

    Uses the already-projected person anchors, so under a calibrated projector
    this is a floor distance in metres and under a normalized one it is a
    fraction of the frame. Never a bounding-box centre distance, and never
    pixels.

    Raises:
        InteractionError: if the two points are in different coordinate spaces.
            :class:`~atreides_vision.behavior.coordinates.MotionPoint` refuses
            the subtraction; this restates it as a pairwise failure so the
            caller learns which comparison was invalid.
    """
    try:
        return first.distance_to(second)
    except ValueError as exc:
        raise InteractionError(f"Cannot measure separation: {exc}") from exc


def closing_speed(samples: Sequence[TimedDistance]) -> float | None:
    """Rate at which the gap is closing, in distance units per second.

    Defined as::

        closing_speed = -(distance_last - distance_first) / (t_last - t_first)

    so a shrinking gap is **positive**. Measured across the window's endpoints
    rather than between consecutive frames: a frame-to-frame difference divides
    tracker jitter by a very small ``dt`` and produces a number dominated by
    noise, which is the same reasoning behind the behaviour layer's windowed
    speed.

    Because it is a windowed mean, a pair that approaches and then retreats
    inside one window reports the *net* rate, not the peak.

    Returns:
        ``None`` with fewer than two samples or no elapsed time — never a
        fabricated zero, which would be indistinguishable from a genuinely
        constant separation.
    """
    if len(samples) < MIN_SAMPLES:
        return None
    elapsed = samples[-1].timestamp_seconds - samples[0].timestamp_seconds
    if elapsed <= 0:
        return None
    return -(samples[-1].distance - samples[0].distance) / elapsed


def relative_motion_heading(
    first: SubjectMotion | None,
    second: SubjectMotion | None,
    min_displacement: float,
) -> float | None:
    """Unsigned angle between two directions of travel, in ``[0, π]``.

    ``0`` is travelling the same way, ``π`` opposite ways, ``π/2``
    perpendicular. Wrapped, so a pair at ``+170°`` and ``-170°`` reads as the
    20° difference it is rather than 340°.

    Returns:
        ``None`` when either subject has moved too little for a direction to
        exist. That is not the same as ``0`` — "neither is going anywhere" and
        "both are going the same way" are different facts, and reporting the
        latter for the former would invent an alignment nobody has.

    Raises:
        InteractionError: if the two motions are in different spaces.
    """
    if first is None or second is None:
        return None
    if first.space is not second.space:
        raise InteractionError(
            "Cannot compare motion across coordinate spaces: "
            f"{first.space.value} and {second.space.value}"
        )

    first_heading = first.heading_radians(min_displacement)
    second_heading = second.heading_radians(min_displacement)
    if first_heading is None or second_heading is None:
        return None
    return abs(wrap_angle(second_heading - first_heading))


def proximity_since(
    previous: float | None,
    *,
    distance: float,
    threshold: float,
    window_start_seconds: float,
    continuous: bool,
) -> float | None:
    """When the pair's current spell within ``threshold`` began, or ``None``.

    A pure transition function, so proximity duration — which must outlive the
    retained history window, since two people can stand together far longer — is
    still fully determined by its inputs rather than by hidden accumulation.
    The behaviour layer's stationary duration works the same way.

    Rules, in order:

    * A discontinuity ends the spell. Where the pair went while unobserved is
      unknown, so claiming they stayed together through it would be unfounded.
    * A separation strictly greater than ``threshold`` ends the spell. The
      boundary itself counts as *in* proximity, so ``distance == threshold``
      continues it.
    * Otherwise an existing spell continues, and a new one starts at
      ``window_start_seconds`` — the earliest observation in the current window
      rather than this frame, so the duration does not lose the time it took to
      establish.
    """
    if not continuous or distance > threshold:
        return None
    return previous if previous is not None else window_start_seconds

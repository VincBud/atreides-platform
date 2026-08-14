"""The maths: positions over time into speed, acceleration and turn rate.

Everything here is a pure function of a sequence of timed positions. No state,
no configuration objects, no knowledge of tracks, cameras or frames — which is
what makes each formula testable against a synthetic trajectory with a known
analytic answer.

Two conventions run through the module.

**Path length, not straight-line displacement, for speed.** Someone dancing on
the spot travels a real distance and returns to where they started; net
displacement would report them as motionless. Speed therefore sums the distance
actually covered between consecutive samples. This matches the zone layer's
definition.

**Split-half differencing for rates of change.** Acceleration and turn rate are
differences of a quantity that is itself measured over a window. Differencing
consecutive frames instead would divide sensor noise by a very small ``dt`` and
produce a number dominated by jitter. So the window is cut at its midpoint, the
quantity is measured over each half, and the two are differenced across the gap
between the halves' centres. For constant acceleration this is *exact*, not
approximate, which is precisely what the tests assert.

Every function returns ``None`` rather than a fabricated number when there is
not enough evidence: too few samples, no elapsed time, or movement too small
for direction to mean anything. Callers must not substitute zero for ``None``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

from atreides_vision.behavior.coordinates import MotionPoint

__all__ = [
    "TimedPosition",
    "acceleration",
    "heading_change_rate",
    "heading_radians",
    "mean_speed",
    "net_displacement",
    "path_length",
    "span_center_seconds",
    "span_seconds",
    "stationary_since",
    "wrap_angle",
]

#: Two samples are the minimum from which any motion can be derived.
MIN_SAMPLES = 2


class TimedPosition(Protocol):
    """A position with a time. :class:`~atreides_vision.behavior.models.BehaviorObservation`
    satisfies this structurally.

    Kinematics depends on this narrow shape rather than the full observation so
    the formulas cannot quietly start using boxes, confidences or zone ids.
    """

    @property
    def timestamp_seconds(self) -> float: ...

    @property
    def position(self) -> MotionPoint: ...


def path_length(samples: Sequence[TimedPosition]) -> float:
    """Total distance travelled along the sampled path, in space distance units.

    Zero for fewer than two samples.
    """
    return sum(
        previous.position.distance_to(current.position) for previous, current in pairwise(samples)
    )


def net_displacement(samples: Sequence[TimedPosition]) -> tuple[float, float]:
    """Straight-line displacement ``(dx, dy)`` from first to last sample.

    ``(0.0, 0.0)`` for fewer than two samples.
    """
    if len(samples) < MIN_SAMPLES:
        return (0.0, 0.0)
    return samples[0].position.vector_to(samples[-1].position)


def span_seconds(samples: Sequence[TimedPosition]) -> float:
    """Elapsed time from the first sample to the last. ``0.0`` if fewer than two."""
    if len(samples) < MIN_SAMPLES:
        return 0.0
    return samples[-1].timestamp_seconds - samples[0].timestamp_seconds


def span_center_seconds(samples: Sequence[TimedPosition]) -> float:
    """The midpoint of the sampled span.

    This — not the mean of the sample times — is the instant a mean speed over
    the span actually refers to, and using it is what makes :func:`acceleration`
    exact for constant acceleration.

    Raises:
        ValueError: if there are no samples.
    """
    if not samples:
        raise ValueError("span_center_seconds requires at least one sample")
    return (samples[0].timestamp_seconds + samples[-1].timestamp_seconds) / 2.0


def mean_speed(samples: Sequence[TimedPosition]) -> float | None:
    """Mean speed over the sampled span, in distance units per second.

    Path length divided by elapsed time, so movement in place counts.

    Returns:
        ``None`` if there are fewer than two samples or no time elapsed
        between the first and last — never a fabricated ``0.0``.
    """
    if len(samples) < MIN_SAMPLES:
        return None
    elapsed = span_seconds(samples)
    if elapsed <= 0:
        return None
    return path_length(samples) / elapsed


def heading_radians(
    samples: Sequence[TimedPosition],
    min_displacement: float,
) -> float | None:
    """Direction of net travel across the samples, in radians.

    Measured with ``atan2(dy, dx)`` in the active coordinate space. Note that
    in normalized image space ``y`` grows *downward*, so an increasing angle
    corresponds to a clockwise turn as seen on screen.

    Returns:
        ``None`` when net displacement is below ``min_displacement``. A person
        standing still has a direction of travel that is undefined, not zero,
        and reporting an angle derived from box jitter would manufacture a
        heading — and, worse, a turn rate — out of noise.
    """
    dx, dy = net_displacement(samples)
    if math.hypot(dx, dy) < min_displacement:
        return None
    return math.atan2(dy, dx)


def wrap_angle(radians: float) -> float:
    """Wrap an angle into ``(-pi, pi]``.

    Without this, a turn from ``+170°`` to ``-170°`` reads as a 340° swing
    rather than the 20° one that actually happened.
    """
    wrapped = math.remainder(radians, math.tau)
    # math.remainder yields [-pi, pi]; normalise the -pi endpoint to +pi so the
    # result is a half-open interval and equal turns compare equal.
    return math.pi if wrapped == -math.pi else wrapped


def acceleration(samples: Sequence[TimedPosition]) -> float | None:
    """Rate of change of speed, in distance units per second squared.

    The samples are split at their midpoint; each half's mean speed is taken to
    apply at that half's own span centre, and the two are differenced. Positive
    means speeding up.

    Returns:
        ``None`` unless both halves hold at least two samples spanning real
        time.
    """
    halves = _split_halves(samples)
    if halves is None:
        return None
    earlier, later = halves

    first_speed = mean_speed(earlier)
    second_speed = mean_speed(later)
    if first_speed is None or second_speed is None:
        return None

    delta_t = span_center_seconds(later) - span_center_seconds(earlier)
    if delta_t <= 0:
        return None
    return (second_speed - first_speed) / delta_t


def heading_change_rate(
    samples: Sequence[TimedPosition],
    min_displacement: float,
) -> float | None:
    """Rate of change of travel direction, in radians per second.

    Same split-half construction as :func:`acceleration`: the direction of net
    travel in each half is compared, and the wrapped difference is divided by
    the time between the halves' centres. Sign follows
    :func:`heading_radians`'s convention.

    Returns:
        ``None`` if either half moved less than ``min_displacement`` — with no
        direction of travel there is no turn to measure.
    """
    halves = _split_halves(samples)
    if halves is None:
        return None
    earlier, later = halves

    first_heading = heading_radians(earlier, min_displacement)
    second_heading = heading_radians(later, min_displacement)
    if first_heading is None or second_heading is None:
        return None

    delta_t = span_center_seconds(later) - span_center_seconds(earlier)
    if delta_t <= 0:
        return None
    return wrap_angle(second_heading - first_heading) / delta_t


def stationary_since(
    previous: float | None,
    *,
    speed: float | None,
    window_start_seconds: float,
    threshold: float,
    continuous: bool,
) -> float | None:
    """When the subject's current stationary spell began, or ``None``.

    Kept as a pure transition function so that stationary duration — the one
    signal that must outlive the retained history window, since a person can
    stand still for far longer than a couple of seconds — is still fully
    determined by its inputs rather than by hidden accumulation.

    The rules, in order:

    * A discontinuity (``continuous=False``) ends any spell. Motion across an
      unobserved gap is unknown, so a claim that someone stood still through it
      would be unfounded.
    * An unknown speed ends any spell, for the same reason.
    * Speed at or above ``threshold`` ends any spell.
    * Otherwise an existing spell continues unchanged, and a new one starts at
      ``window_start_seconds`` — the earliest observation the speed estimate
      rests on, not the current frame, so the duration does not lose the window
      it took to establish that the subject was stationary.

    Args:
        previous: Start of the spell in progress, or ``None``.
        speed: Current speed estimate, or ``None`` when it cannot be computed.
        window_start_seconds: Timestamp of the oldest sample behind ``speed``.
        threshold: Speed below which the subject counts as stationary.
        continuous: Whether history has been unbroken since the last update.
    """
    if not continuous or speed is None or speed >= threshold:
        return None
    return previous if previous is not None else window_start_seconds


def _split_halves(
    samples: Sequence[TimedPosition],
) -> tuple[Sequence[TimedPosition], Sequence[TimedPosition]] | None:
    """Split samples at their time midpoint into two usable halves.

    Splitting by time rather than by count keeps the two halves comparable when
    sampling is irregular — after a dropped frame, for instance. A sample lying
    exactly on the midpoint is included in both halves so no motion falls
    between them.

    Returns ``None`` if either half would hold fewer than two samples.
    """
    if len(samples) < 2 * MIN_SAMPLES:
        return None

    midpoint = span_center_seconds(samples)
    earlier = [sample for sample in samples if sample.timestamp_seconds <= midpoint]
    later = [sample for sample in samples if sample.timestamp_seconds >= midpoint]

    if len(earlier) < MIN_SAMPLES or len(later) < MIN_SAMPLES:
        return None
    return (earlier, later)

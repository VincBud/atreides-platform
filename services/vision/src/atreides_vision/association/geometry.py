"""The comparability maths: venue-time gaps, timing slack, separation, radii.

Pure functions over plain values. No configuration objects beyond the tunables
they are handed, no state, no knowledge of cameras or ticks — so every formula
can be checked against a layout whose answer is known by arithmetic.

Distances bottom out in
:meth:`~atreides_vision.behavior.coordinates.MotionPoint.distance_to`, the same
primitive the interaction and crowd layers use. There is deliberately no second
distance implementation anywhere in Atreides.

Units are metres and seconds throughout, and only ever metres: every function
here refuses a non-metric space rather than returning a number whose unit
depends on which camera asked. Nothing divides by a venue size, assumes an
overlap, or knows how many cameras exist.
"""

from __future__ import annotations

from atreides_vision.association.config import AssociationGatingConfig
from atreides_vision.association.models import TrackObservation
from atreides_vision.behavior.coordinates import MotionSpace
from atreides_vision.errors import AssociationError
from atreides_vision.timing import TimeUncertainty

__all__ = [
    "admissible_radius_m",
    "earliest_possible_gap_seconds",
    "latest_possible_gap_seconds",
    "motion_allowance_m",
    "nominal_gap_seconds",
    "separation_metres",
    "timing_slack_seconds",
]


def nominal_gap_seconds(first: TrackObservation, second: TrackObservation) -> float:
    """Venue-time difference between two observations, as stated.

    Venue seconds, never source seconds: source time restarts at every stream
    epoch boundary, so a difference of source seconds across cameras is not a
    duration at all.
    """
    return abs(first.venue_seconds - second.venue_seconds)


def timing_slack_seconds(first: TimeUncertainty, second: TimeUncertainty) -> float | None:
    """How much the stated gap could be wrong, in seconds, in either direction.

    The two half-widths added together: each stream's true instant may lie
    anywhere within its own bound, so the worst case for the pair is the sum.

    Returns:
        ``None`` when either alignment is ``NOT_ESTABLISHED``. ``None`` is not
        zero and does not do arithmetic: nobody related that stream to the venue
        timeline, so there is no bound to add. Callers must treat it as "this
        pair has no common clock", not as "this pair is perfectly aligned".
    """
    if first.half_width_seconds is None or second.half_width_seconds is None:
        return None
    return first.half_width_seconds + second.half_width_seconds


def earliest_possible_gap_seconds(nominal_gap: float, slack: float) -> float:
    """The smallest real gap consistent with the stated times, never below zero.

    The number a *temporal* rejection must be based on: refusing a pair as
    non-simultaneous is only honest if even the pair's best case exceeds the
    window.
    """
    return max(0.0, nominal_gap - slack)


def latest_possible_gap_seconds(nominal_gap: float, slack: float) -> float:
    """The largest real gap consistent with the stated times.

    The number a *spatial* rejection must be based on: the person had at most
    this long to move, so anything less would exclude pairs on time the clocks
    cannot rule out.
    """
    return nominal_gap + slack


def separation_metres(first: TrackObservation, second: TrackObservation) -> float:
    """Distance in venue metres between two observations' projected positions.

    Raises:
        AssociationError: if either position is not in calibrated ground-plane
            metres, or the two are in different spaces. Both would otherwise
            yield a plausible-looking float in no unit at all.
    """
    for observation in (first, second):
        if observation.space is not MotionSpace.GROUND_PLANE_METRES:
            raise AssociationError(
                f"Cannot measure cross-camera separation for {observation.subject}: its position "
                f"is in {observation.space.value}, not calibrated metres. Image-plane coordinates "
                "are camera-local, so a distance between two cameras' values means nothing."
            )
    try:
        return first.position.distance_to(second.position)
    except ValueError as exc:  # pragma: no cover - the space check above precedes it
        raise AssociationError(f"Cannot measure cross-camera separation: {exc}") from exc


def motion_allowance_m(latest_gap_seconds: float, config: AssociationGatingConfig) -> float:
    """Metres a person could have covered in the time that may separate two observations.

    ``max_subject_speed_m_per_s * latest_gap_seconds``. Grows with timing
    uncertainty on purpose: worse-known clocks admit more, never fewer,
    candidates. A radius that shrank as alignment degraded would be deriving a
    rejection from precision nobody established.
    """
    if latest_gap_seconds < 0:
        raise AssociationError(
            f"A time gap cannot be negative, got {latest_gap_seconds!r} seconds."
        )
    return config.max_subject_speed_m_per_s * latest_gap_seconds


def admissible_radius_m(latest_gap_seconds: float, config: AssociationGatingConfig) -> float:
    """Separation at or below which geometry does not exclude a pair, in metres.

    ``position_tolerance_m + motion_allowance_m``: how far two cameras may
    disagree about one stationary person, plus how far that person could have
    moved in the time that may separate the two observations.
    """
    return config.position_tolerance_m + motion_allowance_m(latest_gap_seconds, config)

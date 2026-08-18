"""Tunable parameters for cross-camera comparability gating.

Every number that can reject a candidate lives here with its unit and its
rationale. No gate in this package contains a bare constant.

**These are DEVELOPMENT defaults.** Nothing here was measured against real
multi-camera footage, because none exists in this repository. They were chosen
to be defensible and to make the gates testable, and each one states which
direction it errs in when it is wrong.

The one rule they all obey
--------------------------
A gate's only power is rejection, so every default is set to reject only what is
*implausible*, never what is merely *unlikely*. A tighter value here does not
make a merge safer — it hides a candidate from the layer whose job is to decide,
and turns a visible ambiguity into an invisible miss. Discrimination between
surviving candidates belongs to the decision layer above, on evidence stronger
than geometry.

There is deliberately no threshold on
:class:`~atreides_vision.timing.TimeUncertainty` itself. A half-width is not
comparable to a constant — it is combined with the venue-time gap and converted
into metres of allowance, so a badly aligned pair of cameras produces an
enormous admissible radius that discriminates nothing, and says so in the
evidence, rather than being silently declared "aligned enough".

Unlike the behaviour, interaction and crowd configs there is no normalized-image
variant, and no ``intended_space``. Cross-camera geometry requires calibrated
metres by construction: a frame fraction on one camera has no relationship to a
frame fraction on another, and no fallback could give one.
"""

from __future__ import annotations

from dataclasses import dataclass

from atreides_vision.errors import ConfigurationError

__all__ = ["AssociationGatingConfig"]


@dataclass(frozen=True, slots=True)
class AssociationGatingConfig:
    """Thresholds for deciding which candidate pairs geometry cannot exclude."""

    #: Largest nominal venue-time difference, in seconds, at which two
    #: observations are still treated as describing one instant.
    #:
    #: Ticks are source-driven, so two cameras almost never land on the same
    #: venue microsecond; unequal frame rates guarantee it. 0.5 s covers a full
    #: frame period at rates down to 2 FPS, which real CCTV exports reach.
    #:
    #: It is **not** a claim about synchronisation. What the two clocks are
    #: actually worth is each stream's ``TimeUncertainty``, which is added to
    #: this rather than replaced by it.
    #:
    #: Too small rejects true simultaneous pairs on slow cameras; too large
    #: admits pairs separated by real time, which the motion allowance then has
    #: to cover in metres.
    max_pairing_gap_seconds: float = 0.5

    #: Fastest a person is assumed to travel on foot, in metres per second.
    #: Used only to widen a rejection radius by the distance somebody could have
    #: covered between two observations — never to measure a speed.
    #:
    #: Set near the human sprint limit rather than at a walking pace, because
    #: this gate exists to exclude the physically impossible. A "typical" value
    #: would quietly turn a gate into a discriminator and reject true pairs.
    max_subject_speed_m_per_s: float = 8.0

    #: How far apart two calibrated cameras may place one person at one instant,
    #: in metres, before geometry excludes the pair.
    #:
    #: Absorbs the bottom-centre anchor landing differently when feet are
    #: occluded, homography error at the far end of a view, and box jitter. **The
    #: dominant free parameter of this milestone, and entirely unvalidated.**
    #: Too small rejects true pairs; too large admits more candidates, which
    #: surfaces as ambiguity rather than as a merge.
    position_tolerance_m: float = 1.5

    #: Difference in separation, in metres, below which two candidates on one
    #: counterpart camera are not distinguished by geometry.
    #:
    #: Set just under :attr:`position_tolerance_m`: if two candidates differ by
    #: less than the tolerance the geometry itself is built on, preferring the
    #: nearer one is preferring noise. Below this the honest answer is
    #: ``AMBIGUOUS``, which refuses rather than guesses.
    ambiguity_margin_m: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("max_pairing_gap_seconds", self.max_pairing_gap_seconds),
            ("max_subject_speed_m_per_s", self.max_subject_speed_m_per_s),
            ("position_tolerance_m", self.position_tolerance_m),
            ("ambiguity_margin_m", self.ambiguity_margin_m),
        ):
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive, got {value}")

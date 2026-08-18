"""Tunable parameters for cross-camera comparability gating and decisions.

Every number that can reject a candidate — or authorise an association — lives
here with its unit and its rationale. No gate and no decision rule in this
package contains a bare constant.

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

:class:`AssociationDecisionConfig` is the exception that proves it: it is the
only place a number may *authorise* something, and every default in it is set
so that too little evidence produces ``AMBIGUOUS`` rather than an association.
Loosening one of its values trades missed associations for false ones, which is
the wrong direction — a wrong merge corrupts every venue-level number above it
and is far harder to notice than a missed one.

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

__all__ = ["AssociationDecisionConfig", "AssociationGatingConfig"]


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


@dataclass(frozen=True, slots=True)
class AssociationDecisionConfig:
    """Thresholds the decision core must clear before it asserts an association.

    Separate from :class:`AssociationGatingConfig` because the two answer
    different questions. Gating asks "could geometry rule this out at this
    instant"; these ask "has enough agreement accumulated over time to say yes".
    Keeping them apart means a venue can widen a rejection radius without
    loosening what it takes to be believed.
    """

    #: Fewest agreeing observations, in a row, before an association may be
    #: asserted. Counted, never assumed from a frame rate, because none is known
    #: here.
    #:
    #: Two would let a single coincidence repeated once carry a merge; three is
    #: the smallest count where a run has to survive a change of frame. Too high
    #: and short crossings are never associated at all.
    min_support_samples: int = 3

    #: Shortest venue time, in seconds, that a run of agreement must span.
    #:
    #: Exists because a sample count alone means different things at 2 FPS and
    #: 30 FPS: three consecutive frames of a fast camera is a tenth of a second,
    #: which two people passing each other can easily produce. This floor makes
    #: the requirement a statement about the venue, not about the encoder.
    min_support_seconds: float = 0.4

    #: Longest interruption, in seconds, a run of agreement survives.
    #:
    #: A pair unobserved for longer starts a new run rather than continuing the
    #: old one: after a gap the two tracklets may be different people, and a
    #: dropped frame is indistinguishable here from somebody leaving the view.
    #: Longer gaps are exactly the handoff problem this milestone does not solve.
    max_support_gap_seconds: float = 1.0

    #: How much nearer, in metres, the preferred candidate must be than the
    #: nearest rival on the same counterpart camera before it may be accepted.
    #:
    #: The rival matters more than the distance: being 0.4 m away means nothing
    #: if somebody else is 0.5 m away. Set at the gating layer's
    #: ``ambiguity_margin_m`` by default so a lead geometry calls indistinct is
    #: never enough to accept; raising it demands a clearer winner, which turns
    #: would-be associations into ``AMBIGUOUS``, never into wrong ones.
    acceptance_margin_m: float = 1.0

    #: How long, in seconds of venue time, a candidate pair's evidence is kept
    #: after it was last observed. Bounded state, not a memory: association here
    #: is operational and dies with the session.
    #:
    #: Set well above :attr:`max_support_gap_seconds` so a pair that reappears
    #: within a survivable gap still finds its run, and low enough that a venue
    #: full of passers-by does not accumulate evidence about pairs that are over.
    evidence_ttl_seconds: float = 5.0

    #: Hard ceiling on how many candidate pairs carry evidence at once. Reached
    #: only when eviction by age has already failed to contain growth; the
    #: least-recently-observed pairs go first.
    #:
    #: A ceiling on *pairs*, which grow with the square of concurrent tracks, so
    #: it is deliberately generous: 4096 covers well over 60 simultaneously
    #: visible tracklets across a venue before anything is dropped. It is not a
    #: statement about how many cameras or people a venue has.
    max_tracked_pairs: int = 4096

    def __post_init__(self) -> None:
        for name, value in (
            ("min_support_seconds", self.min_support_seconds),
            ("max_support_gap_seconds", self.max_support_gap_seconds),
            ("acceptance_margin_m", self.acceptance_margin_m),
            ("evidence_ttl_seconds", self.evidence_ttl_seconds),
        ):
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive, got {value}")
        if self.min_support_samples < 2:
            raise ConfigurationError(
                "min_support_samples must be at least 2, got "
                f"{self.min_support_samples}: a single observation is a coincidence, not a "
                "sustained agreement."
            )
        if self.max_tracked_pairs < 1:
            raise ConfigurationError(
                f"max_tracked_pairs must be at least 1, got {self.max_tracked_pairs}"
            )
        if self.evidence_ttl_seconds < self.max_support_gap_seconds:
            raise ConfigurationError(
                f"evidence_ttl_seconds ({self.evidence_ttl_seconds}) must be at least "
                f"max_support_gap_seconds ({self.max_support_gap_seconds}), otherwise evidence is "
                "discarded while the run it belongs to is still allowed to continue."
            )

"""Outcome vocabulary for association -> venue identity integration.

One :class:`~atreides_vision.association.models.PairDecision` meets the venue
identity registry and something happens, or explicitly does not. This module
names the four things that can happen, and the record that keeps a decision's
evidence attached to whatever the registry did with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from atreides_vision.association.models import CandidateKey, PairDecision
from atreides_vision.errors import FusionError
from atreides_vision.identity.models import VenueSubjectId

__all__ = [
    "IntegrationOutcome",
    "IntegrationResult",
    "MembershipOutcome",
]


class IntegrationOutcome(StrEnum):
    """What happened when one association decision met the identity registry.

    All four are first class, matching the vocabulary
    :class:`~atreides_vision.association.models.AssociationOutcome` itself
    established: a caller written against three of these would break the
    moment a decision produced the fourth.
    """

    #: The registry changed: a new venue subject was created holding both
    #: tracklets, or an unowned tracklet was added to one that already existed.
    APPLIED = "applied"

    #: Both tracklets were already active members of the same venue subject.
    #: The registry did not change. This is what makes reprocessing the same
    #: :class:`~atreides_vision.association.models.AssociationRound` safe: a
    #: decision already applied changes nothing the second time.
    NO_OP = "no_op"

    #: The registry was left exactly as it was because applying the decision
    #: would require a merge, a split, or a silent reassignment this layer is
    #: not permitted to perform. Never a partial application.
    REFUSED = "refused"

    #: The decision was not a positive association (``AMBIGUOUS``,
    #: ``UNASSOCIATED`` or ``NOT_COMPARABLE``) and was never eligible to
    #: change the registry.
    IGNORED = "ignored"


@dataclass(frozen=True, slots=True)
class MembershipOutcome:
    """One decision's fate against the registry, with its provenance intact.

    ``decision`` is the full V0.4.2 :class:`PairDecision` — assessment,
    support and confidence — carried through unmodified, so a caller can see
    exactly why a membership was accepted, refused or ignored without this
    layer inventing a second copy of that evidence.
    """

    decision: PairDecision
    outcome: IntegrationOutcome
    #: A human-readable account of the rule that produced ``outcome``, in the
    #: same spirit as :attr:`~atreides_vision.association.models.GateRejection.detail`.
    detail: str
    #: The venue subject(s) this outcome concerns, in canonical (sequence)
    #: order: empty for ``IGNORED``, one entry for ``APPLIED``/``NO_OP``, and
    #: the two conflicting subjects — never merged — for a ``REFUSED``
    #: cross-subject conflict.
    subject_ids: tuple[VenueSubjectId, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.detail or not self.detail.strip():
            raise FusionError(
                f"MembershipOutcome({self.outcome.value}) must explain itself: an outcome with "
                "no detail cannot be acted on by whoever reads it."
            )

    @property
    def key(self) -> CandidateKey:
        return self.decision.key


@dataclass(frozen=True, slots=True)
class IntegrationResult:
    """Every outcome produced from one
    :class:`~atreides_vision.association.models.AssociationRound`.

    Complete rather than filtered, matching ``AssociationRound`` itself: one
    outcome per decision, including every ``IGNORED`` one, so "nothing
    happened this round" is as auditable as "three tracklets joined".
    """

    venue_seconds: float
    outcomes: tuple[MembershipOutcome, ...]

    @property
    def applied(self) -> tuple[MembershipOutcome, ...]:
        return tuple(o for o in self.outcomes if o.outcome is IntegrationOutcome.APPLIED)

    @property
    def no_ops(self) -> tuple[MembershipOutcome, ...]:
        return tuple(o for o in self.outcomes if o.outcome is IntegrationOutcome.NO_OP)

    @property
    def refused(self) -> tuple[MembershipOutcome, ...]:
        return tuple(o for o in self.outcomes if o.outcome is IntegrationOutcome.REFUSED)

    @property
    def ignored(self) -> tuple[MembershipOutcome, ...]:
        return tuple(o for o in self.outcomes if o.outcome is IntegrationOutcome.IGNORED)

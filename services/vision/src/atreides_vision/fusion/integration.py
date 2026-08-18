"""Association -> venue identity integration — V0.4.3b.

The seam between two independent layers: ``association`` decides whether two
camera-local tracklets correspond right now; ``identity`` owns venue-local
identity state and refuses to reconcile a claim on its own. Neither absorbs
the other's responsibility, so this module holds the one policy that belongs
to neither: which of a round's decisions may change registry state, and what
happens when they cannot agree.

Only ``AssociationOutcome.ASSOCIATED`` decisions are ever attempted against
the registry. Every other outcome — ``AMBIGUOUS``, ``UNASSOCIATED``,
``NOT_COMPARABLE`` — is IGNORED by construction: "no association here" from
V0.4.2 must never become a membership change here, whatever the reason for it.

The four outcomes a positive association can produce
------------------------------------------------------
1. Neither tracklet is currently owned: a new venue subject is created
   holding both.
2. One tracklet is owned, the other is not: the unowned tracklet is added to
   the existing venue subject — the ordinary way an identity grows across a
   third, fourth, ... camera.
3. Both tracklets are already owned by the *same* venue subject: NO_OP. This
   is what makes reprocessing one ``AssociationRound`` idempotent — a decision
   that has already been applied changes nothing the second time.
4. Both tracklets are owned by *different* venue subjects: REFUSED. Merging
   two already-distinct venue identities is not a decision this layer may
   make from one pairwise association; it would corrupt both subjects'
   provenance, and there is no confidence-aware policy in this milestone's
   scope for choosing which one should absorb the other, or whether either
   should. Below the evidence for a full merge, the correct output is no
   merge — reported, not swallowed. Resolving genuine cross-subject conflicts
   is a distinct, unbuilt capability and a product decision, not something to
   improvise here.

A fifth refusal exists at the registry boundary: the tracklet that would be
*added* under case 2 belongs to a camera the owning venue subject already has
an active member on. One venue subject with two live tracklets on one camera
would claim to be two places on that camera's view at once, which
:class:`~atreides_vision.identity.registry.VenueSubjectRegistry` never
allows. This layer checks for it before calling the registry, so the outcome
is a reported REFUSED rather than an :class:`~atreides_vision.errors.IdentityError`
interrupting the round.

Every other :class:`~atreides_vision.errors.IdentityError` the registry can
raise — the registry at capacity, venue time moving backwards relative to a
subject's last activity — is a precondition violation this layer does not
model as a fourth kind of refusal, and is left to propagate. Both are the
caller's responsibility (retire or expire subjects; process rounds in
non-decreasing venue time, exactly as V0.4.0 through V0.4.2 already require)
rather than an ordinary outcome of a well-formed round.

TrackletKey reconciliation
---------------------------
``association.models.TrackletKey`` and ``identity.models.TrackletKey`` are
independently declared, structurally identical aliases for
``tuple[str, int, int]`` — deliberately so, per each package's own boundary
guard, which forbids either from importing the other. This module is the one
place both meanings are in scope, and it never treats either alias as a value
to pass across the seam: every registry lookup here is built fresh from a
``TrackObservation``'s own ``subject`` (``SubjectRef``) and ``stream``
(``StreamRef``) through ``identity.tracklet_key()`` — the actual shared
vocabulary both packages already key their own aliases from. No raw tuple
crosses the boundary, and no third alias is introduced.
"""

from __future__ import annotations

from atreides_vision.association.models import AssociationRound, PairDecision, TrackObservation
from atreides_vision.fusion.models import IntegrationOutcome, IntegrationResult, MembershipOutcome
from atreides_vision.identity.models import TrackletMembership, VenueSubjectId, tracklet_key
from atreides_vision.identity.registry import VenueSubjectRegistry

__all__ = ["integrate_round"]


def integrate_round(registry: VenueSubjectRegistry, round_: AssociationRound) -> IntegrationResult:
    """Apply one association round's positive decisions to a venue identity registry.

    Processes ``round_.decisions`` in their already-canonical order (see
    :class:`~atreides_vision.association.models.AssociationRound`), so the
    result never depends on how the round's cameras were enumerated. Mutates
    ``registry`` in place, matching the registry's own design — it is the
    only place a membership changes — and returns a complete, ordered account
    of what happened to every decision, including the ones that were never
    eligible to apply.

    Calling this twice with the same round is safe: every decision that
    already changed the registry resolves to a ``NO_OP`` the second time, and
    every ``REFUSED`` or ``IGNORED`` decision left the registry untouched to
    begin with.
    """
    outcomes = tuple(
        _integrate_decision(registry, decision, round_.venue_seconds)
        for decision in round_.decisions
    )
    return IntegrationResult(venue_seconds=round_.venue_seconds, outcomes=outcomes)


def _integrate_decision(
    registry: VenueSubjectRegistry, decision: PairDecision, at_seconds: float
) -> MembershipOutcome:
    if not decision.is_associated:
        return MembershipOutcome(
            decision=decision,
            outcome=IntegrationOutcome.IGNORED,
            detail=f"outcome is {decision.outcome.value}, not a positive association",
        )

    first, second = decision.pair.first, decision.pair.second
    owner_first = registry.owner_of(tracklet_key(first.subject, first.stream))
    owner_second = registry.owner_of(tracklet_key(second.subject, second.stream))

    if owner_first is None and owner_second is None:
        return _create_both(registry, decision, first, second, at_seconds)
    if owner_first is not None and owner_second is not None:
        return _reconcile_existing(decision, owner_first, owner_second)
    if owner_first is not None:
        return _extend(registry, decision, owner_first, first, second, at_seconds)
    assert owner_second is not None
    return _extend(registry, decision, owner_second, second, first, at_seconds)


def _create_both(
    registry: VenueSubjectRegistry,
    decision: PairDecision,
    first: TrackObservation,
    second: TrackObservation,
    at_seconds: float,
) -> MembershipOutcome:
    subject_id = registry.create(_membership(first, at_seconds))
    registry.add_membership(subject_id, _membership(second, at_seconds))
    return MembershipOutcome(
        decision=decision,
        outcome=IntegrationOutcome.APPLIED,
        detail=(
            f"created {subject_id} from {first.subject} and {second.subject}: neither tracklet "
            "was previously owned"
        ),
        subject_ids=(subject_id,),
    )


def _extend(
    registry: VenueSubjectRegistry,
    decision: PairDecision,
    owned_id: VenueSubjectId,
    owned_obs: TrackObservation,
    unowned_obs: TrackObservation,
    at_seconds: float,
) -> MembershipOutcome:
    existing = registry.get(owned_id)
    if existing.member_on(unowned_obs.camera_id) is not None:
        return MembershipOutcome(
            decision=decision,
            outcome=IntegrationOutcome.REFUSED,
            detail=(
                f"{owned_id} already has an active tracklet on camera "
                f"'{unowned_obs.camera_id}'; adding {unowned_obs.subject} would claim two "
                "simultaneous tracklets on one camera"
            ),
            subject_ids=(owned_id,),
        )
    registry.add_membership(owned_id, _membership(unowned_obs, at_seconds))
    return MembershipOutcome(
        decision=decision,
        outcome=IntegrationOutcome.APPLIED,
        detail=f"added {unowned_obs.subject} to {owned_id}, already owning {owned_obs.subject}",
        subject_ids=(owned_id,),
    )


def _reconcile_existing(
    decision: PairDecision, owner_first: VenueSubjectId, owner_second: VenueSubjectId
) -> MembershipOutcome:
    if owner_first == owner_second:
        return MembershipOutcome(
            decision=decision,
            outcome=IntegrationOutcome.NO_OP,
            detail=f"both tracklets are already active members of {owner_first}",
            subject_ids=(owner_first,),
        )
    low, high = sorted((owner_first, owner_second))
    return MembershipOutcome(
        decision=decision,
        outcome=IntegrationOutcome.REFUSED,
        detail=(
            f"tracklets are owned by two different venue subjects, {low} and {high}; merging "
            "existing venue identities is not a decision this layer makes from one association"
        ),
        subject_ids=(low, high),
    )


def _membership(observation: TrackObservation, at_seconds: float) -> TrackletMembership:
    return TrackletMembership(
        subject=observation.subject, stream=observation.stream, joined_at_seconds=at_seconds
    )

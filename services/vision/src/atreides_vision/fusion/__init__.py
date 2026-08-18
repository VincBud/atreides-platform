"""Association -> venue identity integration — V0.4.3b: the seam itself.

Where this sits
----------------
::

    detection -> camera-local tracking -> calibration / spatial context
      -> venue-time alignment                     (V0.4.0)
      -> comparability gating + geometry          (V0.4.1)
      -> deterministic association                (V0.4.2)
      -> temporary venue-local identity            (V0.4.3a)
      -> association -> venue identity integration <- V0.4.3b builds this
      -> venue-level evidence -> decision support (later)

What V0.4.3b builds
--------------------
:func:`~atreides_vision.fusion.integration.integrate_round` — the one function
that turns one :class:`~atreides_vision.association.models.AssociationRound`
into safe, deterministic membership operations against a
:class:`~atreides_vision.identity.registry.VenueSubjectRegistry`. Only a
positive ``ASSOCIATED`` decision may ever create or extend venue-local
identity; every other decision is recorded IGNORED. Every decision's fate —
applied, a no-op, refused, or ignored — is reported as a
:class:`~atreides_vision.fusion.models.MembershipOutcome`, carrying its full
V0.4.2 evidence and an explanation of the rule that produced it.

What V0.4.3b does not do
-------------------------
It does not decide whether two tracklets correspond — that is association's
job, unchanged. It does not own venue-local identity state — that is the
registry's job, unchanged. It never merges two already-distinct venue
subjects and never splits one: a cross-subject conflict is reported REFUSED,
not resolved. No automatic duplicate-count suppression, no venue-level
interaction or crowd fusion, no ReID or appearance evidence, no long-gap or
non-overlap handoff, no pipeline/CLI wiring, no reporting/JSONL schema, no
risk or decision engine. All of that is later or separate work.
"""

from __future__ import annotations

from atreides_vision.fusion.integration import integrate_round
from atreides_vision.fusion.models import IntegrationOutcome, IntegrationResult, MembershipOutcome

__all__ = [
    "IntegrationOutcome",
    "IntegrationResult",
    "MembershipOutcome",
    "integrate_round",
]

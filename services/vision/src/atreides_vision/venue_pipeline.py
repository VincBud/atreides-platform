"""Venue-level runtime wiring — V0.4.4: association -> fusion -> venue identity.

V0.4.2 built :class:`~atreides_vision.association.engine.AssociationEngine`,
the facade that decides every candidate pair at one venue instant.
V0.4.3a/b built :class:`~atreides_vision.identity.registry.VenueSubjectRegistry`
and :func:`~atreides_vision.fusion.integration.integrate_round`, the identity
state and the policy that turns a positive association into a membership
change. Nothing before this milestone ran them together: each was fully
implemented and tested in isolation, with no pipeline calling them in
sequence for one venue session.

This module is that pipeline, and nothing more. It owns the lifecycle of the
two stateful, per-session collaborators — one :class:`AssociationEngine`, one
:class:`VenueSubjectRegistry` — and drives them through the one sequence each
already documented as its own contract:

::

    camera-local observations, already venue-time aligned and projected
      -> AssociationEngine.observe(venue_seconds, observations)   (V0.4.2, unchanged)
      -> AssociationRound
      -> fusion.integrate_round(registry, round_)                 (V0.4.3b, unchanged)
      -> VenueSubjectRegistry mutated only through that call        (V0.4.3a, unchanged)
      -> VenueInstantResult                                        <- V0.4.4 builds this

Where this sits
----------------
Nothing in :mod:`atreides_vision.venue`, :mod:`atreides_vision.association`,
:mod:`atreides_vision.identity` or :mod:`atreides_vision.fusion` is touched or
reimplemented here — each package's own boundary guard already forbids the
cross-cutting reach this orchestration needs (``association`` cannot see
``identity``, ``venue`` cannot see any of the three), so this module lives at
the top level, free to depend on all three, and depends on nothing else. It
does not import :mod:`atreides_vision.venue` or :mod:`atreides_vision.video`:
this milestone wires already-built :class:`~atreides_vision.association.models.TrackObservation`
values into association and fusion, it does not derive them from raw frames,
pixels or camera-local tracks. That adapter — projecting a camera-local track
into a venue-time-aligned, spatially-contextualised observation — needs
calibration and projector access this milestone does not own, and inventing
missing timing confidence, calibration trust or spatial context here would be
exactly the kind of silent fabrication the association gates exist to refuse.
Building an observation is the caller's responsibility; this pipeline's
contract begins at the observation.

What this milestone does not do
--------------------------------
No occupancy, no duplicate-suppression product metric, no cumulative unique
visitor count, no JSONL/reporting schema, no dashboard, no risk or decision
engine, no venue-level behaviour/crowd/interaction fusion, no automatic merge
of two already-distinct venue subjects — every one of those stays exactly
where V0.4.2/V0.4.3 left it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from atreides_vision.association import (
    AssociationDecisionConfig,
    AssociationEngine,
    AssociationGatingConfig,
    AssociationRound,
    TrackObservation,
)
from atreides_vision.fusion import IntegrationResult, integrate_round
from atreides_vision.identity import VenueIdentityConfig, VenueSubjectId, VenueSubjectRegistry

__all__ = ["VenueInstantResult", "VenuePipeline"]


@dataclass(frozen=True, slots=True)
class VenueInstantResult:
    """Everything one venue instant produced, from observations to identity.

    Immutable and complete, matching :class:`AssociationRound` and
    :class:`~atreides_vision.fusion.models.IntegrationResult` themselves:
    nothing here is filtered, so "nothing happened this instant" is as
    inspectable as "a venue subject was created".
    """

    #: The session-relative venue instant this result describes.
    venue_seconds: float
    #: The observations the engine considered, in the canonical order
    #: :class:`AssociationRound` itself already guarantees.
    observations: tuple[TrackObservation, ...]
    #: V0.4.2's own decision for every candidate pair at this instant.
    round: AssociationRound
    #: V0.4.3b's own account of what each positive decision did to identity.
    integration: IntegrationResult
    #: Every venue subject currently active, in canonical (sequence) order,
    #: after this instant's integration — a read of registry state, not a
    #: second copy of it.
    active_subjects: tuple[VenueSubjectId, ...]


class VenuePipeline:
    """Coordinates one venue session's association, fusion and identity state.

    One pipeline instance is one venue session: it owns exactly one
    :class:`AssociationEngine` and one :class:`VenueSubjectRegistry` for the
    lifetime of the object, and nothing here is global or shared between
    instances. A single-camera session takes the same path as any other —
    the engine already produces ``NOT_COMPARABLE`` for every pair on its own,
    fusion ignores every one of them, and the registry never gains a member.

    Camera-local ``SubjectRef`` stays camera-local throughout: this class
    never allocates one, never mutates one, and every ``VenueSubjectId`` it
    can produce comes only from the registry it owns.
    """

    __slots__ = ("_engine", "_registry")

    def __init__(
        self,
        *,
        gating: AssociationGatingConfig | None = None,
        decisions: AssociationDecisionConfig | None = None,
        identity: VenueIdentityConfig | None = None,
    ) -> None:
        self._engine = AssociationEngine(gating=gating, decisions=decisions)
        self._registry = VenueSubjectRegistry(identity)

    @property
    def registry(self) -> VenueSubjectRegistry:
        """The session's venue identity registry.

        Exposed for its own read surface (:meth:`VenueSubjectRegistry.get`,
        :meth:`~VenueSubjectRegistry.provenance_for`, and so on) — every
        mutation still happens only through :meth:`process_instant` and
        :meth:`expire`, exactly as :func:`~atreides_vision.fusion.integrate_round`
        and the registry itself already require.
        """
        return self._registry

    @property
    def tracked_pairs(self) -> int:
        """How many candidate pairs the association engine currently holds evidence for."""
        return self._engine.tracked_pairs

    def process_instant(
        self, venue_seconds: float, observations: Iterable[TrackObservation]
    ) -> VenueInstantResult:
        """Run one venue instant through association and into venue identity.

        Calls :meth:`AssociationEngine.observe` exactly once, then
        :func:`~atreides_vision.fusion.integrate_round` exactly once, in that
        order — the full sequence this milestone wires and nothing more.
        Observation ordering is not re-canonicalised here: the engine already
        sorts by :func:`~atreides_vision.association.observation_order_key`
        and rejects a duplicate tracklet within the instant, so this call's
        result is independent of the order ``observations`` was iterated in.

        An empty ``observations`` iterable is a legal, idempotent no-op
        instant: an ``AssociationRound`` with no observations and no
        decisions, an ``IntegrationResult`` with no outcomes, and the
        registry's active subjects unchanged. This is what a stream that has
        ended, or one that has not started decoding yet, produces.

        Raises:
            AssociationError: propagated unchanged from
                :meth:`AssociationEngine.observe` — if ``venue_seconds`` does
                not strictly advance past this pipeline's last processed
                instant, or if one tracklet appears twice in ``observations``.
                Both are structural violations of this pipeline's own
                contract, never an ordinary association outcome.
            FusionError: propagated unchanged from
                :func:`~atreides_vision.fusion.integrate_round`.
            IdentityError: propagated unchanged from the registry, for the
                precondition violations fusion itself does not model as a
                refusal (registry at capacity, venue time moving backwards
                for a subject fusion is not currently reconciling).
        """
        round_ = self._engine.observe(venue_seconds, observations)
        integration = integrate_round(self._registry, round_)
        return VenueInstantResult(
            venue_seconds=venue_seconds,
            observations=round_.observations,
            round=round_,
            integration=integration,
            active_subjects=self._registry.active_subjects(),
        )

    def expire(self, now_seconds: float) -> tuple[VenueSubjectId, ...]:
        """Advance venue-local identity lifecycle to ``now_seconds`` of venue time.

        Delegates entirely to :meth:`VenueSubjectRegistry.expire` — this
        pipeline introduces no second expiry policy. Callers decide when to
        call it (for example, once per processed instant, using that
        instant's own ``venue_seconds``); it is never called implicitly by
        :meth:`process_instant`, so an instant's own membership changes are
        never retired by the same call that produced them.
        """
        return self._registry.expire(now_seconds)

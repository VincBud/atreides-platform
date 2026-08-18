"""The deterministic association core: candidates in, decided pairs out.

This is the facade the pipeline orchestrates. One call per venue instant:
give it the observations that instant contains and it returns an
:class:`~atreides_vision.association.models.AssociationRound` — a decision for
every candidate pair, including the ones a hard gate refused, each carrying the
evidence it was decided from.

What it takes to be associated
------------------------------
All of the following, at the same time, or the answer is not ``ASSOCIATED``:

1. every hard comparability gate passed — same venue frame, same surface,
   calibrated metres, trusted geometry, established venue time, simultaneous,
   overlapping calibrated region;
2. geometry admits the pair: their separation is inside the radius that position
   tolerance plus the motion the timing uncertainty allows;
3. each is the other's nearest admissible candidate **on the other's camera**;
4. the runner-up on each side is at least the acceptance margin further away,
   and neither side's counterpart camera was already declared ambiguous;
5. that agreement has held, unbroken, for at least the configured number of
   observations **and** the configured span of venue time;
6. nothing in the wider group the association would form contradicts itself.

Anything less is ``AMBIGUOUS`` or ``UNASSOCIATED``, and which one is recorded
with the rule that produced it. Geometry alone never suffices: step 2 can be
satisfied at every instant by two strangers standing near each other, and steps
3 to 6 are what stop that becoming an association.

What it deliberately is not
---------------------------
It produces no identity. A decision here says two tracklets appear to be one
subject *right now*; it does not name that subject, does not count subjects, and
does not join the pairs into anything. Turning associations into a venue-local
temporary identity is a distinct capability with its own failure modes, and the
clean, provenance-carrying input for it is exactly what this module emits.

Determinism
-----------
Observations are processed in canonical order, pairs are decided in canonical
key order, groups are formed from sorted edges, and every lookup structure is
looked up rather than iterated. The same instants in a different camera
enumeration order produce byte-identical records.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from atreides_vision.association.candidates import gate_observation
from atreides_vision.association.config import (
    AssociationDecisionConfig,
    AssociationGatingConfig,
)
from atreides_vision.association.history import EvidenceSample, PairEvidenceStore
from atreides_vision.association.matching import (
    CameraPreference,
    camera_preferences,
    inconsistent_pairs,
    pair_agreement,
)
from atreides_vision.association.models import (
    AssociationOutcome,
    AssociationRound,
    CandidateKey,
    DecisionReason,
    PairAssessment,
    PairDecision,
    SupportSummary,
    TrackletKey,
    TrackObservation,
    candidate_key,
    observation_order_key,
    tracklet_key,
)
from atreides_vision.errors import AssociationError

__all__ = ["AssociationEngine"]


class AssociationEngine:
    """Decides candidate pairs over time, holding only short-lived evidence.

    One engine per session. Its whole state is the bounded evidence store, which
    is temporary, operational and discarded with the process — there is no
    identity here to persist and nothing here survives the session.

    A single-camera venue is not a special case: every pair is refused by the
    same-camera gate, every decision comes back ``NOT_COMPARABLE``, no evidence
    accumulates, and the same code path runs.
    """

    __slots__ = ("_decisions", "_evidence", "_gating", "_last_seconds")

    def __init__(
        self,
        gating: AssociationGatingConfig | None = None,
        decisions: AssociationDecisionConfig | None = None,
    ) -> None:
        self._gating = gating or AssociationGatingConfig()
        self._decisions = decisions or AssociationDecisionConfig()
        self._evidence = PairEvidenceStore()
        self._last_seconds: float | None = None

    @property
    def tracked_pairs(self) -> int:
        """How many candidate pairs currently hold evidence. For observability."""
        return len(self._evidence)

    def observe(
        self, venue_seconds: float, observations: Iterable[TrackObservation]
    ) -> AssociationRound:
        """Decide every candidate pair among ``observations`` at one venue instant.

        ``venue_seconds`` is the session-relative venue time the instant belongs
        to; it must advance between calls, because evidence about how long an
        agreement held is meaningless if instants may be replayed or reordered.

        Raises:
            AssociationError: if the instant does not advance, or if one
                tracklet appears twice in it.
        """
        ordered = self._ordered(venue_seconds, observations)
        gatings = [gate_observation(observation, ordered, self._gating) for observation in ordered]

        assessments: dict[CandidateKey, PairAssessment] = {}
        for gating in gatings:
            for assessment in gating.assessments:
                assessments.setdefault(candidate_key(assessment.pair), assessment)
        preferences: dict[TrackletKey, Sequence[CameraPreference]] = {
            tracklet_key(gating.observation): camera_preferences(gating) for gating in gatings
        }

        provisional: dict[CandidateKey, tuple[PairAssessment, SupportSummary]] = {}
        settled: list[PairDecision] = []
        for key in sorted(assessments):
            assessment = assessments[key]
            decision = self._decide(key, assessment, preferences, venue_seconds)
            if isinstance(decision, PairDecision):
                settled.append(decision)
            else:
                provisional[key] = decision

        settled.extend(self._resolve(provisional, assessments))
        self._evidence.prune(venue_seconds, self._decisions)
        self._last_seconds = venue_seconds
        return AssociationRound(
            venue_seconds=venue_seconds,
            observations=tuple(ordered),
            decisions=tuple(sorted(settled, key=lambda decision: decision.key)),
        )

    def _ordered(
        self, venue_seconds: float, observations: Iterable[TrackObservation]
    ) -> list[TrackObservation]:
        if self._last_seconds is not None and venue_seconds <= self._last_seconds:
            raise AssociationError(
                f"Venue instants must advance: {venue_seconds} follows {self._last_seconds}. "
                "Evidence about how long an agreement held cannot be replayed."
            )
        ordered = sorted(observations, key=observation_order_key)
        keys = [tracklet_key(observation) for observation in ordered]
        if len(set(keys)) != len(keys):
            raise AssociationError(
                f"One tracklet appears twice at venue second {venue_seconds}. At one instant a "
                "tracklet is in one place, and two positions for it would be compared as rivals."
            )
        return ordered

    def _decide(
        self,
        key: CandidateKey,
        assessment: PairAssessment,
        preferences: dict[TrackletKey, Sequence[CameraPreference]],
        venue_seconds: float,
    ) -> PairDecision | tuple[PairAssessment, SupportSummary]:
        """Decide one pair, or defer it pending the group consistency check."""
        if not assessment.is_comparable:
            # Nothing was measured, so whatever was agreed before happened under
            # conditions that no longer hold. Forgetting it means a later run
            # starts from evidence, not from a gap.
            self._evidence.discard(key)
            return PairDecision(
                assessment=assessment,
                outcome=AssociationOutcome.NOT_COMPARABLE,
                reasons=(DecisionReason.HARD_GATE_REJECTED,),
            )

        evidence = assessment.evidence
        if evidence is None:  # pragma: no cover - a comparable pair carries evidence
            raise AssertionError("a comparable pair must carry its geometric evidence")

        agreement = pair_agreement(assessment.pair, preferences, self._decisions)
        agrees = evidence.is_admissible and agreement.agrees
        support = self._evidence.observe(
            key,
            EvidenceSample(
                venue_seconds=venue_seconds,
                agrees=agrees,
                trust=evidence.trust,
                separation_m=evidence.separation_m,
                rival_margin_m=agreement.rival_margin_m,
            ),
            self._decisions,
        )

        if not evidence.is_admissible:
            return PairDecision(
                assessment=assessment,
                outcome=AssociationOutcome.UNASSOCIATED,
                reasons=(DecisionReason.GEOMETRY_EXCLUDED,),
                support=support.summary(),
            )
        if not agrees:
            # ``pair_agreement`` names a reason whenever it refuses.
            reason = agreement.reason or DecisionReason.NOT_MUTUAL_BEST
            return PairDecision(
                assessment=assessment,
                outcome=AssociationOutcome.AMBIGUOUS,
                reasons=(reason,),
                support=support.summary(),
            )
        if not support.is_sustained(self._decisions):
            return PairDecision(
                assessment=assessment,
                outcome=AssociationOutcome.AMBIGUOUS,
                reasons=(DecisionReason.SUPPORT_NOT_SUSTAINED,),
                support=support.summary(),
            )
        return (assessment, support.summary())

    def _resolve(
        self,
        provisional: dict[CandidateKey, tuple[PairAssessment, SupportSummary]],
        assessments: dict[CandidateKey, PairAssessment],
    ) -> list[PairDecision]:
        """Turn sustained agreements into associations, minus the contradictory ones."""
        contradicted = dict(inconsistent_pairs(sorted(provisional), assessments))
        resolved = []
        for key in sorted(provisional):
            assessment, support = provisional[key]
            reason = contradicted.get(key)
            if reason is not None:
                resolved.append(
                    PairDecision(
                        assessment=assessment,
                        outcome=AssociationOutcome.AMBIGUOUS,
                        reasons=(reason,),
                        support=support,
                    )
                )
                continue
            resolved.append(
                PairDecision(
                    assessment=assessment,
                    outcome=AssociationOutcome.ASSOCIATED,
                    reasons=(DecisionReason.SUSTAINED_MUTUAL_AGREEMENT,),
                    support=support,
                    confidence=support.confidence,
                )
            )
        return resolved

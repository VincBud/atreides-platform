"""Multi-camera association — V0.4.2: deciding candidates, conservatively, over time.

Association is the V1 capability that turns several cameras into one venue view.
V0.4.0 built the time domain underneath it: N streams on one session-relative
venue timeline, each frame carrying its stream provenance and its timing
uncertainty. V0.4.1 built the vocabulary of a cross-camera comparison and the
hard preconditions one must clear. V0.4.2 adds the layer that decides: it
accumulates short-lived evidence about candidate pairs and, when mutual,
exclusive, unambiguous agreement has held long enough, says so.

Where this sits
---------------
::

    detection -> camera-local tracking -> calibration / spatial context
      -> venue-time alignment                     (V0.4.0)
      -> comparability gating + geometry          (V0.4.1)
      -> deterministic association                <- V0.4.2 builds this
      -> temporary venue-local identity           (later)
      -> venue-level evidence -> decision support (later)

Cross-camera logic lives here and nowhere else. No tracker, tracker adapter,
detector or camera-local analyzer knows another camera exists, and this package
imports none of them: it consumes Atreides domain values — a subject reference,
a stream reference, a venue time with its uncertainty, a projected position, a
camera's spatial context — and nothing that could reach a pixel.

What geometry may and may not do
--------------------------------
**Geometry rejects; it never associates on its own.** Two people standing on one
square metre of floor at one instant, seen by two cameras, are geometrically
identical to one person seen twice. Agreement in space and time is therefore a *precondition* for
believing two tracklets are the same person and never a *reason* to. So this
gating layer produces ``NOT_COMPARABLE``, ``UNASSOCIATED`` and ``AMBIGUOUS``,
and :class:`~atreides_vision.association.models.ObservationGating` refuses to be
constructed with ``ASSOCIATED`` at all.

``ASSOCIATED`` exists one level up, in
:class:`~atreides_vision.association.engine.AssociationEngine`, and needs far
more than proximity: both sides preferring each other on each other's camera,
no rival close enough to make that a coin flip, and that agreement holding
unbroken across a configured number of observations *and* a configured span of
venue time — then a group-level consistency check that stops pairwise agreement
becoming a transitive merge nobody measured. Anything short of all of it is
``AMBIGUOUS`` or ``UNASSOCIATED``, recorded with the rule that refused it. A
wrong merge corrupts every venue-level number above it and is harder to detect
than a missed one; the failure this package is built to prefer is the missed
one.

What V0.4.2 does not do
-----------------------
No venue-local identity type, no merge, no fused subject, no duplicate-count
suppression, no headcount — a decision here concerns two tracklets and names
nobody. No appearance descriptor, embedding, ReID or biometric of any kind: an
observation carries a projected floor position and nothing that could describe
how somebody looks. No hand-off matching across
non-overlapping cameras, no long-gap association, no same-camera fragment
stitching, no learned or trained matching, no venue-level crowd or interaction
fusion, and no risk or decision engine.

Single camera is the same path
------------------------------
One camera produces no comparable pairs, so every decision comes back
``NOT_COMPARABLE`` and no evidence accumulates: there is no cross-camera
evidence to have, and the layer says so instead of erroring, branching or
inventing one. Non-overlapping cameras
behave the same way, and are an ordinary deployment rather than a defect.
"""

from __future__ import annotations

from atreides_vision.association.candidates import (
    assess_pair,
    candidate_pairs,
    gate_observation,
)
from atreides_vision.association.config import (
    AssociationDecisionConfig,
    AssociationGatingConfig,
)
from atreides_vision.association.engine import AssociationEngine
from atreides_vision.association.gates import HARD_GATE_ORDER, comparability_rejections
from atreides_vision.association.geometry import (
    admissible_radius_m,
    earliest_possible_gap_seconds,
    latest_possible_gap_seconds,
    motion_allowance_m,
    nominal_gap_seconds,
    separation_metres,
    timing_slack_seconds,
)
from atreides_vision.association.history import (
    EvidenceSample,
    PairEvidence,
    PairEvidenceStore,
)
from atreides_vision.association.matching import (
    CameraPreference,
    PairAgreement,
    camera_preferences,
    inconsistent_pairs,
    pair_agreement,
)
from atreides_vision.association.models import (
    AssociationOutcome,
    AssociationRound,
    CameraSpatialContext,
    CandidateKey,
    CandidatePair,
    DecisionConfidence,
    DecisionReason,
    GateReason,
    GateRejection,
    GeometricEvidence,
    GeometryTrust,
    ObservationGating,
    PairAssessment,
    PairDecision,
    SupportSummary,
    TrackletKey,
    TrackObservation,
    VenueRegion,
    candidate_key,
    observation_order_key,
    tracklet_key,
)

__all__ = [
    "HARD_GATE_ORDER",
    "AssociationDecisionConfig",
    "AssociationEngine",
    "AssociationGatingConfig",
    "AssociationOutcome",
    "AssociationRound",
    "CameraPreference",
    "CameraSpatialContext",
    "CandidateKey",
    "CandidatePair",
    "DecisionConfidence",
    "DecisionReason",
    "EvidenceSample",
    "GateReason",
    "GateRejection",
    "GeometricEvidence",
    "GeometryTrust",
    "ObservationGating",
    "PairAgreement",
    "PairAssessment",
    "PairDecision",
    "PairEvidence",
    "PairEvidenceStore",
    "SupportSummary",
    "TrackObservation",
    "TrackletKey",
    "VenueRegion",
    "admissible_radius_m",
    "assess_pair",
    "camera_preferences",
    "candidate_key",
    "candidate_pairs",
    "comparability_rejections",
    "earliest_possible_gap_seconds",
    "gate_observation",
    "inconsistent_pairs",
    "latest_possible_gap_seconds",
    "motion_allowance_m",
    "nominal_gap_seconds",
    "observation_order_key",
    "pair_agreement",
    "separation_metres",
    "timing_slack_seconds",
    "tracklet_key",
]

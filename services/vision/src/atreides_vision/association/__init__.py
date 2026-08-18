"""Multi-camera association — V0.4.1: the domain, and the gates geometry can enforce.

Association is the V1 capability that turns several cameras into one venue view.
V0.4.0 built the time domain underneath it: N streams on one session-relative
venue timeline, each frame carrying its stream provenance and its timing
uncertainty. V0.4.1 builds the layer immediately above — the vocabulary in which
a cross-camera comparison is expressed, and the hard preconditions such a
comparison must clear before it may be attempted at all.

Where this sits
---------------
::

    detection -> camera-local tracking -> calibration / spatial context
      -> venue-time alignment                     (V0.4.0)
      -> comparability gating + geometry          <- V0.4.1 builds this
      -> deterministic association                (later)
      -> temporary venue-local identity           (later)
      -> venue-level evidence -> decision support (later)

Cross-camera logic lives here and nowhere else. No tracker, tracker adapter,
detector or camera-local analyzer knows another camera exists, and this package
imports none of them: it consumes Atreides domain values — a subject reference,
a stream reference, a venue time with its uncertainty, a projected position, a
camera's spatial context — and nothing that could reach a pixel.

The one thing it will not do
----------------------------
**Geometry rejects; it never merges.** Two people standing on one square metre
of floor at one instant, seen by two cameras, are geometrically identical to one
person seen twice. Agreement in space and time is therefore a *precondition* for
believing two tracklets are the same person and never a *reason* to. So this
layer produces ``NOT_COMPARABLE``, ``UNASSOCIATED`` and ``AMBIGUOUS``, and
:class:`~atreides_vision.association.models.ObservationGating` refuses to be
constructed with ``ASSOCIATED`` at all. A wrong merge corrupts every venue-level
number above it and is harder to detect than a missed one; the failure this
package is built to prefer is the missed one.

What V0.4.1 does not do
-----------------------
No venue-local identity type, no merge, no fused subject, no duplicate-count
suppression, no headcount. No appearance descriptor, embedding, ReID or
biometric of any kind — an observation carries a projected floor position and
nothing that could describe how somebody looks. No hand-off matching across
non-overlapping cameras, no long-gap association, no same-camera fragment
stitching, no learned or trained matching, no venue-level crowd or interaction
fusion, and no risk or decision engine.

Single camera is the same path
------------------------------
One camera produces no candidate pairs, so every observation gates to
``NOT_COMPARABLE``: there is no cross-camera evidence to have, and the layer
says so instead of erroring, branching or inventing one. Non-overlapping cameras
behave the same way, and are an ordinary deployment rather than a defect.
"""

from __future__ import annotations

from atreides_vision.association.candidates import (
    assess_pair,
    candidate_pairs,
    gate_observation,
)
from atreides_vision.association.config import AssociationGatingConfig
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
from atreides_vision.association.models import (
    AssociationOutcome,
    CameraSpatialContext,
    CandidatePair,
    GateReason,
    GateRejection,
    GeometricEvidence,
    GeometryTrust,
    ObservationGating,
    PairAssessment,
    TrackObservation,
    VenueRegion,
    observation_order_key,
)

__all__ = [
    "HARD_GATE_ORDER",
    "AssociationGatingConfig",
    "AssociationOutcome",
    "CameraSpatialContext",
    "CandidatePair",
    "GateReason",
    "GateRejection",
    "GeometricEvidence",
    "GeometryTrust",
    "ObservationGating",
    "PairAssessment",
    "TrackObservation",
    "VenueRegion",
    "admissible_radius_m",
    "assess_pair",
    "candidate_pairs",
    "comparability_rejections",
    "earliest_possible_gap_seconds",
    "gate_observation",
    "latest_possible_gap_seconds",
    "motion_allowance_m",
    "nominal_gap_seconds",
    "observation_order_key",
    "separation_metres",
    "timing_slack_seconds",
]

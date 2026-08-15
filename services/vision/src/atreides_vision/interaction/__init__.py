"""Interaction signals — V0.3.4 PERSON-TO-PERSON MEASUREMENT.

Turns the behaviour layer's per-subject snapshots into **neutral pairwise
measurements**: how far apart two observed subjects are, how fast that gap is
changing, how their directions of travel relate, and how long they have stayed
close.

Measurement, not interpretation
-------------------------------
"0.8 m apart, closing at 1.1 m/s, together for 3.4 s" is a measurement.
Whether it means anything — a conversation, a queue, a scuffle, two strangers
passing — is a question this package deliberately cannot answer, and there is no
signal type for confrontation, aggression, fighting, threat, conflict,
escalation, harassment or stalking. Those require evidence Atreides does not
have, and a package that measured *and* judged would make both jobs impossible
to audit.

Identity boundary
-----------------
A pair is two **camera-local** subjects on **one** camera. Cross-camera pairing
is rejected outright, not merely unimplemented: without multi-camera
association, one real person visible in two cameras would be paired with
themselves and produce a confident measurement of a relationship that does not
exist. See :class:`SubjectPairRef`.

A camera-local track id is not a unique real-world person. If a tracker loses
someone and re-acquires them under a new id, the old pairs end and different
ones begin — nothing here stitches them together.

Nothing in this package computes appearance descriptors, biometric identifiers,
persistent identity or any cross-camera association.

Coordinate safety
-----------------
Positions are only ever subtracted when they are legitimately comparable:
matching coordinate space, matching camera, and — for metric analysis —
matching venue and surface. A distance between the main floor and a balcony
would be a fabrication, so :class:`SpatialContext` refuses to produce one.

Layering
--------
``tracks → BehaviorAnalyzer → BehaviorFrame → InteractionAnalyzer``

Pairwise state lives here, never in the behaviour analyzer. This package
imports no detector, no tracker and no zone engine, and performs no pixel
arithmetic.
"""

from __future__ import annotations

from atreides_vision.interaction.analyzer import InteractionAnalyzer
from atreides_vision.interaction.config import InteractionAnalysisConfig
from atreides_vision.interaction.geometry import (
    closing_speed,
    pair_distance,
    proximity_since,
    relative_motion_heading,
)
from atreides_vision.interaction.history import PairHistory, PairHistoryStore, PairSample
from atreides_vision.interaction.models import (
    InteractionFrame,
    InteractionSignal,
    InteractionSignalType,
    InteractionSnapshot,
    PairGeometry,
    SpatialContext,
    SubjectPairRef,
    interaction_unit_for,
)
from atreides_vision.interaction.reporting import (
    INTERACTION_SCHEMA_VERSION,
    InteractionRunMetadata,
    InteractionWriter,
    JsonlInteractionWriter,
    interaction_records,
)

__all__ = [
    "INTERACTION_SCHEMA_VERSION",
    "InteractionAnalysisConfig",
    "InteractionAnalyzer",
    "InteractionFrame",
    "InteractionRunMetadata",
    "InteractionSignal",
    "InteractionSignalType",
    "InteractionSnapshot",
    "InteractionWriter",
    "JsonlInteractionWriter",
    "PairGeometry",
    "PairHistory",
    "PairHistoryStore",
    "PairSample",
    "SpatialContext",
    "SubjectPairRef",
    "closing_speed",
    "interaction_records",
    "interaction_unit_for",
    "pair_distance",
    "proximity_since",
    "relative_motion_heading",
]

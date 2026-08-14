"""Behaviour signals — V0.3 SEE + TRACK + MEASURE HOW PEOPLE MOVE.

Turns camera-local track observations into neutral, measurable temporal
signals: speed, acceleration, turn rate and stationary duration. Nothing here
interprets those numbers. There is no aggression, intoxication, suspicion,
distress or threat classifier, and no risk score — a measurement layer that
also judged would make both jobs impossible to audit.

This layer consumes :class:`~atreides_vision.models.Track` objects and a
:class:`MotionProjector`, and nothing else. It has no dependency on RF-DETR,
OC-SORT, ByteTrack or the zone layer, so all of those remain replaceable and
zones remain optional.

Scale independence
------------------
No part of this package assumes a venue size, a camera's coverage area, a
mounting height, a field of view, or a resolution. Signals are computed in an
abstract :class:`MotionSpace` supplied by the projector, so a camera covering a
doorway and one covering a festival field run the same code. V0.3.0 ships one
projector, :class:`NormalizedFrameProjector`, which is resolution-independent
but **camera-local**: its distances are fractions of a frame, not metres, and
they are not comparable between arbitrary cameras. Calibrated ground-plane
projection is the planned next step and requires no change to the signals.

Identity and privacy boundary
-----------------------------
**A camera-local track id is not a unique real-world person.** Signals carry a
:class:`SubjectRef` — a *(camera_id, track_id)* pair — which is ephemeral,
session-local and discarded when the process exits.

This milestone computes no appearance embeddings, performs no facial or other
biometric recognition, persists no identity, and matches nobody across cameras.
Any future multi-camera fusion will sit *above* camera-local tracking and
*below* venue-level intelligence, mapping subject references onto venue
identities without changing anything in this package.
"""

from __future__ import annotations

from atreides_vision.behavior.analyzer import BehaviorAnalyzer
from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.behavior.coordinates import (
    MotionPoint,
    MotionProjector,
    MotionSpace,
    NormalizedFrameProjector,
    PersonAnchor,
    bottom_center_anchor,
)
from atreides_vision.behavior.history import SubjectHistory, SubjectHistoryStore
from atreides_vision.behavior.models import (
    BehaviorFrame,
    BehaviorObservation,
    BehaviorSignal,
    SignalType,
    SubjectRef,
    unit_for,
)

__all__ = [
    "BehaviorAnalysisConfig",
    "BehaviorAnalyzer",
    "BehaviorFrame",
    "BehaviorObservation",
    "BehaviorSignal",
    "MotionPoint",
    "MotionProjector",
    "MotionSpace",
    "NormalizedFrameProjector",
    "PersonAnchor",
    "SignalType",
    "SubjectHistory",
    "SubjectHistoryStore",
    "SubjectRef",
    "bottom_center_anchor",
    "unit_for",
]

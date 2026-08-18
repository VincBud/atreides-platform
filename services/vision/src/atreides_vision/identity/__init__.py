"""Temporary venue-local identity — V0.4.3a: the domain and its registry state.

Camera-local tracking produces a tracklet on one camera in one session; it is
not a person and is not unique across cameras. Venue-local identity is the
layer above it that gives a *hypothesis* — "these tracklets are one person,
right now, in this venue, in this session" — a name of its own, distinct from
every tracker integer that contributed to it.

Where this sits
----------------
::

    detection -> camera-local tracking -> calibration / spatial context
      -> venue-time alignment                     (V0.4.0)
      -> comparability gating + geometry          (V0.4.1)
      -> deterministic association                (V0.4.2)
      -> temporary venue-local identity            <- V0.4.3a builds this
      -> venue-level evidence -> decision support (later)

What V0.4.3a builds
--------------------
The identity namespace (:class:`~atreides_vision.identity.models.VenueSubjectId`),
deterministic session-local allocation, the shape of membership
(:class:`~atreides_vision.identity.models.TrackletMembership`) and its
provenance (:class:`~atreides_vision.identity.models.MembershipEvent`), and the
bounded, session-scoped registry
(:class:`~atreides_vision.identity.registry.VenueSubjectRegistry`) that
enforces one active tracklet per camera per identity and refuses conflicting
membership rather than reconciling it silently.

What V0.4.3a does not do
-------------------------
This package does not consume
:class:`~atreides_vision.association.models.AssociationRound` or any other
association output, and does not import the ``association`` package at all.
Every membership change here is an explicit registry operation performed by a
caller that already has a reason; nothing here decides *which* tracklets
belong together. No automatic merge, no automatic split, no duplicate-count
suppression, no venue-level crowd or interaction fusion, no appearance or
ReID, no cross-session identity, no reporting/JSONL schema, no CLI or
pipeline wiring, no risk or decision engine. All of that is later work.

Single camera is the same path
-------------------------------
A venue subject with exactly one camera behaves identically to one with many:
one membership, one camera claimed, the same invariants. There is no
single-camera mode and no special case.

Temporary by construction
--------------------------
A :class:`~atreides_vision.identity.models.VenueSubjectId` is session-scoped,
never written to a durable store, never carried between sessions or venues,
and never biometric — it names a membership hypothesis, not a real person.
"""

from __future__ import annotations

from atreides_vision.identity.config import VenueIdentityConfig
from atreides_vision.identity.models import (
    MembershipAction,
    MembershipEvent,
    TrackletKey,
    TrackletMembership,
    VenueSubject,
    VenueSubjectId,
    membership_order_key,
    tracklet_key,
)
from atreides_vision.identity.registry import VenueSubjectRegistry

__all__ = [
    "MembershipAction",
    "MembershipEvent",
    "TrackletKey",
    "TrackletMembership",
    "VenueIdentityConfig",
    "VenueSubject",
    "VenueSubjectId",
    "VenueSubjectRegistry",
    "membership_order_key",
    "tracklet_key",
]

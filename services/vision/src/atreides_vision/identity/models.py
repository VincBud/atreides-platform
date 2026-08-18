"""Domain types for temporary venue-local identity.

A camera-local track — :class:`~atreides_vision.behavior.models.SubjectRef` —
is a tracklet on one camera in one session; it is not a person and is not
unique across cameras. This module names the layer above it: a hypothesis,
scoped to one venue session, that a set of camera-local tracklets are one
person right now. That hypothesis is a :class:`VenueSubject`, addressed by a
:class:`VenueSubjectId` — a distinct type in its own namespace, never a reused
tracker integer and never derived from "lowest id wins".

Nothing here consumes an association decision. This milestone builds the
identity namespace and the shape of membership and provenance; deciding
*which* tracklets belong together is V0.4.2's
:class:`~atreides_vision.association.models.AssociationRound`, wired in by a
later milestone. Until then every membership change here is an explicit
operation performed by a caller, not an inference this layer makes on its own.

Nothing here is biometric. A membership is nothing more than *which*
camera-local tracklet contributed and *when* it was added or removed — no
appearance, no face, no template.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from atreides_vision.behavior.models import SubjectRef
from atreides_vision.errors import IdentityError
from atreides_vision.timing import StreamRef

__all__ = [
    "MembershipAction",
    "MembershipEvent",
    "TrackletKey",
    "TrackletMembership",
    "VenueSubject",
    "VenueSubjectId",
    "membership_order_key",
    "tracklet_key",
]

#: The stable camera-local identity of one tracklet: which camera, which
#: recording of that camera, which track number. Deliberately a plain tuple
#: of primitives rather than an import of
#: :class:`atreides_vision.association.models.TrackletKey` — this package
#: does not depend on the association layer at all in this milestone, and
#: the two are structurally identical only because they describe the same
#: underlying thing from two sides of the same boundary.
TrackletKey = tuple[str, int, int]


def tracklet_key(subject: SubjectRef, stream: StreamRef) -> TrackletKey:
    """The canonical, orderable key for one camera-local tracklet.

    Raises if ``subject`` and ``stream`` disagree about which camera they
    describe — a contradiction no caller should be able to construct.
    """
    if subject.camera_id != stream.camera_id:
        raise IdentityError(
            f"SubjectRef camera '{subject.camera_id}' does not match "
            f"StreamRef camera '{stream.camera_id}'"
        )
    return (subject.camera_id, stream.stream_epoch, subject.track_id)


@dataclass(frozen=True, slots=True, order=True)
class VenueSubjectId:
    """An opaque, temporary, venue/session-scoped identity.

    Not a track id, not a camera id, not derived from either. ``sequence`` is
    a monotonically increasing allocation counter private to one
    :class:`~atreides_vision.identity.registry.VenueSubjectRegistry` — it
    exists to make allocation deterministic and ids orderable, not to carry
    meaning. It never survives the session that allocated it and is never
    written to a durable store.
    """

    sequence: int

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise IdentityError(f"VenueSubjectId.sequence must be >= 1, got {self.sequence!r}")

    def __str__(self) -> str:
        return f"venue_subject_{self.sequence:06d}"

    def to_record(self) -> str:
        """Serialization-ready representation: the opaque string form."""
        return str(self)


@dataclass(frozen=True, slots=True)
class TrackletMembership:
    """One camera-local tracklet's contribution to one venue subject.

    Provenance, not evidence: this milestone does not carry the confidence or
    geometry that justified the membership, only *which* tracklet and *when*
    the registry was told about it. A later milestone that wires in
    :class:`~atreides_vision.association.models.AssociationRound` decisions is
    where that evidence attaches.
    """

    subject: SubjectRef
    stream: StreamRef
    joined_at_seconds: float

    def __post_init__(self) -> None:
        if self.subject.camera_id != self.stream.camera_id:
            raise IdentityError(
                f"TrackletMembership subject camera '{self.subject.camera_id}' does not match "
                f"stream camera '{self.stream.camera_id}'"
            )
        if not math.isfinite(self.joined_at_seconds) or self.joined_at_seconds < 0:
            raise IdentityError(
                f"joined_at_seconds must be a finite, non-negative venue time, got "
                f"{self.joined_at_seconds!r}"
            )

    @property
    def camera_id(self) -> str:
        return self.subject.camera_id

    @property
    def tracklet_key(self) -> TrackletKey:
        return tracklet_key(self.subject, self.stream)

    def to_record(self) -> dict[str, Any]:
        return {
            "camera_id": self.subject.camera_id,
            "track_id": self.subject.track_id,
            "stream_epoch": self.stream.stream_epoch,
            "joined_at_seconds": self.joined_at_seconds,
        }


def membership_order_key(membership: TrackletMembership) -> TrackletKey:
    """Canonical ordering for memberships, independent of insertion order."""
    return membership.tracklet_key


class MembershipAction(StrEnum):
    """What a :class:`MembershipEvent` records happening to a membership."""

    ADDED = "added"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class MembershipEvent:
    """One membership change, kept so provenance survives a removal.

    A venue subject's current members are only its *active* state; a
    tracklet that was added and later removed is still part of the identity's
    history and must remain recoverable. This is that record.
    """

    membership: TrackletMembership
    action: MembershipAction
    at_seconds: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.at_seconds) or self.at_seconds < 0:
            raise IdentityError(
                f"at_seconds must be a finite, non-negative venue time, got {self.at_seconds!r}"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "action": str(self.action),
            "at_seconds": self.at_seconds,
            **self.membership.to_record(),
        }


@dataclass(frozen=True, slots=True)
class VenueSubject:
    """A snapshot of one venue-local identity's current state.

    ``members`` is the *active* membership set — the camera-local tracklets
    currently contributing to this identity — in canonical order, never
    insertion order. Removed memberships are not lost; they live in the
    registry's provenance log for this subject, not in this snapshot.

    Invariant: at most one active member per camera. A venue subject claiming
    two simultaneous tracklets on the same camera would be claiming to be two
    places on one camera's view at once, which is never something this layer
    may assert.
    """

    subject_id: VenueSubjectId
    created_at_seconds: float
    last_activity_seconds: float
    members: tuple[TrackletMembership, ...]
    retired_at_seconds: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("created_at_seconds", self.created_at_seconds),
            ("last_activity_seconds", self.last_activity_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise IdentityError(
                    f"{name} must be a finite, non-negative venue time, got {value!r}"
                )
        if self.last_activity_seconds < self.created_at_seconds:
            raise IdentityError(
                f"{self.subject_id} has last_activity_seconds "
                f"({self.last_activity_seconds}) before created_at_seconds "
                f"({self.created_at_seconds})"
            )
        if self.retired_at_seconds is not None:
            if not math.isfinite(self.retired_at_seconds) or self.retired_at_seconds < 0:
                raise IdentityError(
                    f"retired_at_seconds must be a finite, non-negative venue time, got "
                    f"{self.retired_at_seconds!r}"
                )
            if self.retired_at_seconds < self.last_activity_seconds:
                raise IdentityError(
                    f"{self.subject_id} retired at {self.retired_at_seconds} before its last "
                    f"activity at {self.last_activity_seconds}"
                )
        ordered = tuple(sorted(self.members, key=membership_order_key))
        if ordered != self.members:
            raise IdentityError(
                f"{self.subject_id} members are not in canonical order; construct with "
                "members already sorted by tracklet_key"
            )
        keys = [m.tracklet_key for m in self.members]
        if len(set(keys)) != len(keys):
            raise IdentityError(f"{self.subject_id} has a duplicate tracklet among its members")
        cameras = [m.camera_id for m in self.members]
        if len(set(cameras)) != len(cameras):
            raise IdentityError(
                f"{self.subject_id} has more than one active tracklet on the same camera"
            )

    @property
    def is_retired(self) -> bool:
        return self.retired_at_seconds is not None

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(sorted(m.camera_id for m in self.members))

    def member_on(self, camera_id: str) -> TrackletMembership | None:
        for member in self.members:
            if member.camera_id == camera_id:
                return member
        return None

    def to_record(self) -> dict[str, Any]:
        return {
            "venue_subject_id": self.subject_id.to_record(),
            "created_at_seconds": self.created_at_seconds,
            "last_activity_seconds": self.last_activity_seconds,
            "retired_at_seconds": self.retired_at_seconds,
            "members": [m.to_record() for m in self.members],
        }

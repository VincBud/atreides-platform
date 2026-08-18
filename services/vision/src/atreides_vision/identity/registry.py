"""The bounded, session-scoped store of venue-local identities.

:class:`VenueSubjectRegistry` is the only place a :class:`VenueSubjectId` is
allocated and the only place a :class:`VenueSubject`'s membership changes. It
does not decide *which* tracklets belong together — every operation here is
explicit, performed by a caller that already believes it, never inferred from
proximity, appearance or a track id. That decision is a later milestone's to
wire in through :class:`~atreides_vision.association.models.AssociationRound`.

Every read comes back in a canonical order that does not depend on insertion,
dict or set order, so two callers who create the same subjects and memberships
in a different sequence see byte-identical output.
"""

from __future__ import annotations

from dataclasses import replace

from atreides_vision.behavior.models import SubjectRef
from atreides_vision.errors import IdentityError
from atreides_vision.identity.config import VenueIdentityConfig
from atreides_vision.identity.models import (
    MembershipAction,
    MembershipEvent,
    TrackletKey,
    TrackletMembership,
    VenueSubject,
    VenueSubjectId,
)

__all__ = ["VenueSubjectRegistry"]


class VenueSubjectRegistry:
    """Session-scoped registry of venue-local identities.

    Nothing here persists past the object's lifetime: no file, no database, no
    cross-session or cross-venue linkage. Discarding the instance discards
    every identity it ever allocated.
    """

    __slots__ = ("_claims", "_config", "_events", "_next_sequence", "_subjects")

    def __init__(self, config: VenueIdentityConfig | None = None) -> None:
        self._config = config if config is not None else VenueIdentityConfig()
        self._subjects: dict[VenueSubjectId, VenueSubject] = {}
        self._events: dict[VenueSubjectId, list[MembershipEvent]] = {}
        self._claims: dict[TrackletKey, VenueSubjectId] = {}
        self._next_sequence = 1

    def __len__(self) -> int:
        return len(self._subjects)

    def __contains__(self, subject_id: VenueSubjectId) -> bool:
        return subject_id in self._subjects

    # -- lifecycle: create / add / remove / retire -------------------------

    def create(self, membership: TrackletMembership) -> VenueSubjectId:
        """Allocate a new venue subject with one initial tracklet.

        A venue subject cannot represent nobody, so creation always requires
        at least one contributing tracklet. Refuses if that tracklet is
        already claimed by another subject, or if the registry is at
        capacity — see :attr:`VenueIdentityConfig.max_venue_subjects`.
        """
        key = membership.tracklet_key
        if key in self._claims:
            raise IdentityError(
                f"{membership.subject} is already a member of {self._claims[key]}; "
                "remove it there before creating a new subject with it."
            )
        if len(self._subjects) >= self._config.max_venue_subjects:
            raise IdentityError(
                f"Venue subject registry is at capacity ({self._config.max_venue_subjects}); "
                "call expire() to release retired subjects before creating another."
            )
        subject_id = VenueSubjectId(self._next_sequence)
        self._next_sequence += 1
        subject = VenueSubject(
            subject_id=subject_id,
            created_at_seconds=membership.joined_at_seconds,
            last_activity_seconds=membership.joined_at_seconds,
            members=(membership,),
        )
        self._subjects[subject_id] = subject
        self._claims[key] = subject_id
        self._events[subject_id] = [
            MembershipEvent(membership, MembershipAction.ADDED, membership.joined_at_seconds)
        ]
        return subject_id

    def add_membership(self, subject_id: VenueSubjectId, membership: TrackletMembership) -> None:
        """Add a camera-local tracklet to an existing, active venue subject.

        Refuses: an unknown or retired subject; a tracklet already claimed by
        this or any other subject; a second active tracklet on a camera this
        subject already has one on; and venue time moving backwards relative
        to the subject's last activity.
        """
        subject = self._require_active(subject_id)
        key = membership.tracklet_key
        if key in self._claims:
            owner = self._claims[key]
            if owner == subject_id:
                raise IdentityError(f"{membership.subject} is already a member of {subject_id}")
            raise IdentityError(
                f"{membership.subject} is already a member of {owner}; "
                "remove it there before adding it here."
            )
        if membership.joined_at_seconds < subject.last_activity_seconds:
            raise IdentityError(
                f"Membership for {subject_id} went backwards: "
                f"{membership.joined_at_seconds} is before last activity "
                f"{subject.last_activity_seconds}"
            )
        if subject.member_on(membership.camera_id) is not None:
            raise IdentityError(
                f"{subject_id} already has an active tracklet on camera "
                f"'{membership.camera_id}'; remove it before adding another."
            )
        members = tuple(sorted((*subject.members, membership), key=lambda m: m.tracklet_key))
        updated = replace(
            subject, members=members, last_activity_seconds=membership.joined_at_seconds
        )
        self._subjects[subject_id] = updated
        self._claims[key] = subject_id
        self._events[subject_id].append(
            MembershipEvent(membership, MembershipAction.ADDED, membership.joined_at_seconds)
        )

    def remove_membership(
        self, subject_id: VenueSubjectId, subject_ref: SubjectRef, at_seconds: float
    ) -> None:
        """Remove a camera-local tracklet from an active venue subject.

        Removal does not destroy provenance: the membership remains
        recoverable through :meth:`provenance_for`. It also does not retire
        the subject, even if no members remain — retirement is a separate,
        explicit operation.
        """
        subject = self._require_active(subject_id)
        match = next((m for m in subject.members if m.subject == subject_ref), None)
        if match is None:
            raise IdentityError(f"{subject_ref} is not an active member of {subject_id}")
        if at_seconds < subject.last_activity_seconds:
            raise IdentityError(
                f"Removal for {subject_id} went backwards: {at_seconds} is before last "
                f"activity {subject.last_activity_seconds}"
            )
        members = tuple(m for m in subject.members if m.subject != subject_ref)
        updated = replace(subject, members=members, last_activity_seconds=at_seconds)
        self._subjects[subject_id] = updated
        del self._claims[match.tracklet_key]
        self._events[subject_id].append(
            MembershipEvent(match, MembershipAction.REMOVED, at_seconds)
        )

    def retire(self, subject_id: VenueSubjectId, at_seconds: float) -> None:
        """Mark a venue subject as no longer current.

        A retired subject accepts no further membership changes. Its
        tracklet claims are released, so a tracklet it held may be claimed by
        a new venue subject afterward — retirement ends the identity, not the
        tracklet.
        """
        subject = self._require_active(subject_id)
        if at_seconds < subject.last_activity_seconds:
            raise IdentityError(
                f"Retirement for {subject_id} went backwards: {at_seconds} is before last "
                f"activity {subject.last_activity_seconds}"
            )
        updated = replace(subject, retired_at_seconds=at_seconds)
        self._subjects[subject_id] = updated
        for member in subject.members:
            self._claims.pop(member.tracklet_key, None)

    def expire(self, now_seconds: float) -> tuple[VenueSubjectId, ...]:
        """Advance the registry to ``now_seconds`` of venue time.

        Two deterministic passes, in canonical subject order:

        1. Any active subject idle longer than
           :attr:`VenueIdentityConfig.idle_expiry_seconds` is retired, timestamped
           at its own last activity rather than at ``now_seconds``.
        2. Any subject retired longer than
           :attr:`VenueIdentityConfig.retention_seconds` is purged entirely,
           freeing registry capacity. Purged subjects are no longer queryable.

        Returns the ids purged in this call, in canonical order. Based only
        on venue time — never wall-clock time.
        """
        for subject_id in sorted(self._subjects):
            subject = self._subjects[subject_id]
            if subject.is_retired:
                continue
            idle_for = now_seconds - subject.last_activity_seconds
            if idle_for > self._config.idle_expiry_seconds:
                self.retire(subject_id, subject.last_activity_seconds)

        purged = tuple(
            subject_id
            for subject_id in sorted(self._subjects)
            if self._subjects[subject_id].is_retired
            and now_seconds - (self._subjects[subject_id].retired_at_seconds or 0.0)
            > self._config.retention_seconds
        )
        for subject_id in purged:
            del self._subjects[subject_id]
            del self._events[subject_id]
        return purged

    # -- reads ---------------------------------------------------------

    def get(self, subject_id: VenueSubjectId) -> VenueSubject:
        subject = self._subjects.get(subject_id)
        if subject is None:
            raise IdentityError(f"{subject_id} is not registered")
        return subject

    def subjects(self) -> tuple[VenueSubjectId, ...]:
        """Every registered subject, active or retired, in canonical order."""
        return tuple(sorted(self._subjects))

    def active_subjects(self) -> tuple[VenueSubjectId, ...]:
        return tuple(sid for sid in sorted(self._subjects) if not self._subjects[sid].is_retired)

    def owner_of(self, key: TrackletKey) -> VenueSubjectId | None:
        """The venue subject currently, actively claiming this tracklet, if any."""
        return self._claims.get(key)

    def provenance_for(self, subject_id: VenueSubjectId) -> tuple[MembershipEvent, ...]:
        """Every membership change ever recorded for this subject, in order.

        Chronological by construction: every event is appended under a
        venue-time non-decreasing check, so replaying this tuple in order
        never has to be re-sorted to be meaningful.
        """
        events = self._events.get(subject_id)
        if events is None:
            raise IdentityError(f"{subject_id} is not registered")
        return tuple(events)

    def _require_active(self, subject_id: VenueSubjectId) -> VenueSubject:
        subject = self._subjects.get(subject_id)
        if subject is None:
            raise IdentityError(f"{subject_id} is not registered")
        if subject.is_retired:
            raise IdentityError(
                f"{subject_id} was retired at venue second {subject.retired_at_seconds} and "
                "accepts no further membership changes"
            )
        return subject

"""Domain types for one venue instant's current occupancy — V0.4.6.

An :class:`OccupancySnapshot` is deliberately narrow: it describes exactly
one venue instant's currently observed camera-local tracklets, grouped by
whatever venue-local identity association and fusion have already,
confidently, established for it. It is not a count of everything the
:class:`~atreides_vision.identity.registry.VenueSubjectRegistry` currently
holds — a subject whose tracklets have stopped appearing contributes nothing
here, even though the registry keeps its record for provenance until it is
retired and, later, purged.

Every field name states exactly what it counts. There is no field here that
could be mistaken for cumulative footfall, a unique-visitor total or a
persistent headcount: this is a per-instant, venue-session-scoped snapshot,
discarded the moment the next one is computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atreides_vision.behavior.models import SubjectRef
from atreides_vision.errors import OccupancyError
from atreides_vision.identity.models import VenueSubjectId

__all__ = ["OccupancyContribution", "OccupancySnapshot"]


@dataclass(frozen=True, slots=True)
class OccupancyContribution:
    """One venue subject's currently observed camera-local tracklets.

    ``members`` lists only tracklets present in the venue instant this
    contribution was built from — never every tracklet the subject's
    provenance has ever held. A subject whose second camera has gone quiet
    drops that tracklet from ``members`` the very next instant, even though
    :meth:`~atreides_vision.identity.registry.VenueSubjectRegistry.get` still
    remembers it as historical membership.

    A contribution with exactly one member is a fully legitimate,
    zero-suppression occupancy unit: a venue subject currently represented by
    only one camera contributes one occupancy count and suppresses nothing,
    exactly as an unassociated tracklet does — the two differ only in
    provenance, never in how they are counted.
    """

    subject_id: VenueSubjectId
    members: tuple[SubjectRef, ...]

    def __post_init__(self) -> None:
        if not self.members:
            raise OccupancyError(
                f"{self.subject_id} contributes to occupancy with zero currently observed "
                "tracklets; a contribution must be backed by at least one observation, or it "
                "should not exist for this venue instant."
            )
        if len(set(self.members)) != len(self.members):
            raise OccupancyError(
                f"{self.subject_id} lists the same camera-local tracklet more than once "
                f"among its currently observed members: {self.members}"
            )

    @property
    def suppressed_count(self) -> int:
        """How many of this instant's currently observed tracklets this subject suppressed.

        Zero for a subject represented by exactly one currently observed
        tracklet; ``N - 1`` for one represented by ``N``.
        """
        return len(self.members) - 1

    def to_record(self) -> dict[str, Any]:
        return {
            "venue_subject_id": self.subject_id.to_record(),
            "members": [str(member) for member in self.members],
            "suppressed_count": self.suppressed_count,
        }


@dataclass(frozen=True, slots=True)
class OccupancySnapshot:
    """How many people one venue instant's observations currently represent.

    This is *current* occupancy, not cumulative footfall and not a persistent
    headcount: it describes exactly one venue instant, built from exactly the
    camera-local tracklets observed at it. Two snapshots for two different
    instants are unrelated values — nothing here accumulates across them.

    Arithmetic invariant, enforced at construction::

        raw_active_track_count == occupancy_count + suppressed_duplicate_count

    where ``occupancy_count`` is the number of independently counted units —
    one per :class:`OccupancyContribution` plus one per unresolved
    tracklet — and ``suppressed_duplicate_count`` is the sum of every
    contribution's :attr:`~OccupancyContribution.suppressed_count`.
    """

    venue_seconds: float
    #: Every camera-local tracklet observed at this venue instant, whether or
    #: not it currently belongs to a fused venue subject. The un-deduplicated
    #: count this snapshot's suppression is measured against.
    raw_active_track_count: int
    #: One entry per venue subject currently represented by two or more, or
    #: exactly one, currently observed tracklets — in canonical
    #: (:class:`~atreides_vision.identity.models.VenueSubjectId` sequence)
    #: order.
    contributing_subjects: tuple[OccupancyContribution, ...]
    #: Camera-local tracklets observed at this instant that no venue subject
    #: currently owns — AMBIGUOUS, UNASSOCIATED, NOT_COMPARABLE or REFUSED,
    #: or simply not yet evaluated. Each counts independently, in canonical
    #: observation order.
    unresolved_tracklets: tuple[SubjectRef, ...]

    def __post_init__(self) -> None:
        if self.raw_active_track_count < 0:
            raise OccupancyError(
                f"raw_active_track_count must be non-negative, got {self.raw_active_track_count!r}"
            )
        subject_ids = [contribution.subject_id for contribution in self.contributing_subjects]
        if subject_ids != sorted(subject_ids):
            raise OccupancyError(
                "contributing_subjects must be in canonical VenueSubjectId order, got "
                f"{[str(s) for s in subject_ids]}"
            )
        if len(set(subject_ids)) != len(subject_ids):
            raise OccupancyError(
                "the same venue subject contributes more than once: "
                f"{[str(s) for s in subject_ids]}"
            )
        claimed: set[SubjectRef] = set()
        for contribution in self.contributing_subjects:
            for member in contribution.members:
                if member in claimed:
                    raise OccupancyError(
                        f"{member} is claimed by more than one contributing venue subject"
                    )
                claimed.add(member)
        for tracklet in self.unresolved_tracklets:
            if tracklet in claimed:
                raise OccupancyError(
                    f"{tracklet} is both an unresolved tracklet and a contributing subject's "
                    "member; a tracklet must be counted exactly once"
                )
            claimed.add(tracklet)
        if len(claimed) != self.raw_active_track_count:
            raise OccupancyError(
                f"raw_active_track_count is {self.raw_active_track_count} but "
                f"{len(claimed)} distinct currently observed tracklet(s) were accounted for; "
                "every observed tracklet must be counted exactly once, as a contribution member "
                "or as unresolved."
            )

    @property
    def occupancy_count(self) -> int:
        """The deduplicated count this snapshot exists to produce."""
        return len(self.contributing_subjects) + len(self.unresolved_tracklets)

    @property
    def suppressed_duplicate_count(self) -> int:
        """How many currently observed tracklets were suppressed as duplicates."""
        return self.raw_active_track_count - self.occupancy_count

    @property
    def active_venue_subject_count(self) -> int:
        """How many distinct venue subjects currently contribute to occupancy."""
        return len(self.contributing_subjects)

    def to_record(self) -> dict[str, Any]:
        return {
            "venue_seconds": self.venue_seconds,
            "raw_active_track_count": self.raw_active_track_count,
            "occupancy_count": self.occupancy_count,
            "suppressed_duplicate_count": self.suppressed_duplicate_count,
            "active_venue_subject_count": self.active_venue_subject_count,
            "contributing_subjects": [c.to_record() for c in self.contributing_subjects],
            "unresolved_tracklets": [str(t) for t in self.unresolved_tracklets],
        }

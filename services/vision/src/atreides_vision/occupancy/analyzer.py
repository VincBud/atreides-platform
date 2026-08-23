"""Computing one venue instant's current occupancy — V0.4.6.

The narrowest safe input already available: one already-produced
:class:`~atreides_vision.venue_pipeline.VenueInstantResult` and the
:class:`~atreides_vision.identity.registry.VenueSubjectRegistry` it was
produced against. Nothing here re-runs association or fusion, and nothing
here reads :mod:`atreides_vision.venue_pipeline_reporting` — reporting is an
observer of runtime truth, never a source of it.
"""

from __future__ import annotations

from atreides_vision.association.models import TrackObservation, observation_order_key
from atreides_vision.identity.models import VenueSubjectId, tracklet_key
from atreides_vision.identity.registry import VenueSubjectRegistry
from atreides_vision.occupancy.models import OccupancyContribution, OccupancySnapshot
from atreides_vision.venue_pipeline import VenueInstantResult

__all__ = ["compute_occupancy"]


def compute_occupancy(
    result: VenueInstantResult, registry: VenueSubjectRegistry
) -> OccupancySnapshot:
    """Deduplicate one venue instant's observations by current venue identity.

    ``registry`` must reflect the state immediately after ``result`` was
    produced — the ordinary case of calling this right after
    :meth:`~atreides_vision.venue_pipeline.VenuePipeline.process_instant`,
    before any later instant or
    :meth:`~atreides_vision.venue_pipeline.VenuePipeline.expire` call has had
    a chance to retire a subject this instant's observations still name.
    This function performs no registry mutation of its own and calls only
    :meth:`~atreides_vision.identity.registry.VenueSubjectRegistry.owner_of`
    — the registry's own read of *live* ownership, which already excludes a
    tracklet whose subject has been retired or whose membership has been
    removed. No other registry state (retained provenance, historical
    members no longer observed) reaches this function at all.

    Grouping is by :meth:`~atreides_vision.identity.registry.VenueSubjectRegistry.owner_of`
    alone: a tracklet with no current owner counts independently, whatever
    the reason (AMBIGUOUS, UNASSOCIATED, NOT_COMPARABLE, a REFUSED fusion
    outcome, or simply not yet evaluated). Two tracklets sharing one live
    owner count once. This function never re-derives, second-guesses or
    overrides an association or fusion decision — it only reads where each
    currently observed tracklet's ownership already, independently, landed.
    """
    groups: dict[VenueSubjectId, list[TrackObservation]] = {}
    unresolved: list[TrackObservation] = []
    for observation in result.observations:
        owner = registry.owner_of(tracklet_key(observation.subject, observation.stream))
        if owner is None:
            unresolved.append(observation)
        else:
            groups.setdefault(owner, []).append(observation)

    contributing_subjects = tuple(
        OccupancyContribution(
            subject_id=subject_id,
            members=tuple(
                observation.subject
                for observation in sorted(groups[subject_id], key=observation_order_key)
            ),
        )
        for subject_id in sorted(groups)
    )
    unresolved_tracklets = tuple(
        observation.subject for observation in sorted(unresolved, key=observation_order_key)
    )
    return OccupancySnapshot(
        venue_seconds=result.venue_seconds,
        raw_active_track_count=len(result.observations),
        contributing_subjects=contributing_subjects,
        unresolved_tracklets=unresolved_tracklets,
    )

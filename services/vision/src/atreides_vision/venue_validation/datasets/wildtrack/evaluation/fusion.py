"""Per-instant fusion / venue-subject purity scoring for the WILDTRACK evaluator.

Consumes only the frozen snapshot and WILDTRACK ground truth. Purity is measured
both directions:

* **impurity** — a venue subject whose GT-resolved members belong to more than
  one WILDTRACK person (an over-merge);
* **fragmentation** — a WILDTRACK person whose owned, matched tracklets are split
  across more than one venue subject (an under-merge);
* **matched-unowned** — a GT-matched tracklet that no venue subject owns.

Nothing here writes back into runtime state and ``WildtrackPersonId`` never
leaves the evaluator.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from atreides_vision.association.models import TrackletKey
from atreides_vision.identity.models import VenueSubjectId
from atreides_vision.venue_validation.datasets.wildtrack.annotations import (
    WildtrackFrameAnnotations,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.models import (
    CameraMatch,
    InstantFusionEvaluation,
)
from atreides_vision.venue_validation.observer import VenueValidationInstantSnapshot

__all__ = ["score_instant_fusion"]


def score_instant_fusion(
    snapshot: VenueValidationInstantSnapshot,
    frame: WildtrackFrameAnnotations,
    camera_matches: Mapping[str, CameraMatch],
    *,
    selected_camera_ids: Sequence[str],
) -> InstantFusionEvaluation:
    """Score one venue instant's fusion output against WILDTRACK identity truth."""
    owner: dict[TrackletKey, VenueSubjectId] = {}
    for membership in snapshot.memberships:
        for member in membership.members:
            owner[member] = membership.subject_id

    person_by_tracklet: dict[TrackletKey, int] = {}
    for camera_id in selected_camera_ids:
        match = camera_matches[camera_id]
        for row, column in match.pairs:
            person_by_tracklet[match.observation_keys[row]] = match.person_ids[column]

    matched_tracklet_count = len(person_by_tracklet)
    unowned_set = set(snapshot.unowned_tracklets)
    matched_unowned_count = sum(1 for key in person_by_tracklet if key in unowned_set)
    #: GT person instances with >=1 locally matched runtime tracklet that no venue
    #: subject owns at this instant, deduplicated by WILDTRACK personID.
    matched_unowned_person_count = len(
        {person_by_tracklet[key] for key in person_by_tracklet if key in unowned_set}
    )

    evaluable_subject_count = 0
    impure_subject_count = 0
    for membership in snapshot.memberships:
        persons = {
            person_by_tracklet[member]
            for member in membership.members
            if member in person_by_tracklet
        }
        if not persons:
            continue
        evaluable_subject_count += 1
        if len(persons) > 1:
            impure_subject_count += 1

    persons_with_owned_match = 0
    fragmented_person_count = 0
    for person in frame.persons:
        person_id = person.person_id.value
        owners: set[VenueSubjectId] = set()
        has_matched = False
        for camera_id in selected_camera_ids:
            key = camera_matches[camera_id].observation_for_person(person_id)
            if key is None:
                continue
            has_matched = True
            owner_id = owner.get(key)
            if owner_id is not None:
                owners.add(owner_id)
        if has_matched and owners:
            persons_with_owned_match += 1
            if len(owners) > 1:
                fragmented_person_count += 1

    outcome_counts: Counter[str] = Counter()
    for outcome in snapshot.integration.outcomes:
        outcome_counts[outcome.outcome.value] += 1

    return InstantFusionEvaluation(
        evaluable_subject_count=evaluable_subject_count,
        impure_subject_count=impure_subject_count,
        persons_with_owned_match=persons_with_owned_match,
        fragmented_person_count=fragmented_person_count,
        matched_tracklet_count=matched_tracklet_count,
        matched_unowned_count=matched_unowned_count,
        matched_unowned_person_count=matched_unowned_person_count,
        integration_outcome_counts=dict(outcome_counts),
    )

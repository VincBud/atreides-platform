"""V0.4.9b-C — a read-only validation observer seam at the venue instant.

The direction is strictly one-way::

    runtime  ->  immutable prediction snapshot  ->  optional validation observer

The runtime produces one venue instant; this module freezes exactly what it
produced into an immutable snapshot; an optional observer callback inspects
that snapshot and returns nothing. The observer cannot mutate the
:class:`~atreides_vision.venue_pipeline.VenuePipeline`, the association engine,
the :class:`~atreides_vision.identity.registry.VenueSubjectRegistry`, fusion or
occupancy; it cannot supply observations, replace runtime state, or influence
the next instant. It observes facts already established, and that is all.

Generic by construction. No dataset type and no annotation semantics appears
here — this module sees runtime *predictions* only. A later offline evaluator
(V0.4.9b-D) is free to consume these snapshots alongside whatever reference it
has; the observer itself never does, and imports no dataset package.

Determinism. Every field is an already-canonical immutable domain value, and
this module imposes no ordering of its own: membership, unowned tracklets and
observations are all handed on in the exact order the producing layer
(:class:`~atreides_vision.occupancy.models.OccupancySnapshot`,
:class:`~atreides_vision.association.models.AssociationRound`) already
guarantees, which is independent of manifest camera order. The only
transformation is resolving each camera-local :class:`SubjectRef` from the
occupancy snapshot to its canonical
:data:`~atreides_vision.association.models.TrackletKey` using *this instant's*
observations — a per-element rename that preserves source order. Nothing here
reads a wall clock, a PID, a hostname, an absolute path or a random source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from atreides_vision.association.models import (
    AssociationRound,
    TrackletKey,
    TrackObservation,
    tracklet_key,
)
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.errors import ValidationError
from atreides_vision.fusion.models import IntegrationResult
from atreides_vision.identity.models import VenueSubjectId
from atreides_vision.occupancy.models import OccupancySnapshot
from atreides_vision.venue.models import stream_sort_key
from atreides_vision.venue_pipeline import VenueInstantResult
from atreides_vision.venue_validation.stage_observation import CameraStageSnapshot

__all__ = [
    "VenueSubjectMembershipSnapshot",
    "VenueValidationInstantSnapshot",
    "VenueValidationObserver",
]


@dataclass(frozen=True, slots=True)
class VenueSubjectMembershipSnapshot:
    """Which currently observed camera-local tracklets one venue subject owns
    at one venue instant.

    Instant-local and non-persistent. ``members`` are only the tracklets
    observed at *this* venue instant whose live registry owner is
    ``subject_id``. It is never the subject's full historical membership: a
    tracklet the registry still retains for provenance but that was not
    observed this instant does not appear here. This mirrors
    :class:`~atreides_vision.occupancy.models.OccupancyContribution`, from
    which it is derived — it is not a second identity algorithm.

    Each member is a :data:`~atreides_vision.association.models.TrackletKey`
    — ``(camera_id, stream_epoch, track_id)`` — so a tracklet observed before
    a camera stream restart is never conflated with one that reuses the same
    track integer afterwards.
    """

    subject_id: VenueSubjectId
    #: Canonical order inherited verbatim from the immutable
    #: :class:`~atreides_vision.occupancy.models.OccupancyContribution` this
    #: snapshot is derived from; no ordering is imposed here.
    members: tuple[TrackletKey, ...]


@dataclass(frozen=True, slots=True)
class VenueValidationInstantSnapshot:
    """One immutable, factual record of what the runtime produced for one
    venue instant, for a later offline evaluator to inspect.

    Every component belongs to the same venue instant — enforced in
    :meth:`__post_init__`, so a directly constructed snapshot cannot stitch
    an association round, an integration result and an occupancy snapshot
    from different moments together. Nothing here is a score, a grade or an
    accuracy figure; it is the runtime's own prediction, frozen.
    """

    #: The session-relative venue instant, passed through verbatim.
    venue_seconds: float
    #: Every camera-local observation the engine considered this instant, in
    #: the canonical order :class:`AssociationRound` already guarantees.
    observations: tuple[TrackObservation, ...]
    #: V0.4.2's cross-camera decision for every candidate pair this instant.
    association_round: AssociationRound
    #: V0.4.3b's account of what each positive decision did to venue identity.
    integration: IntegrationResult
    #: Current venue-subject membership: which currently observed tracklets
    #: each venue subject owns right now, in the canonical ``VenueSubjectId``
    #: order the occupancy snapshot already guarantees. Retained-but-unobserved
    #: historical members are excluded.
    memberships: tuple[VenueSubjectMembershipSnapshot, ...]
    #: Currently observed tracklets that no venue subject owns this instant
    #: (AMBIGUOUS, UNASSOCIATED, NOT_COMPARABLE, a REFUSED fusion outcome, or
    #: simply not yet evaluated). Represented explicitly rather than dropped,
    #: in the canonical order the occupancy snapshot already guarantees.
    unowned_tracklets: tuple[TrackletKey, ...]
    #: V0.4.6's current-occupancy snapshot for this instant, reused whole.
    occupancy: OccupancySnapshot
    #: V0.4.10a's camera-local perception/tracking stage facts for every
    #: contributing stream this instant, in canonical ``(camera_id,
    #: stream_epoch)`` order. Empty unless a caller opted into observation —
    #: never populated as a side effect of anything else this snapshot holds.
    camera_stages: tuple[CameraStageSnapshot, ...] = ()

    def __post_init__(self) -> None:
        seconds = self.venue_seconds
        if not (
            seconds
            == self.association_round.venue_seconds
            == self.integration.venue_seconds
            == self.occupancy.venue_seconds
        ):
            raise ValidationError(
                "A validation instant snapshot must describe one venue instant: got "
                f"venue_seconds={seconds}, round={self.association_round.venue_seconds}, "
                f"integration={self.integration.venue_seconds}, "
                f"occupancy={self.occupancy.venue_seconds}."
            )
        for stage in self.camera_stages:
            if stage.venue_seconds != seconds:
                raise ValidationError(
                    "A validation instant snapshot must describe one venue instant: camera "
                    f"stage for stream {stage.stream} has venue_seconds={stage.venue_seconds}, "
                    f"expected {seconds}."
                )
        streams = [stage.stream for stage in self.camera_stages]
        if len(set(streams)) != len(streams):
            raise ValidationError(
                "A validation instant snapshot cannot hold two camera stages for the same "
                f"stream: {streams}."
            )
        canonical = sorted(streams, key=stream_sort_key)
        if streams != canonical:
            raise ValidationError(
                "camera_stages must be in canonical (camera_id, stream_epoch) order: got "
                f"{streams}, expected {canonical}."
            )


class VenueValidationObserver(Protocol):
    """A read-only sink for :class:`VenueValidationInstantSnapshot`.

    Called exactly once per venue instant, after association, fusion and
    occupancy have all completed for that instant. The return value is
    ignored — an observer that returns something cannot thereby change
    anything. If it raises, the validation run fails (see
    :func:`~atreides_vision.venue_validation.runner.run_venue_validation`):
    a partial or misleading validation output is worse than a failed one.
    """

    def __call__(self, snapshot: VenueValidationInstantSnapshot, /) -> None: ...


def build_instant_snapshot(
    result: VenueInstantResult,
    occupancy: OccupancySnapshot,
    camera_stages: tuple[CameraStageSnapshot, ...] = (),
) -> VenueValidationInstantSnapshot:
    """Freeze one venue instant's runtime output into an immutable snapshot.

    Runner machinery, not part of the package's public API. ``occupancy`` must
    be the snapshot
    :func:`~atreides_vision.occupancy.analyzer.compute_occupancy` produced
    from ``result`` and the registry state immediately after
    :meth:`~atreides_vision.venue_pipeline.VenuePipeline.process_instant` —
    the ordinary runner call sequence. The current-membership view is taken
    from that occupancy snapshot, which has already intersected registry
    membership with this instant's observations via the registry's own live
    ``owner_of`` read; this function does not touch the registry and does not
    re-derive identity.

    The occupancy snapshot names its members and unresolved tracklets by
    :class:`SubjectRef` — ``(camera_id, track_id)`` only. Each is resolved
    here to the canonical
    :data:`~atreides_vision.association.models.TrackletKey` of the matching
    observation from *this same instant*, restoring ``stream_epoch`` without
    reading any historical registry state or manufacturing a value.

    Raises:
        ValidationError: if the components do not all describe the same venue
            instant, or if an occupancy :class:`SubjectRef` does not resolve
            to exactly one of this instant's observations.
    """
    keys_by_subject: dict[SubjectRef, list[TrackletKey]] = {}
    for observation in result.observations:
        keys_by_subject.setdefault(observation.subject, []).append(tracklet_key(observation))

    def resolve(ref: SubjectRef) -> TrackletKey:
        candidates = keys_by_subject.get(ref, [])
        if len(candidates) != 1:
            raise ValidationError(
                f"Occupancy names camera-local tracklet {ref}, but this venue instant has "
                f"{len(candidates)} matching observation(s); it must resolve to exactly one "
                "current TrackletKey."
            )
        return candidates[0]

    memberships = tuple(
        VenueSubjectMembershipSnapshot(
            subject_id=contribution.subject_id,
            members=tuple(resolve(member) for member in contribution.members),
        )
        for contribution in occupancy.contributing_subjects
    )
    return VenueValidationInstantSnapshot(
        venue_seconds=result.venue_seconds,
        observations=result.observations,
        association_round=result.round,
        integration=result.integration,
        memberships=memberships,
        unowned_tracklets=tuple(resolve(ref) for ref in occupancy.unresolved_tracklets),
        occupancy=occupancy,
        camera_stages=camera_stages,
    )

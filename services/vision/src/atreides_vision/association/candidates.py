"""Candidate enumeration and what geometry alone establishes about them.

Three pure functions, no state, no history, nothing retained between calls:

* :func:`candidate_pairs` — the deterministic total order over candidates.
* :func:`assess_pair` — one pair through the hard gates, then the geometry.
* :func:`gate_observation` — one observation against a pool, with ambiguity
  assessed per counterpart camera.

Determinism
-----------
Output order is fixed by ``(camera_id, stream_epoch, track_id, frame_index)``,
never by the order observations arrive in. Cameras supplied A, B, C and C, A, B
produce identical candidates, identical assessments and identical records. This
matters more here than anywhere else in Atreides: an association that depends on
camera enumeration order is an association nobody can reproduce or falsify.

Ambiguity is per counterpart camera
-----------------------------------
Two candidates on *different* cameras are not rivals — a person standing in
three cameras' views produces three mutually consistent observations, and
calling that ambiguous would penalise good coverage. Two candidates on the *same*
counterpart camera are rivals: at one instant, one person is one track on one
camera. So ambiguity is assessed within each counterpart camera and reported as
the list of cameras where geometry cannot choose. Nothing here knows or assumes
how many cameras exist.

What this module does not do
----------------------------
It never merges, never assigns an identity, never counts people and never
returns :attr:`~atreides_vision.association.models.AssociationOutcome.ASSOCIATED`.
Surviving a gate is not evidence of anything; it only means geometry had no
grounds to refuse. Deciding is the next layer's job, on evidence this one does
not have.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations

from atreides_vision.association.config import AssociationGatingConfig
from atreides_vision.association.gates import comparability_rejections
from atreides_vision.association.geometry import (
    latest_possible_gap_seconds,
    motion_allowance_m,
    nominal_gap_seconds,
    separation_metres,
    timing_slack_seconds,
)
from atreides_vision.association.models import (
    AssociationOutcome,
    CandidatePair,
    GeometricEvidence,
    ObservationGating,
    PairAssessment,
    TrackObservation,
    observation_order_key,
)

__all__ = ["assess_pair", "candidate_pairs", "gate_observation"]


def candidate_pairs(observations: Iterable[TrackObservation]) -> tuple[CandidatePair, ...]:
    """Every unordered pair of distinct observations, in one canonical order.

    Each pair appears exactly once, ordered by
    ``(first, second)`` under :func:`observation_order_key`. Same-camera pairs
    are **included** and later rejected by a gate rather than filtered out here:
    a candidate that was refused, with the reason it was refused, is the record
    that makes "no association" auditable.

    A pool from one camera, or a pool of one observation, yields no pairs. That
    is the ordinary single-camera case, not an error.
    """
    ordered = sorted(observations, key=observation_order_key)
    return tuple(CandidatePair(first, second) for first, second in combinations(ordered, 2))


def assess_pair(pair: CandidatePair, config: AssociationGatingConfig) -> PairAssessment:
    """Run the hard gates, then measure the geometry if they all passed.

    A rejected pair carries no evidence: a distance computed across a closed
    gate would be read later as though it meant something.
    """
    rejections = comparability_rejections(pair, config)
    if rejections:
        return PairAssessment(pair=pair, rejections=rejections)

    slack = timing_slack_seconds(pair.first.timing, pair.second.timing)
    # Unreachable once the gates pass: VENUE_TIME_NOT_ESTABLISHED is exactly the
    # condition under which slack is None. Asserted rather than assumed because
    # the alternative is a fabricated zero.
    if slack is None:  # pragma: no cover - guarded by the timing gate above
        raise AssertionError("a comparable pair must have established timing on both streams")

    nominal = nominal_gap_seconds(pair.first, pair.second)
    latest_gap = latest_possible_gap_seconds(nominal, slack)
    evidence = GeometricEvidence(
        separation_m=separation_metres(pair.first, pair.second),
        position_tolerance_m=config.position_tolerance_m,
        motion_allowance_m=motion_allowance_m(latest_gap, config),
        nominal_gap_seconds=nominal,
        timing_slack_seconds=slack,
        trust=pair.first.spatial.trust.worst_of(pair.second.spatial.trust),
    )
    return PairAssessment(pair=pair, evidence=evidence)


def gate_observation(
    observation: TrackObservation,
    pool: Sequence[TrackObservation],
    config: AssociationGatingConfig,
) -> ObservationGating:
    """Everything geometry establishes about one observation against ``pool``.

    ``pool`` is whatever the caller considers concurrent — typically the other
    observations at one venue instant. The observation itself may appear in it
    and is skipped; nothing else is filtered out in advance, because a pair
    rejected by a named gate is more useful than a pair silently never formed.

    The outcome is:

    ``NOT_COMPARABLE``
        Nothing in the pool cleared the hard gates — including the ordinary case
        of an empty pool, which is what a single-camera venue produces for every
        observation, every time. Not an error and not a special mode: there is
        simply no cross-camera evidence to have.
    ``UNASSOCIATED``
        Something was comparable and geometry excluded all of it.
    ``AMBIGUOUS``
        Some counterpart camera offers candidates geometry cannot separate.
        The answer is no merge, and it is recorded as unresolved rather than
        resolved by picking the nearer one.
    ``None``
        Geometry has no verdict. Candidates survived and are carried forward,
        nearest first, for a layer with stronger evidence to decide on.
        ``ASSOCIATED`` is never produced here.
    """
    assessments = tuple(
        assess_pair(CandidatePair(observation, other), config)
        for other in sorted(pool, key=observation_order_key)
        if other.subject != observation.subject
    )

    ranked: list[tuple[float, tuple[str, int, int, int], PairAssessment]] = []
    for assessment in assessments:
        evidence = assessment.evidence
        if evidence is None or not evidence.is_admissible:
            continue
        counterpart = assessment.pair.other_than(observation.subject)
        ranked.append((evidence.separation_m, observation_order_key(counterpart), assessment))
    ranked.sort(key=lambda entry: (entry[0], entry[1]))

    candidates = tuple(assessment for _, _, assessment in ranked)
    ambiguous_cameras = _ambiguous_cameras(ranked, config)

    return ObservationGating(
        observation=observation,
        assessments=assessments,
        candidates=candidates,
        ambiguous_cameras=ambiguous_cameras,
        geometric_outcome=_outcome(assessments, candidates, ambiguous_cameras),
    )


def _ambiguous_cameras(
    ranked: Sequence[tuple[float, tuple[str, int, int, int], PairAssessment]],
    config: AssociationGatingConfig,
) -> tuple[str, ...]:
    """Counterpart cameras whose two nearest candidates geometry cannot separate.

    Compares the two nearest per camera: anything further is further still from
    the best, so the pair that decides ambiguity is always the closest two.
    """
    by_camera: dict[str, list[float]] = {}
    for separation, key, _ in ranked:
        by_camera.setdefault(key[0], []).append(separation)
    return tuple(
        camera
        for camera in sorted(by_camera)
        if len(by_camera[camera]) >= 2
        and by_camera[camera][1] - by_camera[camera][0] < config.ambiguity_margin_m
    )


def _outcome(
    assessments: Sequence[PairAssessment],
    candidates: Sequence[PairAssessment],
    ambiguous_cameras: Sequence[str],
) -> AssociationOutcome | None:
    if not any(assessment.is_comparable for assessment in assessments):
        return AssociationOutcome.NOT_COMPARABLE
    if not candidates:
        return AssociationOutcome.UNASSOCIATED
    if ambiguous_cameras:
        return AssociationOutcome.AMBIGUOUS
    return None

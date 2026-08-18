"""Who prefers whom, and when a group of preferences contradicts itself.

Pure functions over one venue instant's gating results. Nothing here keeps
state, reads a clock, or knows how many cameras exist.

Two rules, in order
-------------------
**Mutual preference.** An association needs both sides to agree. Each
observation names, per counterpart camera, the nearest candidate geometry
admitted; a pair is agreed only when each is the other's nearest on the other's
camera and the runner-up is far enough behind that preferring the leader is not
a coin flip. Because the nearest candidate on a camera is unique, this is
exclusive by construction: at one instant one observation cannot positively
agree with two tracklets on the same camera.

**Pairwise consistency.** Agreement is pairwise, but its consequences are not:
A agreeing with B and B with C would join A and C without anybody having
compared them. So the groups formed by agreement are checked against the
evidence that already exists — two members on one camera, or two members
geometry positively excluded from each other, and the whole group is refused
rather than a member being picked to sacrifice. Transitivity is not evidence.

A pair that was never comparable is **not** a contradiction. Cameras with no
shared calibrated region are a supported deployment, not a fault, and treating
"never measured" as "measured and disagreed" would make association impossible
exactly where coverage is sparsest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from atreides_vision.association.config import AssociationDecisionConfig
from atreides_vision.association.models import (
    CandidateKey,
    CandidatePair,
    DecisionReason,
    ObservationGating,
    PairAssessment,
    TrackletKey,
    tracklet_key,
)

__all__ = [
    "CameraPreference",
    "PairAgreement",
    "camera_preferences",
    "inconsistent_pairs",
    "pair_agreement",
]


@dataclass(frozen=True, slots=True)
class CameraPreference:
    """One observation's nearest admissible candidate on one counterpart camera.

    Per camera rather than overall, because candidates on *different* cameras
    are not rivals: one person standing in three views produces three mutually
    consistent observations, and only the ones sharing a camera compete.
    """

    camera_id: str
    #: The preferred counterpart tracklet, ``(camera_id, stream_epoch, track_id)``.
    partner: TrackletKey
    separation_m: float
    #: How much further away the runner-up on this camera was, in metres.
    #: ``None`` when the camera offered only one candidate — no rival at all,
    #: which is a different situation from a rival at zero distance.
    rival_margin_m: float | None
    #: Whether the gating layer already declared this camera's candidates
    #: geometrically indistinguishable.
    ambiguous: bool


@dataclass(frozen=True, slots=True)
class PairAgreement:
    """Whether one instant supports a pair, and if not, which rule refused it."""

    agrees: bool
    #: ``None`` exactly when the pair agrees; otherwise the reason it did not.
    reason: DecisionReason | None
    #: The tighter of the two sides' rival margins, in metres, when both sides
    #: prefer each other. ``None`` when no rival was offered or the question did
    #: not arise.
    rival_margin_m: float | None


def camera_preferences(gating: ObservationGating) -> tuple[CameraPreference, ...]:
    """What this observation prefers on each counterpart camera, camera-sorted.

    Reads the ranking the gating layer already produced rather than re-sorting
    it, so preference order and gating order can never disagree.
    """
    ranked: dict[str, list[tuple[TrackletKey, float]]] = {}
    for assessment in gating.candidates:
        counterpart = assessment.pair.other_than(gating.subject)
        evidence = assessment.evidence
        # A candidate is admissible by construction, so it has evidence; the
        # check is here because a silent None would become a fabricated zero.
        if evidence is None:  # pragma: no cover - candidates always carry evidence
            raise AssertionError("a surviving candidate must carry its geometric evidence")
        ranked.setdefault(counterpart.camera_id, []).append(
            (tracklet_key(counterpart), evidence.separation_m)
        )

    preferences = []
    for camera_id in sorted(ranked):
        entries = ranked[camera_id]
        partner, separation = entries[0]
        margin = entries[1][1] - separation if len(entries) > 1 else None
        preferences.append(
            CameraPreference(
                camera_id=camera_id,
                partner=partner,
                separation_m=separation,
                rival_margin_m=margin,
                ambiguous=camera_id in gating.ambiguous_cameras,
            )
        )
    return tuple(preferences)


def pair_agreement(
    pair: CandidatePair,
    preferences: Mapping[TrackletKey, Sequence[CameraPreference]],
    config: AssociationDecisionConfig,
) -> PairAgreement:
    """Whether both sides of ``pair`` prefer each other clearly enough.

    ``preferences`` is looked up, never iterated, so no result here depends on
    the order cameras or observations arrived in.
    """
    first = tracklet_key(pair.first)
    second = tracklet_key(pair.second)
    towards_second = _preference_for(preferences.get(first, ()), second[0])
    towards_first = _preference_for(preferences.get(second, ()), first[0])

    if (
        towards_second is None
        or towards_first is None
        or towards_second.partner != second
        or towards_first.partner != first
    ):
        return PairAgreement(
            agrees=False, reason=DecisionReason.NOT_MUTUAL_BEST, rival_margin_m=None
        )

    margin = _tighter_margin(towards_second.rival_margin_m, towards_first.rival_margin_m)
    separated = not (towards_second.ambiguous or towards_first.ambiguous) and (
        margin is None or margin >= config.acceptance_margin_m
    )
    if not separated:
        return PairAgreement(
            agrees=False, reason=DecisionReason.RIVAL_NOT_SEPARATED, rival_margin_m=margin
        )
    return PairAgreement(agrees=True, reason=None, rival_margin_m=margin)


def inconsistent_pairs(
    accepted: Sequence[CandidateKey],
    assessments: Mapping[CandidateKey, PairAssessment],
) -> tuple[tuple[CandidateKey, DecisionReason], ...]:
    """Accepted pairs whose wider group contradicts itself, in canonical order.

    A group is every tracklet reachable through accepted pairs. It is refused
    whole — every pair in it, not a chosen victim — when two of its members sit
    on one camera, or when two of them were compared and geometry excluded each
    other. Refusing the whole group is both deterministic and the conservative
    direction: a group whose own evidence disagrees is not the place to guess
    which part was right.
    """
    groups = _groups(accepted)
    contradicted: list[tuple[CandidateKey, DecisionReason]] = []
    for members in groups:
        reason = _contradiction(members, assessments)
        if reason is None:
            continue
        contradicted.extend(
            (key, reason) for key in accepted if key[0] in members and key[1] in members
        )
    return tuple(sorted(contradicted))


def _preference_for(
    preferences: Sequence[CameraPreference], camera_id: str
) -> CameraPreference | None:
    for preference in preferences:
        if preference.camera_id == camera_id:
            return preference
    return None


def _tighter_margin(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _groups(accepted: Sequence[CandidateKey]) -> tuple[tuple[TrackletKey, ...], ...]:
    """Connected groups of tracklets joined by accepted pairs, canonically ordered."""
    parents: dict[TrackletKey, TrackletKey] = {}
    for first, second in sorted(accepted):
        parents.setdefault(first, first)
        parents.setdefault(second, second)
        root_first, root_second = _root(parents, first), _root(parents, second)
        if root_first != root_second:
            parents[max(root_first, root_second)] = min(root_first, root_second)

    grouped: dict[TrackletKey, list[TrackletKey]] = {}
    for member in sorted(parents):
        grouped.setdefault(_root(parents, member), []).append(member)
    return tuple(tuple(grouped[root]) for root in sorted(grouped))


def _root(parents: dict[TrackletKey, TrackletKey], key: TrackletKey) -> TrackletKey:
    while parents[key] != key:
        key = parents[key]
    return key


def _contradiction(
    members: Sequence[TrackletKey], assessments: Mapping[CandidateKey, PairAssessment]
) -> DecisionReason | None:
    cameras = [member[0] for member in members]
    if len(set(cameras)) != len(cameras):
        return DecisionReason.EXCLUSIVITY_CONFLICT
    for first, second in combinations(members, 2):
        assessment = assessments.get((first, second))
        if assessment is not None and assessment.is_comparable and not assessment.is_admissible:
            return DecisionReason.INCONSISTENT_EVIDENCE
    return None

"""Domain types for cross-camera association: what may be compared, and what geometry alone says.

Association is the capability that turns several cameras into one venue view.
It is also the capability where a mistake is most expensive: a wrong merge
corrupts every venue-level number above it and is harder to detect than a
missed one. So this module defines the vocabulary first, and defines it so that
the dangerous answer is hard to express.

What lives here
---------------
* :class:`TrackObservation` — one camera-local track, at one venue instant,
  carried with everything needed to decide whether it may be compared at all:
  its stream provenance, its venue time and that time's uncertainty, its
  position in a named coordinate space, and its camera's spatial context.
* :class:`CandidatePair` — two observations in a canonical order, so the same
  two observations are one candidate however they were enumerated.
* :class:`GeometricEvidence` — what the shared floor plane says about a pair.
* :class:`PairAssessment` and :class:`ObservationGating` — the geometric
  verdicts, with every rejection reason recorded.
* :class:`SupportSummary`, :class:`PairDecision` and :class:`AssociationRound`
  — V0.4.2's decision vocabulary: what sustained cross-camera agreement was
  observed for a candidate, what was concluded from it, and why.

Geometry rejects; it never associates
-------------------------------------
:class:`AssociationOutcome` names all four answers the association capability
can ever give, because a consumer must be able to handle them from the start.
Two people standing on the same square metre of floor at the same instant, seen
by two cameras, are geometrically indistinguishable from one person seen twice;
so agreement in space and time is *necessary* for a positive association and
nowhere near *sufficient* for one. Geometry can therefore veto and never
authorise, and :class:`ObservationGating` — the V0.4.1 geometric result —
refuses to be constructed with an ``ASSOCIATED`` outcome rather than trusting a
caller not to write it.

:class:`AssociationOutcome.ASSOCIATED` becomes reachable in V0.4.2, and only
through :class:`PairDecision`, which is constructable with that outcome only
when the evidence that authorises it is attached: a sustained agreement run, a
confidence, and an admissible geometric assessment across two different
cameras. The dangerous answer stays hard to express.

Identity boundary
-----------------
:class:`~atreides_vision.behavior.models.SubjectRef` stays exactly
``(camera_id, track_id)`` — camera-scoped, tracker-assigned, not a person.
Nothing here defines a venue-local identity, a subject store or a count of
people. A :class:`PairDecision` is a statement about *two tracklets*, not about
a person: it says the evidence supports treating them as one subject right now,
and it names both ``(camera, track)`` observations that produced it. Turning a
set of such decisions into one venue-local temporary identity is a later
capability, and when that type arrives it is a distinct type in its own
namespace, built from these decisions rather than from a reused tracker
integer.

Nothing here is biometric. No appearance descriptor, embedding, colour
histogram, crop or pixel of any kind reaches this layer: an observation carries
a projected position and nothing that could describe what a person looks like.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Self

from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.errors import AssociationError
from atreides_vision.timing import StreamRef, TimeUncertainty

__all__ = [
    "AssociationOutcome",
    "AssociationRound",
    "CameraSpatialContext",
    "CandidateKey",
    "CandidatePair",
    "DecisionConfidence",
    "DecisionReason",
    "GateReason",
    "GateRejection",
    "GeometricEvidence",
    "GeometryTrust",
    "ObservationGating",
    "PairAssessment",
    "PairDecision",
    "SupportSummary",
    "TrackObservation",
    "TrackletKey",
    "VenueRegion",
    "candidate_key",
    "observation_order_key",
    "tracklet_key",
]


class AssociationOutcome(StrEnum):
    """Every answer the association capability may give about a candidate.

    All four are first class and every caller must handle all four. "We could
    not tell" and "we were not allowed to look" are different facts, and
    collapsing either into "no" would hide a blind spot rather than report it.
    """

    #: The evidence supports treating these observations as one person.
    #: **Never produced by geometry alone** — see the module docstring. It is
    #: named here because the vocabulary is the contract, and a consumer written
    #: against three outcomes would break when the fourth arrives.
    ASSOCIATED = "associated"

    #: Comparable, and the available evidence does not single out one partner.
    #: The correct output when two candidates cannot be told apart: an
    #: arbitrary choice between them would be a coin flip presented as a fact.
    AMBIGUOUS = "ambiguous"

    #: Comparable, and the evidence excludes every candidate. A statement about
    #: this instant only, never about the person.
    UNASSOCIATED = "unassociated"

    #: The comparison could not be made at all — a hard gate refused it. Not a
    #: weaker ``UNASSOCIATED``: nothing was measured, so nothing was found.
    NOT_COMPARABLE = "not_comparable"


class GateReason(StrEnum):
    """Why a pair could not be compared. One member per hard gate.

    Recorded rather than logged, because "no association here" is only
    trustworthy if it can be traced to the rule that produced it. Every member
    describes a *precondition of the measurement* that failed, never a
    conclusion about the people in the frames.
    """

    #: Both observations are from one camera. Association is cross-camera by
    #: definition; joining two tracklets on one camera is fragment stitching,
    #: which is a different capability with different failure modes.
    SAME_CAMERA = "same_camera"

    #: The two cameras' metres are expressed in different venue coordinate
    #: frames, so their coordinates are unrelated numbers that share a unit.
    DIFFERENT_VENUE_FRAME = "different_venue_frame"

    #: The two cameras are calibrated to different planar surfaces. One
    #: homography describes one plane: a point on the main floor and a point on
    #: a balcony can have near-identical ``(x, y)`` and be a storey apart.
    DIFFERENT_SURFACE = "different_surface"

    #: At least one observation is not in calibrated ground-plane metres.
    #: Normalized image coordinates are camera-local by construction, and no
    #: physical distance may be derived from an uncalibrated scene.
    NOT_GROUND_PLANE_METRES = "not_ground_plane_metres"

    #: At least one camera's viewpoint no longer matches the one its
    #: calibration was authored against. Its metres are not trusted, so no
    #: distance computed from them may gate anything.
    CAMERA_GEOMETRY_UNTRUSTED = "camera_geometry_untrusted"

    #: At least one observation's position on the venue timeline was never
    #: established. Without that, the two observations cannot be placed on a
    #: common clock at all, and "at the same moment" has no meaning.
    VENUE_TIME_NOT_ESTABLISHED = "venue_time_not_established"

    #: Even allowing for the stated timing uncertainty in the pair's favour,
    #: the two observations are too far apart in venue time to describe one
    #: instant.
    NOT_SIMULTANEOUS = "not_simultaneous"

    #: Both cameras declared the region their calibration covers, and those
    #: regions are further apart than any person could bridge in the time
    #: between the two observations. Non-overlapping cameras are an ordinary
    #: deployment, not an error.
    NO_SHARED_CALIBRATED_REGION = "no_shared_calibrated_region"


@dataclass(frozen=True, slots=True)
class GateRejection:
    """One hard gate's refusal, with the sentence an operator needs.

    ``reason`` is for a machine, ``detail`` is for a human reading a diagnostic
    and asking why a camera pair never associates.
    """

    reason: GateReason
    detail: str

    def __post_init__(self) -> None:
        if not self.detail or not self.detail.strip():
            raise AssociationError(
                f"GateRejection({self.reason.value}) must explain itself: a rejection with no "
                "detail cannot be acted on by whoever reads it."
            )

    def to_record(self) -> dict[str, Any]:
        return {"reason": self.reason.value, "detail": self.detail}

    def __str__(self) -> str:
        return f"{self.reason.value}: {self.detail}"


class GeometryTrust(StrEnum):
    """How far a camera's calibrated metres may currently be relied on.

    A deliberate narrowing of
    :class:`~atreides_vision.calibration.integrity.models.IntegrityState`, made
    by whoever owns the integrity guard and passed in as a value. Association
    consumes the *conclusion* rather than importing the calibration machinery,
    which is what keeps OpenCV — and every pixel — out of this layer.

    The intended mapping, which the composition root performs::

        CONSISTENT           -> VERIFIED
        UNCHECKED            -> UNVERIFIED
        INSUFFICIENT_EVIDENCE-> UNVERIFIED
        DRIFT_DETECTED       -> UNTRUSTED

    ``UNCHECKED`` and ``INSUFFICIENT_EVIDENCE`` collapse into one level on
    purpose: neither is evidence that the camera moved, and neither is evidence
    that it did not. What they share — nobody confirmed the viewpoint — is
    exactly what a consumer of a geometric comparison needs to know.
    """

    #: The viewpoint was checked against the calibration reference and matched.
    VERIFIED = "verified"

    #: Nobody confirmed the viewpoint: monitoring is off, has not run yet, or
    #: could not reach a conclusion. Comparison proceeds and says so.
    UNVERIFIED = "unverified"

    #: The camera moved. Its metres are no longer trusted, so they gate nothing.
    UNTRUSTED = "untrusted"

    @property
    def permits_comparison(self) -> bool:
        """Whether geometry from this camera may be compared at all.

        Only positively detected movement forbids it. An unverified camera is
        not a trusted one, but refusing it outright would make association
        depend on integrity monitoring being deployed, which is a deployment
        fact rather than a property of the data.
        """
        return self is not GeometryTrust.UNTRUSTED

    def worst_of(self, other: GeometryTrust) -> GeometryTrust:
        """The weaker of two cameras' trust — what a pair drawn from both is worth."""
        return self if _TRUST_RANK[self] >= _TRUST_RANK[other] else other


#: Higher is weaker. Ordering lives here rather than in member order so the
#: serialised values stay free to change without changing the comparison.
_TRUST_RANK: Final[dict[GeometryTrust, int]] = {
    GeometryTrust.VERIFIED: 0,
    GeometryTrust.UNVERIFIED: 1,
    GeometryTrust.UNTRUSTED: 2,
}


@dataclass(frozen=True, slots=True)
class VenueRegion:
    """The venue-metre footprint a camera's calibration covers.

    Built from the venue coordinates an operator actually supplied — typically
    ``GroundPlaneCalibration.venue_points()`` — so it describes *where this
    camera's metres were fitted*, not where the camera can see. The two differ:
    a camera calibrated against one doorway may view half a hall, and positions
    outside the region are extrapolations of the homography.

    Deliberately an axis-aligned bounding box rather than a convex hull. The box
    is a superset of the hull, so a gate built on it can only ever under-reject,
    and the failure direction of an over-simple footprint is therefore a missed
    association rather than a fabricated one.

    Nothing here assumes a venue size, shape, origin or camera coverage: the
    numbers come from the operator's own correspondences and may be negative.
    """

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        for name, value in (
            ("min_x", self.min_x),
            ("min_y", self.min_y),
            ("max_x", self.max_x),
            ("max_y", self.max_y),
        ):
            if not math.isfinite(value):
                raise AssociationError(f"VenueRegion.{name} must be finite metres, got {value!r}")
        if self.max_x < self.min_x or self.max_y < self.min_y:
            raise AssociationError(
                "VenueRegion requires min <= max on both axes, got "
                f"({self.min_x}, {self.min_y})-({self.max_x}, {self.max_y})"
            )

    @classmethod
    def covering(cls, points: Iterable[tuple[float, float]]) -> Self:
        """The smallest axis-aligned region containing every ``(x, y)`` in metres."""
        collected = [(float(x), float(y)) for x, y in points]
        if not collected:
            raise AssociationError(
                "A venue region needs at least one point. An empty region would claim a camera "
                "covers nothing, which is different from not knowing what it covers (None)."
            )
        xs = [x for x, _ in collected]
        ys = [y for _, y in collected]
        return cls(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))

    def gap_to(self, other: VenueRegion) -> float:
        """Shortest distance in metres between the two regions; ``0.0`` if they meet."""
        dx = max(0.0, self.min_x - other.max_x, other.min_x - self.max_x)
        dy = max(0.0, self.min_y - other.max_y, other.min_y - self.max_y)
        return math.hypot(dx, dy)

    def to_record(self) -> dict[str, Any]:
        return {
            "min_x": round(self.min_x, 6),
            "min_y": round(self.min_y, 6),
            "max_x": round(self.max_x, 6),
            "max_y": round(self.max_y, 6),
        }


@dataclass(frozen=True, slots=True)
class CameraSpatialContext:
    """What one camera's geometry is currently worth for cross-camera comparison.

    Distinct from :class:`~atreides_vision.interaction.models.SpatialContext`,
    which encodes the opposite rule: interaction requires both subjects on the
    *same* camera, because without association one person seen twice would be
    paired with themselves. Association requires *different* cameras and adds
    the two facts that only matter once cameras are compared — whether the
    viewpoint is still trusted, and which part of the floor the metres were
    fitted over.

    Raises:
        AssociationError: if metric context omits the venue or surface it is
            expressed in, or if a region is declared for a non-metric space.
    """

    camera_id: str
    space: MotionSpace
    #: Required for metric space, meaningless without one.
    venue_id: str | None = None
    surface_id: str | None = None
    #: Defaults to the honest state: nobody confirmed this camera's viewpoint.
    trust: GeometryTrust = GeometryTrust.UNVERIFIED
    #: ``None`` means *not declared*, which is different from *covers nothing*.
    #: Where it is ``None`` the region gate does not run at all.
    calibrated_region: VenueRegion | None = None

    def __post_init__(self) -> None:
        if not self.camera_id or not self.camera_id.strip():
            raise AssociationError("CameraSpatialContext.camera_id must be a non-empty string")
        if self.space.is_physical:
            if not self.venue_id or not self.surface_id:
                raise AssociationError(
                    "Metric spatial context needs both a venue_id and a surface_id: without them "
                    "there is no way to tell whether two cameras' metres describe the same "
                    "coordinate frame and the same physical plane."
                )
        elif self.calibrated_region is not None:
            raise AssociationError(
                f"Camera '{self.camera_id}' declares a calibrated region but measures in "
                f"{self.space.value}. A region is venue metres; an uncalibrated space has none."
            )

    @property
    def is_metric(self) -> bool:
        return self.space.is_physical

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "camera_id": self.camera_id,
            "space": self.space.value,
            "trust": self.trust.value,
        }
        if self.venue_id is not None:
            record["venue_id"] = self.venue_id
        if self.surface_id is not None:
            record["surface_id"] = self.surface_id
        if self.calibrated_region is not None:
            record["calibrated_region"] = self.calibrated_region.to_record()
        return record


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """One camera-local track, at one venue instant, ready to be compared.

    Everything a hard gate needs, and nothing more. In particular there is no
    image, no bounding box and no appearance of any kind: the position arrives
    already projected, so no pixel exists above the projector seam.

    Provenance is complete by construction. ``subject`` says which tracklet on
    which camera; ``stream`` says which recording of that camera, so a tracklet
    from before a reconnect is never silently continued into one from after it;
    ``frame_index`` resolves back to the exact frame of the exact file.

    Raises:
        AssociationError: if the subject, stream and spatial context disagree
            about which camera this is, if the position is not in the space the
            context declares, or if the venue time is not a real instant.
    """

    subject: SubjectRef
    stream: StreamRef
    #: Frame number within this stream, as the pipeline counts it (1-based).
    frame_index: int
    #: The session-relative venue instant. Never a wall clock, never a source
    #: second: source seconds restart at an epoch boundary and cannot be
    #: compared across cameras.
    venue_seconds: float
    #: How well this stream's clock is related to the venue timeline.
    timing: TimeUncertainty
    #: The subject's floor anchor, already projected into ``spatial.space``.
    position: MotionPoint
    spatial: CameraSpatialContext

    def __post_init__(self) -> None:
        cameras = {self.subject.camera_id, self.stream.camera_id, self.spatial.camera_id}
        if len(cameras) != 1:
            raise AssociationError(
                "A track observation must describe one camera, got subject "
                f"'{self.subject.camera_id}', stream '{self.stream.camera_id}' and context "
                f"'{self.spatial.camera_id}'."
            )
        if self.position.space is not self.spatial.space:
            raise AssociationError(
                f"Observation {self.subject} is positioned in {self.position.space.value} but its "
                f"camera context declares {self.spatial.space.value}. One of them is wrong, and "
                "guessing which would put a person in the wrong coordinate system."
            )
        if self.frame_index < 0:
            raise AssociationError(
                f"TrackObservation.frame_index must be non-negative, got {self.frame_index!r}"
            )
        if not math.isfinite(self.venue_seconds) or self.venue_seconds < 0:
            raise AssociationError(
                "TrackObservation.venue_seconds must be finite and non-negative, got "
                f"{self.venue_seconds!r}. Venue time is session-relative, so nothing precedes zero."
            )

    @property
    def camera_id(self) -> str:
        return self.subject.camera_id

    @property
    def space(self) -> MotionSpace:
        return self.position.space

    def to_record(self) -> dict[str, Any]:
        return {
            "camera_id": self.subject.camera_id,
            "track_id": self.subject.track_id,
            "stream_epoch": self.stream.stream_epoch,
            "frame_index": self.frame_index,
            "venue_seconds": round(self.venue_seconds, 3),
            "timing_basis": self.timing.basis.value,
            "timing_half_width_seconds": self.timing.half_width_seconds,
            "position": [round(self.position.x, 6), round(self.position.y, 6)],
            "space": self.position.space.value,
            "trust": self.spatial.trust.value,
        }

    def __str__(self) -> str:
        return f"{self.subject}@{self.stream}"


def observation_order_key(observation: TrackObservation) -> tuple[str, int, int, int]:
    """Total order over observations, independent of how they were enumerated.

    ``(camera_id, stream_epoch, track_id, frame_index)``. ``camera_id`` compares
    by Python's default string ordering — Unicode code point, never locale and
    never insertion order — matching the venue scheduler's rule, so association
    order cannot depend on manifest camera order.
    """
    return (
        observation.subject.camera_id,
        observation.stream.stream_epoch,
        observation.subject.track_id,
        observation.frame_index,
    )


@dataclass(frozen=True, slots=True)
class CandidatePair:
    """Two observations that might describe one person, in a canonical order.

    ``(A, B)`` and ``(B, A)`` are one candidate, so the pair sorts its members
    at construction by :func:`observation_order_key`. Without that, every
    downstream count, record and set would silently double the moment two
    cameras were iterated in a different order.

    A pair may be constructed for two observations that turn out not to be
    comparable at all — that is the point: a rejection has to be *representable*
    to be reportable. The only pairing refused here is an observation with
    itself, which is not a pair in any ordering.

    Raises:
        AssociationError: for a self-pair.
    """

    first: TrackObservation
    second: TrackObservation

    def __init__(self, first: TrackObservation, second: TrackObservation) -> None:
        if first.subject == second.subject:
            raise AssociationError(
                f"An observation cannot be a candidate for itself: {first.subject} paired with "
                f"{second.subject}."
            )
        low, high = sorted((first, second), key=observation_order_key)
        object.__setattr__(self, "first", low)
        object.__setattr__(self, "second", high)

    @property
    def subjects(self) -> tuple[SubjectRef, SubjectRef]:
        return (self.first.subject, self.second.subject)

    @property
    def cameras(self) -> tuple[str, str]:
        return (self.first.camera_id, self.second.camera_id)

    @property
    def key(self) -> tuple[tuple[str, int, int, int], tuple[str, int, int, int]]:
        """The pair's position in the total order over candidates."""
        return (observation_order_key(self.first), observation_order_key(self.second))

    def other_than(self, subject: SubjectRef) -> TrackObservation:
        """The member of the pair that is not ``subject``."""
        if subject == self.first.subject:
            return self.second
        if subject == self.second.subject:
            return self.first
        raise AssociationError(f"{subject} is not part of candidate {self}")

    def to_record(self) -> dict[str, Any]:
        return {"first": self.first.to_record(), "second": self.second.to_record()}

    def __iter__(self) -> Iterator[TrackObservation]:
        return iter((self.first, self.second))

    def __str__(self) -> str:
        return f"{self.first} <-> {self.second}"


@dataclass(frozen=True, slots=True)
class GeometricEvidence:
    """What the shared floor plane says about one candidate pair.

    Measurements and the tolerance they were judged against — never a score, a
    likelihood or a similarity. The admissible radius is reported alongside the
    separation precisely so a reader can see *why* a pair survived: a pair that
    passed only because the two clocks are half a minute apart looks exactly
    like one that passed on a tight metre, unless the radius is on the record.

    Uncertainty widens this radius; it never narrows it. A gate that grew
    stricter as the clocks got worse would be deriving a rejection from a number
    nobody established, which is the same laundering V0.4.0 exists to prevent.
    """

    #: Distance between the two projected positions, in venue metres.
    separation_m: float
    #: How far apart two cameras may place one person at one instant.
    position_tolerance_m: float
    #: How far a person could have moved in the time that may separate the two
    #: observations, including the timing uncertainty of both streams.
    motion_allowance_m: float
    #: ``|venue time difference|`` as stated, before uncertainty.
    nominal_gap_seconds: float
    #: The two streams' timing half-widths added together: how much the stated
    #: gap could be wrong in either direction.
    timing_slack_seconds: float
    #: The weaker of the two cameras' geometry trust.
    trust: GeometryTrust

    def __post_init__(self) -> None:
        for name, value in (
            ("separation_m", self.separation_m),
            ("position_tolerance_m", self.position_tolerance_m),
            ("motion_allowance_m", self.motion_allowance_m),
            ("nominal_gap_seconds", self.nominal_gap_seconds),
            ("timing_slack_seconds", self.timing_slack_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise AssociationError(
                    f"GeometricEvidence.{name} must be finite and non-negative, got {value!r}"
                )

    @property
    def admissible_radius_m(self) -> float:
        """The separation at or below which geometry does not exclude the pair."""
        return self.position_tolerance_m + self.motion_allowance_m

    @property
    def margin_m(self) -> float:
        """Metres of room left before exclusion. Negative when excluded."""
        return self.admissible_radius_m - self.separation_m

    @property
    def is_admissible(self) -> bool:
        """Whether geometry fails to exclude this pair.

        Deliberately phrased as a double negative, because that is what it
        means. An admissible pair is one geometry cannot rule out — not one
        geometry supports.
        """
        return self.separation_m <= self.admissible_radius_m

    def to_record(self) -> dict[str, Any]:
        return {
            "separation_m": round(self.separation_m, 6),
            "admissible_radius_m": round(self.admissible_radius_m, 6),
            "position_tolerance_m": round(self.position_tolerance_m, 6),
            "motion_allowance_m": round(self.motion_allowance_m, 6),
            "margin_m": round(self.margin_m, 6),
            "nominal_gap_seconds": round(self.nominal_gap_seconds, 3),
            "timing_slack_seconds": round(self.timing_slack_seconds, 3),
            "trust": self.trust.value,
        }


@dataclass(frozen=True, slots=True)
class PairAssessment:
    """One candidate pair, its hard-gate rejections, and its geometry if any.

    ``evidence`` exists exactly when every hard gate passed. A rejected pair
    carries no measurement on purpose: numbers computed across a gate nobody
    should have crossed would be read later as if they meant something.
    """

    pair: CandidatePair
    #: Every hard gate that refused this pair, in the fixed order they are
    #: evaluated. All applicable gates are recorded, not just the first: an
    #: operator fixing a deployment needs the whole list.
    rejections: tuple[GateRejection, ...] = ()
    evidence: GeometricEvidence | None = None

    def __post_init__(self) -> None:
        if self.rejections and self.evidence is not None:
            raise AssociationError(
                f"Candidate {self.pair} was rejected by "
                f"{[rejection.reason.value for rejection in self.rejections]} yet carries "
                "geometric evidence. Measurements across a closed gate are not measurements."
            )
        if not self.rejections and self.evidence is None:
            raise AssociationError(
                f"Candidate {self.pair} passed every hard gate but carries no geometry. A "
                "comparable pair that was never measured is an unreported blind spot."
            )

    @property
    def is_comparable(self) -> bool:
        """Whether the pair cleared every hard gate and could be measured."""
        return not self.rejections

    @property
    def is_admissible(self) -> bool:
        """Whether the pair is comparable *and* geometry does not exclude it."""
        return self.evidence is not None and self.evidence.is_admissible

    @property
    def geometric_outcome(self) -> AssociationOutcome | None:
        """The outcome geometry alone establishes for this pair.

        ``None`` means geometry has no verdict: the pair survived, and whether
        it is one person is a question for evidence this layer does not have.
        :attr:`AssociationOutcome.ASSOCIATED` is structurally unreachable here.
        """
        if self.rejections:
            return AssociationOutcome.NOT_COMPARABLE
        if not self.is_admissible:
            return AssociationOutcome.UNASSOCIATED
        return None

    def reasons(self) -> tuple[GateReason, ...]:
        return tuple(rejection.reason for rejection in self.rejections)

    def to_record(self) -> dict[str, Any]:
        outcome = self.geometric_outcome
        record: dict[str, Any] = {
            "pair": self.pair.to_record(),
            "geometric_outcome": None if outcome is None else outcome.value,
        }
        if self.rejections:
            record["rejections"] = [rejection.to_record() for rejection in self.rejections]
        if self.evidence is not None:
            record["evidence"] = self.evidence.to_record()
        return record


@dataclass(frozen=True, slots=True)
class ObservationGating:
    """Everything geometry established about one observation at one instant.

    Carries the full considered set, not just the survivors: a candidate that
    was refused, and the gate that refused it, is the provenance that makes
    "no association here" auditable rather than merely asserted.

    Raises:
        AssociationError: if the outcome is
            :attr:`AssociationOutcome.ASSOCIATED`, or if it disagrees with the
            candidates it was supposedly derived from. Geometry authorising a
            merge is prevented by construction rather than by convention.
    """

    observation: TrackObservation
    #: Every pair considered, in candidate order.
    assessments: tuple[PairAssessment, ...] = ()
    #: The subset geometry could not exclude, nearest first. Not a ranking of
    #: how likely each is to be the same person — only how far away it was.
    candidates: tuple[PairAssessment, ...] = ()
    #: Counterpart cameras offering candidates geometry cannot tell apart.
    ambiguous_cameras: tuple[str, ...] = ()
    #: The outcome geometry alone establishes, or ``None`` when it defers.
    geometric_outcome: AssociationOutcome | None = None

    def __post_init__(self) -> None:
        if self.geometric_outcome is AssociationOutcome.ASSOCIATED:
            raise AssociationError(
                "Geometry cannot establish ASSOCIATED. Two people on one square metre of floor "
                "look exactly like one person seen twice, so spatial and temporal agreement is a "
                "precondition for a merge and never a reason for one."
            )
        if self.ambiguous_cameras and self.geometric_outcome is not AssociationOutcome.AMBIGUOUS:
            raise AssociationError(
                f"Observation {self.observation.subject} has candidates geometry cannot separate "
                f"on {list(self.ambiguous_cameras)} but reports "
                f"{self.geometric_outcome}. Unresolved ambiguity must not be reported as a "
                "resolved outcome."
            )
        if self.geometric_outcome is AssociationOutcome.AMBIGUOUS and not self.ambiguous_cameras:
            raise AssociationError(
                f"Observation {self.observation.subject} reports AMBIGUOUS without naming a "
                "camera whose candidates could not be separated. An ambiguity nobody can locate "
                "is not reportable."
            )
        if self.geometric_outcome is AssociationOutcome.UNASSOCIATED and self.candidates:
            raise AssociationError(
                f"Observation {self.observation.subject} reports UNASSOCIATED while carrying "
                f"{len(self.candidates)} admissible candidate(s)."
            )

    @property
    def subject(self) -> SubjectRef:
        return self.observation.subject

    @property
    def comparable(self) -> tuple[PairAssessment, ...]:
        """The pairs that cleared every hard gate, whatever geometry then said."""
        return tuple(assessment for assessment in self.assessments if assessment.is_comparable)

    def to_record(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_record(),
            "geometric_outcome": (
                None if self.geometric_outcome is None else self.geometric_outcome.value
            ),
            "ambiguous_cameras": list(self.ambiguous_cameras),
            "considered": len(self.assessments),
            "comparable": len(self.comparable),
            "assessments": [assessment.to_record() for assessment in self.assessments],
        }


#: A camera-local tracklet's stable identity across ticks:
#: ``(camera_id, stream_epoch, track_id)``. It is *not* a venue-local identity
#: and not a person — it is the same tracklet a camera-local tracker produced,
#: named so that evidence about it can be carried from one venue instant to the
#: next. ``stream_epoch`` is part of it because ``track_7`` before a reconnect
#: and ``track_7`` after one are unrelated tracklets that share an integer.
TrackletKey = tuple[str, int, int]

#: Two tracklets in canonical order — the stable name of a candidate pair,
#: independent of the frame it was observed on and of camera enumeration order.
CandidateKey = tuple[TrackletKey, TrackletKey]


def tracklet_key(observation: TrackObservation) -> TrackletKey:
    """The observation's tracklet, without the frame it happened to be seen on.

    :func:`observation_order_key` orders *observations*; this names the
    *tracklet* behind them, which is what evidence accumulates against over
    time. Sorting agrees with :func:`observation_order_key` because it is that
    key with its last component dropped.
    """
    return (
        observation.subject.camera_id,
        observation.stream.stream_epoch,
        observation.subject.track_id,
    )


def candidate_key(pair: CandidatePair) -> CandidateKey:
    """The pair's stable name across venue instants.

    :class:`CandidatePair` already sorts its members by
    :func:`observation_order_key`, whose first three components are exactly
    :func:`tracklet_key`, so this preserves that canonical order. Two distinct
    tracklets can never tie on it, which is what makes the same two tracklets
    reach the same evidence entry however the cameras were enumerated.
    """
    return (tracklet_key(pair.first), tracklet_key(pair.second))


class DecisionReason(StrEnum):
    """Why a candidate was decided the way it was. Recorded, never inferred.

    A decision is only trustworthy if it can be traced to the rule that
    produced it, and the *negative* reasons matter most: "no association here"
    has to say whether nothing was measured, whether geometry excluded it,
    whether the evidence was not yet sustained, or whether a rival could not be
    separated. Those are four different facts about the venue and only one of
    them is about the people.
    """

    #: A hard comparability gate rejected the pair. Which gates is in the
    #: assessment's own rejection list.
    HARD_GATE_REJECTED = "hard_gate_rejected"

    #: Comparable, and the separation exceeded the admissible radius. A
    #: statement about this instant only, never about the people.
    GEOMETRY_EXCLUDED = "geometry_excluded"

    #: The agreement run is real but too short — too few observations, or too
    #: little venue time, or both. Not a rejection: an association that has not
    #: yet earned the right to be asserted.
    SUPPORT_NOT_SUSTAINED = "support_not_sustained"

    #: A rival candidate on the same counterpart camera is too close in
    #: separation for either to be preferred. Choosing the nearer would be a
    #: coin flip presented as a fact.
    RIVAL_NOT_SEPARATED = "rival_not_separated"

    #: One side of the pair prefers a different partner on the other's camera.
    #: An association neither side agrees on is not an association.
    NOT_MUTUAL_BEST = "not_mutual_best"

    #: Accepting this pair would give one observation two positive partners on
    #: one counterpart camera. At one instant, one person is one track on one
    #: camera, so all the rivals are refused rather than one being picked.
    EXCLUSIVITY_CONFLICT = "exclusivity_conflict"

    #: Accepting this pair would join a group of tracklets whose own pairwise
    #: evidence contradicts itself — two of them on one camera, or two of them
    #: geometrically excluded from each other. Transitivity is not evidence.
    INCONSISTENT_EVIDENCE = "inconsistent_evidence"

    #: The one positive reason: mutual, exclusive, separated agreement,
    #: sustained across enough venue time and enough observations.
    SUSTAINED_MUTUAL_AGREEMENT = "sustained_mutual_agreement"


class DecisionConfidence(StrEnum):
    """How far a positive association may be relied on. Not a probability.

    Deliberately two coarse levels derived from a fact that was *established*
    rather than from a score nobody calibrated: whether both contributing
    cameras' viewpoints were verified against the calibration they are being
    measured with. Degraded integrity therefore lowers confidence visibly
    instead of being swallowed, which is the property the architecture asks
    for; a number between 0 and 1 would imply a calibration this project has
    not done.

    The temporal strength of the evidence is not folded in here — it is
    reported as measurements on :class:`SupportSummary`, where a reader can see
    how many observations over how many seconds produced the decision.
    """

    #: At least one contributing camera's viewpoint was never confirmed.
    PROVISIONAL = "provisional"

    #: Every contributing camera's geometry was verified against its
    #: calibration reference.
    CORROBORATED = "corroborated"


@dataclass(frozen=True, slots=True)
class SupportSummary:
    """The temporal evidence standing behind one candidate pair, as measurements.

    ``samples`` and ``span_seconds`` describe the *current unbroken run of
    agreement* — not the pair's whole life. A contradicting observation or a
    gap ends the run and the counters restart, because a claim that two
    tracklets agreed for three seconds must not span a second in which they
    disagreed.

    Both a count and a duration are reported because either alone is
    misleading: a count is meaningless without a frame rate, and no frame rate
    may be assumed here, while a duration alone could be two observations an
    hour apart.
    """

    #: Consecutive agreeing observations in the run in progress. ``0`` when the
    #: most recent observation broke it.
    samples: int
    #: Every observation ever recorded for this pair, agreeing or not. Lets a
    #: reader see that a run was interrupted rather than never started.
    total_samples: int
    #: ``last_seconds - started_seconds`` of the current run, in venue seconds.
    span_seconds: float
    #: Venue time the current run began, or ``None`` when no run is in progress.
    started_seconds: float | None
    #: Venue time of the most recent observation of this pair.
    last_seconds: float
    #: Weakest geometry trust seen during the run — verified only if every
    #: contributing observation was. With no run in progress, the most recent
    #: observation's own trust.
    trust: GeometryTrust
    #: Largest separation, in metres, seen during the run; with no run in
    #: progress, the most recent observation's own separation.
    max_separation_m: float
    #: Smallest lead the preferred partner held over its nearest rival during
    #: the run, in metres. ``None`` when no rival was ever offered.
    min_rival_margin_m: float | None

    def __post_init__(self) -> None:
        if self.samples < 0 or self.total_samples < self.samples:
            raise AssociationError(
                f"SupportSummary counts are incoherent: {self.samples} agreeing observations out "
                f"of {self.total_samples} recorded."
            )
        for name, value in (
            ("span_seconds", self.span_seconds),
            ("last_seconds", self.last_seconds),
            ("max_separation_m", self.max_separation_m),
        ):
            if not math.isfinite(value) or value < 0:
                raise AssociationError(
                    f"SupportSummary.{name} must be finite and non-negative, got {value!r}"
                )
        if (self.samples == 0) != (self.started_seconds is None):
            raise AssociationError(
                "A run in progress has a start and an empty run has none, got "
                f"{self.samples} sample(s) with started_seconds={self.started_seconds!r}."
            )

    @property
    def confidence(self) -> DecisionConfidence:
        """Corroborated only when every contributing camera's geometry was verified."""
        if self.trust is GeometryTrust.VERIFIED:
            return DecisionConfidence.CORROBORATED
        return DecisionConfidence.PROVISIONAL

    def to_record(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "total_samples": self.total_samples,
            "span_seconds": round(self.span_seconds, 3),
            "started_seconds": (
                None if self.started_seconds is None else round(self.started_seconds, 3)
            ),
            "last_seconds": round(self.last_seconds, 3),
            "trust": self.trust.value,
            "max_separation_m": round(self.max_separation_m, 6),
            "min_rival_margin_m": (
                None if self.min_rival_margin_m is None else round(self.min_rival_margin_m, 6)
            ),
        }


@dataclass(frozen=True, slots=True)
class PairDecision:
    """What the association core concluded about one candidate pair, and why.

    Carries its whole provenance: the V0.4.1 assessment it was decided from —
    with both ``(camera, track)`` observations, every hard gate that refused
    them and the geometry that measured them — plus the temporal support and
    the reasons for the outcome. Nothing here is a person, an identity or a
    count; it is a statement about two tracklets at one venue instant.

    ``ASSOCIATED`` is constrained rather than trusted. It cannot be built
    without an admissible cross-camera assessment, the sustained-agreement
    reason, a support summary and a confidence, so no caller can assert a
    positive association it has no evidence for.

    Raises:
        AssociationError: for a decision that does not match the evidence
            attached to it.
    """

    assessment: PairAssessment
    outcome: AssociationOutcome
    #: Why, in the fixed order the core evaluates them. Never empty.
    reasons: tuple[DecisionReason, ...]
    #: The temporal evidence, present for every comparable pair. ``None``
    #: exactly when nothing was measurable.
    support: SupportSummary | None = None
    #: Present only on a positive association: an outcome that is not an
    #: association has no confidence to state.
    confidence: DecisionConfidence | None = None

    def __post_init__(self) -> None:
        if not self.reasons:
            raise AssociationError(
                f"Decision {self.outcome.value} for {self.assessment.pair} states no reason. An "
                "association nobody can trace to a rule is not auditable."
            )
        comparable = self.assessment.is_comparable
        if comparable == (self.outcome is AssociationOutcome.NOT_COMPARABLE):
            raise AssociationError(
                f"Candidate {self.assessment.pair} is "
                f"{'comparable' if comparable else 'not comparable'} yet was decided "
                f"{self.outcome.value}. NOT_COMPARABLE means a hard gate refused the "
                "measurement, and is not a weaker UNASSOCIATED."
            )
        if comparable and self.support is None:
            raise AssociationError(
                f"Comparable candidate {self.assessment.pair} carries no support summary. A pair "
                "that was measured and not recorded is an unreported blind spot."
            )
        if self.outcome is not AssociationOutcome.ASSOCIATED:
            self._check_not_associated()
            return
        self._check_associated()

    def _check_not_associated(self) -> None:
        if self.confidence is not None:
            raise AssociationError(
                f"Candidate {self.assessment.pair} was decided {self.outcome.value} yet states a "
                f"confidence of {self.confidence.value}. Only an association has one."
            )
        if DecisionReason.SUSTAINED_MUTUAL_AGREEMENT in self.reasons:
            raise AssociationError(
                f"Candidate {self.assessment.pair} was decided {self.outcome.value} while "
                "claiming sustained mutual agreement, which is the reason for an association."
            )

    def _check_associated(self) -> None:
        if DecisionReason.SUSTAINED_MUTUAL_AGREEMENT not in self.reasons:
            raise AssociationError(
                f"Candidate {self.assessment.pair} was decided ASSOCIATED without sustained "
                "mutual agreement. A positive association states the evidence that earned it."
            )
        if not self.assessment.is_admissible:
            raise AssociationError(
                f"Candidate {self.assessment.pair} was decided ASSOCIATED although geometry "
                "excludes it. A merge geometry rules out corrupts every venue-level number "
                "above it."
            )
        if self.support is None or self.confidence is None:
            raise AssociationError(
                f"Candidate {self.assessment.pair} was decided ASSOCIATED without its support "
                "and confidence. Every association carries its evidence and its uncertainty."
            )
        first, second = self.assessment.pair.cameras
        if first == second:
            raise AssociationError(
                f"Candidate {self.assessment.pair} was decided ASSOCIATED within camera "
                f"'{first}'. Joining two tracklets on one camera is fragment stitching, a "
                "different capability with different failure modes."
            )

    @property
    def pair(self) -> CandidatePair:
        return self.assessment.pair

    @property
    def key(self) -> CandidateKey:
        return candidate_key(self.assessment.pair)

    @property
    def subjects(self) -> tuple[SubjectRef, SubjectRef]:
        """The two ``(camera_id, track_id)`` references this decision is about."""
        return self.assessment.pair.subjects

    @property
    def is_associated(self) -> bool:
        return self.outcome is AssociationOutcome.ASSOCIATED

    def involves(self, subject: SubjectRef) -> bool:
        return subject in self.assessment.pair.subjects

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "outcome": self.outcome.value,
            "reasons": [reason.value for reason in self.reasons],
            "assessment": self.assessment.to_record(),
        }
        if self.support is not None:
            record["support"] = self.support.to_record()
        if self.confidence is not None:
            record["confidence"] = self.confidence.value
        return record

    def __str__(self) -> str:
        return f"{self.assessment.pair}: {self.outcome.value}"


@dataclass(frozen=True, slots=True)
class AssociationRound:
    """Every cross-camera decision made at one venue instant.

    Complete rather than filtered: a decision is emitted for every candidate
    pair, including the ones a hard gate refused, because "these two cameras
    never associate" is only diagnosable if the refusals are on the record too.

    Ordering is canonical throughout — observations by
    :func:`observation_order_key`, decisions by :func:`candidate_key` — so a
    round is byte-identical between runs and independent of the order the
    cameras were enumerated in.

    Raises:
        AssociationError: if the venue instant is not a real one, or the
            contents are not in canonical order.
    """

    #: The session-relative venue instant these decisions describe.
    venue_seconds: float
    observations: tuple[TrackObservation, ...]
    decisions: tuple[PairDecision, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.venue_seconds) or self.venue_seconds < 0:
            raise AssociationError(
                "AssociationRound.venue_seconds must be finite and non-negative, got "
                f"{self.venue_seconds!r}."
            )
        keys = [observation_order_key(observation) for observation in self.observations]
        if keys != sorted(keys):
            raise AssociationError(
                "AssociationRound observations must be in canonical order; an order that "
                "depends on camera enumeration is not reproducible."
            )
        decision_keys = [decision.key for decision in self.decisions]
        if decision_keys != sorted(decision_keys):
            raise AssociationError(
                "AssociationRound decisions must be in canonical candidate order; an order that "
                "depends on camera enumeration is not reproducible."
            )

    @property
    def associated(self) -> tuple[PairDecision, ...]:
        """The positive associations, in canonical order."""
        return tuple(decision for decision in self.decisions if decision.is_associated)

    def decisions_for(self, subject: SubjectRef) -> tuple[PairDecision, ...]:
        """Every decision this subject took part in, in canonical order."""
        self._require_present(subject)
        return tuple(decision for decision in self.decisions if decision.involves(subject))

    def outcome_for(self, subject: SubjectRef) -> AssociationOutcome:
        """What this instant established about one observation.

        The strongest outcome any of its candidates reached: an association if
        one was made, otherwise an unresolved candidate if one remains,
        otherwise exclusion — and ``NOT_COMPARABLE`` when nothing about this
        observation could be measured at all, which is what a single-camera
        venue produces for every observation, every time.
        """
        decisions = self.decisions_for(subject)
        outcomes = {decision.outcome for decision in decisions}
        for outcome in (
            AssociationOutcome.ASSOCIATED,
            AssociationOutcome.AMBIGUOUS,
            AssociationOutcome.UNASSOCIATED,
        ):
            if outcome in outcomes:
                return outcome
        return AssociationOutcome.NOT_COMPARABLE

    def partners_of(self, subject: SubjectRef) -> tuple[SubjectRef, ...]:
        """The subjects positively associated with ``subject``, in canonical order.

        A list of ``(camera_id, track_id)`` references and nothing more. It is
        not an identity, not a group and not a count of people: resolving a set
        of associations into one venue-local temporary subject is a later
        capability that must not be improvised from this.
        """
        return tuple(
            decision.pair.other_than(subject).subject
            for decision in self.decisions_for(subject)
            if decision.is_associated
        )

    def _require_present(self, subject: SubjectRef) -> None:
        if all(observation.subject != subject for observation in self.observations):
            raise AssociationError(
                f"{subject} was not observed at venue second {self.venue_seconds}"
            )

    def to_record(self) -> dict[str, Any]:
        counts = {outcome.value: 0 for outcome in AssociationOutcome}
        for decision in self.decisions:
            counts[decision.outcome.value] += 1
        return {
            "venue_seconds": round(self.venue_seconds, 3),
            "observations": [observation.to_record() for observation in self.observations],
            "outcome_counts": counts,
            "decisions": [decision.to_record() for decision in self.decisions],
        }

"""Domain models for pairwise interaction measurement.

Everything here describes **how two observed subjects relate spatially**, and
nothing here interprets that relation. Two people 0.4 m apart and closing at
1.5 m/s is a measurement. Whether it means anything is a question for a later
layer with evidence this one does not have, and there is deliberately no signal
type for confrontation, aggression, threat, conflict or escalation.

Identity boundary
-----------------
A pair is two **camera-local** subjects. Since a camera-local track id is not a
unique real-world person, a pair is not "two people" — it is two tracklets a
single tracker was following at the same moment. If one of them is lost and
re-acquired under a new id, the pair ends and a different pair begins; nothing
here stitches them together.

Pairs never span cameras. See :class:`SpatialContext` for why.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Self

from atreides_vision.behavior.coordinates import MotionSpace
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.errors import InteractionError

__all__ = [
    "InteractionFrame",
    "InteractionSignal",
    "InteractionSignalType",
    "InteractionSnapshot",
    "PairGeometry",
    "SpatialContext",
    "SubjectPairRef",
    "interaction_unit_for",
]


@dataclass(frozen=True, slots=True)
class SpatialContext:
    """The coordinate frame a set of positions may legitimately be compared in.

    Subtracting two positions only means something when both were produced the
    same way. This makes the preconditions explicit instead of leaving them to
    whoever wrote the call site:

    * **Normalized image space** is camera-local by construction. A frame
      fraction on one camera has no relationship to a frame fraction on another,
      so comparison requires the *same camera*.
    * **Ground-plane metres** are venue-local, so comparison requires the same
      ``venue_id`` *and* the same ``surface_id``. Two positions on the main floor
      and on a balcony can have nearby ``(x, y)`` values and be metres apart
      vertically; treating them as neighbours would be a fabrication.

    V0.3.4 additionally requires the same camera in **both** spaces — see
    :meth:`is_compatible_with`.

    Raises:
        InteractionError: if metric context omits the venue or surface it is
            expressed in. Assuming a shared plane is exactly the silent
            wrongness this type exists to prevent.
    """

    camera_id: str
    space: MotionSpace
    #: Required for metric space, meaningless for normalized.
    venue_id: str | None = None
    surface_id: str | None = None

    def __post_init__(self) -> None:
        if not self.camera_id or not self.camera_id.strip():
            raise InteractionError("SpatialContext.camera_id must be a non-empty string")
        if self.space is MotionSpace.GROUND_PLANE_METRES and (
            not self.venue_id or not self.surface_id
        ):
            raise InteractionError(
                "Metric interaction analysis needs both a venue_id and a surface_id: without "
                "them there is no way to tell whether two positions describe the same physical "
                "plane, and distances between different surfaces would be meaningless."
            )

    @property
    def is_metric(self) -> bool:
        return self.space.is_physical

    def incompatibility_with(self, other: SpatialContext) -> str | None:
        """Why these two contexts cannot be compared, or ``None`` if they can.

        Returns a reason rather than a bare boolean so a caller can say what was
        rejected instead of silently skipping.
        """
        if self.space is not other.space:
            return f"different coordinate spaces ({self.space.value} vs {other.space.value})"
        if self.camera_id != other.camera_id:
            # True for both spaces in V0.3.4. Normalized coordinates are
            # camera-local by nature; metric ones are excluded because without
            # multi-camera association the same person seen twice would be
            # paired with themselves.
            return f"different cameras ({self.camera_id} vs {other.camera_id})"
        if self.space is MotionSpace.GROUND_PLANE_METRES:
            if self.venue_id != other.venue_id:
                return f"different venues ({self.venue_id} vs {other.venue_id})"
            if self.surface_id != other.surface_id:
                return f"different surfaces ({self.surface_id} vs {other.surface_id})"
        return None

    def is_compatible_with(self, other: SpatialContext) -> bool:
        return self.incompatibility_with(other) is None


@dataclass(frozen=True, slots=True)
class SubjectPairRef:
    """Two camera-local subjects, in a canonical order.

    ``(track_7, track_12)`` and ``(track_12, track_7)`` are the same pair, so
    the reference sorts its members at construction. Without that, pair history
    would silently duplicate the moment the tracker returned its tracks in a
    different order, and proximity durations would restart for no reason.

    Both subjects must be on the same camera. Pairing across cameras is
    rejected: with no multi-camera association, the same real person visible in
    two cameras would be paired **with themselves**, producing a confident
    measurement of a relationship that does not exist.

    Raises:
        InteractionError: for a self-pair or a cross-camera pair.
    """

    first: SubjectRef
    second: SubjectRef

    def __init__(self, first: SubjectRef, second: SubjectRef) -> None:
        if first == second:
            raise InteractionError(
                f"A subject cannot interact with itself: {first} paired with {second}"
            )
        if first.camera_id != second.camera_id:
            raise InteractionError(
                f"Cross-camera interaction pairs are not supported: {first} and {second} are on "
                "different cameras. Without multi-camera association, one real person seen by "
                "both cameras would be paired with themselves."
            )
        low, high = sorted((first, second), key=lambda ref: (ref.camera_id, ref.track_id))
        object.__setattr__(self, "first", low)
        object.__setattr__(self, "second", high)

    @property
    def camera_id(self) -> str:
        """The camera both subjects belong to."""
        return self.first.camera_id

    @property
    def track_ids(self) -> tuple[int, int]:
        return (self.first.track_id, self.second.track_id)

    def contains(self, subject: SubjectRef) -> bool:
        return subject in (self.first, self.second)

    def other_than(self, subject: SubjectRef) -> SubjectRef:
        """The member of the pair that is not ``subject``."""
        if subject == self.first:
            return self.second
        if subject == self.second:
            return self.first
        raise InteractionError(f"{subject} is not part of pair {self}")

    def __str__(self) -> str:
        return f"{self.first}+{self.second.track_id}"


class InteractionSignalType(StrEnum):
    """The neutral pairwise quantities V0.3.4 produces.

    Measurements, not conclusions. There is deliberately no member for
    confrontation, aggression, fighting, threat, escalation, harassment or
    stalking: those are interpretations requiring evidence this layer does not
    have, and a measurement package that also judged would make both jobs
    impossible to audit.
    """

    #: Rate at which the gap between the two subjects is shrinking.
    CLOSING_SPEED = "closing_speed"
    #: Angle between the two subjects' directions of travel, in radians.
    RELATIVE_MOTION_HEADING = "relative_motion_heading"
    #: Continuous seconds spent within the configured proximity threshold.
    PROXIMITY_DURATION = "proximity_duration"


#: Unit template per signal. ``{distance}`` is filled from the coordinate space,
#: so the same signal reads ``m/s`` under calibration and
#: ``normalized_frame_unit/s`` without it. A type with no entry raises rather
#: than reporting a wrong unit.
_UNIT_TEMPLATES: Final[Mapping[InteractionSignalType, str]] = {
    InteractionSignalType.CLOSING_SPEED: "{distance}/s",
    InteractionSignalType.RELATIVE_MOTION_HEADING: "rad",
    InteractionSignalType.PROXIMITY_DURATION: "s",
}


def interaction_unit_for(signal_type: InteractionSignalType, space: MotionSpace) -> str:
    """The unit a pairwise signal of this type is measured in."""
    return _UNIT_TEMPLATES[signal_type].format(distance=space.distance_unit)


@dataclass(frozen=True, slots=True)
class InteractionSignal:
    """One measured pairwise quantity, at one moment."""

    signal_type: InteractionSignalType
    value: float
    space: MotionSpace

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise InteractionError(
                f"InteractionSignal.value must be finite, got {self.value!r} "
                f"for {self.signal_type.value}"
            )

    @property
    def unit(self) -> str:
        return interaction_unit_for(self.signal_type, self.space)

    def to_record(self) -> dict[str, Any]:
        return {
            "type": self.signal_type.value,
            "value": round(self.value, 6),
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class PairGeometry:
    """Where the two subjects are relative to each other, on this frame alone.

    Kept apart from the derived signals for the same reason a behaviour
    snapshot separates position from signals: this is an observation about one
    instant, while the signals are statements about a window of frames.
    """

    #: Euclidean separation in the active space's distance units.
    distance: float
    space: MotionSpace

    def __post_init__(self) -> None:
        if not math.isfinite(self.distance) or self.distance < 0:
            raise InteractionError(
                f"PairGeometry.distance must be finite and non-negative, got {self.distance!r}"
            )

    @property
    def distance_unit(self) -> str:
        return self.space.distance_unit

    def to_record(self) -> dict[str, Any]:
        return {
            "distance": {
                "value": round(self.distance, 6),
                "unit": self.distance_unit,
                "space": self.space.value,
            }
        }


@dataclass(frozen=True, slots=True)
class InteractionSnapshot:
    """One pair, at one frame: current geometry plus derived signals."""

    pair: SubjectPairRef
    frame_index: int
    timestamp_seconds: float
    geometry: PairGeometry
    signals: tuple[InteractionSignal, ...] = field(default_factory=tuple)
    #: Zone ids both subjects were inside, in configuration order. Context only
    #: — sharing a zone is a fact about labels, never a claim about proximity.
    shared_zone_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise InteractionError(
                f"InteractionSnapshot.frame_index must be non-negative, got {self.frame_index!r}"
            )

    @property
    def space(self) -> MotionSpace:
        return self.geometry.space

    @property
    def distance(self) -> float:
        return self.geometry.distance

    def value_for(self, signal_type: InteractionSignalType) -> float | None:
        """This pair's value for one signal, or ``None`` if it was not emitted."""
        for signal in self.signals:
            if signal.signal_type is signal_type:
                return signal.value
        return None

    def __len__(self) -> int:
        return len(self.signals)


@dataclass(frozen=True, slots=True)
class InteractionFrame:
    """Every pair measured on one frame, in deterministic pair order."""

    frame_index: int
    timestamp_seconds: float
    snapshots: tuple[InteractionSnapshot, ...] = field(default_factory=tuple)
    #: Pairs considered before compatibility and distance gating. Exposed so a
    #: run can show how much work gating avoided.
    candidate_pairs: int = 0

    @classmethod
    def from_snapshots(
        cls,
        frame_index: int,
        timestamp_seconds: float,
        snapshots: Sequence[InteractionSnapshot],
        candidate_pairs: int = 0,
    ) -> Self:
        return cls(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            snapshots=tuple(snapshots),
            candidate_pairs=candidate_pairs,
        )

    def for_pair(self, pair: SubjectPairRef) -> InteractionSnapshot | None:
        for snapshot in self.snapshots:
            if snapshot.pair == pair:
                return snapshot
        return None

    def __len__(self) -> int:
        return len(self.snapshots)

    def __iter__(self) -> Iterator[InteractionSnapshot]:
        return iter(self.snapshots)

"""Domain models for behaviour signals: who, when, what, and in what unit.

These types are the contract between behaviour measurement and any future
intelligence layer. A consumer of a :class:`BehaviorSignal` can interpret it
completely — value, unit, coordinate space, subject, time and quality — without
knowing that RF-DETR produced the box or that OC-SORT assigned the id. That is
deliberate: detector and tracker stay replaceable.

Identity boundary
-----------------
**A camera-local track id is not a unique real-world person.** It is a handle
on a box that one tracker followed through one camera's view during one
session. The same person walking out of frame and back in may return with a new
id; two cameras hand out the same integers to unrelated people. Every signal
therefore carries a :class:`SubjectRef` — the *(camera, track)* pair — and never
a bare integer.

Nothing here is biometric. No appearance descriptor, face, gait or clothing
feature is computed, stored or referenced. Signals are derived from box
geometry and elapsed time only.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.models import BoundingBox

__all__ = [
    "BehaviorFrame",
    "BehaviorObservation",
    "BehaviorSignal",
    "SignalType",
    "SubjectRef",
    "unit_for",
]


@dataclass(frozen=True, slots=True)
class SubjectRef:
    """The subject a behaviour signal is about: one track, on one camera.

    Both components are required. ``track_id`` alone is ambiguous the moment a
    second camera exists — two cameras routinely assign the same small integers
    to entirely different people — so the pair is the identity used everywhere,
    including as the key of every per-subject store.

    Future multi-camera fusion
    --------------------------
    A later milestone may decide that ``(cam_a, 7)`` and ``(cam_b, 3)`` are the
    same person and give that person a venue-local identity. That layer will sit
    *above* this one and map ``SubjectRef`` values onto venue subjects. It is
    deliberately not a field here: adding one now would imply a fusion
    capability that does not exist, and every signal carrying a permanently
    ``None`` identity column would be a lie by omission. Because signals travel
    with a whole ``SubjectRef`` rather than a raw id, that mapping can be
    introduced without changing any behaviour algorithm.
    """

    camera_id: str
    track_id: int

    def __post_init__(self) -> None:
        if not self.camera_id or not self.camera_id.strip():
            raise ValueError("SubjectRef.camera_id must be a non-empty string")
        if self.track_id < 0:
            raise ValueError(f"SubjectRef.track_id must be non-negative, got {self.track_id!r}")

    def __str__(self) -> str:
        return f"{self.camera_id}/{self.track_id}"


class SignalType(StrEnum):
    """The neutral, measurable quantities V0.3.0 produces.

    Every member is a *measurement*, not a judgement. There is deliberately no
    member for aggression, intoxication, suspicion, distress or threat: those
    are interpretations, they require evidence this milestone does not have,
    and they do not belong in a measurement layer.
    """

    #: Movement magnitude per second in the active coordinate space.
    SPEED = "speed"
    #: Rate of change of :attr:`SPEED`, per second.
    ACCELERATION = "acceleration"
    #: Rate of change of travel direction, in radians per second.
    HEADING_CHANGE_RATE = "heading_change_rate"
    #: Continuous seconds spent below the configured movement threshold.
    STATIONARY_DURATION = "stationary_duration"


#: Unit template per signal type. ``{distance}`` is filled from the coordinate
#: space, which is what lets the same signal read "normalized_frame_unit/s"
#: today and "m/s" once a calibrated projector exists. A signal type without an
#: entry raises immediately rather than silently reporting a wrong unit.
_UNIT_TEMPLATES: Final[Mapping[SignalType, str]] = {
    SignalType.SPEED: "{distance}/s",
    SignalType.ACCELERATION: "{distance}/s^2",
    SignalType.HEADING_CHANGE_RATE: "rad/s",
    SignalType.STATIONARY_DURATION: "s",
}


def unit_for(signal_type: SignalType, space: MotionSpace) -> str:
    """The unit a signal of this type is measured in, within ``space``.

    Angles and durations are space-independent; distances are not.
    """
    return _UNIT_TEMPLATES[signal_type].format(distance=space.distance_unit)


@dataclass(frozen=True, slots=True)
class BehaviorObservation:
    """One subject, seen at one frame, reduced to what kinematics needs.

    This is the only thing retained per frame. The frame's pixels are never
    kept, and the bounding box is optional context rather than an input to any
    signal.
    """

    subject: SubjectRef
    frame_index: int
    timestamp_seconds: float
    #: The subject's anchor, already projected into a motion space.
    position: MotionPoint
    #: The source box, kept for diagnostics and for future signals that need
    #: apparent size. No current signal reads it.
    bbox: BoundingBox | None = None
    #: Tracker confidence where the tracker reports one.
    confidence: float | None = None
    #: Ids of any zones the subject was inside, as opaque strings supplied by
    #: the caller. The behaviour layer never interprets them and does not
    #: import the zone layer, so zones stay optional and independent.
    zone_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError(
                f"BehaviorObservation.frame_index must be non-negative, got {self.frame_index!r}"
            )
        if not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise ValueError(
                "BehaviorObservation.timestamp_seconds must be finite and non-negative, "
                f"got {self.timestamp_seconds!r}"
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"BehaviorObservation.confidence must be in [0, 1], got {self.confidence!r}"
            )

    @property
    def space(self) -> MotionSpace:
        return self.position.space


@dataclass(frozen=True, slots=True)
class BehaviorSignal:
    """One measured quantity, about one subject, at one moment.

    Carries enough context to be interpreted standalone by a downstream layer:
    what was measured, about whom, when, how much, in what unit, and how much
    evidence it rests on.

    Fields were chosen for what this milestone can actually justify. Anything a
    future risk engine might want — smoothed aggregates, per-venue baselines,
    cross-subject comparisons — belongs to that engine, built from streams of
    these, not bolted on here.
    """

    signal_type: SignalType
    subject: SubjectRef
    frame_index: int
    timestamp_seconds: float
    value: float
    #: The coordinate space the measurement was taken in. For durations this
    #: records the space the movement threshold was applied in, so a duration
    #: measured against normalized-image motion is never mistaken for one
    #: measured against calibrated metres.
    space: MotionSpace
    #: How much of the configured window the estimate rests on, in ``[0, 1]``.
    #: ``1.0`` means a full window of observations; ``0.2`` means a fifth of
    #: one, typically a freshly acquired track. Not a probability, and not a
    #: statement that the measurement is *right* — only how much data it saw.
    quality: float | None = None
    #: Zone context copied from the observation, if the caller supplied any.
    zone_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError(
                f"BehaviorSignal.value must be finite, got {self.value!r} "
                f"for {self.signal_type.value}"
            )
        if self.frame_index < 0:
            raise ValueError(
                f"BehaviorSignal.frame_index must be non-negative, got {self.frame_index!r}"
            )
        if self.quality is not None and not 0.0 <= self.quality <= 1.0:
            raise ValueError(f"BehaviorSignal.quality must be in [0, 1], got {self.quality!r}")

    @property
    def unit(self) -> str:
        """The unit of :attr:`value`, derived from type and space."""
        return unit_for(self.signal_type, self.space)

    def to_record(self) -> dict[str, Any]:
        """A compact JSON-serialisable record.

        The unit and space travel with the value on purpose: a consumer must
        never have to guess whether a number is normalized-image or metric.
        """
        record: dict[str, Any] = {
            "type": self.signal_type.value,
            "camera_id": self.subject.camera_id,
            "track_id": self.subject.track_id,
            "frame": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "value": round(self.value, 6),
            "unit": self.unit,
            "space": self.space.value,
        }
        if self.quality is not None:
            record["quality"] = round(self.quality, 3)
        if self.zone_ids:
            record["zone_ids"] = list(self.zone_ids)
        return record


@dataclass(frozen=True, slots=True)
class BehaviorFrame:
    """Every signal produced for one frame, in stable subject order."""

    frame_index: int
    timestamp_seconds: float
    signals: tuple[BehaviorSignal, ...] = field(default_factory=tuple)

    def for_subject(self, subject: SubjectRef) -> tuple[BehaviorSignal, ...]:
        return tuple(signal for signal in self.signals if signal.subject == subject)

    def of_type(self, signal_type: SignalType) -> tuple[BehaviorSignal, ...]:
        return tuple(signal for signal in self.signals if signal.signal_type is signal_type)

    def value_for(self, subject: SubjectRef, signal_type: SignalType) -> float | None:
        """The value of one signal for one subject, or ``None`` if not emitted."""
        for signal in self.signals:
            if signal.subject == subject and signal.signal_type is signal_type:
                return signal.value
        return None

    @property
    def subjects(self) -> tuple[SubjectRef, ...]:
        """Distinct subjects with at least one signal this frame, in order."""
        seen: dict[SubjectRef, None] = {}
        for signal in self.signals:
            seen.setdefault(signal.subject, None)
        return tuple(seen)

    def to_records(self) -> list[dict[str, Any]]:
        return [signal.to_record() for signal in self.signals]

    def __len__(self) -> int:
        return len(self.signals)

    def __iter__(self) -> Iterator[BehaviorSignal]:
        return iter(self.signals)

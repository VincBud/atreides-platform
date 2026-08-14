"""Subject references, observations, signals and their units."""

from __future__ import annotations

import pytest

from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.behavior.models import (
    BehaviorFrame,
    BehaviorObservation,
    BehaviorSignal,
    SignalType,
    SubjectRef,
    unit_for,
)
from atreides_vision.models import BoundingBox

NORMALIZED = MotionSpace.NORMALIZED_IMAGE
GROUND = MotionSpace.GROUND_PLANE_METRES


def position(x: float = 0.5, y: float = 0.5) -> MotionPoint:
    return MotionPoint(x, y, NORMALIZED)


def observation(
    subject: SubjectRef,
    frame_index: int = 1,
    timestamp: float = 0.0,
    **kwargs: object,
) -> BehaviorObservation:
    return BehaviorObservation(
        subject=subject,
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        position=position(),
        **kwargs,  # type: ignore[arg-type]
    )


def signal(
    subject: SubjectRef,
    signal_type: SignalType = SignalType.SPEED,
    value: float = 0.1,
    **kwargs: object,
) -> BehaviorSignal:
    return BehaviorSignal(
        signal_type=signal_type,
        subject=subject,
        frame_index=1,
        timestamp_seconds=0.0,
        value=value,
        space=NORMALIZED,
        **kwargs,  # type: ignore[arg-type]
    )


class TestSubjectRef:
    def test_carries_camera_and_track(self) -> None:
        subject = SubjectRef("cam_01", 7)
        assert subject.camera_id == "cam_01"
        assert subject.track_id == 7

    def test_same_track_id_on_two_cameras_is_two_subjects(self) -> None:
        """The reason a bare track id is never used as an identity."""
        assert SubjectRef("cam_a", 7) != SubjectRef("cam_b", 7)
        assert len({SubjectRef("cam_a", 7), SubjectRef("cam_b", 7)}) == 2

    def test_is_hashable_and_usable_as_a_key(self) -> None:
        store = {SubjectRef("cam_a", 1): "a", SubjectRef("cam_a", 2): "b"}
        assert store[SubjectRef("cam_a", 1)] == "a"

    def test_equal_refs_are_the_same_key(self) -> None:
        assert SubjectRef("cam_a", 7) == SubjectRef("cam_a", 7)
        assert hash(SubjectRef("cam_a", 7)) == hash(SubjectRef("cam_a", 7))

    def test_is_immutable(self) -> None:
        subject = SubjectRef("cam_01", 7)
        with pytest.raises(AttributeError):
            subject.track_id = 9  # type: ignore[misc]

    def test_readable_string_form(self) -> None:
        assert str(SubjectRef("cam_01", 7)) == "cam_01/7"

    @pytest.mark.parametrize("camera_id", ["", "   "])
    def test_rejects_empty_camera_id(self, camera_id: str) -> None:
        with pytest.raises(ValueError, match="camera_id must be a non-empty string"):
            SubjectRef(camera_id, 1)

    def test_rejects_negative_track_id(self) -> None:
        with pytest.raises(ValueError, match="track_id must be non-negative"):
            SubjectRef("cam_01", -1)

    def test_has_no_venue_identity_field(self) -> None:
        """V0.3.0 does no cross-camera fusion, so it claims none.

        A permanently-``None`` venue identity column would imply a capability
        that does not exist. Fusion, when it arrives, maps SubjectRefs from
        above rather than adding a field here.
        """
        assert set(SubjectRef.__dataclass_fields__) == {"camera_id", "track_id"}


class TestSignalUnits:
    @pytest.mark.parametrize("signal_type", list(SignalType))
    def test_every_signal_type_has_a_unit_in_every_space(self, signal_type: SignalType) -> None:
        for space in MotionSpace:
            assert unit_for(signal_type, space)

    def test_distance_units_follow_the_space(self) -> None:
        """The same code reads metres once a calibrated projector exists."""
        assert unit_for(SignalType.SPEED, NORMALIZED) == "normalized_frame_unit/s"
        assert unit_for(SignalType.SPEED, GROUND) == "m/s"
        assert unit_for(SignalType.ACCELERATION, NORMALIZED) == "normalized_frame_unit/s^2"
        assert unit_for(SignalType.ACCELERATION, GROUND) == "m/s^2"

    def test_angles_and_durations_are_space_independent(self) -> None:
        for space in MotionSpace:
            assert unit_for(SignalType.HEADING_CHANGE_RATE, space) == "rad/s"
            assert unit_for(SignalType.STATIONARY_DURATION, space) == "s"

    def test_no_semantic_signal_types_exist(self) -> None:
        """V0.3.0 measures; it does not judge."""
        assert {member.value for member in SignalType} == {
            "speed",
            "acceleration",
            "heading_change_rate",
            "stationary_duration",
        }


class TestBehaviorObservation:
    def test_reports_its_space(self) -> None:
        assert observation(SubjectRef("cam_01", 1)).space is NORMALIZED

    def test_optional_context_defaults_to_empty(self) -> None:
        record = observation(SubjectRef("cam_01", 1))
        assert record.bbox is None
        assert record.confidence is None
        assert record.zone_ids == ()

    def test_carries_optional_context(self) -> None:
        record = observation(
            SubjectRef("cam_01", 1),
            bbox=BoundingBox(0.0, 0.0, 10.0, 20.0),
            confidence=0.8,
            zone_ids=("bar", "lounge"),
        )
        assert record.confidence == 0.8
        assert record.zone_ids == ("bar", "lounge")

    def test_rejects_negative_frame_index(self) -> None:
        with pytest.raises(ValueError, match="frame_index must be non-negative"):
            observation(SubjectRef("cam_01", 1), frame_index=-1)

    @pytest.mark.parametrize("timestamp", [-0.1, float("nan"), float("inf")])
    def test_rejects_invalid_timestamps(self, timestamp: float) -> None:
        with pytest.raises(ValueError, match="timestamp_seconds must be finite"):
            observation(SubjectRef("cam_01", 1), timestamp=timestamp)

    @pytest.mark.parametrize("confidence", [-0.1, 1.1])
    def test_rejects_out_of_range_confidence(self, confidence: float) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            observation(SubjectRef("cam_01", 1), confidence=confidence)


class TestBehaviorSignal:
    def test_derives_its_unit(self) -> None:
        assert signal(SubjectRef("cam_01", 1)).unit == "normalized_frame_unit/s"

    def test_rejects_non_finite_values(self) -> None:
        with pytest.raises(ValueError, match="value must be finite"):
            signal(SubjectRef("cam_01", 1), value=float("inf"))

    @pytest.mark.parametrize("quality", [-0.01, 1.01])
    def test_rejects_out_of_range_quality(self, quality: float) -> None:
        with pytest.raises(ValueError, match="quality must be in"):
            signal(SubjectRef("cam_01", 1), quality=quality)

    def test_record_carries_unit_and_space(self) -> None:
        """A consumer must never have to guess whether a value is metric."""
        record = signal(SubjectRef("cam_01", 7), value=0.1234567, quality=0.5).to_record()
        assert record["type"] == "speed"
        assert record["camera_id"] == "cam_01"
        assert record["track_id"] == 7
        assert record["unit"] == "normalized_frame_unit/s"
        assert record["space"] == "normalized_image"
        assert record["value"] == pytest.approx(0.123457)
        assert record["quality"] == 0.5

    def test_record_omits_absent_optionals(self) -> None:
        record = signal(SubjectRef("cam_01", 7)).to_record()
        assert "quality" not in record
        assert "zone_ids" not in record

    def test_record_includes_zone_context_when_present(self) -> None:
        record = signal(SubjectRef("cam_01", 7), zone_ids=("bar",)).to_record()
        assert record["zone_ids"] == ["bar"]


class TestBehaviorFrame:
    def test_queries_by_subject_and_type(self) -> None:
        one, two = SubjectRef("cam_01", 1), SubjectRef("cam_01", 2)
        frame = BehaviorFrame(
            frame_index=4,
            timestamp_seconds=0.16,
            signals=(
                signal(one, SignalType.SPEED, 0.1),
                signal(one, SignalType.ACCELERATION, 0.2),
                signal(two, SignalType.SPEED, 0.3),
            ),
        )
        assert len(frame) == 3
        assert len(frame.for_subject(one)) == 2
        assert len(frame.of_type(SignalType.SPEED)) == 2
        assert frame.value_for(two, SignalType.SPEED) == pytest.approx(0.3)
        assert frame.value_for(two, SignalType.ACCELERATION) is None
        assert frame.subjects == (one, two)

    def test_empty_frame(self) -> None:
        frame = BehaviorFrame(frame_index=1, timestamp_seconds=0.0)
        assert len(frame) == 0
        assert frame.subjects == ()
        assert frame.to_records() == []

    def test_is_iterable(self) -> None:
        subject = SubjectRef("cam_01", 1)
        frame = BehaviorFrame(1, 0.0, (signal(subject),))
        assert [item.subject for item in frame] == [subject]

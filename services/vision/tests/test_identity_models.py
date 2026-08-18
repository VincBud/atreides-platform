"""The identity domain types: an opaque id, a membership, and a valid subject.

Every test here exists to stop a :class:`VenueSubject` being constructed in a
state its invariants forbid: members out of canonical order, two active
tracklets on one camera, time running backwards.
"""

from __future__ import annotations

import pytest

from atreides_vision.behavior.models import SubjectRef
from atreides_vision.errors import IdentityError
from atreides_vision.identity import (
    MembershipAction,
    MembershipEvent,
    TrackletMembership,
    VenueSubject,
    VenueSubjectId,
    tracklet_key,
)
from atreides_vision.timing import StreamRef
from identity_builders import membership


class TestVenueSubjectId:
    def test_sequence_must_be_positive(self) -> None:
        with pytest.raises(IdentityError, match="sequence"):
            VenueSubjectId(0)

    def test_string_form_is_opaque_and_not_the_raw_sequence(self) -> None:
        assert str(VenueSubjectId(7)) == "venue_subject_000007"

    def test_two_ids_with_the_same_sequence_are_equal(self) -> None:
        assert VenueSubjectId(3) == VenueSubjectId(3)

    def test_ids_order_by_sequence(self) -> None:
        assert VenueSubjectId(1) < VenueSubjectId(2)

    def test_to_record_is_the_string_form(self) -> None:
        assert VenueSubjectId(5).to_record() == str(VenueSubjectId(5))


class TestTrackletKey:
    def test_key_is_camera_epoch_track(self) -> None:
        key = tracklet_key(SubjectRef("cam_a", 1), StreamRef("cam_a", 2))
        assert key == ("cam_a", 2, 1)

    def test_a_subject_and_stream_naming_different_cameras_is_refused(self) -> None:
        with pytest.raises(IdentityError, match="does not match"):
            tracklet_key(SubjectRef("cam_a", 1), StreamRef("cam_b", 0))


class TestTrackletMembership:
    def test_camera_mismatch_between_subject_and_stream_is_refused(self) -> None:
        with pytest.raises(IdentityError, match="does not match"):
            TrackletMembership(
                subject=SubjectRef("cam_a", 1),
                stream=StreamRef("cam_b", 0),
                joined_at_seconds=1.0,
            )

    def test_negative_joined_at_is_refused(self) -> None:
        with pytest.raises(IdentityError, match="joined_at_seconds"):
            membership(joined_at_seconds=-1.0)

    def test_tracklet_key_matches_the_module_function(self) -> None:
        m = membership("cam_a", 3, stream_epoch=1)
        assert m.tracklet_key == ("cam_a", 1, 3)

    def test_to_record_carries_camera_track_epoch_and_time(self) -> None:
        m = membership("cam_a", 3, stream_epoch=1, joined_at_seconds=12.5)
        assert m.to_record() == {
            "camera_id": "cam_a",
            "track_id": 3,
            "stream_epoch": 1,
            "joined_at_seconds": 12.5,
        }


class TestMembershipEvent:
    def test_negative_at_seconds_is_refused(self) -> None:
        with pytest.raises(IdentityError, match="at_seconds"):
            MembershipEvent(membership(), MembershipAction.ADDED, -1.0)

    def test_to_record_includes_the_action(self) -> None:
        event = MembershipEvent(membership(), MembershipAction.REMOVED, 5.0)
        record = event.to_record()
        assert record["action"] == "removed"
        assert record["at_seconds"] == 5.0


class TestVenueSubjectInvariants:
    def test_a_single_member_subject_is_valid(self) -> None:
        subject = VenueSubject(
            subject_id=VenueSubjectId(1),
            created_at_seconds=10.0,
            last_activity_seconds=10.0,
            members=(membership("cam_a", 1),),
        )
        assert subject.members == (membership("cam_a", 1),)
        assert not subject.is_retired

    def test_last_activity_before_created_at_is_refused(self) -> None:
        with pytest.raises(IdentityError, match="last_activity_seconds"):
            VenueSubject(
                subject_id=VenueSubjectId(1),
                created_at_seconds=10.0,
                last_activity_seconds=5.0,
                members=(membership("cam_a", 1),),
            )

    def test_retiring_before_last_activity_is_refused(self) -> None:
        with pytest.raises(IdentityError, match="retired"):
            VenueSubject(
                subject_id=VenueSubjectId(1),
                created_at_seconds=10.0,
                last_activity_seconds=10.0,
                members=(membership("cam_a", 1),),
                retired_at_seconds=5.0,
            )

    def test_members_out_of_canonical_order_are_refused(self) -> None:
        out_of_order = (
            membership("cam_b", 1, joined_at_seconds=10.0),
            membership("cam_a", 1, joined_at_seconds=10.0),
        )
        with pytest.raises(IdentityError, match="canonical order"):
            VenueSubject(
                subject_id=VenueSubjectId(1),
                created_at_seconds=10.0,
                last_activity_seconds=10.0,
                members=out_of_order,
            )

    def test_two_active_members_on_the_same_camera_are_refused(self) -> None:
        same_camera = (
            membership("cam_a", 1, stream_epoch=0, joined_at_seconds=10.0),
            membership("cam_a", 2, stream_epoch=0, joined_at_seconds=10.0),
        )
        with pytest.raises(IdentityError, match="more than one active tracklet"):
            VenueSubject(
                subject_id=VenueSubjectId(1),
                created_at_seconds=10.0,
                last_activity_seconds=10.0,
                members=tuple(sorted(same_camera, key=lambda m: m.tracklet_key)),
            )

    def test_camera_ids_are_the_sorted_distinct_cameras(self) -> None:
        members = tuple(
            sorted(
                (
                    membership("cam_b", 1, joined_at_seconds=10.0),
                    membership("cam_a", 1, joined_at_seconds=10.0),
                ),
                key=lambda m: m.tracklet_key,
            )
        )
        subject = VenueSubject(
            subject_id=VenueSubjectId(1),
            created_at_seconds=10.0,
            last_activity_seconds=10.0,
            members=members,
        )
        assert subject.camera_ids == ("cam_a", "cam_b")

    def test_member_on_returns_none_for_an_absent_camera(self) -> None:
        subject = VenueSubject(
            subject_id=VenueSubjectId(1),
            created_at_seconds=10.0,
            last_activity_seconds=10.0,
            members=(membership("cam_a", 1),),
        )
        assert subject.member_on("cam_z") is None

    def test_to_record_is_serialization_ready(self) -> None:
        subject = VenueSubject(
            subject_id=VenueSubjectId(1),
            created_at_seconds=10.0,
            last_activity_seconds=10.0,
            members=(membership("cam_a", 1),),
        )
        record = subject.to_record()
        assert record["venue_subject_id"] == "venue_subject_000001"
        assert record["retired_at_seconds"] is None
        assert len(record["members"]) == 1

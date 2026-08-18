"""The venue subject registry: allocation, membership, provenance, lifecycle.

Every test in ``TestConflictsAreRefused`` exists to prove the registry never
silently reconciles two claims on one tracklet or two active tracklets on one
camera — the wrong answer is always a refusal, never a guess.
"""

from __future__ import annotations

import pytest

from atreides_vision.behavior.models import SubjectRef
from atreides_vision.errors import IdentityError
from atreides_vision.identity import MembershipAction, VenueSubjectId, VenueSubjectRegistry
from identity_builders import CONFIG, membership


class TestAllocation:
    def test_the_first_subject_gets_sequence_one(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1))
        assert subject_id == VenueSubjectId(1)

    def test_allocation_is_monotonic_regardless_of_camera(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        first = registry.create(membership("cam_b", 9))
        second = registry.create(membership("cam_a", 1))
        assert (first, second) == (VenueSubjectId(1), VenueSubjectId(2))

    def test_a_venue_subject_id_is_not_the_track_id(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 999))
        assert subject_id.sequence != 999

    def test_single_camera_identity_works_the_same_way(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_only", 1))
        subject = registry.get(subject_id)
        assert subject.camera_ids == ("cam_only",)


class TestMembership:
    def test_a_second_camera_can_be_added(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))
        registry.add_membership(subject_id, membership("cam_b", 2, joined_at_seconds=10.5))

        subject = registry.get(subject_id)
        assert subject.camera_ids == ("cam_a", "cam_b")
        assert subject.last_activity_seconds == 10.5

    def test_membership_can_be_removed_without_retiring_the_subject(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))
        registry.add_membership(subject_id, membership("cam_b", 2, joined_at_seconds=10.5))

        registry.remove_membership(subject_id, SubjectRef("cam_b", 2), 11.0)

        subject = registry.get(subject_id)
        assert subject.camera_ids == ("cam_a",)
        assert not subject.is_retired

    def test_removing_a_tracklet_releases_its_claim(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))
        registry.remove_membership(subject_id, SubjectRef("cam_a", 1), 11.0)

        assert registry.owner_of(("cam_a", 0, 1)) is None

    def test_removing_a_tracklet_that_is_not_a_member_is_refused(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1))

        with pytest.raises(IdentityError, match="not an active member"):
            registry.remove_membership(subject_id, SubjectRef("cam_z", 9), 11.0)


class TestConflictsAreRefused:
    def test_creating_with_an_already_claimed_tracklet_is_refused(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        registry.create(membership("cam_a", 1))

        with pytest.raises(IdentityError, match="already a member"):
            registry.create(membership("cam_a", 1))

    def test_adding_a_tracklet_already_owned_by_another_subject_is_refused(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        owner = registry.create(membership("cam_a", 1))
        other = registry.create(membership("cam_b", 2))

        with pytest.raises(IdentityError, match="already a member"):
            registry.add_membership(other, membership("cam_a", 1))
        assert registry.owner_of(("cam_a", 0, 1)) == owner

    def test_a_second_active_tracklet_on_the_same_camera_is_refused(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))

        with pytest.raises(IdentityError, match="already has an active tracklet"):
            registry.add_membership(subject_id, membership("cam_a", 2, joined_at_seconds=10.5))

    def test_after_removal_the_same_camera_may_be_added_again(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))
        registry.remove_membership(subject_id, SubjectRef("cam_a", 1), 11.0)

        registry.add_membership(subject_id, membership("cam_a", 2, joined_at_seconds=12.0))

        assert registry.get(subject_id).camera_ids == ("cam_a",)

    def test_backwards_venue_time_on_add_is_refused(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))

        with pytest.raises(IdentityError, match="backwards"):
            registry.add_membership(subject_id, membership("cam_b", 2, joined_at_seconds=9.0))

    def test_backwards_venue_time_on_remove_is_refused(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))
        registry.add_membership(subject_id, membership("cam_b", 2, joined_at_seconds=10.5))

        with pytest.raises(IdentityError, match="backwards"):
            registry.remove_membership(subject_id, SubjectRef("cam_b", 2), 9.0)

    def test_operating_on_an_unknown_subject_is_refused(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)

        with pytest.raises(IdentityError, match="not registered"):
            registry.add_membership(VenueSubjectId(1), membership("cam_a", 1))

    def test_operating_on_a_retired_subject_is_refused(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))
        registry.retire(subject_id, 10.0)

        with pytest.raises(IdentityError, match="retired"):
            registry.add_membership(subject_id, membership("cam_b", 2, joined_at_seconds=11.0))

    def test_retiring_twice_is_refused(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))
        registry.retire(subject_id, 10.0)

        with pytest.raises(IdentityError, match="retired"):
            registry.retire(subject_id, 11.0)


class TestProvenance:
    def test_provenance_survives_removal(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))
        registry.remove_membership(subject_id, SubjectRef("cam_a", 1), 11.0)

        events = registry.provenance_for(subject_id)
        assert [e.action for e in events] == [MembershipAction.ADDED, MembershipAction.REMOVED]
        assert events[0].membership.subject == SubjectRef("cam_a", 1)

    def test_provenance_for_an_unknown_subject_is_refused(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)

        with pytest.raises(IdentityError, match="not registered"):
            registry.provenance_for(VenueSubjectId(1))

    def test_provenance_is_chronological(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))
        registry.add_membership(subject_id, membership("cam_b", 2, joined_at_seconds=10.5))
        registry.add_membership(subject_id, membership("cam_c", 3, joined_at_seconds=11.0))

        seconds = [e.at_seconds for e in registry.provenance_for(subject_id)]
        assert seconds == sorted(seconds)


class TestRetirement:
    def test_retiring_releases_all_camera_claims(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))
        registry.add_membership(subject_id, membership("cam_b", 2, joined_at_seconds=10.5))

        registry.retire(subject_id, 11.0)

        assert registry.owner_of(("cam_a", 0, 1)) is None
        assert registry.owner_of(("cam_b", 0, 2)) is None

    def test_a_retired_subject_still_reports_its_last_active_members(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))
        registry.retire(subject_id, 11.0)

        subject = registry.get(subject_id)
        assert subject.is_retired
        assert subject.camera_ids == ("cam_a",)

    def test_a_freed_tracklet_can_be_claimed_by_a_new_subject(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        first = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))
        registry.retire(first, 10.0)

        second = registry.create(membership("cam_a", 1, joined_at_seconds=12.0))

        assert registry.owner_of(("cam_a", 0, 1)) == second

    def test_active_subjects_excludes_retired_ones(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        active = registry.create(membership("cam_a", 1))
        retired = registry.create(membership("cam_b", 2))
        registry.retire(retired, 10.0)

        assert registry.active_subjects() == (active,)
        assert registry.subjects() == (active, retired)


class TestExpiry:
    def test_an_idle_active_subject_is_auto_retired(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))

        registry.expire(10.0 + CONFIG.idle_expiry_seconds + 0.1)

        assert registry.get(subject_id).is_retired

    def test_a_recently_active_subject_is_not_expired(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))

        registry.expire(10.0 + CONFIG.idle_expiry_seconds - 1.0)

        assert not registry.get(subject_id).is_retired

    def test_a_long_retired_subject_is_purged(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))
        registry.retire(subject_id, 10.0)

        purged = registry.expire(10.0 + CONFIG.retention_seconds + 0.1)

        assert purged == (subject_id,)
        assert subject_id not in registry
        with pytest.raises(IdentityError, match="not registered"):
            registry.provenance_for(subject_id)

    def test_a_recently_retired_subject_is_not_purged(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))
        registry.retire(subject_id, 10.0)

        purged = registry.expire(10.0 + CONFIG.retention_seconds - 1.0)

        assert purged == ()
        assert subject_id in registry

    def test_expiry_is_based_only_on_venue_time_not_call_count(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))

        for _ in range(50):
            registry.expire(10.0)

        assert not registry.get(subject_id).is_retired


class TestCapacity:
    def test_creating_past_the_ceiling_is_refused(self) -> None:
        tiny = CONFIG.__class__(
            idle_expiry_seconds=CONFIG.idle_expiry_seconds,
            retention_seconds=CONFIG.retention_seconds,
            max_venue_subjects=1,
        )
        registry = VenueSubjectRegistry(tiny)
        registry.create(membership("cam_a", 1))

        with pytest.raises(IdentityError, match="capacity"):
            registry.create(membership("cam_b", 2))

    def test_churning_many_subjects_never_exceeds_the_ceiling(self) -> None:
        """Repeated create/retire/expire keeps the registry bounded under churn."""
        registry = VenueSubjectRegistry(CONFIG)
        now = 10.0
        step = CONFIG.retention_seconds + 1.0
        for index in range(500):
            subject_id = registry.create(membership(f"cam_{index}", 1, joined_at_seconds=now))
            registry.retire(subject_id, now)
            now += step
            registry.expire(now)
            assert len(registry) <= CONFIG.max_venue_subjects

    def test_expiring_after_capacity_is_reached_frees_room_to_create_again(self) -> None:
        tiny = CONFIG.__class__(
            idle_expiry_seconds=CONFIG.idle_expiry_seconds,
            retention_seconds=1.0,
            max_venue_subjects=1,
        )
        registry = VenueSubjectRegistry(tiny)
        first = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))
        registry.retire(first, 10.0)

        registry.expire(12.0)
        second = registry.create(membership("cam_b", 2, joined_at_seconds=12.0))

        assert second == VenueSubjectId(2)


class TestDeterminism:
    def test_subjects_come_back_in_canonical_order_not_insertion_order(self) -> None:
        registry = VenueSubjectRegistry(CONFIG)
        first = registry.create(membership("cam_z", 1))
        second = registry.create(membership("cam_a", 2))

        assert registry.subjects() == (first, second)

    def test_membership_order_does_not_depend_on_camera_enumeration_order(self) -> None:
        forward = VenueSubjectRegistry(CONFIG)
        s1 = forward.create(membership("cam_a", 1, joined_at_seconds=10.0))
        forward.add_membership(s1, membership("cam_b", 2, joined_at_seconds=10.0))
        forward.add_membership(s1, membership("cam_c", 3, joined_at_seconds=10.0))

        backward = VenueSubjectRegistry(CONFIG)
        s2 = backward.create(membership("cam_a", 1, joined_at_seconds=10.0))
        backward.add_membership(s2, membership("cam_c", 3, joined_at_seconds=10.0))
        backward.add_membership(s2, membership("cam_b", 2, joined_at_seconds=10.0))

        assert forward.get(s1).to_record()["members"] == backward.get(s2).to_record()["members"]

    def test_two_runs_of_the_same_operations_produce_byte_identical_records(self) -> None:
        def run() -> dict[str, object]:
            registry = VenueSubjectRegistry(CONFIG)
            subject_id = registry.create(membership("cam_a", 1, joined_at_seconds=10.0))
            registry.add_membership(subject_id, membership("cam_b", 2, joined_at_seconds=10.5))
            registry.retire(subject_id, 11.0)
            return registry.get(subject_id).to_record()

        assert run() == run()

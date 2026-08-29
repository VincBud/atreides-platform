"""V0.4.9b-C — the generic, read-only validation observer seam.

The seam has exactly one job: after the runtime has finished deciding a venue
instant, hand a later offline evaluator an immutable, factual snapshot of what
the runtime produced. These tests prove it observes and never participates —
the artifacts are byte-identical with and without an observer, an observer that
returns something changes nothing, and an observer that raises fails the run
the way any other layer would.

The camera-local pipeline, ``VenuePipeline``, occupancy and both writers are
the real ones; only the detector, tracker and source seams are faked, exactly
as :mod:`test_venue_validation_runner` does. This module reuses that module's
synthetic two-camera fixtures rather than rebuilding them.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from association_builders import observation as make_observation
from atreides_vision.association.models import (
    AssociationRound,
    TrackObservation,
    observation_order_key,
    tracklet_key,
)
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.errors import ValidationError
from atreides_vision.fusion.models import IntegrationResult
from atreides_vision.identity.models import VenueSubjectId
from atreides_vision.occupancy.models import OccupancyContribution, OccupancySnapshot
from atreides_vision.venue.manifest import VenueCamera
from atreides_vision.venue_pipeline import VenueInstantResult
from atreides_vision.venue_validation import (
    VenueValidationConfig,
    VenueValidationSummary,
    run_venue_validation,
)
from atreides_vision.venue_validation import observer as observer_module
from atreides_vision.venue_validation.observer import (
    VenueSubjectMembershipSnapshot,
    VenueValidationInstantSnapshot,
    VenueValidationObserver,
    build_instant_snapshot,
)
from atreides_vision.video.source import VideoSource
from fakes import FakeTracker
from test_venue_validation_runner import (
    CAM_A,
    CAM_B,
    DIMS_A,
    ShapeDetector,
    camera_entry,
    manifest_for,
    records,
    source_for,
    standing_detector,
    two_camera_setup,
)


class RecordingObserver:
    """Keeps every snapshot it is handed, in call order."""

    def __init__(self) -> None:
        self.snapshots: list[VenueValidationInstantSnapshot] = []

    def __call__(self, snapshot: VenueValidationInstantSnapshot, /) -> None:
        self.snapshots.append(snapshot)


class MeddlingObserver:
    """Returns a value and mutates its own state — neither may reach the run."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, snapshot: VenueValidationInstantSnapshot, /) -> object:
        self.calls += 1
        return {"replace": "everything", "occupancy": 999}


class ExplodingObserver:
    def __init__(self, *, fail_on: int = 1) -> None:
        self.calls = 0
        self._fail_on = fail_on

    def __call__(self, snapshot: VenueValidationInstantSnapshot, /) -> None:
        self.calls += 1
        if self.calls >= self._fail_on:
            raise RuntimeError("observer failed deliberately")


def run_observed(
    config: VenueValidationConfig,
    sources: Mapping[Path, VideoSource],
    detector: ShapeDetector,
    observer: VenueValidationObserver | None,
) -> VenueValidationSummary:
    return run_venue_validation(
        config,
        detector=detector,
        tracker_factory=lambda _config: FakeTracker(),
        source_factory=lambda entry: sources[entry.input_path],
        observer=observer,
    )


def subject_refs(snapshot: VenueValidationInstantSnapshot) -> set[tuple[str, int, int]]:
    owned = {key for membership in snapshot.memberships for key in membership.members}
    unowned = set(snapshot.unowned_tracklets)
    return owned | unowned


def observed_subjects(snapshot: VenueValidationInstantSnapshot) -> set[tuple[str, int, int]]:
    return {tracklet_key(o) for o in snapshot.observations}


def digest(snapshots: list[VenueValidationInstantSnapshot]) -> list[object]:
    return [
        (
            snap.venue_seconds,
            tuple(observation_order_key(o) for o in snap.observations),
            tuple(
                (str(membership.subject_id), tuple(membership.members))
                for membership in snap.memberships
            ),
            tuple(snap.unowned_tracklets),
        )
        for snap in snapshots
    ]


class TestOptionalAndAdditive:
    def test_the_observer_defaults_to_none_and_existing_callers_are_unchanged(
        self, tmp_path: Path
    ) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=4, frames_b=4)
        summary = run_venue_validation(
            config,
            detector=detector,
            tracker_factory=lambda _config: FakeTracker(),
            source_factory=lambda entry: sources[entry.input_path],
        )
        assert summary.venue_instants == 4

    def test_observer_none_produces_byte_identical_artifacts(self, tmp_path: Path) -> None:
        plain_config, plain_sources, plain_detector = two_camera_setup(
            tmp_path, output_dir=tmp_path / "plain"
        )
        run_observed(plain_config, plain_sources, plain_detector, None)

        observed_config, observed_sources, observed_detector = two_camera_setup(
            tmp_path, output_dir=tmp_path / "observed"
        )
        run_observed(observed_config, observed_sources, observed_detector, RecordingObserver())

        for name in ("association_report_path", "occupancy_report_path", "summary_path"):
            assert (
                getattr(plain_config, name).read_bytes()
                == getattr(observed_config, name).read_bytes()
            )


class TestInvocation:
    def test_the_observer_is_called_exactly_once_per_venue_instant(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=7, frames_b=5)
        recorder = RecordingObserver()
        summary = run_observed(config, sources, detector, recorder)

        assert len(recorder.snapshots) == summary.venue_instants
        assert len(recorder.snapshots) == len(records(config.association_report_path)) - 1

    def test_each_snapshot_carries_the_venue_instant_verbatim(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=6, frames_b=6)
        recorder = RecordingObserver()
        run_observed(config, sources, detector, recorder)

        instant_records = records(config.association_report_path)[1:]
        assert [snap.venue_seconds for snap in recorder.snapshots] == [
            record["venue_seconds"] for record in instant_records
        ]

    def test_snapshot_components_all_describe_the_same_instant(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=6, frames_b=4)
        recorder = RecordingObserver()
        run_observed(config, sources, detector, recorder)

        for snap in recorder.snapshots:
            assert snap.association_round.venue_seconds == snap.venue_seconds
            assert snap.integration.venue_seconds == snap.venue_seconds
            assert snap.occupancy.venue_seconds == snap.venue_seconds


class TestObservations:
    def test_the_snapshot_holds_every_current_observation_in_canonical_order(
        self, tmp_path: Path
    ) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=5, frames_b=5)
        recorder = RecordingObserver()
        run_observed(config, sources, detector, recorder)

        for snap in recorder.snapshots:
            assert list(snap.observations) == sorted(snap.observations, key=observation_order_key)
            assert snap.observations == snap.association_round.observations
        assert {c for snap in recorder.snapshots for c in observed_subjects(snap)} != set()
        cameras = {o.subject.camera_id for snap in recorder.snapshots for o in snap.observations}
        assert cameras == {CAM_A, CAM_B}

    def test_a_zero_frame_stream_contributes_no_fabricated_observation(
        self, tmp_path: Path
    ) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=5, frames_b=0)
        recorder = RecordingObserver()
        summary = run_observed(config, sources, detector, recorder)

        assert len(recorder.snapshots) == summary.venue_instants == 5
        for snap in recorder.snapshots:
            assert {o.subject.camera_id for o in snap.observations} == {CAM_A}


class TestCurrentMembership:
    def test_current_refs_map_to_their_owning_venue_subject(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=10, frames_b=10)
        recorder = RecordingObserver()
        run_observed(config, sources, detector, recorder)

        joined = [
            snap
            for snap in recorder.snapshots
            if any(
                {key[0] for key in membership.members} == {CAM_A, CAM_B}
                for membership in snap.memberships
            )
        ]
        assert joined, "the two cameras never associated in this synthetic run"
        for snap in joined:
            for membership in snap.memberships:
                assert all(len(key) == 3 for key in membership.members)
                assert len(membership.members) == len(set(membership.members))
                assert len({key[0] for key in membership.members}) == len(membership.members)

    def test_owned_and_unowned_partition_the_current_observations(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=8, frames_b=8)
        recorder = RecordingObserver()
        run_observed(config, sources, detector, recorder)

        for snap in recorder.snapshots:
            owned = [key for membership in snap.memberships for key in membership.members]
            unowned = list(snap.unowned_tracklets)
            assert len(owned) + len(unowned) == len(set(owned) | set(unowned))
            assert set(owned).isdisjoint(unowned)
            assert set(owned) | set(unowned) == observed_subjects(snap)

    def test_a_retained_but_unobserved_member_is_not_reported_as_current(
        self, tmp_path: Path
    ) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=8, frames_b=3)
        recorder = RecordingObserver()
        run_observed(config, sources, detector, recorder)

        # The short stream stops after instant 2; later instants observe CAM_A
        # only, so no CAM_B ref may appear anywhere in those snapshots even
        # though the registry still retains the association for provenance.
        tail = recorder.snapshots[4:]
        assert tail
        for snap in tail:
            assert {o.subject.camera_id for o in snap.observations} == {CAM_A}
            assert all(key[0] == CAM_A for key in subject_refs(snap))
            for membership in snap.memberships:
                assert all(key[0] == CAM_A for key in membership.members)

    def test_single_camera_observations_are_all_explicitly_unowned(self, tmp_path: Path) -> None:
        path = tmp_path / "solo.mp4"
        config = VenueValidationConfig(
            manifest=manifest_for(VenueCamera(camera_id=CAM_A, sources=(camera_entry(path),))),
            output_dir=tmp_path / "artifacts",
        )
        recorder = RecordingObserver()
        summary = run_observed(
            config, {path: source_for(*DIMS_A, 6)}, standing_detector(), recorder
        )

        assert len(recorder.snapshots) == summary.venue_instants == 6
        for snap in recorder.snapshots:
            assert snap.memberships == ()
            assert set(snap.unowned_tracklets) == observed_subjects(snap)


class TestOrdering:
    def test_ordering_is_inherited_from_the_occupancy_snapshot_not_reimposed(
        self, tmp_path: Path
    ) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=8, frames_b=8)
        recorder = RecordingObserver()
        run_observed(config, sources, detector, recorder)

        saw_members = False
        for snap in recorder.snapshots:
            ids = [membership.subject_id for membership in snap.memberships]
            assert ids == [c.subject_id for c in snap.occupancy.contributing_subjects]
            by_id = {c.subject_id: c for c in snap.occupancy.contributing_subjects}
            for membership in snap.memberships:
                contribution = by_id[membership.subject_id]
                saw_members = saw_members or bool(membership.members)
                # position-for-position: the producer's order is preserved, and
                # each member is that producer SubjectRef plus its stream_epoch.
                assert [(key[0], key[2]) for key in membership.members] == [
                    (ref.camera_id, ref.track_id) for ref in contribution.members
                ]
                assert all(len(key) == 3 for key in membership.members)
            assert [(key[0], key[2]) for key in snap.unowned_tracklets] == [
                (ref.camera_id, ref.track_id) for ref in snap.occupancy.unresolved_tracklets
            ]
        assert saw_members, "no owned membership ever appeared in this synthetic run"

    def test_manifest_camera_order_does_not_change_the_snapshots(self, tmp_path: Path) -> None:
        forward_config, forward_sources, forward_detector = two_camera_setup(
            tmp_path, order=(CAM_A, CAM_B), output_dir=tmp_path / "forward"
        )
        forward = RecordingObserver()
        run_observed(forward_config, forward_sources, forward_detector, forward)

        reverse_config, reverse_sources, reverse_detector = two_camera_setup(
            tmp_path, order=(CAM_B, CAM_A), output_dir=tmp_path / "reverse"
        )
        reverse = RecordingObserver()
        run_observed(reverse_config, reverse_sources, reverse_detector, reverse)

        assert digest(forward.snapshots) == digest(reverse.snapshots)

    def test_two_runs_produce_identical_snapshots(self, tmp_path: Path) -> None:
        first_config, first_sources, first_detector = two_camera_setup(
            tmp_path, output_dir=tmp_path / "one"
        )
        first = RecordingObserver()
        run_observed(first_config, first_sources, first_detector, first)

        second_config, second_sources, second_detector = two_camera_setup(
            tmp_path, output_dir=tmp_path / "two"
        )
        second = RecordingObserver()
        run_observed(second_config, second_sources, second_detector, second)

        assert digest(first.snapshots) == digest(second.snapshots)


class TestImmutability:
    def test_the_snapshot_and_its_members_are_frozen(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=3, frames_b=3)
        recorder = RecordingObserver()
        run_observed(config, sources, detector, recorder)
        snap = recorder.snapshots[0]

        with pytest.raises(FrozenInstanceError):
            snap.venue_seconds = 99.0  # type: ignore[misc]
        assert isinstance(snap.observations, tuple)
        assert isinstance(snap.memberships, tuple)
        assert isinstance(snap.unowned_tracklets, tuple)
        membership = VenueSubjectMembershipSnapshot(
            subject_id=VenueSubjectId(sequence=1), members=()
        )
        with pytest.raises(FrozenInstanceError):
            membership.members = ()  # type: ignore[misc]


class TestRuntimeIsolation:
    def test_an_observer_return_value_and_state_cannot_change_the_artifacts(
        self, tmp_path: Path
    ) -> None:
        baseline_config, baseline_sources, baseline_detector = two_camera_setup(
            tmp_path, output_dir=tmp_path / "baseline"
        )
        run_observed(baseline_config, baseline_sources, baseline_detector, RecordingObserver())

        meddling_config, meddling_sources, meddling_detector = two_camera_setup(
            tmp_path, output_dir=tmp_path / "meddling"
        )
        meddler = MeddlingObserver()
        # MeddlingObserver deliberately returns a value; the seam must ignore it,
        # so it is not structurally a VenueValidationObserver (-> None).
        run_observed(meddling_config, meddling_sources, meddling_detector, meddler)  # type: ignore[arg-type]

        assert meddler.calls > 0
        for name in ("association_report_path", "occupancy_report_path", "summary_path"):
            assert (
                getattr(baseline_config, name).read_bytes()
                == getattr(meddling_config, name).read_bytes()
            )

    def test_the_generic_observer_module_names_no_dataset_type(self) -> None:
        source = inspect.getsource(observer_module)
        lowered = source.lower()
        assert "wildtrack" not in lowered
        assert "ground_truth" not in lowered
        for name in dir(observer_module):
            assert "wildtrack" not in name.lower()


class TestFailureSemantics:
    def test_an_observer_that_raises_fails_the_run(self, tmp_path: Path) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=6, frames_b=6)
        with pytest.raises(RuntimeError, match="deliberately"):
            run_observed(config, sources, detector, ExplodingObserver(fail_on=2))

    def test_a_failed_observer_run_releases_sources_and_writes_no_summary(
        self, tmp_path: Path
    ) -> None:
        config, sources, detector = two_camera_setup(tmp_path, frames_a=6, frames_b=6)
        with pytest.raises(RuntimeError):
            run_observed(config, sources, detector, ExplodingObserver(fail_on=1))

        assert all(source.released for source in sources.values())  # type: ignore[attr-defined]
        assert not config.summary_path.exists()


def _instant(seconds: float, observations: tuple[TrackObservation, ...] = ()) -> VenueInstantResult:
    ordered = tuple(sorted(observations, key=observation_order_key))
    return VenueInstantResult(
        venue_seconds=seconds,
        observations=ordered,
        round=AssociationRound(venue_seconds=seconds, observations=ordered, decisions=()),
        integration=IntegrationResult(venue_seconds=seconds, outcomes=()),
        active_subjects=(),
    )


def _occupancy(
    seconds: float,
    *,
    raw: int = 0,
    contributing: tuple[OccupancyContribution, ...] = (),
    unresolved: tuple[SubjectRef, ...] = (),
) -> OccupancySnapshot:
    return OccupancySnapshot(
        venue_seconds=seconds,
        raw_active_track_count=raw,
        contributing_subjects=contributing,
        unresolved_tracklets=unresolved,
    )


class TestBuildInstantSnapshot:
    def test_matching_instants_build_an_empty_snapshot(self) -> None:
        snapshot = build_instant_snapshot(_instant(1.5), _occupancy(1.5))
        assert snapshot.venue_seconds == 1.5
        assert snapshot.memberships == ()
        assert snapshot.unowned_tracklets == ()

    def test_a_mismatched_occupancy_instant_is_refused_via_the_type_invariant(self) -> None:
        with pytest.raises(ValidationError, match="one venue instant"):
            build_instant_snapshot(_instant(1.0), _occupancy(2.0))

    def test_owned_and_unowned_refs_resolve_to_tracklet_keys_with_stream_epoch(self) -> None:
        owned = make_observation(CAM_A, 7, stream_epoch=2)
        loose = make_observation(CAM_B, 4, stream_epoch=1)
        result = _instant(10.0, (owned, loose))
        occupancy = _occupancy(
            10.0,
            raw=2,
            contributing=(
                OccupancyContribution(
                    subject_id=VenueSubjectId(sequence=1), members=(owned.subject,)
                ),
            ),
            unresolved=(loose.subject,),
        )
        snapshot = build_instant_snapshot(result, occupancy)

        assert snapshot.memberships[0].members == ((CAM_A, 2, 7),)
        assert snapshot.unowned_tracklets == ((CAM_B, 1, 4),)

    def test_two_refs_differing_only_by_stream_epoch_stay_distinct(self) -> None:
        early = make_observation(CAM_A, 5, stream_epoch=0)
        late = make_observation(CAM_B, 9, stream_epoch=3)
        result = _instant(4.0, (early, late))
        occupancy = _occupancy(4.0, raw=2, unresolved=(early.subject, late.subject))

        snapshot = build_instant_snapshot(result, occupancy)

        assert set(snapshot.unowned_tracklets) == {(CAM_A, 0, 5), (CAM_B, 3, 9)}

    def test_an_unresolvable_occupancy_ref_raises_rather_than_being_dropped(self) -> None:
        result = _instant(2.0, ())
        occupancy = _occupancy(2.0, raw=1, unresolved=(SubjectRef(CAM_A, 1),))
        with pytest.raises(ValidationError, match="exactly one"):
            build_instant_snapshot(result, occupancy)

    def test_an_ambiguous_occupancy_ref_two_matching_observations_raises(self) -> None:
        first = make_observation(CAM_A, 3, stream_epoch=0)
        second = make_observation(CAM_A, 3, stream_epoch=1)
        result = _instant(6.0, (first, second))
        occupancy = _occupancy(6.0, raw=1, unresolved=(SubjectRef(CAM_A, 3),))
        with pytest.raises(ValidationError, match="exactly one"):
            build_instant_snapshot(result, occupancy)


class TestInstantSnapshotTypeInvariant:
    """The same-instant invariant is owned by the frozen type, not the builder."""

    def _snapshot(
        self,
        *,
        venue_seconds: float = 3.0,
        round_seconds: float = 3.0,
        integration_seconds: float = 3.0,
        occupancy_seconds: float = 3.0,
    ) -> VenueValidationInstantSnapshot:
        return VenueValidationInstantSnapshot(
            venue_seconds=venue_seconds,
            observations=(),
            association_round=AssociationRound(
                venue_seconds=round_seconds, observations=(), decisions=()
            ),
            integration=IntegrationResult(venue_seconds=integration_seconds, outcomes=()),
            memberships=(),
            unowned_tracklets=(),
            occupancy=_occupancy(occupancy_seconds),
        )

    def test_direct_construction_with_matching_components_is_accepted(self) -> None:
        assert self._snapshot().venue_seconds == 3.0

    def test_direct_construction_rejects_a_mismatched_association_round(self) -> None:
        with pytest.raises(ValidationError, match="one venue instant"):
            self._snapshot(round_seconds=9.0)

    def test_direct_construction_rejects_a_mismatched_integration(self) -> None:
        with pytest.raises(ValidationError, match="one venue instant"):
            self._snapshot(integration_seconds=9.0)

    def test_direct_construction_rejects_a_mismatched_occupancy(self) -> None:
        with pytest.raises(ValidationError, match="one venue instant"):
            self._snapshot(occupancy_seconds=9.0)


class TestPackageApi:
    def test_the_builder_is_not_exported_from_the_package_root(self) -> None:
        import atreides_vision.venue_validation as package

        assert "build_instant_snapshot" not in package.__all__
        assert not hasattr(package, "build_instant_snapshot")

    def test_the_observer_types_remain_exported(self) -> None:
        import atreides_vision.venue_validation as package

        for name in (
            "VenueSubjectMembershipSnapshot",
            "VenueValidationInstantSnapshot",
            "VenueValidationObserver",
        ):
            assert name in package.__all__
            assert hasattr(package, name)

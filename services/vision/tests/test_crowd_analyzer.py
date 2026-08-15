"""The spatial-cluster analyzer end to end, driven by synthetic behaviour frames.

:class:`TestSameCameraBoundary` is the architectural test: subjects from two
cameras must never end up in one cluster, because without association one real
person could be counted twice and clustered with themselves.

:class:`TestPersistence` pins the exact-member-set semantics, which is what stops
the layer from silently claiming group continuity through changing membership.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.behavior.models import (
    BehaviorFrame,
    BehaviorSubjectSnapshot,
    SubjectMotion,
    SubjectRef,
)
from atreides_vision.crowd.analyzer import SpatialClusterAnalyzer
from atreides_vision.crowd.config import CrowdAnalysisConfig
from atreides_vision.crowd.models import CrowdFrame, canonical_members
from atreides_vision.errors import ConfigurationError, CrowdError
from atreides_vision.interaction.models import SpatialContext

METRIC = MotionSpace.GROUND_PLANE_METRES
NORMALIZED = MotionSpace.NORMALIZED_IMAGE
FPS = 25.0
DT = 1.0 / FPS
CAMERA = "cam_01"

#: ``(x, y, dx, dy)`` at a moment: position plus recent motion.
Path = Callable[[float], tuple[float, float, float, float]]


def metric_context(camera_id: str = CAMERA) -> SpatialContext:
    return SpatialContext(camera_id, METRIC, "venue_01", "main_floor")


def analyzer(context: SpatialContext | None = None, **overrides: object) -> SpatialClusterAnalyzer:
    config = CrowdAnalysisConfig.for_ground_plane_metres()
    if overrides:
        from dataclasses import replace

        config = replace(config, **overrides)  # type: ignore[arg-type]
    return SpatialClusterAnalyzer(context or metric_context(), config)


def snapshot(
    track_id: int,
    x: float,
    y: float,
    dx: float = 0.0,
    dy: float = 0.0,
    frame_index: int = 1,
    timestamp: float = 0.0,
    camera_id: str = CAMERA,
    space: MotionSpace = METRIC,
    zone_ids: tuple[str, ...] = (),
) -> BehaviorSubjectSnapshot:
    return BehaviorSubjectSnapshot(
        subject=SubjectRef(camera_id, track_id),
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        position=MotionPoint(x, y, space),
        zone_ids=zone_ids,
        motion=SubjectMotion(dx, dy, 1.0, space),
    )


def frame_of(
    *snapshots: BehaviorSubjectSnapshot, index: int = 1, timestamp: float = 0.0
) -> BehaviorFrame:
    return BehaviorFrame.from_snapshots(index, timestamp, snapshots)


def run(
    instance: SpatialClusterAnalyzer, paths: dict[int, Path], frames: int = 20
) -> list[CrowdFrame]:
    """Drive several subjects along known paths and collect every frame."""
    results = []
    for index in range(frames):
        timestamp = index * DT
        snapshots = [
            snapshot(
                track_id,
                *path(timestamp),
                frame_index=index + 1,
                timestamp=timestamp,
                camera_id=instance.camera_id,
            )
            for track_id, path in paths.items()
        ]
        results.append(
            instance.update(BehaviorFrame.from_snapshots(index + 1, timestamp, snapshots))
        )
    return results


def still(x: float, y: float = 0.0) -> Path:
    return lambda _t: (x, y, 0.0, 0.0)


def moving(start: float, velocity: float, y: float = 0.0) -> Path:
    return lambda t: (start + velocity * t, y, velocity, 0.0)


def members_of(frame: CrowdFrame) -> list[list[int]]:
    return [[m.track_id for m in cluster.members] for cluster in frame.clusters]


class TestConstruction:
    def test_reports_its_context(self) -> None:
        instance = analyzer()
        assert instance.camera_id == CAMERA
        assert instance.space is METRIC

    def test_rejects_a_config_written_for_another_space(self) -> None:
        """A normalized radius read as metres connects only people touching."""
        with pytest.raises(ConfigurationError, match="written for normalized_image"):
            SpatialClusterAnalyzer(metric_context(), CrowdAnalysisConfig.for_normalized_image())

    def test_rejects_metric_config_in_normalized_space(self) -> None:
        with pytest.raises(ConfigurationError, match="written for ground_plane_metres"):
            SpatialClusterAnalyzer(
                SpatialContext(CAMERA, NORMALIZED),
                CrowdAnalysisConfig.for_ground_plane_metres(),
            )

    def test_an_undeclared_config_is_allowed_anywhere(self) -> None:
        assert SpatialClusterAnalyzer(metric_context(), CrowdAnalysisConfig()).space is METRIC

    def test_starts_empty(self) -> None:
        instance = analyzer()
        assert instance.latest_frame is None
        assert instance.clusters_measured == 0
        assert instance.tracked_compositions == 0


class TestFrameSummary:
    def test_no_subjects(self) -> None:
        result = analyzer().update(frame_of())
        assert len(result) == 0
        assert result.summary.observed_subject_count == 0
        assert result.summary.isolated_subject_count == 0

    def test_one_subject(self) -> None:
        result = analyzer().update(frame_of(snapshot(1, 0.0, 0.0)))
        assert len(result) == 0
        assert result.summary.observed_subject_count == 1
        assert result.summary.isolated_subject_count == 1

    def test_two_subjects_below_the_minimum(self) -> None:
        result = analyzer().update(frame_of(snapshot(1, 0.0, 0.0), snapshot(2, 1.0, 0.0)))
        assert result.summary.cluster_count == 0
        assert result.summary.clustered_subject_count == 0
        assert result.summary.isolated_subject_count == 2

    def test_counts_add_up(self) -> None:
        result = analyzer().update(
            frame_of(
                snapshot(1, 0.0, 0.0),
                snapshot(2, 1.0, 0.0),
                snapshot(3, 2.0, 0.0),
                snapshot(9, 60.0, 60.0),
            )
        )
        summary = result.summary
        assert summary.observed_subject_count == 4
        assert summary.cluster_count == 1
        assert summary.clustered_subject_count == 3
        assert summary.isolated_subject_count == 1
        assert summary.largest_cluster_size == 3

    def test_largest_cluster_size(self) -> None:
        result = analyzer().update(
            frame_of(
                snapshot(1, 0.0, 0.0),
                snapshot(2, 1.0, 0.0),
                snapshot(3, 2.0, 0.0),
                snapshot(4, 3.0, 0.0),
                snapshot(7, 60.0, 0.0),
                snapshot(8, 61.0, 0.0),
                snapshot(9, 62.0, 0.0),
            )
        )
        assert result.summary.cluster_count == 2
        assert result.summary.largest_cluster_size == 4

    def test_graph_counters_are_reported(self) -> None:
        """Three collinear subjects at a 1.5 m radius form a chain, not a clique.

        The end pair is 2.0 m apart, beyond the radius, so only two of the three
        examined pairs become edges — and they are still one cluster.
        """
        result = analyzer().update(
            frame_of(snapshot(1, 0.0, 0.0), snapshot(2, 1.0, 0.0), snapshot(3, 2.0, 0.0))
        )
        assert result.pair_comparisons == 3
        assert result.edge_count == 2
        assert result.summary.cluster_count == 1


class TestClusterMetrics:
    def cluster(self, *snapshots: BehaviorSubjectSnapshot) -> object:
        result = analyzer().update(frame_of(*snapshots))
        assert len(result) == 1
        return result.clusters[0]

    def test_centroid_of_a_simple_triangle(self) -> None:
        cluster = self.cluster(snapshot(1, 0.0, 0.0), snapshot(2, 1.5, 0.0), snapshot(3, 0.0, 1.5))
        assert (cluster.centroid.x, cluster.centroid.y) == pytest.approx((0.5, 0.5))  # type: ignore[attr-defined]

    def test_spread_of_three_in_a_row(self) -> None:
        cluster = self.cluster(snapshot(1, 0.0, 0.0), snapshot(2, 1.0, 0.0), snapshot(3, 2.0, 0.0))
        assert cluster.spatial_spread == pytest.approx(math.sqrt(2.0 / 3.0))  # type: ignore[attr-defined]

    def test_mean_and_max_pair_distance(self) -> None:
        cluster = self.cluster(snapshot(1, 0.0, 0.0), snapshot(2, 1.0, 0.0), snapshot(3, 2.0, 0.0))
        assert cluster.mean_pair_distance == pytest.approx(4.0 / 3.0)  # type: ignore[attr-defined]
        assert cluster.max_pair_distance == pytest.approx(2.0)  # type: ignore[attr-defined]

    def test_max_pair_distance_can_exceed_the_cluster_radius(self) -> None:
        """Chaining, made visible in the reported extent."""
        cluster = self.cluster(
            snapshot(1, 0.0, 0.0),
            snapshot(2, 1.4, 0.0),
            snapshot(3, 2.8, 0.0),
            snapshot(4, 4.2, 0.0),
        )
        assert cluster.max_pair_distance == pytest.approx(4.2)  # type: ignore[attr-defined]
        assert cluster.max_pair_distance > 1.5  # type: ignore[attr-defined]

    def test_member_count(self) -> None:
        cluster = self.cluster(snapshot(1, 0.0, 0.0), snapshot(2, 1.0, 0.0), snapshot(3, 2.0, 0.0))
        assert cluster.member_count == 3  # type: ignore[attr-defined]

    def test_density_counts_every_observed_subject_not_only_members(self) -> None:
        """Density describes the neighbourhood, not cluster membership."""
        sparse = analyzer().update(
            frame_of(snapshot(1, 0.0, 0.0), snapshot(2, 1.0, 0.0), snapshot(3, 2.0, 0.0))
        )
        crowded = analyzer().update(
            frame_of(
                snapshot(1, 0.0, 0.0),
                snapshot(2, 1.0, 0.0),
                snapshot(3, 2.0, 0.0),
                # Too far to join the cluster, close enough to raise density.
                snapshot(4, 0.0, 1.9),
                snapshot(5, 1.0, 1.9),
            )
        )
        dense_density = crowded.clusters[0].mean_local_density
        sparse_density = sparse.clusters[0].mean_local_density
        assert dense_density is not None and sparse_density is not None
        assert dense_density > sparse_density

    def test_density_matches_the_formula(self) -> None:
        """Three subjects within a 2 m disk of each other: each sees two others."""
        result = analyzer().update(
            frame_of(snapshot(1, 0.0, 0.0), snapshot(2, 1.0, 0.0), snapshot(3, 2.0, 0.0))
        )
        cluster = result.clusters[0]
        # The middle subject sees both others; the ends see one each at r=2.0
        # (0->2 is exactly 2.0, so inclusive). All three therefore see two.
        assert cluster.mean_local_density == pytest.approx(3.0 / (math.pi * 4.0))
        assert cluster.max_local_density == pytest.approx(3.0 / (math.pi * 4.0))

    def test_coherence_when_all_move_together(self) -> None:
        cluster = self.cluster(
            snapshot(1, 0.0, 0.0, 1.0, 0.0),
            snapshot(2, 1.0, 0.0, 1.0, 0.0),
            snapshot(3, 2.0, 0.0, 1.0, 0.0),
        )
        assert cluster.motion_heading_coherence == pytest.approx(1.0)  # type: ignore[attr-defined]

    def test_coherence_when_headings_oppose(self) -> None:
        cluster = self.cluster(
            snapshot(1, 0.0, 0.0, 1.0, 0.0),
            snapshot(2, 1.0, 0.0, -1.0, 0.0),
            snapshot(3, 2.0, 0.0, 1.0, 0.0),
        )
        assert cluster.motion_heading_coherence == pytest.approx(1.0 / 3.0)  # type: ignore[attr-defined]

    def test_coherence_omitted_when_nobody_moves(self) -> None:
        """Not zero: stationary members have no direction to compare."""
        cluster = self.cluster(snapshot(1, 0.0, 0.0), snapshot(2, 1.0, 0.0), snapshot(3, 2.0, 0.0))
        assert cluster.motion_heading_coherence is None  # type: ignore[attr-defined]

    def test_coherence_omitted_with_only_one_mover(self) -> None:
        cluster = self.cluster(
            snapshot(1, 0.0, 0.0, 1.0, 0.0),
            snapshot(2, 1.0, 0.0),
            snapshot(3, 2.0, 0.0),
        )
        assert cluster.motion_heading_coherence is None  # type: ignore[attr-defined]


class TestSameCameraBoundary:
    def test_a_foreign_camera_snapshot_is_rejected(self) -> None:
        instance = analyzer()
        mixed = frame_of(
            snapshot(1, 0.0, 0.0),
            snapshot(2, 1.0, 0.0),
            snapshot(9, 1.5, 0.0, camera_id="cam_other"),
        )
        with pytest.raises(CrowdError, match="does not belong to camera"):
            instance.update(mixed)

    def test_the_rejection_explains_the_double_counting_risk(self) -> None:
        instance = analyzer()
        with pytest.raises(CrowdError) as caught:
            instance.update(frame_of(snapshot(1, 0.0, 0.0, camera_id="cam_other")))
        assert "count one real person twice" in str(caught.value)

    def test_a_member_set_spanning_cameras_is_impossible(self) -> None:
        with pytest.raises(CrowdError, match="same-camera only"):
            canonical_members(
                [SubjectRef("cam_a", 1), SubjectRef("cam_a", 2), SubjectRef("cam_b", 9)]
            )

    def test_two_cameras_use_two_analyzers_with_separate_state(self) -> None:
        left = analyzer(metric_context("cam_a"))
        right = analyzer(metric_context("cam_b"))

        left_frames = run(left, {1: still(0.0), 2: still(1.0), 3: still(2.0)}, frames=10)
        right_frames = run(right, {1: still(0.0), 2: still(1.0), 3: still(2.0)}, frames=10)

        assert left_frames[-1].clusters[0].camera_id == "cam_a"
        assert right_frames[-1].clusters[0].camera_id == "cam_b"
        # Same numeric track ids, and still two entirely separate clusters.
        assert left_frames[-1].clusters[0].members != right_frames[-1].clusters[0].members
        assert left.tracked_compositions == right.tracked_compositions == 1

    def test_one_camera_does_not_affect_another_s_counts(self) -> None:
        left = analyzer(metric_context("cam_a"))
        right = analyzer(metric_context("cam_b"))
        run(left, {1: still(0.0), 2: still(1.0), 3: still(2.0)}, frames=5)
        result = run(right, {1: still(0.0)}, frames=5)[-1]
        assert result.summary.observed_subject_count == 1
        assert result.summary.cluster_count == 0


class TestCoordinateCompatibility:
    def test_a_foreign_space_snapshot_is_rejected(self) -> None:
        instance = analyzer()
        with pytest.raises(CrowdError, match="measures in ground_plane_metres"):
            instance.update(
                frame_of(snapshot(1, 0.0, 0.0), snapshot(2, 0.5, 0.5, space=NORMALIZED))
            )

    def test_metric_context_requires_venue_and_surface(self) -> None:
        from atreides_vision.errors import InteractionError

        with pytest.raises(InteractionError, match="venue_id and a surface_id"):
            SpatialContext(CAMERA, METRIC)

    def test_normalized_clustering_works(self) -> None:
        instance = SpatialClusterAnalyzer(
            SpatialContext(CAMERA, NORMALIZED), CrowdAnalysisConfig.for_normalized_image()
        )
        result = instance.update(
            frame_of(
                snapshot(1, 0.10, 0.10, space=NORMALIZED),
                snapshot(2, 0.15, 0.10, space=NORMALIZED),
                snapshot(3, 0.20, 0.10, space=NORMALIZED),
            )
        )
        assert len(result) == 1
        assert result.clusters[0].space is NORMALIZED

    def test_large_coordinates_are_handled(self) -> None:
        result = analyzer().update(
            frame_of(
                snapshot(1, 5000.0, 5000.0),
                snapshot(2, 5001.0, 5000.0),
                snapshot(3, 5002.0, 5000.0),
            )
        )
        assert len(result) == 1
        assert result.clusters[0].centroid.x == pytest.approx(5001.0)

    def test_identical_positions_do_not_merge_subjects(self) -> None:
        result = analyzer().update(
            frame_of(snapshot(1, 2.0, 2.0), snapshot(2, 2.0, 2.0), snapshot(3, 2.0, 2.0))
        )
        cluster = result.clusters[0]
        assert cluster.member_count == 3
        assert cluster.spatial_spread == pytest.approx(0.0)
        assert cluster.max_pair_distance == pytest.approx(0.0)


class TestPersistence:
    def test_a_stable_composition_accumulates(self) -> None:
        frames = run(analyzer(), {1: still(0.0), 2: still(1.0), 3: still(2.0)}, frames=51)
        assert frames[-1].clusters[0].persistence_duration == pytest.approx(2.0, abs=DT)

    def test_the_first_sighting_reports_zero(self) -> None:
        result = analyzer().update(
            frame_of(snapshot(1, 0.0, 0.0), snapshot(2, 1.0, 0.0), snapshot(3, 2.0, 0.0))
        )
        assert result.clusters[0].persistence_duration == 0.0

    def test_a_joining_member_starts_a_new_composition(self) -> None:
        """Exact-member semantics: {1,2,3} and {1,2,3,4} are different clusters."""
        instance = analyzer()
        for index in range(20):
            timestamp = index * DT
            instance.update(
                frame_of(
                    snapshot(1, 0.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    snapshot(2, 1.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    snapshot(3, 2.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    index=index + 1,
                    timestamp=timestamp,
                )
            )
        joined = instance.update(
            frame_of(
                snapshot(1, 0.0, 0.0, frame_index=21, timestamp=20 * DT),
                snapshot(2, 1.0, 0.0, frame_index=21, timestamp=20 * DT),
                snapshot(3, 2.0, 0.0, frame_index=21, timestamp=20 * DT),
                snapshot(4, 3.0, 0.0, frame_index=21, timestamp=20 * DT),
                index=21,
                timestamp=20 * DT,
            )
        )
        assert joined.clusters[0].member_count == 4
        assert joined.clusters[0].persistence_duration == 0.0

    def test_a_leaving_member_starts_a_new_composition(self) -> None:
        instance = analyzer()
        for index in range(20):
            timestamp = index * DT
            instance.update(
                frame_of(
                    snapshot(1, 0.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    snapshot(2, 1.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    snapshot(3, 2.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    snapshot(4, 3.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    index=index + 1,
                    timestamp=timestamp,
                )
            )
        reduced = instance.update(
            frame_of(
                snapshot(1, 0.0, 0.0, frame_index=21, timestamp=20 * DT),
                snapshot(2, 1.0, 0.0, frame_index=21, timestamp=20 * DT),
                snapshot(3, 2.0, 0.0, frame_index=21, timestamp=20 * DT),
                index=21,
                timestamp=20 * DT,
            )
        )
        assert reduced.clusters[0].persistence_duration == 0.0

    def test_track_id_churn_starts_a_new_composition(self) -> None:
        """A re-acquired person is a new subject, so the composition changes."""
        instance = analyzer()
        for index in range(20):
            timestamp = index * DT
            instance.update(
                frame_of(
                    snapshot(1, 0.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    snapshot(2, 1.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    snapshot(3, 2.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    index=index + 1,
                    timestamp=timestamp,
                )
            )
        churned = instance.update(
            frame_of(
                snapshot(1, 0.0, 0.0, frame_index=21, timestamp=20 * DT),
                snapshot(2, 1.0, 0.0, frame_index=21, timestamp=20 * DT),
                snapshot(31, 2.0, 0.0, frame_index=21, timestamp=20 * DT),
                index=21,
                timestamp=20 * DT,
            )
        )
        assert [m.track_id for m in churned.clusters[0].members] == [1, 2, 31]
        assert churned.clusters[0].persistence_duration == 0.0

    def test_a_short_gap_is_tolerated(self) -> None:
        instance = analyzer()
        last = None
        for index in range(40):
            if index in (20, 21):  # boxes dropped briefly
                continue
            timestamp = index * DT
            last = instance.update(
                frame_of(
                    snapshot(1, 0.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    snapshot(2, 1.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    snapshot(3, 2.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    index=index + 1,
                    timestamp=timestamp,
                )
            )
        assert last is not None
        assert last.clusters[0].persistence_duration > 1.0

    def test_a_long_gap_resets(self) -> None:
        """Where they went while unobserved is unknown."""
        instance = analyzer()
        for index in range(20):
            timestamp = index * DT
            instance.update(
                frame_of(
                    snapshot(1, 0.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    snapshot(2, 1.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    snapshot(3, 2.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    index=index + 1,
                    timestamp=timestamp,
                )
            )
        # Everyone disappears long enough for the composition to expire.
        for index in range(20, 220):
            timestamp = index * DT
            instance.update(frame_of(index=index + 1, timestamp=timestamp))

        returned = instance.update(
            frame_of(
                snapshot(1, 0.0, 0.0, frame_index=300, timestamp=12.0),
                snapshot(2, 1.0, 0.0, frame_index=300, timestamp=12.0),
                snapshot(3, 2.0, 0.0, frame_index=300, timestamp=12.0),
                index=300,
                timestamp=12.0,
            )
        )
        assert returned.clusters[0].persistence_duration == 0.0

    def test_two_compositions_do_not_leak_state(self) -> None:
        instance = analyzer()
        frames = run(
            instance,
            {
                1: still(0.0),
                2: still(1.0),
                3: still(2.0),
                7: still(40.0),
                8: still(41.0),
                9: still(42.0),
            },
            frames=30,
        )
        first, second = frames[-1].clusters
        assert first.persistence_duration == pytest.approx(second.persistence_duration)
        assert instance.tracked_compositions == 2

    def test_history_is_bounded_under_churn(self) -> None:
        instance = analyzer(stale_cluster_ttl_seconds=0.5)
        for index in range(120):
            timestamp = index * DT
            base = index * 3
            instance.update(
                frame_of(
                    snapshot(base, 0.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    snapshot(base + 1, 1.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    snapshot(base + 2, 2.0, 0.0, frame_index=index + 1, timestamp=timestamp),
                    index=index + 1,
                    timestamp=timestamp,
                )
            )
        assert instance.tracked_compositions <= 2


class TestZoneContext:
    def test_common_zones_are_those_all_members_share(self) -> None:
        result = analyzer().update(
            frame_of(
                snapshot(1, 0.0, 0.0, zone_ids=("dancefloor", "bar")),
                snapshot(2, 1.0, 0.0, zone_ids=("dancefloor",)),
                snapshot(3, 2.0, 0.0, zone_ids=("dancefloor", "walkway")),
            )
        )
        assert result.clusters[0].common_zone_ids == ("dancefloor",)

    def test_member_zones_are_the_union(self) -> None:
        result = analyzer().update(
            frame_of(
                snapshot(1, 0.0, 0.0, zone_ids=("bar",)),
                snapshot(2, 1.0, 0.0, zone_ids=("dancefloor",)),
                snapshot(3, 2.0, 0.0, zone_ids=("bar", "walkway")),
            )
        )
        assert result.clusters[0].member_zone_ids == ("bar", "dancefloor", "walkway")

    def test_mixed_zones_share_nothing(self) -> None:
        result = analyzer().update(
            frame_of(
                snapshot(1, 0.0, 0.0, zone_ids=("bar",)),
                snapshot(2, 1.0, 0.0, zone_ids=("entrance",)),
                snapshot(3, 2.0, 0.0, zone_ids=("stage",)),
            )
        )
        assert result.clusters[0].common_zone_ids == ()

    def test_one_member_without_zones_leaves_no_common_zone(self) -> None:
        result = analyzer().update(
            frame_of(
                snapshot(1, 0.0, 0.0, zone_ids=("bar",)),
                snapshot(2, 1.0, 0.0, zone_ids=("bar",)),
                snapshot(3, 2.0, 0.0),
            )
        )
        assert result.clusters[0].common_zone_ids == ()

    def test_no_zones_at_all(self) -> None:
        result = analyzer().update(
            frame_of(snapshot(1, 0.0, 0.0), snapshot(2, 1.0, 0.0), snapshot(3, 2.0, 0.0))
        )
        assert result.clusters[0].common_zone_ids == ()
        assert result.clusters[0].member_zone_ids == ()

    def test_zones_do_not_form_clusters(self) -> None:
        """Same zone, far apart: no cluster. Geometry decides, not labels."""
        result = analyzer().update(
            frame_of(
                snapshot(1, 0.0, 0.0, zone_ids=("dancefloor",)),
                snapshot(2, 40.0, 0.0, zone_ids=("dancefloor",)),
                snapshot(3, 80.0, 0.0, zone_ids=("dancefloor",)),
            )
        )
        assert result.summary.cluster_count == 0

    def test_zones_do_not_prevent_clusters(self) -> None:
        """Different zones, close together: still one cluster."""
        result = analyzer().update(
            frame_of(
                snapshot(1, 0.0, 0.0, zone_ids=("left",)),
                snapshot(2, 0.5, 0.0, zone_ids=("right",)),
                snapshot(3, 1.0, 0.0, zone_ids=("right",)),
            )
        )
        assert result.summary.cluster_count == 1
        assert result.clusters[0].common_zone_ids == ()


class TestDeterminism:
    def test_repeated_runs_agree(self) -> None:
        def measure() -> list[tuple[tuple[int, ...], float]]:
            frames = run(
                analyzer(),
                {1: still(0.0), 2: still(1.0), 3: still(2.0), 4: moving(20.0, -1.0)},
                frames=30,
            )
            return [
                (tuple(m.track_id for m in cluster.members), cluster.spatial_spread)
                for cluster in frames[-1].clusters
            ]

        assert measure() == measure()

    def test_snapshot_ordering_does_not_change_clusters(self) -> None:
        forward = analyzer().update(
            frame_of(snapshot(1, 0.0, 0.0), snapshot(2, 1.0, 0.0), snapshot(3, 2.0, 0.0))
        )
        backward = analyzer().update(
            frame_of(snapshot(3, 2.0, 0.0), snapshot(2, 1.0, 0.0), snapshot(1, 0.0, 0.0))
        )
        assert members_of(forward) == members_of(backward) == [[1, 2, 3]]


class TestFrameLookups:
    def test_lookup_by_member_set_is_order_independent(self) -> None:
        result = analyzer().update(
            frame_of(snapshot(1, 0.0, 0.0), snapshot(2, 1.0, 0.0), snapshot(3, 2.0, 0.0))
        )
        members = [SubjectRef(CAMERA, 3), SubjectRef(CAMERA, 1), SubjectRef(CAMERA, 2)]
        assert result.for_members(members) is not None

    def test_lookup_by_subject(self) -> None:
        result = analyzer().update(
            frame_of(
                snapshot(1, 0.0, 0.0),
                snapshot(2, 1.0, 0.0),
                snapshot(3, 2.0, 0.0),
                snapshot(9, 60.0, 0.0),
            )
        )
        assert result.containing(SubjectRef(CAMERA, 2)) is not None
        assert result.containing(SubjectRef(CAMERA, 9)) is None

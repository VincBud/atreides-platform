"""Radius-connectivity clustering: edges, components, chaining and determinism.

The chaining test is the important one — it pins the documented semantics of the
chosen rule so nobody later "fixes" it into a diameter constraint.
"""

from __future__ import annotations

import pytest

from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.crowd.clustering import build_cluster_graph, connected_components
from atreides_vision.errors import CrowdError

METRIC = MotionSpace.GROUND_PLANE_METRES
NORMALIZED = MotionSpace.NORMALIZED_IMAGE
CAMERA = "cam_01"


def subject(track_id: int, camera_id: str = CAMERA) -> SubjectRef:
    return SubjectRef(camera_id, track_id)


def layout(
    *positions: tuple[int, float, float], space: MotionSpace = METRIC
) -> dict[SubjectRef, MotionPoint]:
    return {subject(track_id): MotionPoint(x, y, space) for track_id, x, y in positions}


def track_ids(components: list[tuple[SubjectRef, ...]]) -> list[list[int]]:
    return [[member.track_id for member in component] for component in components]


class TestComponentFormation:
    def test_no_subjects(self) -> None:
        assert connected_components({}, 1.5, 3) == []

    def test_one_subject_forms_no_cluster(self) -> None:
        assert connected_components(layout((1, 0.0, 0.0)), 1.5, 3) == []

    def test_two_subjects_do_not_meet_a_minimum_of_three(self) -> None:
        """Two-person relationships belong to the pairwise layer."""
        positions = layout((1, 0.0, 0.0), (2, 1.0, 0.0))
        assert connected_components(positions, 1.5, 3) == []

    def test_three_within_radius_form_one_cluster(self) -> None:
        positions = layout((1, 0.0, 0.0), (2, 1.0, 0.0), (3, 0.0, 1.0))
        assert track_ids(connected_components(positions, 1.5, 3)) == [[1, 2, 3]]

    def test_subjects_outside_the_radius_form_nothing(self) -> None:
        positions = layout((1, 0.0, 0.0), (2, 50.0, 0.0), (3, 100.0, 0.0))
        assert connected_components(positions, 1.5, 3) == []

    def test_two_separate_clusters(self) -> None:
        positions = layout(
            (1, 0.0, 0.0),
            (2, 1.0, 0.0),
            (3, 2.0, 0.0),
            (4, 40.0, 0.0),
            (5, 41.0, 0.0),
            (6, 42.0, 0.0),
        )
        assert track_ids(connected_components(positions, 1.5, 3)) == [[1, 2, 3], [4, 5, 6]]

    def test_a_lone_subject_beside_a_cluster_is_excluded(self) -> None:
        positions = layout((1, 0.0, 0.0), (2, 1.0, 0.0), (3, 2.0, 0.0), (9, 80.0, 80.0))
        assert track_ids(connected_components(positions, 1.5, 3)) == [[1, 2, 3]]

    def test_minimum_size_is_configurable(self) -> None:
        positions = layout((1, 0.0, 0.0), (2, 1.0, 0.0))
        assert track_ids(connected_components(positions, 1.5, 2)) == [[1, 2]]

    def test_identical_positions_still_form_a_cluster(self) -> None:
        """Distinct tracks at one point remain distinct subjects."""
        positions = layout((1, 2.0, 2.0), (2, 2.0, 2.0), (3, 2.0, 2.0))
        assert track_ids(connected_components(positions, 1.5, 3)) == [[1, 2, 3]]


class TestRadiusBoundary:
    def test_exactly_at_the_radius_connects(self) -> None:
        """The boundary counts as connected, defined rather than incidental."""
        positions = layout((1, 0.0, 0.0), (2, 1.5, 0.0), (3, 3.0, 0.0))
        assert track_ids(connected_components(positions, 1.5, 3)) == [[1, 2, 3]]

    def test_just_beyond_the_radius_does_not(self) -> None:
        positions = layout((1, 0.0, 0.0), (2, 1.5000001, 0.0), (3, 3.0000002, 0.0))
        assert connected_components(positions, 1.5, 3) == []

    def test_diagonal_separation_uses_euclidean_distance(self) -> None:
        """(3,4) is 5 apart, so a radius of 4 must not connect them."""
        positions = layout((1, 0.0, 0.0), (2, 3.0, 4.0), (3, 6.0, 8.0))
        assert connected_components(positions, 4.0, 3) == []
        assert track_ids(connected_components(positions, 5.0, 3)) == [[1, 2, 3]]


class TestChaining:
    def test_a_chain_forms_one_cluster(self) -> None:
        """Connectivity is transitive; this is the rule, not a defect.

        Four subjects 1.4 apart with a 1.5 radius are one cluster, even though
        the outermost two are 4.2 apart — nearly three radii.
        """
        positions = layout((1, 0.0, 0.0), (2, 1.4, 0.0), (3, 2.8, 0.0), (4, 4.2, 0.0))
        components = connected_components(positions, 1.5, 3)
        assert track_ids(components) == [[1, 2, 3, 4]]

        ends = positions[subject(1)].distance_to(positions[subject(4)])
        assert ends == pytest.approx(4.2)
        assert ends > 1.5 * 2, "the chain must span well beyond the radius"

    def test_a_long_chain_stays_one_cluster(self) -> None:
        positions = layout(*[(index, index * 1.4, 0.0) for index in range(1, 11)])
        assert track_ids(connected_components(positions, 1.5, 3)) == [list(range(1, 11))]

    def test_breaking_one_link_splits_the_chain(self) -> None:
        positions = layout(
            (1, 0.0, 0.0),
            (2, 1.4, 0.0),
            (3, 2.8, 0.0),
            (4, 20.0, 0.0),
            (5, 21.4, 0.0),
            (6, 22.8, 0.0),
        )
        assert track_ids(connected_components(positions, 1.5, 3)) == [[1, 2, 3], [4, 5, 6]]


class TestMembershipUniqueness:
    def test_a_subject_belongs_to_at_most_one_cluster(self) -> None:
        positions = layout(*[(index, index * 1.0, 0.0) for index in range(1, 8)])
        components = connected_components(positions, 1.5, 2)
        seen = [member for component in components for member in component]
        assert len(seen) == len(set(seen))

    def test_every_subject_appears_at_most_once_across_clusters(self) -> None:
        positions = layout(
            (1, 0.0, 0.0),
            (2, 1.0, 0.0),
            (3, 2.0, 0.0),
            (4, 30.0, 0.0),
            (5, 31.0, 0.0),
            (6, 32.0, 0.0),
        )
        components = connected_components(positions, 1.5, 3)
        flat = [member for component in components for member in component]
        assert len(flat) == len(set(flat)) == 6


class TestGraphCounters:
    def test_comparisons_are_n_choose_2(self) -> None:
        positions = layout(*[(index, index * 10.0, 0.0) for index in range(1, 6)])
        graph = build_cluster_graph(positions, 1.5)
        assert graph.comparisons == 10
        assert graph.edge_count == 0

    def test_edges_count_connected_pairs(self) -> None:
        """Three mutually-close subjects give three edges."""
        positions = layout((1, 0.0, 0.0), (2, 1.0, 0.0), (3, 0.0, 1.0))
        graph = build_cluster_graph(positions, 1.5)
        assert graph.edge_count == 3

    def test_a_chain_has_fewer_edges_than_a_clique(self) -> None:
        chain = layout((1, 0.0, 0.0), (2, 1.4, 0.0), (3, 2.8, 0.0))
        assert build_cluster_graph(chain, 1.5).edge_count == 2

    def test_neighbours_are_reported(self) -> None:
        positions = layout((1, 0.0, 0.0), (2, 1.0, 0.0), (3, 40.0, 0.0))
        graph = build_cluster_graph(positions, 1.5)
        assert graph.neighbours(subject(1)) == frozenset({subject(2)})
        assert graph.neighbours(subject(3)) == frozenset()


class TestDeterminism:
    def test_member_ordering_is_canonical(self) -> None:
        positions = layout((9, 0.0, 0.0), (2, 1.0, 0.0), (5, 0.0, 1.0))
        assert track_ids(connected_components(positions, 1.5, 3)) == [[2, 5, 9]]

    def test_input_ordering_does_not_change_the_result(self) -> None:
        forward = layout((1, 0.0, 0.0), (2, 1.0, 0.0), (3, 2.0, 0.0))
        backward = dict(reversed(list(forward.items())))
        assert connected_components(forward, 1.5, 3) == connected_components(backward, 1.5, 3)

    def test_cluster_ordering_is_stable(self) -> None:
        positions = layout(
            (7, 40.0, 0.0),
            (8, 41.0, 0.0),
            (9, 42.0, 0.0),
            (1, 0.0, 0.0),
            (2, 1.0, 0.0),
            (3, 2.0, 0.0),
        )
        assert track_ids(connected_components(positions, 1.5, 3)) == [[1, 2, 3], [7, 8, 9]]

    def test_repeated_runs_agree(self) -> None:
        positions = layout(
            *[(index, (index % 4) * 1.0, (index // 4) * 1.0) for index in range(1, 13)]
        )
        first = connected_components(positions, 1.5, 3)
        for _ in range(4):
            assert connected_components(positions, 1.5, 3) == first


class TestValidation:
    def test_rejects_a_non_positive_radius(self) -> None:
        with pytest.raises(CrowdError, match="cluster_radius must be positive"):
            build_cluster_graph(layout((1, 0.0, 0.0)), 0.0)

    def test_rejects_a_minimum_size_below_one(self) -> None:
        graph = build_cluster_graph(layout((1, 0.0, 0.0)), 1.5)
        with pytest.raises(CrowdError, match="minimum_size must be >= 1"):
            graph.components(0)

    def test_rejects_mixed_coordinate_spaces(self) -> None:
        positions = {
            subject(1): MotionPoint(0.0, 0.0, METRIC),
            subject(2): MotionPoint(0.5, 0.5, NORMALIZED),
        }
        with pytest.raises(CrowdError, match="Cannot measure separation"):
            build_cluster_graph(positions, 1.5)


class TestNormalizedSpace:
    def test_clustering_works_in_normalized_units(self) -> None:
        positions = layout((1, 0.10, 0.10), (2, 0.15, 0.10), (3, 0.20, 0.10), space=NORMALIZED)
        assert track_ids(connected_components(positions, 0.12, 3)) == [[1, 2, 3]]

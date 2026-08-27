"""Ground-truth-only cross-camera correspondence and occupancy for WILDTRACK."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atreides_vision.venue_validation.datasets.wildtrack.annotations import (
    load_frame_annotations,
)
from atreides_vision.venue_validation.datasets.wildtrack.cameras import CAMERA_IDS
from atreides_vision.venue_validation.datasets.wildtrack.correspondence import (
    WildtrackCrossCameraCorrespondence,
    WildtrackGroundTruthOccupancy,
    cross_camera_correspondences,
    ground_truth_occupancy,
)
from atreides_vision.venue_validation.datasets.wildtrack.time import WildtrackFrameId
from wildtrack_fixtures import make_view, write_annotation_json

_VISIBLE_BBOX = (10, 20, 30, 40)


def _person_record(
    person_id: int, *, position_id: int, visible_in: tuple[str, ...]
) -> dict[str, Any]:
    views = [
        make_view(view_num, _VISIBLE_BBOX if camera_id in visible_in else None)
        for view_num, camera_id in enumerate(CAMERA_IDS)
    ]
    return {"personID": person_id, "positionID": position_id, "views": views}


def _write_frame(
    path: Path, frame_value: int, records: list[dict[str, Any]]
) -> tuple[WildtrackFrameId, Path]:
    frame_id = WildtrackFrameId(frame_value)
    annotation_path = path / f"{frame_id.filename_stem}.json"
    write_annotation_json(annotation_path, records)
    return frame_id, annotation_path


class TestCrossCameraCorrespondences:
    def test_multi_camera_person(self, tmp_path: Path) -> None:
        records = [_person_record(1, position_id=0, visible_in=("C1", "C3"))]
        frame_id, annotation_path = _write_frame(tmp_path, 0, records)
        frame = load_frame_annotations(annotation_path, frame_id)
        correspondences = cross_camera_correspondences(frame)
        assert correspondences == (
            WildtrackCrossCameraCorrespondence(
                person_id=frame.persons[0].person_id, camera_ids=("C1", "C3")
            ),
        )
        assert correspondences[0].camera_count == 2
        assert correspondences[0].is_single_camera is False

    def test_single_camera_person(self, tmp_path: Path) -> None:
        records = [_person_record(1, position_id=0, visible_in=("C2",))]
        frame_id, annotation_path = _write_frame(tmp_path, 0, records)
        frame = load_frame_annotations(annotation_path, frame_id)
        correspondence = cross_camera_correspondences(frame)[0]
        assert correspondence.camera_count == 1
        assert correspondence.is_single_camera is True

    def test_person_visible_in_no_camera(self, tmp_path: Path) -> None:
        records = [_person_record(1, position_id=0, visible_in=())]
        frame_id, annotation_path = _write_frame(tmp_path, 0, records)
        frame = load_frame_annotations(annotation_path, frame_id)
        correspondence = cross_camera_correspondences(frame)[0]
        assert correspondence.camera_ids == ()
        assert correspondence.camera_count == 0
        assert correspondence.is_single_camera is False

    def test_ordering_follows_canonical_person_id_order_not_json_order(
        self, tmp_path: Path
    ) -> None:
        records = [
            _person_record(5, position_id=0, visible_in=("C1",)),
            _person_record(2, position_id=1, visible_in=("C2",)),
        ]
        frame_id, annotation_path = _write_frame(tmp_path, 0, records)
        frame = load_frame_annotations(annotation_path, frame_id)
        correspondences = cross_camera_correspondences(frame)
        assert [c.person_id.value for c in correspondences] == [2, 5]

    def test_empty_frame(self, tmp_path: Path) -> None:
        frame_id, annotation_path = _write_frame(tmp_path, 0, [])
        frame = load_frame_annotations(annotation_path, frame_id)
        assert cross_camera_correspondences(frame) == ()

    def test_excludes_view_with_single_sentinel_coordinate(self, tmp_path: Path) -> None:
        # A real-corpus shape: one bbox field is -1, the other three are real
        # pixel values. Per the official WILDTRACK visibility rule this view
        # is non-visible and must not appear in the correspondence.
        views = [
            make_view(view_num, _VISIBLE_BBOX if camera_id == "C1" else None)
            for view_num, camera_id in enumerate(CAMERA_IDS)
        ]
        c3_index = CAMERA_IDS.index("C3")
        views[c3_index] = {"viewNum": c3_index, "xmin": -1, "ymin": 20, "xmax": 30, "ymax": 40}
        records = [{"personID": 1, "positionID": 0, "views": views}]
        frame_id, annotation_path = _write_frame(tmp_path, 0, records)
        frame = load_frame_annotations(annotation_path, frame_id)
        correspondence = cross_camera_correspondences(frame)[0]
        assert correspondence.camera_ids == ("C1",)
        assert "C3" not in correspondence.camera_ids


class TestGroundTruthOccupancy:
    def test_full_scene_counts_every_annotated_person(self, tmp_path: Path) -> None:
        records = [
            _person_record(1, position_id=0, visible_in=("C1",)),
            _person_record(2, position_id=1, visible_in=()),
        ]
        frame_id, annotation_path = _write_frame(tmp_path, 0, records)
        frame = load_frame_annotations(annotation_path, frame_id)
        occupancy = ground_truth_occupancy(frame)
        assert occupancy.full_scene_person_count == 2

    def test_default_considers_all_cameras(self, tmp_path: Path) -> None:
        records = [
            _person_record(1, position_id=0, visible_in=("C1",)),
            _person_record(2, position_id=1, visible_in=()),
        ]
        frame_id, annotation_path = _write_frame(tmp_path, 0, records)
        frame = load_frame_annotations(annotation_path, frame_id)
        occupancy = ground_truth_occupancy(frame)
        assert occupancy.observed_person_count == 1
        assert occupancy.considered_camera_ids == CAMERA_IDS

    def test_restricted_camera_subset_excludes_persons_seen_only_elsewhere(
        self, tmp_path: Path
    ) -> None:
        records = [
            _person_record(1, position_id=0, visible_in=("C1",)),
            _person_record(2, position_id=1, visible_in=("C7",)),
        ]
        frame_id, annotation_path = _write_frame(tmp_path, 0, records)
        frame = load_frame_annotations(annotation_path, frame_id)
        occupancy = ground_truth_occupancy(frame, considered_camera_ids=("C1",))
        assert occupancy.observed_person_count == 1
        assert occupancy.full_scene_person_count == 2
        assert occupancy.considered_camera_ids == ("C1",)

    def test_considered_camera_ids_output_order_ignores_input_order(self, tmp_path: Path) -> None:
        records = [_person_record(1, position_id=0, visible_in=("C1",))]
        frame_id, annotation_path = _write_frame(tmp_path, 0, records)
        frame = load_frame_annotations(annotation_path, frame_id)
        occupancy = ground_truth_occupancy(frame, considered_camera_ids=("C7", "C1", "C3"))
        assert occupancy.considered_camera_ids == ("C1", "C3", "C7")

    def test_person_visible_in_multiple_considered_cameras_counts_once(
        self, tmp_path: Path
    ) -> None:
        records = [_person_record(1, position_id=0, visible_in=("C1", "C2", "C3"))]
        frame_id, annotation_path = _write_frame(tmp_path, 0, records)
        frame = load_frame_annotations(annotation_path, frame_id)
        occupancy = ground_truth_occupancy(frame, considered_camera_ids=("C1", "C2", "C3"))
        assert occupancy.observed_person_count == 1

    def test_frame_id_value_recorded(self, tmp_path: Path) -> None:
        frame_id, annotation_path = _write_frame(tmp_path, 10, [])
        frame = load_frame_annotations(annotation_path, frame_id)
        occupancy = ground_truth_occupancy(frame)
        assert occupancy.frame_id_value == 10

    def test_result_type(self, tmp_path: Path) -> None:
        frame_id, annotation_path = _write_frame(tmp_path, 0, [])
        frame = load_frame_annotations(annotation_path, frame_id)
        assert isinstance(ground_truth_occupancy(frame), WildtrackGroundTruthOccupancy)

    def test_single_sentinel_coordinate_excludes_camera_from_observed_occupancy(
        self, tmp_path: Path
    ) -> None:
        # Person is fully visible in C1 but their C2 record has one sentinel
        # coordinate; per the official rule that view is non-visible, so
        # restricting occupancy to C2 alone must not count this person.
        views = [
            make_view(view_num, _VISIBLE_BBOX if camera_id == "C1" else None)
            for view_num, camera_id in enumerate(CAMERA_IDS)
        ]
        c2_index = CAMERA_IDS.index("C2")
        views[c2_index] = {"viewNum": c2_index, "xmin": 10, "ymin": -1, "xmax": 30, "ymax": 40}
        records = [{"personID": 1, "positionID": 0, "views": views}]
        frame_id, annotation_path = _write_frame(tmp_path, 0, records)
        frame = load_frame_annotations(annotation_path, frame_id)
        occupancy = ground_truth_occupancy(frame, considered_camera_ids=("C2",))
        assert occupancy.observed_person_count == 0
        assert occupancy.full_scene_person_count == 1

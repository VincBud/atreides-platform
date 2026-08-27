"""A deterministic, non-identifying summary of one WILDTRACK dataset root."""

from __future__ import annotations

from pathlib import Path

from atreides_vision.venue_validation.datasets.wildtrack.cameras import CAMERA_COUNT, CAMERA_IDS
from atreides_vision.venue_validation.datasets.wildtrack.inspection import (
    ADAPTER_SCHEMA_VERSION,
    WildtrackDatasetInspection,
    inspect_dataset,
)
from wildtrack_fixtures import build_layout, make_view

_BBOX = (1, 2, 3, 4)


def _record(person_id: int, position_id: int, visible_camera: str) -> dict[str, object]:
    return {
        "personID": person_id,
        "positionID": position_id,
        "views": [
            make_view(view_num, _BBOX if camera_id == visible_camera else None)
            for view_num, camera_id in enumerate(CAMERA_IDS)
        ],
    }


class TestInspectDataset:
    def test_counts_and_frame_bounds(self, tmp_path: Path) -> None:
        layout = build_layout(
            tmp_path,
            image_frame_values=(0, 5, 10),
            annotation_frame_values=(0, 5),
            person_records_by_frame={
                0: [_record(1, 0, "C1"), _record(2, 24, "C2")],
                5: [_record(1, 0, "C1")],
            },
        )
        inspection = inspect_dataset(layout)
        assert inspection.image_frame_count == 3
        assert inspection.annotated_frame_count == 2
        assert inspection.first_annotated_frame_value == 0
        assert inspection.last_annotated_frame_value == 5
        assert inspection.total_person_observation_count == 3
        assert inspection.distinct_person_id_count == 2
        assert inspection.camera_ids == CAMERA_IDS
        assert inspection.calibrated_camera_count == CAMERA_COUNT

    def test_annotation_cadence_is_two_hz(self, tmp_path: Path) -> None:
        layout = build_layout(tmp_path)
        inspection = inspect_dataset(layout)
        assert inspection.annotation_cadence_hz == 2.0

    def test_empty_annotated_frames_still_counted(self, tmp_path: Path) -> None:
        layout = build_layout(
            tmp_path,
            image_frame_values=(0, 5),
            annotation_frame_values=(0, 5),
            person_records_by_frame={},
        )
        inspection = inspect_dataset(layout)
        assert inspection.total_person_observation_count == 0
        assert inspection.distinct_person_id_count == 0
        assert inspection.annotated_frame_count == 2

    def test_determinism_across_repeated_calls(self, tmp_path: Path) -> None:
        layout = build_layout(
            tmp_path,
            annotation_frame_values=(0, 5),
            person_records_by_frame={
                0: [_record(3, 0, "C1"), _record(1, 24, "C2")],
                5: [_record(2, 48, "C3")],
            },
        )
        first = inspect_dataset(layout)
        second = inspect_dataset(layout)
        assert first == second
        assert first.to_record() == second.to_record()

    def test_result_type(self, tmp_path: Path) -> None:
        layout = build_layout(tmp_path)
        assert isinstance(inspect_dataset(layout), WildtrackDatasetInspection)


class TestToRecordHasNoPathTimestampOrAccuracyFigure:
    def test_record_shape(self, tmp_path: Path) -> None:
        layout = build_layout(tmp_path)
        record = inspect_dataset(layout).to_record()
        assert record["adapter_schema_version"] == ADAPTER_SCHEMA_VERSION
        assert record["dataset_name"] == "WILDTRACK"
        forbidden_keys = {"path", "root", "timestamp", "accuracy", "precision", "recall"}
        assert forbidden_keys.isdisjoint(record.keys())

    def test_record_values_contain_no_filesystem_path(self, tmp_path: Path) -> None:
        layout = build_layout(tmp_path)
        record = inspect_dataset(layout).to_record()
        for value in record.values():
            assert str(tmp_path) not in str(value)

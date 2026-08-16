"""Manual annotation: what parses, what is refused, and what is never fabricated.

The rule under test everywhere: no manual metric exists unless a human supplied
the numbers it is computed from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from atreides_vision.errors import ValidationError
from atreides_vision.validation.ground_truth import (
    CONTINUITY_ANNOTATION_FILENAME,
    CONTINUITY_COLUMNS,
    FRAME_ANNOTATION_FILENAME,
    FRAME_COLUMNS,
    FrameAnnotation,
    GroundTruthStatus,
    ManualGroundTruth,
    load_manual_ground_truth,
    write_annotation_templates,
)

FRAME_ROWS = ((1.0, 10), (2.0, 20), (3.0, 30))
CONTINUITY_ROWS = (("w01", 0.0, 8.0), ("w02", 10.0, 18.0))


def templates(tmp_path: Path, clip_id: str = "clip") -> dict[str, Path]:
    return write_annotation_templates(tmp_path, clip_id, FRAME_ROWS, CONTINUITY_ROWS)


def write_frame_csv(tmp_path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path = tmp_path / FRAME_ANNOTATION_FILENAME
    lines = [",".join(FRAME_COLUMNS)]
    lines.extend(",".join(str(row.get(column, "")) for column in FRAME_COLUMNS) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_continuity_csv(tmp_path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path = tmp_path / CONTINUITY_ANNOTATION_FILENAME
    lines = [",".join(CONTINUITY_COLUMNS)]
    lines.extend(
        ",".join(str(row.get(column, "")) for column in CONTINUITY_COLUMNS) for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def frame_row(**overrides: object) -> dict[str, object]:
    row = {
        "clip_id": "clip",
        "timestamp_seconds": 1.0,
        "frame_index": 10,
        "visible_people": 8,
        "correct_person_detections": 6,
        "false_positive_detections": 1,
        "missed_people": 2,
        "notes": "",
    }
    row.update(overrides)
    return row


def continuity_row(**overrides: object) -> dict[str, object]:
    row = {
        "clip_id": "clip",
        "sequence_id": "w01",
        "start_time": 0.0,
        "end_time": 8.0,
        "target_label": "red jacket",
        "observed_track_ids": "3 11",
        "id_switch_count": 1,
        "fragment_count": 2,
        "lost_track_events": 0,
        "notes": "",
    }
    row.update(overrides)
    return row


# --- templates -------------------------------------------------------------


def test_templates_are_written_pre_filled_but_unannotated(tmp_path: Path) -> None:
    paths = templates(tmp_path)

    text = paths["frames"].read_text(encoding="utf-8")
    assert text.splitlines()[0] == ",".join(FRAME_COLUMNS)
    # Coordinates pre-filled, annotation columns empty.
    assert "clip,1.000,10,,,,," in text

    ground_truth = load_manual_ground_truth(tmp_path, "clip")
    assert ground_truth.status is GroundTruthStatus.NOT_ANNOTATED
    assert ground_truth.frame_rows_available == 3
    assert ground_truth.continuity_rows_available == 2


def test_templates_never_overwrite_existing_annotation(tmp_path: Path) -> None:
    templates(tmp_path)
    write_frame_csv(tmp_path, [frame_row()])
    before = (tmp_path / FRAME_ANNOTATION_FILENAME).read_text(encoding="utf-8")

    templates(tmp_path)

    assert (tmp_path / FRAME_ANNOTATION_FILENAME).read_text(encoding="utf-8") == before


# --- absent annotation -----------------------------------------------------


def test_no_annotation_files_at_all_is_not_an_error(tmp_path: Path) -> None:
    ground_truth = load_manual_ground_truth(tmp_path, "clip")

    assert ground_truth.status is GroundTruthStatus.NOT_ANNOTATED
    assert ground_truth.detection_recall_sample is None
    assert ground_truth.detection_precision_sample is None
    assert ground_truth.total_id_switches is None


def test_unannotated_record_emits_no_metrics(tmp_path: Path) -> None:
    templates(tmp_path)

    record = load_manual_ground_truth(tmp_path, "clip").to_record()

    assert record["manual_ground_truth_status"] == "NOT_ANNOTATED"
    assert "detection_sample" not in record
    assert "continuity_sample" not in record
    assert "no human annotation exists" in record["note"]


# --- recall and precision --------------------------------------------------


def test_recall_and_precision_over_annotated_frames(tmp_path: Path) -> None:
    write_frame_csv(
        tmp_path,
        [
            frame_row(visible_people=10, correct_person_detections=8, false_positive_detections=2),
            frame_row(visible_people=10, correct_person_detections=7, false_positive_detections=0),
        ],
    )

    ground_truth = load_manual_ground_truth(tmp_path, "clip")

    assert ground_truth.visible_people == 20
    assert ground_truth.correct_detections == 15
    assert ground_truth.detection_recall_sample == pytest.approx(0.75)
    assert ground_truth.detection_precision_sample == pytest.approx(15 / 17)


def test_zero_visible_people_yields_no_recall_rather_than_zero(tmp_path: Path) -> None:
    """Recall of nothing is undefined; 0.0 would read as a total miss."""
    write_frame_csv(
        tmp_path,
        [
            frame_row(
                visible_people=0,
                correct_person_detections=0,
                false_positive_detections=0,
                missed_people=0,
            )
        ],
    )

    ground_truth = load_manual_ground_truth(tmp_path, "clip")

    assert ground_truth.status is not GroundTruthStatus.NOT_ANNOTATED
    assert ground_truth.detection_recall_sample is None
    assert ground_truth.detection_precision_sample is None


def test_a_false_positive_only_frame_yields_zero_precision(tmp_path: Path) -> None:
    """Nothing correct and three spurious boxes: precision really is 0.0."""
    write_frame_csv(
        tmp_path,
        [
            frame_row(
                visible_people=0,
                correct_person_detections=0,
                false_positive_detections=3,
                missed_people=0,
            )
        ],
    )

    ground_truth = load_manual_ground_truth(tmp_path, "clip")

    assert ground_truth.detection_precision_sample == pytest.approx(0.0)
    assert ground_truth.detection_recall_sample is None


def test_partial_annotation_is_reported_as_partial(tmp_path: Path) -> None:
    templates(tmp_path)
    # Overwrite with one annotated row and two blank ones.
    write_frame_csv(
        tmp_path,
        [
            frame_row(),
            {"clip_id": "clip", "timestamp_seconds": 2.0, "frame_index": 20},
            {"clip_id": "clip", "timestamp_seconds": 3.0, "frame_index": 30},
        ],
    )

    ground_truth = load_manual_ground_truth(tmp_path, "clip")

    assert ground_truth.status is GroundTruthStatus.PARTIALLY_ANNOTATED
    assert len(ground_truth.frames) == 1
    assert ground_truth.frame_rows_available == 3


def test_fully_annotated_is_reported_as_annotated(tmp_path: Path) -> None:
    write_frame_csv(tmp_path, [frame_row(), frame_row(frame_index=20)])
    write_continuity_csv(tmp_path, [continuity_row(), continuity_row(sequence_id="w02")])

    ground_truth = load_manual_ground_truth(tmp_path, "clip")

    assert ground_truth.status is GroundTruthStatus.ANNOTATED


# --- continuity ------------------------------------------------------------


def test_continuity_annotation_parses(tmp_path: Path) -> None:
    write_continuity_csv(
        tmp_path,
        [
            continuity_row(id_switch_count=1, fragment_count=2, lost_track_events=1),
            continuity_row(
                sequence_id="w02",
                start_time=10.0,
                end_time=18.0,
                observed_track_ids="5;9;14",
                id_switch_count=2,
                fragment_count=0,
                lost_track_events=0,
            ),
        ],
    )

    ground_truth = load_manual_ground_truth(tmp_path, "clip")

    assert len(ground_truth.continuity) == 2
    assert ground_truth.total_id_switches == 3
    assert ground_truth.total_fragments == 2
    assert ground_truth.total_lost_track_events == 1
    assert ground_truth.annotated_continuity_seconds == pytest.approx(16.0)
    assert ground_truth.continuity[0].observed_track_ids == (3, 11)
    assert ground_truth.continuity[1].observed_track_ids == (5, 9, 14)


def test_continuity_record_states_that_target_label_is_not_an_identity(tmp_path: Path) -> None:
    write_continuity_csv(tmp_path, [continuity_row()])

    record = load_manual_ground_truth(tmp_path, "clip").to_record()

    assert "no ReID is implied" in record["continuity_sample"]["scope"]


# --- malformed input -------------------------------------------------------


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / FRAME_ANNOTATION_FILENAME
    path.write_text("clip_id,frame_index\nclip,10\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="missing required column"):
        load_manual_ground_truth(tmp_path, "clip")


def test_non_numeric_count_is_rejected(tmp_path: Path) -> None:
    write_frame_csv(tmp_path, [frame_row(visible_people="lots")])

    with pytest.raises(ValidationError, match="must be a whole number"):
        load_manual_ground_truth(tmp_path, "clip")


def test_negative_count_is_rejected(tmp_path: Path) -> None:
    write_frame_csv(tmp_path, [frame_row(missed_people=-1)])

    with pytest.raises(ValidationError, match="non-negative"):
        load_manual_ground_truth(tmp_path, "clip")


def test_more_correct_detections_than_visible_people_is_rejected(tmp_path: Path) -> None:
    write_frame_csv(tmp_path, [frame_row(visible_people=3, correct_person_detections=5)])

    with pytest.raises(ValidationError, match="exceeds visible_people"):
        load_manual_ground_truth(tmp_path, "clip")


def test_a_half_filled_row_is_rejected_rather_than_completed(tmp_path: Path) -> None:
    """Treating it as blank would drop the annotator's work; completing it would invent."""
    write_frame_csv(
        tmp_path, [frame_row(correct_person_detections="", false_positive_detections="")]
    )

    with pytest.raises(ValidationError, match="partially filled row"):
        load_manual_ground_truth(tmp_path, "clip")


def test_non_numeric_track_id_list_is_rejected(tmp_path: Path) -> None:
    write_continuity_csv(tmp_path, [continuity_row(observed_track_ids="3 red")])

    with pytest.raises(ValidationError, match="observed_track_ids"):
        load_manual_ground_truth(tmp_path, "clip")


def test_end_time_before_start_time_is_rejected(tmp_path: Path) -> None:
    write_continuity_csv(tmp_path, [continuity_row(start_time=10.0, end_time=2.0)])

    with pytest.raises(ValidationError, match="precedes"):
        load_manual_ground_truth(tmp_path, "clip")


def test_frame_annotation_validates_at_construction() -> None:
    with pytest.raises(ValidationError, match="exceeds visible_people"):
        FrameAnnotation(
            clip_id="clip",
            timestamp_seconds=1.0,
            frame_index=1,
            visible_people=2,
            correct_person_detections=3,
            false_positive_detections=0,
            missed_people=0,
        )


def test_empty_ground_truth_object_reports_not_annotated() -> None:
    assert ManualGroundTruth(clip_id="clip").status is GroundTruthStatus.NOT_ANNOTATED

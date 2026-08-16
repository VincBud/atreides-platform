"""Manual ground truth: the only place in this package a human's judgement lives.

Everything else in the validation harness measures Atreides against itself. That
can show how the *system* behaved; it cannot show whether the system was right.
Detection recall, precision, real ID switches and real fragmentation all require
somebody to look at a frame and count what is actually there, and this module is
the boundary those numbers come in through.

Two consequences are enforced rather than documented:

**Absent annotation is reported as absent.** A clip nobody annotated reports
``manual_ground_truth_status = NOT_ANNOTATED`` and emits no recall, no precision
and no ID-switch figure at all. There is no "0 so far" and no empty-but-present
section — a report showing 0.0 recall next to a proxy metric would be read as a
measurement.

**Manual metrics never merge with proxy metrics.** They serialise under their own
key, with their own measurement class, and the CSV/Markdown reporting keeps them
in separate sections.

Annotation files
----------------
Two CSVs per clip, generated pre-filled with the deterministic sample plan's rows
and blank annotation columns. A blank row is an un-annotated row and contributes
nothing to any statistic.

``frame_annotations.csv`` — one row per sampled still frame::

    clip_id,timestamp_seconds,frame_index,visible_people,correct_person_detections,
    false_positive_detections,missed_people,notes

``continuity_annotations.csv`` — one row per tracked subject per continuity window::

    clip_id,sequence_id,start_time,end_time,target_label,observed_track_ids,
    id_switch_count,fragment_count,lost_track_events,notes

``target_label`` is a scratch label an annotator uses to keep one subject
straight within one window — "red jacket", "subject_a". It is **run-local
annotation, never identity**: nothing reads it back into the pipeline, it is not
stored against any person, and it is not ReID. Two windows using the same label
are not asserting the same person.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from atreides_vision.errors import ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "CONTINUITY_ANNOTATION_FILENAME",
    "CONTINUITY_COLUMNS",
    "FRAME_ANNOTATION_FILENAME",
    "FRAME_COLUMNS",
    "ContinuityAnnotation",
    "FrameAnnotation",
    "GroundTruthStatus",
    "ManualGroundTruth",
    "load_manual_ground_truth",
    "write_annotation_templates",
]

FRAME_ANNOTATION_FILENAME = "frame_annotations.csv"
CONTINUITY_ANNOTATION_FILENAME = "continuity_annotations.csv"

FRAME_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "timestamp_seconds",
    "frame_index",
    "visible_people",
    "correct_person_detections",
    "false_positive_detections",
    "missed_people",
    "notes",
)

CONTINUITY_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "sequence_id",
    "start_time",
    "end_time",
    "target_label",
    "observed_track_ids",
    "id_switch_count",
    "fragment_count",
    "lost_track_events",
    "notes",
)

_FRAME_REQUIRED = (
    "visible_people",
    "correct_person_detections",
    "false_positive_detections",
    "missed_people",
)
_CONTINUITY_REQUIRED = ("id_switch_count", "fragment_count", "lost_track_events")

_PRECISION = 4


class GroundTruthStatus(StrEnum):
    """How much of a clip's manual annotation exists."""

    #: No annotation rows were filled in. No manual metric is emitted.
    NOT_ANNOTATED = "NOT_ANNOTATED"
    #: Some sampled rows were filled in. Metrics are emitted over those rows
    #: only, with the sample size attached so nobody reads them as a census.
    PARTIALLY_ANNOTATED = "PARTIALLY_ANNOTATED"
    #: Every sampled row was filled in.
    ANNOTATED = "ANNOTATED"


@dataclass(frozen=True, slots=True)
class FrameAnnotation:
    """One human-reviewed still frame."""

    clip_id: str
    timestamp_seconds: float
    frame_index: int
    visible_people: int
    correct_person_detections: int
    false_positive_detections: int
    missed_people: int
    notes: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("visible_people", self.visible_people),
            ("correct_person_detections", self.correct_person_detections),
            ("false_positive_detections", self.false_positive_detections),
            ("missed_people", self.missed_people),
        ):
            if value < 0:
                raise ValidationError(
                    f"{self.clip_id} frame {self.frame_index}: {name} must be non-negative, "
                    f"got {value}"
                )
        if self.correct_person_detections > self.visible_people:
            raise ValidationError(
                f"{self.clip_id} frame {self.frame_index}: correct_person_detections "
                f"({self.correct_person_detections}) exceeds visible_people "
                f"({self.visible_people}). A correct detection is one of the visible people, "
                "so this row describes an impossible frame."
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "timestamp_seconds": self.timestamp_seconds,
            "frame_index": self.frame_index,
            "visible_people": self.visible_people,
            "correct_person_detections": self.correct_person_detections,
            "false_positive_detections": self.false_positive_detections,
            "missed_people": self.missed_people,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ContinuityAnnotation:
    """One human-reviewed subject within one short continuity window."""

    clip_id: str
    sequence_id: str
    start_time: float
    end_time: float
    target_label: str
    observed_track_ids: tuple[int, ...]
    id_switch_count: int
    fragment_count: int
    lost_track_events: int
    notes: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("id_switch_count", self.id_switch_count),
            ("fragment_count", self.fragment_count),
            ("lost_track_events", self.lost_track_events),
        ):
            if value < 0:
                raise ValidationError(
                    f"{self.clip_id}/{self.sequence_id}: {name} must be non-negative, got {value}"
                )
        if self.end_time < self.start_time:
            raise ValidationError(
                f"{self.clip_id}/{self.sequence_id}: end_time {self.end_time} precedes "
                f"start_time {self.start_time}."
            )

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    def to_record(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "sequence_id": self.sequence_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "target_label": self.target_label,
            "observed_track_ids": list(self.observed_track_ids),
            "id_switch_count": self.id_switch_count,
            "fragment_count": self.fragment_count,
            "lost_track_events": self.lost_track_events,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ManualGroundTruth:
    """Every human annotation for one clip, and what can be derived from it.

    ``status`` is the field a consumer must read first. Every derived metric is
    ``None`` when nothing was annotated, and the serialised form omits the
    metrics entirely rather than emitting nulls that could be plotted as zeros.
    """

    clip_id: str
    frames: tuple[FrameAnnotation, ...] = ()
    continuity: tuple[ContinuityAnnotation, ...] = ()
    #: Rows the templates offered, so partial annotation is visible as partial.
    frame_rows_available: int = 0
    continuity_rows_available: int = 0

    @property
    def status(self) -> GroundTruthStatus:
        annotated = len(self.frames) + len(self.continuity)
        if annotated == 0:
            return GroundTruthStatus.NOT_ANNOTATED
        available = self.frame_rows_available + self.continuity_rows_available
        if available and annotated < available:
            return GroundTruthStatus.PARTIALLY_ANNOTATED
        return GroundTruthStatus.ANNOTATED

    @property
    def visible_people(self) -> int:
        return sum(frame.visible_people for frame in self.frames)

    @property
    def correct_detections(self) -> int:
        return sum(frame.correct_person_detections for frame in self.frames)

    @property
    def false_positives(self) -> int:
        return sum(frame.false_positive_detections for frame in self.frames)

    @property
    def missed_people(self) -> int:
        return sum(frame.missed_people for frame in self.frames)

    @property
    def detection_recall_sample(self) -> float | None:
        """Correct detections / visible people, over annotated frames only.

        ``None`` when no frame was annotated, and also when every annotated
        frame had zero visible people: recall of nothing is undefined, and 0.0
        would read as a total miss rather than as an unanswerable question.
        """
        if not self.frames or self.visible_people == 0:
            return None
        return self.correct_detections / self.visible_people

    @property
    def detection_precision_sample(self) -> float | None:
        """Correct detections / (correct + false positives), over annotated frames.

        ``None`` when nothing was annotated or when no detection at all was
        reported on the annotated frames. A frame with false positives and no
        correct detections yields ``0.0``, which is a real measurement.
        """
        if not self.frames:
            return None
        denominator = self.correct_detections + self.false_positives
        if denominator == 0:
            return None
        return self.correct_detections / denominator

    @property
    def total_id_switches(self) -> int | None:
        if not self.continuity:
            return None
        return sum(entry.id_switch_count for entry in self.continuity)

    @property
    def total_fragments(self) -> int | None:
        if not self.continuity:
            return None
        return sum(entry.fragment_count for entry in self.continuity)

    @property
    def total_lost_track_events(self) -> int | None:
        if not self.continuity:
            return None
        return sum(entry.lost_track_events for entry in self.continuity)

    @property
    def annotated_continuity_seconds(self) -> float | None:
        if not self.continuity:
            return None
        return sum(entry.duration_seconds for entry in self.continuity)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "measurement_class": "manual_ground_truth",
            "manual_ground_truth_status": self.status.value,
            "annotated_frame_rows": len(self.frames),
            "available_frame_rows": self.frame_rows_available,
            "annotated_continuity_rows": len(self.continuity),
            "available_continuity_rows": self.continuity_rows_available,
        }
        if self.status is GroundTruthStatus.NOT_ANNOTATED:
            record["note"] = (
                "no human annotation exists for this clip, so no detection recall, precision "
                "or true ID-switch figure is reported. Nothing here is inferred from the "
                "automated proxy metrics."
            )
            return record

        if self.frames:
            record["detection_sample"] = {
                "annotated_frames": len(self.frames),
                "visible_people": self.visible_people,
                "correct_person_detections": self.correct_detections,
                "false_positive_detections": self.false_positives,
                "missed_people": self.missed_people,
                "detection_recall_sample": _round(self.detection_recall_sample),
                "precision_sample": _round(self.detection_precision_sample),
                "scope": (
                    "a small deterministic sample of still frames, not a dataset-level "
                    "detection benchmark."
                ),
            }
        if self.continuity:
            record["continuity_sample"] = {
                "annotated_sequences": len(self.continuity),
                "annotated_seconds": _round(self.annotated_continuity_seconds),
                "id_switches": self.total_id_switches,
                "fragments": self.total_fragments,
                "lost_track_events": self.total_lost_track_events,
                "scope": (
                    "hand-followed subjects within short windows. target_label is run-local "
                    "annotation only and is never an identity; no ReID is implied."
                ),
            }
        return record


def _round(value: float | None, digits: int = _PRECISION) -> float | None:
    return None if value is None else round(value, digits)


def write_annotation_templates(
    directory: Path | str,
    clip_id: str,
    frame_rows: Sequence[tuple[float, int]],
    continuity_rows: Sequence[tuple[str, float, float]],
) -> dict[str, Path]:
    """Write pre-filled, unannotated CSV templates for one clip.

    Rows carry the deterministic sample plan's coordinates and blank annotation
    columns, so an annotator fills in numbers rather than transcribing
    timestamps.

    **Existing files are never overwritten.** Re-running validation on a clip
    that has already been annotated must not destroy the annotation, and a
    silent overwrite would be discovered only when the report changed.

    Args:
        directory: Where the clip's annotation files live.
        clip_id: The clip these rows belong to.
        frame_rows: ``(timestamp_seconds, frame_index)`` per sampled still.
        continuity_rows: ``(sequence_id, start_time, end_time)`` per window.

    Returns:
        Mapping of ``"frames"`` and ``"continuity"`` to the file paths.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    frames_path = target / FRAME_ANNOTATION_FILENAME
    continuity_path = target / CONTINUITY_ANNOTATION_FILENAME

    if not frames_path.exists():
        _write_csv(
            frames_path,
            FRAME_COLUMNS,
            [
                {
                    "clip_id": clip_id,
                    "timestamp_seconds": f"{timestamp:.3f}",
                    "frame_index": frame_index,
                }
                for timestamp, frame_index in frame_rows
            ],
        )
    else:
        logger.info("Keeping existing annotation file %s", frames_path)

    if not continuity_path.exists():
        _write_csv(
            continuity_path,
            CONTINUITY_COLUMNS,
            [
                {
                    "clip_id": clip_id,
                    "sequence_id": sequence_id,
                    "start_time": f"{start:.3f}",
                    "end_time": f"{end:.3f}",
                }
                for sequence_id, start, end in continuity_rows
            ],
        )
    else:
        logger.info("Keeping existing annotation file %s", continuity_path)

    return {"frames": frames_path, "continuity": continuity_path}


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})
    except OSError as exc:
        raise ValidationError(f"Could not write annotation template {path}: {exc}") from exc


def load_manual_ground_truth(directory: Path | str, clip_id: str) -> ManualGroundTruth:
    """Read whatever annotation exists for one clip.

    A missing directory or missing file is a normal state, not an error: most
    clips are never annotated, and that is reported as ``NOT_ANNOTATED``. A file
    that *exists* but cannot be parsed **is** an error — a malformed annotation
    silently skipped would understate exactly the metric it was written to
    supply.

    Raises:
        ValidationError: for a missing required column, a non-integer count, a
            negative count, or a row asserting more correct detections than
            visible people.
    """
    target = Path(directory)
    frames_path = target / FRAME_ANNOTATION_FILENAME
    continuity_path = target / CONTINUITY_ANNOTATION_FILENAME

    frames, frame_rows_available = _load_frame_annotations(frames_path, clip_id)
    continuity, continuity_rows_available = _load_continuity_annotations(continuity_path, clip_id)

    return ManualGroundTruth(
        clip_id=clip_id,
        frames=frames,
        continuity=continuity,
        frame_rows_available=frame_rows_available,
        continuity_rows_available=continuity_rows_available,
    )


def _read_rows(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            missing = [column for column in columns if column not in fieldnames]
            if missing:
                raise ValidationError(
                    f"{path} is missing required column(s): {missing}. Expected header: "
                    f"{','.join(columns)}"
                )
            return [{key: (value or "") for key, value in row.items() if key} for row in reader]
    except csv.Error as exc:
        raise ValidationError(f"{path} is not readable as CSV: {exc}") from exc
    except OSError as exc:
        raise ValidationError(f"Could not read {path}: {exc}") from exc


def _is_blank(row: dict[str, str], required: Sequence[str]) -> bool:
    """Whether this row is an untouched template row.

    All-or-nothing: a row with some required fields filled and others empty is
    a half-finished annotation, and treating it as blank would quietly drop the
    annotator's work while treating it as complete would fabricate the rest.
    """
    filled = [name for name in required if row.get(name, "").strip()]
    return not filled


def _require_int(row: dict[str, str], name: str, where: str) -> int:
    raw = row.get(name, "").strip()
    if not raw:
        raise ValidationError(
            f"{where}: '{name}' is empty but other columns in this row are filled in. "
            "Complete the row or clear it entirely — a partially filled row would make the "
            "derived metric describe something other than what was reviewed."
        )
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValidationError(f"{where}: '{name}' must be a whole number, got {raw!r}") from exc
    return value


def _require_float(row: dict[str, str], name: str, where: str) -> float:
    raw = row.get(name, "").strip()
    if not raw:
        raise ValidationError(f"{where}: '{name}' must be a number, got an empty value")
    try:
        return float(raw)
    except ValueError as exc:
        raise ValidationError(f"{where}: '{name}' must be a number, got {raw!r}") from exc


def _load_frame_annotations(path: Path, clip_id: str) -> tuple[tuple[FrameAnnotation, ...], int]:
    rows = _read_rows(path, FRAME_COLUMNS)
    annotations: list[FrameAnnotation] = []
    for number, row in enumerate(rows, start=2):
        if _is_blank(row, _FRAME_REQUIRED):
            continue
        where = f"{path} line {number}"
        annotations.append(
            FrameAnnotation(
                clip_id=row.get("clip_id", "").strip() or clip_id,
                timestamp_seconds=_require_float(row, "timestamp_seconds", where),
                frame_index=_require_int(row, "frame_index", where),
                visible_people=_require_int(row, "visible_people", where),
                correct_person_detections=_require_int(row, "correct_person_detections", where),
                false_positive_detections=_require_int(row, "false_positive_detections", where),
                missed_people=_require_int(row, "missed_people", where),
                notes=row.get("notes", "").strip(),
            )
        )
    return tuple(annotations), len(rows)


def _load_continuity_annotations(
    path: Path, clip_id: str
) -> tuple[tuple[ContinuityAnnotation, ...], int]:
    rows = _read_rows(path, CONTINUITY_COLUMNS)
    annotations: list[ContinuityAnnotation] = []
    for number, row in enumerate(rows, start=2):
        if _is_blank(row, _CONTINUITY_REQUIRED):
            continue
        where = f"{path} line {number}"
        annotations.append(
            ContinuityAnnotation(
                clip_id=row.get("clip_id", "").strip() or clip_id,
                sequence_id=row.get("sequence_id", "").strip(),
                start_time=_require_float(row, "start_time", where),
                end_time=_require_float(row, "end_time", where),
                target_label=row.get("target_label", "").strip(),
                observed_track_ids=_parse_track_ids(row.get("observed_track_ids", ""), where),
                id_switch_count=_require_int(row, "id_switch_count", where),
                fragment_count=_require_int(row, "fragment_count", where),
                lost_track_events=_require_int(row, "lost_track_events", where),
                notes=row.get("notes", "").strip(),
            )
        )
    return tuple(annotations), len(rows)


def _parse_track_ids(raw: str, where: str) -> tuple[int, ...]:
    """Parse a space- or semicolon-separated list of observed track ids.

    Commas are not a separator: the file is CSV, and a comma-separated list
    inside a cell is a quoting problem waiting to happen.
    """
    text = raw.strip().replace(";", " ")
    if not text:
        return ()
    ids: list[int] = []
    for token in text.split():
        try:
            ids.append(int(token))
        except ValueError as exc:
            raise ValidationError(
                f"{where}: 'observed_track_ids' must be whole numbers separated by spaces or "
                f"semicolons, got {token!r}"
            ) from exc
    return tuple(ids)

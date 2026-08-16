"""Where validation artifacts live, and how they are written deterministically.

One type owns the directory structure so the runner, the analysis pass, the
report writer and the CLI all agree about it without passing paths around::

    <root>/
        runs/<clip_id>/            tracked.mp4, the three streams, run.json
        manual-ground-truth/<clip_id>/
                                   frames/, contact_sheet.jpg, sample_plan.json,
                                   frame_annotations.csv, continuity_annotations.csv
        validation_summary.json
        validation_summary.csv
        V0.3.6_REPORT.md

Everything under the root is generated and git-ignored. Nothing here writes into
``data/`` or anywhere else in the repository.

Determinism is a property of the writers, not a convention: JSON is written with
sorted keys and a fixed separator, and the CSV writer takes an explicit column
order. Two runs over the same inputs produce byte-identical reports, which is
what makes a diff of two validation runs meaningful.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atreides_vision.errors import ValidationError

__all__ = [
    "REPORT_FILENAME",
    "SUMMARY_CSV_FILENAME",
    "SUMMARY_JSON_FILENAME",
    "ValidationLayout",
    "write_csv",
    "write_json",
    "write_text",
]

SUMMARY_JSON_FILENAME = "validation_summary.json"
SUMMARY_CSV_FILENAME = "validation_summary.csv"
REPORT_FILENAME = "V0.3.6_REPORT.md"


@dataclass(frozen=True, slots=True)
class ValidationLayout:
    """The artifact root, and every path derived from it."""

    root: Path

    @classmethod
    def at(cls, root: Path | str) -> ValidationLayout:
        return cls(root=Path(root))

    @property
    def runs_root(self) -> Path:
        return self.root / "runs"

    @property
    def ground_truth_root(self) -> Path:
        return self.root / "manual-ground-truth"

    @property
    def summary_json_path(self) -> Path:
        return self.root / SUMMARY_JSON_FILENAME

    @property
    def summary_csv_path(self) -> Path:
        return self.root / SUMMARY_CSV_FILENAME

    @property
    def report_path(self) -> Path:
        return self.root / REPORT_FILENAME

    def run_dir(self, clip_id: str) -> Path:
        return self.runs_root / clip_id

    def ground_truth_dir(self, clip_id: str) -> Path:
        return self.ground_truth_root / clip_id

    def relative(self, path: Path) -> str:
        """``path`` as a root-relative POSIX string, for the report.

        Absolute paths in a committed-adjacent report would leak whichever
        machine produced it and would not resolve on any other.
        """
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def ensure(self) -> ValidationLayout:
        """Create the root and its two subtrees. Idempotent."""
        for directory in (self.root, self.runs_root, self.ground_truth_root):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ValidationError(
                    f"Could not create validation output directory {directory}: {exc}"
                ) from exc
        return self


def write_json(path: Path | str, record: Any) -> Path:
    """Write one JSON document with sorted keys and a trailing newline."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Could not serialise {target}: {exc}") from exc
    return write_text(target, text)


def write_text(path: Path | str, text: str) -> Path:
    """Write text with ``\\n`` line endings on every platform."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise ValidationError(f"Could not write {target}: {exc}") from exc
    return target


def write_csv(path: Path | str, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> Path:
    """Write a CSV with an explicit column order and a fixed line terminator.

    Column order is the caller's, never the dictionaries' insertion order: a
    summary whose columns move between runs cannot be diffed.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: _cell(row.get(column)) for column in columns})
    except OSError as exc:
        raise ValidationError(f"Could not write {target}: {exc}") from exc
    return target


def _cell(value: Any) -> Any:
    """Render one CSV cell.

    ``None`` becomes an empty cell rather than the string ``"None"`` or ``0``.
    An absent measurement must not arrive in a spreadsheet as a number.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def read_json(path: Path | str) -> dict[str, Any]:
    """Read one JSON document written by :func:`write_json`."""
    target = Path(path)
    if not target.exists():
        raise ValidationError(f"Expected {target}, but it does not exist.")
    try:
        record = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{target} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ValidationError(f"Could not read {target}: {exc}") from exc
    if not isinstance(record, dict):
        raise ValidationError(f"{target} must contain a JSON object.")
    return record

"""Deterministic artifacts for the offline WILDTRACK evaluator (V0.4.9b).

Two files, both byte-for-byte reproducible from the same runtime snapshots and
the same ground truth:

* ``wildtrack_evaluation.jsonl`` — one header record (schema version, selected
  cameras, frozen radius) followed by one record per scored venue instant, in
  instant order;
* ``wildtrack_validation_summary.json`` — the whole-run
  :class:`~...evaluation.models.WildtrackEvaluationSummary`.

No wall clock, process id, hostname, absolute path, UUID or run duration enters
either file. Artifacts are named, never pathed: the output directory is a
caller concern and would break byte equivalence between two equivalent runs.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from atreides_vision.errors import ValidationError
from atreides_vision.jsonl import JsonlStreamWriter
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.models import (
    FalseMergeDiagnostic,
    InstantEvaluation,
    WildtrackEvaluationSummary,
)

__all__ = [
    "WILDTRACK_EVALUATION_REPORT_FILENAME",
    "WILDTRACK_EVALUATION_SCHEMA_VERSION",
    "WILDTRACK_EVALUATION_SUMMARY_FILENAME",
    "WILDTRACK_FALSE_MERGE_DIAGNOSTICS_FILENAME",
    "evaluation_summary_record",
    "write_false_merge_diagnostics",
    "write_wildtrack_evaluation_report",
    "write_wildtrack_evaluation_summary",
]

#: Version of both artifact contracts below. Bumped only with explicit approval.
WILDTRACK_EVALUATION_SCHEMA_VERSION = "0.4.9b"

#: Per-instant streaming artifact.
WILDTRACK_EVALUATION_REPORT_FILENAME = "wildtrack_evaluation.jsonl"

#: Whole-run summary artifact.
WILDTRACK_EVALUATION_SUMMARY_FILENAME = "wildtrack_validation_summary.json"

#: Optional diagnostics artifact — written only when the run contains at least one
#: GT-resolved ``ASSOCIATED`` false merge. Zero false positives means no file: an
#: empty diagnostics file would imply "checked, found nothing wrong" for runs that
#: were never GT-resolvable. Never listed in the summary ``artifacts`` map.
WILDTRACK_FALSE_MERGE_DIAGNOSTICS_FILENAME = "false_merge_diagnostics.jsonl"


def write_wildtrack_evaluation_report(
    path: Path,
    summary: WildtrackEvaluationSummary,
    instants: Sequence[InstantEvaluation],
) -> Path:
    """Write the header + per-instant JSONL report to ``path``.

    Raises:
        ValidationError: if the file cannot be created or written.
    """
    header: dict[str, Any] = {
        "record": "header",
        "schema_version": WILDTRACK_EVALUATION_SCHEMA_VERSION,
        "selected_camera_ids": list(summary.selected_camera_ids),
        "max_ground_plane_distance_m": summary.max_ground_plane_distance_m,
        "instant_count": summary.instant_count,
    }
    with JsonlStreamWriter(path, ValidationError, label="WILDTRACK evaluation report") as writer:
        writer.write_record(header)
        for instant in instants:
            record: dict[str, Any] = {"record": "instant"}
            record.update(instant.to_record())
            writer.write_record(record)
    return path


def write_false_merge_diagnostics(
    path: Path,
    diagnostics: Sequence[FalseMergeDiagnostic],
) -> Path | None:
    """Write the false-merge diagnostics JSONL, or nothing when there are none.

    Records are ordered by ``(dataset_frame_id, first_tracklet, second_tracklet)``
    so two equivalent runs produce byte-identical output. Returns the path when a
    file was written and ``None`` when ``diagnostics`` is empty.

    Raises:
        ValidationError: if the file cannot be created or written.
    """
    if not diagnostics:
        return None
    ordered = sorted(
        diagnostics,
        key=lambda diagnostic: (
            diagnostic.dataset_frame_id,
            diagnostic.first_tracklet,
            diagnostic.second_tracklet,
        ),
    )
    header: dict[str, Any] = {
        "record": "header",
        "schema_version": WILDTRACK_EVALUATION_SCHEMA_VERSION,
        "false_merge_count": len(ordered),
    }
    with JsonlStreamWriter(
        path, ValidationError, label="WILDTRACK false-merge diagnostics"
    ) as writer:
        writer.write_record(header)
        for diagnostic in ordered:
            record: dict[str, Any] = {"record": "false_merge"}
            record.update(diagnostic.to_record())
            writer.write_record(record)
    return path


def evaluation_summary_record(summary: WildtrackEvaluationSummary) -> dict[str, Any]:
    """The whole-run summary as a versioned, deterministic record."""
    record: dict[str, Any] = {
        "schema_version": WILDTRACK_EVALUATION_SCHEMA_VERSION,
        "artifacts": {
            "evaluation_report": WILDTRACK_EVALUATION_REPORT_FILENAME,
            "summary": WILDTRACK_EVALUATION_SUMMARY_FILENAME,
        },
    }
    record.update(summary.to_record())
    return record


def write_wildtrack_evaluation_summary(
    path: Path,
    summary: WildtrackEvaluationSummary,
) -> Path:
    """Write ``summary`` to ``path``, creating the directory if needed.

    Raises:
        ValidationError: if the file cannot be created or written.
    """
    record = evaluation_summary_record(summary)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, indent=2, sort_keys=False, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise ValidationError(
            f"Could not write the WILDTRACK evaluation summary to {path}: {exc}"
        ) from exc
    return path

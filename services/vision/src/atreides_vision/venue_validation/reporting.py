"""The run summary artifact.

One small JSON document per run, versioned like every other Atreides contract
and deterministic by construction: no wall clock, no process id, no absolute
path, no iteration-order-dependent list. The two streaming artifacts beside it
are written by their own owners — V0.4.5's association/fusion writer and
V0.4.6's occupancy writer — and are not re-serialised here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atreides_vision.errors import ValidationError
from atreides_vision.venue_validation.config import (
    ASSOCIATION_REPORT_FILENAME,
    OCCUPANCY_REPORT_FILENAME,
)
from atreides_vision.venue_validation.models import VenueValidationSummary

__all__ = [
    "VENUE_VALIDATION_SUMMARY_SCHEMA_VERSION",
    "summary_record",
    "write_validation_summary",
]

#: Version of the summary contract below. Owned independently of the V0.4.5 and
#: V0.4.6 streams it sits beside.
VENUE_VALIDATION_SUMMARY_SCHEMA_VERSION = "0.4.8"

#: Said in the artifact itself, not only in the docs, because a summary travels
#: further than the document explaining it.
_EXECUTION_ONLY_NOTE = (
    "Execution statistics only. This run was not annotated, so no association, "
    "occupancy or duplicate-suppression accuracy can be derived from these counts."
)


def summary_record(summary: VenueValidationSummary) -> dict[str, Any]:
    """The summary as a versioned, deterministic record.

    Artifacts are named, never pathed: an absolute output directory differs
    between two otherwise equivalent runs and would break byte equivalence.
    """
    record: dict[str, Any] = {
        "schema_version": VENUE_VALIDATION_SUMMARY_SCHEMA_VERSION,
        "artifacts": {
            "association_report": ASSOCIATION_REPORT_FILENAME,
            "occupancy_report": OCCUPANCY_REPORT_FILENAME,
        },
        "note": _EXECUTION_ONLY_NOTE,
    }
    record.update(summary.to_record())
    return record


def write_validation_summary(path: Path, summary: VenueValidationSummary) -> Path:
    """Write ``summary`` to ``path``, creating the directory if needed.

    Raises:
        ValidationError: if the file cannot be created or written.
    """
    record = summary_record(summary)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, indent=2, sort_keys=False, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise ValidationError(f"Could not write the validation summary to {path}: {exc}") from exc
    return path

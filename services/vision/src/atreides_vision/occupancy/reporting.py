"""Observability for :mod:`atreides_vision.occupancy` — V0.4.6.

A separate, additive JSONL stream, not a change to V0.4.5's
``venue_pipeline_reporting`` schema. Occupancy is a *consumer* of the
association/fusion/identity trail that stream already reports, not a
replacement for it — and V0.4.5's schema is a frozen, versioned contract
downstream readers already depend on. Adding an ``occupancy`` field to it
would couple two independently evolving concerns into one schema version;
a separate stream keeps each versioned, and each readable, on its own.

**This is observability only.** It reflects runtime truth; it never produces
it. Every record is built by reading an already-computed
:class:`~atreides_vision.occupancy.models.OccupancySnapshot` via its own
``to_record()``. No reporting output here is ever fed back into occupancy,
identity, fusion or association.

File shape
----------
One JSON object per line. The first line is an ``occupancy_report`` header;
every line after it is one ``occupancy_instant`` record, one per call to
:meth:`JsonlOccupancyReportWriter.write_snapshot`::

    {"record_type":"occupancy_report","schema_version":"0.4.6",...}
    {"record_type":"occupancy_instant","schema_version":"0.4.6","venue_seconds":10.0,...}

Every record repeats ``schema_version`` and ``session_id``, matching every
other stream in this repository.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from atreides_vision.errors import OccupancyError
from atreides_vision.jsonl import GZIP_SUFFIX, JsonlStreamWriter
from atreides_vision.occupancy.models import OccupancySnapshot

logger = logging.getLogger(__name__)

__all__ = [
    "GZIP_SUFFIX",
    "OCCUPANCY_REPORT_SCHEMA_VERSION",
    "JsonlOccupancyReportWriter",
    "OccupancyReportMetadata",
    "occupancy_instant_record",
]

#: Version of the record contract below, owned independently of the V0.4.5
#: venue pipeline report — occupancy is an additive consumer of that trail,
#: not a change to it, and evolves on its own schedule.
OCCUPANCY_REPORT_SCHEMA_VERSION = "0.4.6"

#: Matches the precision every other venue-time-bearing record in this
#: repository already rounds ``venue_seconds`` to.
_SECONDS_PRECISION = 3


@dataclass(frozen=True, slots=True)
class OccupancyReportMetadata:
    """The header record: everything constant for the whole report stream.

    ``session_id`` is a plain, caller-supplied label, exactly as in
    :class:`~atreides_vision.venue_pipeline_reporting.VenuePipelineReportMetadata`,
    so the two streams can be correlated by a caller without either module
    depending on the other.
    """

    session_id: str

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise OccupancyError("An occupancy report requires a non-empty session_id.")

    def to_record(self) -> dict[str, Any]:
        return {
            "record_type": "occupancy_report",
            "schema_version": OCCUPANCY_REPORT_SCHEMA_VERSION,
            "session_id": self.session_id,
            "occupancy_scope": "current_venue_instant",
        }


def occupancy_instant_record(snapshot: OccupancySnapshot, session_id: str) -> dict[str, Any]:
    """One venue instant's occupancy snapshot, as data.

    Delegates entirely to :meth:`~atreides_vision.occupancy.models.OccupancySnapshot.to_record`
    for the snapshot's own fields; this function adds only the envelope
    (record type, schema version, session id) every stream in this
    repository already carries.
    """
    return {
        "record_type": "occupancy_instant",
        "schema_version": OCCUPANCY_REPORT_SCHEMA_VERSION,
        "session_id": session_id,
        "venue_seconds": round(snapshot.venue_seconds, _SECONDS_PRECISION),
        **snapshot.to_record(),
    }


class JsonlOccupancyReportWriter:
    """Streams occupancy snapshots to a ``.jsonl`` file, one line per record.

    Shares :class:`~atreides_vision.jsonl.JsonlStreamWriter` with every other
    output stream in this repository. Snapshots are written as they are
    produced and never accumulated: a long session costs no more memory than
    a short one.

    A write failure raises :class:`~atreides_vision.errors.OccupancyError`
    and never anything the occupancy analyzer itself raises. This writer
    mutates nothing about the pipeline, registry or snapshot it observes: a
    failed or successful write changes nothing about their state.

    Use as a context manager::

        with JsonlOccupancyReportWriter(path, metadata) as writer:
            snapshot = compute_occupancy(result, pipeline.registry)
            writer.write_snapshot(snapshot)
    """

    def __init__(
        self,
        path: Path | str,
        metadata: OccupancyReportMetadata,
        compress: bool | None = None,
    ) -> None:
        self._metadata = metadata
        self._stream = JsonlStreamWriter(path, OccupancyError, "occupancy report", compress)
        self._snapshots_written = 0

        self._stream.write_record(metadata.to_record())

    @property
    def path(self) -> Path:
        return self._stream.path

    @property
    def metadata(self) -> OccupancyReportMetadata:
        return self._metadata

    @property
    def snapshots_written(self) -> int:
        return self._snapshots_written

    @property
    def is_compressed(self) -> bool:
        return self._stream.is_compressed

    def write_snapshot(self, snapshot: OccupancySnapshot) -> None:
        """Append one venue instant's occupancy record.

        Exactly one JSON line per call: this method performs no batching,
        filtering or sampling of its own, so a reader can always recover the
        same number of snapshots the caller itself computed.
        """
        if self._stream.is_released:
            raise OccupancyError(
                f"Cannot write to a released occupancy report writer: {self._stream.path}"
            )
        self._stream.write_record(occupancy_instant_record(snapshot, self._metadata.session_id))
        self._snapshots_written += 1

    def release(self) -> None:
        """Flush and close. Idempotent, and safe after an exception."""
        if not self._stream.is_released:
            self._stream.release()
            logger.debug(
                "Released occupancy report writer %s after %d snapshot(s)",
                self._stream.path,
                self._snapshots_written,
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

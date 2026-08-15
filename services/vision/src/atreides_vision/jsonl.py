"""Streaming JSON Lines output, shared by every machine-readable stream.

Behaviour observations and interaction records are different contracts with
different cardinality, but they need the same plumbing: open a file (optionally
gzipped), serialise one complete record per line, never buffer the whole run,
close cleanly even when processing raises. That plumbing lives here once.

Two properties matter enough to state:

**Whole lines or nothing.** Each record is serialised to a string *before* the
write is attempted, so an interrupted run leaves complete records followed by
nothing. Earlier lines surviving a mid-run failure is expected and useful; a
truncated JSON object would poison the file for every reader.

**Streaming.** Compression writes through to the file as it goes, so memory is
flat regardless of run length and the file can be tailed while it is produced.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Self

logger = logging.getLogger(__name__)

__all__ = ["GZIP_SUFFIX", "JsonlStreamWriter"]

#: Output paths ending in this are gzip-compressed automatically. One
#: convention, visible in the filename, so a reader can tell what a file is
#: without a flag having been recorded anywhere.
GZIP_SUFFIX = ".gz"


class JsonlStreamWriter:
    """Appends one JSON object per line to a file, optionally gzipped.

    Not used directly by the pipeline: the behaviour and interaction writers own
    their own record contracts and wrap this for the file handling. Errors are
    raised as whatever domain exception the owner passes in, so a caller sees
    "could not write behaviour output" rather than a generic I/O error.

    Args:
        path: Destination. A ``.gz`` suffix selects compression unless
            ``compress`` overrides it.
        error_type: Domain exception raised for any open, serialise or write
            failure.
        label: What this stream is, used in error messages. An operator needs
            to know *which* output failed, not merely that one did.
        compress: Force compression on or off. ``None`` follows the suffix.
    """

    def __init__(
        self,
        path: Path | str,
        error_type: type[Exception],
        label: str = "output",
        compress: bool | None = None,
    ) -> None:
        self._path = Path(path)
        self._error_type = error_type
        self._label = label
        self._compressed = (
            self._path.suffix.lower() == GZIP_SUFFIX if compress is None else compress
        )
        self._lines_written = 0
        self._released = False

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Deliberately long-lived: the owning writer is the context manager
            # and closes this in release().
            handle: IO[str] = (
                gzip.open(self._path, "wt", encoding="utf-8")  # noqa: SIM115
                if self._compressed
                else self._path.open("w", encoding="utf-8")
            )
        except OSError as exc:
            raise error_type(f"Could not open {label} {self._path}: {exc}") from exc
        self._handle = handle

    @property
    def path(self) -> Path:
        return self._path

    @property
    def is_compressed(self) -> bool:
        return self._compressed

    @property
    def lines_written(self) -> int:
        return self._lines_written

    @property
    def is_released(self) -> bool:
        return self._released

    def write_record(self, record: dict[str, Any]) -> None:
        """Serialise fully, then write once, so no half-record can reach disk."""
        if self._released:
            raise self._error_type(f"Cannot write to a released {self._label} writer: {self._path}")
        try:
            line = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
        except (TypeError, ValueError) as exc:
            raise self._error_type(
                f"Could not serialise a {self._label} record for {self._path}: {exc}"
            ) from exc
        try:
            self._handle.write(line)
        except OSError as exc:
            raise self._error_type(f"Could not write {self._label} to {self._path}: {exc}") from exc
        self._lines_written += 1

    def release(self) -> None:
        """Flush and close. Idempotent, and safe after an exception."""
        if not getattr(self, "_released", False):
            self._handle.close()
            self._released = True
            logger.debug("Released %s after %d lines", self._path, self._lines_written)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

"""Optional machine-readable zone-metrics output, as JSON Lines.

One compact record per sampled frame, so the file can be tailed while a run is
in progress and read with any JSONL tool afterwards. Metrics are computed every
frame; sampling only controls how often they are written, defaulting to once
per second of source time rather than once per frame.

The records contain zone ids, counts, durations and speeds. They contain no
track ids, no positions and nothing describing an individual.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

from atreides_vision.errors import MetricsOutputError
from atreides_vision.zones.models import ZoneReport

logger = logging.getLogger(__name__)

__all__ = ["JsonlMetricsWriter", "MetricsSampler", "MetricsWriter"]


class MetricsWriter(Protocol):
    """A destination for zone metric records."""

    def write(self, report: ZoneReport) -> None: ...

    def release(self) -> None: ...


class MetricsSampler:
    """Decides which frames get written, based on source time.

    The first frame is always sampled, then one per interval of source time —
    so a 25 FPS and a 60 FPS video of the same scene produce the same number of
    records.
    """

    def __init__(self, interval_seconds: float = 1.0) -> None:
        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be positive, got {interval_seconds}")
        self._interval = interval_seconds
        self._next_due: float | None = None

    @property
    def interval_seconds(self) -> float:
        return self._interval

    def should_sample(self, timestamp_seconds: float) -> bool:
        """Whether the frame at this source timestamp should be recorded."""
        if self._next_due is None or timestamp_seconds >= self._next_due:
            self._next_due = timestamp_seconds + self._interval
            return True
        return False

    def reset(self) -> None:
        self._next_due = None


class JsonlMetricsWriter:
    """Appends one JSON object per line to a ``.jsonl`` file.

    Use as a context manager so the handle is closed even if processing raises::

        with JsonlMetricsWriter(path) as writer:
            writer.write(report)
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._records_written = 0
        self._released = False

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("w", encoding="utf-8")
        except OSError as exc:
            raise MetricsOutputError(f"Could not open metrics output {self._path}: {exc}") from exc

    @property
    def path(self) -> Path:
        return self._path

    @property
    def records_written(self) -> int:
        return self._records_written

    def write(self, report: ZoneReport) -> None:
        """Append one record."""
        if self._released:
            raise MetricsOutputError(f"Cannot write to a released metrics writer: {self._path}")
        try:
            self._handle.write(json.dumps(report.to_record(), separators=(",", ":")) + "\n")
        except OSError as exc:
            raise MetricsOutputError(f"Could not write metrics to {self._path}: {exc}") from exc
        self._records_written += 1

    def release(self) -> None:
        """Flush and close. Idempotent, and safe after an exception."""
        if not getattr(self, "_released", False):
            self._handle.close()
            self._released = True
            logger.debug(
                "Released metrics writer %s after %d records", self._path, self._records_written
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

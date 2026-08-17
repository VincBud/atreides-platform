"""The machine-readable venue session contract, as versioned JSON Lines.

A **separate stream** from behaviour (0.3.3), interaction (0.3.4) and crowd
(0.3.5), with its own schema version. It describes ingestion — which streams
existed, how they were timed, and what happened at each venue instant — which
has nothing in common with a measurement stream's cardinality or lifetime, and
tying the versions together would force a bump on one whenever another changed.

File shape
----------
One JSON object per line: a ``venue_session`` header, then one ``venue_tick``
per venue instant, then a ``venue_session_summary``::

    {"record_type":"venue_session","schema_version":"0.4.0",...}
    {"record_type":"venue_tick","schema_version":"0.4.0","sequence":0,...}
    {"record_type":"venue_session_summary","schema_version":"0.4.0",...}

Every record repeats ``schema_version`` and ``session_id``, because JSONL is
consumed a line at a time and a line that cannot be interpreted alone will
eventually be misinterpreted.

Both times are recorded
-----------------------
Each contribution carries its ``frame_index`` and ``source_seconds`` — where it
sits in its own file — as well as its ``venue_seconds``. Provenance survives the
merge intact: any venue observation resolves back to *(camera, stream epoch,
frame index, position in that recording)*.

``venue_micros`` is the integer ordering key and appears alongside the seconds
rather than instead of them. The seconds are the measurement.

Two alignments, deliberately
----------------------------
``time_alignment`` on a tick covers the streams that contributed to that
instant. ``session_time_alignment`` — on the header and repeated on every tick —
is the worst across every registered stream, contributing or not. They differ,
and the difference is the point: a tick whose worst-aligned camera happens to be
between frames would otherwise read as a better aligned session than the one
that produced it. Anything describing the session reads the session figure.

What is not in this stream
--------------------------
No venue-local person identity, no cross-camera correspondence, no candidate
pairs, no match score, no duplicate suppression, no unique-people or cumulative
visitor figure, no appearance descriptor, embedding or crop, and no topology.
V0.4.0 performs no association at all. Two cameras contributing to one tick were
recording at the same venue instant; nothing here says they saw the same person,
and where ``time_alignment.basis`` is ``not_established`` even the simultaneity
is nominal.

Also absent: any threshold or boolean derived from the alignment. Consumers
carry the uncertainty forward and report it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from atreides_vision.errors import VenueSessionError
from atreides_vision.jsonl import GZIP_SUFFIX, JsonlStreamWriter
from atreides_vision.timing import FrameRateSource, StreamFrame, StreamRef, TimeUncertainty
from atreides_vision.venue.models import StreamAbsence, VenueTick, stream_sort_key

logger = logging.getLogger(__name__)

__all__ = [
    "GZIP_SUFFIX",
    "VENUE_SESSION_SCHEMA_VERSION",
    "JsonlVenueSessionWriter",
    "VenueSessionMetadata",
    "VenueSessionSummary",
    "VenueStreamProvenance",
    "tick_record",
]

#: Version of the record contract below, owned independently of every V0.3.x
#: stream. Written onto every record so a reader can reject a stream it does not
#: understand rather than silently mis-parsing it.
VENUE_SESSION_SCHEMA_VERSION = "0.4.0"

#: Microsecond resolution: the exact precision of the ordering key, so the
#: seconds and the key can never disagree about which instant a tick is at.
_SECONDS_PRECISION = 6
_FPS_PRECISION = 4

#: The narrowest half-width this format can express. Anything finer is widened
#: to it rather than rounded away — see :func:`_alignment_record`.
_MIN_SERIALISED_HALF_WIDTH = 10.0**-_SECONDS_PRECISION


def _alignment_record(timing: TimeUncertainty) -> dict[str, Any]:
    """Uncertainty as data, with no verdict attached.

    ``half_width_seconds`` is ``null`` exactly when the basis is
    ``not_established``. There is deliberately no "aligned" boolean: that
    decision belongs to whoever has a use for it, with their reasoning visible.
    """
    half_width = timing.half_width_seconds
    if half_width is not None:
        # A sub-microsecond half-width would round to 0.0 beside an
        # 'operator_declared' basis — a record TimeUncertainty itself refuses to
        # construct, so this writer would be emitting something its own reader
        # rejects. Widening to the narrowest value the format can express keeps
        # the record parseable and errs towards claiming less, never more.
        half_width = max(round(half_width, _SECONDS_PRECISION), _MIN_SERIALISED_HALF_WIDTH)
    return {"basis": timing.basis.value, "half_width_seconds": half_width}


@dataclass(frozen=True, slots=True)
class VenueStreamProvenance:
    """One stream's identity, input and timing, as recorded in the header.

    ``fps_source`` is here because a frame rate a container invented and one an
    operator measured produce identical-looking timestamps, and everything above
    is derived from them. A reader of this stream can tell them apart; a reader
    of a V0.3.x stream still cannot, which is a known limitation of those
    frozen schemas rather than something this record fixes for them.
    """

    stream: StreamRef
    input_path: str
    width: int
    height: int
    fps: float
    fps_source: FrameRateSource
    #: Where this stream's zero sits on the venue timeline.
    offset_seconds: float
    timing: TimeUncertainty
    #: Frames the container claims, where it claims a usable number. Never used
    #: to drive the schedule — the merge reads until a stream is exhausted.
    frame_count: int | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "camera_id": self.stream.camera_id,
            "stream_epoch": self.stream.stream_epoch,
            "stream_ref": str(self.stream),
            "input_path": self.input_path,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, _FPS_PRECISION),
            "fps_source": self.fps_source.value,
            "fps_is_trustworthy": self.fps_source.is_trustworthy,
            "frame_count": self.frame_count,
            "start_offset_seconds": round(self.offset_seconds, _SECONDS_PRECISION),
            "time_alignment": _alignment_record(self.timing),
        }


@dataclass(frozen=True, slots=True)
class VenueSessionMetadata:
    """The header record: everything constant for the whole session."""

    session_id: str
    streams: tuple[VenueStreamProvenance, ...]
    #: The manifest this session was built from, so a record set traces back to
    #: the exact input list that produced it.
    manifest_path: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise VenueSessionError("A venue session record needs a non-empty session_id.")
        if not self.streams:
            raise VenueSessionError("A venue session record must describe at least one stream.")

    @property
    def camera_ids(self) -> tuple[str, ...]:
        """Distinct cameras, in canonical order."""
        seen: dict[str, None] = {}
        for stream in sorted(self.streams, key=lambda entry: stream_sort_key(entry.stream)):
            seen.setdefault(stream.stream.camera_id, None)
        return tuple(seen)

    @property
    def alignment(self) -> TimeUncertainty:
        """The session's alignment: the worst across **every** stream.

        Every registered stream counts, contributing or not. A session is only
        as aligned as its least aligned camera, and that fact does not change
        from tick to tick — unlike :attr:`VenueTick.alignment`, which covers
        contributors only and therefore improves whenever the worst camera is
        between frames or has ended.
        """
        first, *rest = [entry.timing for entry in self.streams]
        return first.worst_of(*rest)

    def to_record(self) -> dict[str, Any]:
        ordered = sorted(self.streams, key=lambda entry: stream_sort_key(entry.stream))
        record: dict[str, Any] = {
            "record_type": "venue_session",
            "schema_version": VENUE_SESSION_SCHEMA_VERSION,
            "stream": "venue_session",
            "analysis": "multi_camera_time_and_venue_ingestion",
            "session_id": self.session_id,
            "camera_count": len(self.camera_ids),
            "stream_count": len(ordered),
            "session_time_alignment": _alignment_record(self.alignment),
            "venue_time": {
                "origin": "session_relative",
                "unit": "s",
                "ordering_key": "venue_micros",
                "tick_schedule": "source_event_driven",
            },
            # Stated in the data, not only in prose: this stream orders
            # observations, it does not relate them to each other.
            "identity_scope": "camera_local_streams",
            "association": "not_performed",
            "venue_local_identity": "none",
            "streams": [entry.to_record() for entry in ordered],
        }
        if self.manifest_path is not None:
            record["manifest_path"] = self.manifest_path
        return record


@dataclass(frozen=True, slots=True)
class VenueSessionSummary:
    """The closing record: what the session actually produced.

    ``frames_contributed`` counts *frames*, per stream. It is not a count of
    people, of events, or of anything a camera saw, and frames from two cameras
    are never combined into one figure — there is no association to justify it.
    """

    session_id: str
    tick_count: int
    frames_by_stream: tuple[tuple[StreamRef, int], ...]
    first_venue_micros: int | None = None
    last_venue_micros: int | None = None

    def __post_init__(self) -> None:
        if self.tick_count < 0:
            raise VenueSessionError(
                f"VenueSessionSummary.tick_count must be non-negative, got {self.tick_count!r}"
            )

    def to_record(self) -> dict[str, Any]:
        ordered = sorted(self.frames_by_stream, key=lambda entry: stream_sort_key(entry[0]))
        span = (
            None
            if self.first_venue_micros is None or self.last_venue_micros is None
            else round(
                (self.last_venue_micros - self.first_venue_micros) / 1_000_000,
                _SECONDS_PRECISION,
            )
        )
        return {
            "record_type": "venue_session_summary",
            "schema_version": VENUE_SESSION_SCHEMA_VERSION,
            "session_id": self.session_id,
            "tick_count": self.tick_count,
            "first_venue_micros": self.first_venue_micros,
            "last_venue_micros": self.last_venue_micros,
            "venue_span_seconds": span,
            "streams": [
                {
                    "camera_id": stream.camera_id,
                    "stream_epoch": stream.stream_epoch,
                    "frames_contributed": frames,
                }
                for stream, frames in ordered
            ],
        }


def _contribution_record(frame: StreamFrame) -> dict[str, Any]:
    """One frame's full provenance: where it is in its own file, and in the venue."""
    return {
        "camera_id": frame.stream.camera_id,
        "stream_epoch": frame.stream.stream_epoch,
        "frame_index": frame.frame_index,
        "source_seconds": round(frame.source_seconds, _SECONDS_PRECISION),
        "venue_seconds": round(frame.venue_seconds, _SECONDS_PRECISION),
        "time_alignment": _alignment_record(frame.timing),
    }


def _absence_record(absence: StreamAbsence) -> dict[str, Any]:
    return {
        "camera_id": absence.stream.camera_id,
        "stream_epoch": absence.stream.stream_epoch,
        "activity": absence.activity.value,
    }


def tick_record(
    tick: VenueTick, session_id: str, session_alignment: TimeUncertainty
) -> dict[str, Any]:
    """One venue instant, with every stream accounted for.

    ``absent`` is written even when it is empty, so a reader never has to guess
    whether a stream was omitted or simply had nothing to say.

    Two alignments, and the difference matters. ``time_alignment`` covers the
    streams that contributed to *this* instant, so it improves on ticks where a
    poorly aligned camera is between frames — read alone, such a line would
    describe a better aligned session than the one that produced it.
    ``session_time_alignment`` is the worst across every registered stream and
    is the same on every line, which is what makes a line interpretable by
    itself.
    """
    return {
        "record_type": "venue_tick",
        "schema_version": VENUE_SESSION_SCHEMA_VERSION,
        "session_id": session_id,
        "sequence": tick.sequence,
        "venue_micros": tick.venue_micros,
        "venue_seconds": round(tick.venue_seconds, _SECONDS_PRECISION),
        "time_alignment": _alignment_record(tick.alignment),
        "session_time_alignment": _alignment_record(session_alignment),
        "contributions": [
            _contribution_record(contribution) for contribution in tick.contributions
        ],
        "absent": [_absence_record(absence) for absence in tick.absent],
    }


class JsonlVenueSessionWriter:
    """Streams venue session records to a ``.jsonl`` file.

    Shares :class:`~atreides_vision.jsonl.JsonlStreamWriter` with every other
    output stream, so gzip selection, whole-line writes and lifecycle behave
    identically rather than being reimplemented. Ticks are written as they are
    produced and never accumulated: a long session costs no more memory than a
    short one, and the file can be tailed while it is written.
    """

    def __init__(
        self,
        path: Path | str,
        metadata: VenueSessionMetadata,
        compress: bool | None = None,
    ) -> None:
        self._metadata = metadata
        self._stream = JsonlStreamWriter(path, VenueSessionError, "venue session output", compress)
        self._ticks_written = 0
        self._summary_written = False

        self._stream.write_record(metadata.to_record())

    @property
    def path(self) -> Path:
        return self._stream.path

    @property
    def metadata(self) -> VenueSessionMetadata:
        return self._metadata

    @property
    def ticks_written(self) -> int:
        return self._ticks_written

    @property
    def is_compressed(self) -> bool:
        return self._stream.is_compressed

    def write_tick(self, tick: VenueTick) -> None:
        """Append one tick. Ticks must be written in the order they were produced."""
        if self._stream.is_released:
            raise VenueSessionError(
                f"Cannot write to a released venue session writer: {self._stream.path}"
            )
        if self._summary_written:
            raise VenueSessionError(
                f"The summary for session '{self._metadata.session_id}' has already been "
                "written; a tick after it would describe a session the summary does not cover."
            )
        self._stream.write_record(
            tick_record(tick, self._metadata.session_id, self._metadata.alignment)
        )
        self._ticks_written += 1

    def write_summary(self, summary: VenueSessionSummary) -> None:
        """Append the closing record. Once per session."""
        if self._summary_written:
            raise VenueSessionError(
                f"Session '{self._metadata.session_id}' already has a summary record."
            )
        if summary.session_id != self._metadata.session_id:
            raise VenueSessionError(
                f"Summary is for session '{summary.session_id}' but this stream describes "
                f"'{self._metadata.session_id}'."
            )
        self._stream.write_record(summary.to_record())
        self._summary_written = True

    def release(self) -> None:
        """Flush and close. Idempotent, and safe after an exception."""
        if not self._stream.is_released:
            self._stream.release()
            logger.debug(
                "Released venue session writer %s after %d tick(s)",
                self._stream.path,
                self._ticks_written,
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

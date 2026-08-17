"""One venue ingestion session: N video sources walked as one tick schedule.

This is the layer the rest of the system talks to. It owns nothing about
detection, tracking or measurement — it decodes, times and orders frames, and
hands them out. Camera-local processing attaches *above* it, one processor per
camera, driven from the ticks.

What a session is
-----------------
Session-scoped and temporary in every sense. Its stream references, its venue
timeline and its ordering exist for the duration of one run and are discarded
with it. Nothing is persisted, nothing is carried between sessions or venues,
and no identity of any kind is created here.

Deliberately not built
----------------------
No cross-camera association, no venue-local person identity, no duplicate
suppression, no unique-visitor figure, no appearance descriptor, no topology or
overlap model. A tick says which streams had a frame at one instant, and a
consumer that reads more than that into it is reading something the data does
not contain. When association is built it attaches above this loop as an
optional collaborator, consuming ticks and producing its own confidence-carrying
output — it does not move inside a cursor or a tracker.

Lifetimes
---------
The caller owns the sources: this module neither opens nor releases them, so a
failure part-way through still leaves the caller's ``with`` blocks to close
every file. The same rule the single-source pipeline already follows.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence

import numpy as np

from atreides_vision.errors import VenueSessionError
from atreides_vision.timing import StreamFrame, StreamRef, TimeUncertainty
from atreides_vision.venue.models import VenueTick, stream_sort_key
from atreides_vision.venue.scheduler import StreamCursor, merge_streams
from atreides_vision.venue.timebase import VenueTimebase
from atreides_vision.video.source import VideoSource

logger = logging.getLogger(__name__)

__all__ = ["VenueSession", "VideoSourceCursor"]


class VideoSourceCursor:
    """One video source, presented to the merge as a single pending frame.

    Times each frame as it is decoded: ``source_seconds`` from the stream's own
    frame rate, ``venue_seconds`` from the timebase. Both travel with the frame,
    so an observation can always be resolved back to *(camera, epoch, frame
    index, position in that file)* however far downstream it ends up.

    Holds exactly one decoded frame — the peeked one — and nothing else. It
    never seeks, never rewinds, never reads the container's frame count, and
    keeps no history, so a twelve-hour source costs what a twelve-second one
    does.
    """

    def __init__(self, stream: StreamRef, source: VideoSource, timebase: VenueTimebase) -> None:
        if not timebase.is_registered(stream):
            raise VenueSessionError(
                f"Stream {stream} must be registered on the timebase before it is read; "
                "otherwise its frames have no place on the venue timeline."
            )
        self._stream = stream
        self._source = source
        self._timebase = timebase
        metadata = source.metadata
        # Matches the single-source pipeline exactly, so one camera through the
        # venue path is bit-identical to one camera through VideoProcessor.run.
        self._source_fps = metadata.fps if metadata.fps > 0 else 1.0
        self._timing = timebase.uncertainty(stream)
        self._frames: Iterator[np.ndarray] | None = None
        self._next_index = 1
        self._pending: StreamFrame | None = None
        self._exhausted = False

    @property
    def stream(self) -> StreamRef:
        return self._stream

    @property
    def frames_decoded(self) -> int:
        """Frames decoded so far, including the one currently peeked."""
        return self._next_index - 1 + (1 if self._pending is not None else 0)

    def peek(self) -> StreamFrame | None:
        """Decode at most one frame ahead, returning the same one until advanced."""
        if self._pending is not None:
            return self._pending
        if self._exhausted:
            return None
        if self._frames is None:
            self._frames = iter(self._source.frames())

        image = next(self._frames, None)
        if image is None:
            self._exhausted = True
            if self._next_index == 1:
                # Not a failure: an empty recording is a valid recording. It is
                # ENDED from the first tick and contributes nothing, and saying
                # so is better than a session that silently runs a camera short.
                logger.warning(
                    "Stream %s decoded no frames. It will contribute nothing to this session.",
                    self._stream,
                )
            return None

        source_seconds = (self._next_index - 1) / self._source_fps
        self._pending = StreamFrame(
            stream=self._stream,
            frame_index=self._next_index,
            source_seconds=source_seconds,
            venue_seconds=self._timebase.venue_seconds(self._stream, source_seconds),
            timing=self._timing,
            image=image,
        )
        return self._pending

    def advance(self) -> None:
        """Drop the peeked frame. The pixels go with it."""
        if self._pending is None:
            raise VenueSessionError(
                f"Cursor for {self._stream} was advanced with no peeked frame. The scheduler "
                "peeks before it advances; advancing blind would skip a frame."
            )
        self._pending = None
        self._next_index += 1


class VenueSession:
    """A named session over a fixed set of streams, walked once.

    Counts as it goes so a run can be described afterwards — how many ticks,
    which streams contributed how many frames, and what span of venue time was
    covered — without retaining a single tick.
    """

    def __init__(
        self,
        session_id: str,
        timebase: VenueTimebase,
        cursors: Sequence[StreamCursor],
    ) -> None:
        if not session_id or not session_id.strip():
            raise VenueSessionError("A venue session needs a non-empty session_id.")
        if not cursors:
            raise VenueSessionError(
                "A venue session needs at least one stream. One camera is a supported "
                "deployment; none is not a deployment."
            )

        ordered = tuple(sorted(cursors, key=lambda cursor: stream_sort_key(cursor.stream)))
        for cursor in ordered:
            if not timebase.is_registered(cursor.stream):
                raise VenueSessionError(
                    f"Stream {cursor.stream} is not registered on this session's timebase."
                )

        self._session_id = session_id.strip()
        self._timebase = timebase
        self._cursors = ordered
        self._frames_contributed: dict[StreamRef, int] = {cursor.stream: 0 for cursor in ordered}
        self._tick_count = 0
        self._first_venue_micros: int | None = None
        self._last_venue_micros: int | None = None
        self._walked = False

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def timebase(self) -> VenueTimebase:
        return self._timebase

    @property
    def streams(self) -> tuple[StreamRef, ...]:
        """Every stream in this session, in canonical order."""
        return tuple(cursor.stream for cursor in self._cursors)

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def frames_by_stream(self) -> tuple[tuple[StreamRef, int], ...]:
        """Frames contributed per stream, in canonical order."""
        return tuple(
            (stream, self._frames_contributed[stream])
            for stream in sorted(self._frames_contributed, key=stream_sort_key)
        )

    @property
    def first_venue_micros(self) -> int | None:
        """The session's first tick instant, or ``None`` if it produced none."""
        return self._first_venue_micros

    @property
    def last_venue_micros(self) -> int | None:
        return self._last_venue_micros

    @property
    def alignment(self) -> TimeUncertainty:
        """The worst alignment across every registered stream.

        A session is only as aligned as its least aligned camera, so this is
        ``NOT_ESTABLISHED`` unless every stream has a declared offset. It
        describes the session; each tick carries its own, over the streams that
        actually contributed to it.
        """
        first, *rest = [self._timebase.uncertainty(stream) for stream in self.streams]
        return first.worst_of(*rest)

    def ticks(self) -> Iterator[VenueTick]:
        """Walk the whole session, one tick per distinct venue instant.

        A single pass: cursors are consumed as they are read, so a session
        cannot be replayed. A session whose streams are all empty yields nothing
        and returns normally.
        """
        if self._walked:
            raise VenueSessionError(
                f"Session '{self._session_id}' has already been walked. Its streams were "
                "consumed as they were read; nothing is buffered to replay."
            )
        self._walked = True

        for tick in merge_streams(self._cursors):
            self._tick_count += 1
            if self._first_venue_micros is None:
                self._first_venue_micros = tick.venue_micros
            self._last_venue_micros = tick.venue_micros
            for contribution in tick.contributions:
                self._frames_contributed[contribution.stream] += 1
            yield tick

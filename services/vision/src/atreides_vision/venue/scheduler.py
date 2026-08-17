"""N-source fan-in: many streams, one totally ordered schedule of ticks.

The merge is **source-event driven**. A tick happens where a frame happens, one
per distinct venue instant, and a stream that had no frame there simply did not
have one. The alternative — a fixed venue cadence — would be a hardcoded frame
rate wearing a different name, and to fill it in the merge would have to
interpolate positions or hold the last frame, manufacturing observations nobody
made. Consumers that want a regular cadence sample the ticks with
:class:`~atreides_vision.sampling.IntervalSampler`, exactly as the rest of the
pipeline already does.

Heterogeneous by construction
-----------------------------
Nothing here assumes the streams share a frame rate, a start time, a duration or
a resolution, and nothing lines up frame *indices* between streams: a 25 FPS and
a 30 FPS camera have no frame in common, and the schedule is built from venue
times alone. One stream is the degenerate case — its own frames, in order.

Determinism
-----------
The total order is ``(venue_micros, camera_id, stream_epoch, frame_index)``.
``venue_micros`` is an integer, so simultaneity is exact equality with no
tolerance constant, and the remaining components make the order total. Cursors
are sorted canonically on entry, so the order the caller happened to build them
in — which is the manifest's order — cannot reach the output.

Bounded state
-------------
Exactly one decoded frame is in flight per stream: :meth:`StreamCursor.peek`
holds it, :meth:`StreamCursor.advance` drops it. The merge keeps two small values
per stream and no ticks, so memory is O(streams) and independent of how long the
session runs. Nothing seeks, rewinds, or asks a container how many frames it has.

A failing stream ends the session
---------------------------------
An exception from a cursor propagates. V0.4.0 has no tolerant degradation for a
stream that dies mid-session — that belongs with live ingestion — and quietly
continuing without a camera would silently change what every venue-level figure
was computed from. A stream that legitimately decodes *zero* frames is not a
failure: it is ``ENDED`` from the first tick and never contributes.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

from atreides_vision.errors import VenueSessionError
from atreides_vision.timing import StreamFrame, StreamRef
from atreides_vision.venue.models import StreamAbsence, StreamActivity, VenueTick, stream_sort_key

__all__ = ["StreamCursor", "merge_streams"]


class StreamCursor(Protocol):
    """One stream, presented as at most one pending frame at a time.

    Deliberately narrower than an iterator: the merge has to *look* at each
    stream's next frame to decide which instant comes next, without consuming
    it. ``peek()`` decodes at most one frame ahead and returns the same frame
    until ``advance()`` drops it, which is what bounds memory to one frame per
    stream.
    """

    @property
    def stream(self) -> StreamRef: ...

    def peek(self) -> StreamFrame | None:
        """The stream's next frame, or ``None`` when it is exhausted."""
        ...

    def advance(self) -> None:
        """Drop the peeked frame, so the next :meth:`peek` decodes the one after."""
        ...


def merge_streams(cursors: Sequence[StreamCursor]) -> Iterator[VenueTick]:
    """Walk every stream in venue-time order, one tick per distinct instant.

    Args:
        cursors: One cursor per registered stream. Order is irrelevant — they
            are sorted canonically here — but the streams must be distinct.

    Yields:
        Ticks in strictly increasing venue instant, each carrying the frames at
        that instant and the state of every stream that had none.

    Raises:
        VenueSessionError: for duplicate streams, a cursor whose frames do not
            belong to it, or a stream whose venue times fail to advance.
    """
    ordered = tuple(sorted(cursors, key=lambda cursor: stream_sort_key(cursor.stream)))
    _reject_duplicate_streams(ordered)

    started: set[StreamRef] = set()
    emitted_micros: dict[StreamRef, int] = {}
    sequence = 0

    while True:
        heads = [(cursor, head) for cursor in ordered if (head := _peek(cursor)) is not None]
        if not heads:
            return

        for cursor, head in heads:
            previous = emitted_micros.get(cursor.stream)
            if previous is not None and head.venue_micros <= previous:
                raise VenueSessionError(
                    f"Stream {cursor.stream} offered frame {head.frame_index} at venue instant "
                    f"{head.venue_micros}, which is not after its previous frame at {previous}. "
                    "A stream's venue time must advance, or the merge has no order to work in."
                )

        tick_micros = min(head.venue_micros for _, head in heads)

        contributions: list[StreamFrame] = []
        absent: list[StreamAbsence] = []
        contributing: list[StreamCursor] = []
        # Keyed by StreamRef rather than by cursor: a cursor is any object
        # satisfying the protocol, and need not be hashable.
        pending = {cursor.stream: head for cursor, head in heads}

        # ``ordered`` is canonical, and each stream contributes at most one
        # frame, so both tuples come out in canonical order by construction.
        for cursor in ordered:
            head = pending.get(cursor.stream)
            if head is None:
                absent.append(StreamAbsence(cursor.stream, StreamActivity.ENDED))
            elif head.venue_micros == tick_micros:
                contributions.append(head)
                contributing.append(cursor)
            elif cursor.stream in started:
                absent.append(StreamAbsence(cursor.stream, StreamActivity.BETWEEN_FRAMES))
            else:
                absent.append(StreamAbsence(cursor.stream, StreamActivity.NOT_YET_STARTED))

        yield VenueTick(
            venue_micros=tick_micros,
            sequence=sequence,
            contributions=tuple(contributions),
            absent=tuple(absent),
        )
        sequence += 1

        for cursor in contributing:
            started.add(cursor.stream)
            emitted_micros[cursor.stream] = tick_micros
            cursor.advance()


def _peek(cursor: StreamCursor) -> StreamFrame | None:
    """Peek, checking the frame really belongs to the stream that offered it."""
    head = cursor.peek()
    if head is not None and head.stream != cursor.stream:
        raise VenueSessionError(
            f"Cursor for {cursor.stream} offered a frame belonging to {head.stream}. A frame's "
            "provenance must survive the merge intact."
        )
    return head


def _reject_duplicate_streams(cursors: Sequence[StreamCursor]) -> None:
    streams = [cursor.stream for cursor in cursors]
    if len(set(streams)) == len(streams):
        return
    raise VenueSessionError(
        "Two cursors describe the same stream: "
        f"{sorted({str(stream) for stream in streams if streams.count(stream) > 1})}. "
        "Each recording is read once; a camera's several recordings are separate epochs."
    )

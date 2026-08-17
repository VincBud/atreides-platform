"""Domain models for venue ingestion: what happened at one venue instant.

A :class:`VenueTick` is a **scheduling** unit, not a finding. It says: at this
point on the venue timeline, these streams had a frame, and these did not — and
for each stream that did not, why. Nothing in it is a measurement, a judgement
or an identity.

Co-occurrence is not identity
-----------------------------
Two cameras contributing to the same tick means their frames landed on the same
venue instant. It does **not** mean they saw the same person, the same place, or
anything else about each other. When the tick's :attr:`VenueTick.alignment` is
``NOT_ESTABLISHED`` — the default, and the case whenever an operator has not
declared offsets — even the co-occurrence is *nominal*: the streams are ordered
against each other by construction, not by evidence.

No consumer may read a tick as a statement that observations from different
cameras are related. V0.4.0 performs no association: there is no venue-local
person identity here, no cross-camera correspondence, and nothing that suppresses
counting one person twice.

Absence is recorded, not inferred
---------------------------------
Every registered stream that did not contribute appears in
:attr:`VenueTick.absent` with a reason, because "camera 3 is missing from this
tick" has three very different meanings — it has not started yet, it is between
frames because it runs slower, or it has ended — and a consumer that cannot tell
them apart will eventually treat a slow camera as a dead one.

Ticks hold decoded frames. They are consumed and dropped, never retained: a list
of ticks is a list of video frames.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from atreides_vision.errors import VenueSessionError
from atreides_vision.timing import MICROS_PER_SECOND, StreamFrame, StreamRef, TimeUncertainty

__all__ = ["StreamAbsence", "StreamActivity", "VenueTick", "stream_sort_key"]


def stream_sort_key(stream: StreamRef) -> tuple[str, int]:
    """The canonical stream order: camera id by code point, then epoch.

    The one definition, shared by everything that orders streams, so a tick, a
    session record and a summary cannot disagree. ``camera_id`` compares by
    Python's default string ordering and never by the order cameras were listed
    or registered in: the same session with its manifest cameras reversed must
    produce identical output.
    """
    return (stream.camera_id, stream.stream_epoch)


class StreamActivity(StrEnum):
    """What one registered stream was doing at one venue instant.

    The three non-contributing members exist so that a gap in a stream's
    contributions is always explained. A live deployment will need a fourth,
    for a stream that failed mid-session; V0.4.0 fails the session instead, so
    inventing the member now would leave a state nothing can ever produce.
    """

    #: The stream had a frame at exactly this venue instant.
    CONTRIBUTED = "contributed"
    #: The stream is running, but its next frame is later than this instant —
    #: the ordinary case for any camera slower than the fastest one.
    BETWEEN_FRAMES = "between_frames"
    #: The stream's first frame is still ahead: it starts later in the session.
    NOT_YET_STARTED = "not_yet_started"
    #: The stream is exhausted and will never contribute again. Also the state,
    #: from the very first tick, of a stream that decoded no frames at all.
    ENDED = "ended"


@dataclass(frozen=True, slots=True)
class StreamAbsence:
    """One registered stream that did not contribute to a tick, and why."""

    stream: StreamRef
    activity: StreamActivity

    def __post_init__(self) -> None:
        if self.activity is StreamActivity.CONTRIBUTED:
            raise VenueSessionError(
                f"StreamAbsence describes a stream that did not contribute, but {self.stream} "
                "is marked CONTRIBUTED. A contributing stream belongs in the tick's "
                "contributions, with its frame."
            )


@dataclass(frozen=True, slots=True)
class VenueTick:
    """Every stream's state at one venue instant, in a total order.

    One tick exists per distinct :attr:`venue_micros` across all streams. Ticks
    are **source-event driven**: they happen when a frame happens, not on a fixed
    venue cadence. A fixed cadence would be a hardcoded frame rate by another
    name, and would force either interpolation or holding the last frame —
    inventing observations nobody made. A consumer that wants a regular cadence
    samples these with
    :class:`~atreides_vision.sampling.IntervalSampler`, as the rest of the
    pipeline already does.

    ``sequence`` is the 0-based index of this tick within the session. It is
    recorded in the data so a reader can confirm they have every tick, in order,
    without recomputing the schedule.
    """

    #: The venue instant, as the integer ordering key. Shared exactly by every
    #: contribution.
    venue_micros: int
    #: 0-based tick counter within the session.
    sequence: int
    #: The frames at this instant, in canonical order. Never empty.
    contributions: tuple[StreamFrame, ...]
    #: Every registered stream that had no frame here, in canonical order.
    absent: tuple[StreamAbsence, ...] = ()

    def __post_init__(self) -> None:
        if self.venue_micros < 0:
            raise VenueSessionError(
                f"VenueTick.venue_micros must be non-negative, got {self.venue_micros!r}"
            )
        if self.sequence < 0:
            raise VenueSessionError(
                f"VenueTick.sequence must be non-negative, got {self.sequence!r}"
            )
        if not self.contributions:
            raise VenueSessionError(
                "A venue tick must have at least one contributing stream. An instant nobody "
                "contributed a frame to is not an event that happened."
            )

        for contribution in self.contributions:
            if contribution.venue_micros != self.venue_micros:
                raise VenueSessionError(
                    f"Contribution {contribution.stream} is at venue instant "
                    f"{contribution.venue_micros} but the tick is at {self.venue_micros}. "
                    "A tick is one instant; grouping neighbouring frames into it would "
                    "silently claim a simultaneity nobody measured."
                )

        contributing = tuple(contribution.stream for contribution in self.contributions)
        absent = tuple(absence.stream for absence in self.absent)
        _reject_duplicates(contributing, "contributions")
        _reject_duplicates(absent, "absent")

        both = sorted(set(contributing) & set(absent), key=stream_sort_key)
        if both:
            raise VenueSessionError(
                f"Streams appear as both contributing and absent in one tick: "
                f"{[str(stream) for stream in both]}"
            )

        if list(contributing) != sorted(contributing, key=stream_sort_key):
            raise VenueSessionError(
                "VenueTick.contributions must be in canonical (camera_id, stream_epoch) order. "
                "Output order must never depend on the order cameras were registered in."
            )
        if list(absent) != sorted(absent, key=stream_sort_key):
            raise VenueSessionError(
                "VenueTick.absent must be in canonical (camera_id, stream_epoch) order."
            )

    @property
    def venue_seconds(self) -> float:
        """The tick instant in seconds — **quantised** to the ordering key.

        Good enough to place the tick on the timeline, and deliberately not the
        measurement: each contribution carries its own exact
        :attr:`~atreides_vision.timing.StreamFrame.venue_seconds`, computed in
        float from its source time. Anything measuring in seconds reads the
        contribution, not this.
        """
        return self.venue_micros / MICROS_PER_SECOND

    @property
    def alignment(self) -> TimeUncertainty:
        """How well the streams **contributing to this tick** are aligned.

        The worst of the contributions, so ``NOT_ESTABLISHED`` on any one of
        them makes this tick's co-occurrence nominal. Uncertainty travels with
        the tick rather than being reconstructed downstream.

        **This is not the session's alignment**, and must not be read as one. It
        covers contributors only, so it *improves* on ticks where a poorly
        aligned camera happens to be between frames or has ended — the same
        session looks better aligned on some lines than on others. Anything
        describing the session as a whole reads the session-level figure, which
        is the worst across every registered stream whether or not it
        contributed, and which the session record carries on every line.
        """
        first, *rest = self.contributions
        return first.timing.worst_of(*(contribution.timing for contribution in rest))

    def cameras(self) -> tuple[str, ...]:
        """The distinct cameras that contributed, in canonical order.

        A camera absent from this list did not have a frame at this instant;
        :attr:`absent` says why. Contributing is not a claim about what the
        camera saw.
        """
        seen: dict[str, None] = {}
        for contribution in self.contributions:
            seen.setdefault(contribution.stream.camera_id, None)
        return tuple(seen)

    def activity_of(self, camera_id: str) -> StreamActivity | None:
        """What this camera was doing, or ``None`` if it is not in this tick.

        A camera can own several streams — one per recording in the session — so
        the answer is taken over all of them, most-alive first: contributed,
        then between frames, then not yet started, then ended.

        A **restart gap** is the case worth naming: one recording has ended and
        the next has not begun, so the camera has demonstrably started and will
        demonstrably contribute again. That is reported as ``BETWEEN_FRAMES``,
        which at camera level covers both a camera slower than the tick cadence
        and a camera between recordings. Reporting ``NOT_YET_STARTED`` there
        would describe a camera that has not run yet, and ``ENDED`` one that
        never will. Per-stream detail stays in :attr:`absent`, where the two
        epochs are separately visible.

        ``None`` means the camera is not part of this session at all, which is a
        caller error worth seeing rather than a silent ``ENDED``.
        """
        if any(contribution.stream.camera_id == camera_id for contribution in self.contributions):
            return StreamActivity.CONTRIBUTED

        activities = {
            absence.activity for absence in self.absent if absence.stream.camera_id == camera_id
        }
        if not activities:
            return None
        if StreamActivity.BETWEEN_FRAMES in activities:
            return StreamActivity.BETWEEN_FRAMES
        if activities == {StreamActivity.NOT_YET_STARTED, StreamActivity.ENDED}:
            return StreamActivity.BETWEEN_FRAMES
        if StreamActivity.NOT_YET_STARTED in activities:
            return StreamActivity.NOT_YET_STARTED
        return StreamActivity.ENDED


def _reject_duplicates(streams: tuple[StreamRef, ...], where: str) -> None:
    counts = Counter(streams)
    repeated = sorted(
        (stream for stream, count in counts.items() if count > 1), key=stream_sort_key
    )
    if not repeated:
        return
    raise VenueSessionError(
        f"A stream appears more than once in a tick's {where}: "
        f"{[str(stream) for stream in repeated]}. Each stream contributes at most one frame "
        "per venue instant."
    )

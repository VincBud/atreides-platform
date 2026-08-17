"""Time semantics shared by every camera in a venue: whose clock, and how sure.

Until V0.4.0 there was one camera, so "seconds" meant one thing and needed no
qualification. A venue has several, each decoded by its own reader, each with
its own frame rate, and each with its own idea of when zero was. The moment two
of them are read together, a bare number of seconds stops being interpretable.
This module is where the private clocks are named and reconciled.

Two seconds, not one
--------------------
Every frame carries **both** times, and they answer different questions:

* ``source_seconds`` — elapsed time inside that stream, ``(frame_index - 1) /
  fps``. It is provenance: it says where in *that file* the frame is, and it
  restarts at zero when a camera's next recording begins.
* ``venue_seconds`` — the same instant expressed on the one session-relative
  venue timeline, ``source_seconds + offset``. It is the common domain, and it
  is what any measurement above the camera must use, because it is the only one
  that stays monotonic across a stream restart.

Venue time is **session-relative**: zero is the start of the session, not an
absolute wall-clock instant. No date, no time of day and no NTP dependency
enters here.

Uncertainty is first class, and has no threshold
------------------------------------------------
Two cameras' timelines are only related to each other as well as somebody
established the relation. :class:`TimeUncertainty` records exactly that, and
:attr:`TimeAlignmentBasis.NOT_ESTABLISHED` — "nobody established it" — is the
default and a perfectly good answer.

There is deliberately **no comparison operator, no threshold and no
``is_aligned_enough()``**. A boolean derived from a half-width is the first step
towards deciding that two observations are the same person, which is a
capability this milestone does not have and must not appear to have. Consumers
carry the uncertainty forward and report it; they do not collapse it.

Errors here are :class:`~atreides_vision.errors.VenueSessionError` rather than
``ValueError`` because every one of these values arrives from an operator's
manifest, and an operator needs a sentence, not a traceback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

import numpy as np

from atreides_vision.errors import VenueSessionError

__all__ = [
    "MICROS_PER_SECOND",
    "FrameRateSource",
    "StreamFrame",
    "StreamRef",
    "TimeAlignmentBasis",
    "TimeUncertainty",
]

#: Venue time is ordered on an integer microsecond key so that two streams
#: landing on the same instant compare exactly equal, with no tolerance constant
#: anywhere. A microsecond is far finer than any frame rate a camera produces,
#: so the quantisation never merges two genuinely distinct frame times.
MICROS_PER_SECOND = 1_000_000


class FrameRateSource(StrEnum):
    """Where a stream's frame rate came from, which decides what it is worth.

    A frame rate is not a neutral number: every source-relative timestamp in
    the system is derived from it. A container that reports a bogus rate is
    common in real CCTV exports, and the fallback the reader then applies is a
    guess. Recording *which* of the three it was is what stops a guess from
    being read later as a measurement.
    """

    #: Read from the container and usable as reported.
    CONTAINER = "container"
    #: The container reported nothing usable and a fallback was assumed. Every
    #: timestamp derived from it drifts from real time by an unknown factor,
    #: and the error grows with elapsed time rather than staying bounded.
    ASSUMED = "assumed"
    #: Supplied by the operator in a manifest, overriding the container. The
    #: honest remedy for a container that lies about its own frame rate.
    DECLARED = "declared"

    @property
    def is_trustworthy(self) -> bool:
        """Whether timestamps derived from this rate describe elapsed time.

        ``DECLARED`` counts as trustworthy: an operator stating the rate of
        their own recording is evidence, where a fallback constant is not.
        """
        return self is not FrameRateSource.ASSUMED


class TimeAlignmentBasis(StrEnum):
    """How a stream's position on the venue timeline came to be known.

    Only two members exist because only two situations are real today: either
    an operator told us, or nobody did. Members for a detected or estimated
    alignment would describe capabilities that do not exist, and a consumer
    would have no way to tell the difference from the data.
    """

    #: Nothing relates this stream's clock to any other. The default.
    NOT_ESTABLISHED = "not_established"
    #: An operator declared the offset, and stated how accurate it is.
    OPERATOR_DECLARED = "operator_declared"


@dataclass(frozen=True, slots=True)
class TimeUncertainty:
    """How well a stream's time is known relative to the venue timeline.

    ``half_width_seconds`` is a symmetric bound: the true instant is believed to
    lie within ``+/- half_width_seconds`` of the stated venue time. It is not a
    standard deviation and not a probability.

    The two fields move together and are validated as one: an unestablished
    alignment has no half-width, and a declared one always has a positive one.
    "Aligned, accuracy unknown" is not a state — it is ``NOT_ESTABLISHED``.
    """

    basis: TimeAlignmentBasis
    #: ``None`` exactly when the basis is ``NOT_ESTABLISHED``.
    half_width_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.basis is TimeAlignmentBasis.NOT_ESTABLISHED:
            if self.half_width_seconds is not None:
                raise VenueSessionError(
                    "A NOT_ESTABLISHED alignment cannot carry a half-width "
                    f"({self.half_width_seconds!r}). A number here would claim an accuracy "
                    "nobody established. Use TimeUncertainty.declared() when an offset "
                    "really was measured."
                )
            return

        if self.half_width_seconds is None:
            raise VenueSessionError(
                f"A {self.basis.value} alignment must state its accuracy. An offset with no "
                "half-width is a manifest error, not a zero: 'aligned, accuracy unknown' is "
                "NOT_ESTABLISHED."
            )
        if not math.isfinite(self.half_width_seconds):
            raise VenueSessionError(
                f"half_width_seconds must be finite, got {self.half_width_seconds!r}. "
                "An infinite or NaN half-width describes no interval at all."
            )
        if self.half_width_seconds <= 0:
            raise VenueSessionError(
                f"half_width_seconds must be strictly positive, got "
                f"{self.half_width_seconds!r}. Zero would claim the streams are aligned "
                "exactly, which no declared offset ever is."
            )

    @classmethod
    def unknown(cls) -> Self:
        """The default: nothing relates this stream's clock to any other."""
        return cls(basis=TimeAlignmentBasis.NOT_ESTABLISHED)

    @classmethod
    def declared(cls, half_width_seconds: float) -> Self:
        """An operator-declared alignment, accurate to ``+/- half_width_seconds``."""
        return cls(
            basis=TimeAlignmentBasis.OPERATOR_DECLARED,
            half_width_seconds=half_width_seconds,
        )

    @property
    def is_established(self) -> bool:
        """Whether anything at all relates this stream to the venue timeline."""
        return self.basis is not TimeAlignmentBasis.NOT_ESTABLISHED

    def worst_of(self, *others: TimeUncertainty) -> TimeUncertainty:
        """The weakest of this uncertainty and ``others``.

        ``NOT_ESTABLISHED`` **absorbs**: combining a well-declared stream with
        one nobody aligned yields ``NOT_ESTABLISHED``, because the combination
        is only as related as its least related member. Otherwise the widest
        half-width wins.

        Used to describe a whole tick: several streams landing on one venue
        instant are co-located in venue time only as well as the worst of them
        is aligned.
        """
        widest = self
        for other in others:
            # ``half_width_seconds is None`` is exactly ``NOT_ESTABLISHED`` —
            # __post_init__ enforces the equivalence — and narrows for the
            # comparison below, where a property would not.
            if widest.half_width_seconds is None:
                return widest
            if other.half_width_seconds is None:
                return other
            if other.half_width_seconds > widest.half_width_seconds:
                widest = other
        return widest


@dataclass(frozen=True, slots=True)
class StreamRef:
    """One continuous recording from one camera, within one session.

    Not a camera and not a subject. A camera that stops and starts again during
    a session produces two streams, and the second one's tracker numbering has
    nothing to do with the first's: ``(camera_01, track_7)`` before an epoch
    boundary and ``(camera_01, track_7)`` after it are unrelated tracklets that
    happen to share an integer.

    ``stream_epoch`` is the 0-based index of the source in that camera's ordered
    source list in the session manifest. It is not a clock, a UUID, a hash or a
    file timestamp — just a position, unique within ``(session, camera)``,
    session-scoped and never persisted.

    :meth:`__str__` uses ``#`` where
    :class:`~atreides_vision.behavior.models.SubjectRef` uses ``/``. The
    separators differ on purpose: ``cam_01#0`` is a *stream*, ``cam_01/7`` is a
    *tracklet*, and a reader skimming a log must never mistake one for the
    other. A ``StreamRef`` never appears inside a ``SubjectRef``, and never
    becomes one.
    """

    camera_id: str
    stream_epoch: int

    def __post_init__(self) -> None:
        if not self.camera_id or not self.camera_id.strip():
            raise VenueSessionError("StreamRef.camera_id must be a non-empty string")
        if self.stream_epoch < 0:
            raise VenueSessionError(
                f"StreamRef.stream_epoch must be non-negative, got {self.stream_epoch!r}. "
                "It is the index of the source in this camera's manifest list."
            )

    def __str__(self) -> str:
        return f"{self.camera_id}#{self.stream_epoch}"


@dataclass(frozen=True, slots=True)
class StreamFrame:
    """One decoded frame, placed on both its own timeline and the venue's.

    The unit the venue layer schedules on. It carries the pixels only for as
    long as the frame is being processed: nothing retains a ``StreamFrame``, and
    holding a sequence of them would hold a sequence of video frames in memory.

    Provenance is complete by construction — the stream, the frame's index
    within that stream, and its position inside its own source — so a venue
    observation can always be resolved back to the exact frame of the exact file
    it came from.
    """

    stream: StreamRef
    #: 1-based within its stream, matching the pipeline's frame numbering.
    frame_index: int
    #: Elapsed seconds within this stream. Restarts at an epoch boundary.
    source_seconds: float
    #: The same instant on the session-relative venue timeline.
    venue_seconds: float
    #: How well this stream is aligned to the venue timeline.
    timing: TimeUncertainty
    #: The decoded BGR frame. Excluded from equality and hashing on purpose: a
    #: frozen dataclass comparing two ``np.ndarray`` fields evaluates
    #: ``array == array``, which produces an array whose truth value is
    #: ambiguous, and raises. Comparing the metadata is what callers want in any
    #: case — two records describing the same frame of the same stream.
    image: np.ndarray = field(compare=False)

    def __post_init__(self) -> None:
        if self.frame_index < 1:
            raise VenueSessionError(f"StreamFrame.frame_index is 1-based, got {self.frame_index!r}")
        for name, value in (
            ("source_seconds", self.source_seconds),
            ("venue_seconds", self.venue_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise VenueSessionError(
                    f"StreamFrame.{name} must be finite and non-negative, got {value!r}. "
                    "Venue time is session-relative, so nothing precedes zero."
                )

    @property
    def venue_micros(self) -> int:
        """The venue instant as integer microseconds — the **ordering key**.

        Ordering only. Two streams whose venue seconds agree to the microsecond
        share a key exactly, with no tolerance constant, which is what makes the
        merge order total and reproducible.

        It is not the measurement: :attr:`venue_seconds` is, and it is computed
        in float from ``source_seconds + offset``. Deriving a time *from* this
        key would silently quantise every downstream second.
        """
        return round(self.venue_seconds * MICROS_PER_SECOND)

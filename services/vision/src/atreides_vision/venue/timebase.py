"""The one session-relative timeline that every stream's seconds map onto.

Each stream decodes into its own elapsed seconds, starting at zero. Venue time
is the same instants expressed on one shared, session-relative axis, so that
frames from different cameras can be put in order at all:

    venue_seconds = source_seconds + offset_seconds

computed in **float**, from source seconds. Never from
:attr:`~atreides_vision.timing.StreamFrame.venue_micros`, which is an ordering
key: deriving the measurement from the key would quantise every second above it
to a microsecond grid for no reason.

Nothing here is absolute time. Zero is the start of the session, and a session's
numbers mean nothing outside it.

Placement and accuracy are separate concerns
--------------------------------------------
*Where* a stream sits on the venue timeline and *how well that is known* are
different questions, and this module keeps them apart.

**It will not put a fixed accuracy on a drifting clock.** A stream whose frame
rate was only assumed — its container reported nothing usable — has source
seconds that diverge from real elapsed time by an unknown factor, so the error
grows without bound as the recording runs. A ``+/- 0.5 s`` half-width cannot
describe that, and pretending it can is how a guess becomes a confident number.

Such a stream may still be **placed**: an operator's ``start_offset_seconds``
positions it, which is what makes an ordinary NVR export usable — camera
restarted mid-session, container reporting a useless frame rate. Its
:class:`~atreides_vision.timing.TimeUncertainty` is nevertheless forced to
``NOT_ESTABLISHED``, and any accuracy supplied with it is discarded and logged
at WARNING. Refusing the offset outright would leave that deployment
unrepresentable and push an operator into inventing a ``declared_fps`` — which
*is* treated as trustworthy — laundering a guess into a measurement. Accepting
the placement while refusing the accuracy claim keeps the honesty and removes
the incentive.

**It has no default offset that is a guess.** A stream with nothing declared
sits at venue zero with its alignment ``NOT_ESTABLISHED``. That is a default,
not a fallback: it is the honest description of a stream nobody aligned, and it
is the ordinary single-camera case.
"""

from __future__ import annotations

import logging
import math

from atreides_vision.errors import VenueSessionError
from atreides_vision.timing import FrameRateSource, StreamRef, TimeUncertainty
from atreides_vision.venue.models import stream_sort_key

logger = logging.getLogger(__name__)

__all__ = ["VenueTimebase"]


class VenueTimebase:
    """Registers streams and converts their seconds onto the venue timeline.

    Bounded by the number of streams: two small values per stream, and nothing
    per frame. It is consulted once per decoded frame and remembers none of them.
    """

    def __init__(self) -> None:
        self._offsets: dict[StreamRef, float] = {}
        self._uncertainties: dict[StreamRef, TimeUncertainty] = {}

    def register(
        self,
        stream: StreamRef,
        *,
        fps_source: FrameRateSource,
        offset_seconds: float | None = None,
        offset_accuracy_seconds: float | None = None,
    ) -> None:
        """Place one stream on the venue timeline.

        Args:
            stream: The stream being registered. Registering one twice is an
                error: two registrations would mean two different answers for
                the same frames.
            fps_source: Where the stream's frame rate came from. An assumed rate
                is incompatible with a declared offset.
            offset_seconds: Where this stream's zero sits on the venue timeline.
                ``None`` means venue zero, alignment ``NOT_ESTABLISHED``.
            offset_accuracy_seconds: Symmetric accuracy of ``offset_seconds``.
                Required with it, meaningless without it, and **discarded** —
                with a warning — when ``fps_source`` is ``ASSUMED``, because a
                guessed frame rate cannot support an accuracy claim.

        Raises:
            VenueSessionError: for a duplicate registration, an offset without
                an accuracy (or the reverse), or a non-finite or negative
                offset.
        """
        if stream in self._offsets:
            raise VenueSessionError(
                f"Stream {stream} is already registered on this timebase. Two registrations "
                "would give the same frames two different venue times."
            )

        if offset_seconds is None:
            if offset_accuracy_seconds is not None:
                raise VenueSessionError(
                    f"Stream {stream} was given an offset accuracy but no offset. An accuracy "
                    "describes a declaration; on its own it describes nothing."
                )
            self._offsets[stream] = 0.0
            self._uncertainties[stream] = TimeUncertainty.unknown()
            return

        if not math.isfinite(offset_seconds) or offset_seconds < 0:
            raise VenueSessionError(
                f"Stream {stream}: offset_seconds must be finite and non-negative, got "
                f"{offset_seconds!r}. Venue time is session-relative, so nothing precedes zero."
            )
        if offset_accuracy_seconds is None:
            raise VenueSessionError(
                f"Stream {stream}: a declared offset needs an accuracy. Without one the offset "
                "would be read as exact, which no declared offset is."
            )

        # The offset still places the stream — that part is the operator's to
        # state. The accuracy is not, when the seconds it would qualify come
        # from a guessed frame rate: the error grows with elapsed time, so no
        # fixed half-width describes it. Discarded loudly, never quietly.
        self._offsets[stream] = float(offset_seconds)
        if fps_source is FrameRateSource.ASSUMED:
            logger.warning(
                "Stream %s: discarding the declared offset accuracy of %.3f s. Its container "
                "reported no usable frame rate, so its source seconds drift from real elapsed "
                "time by an unknown factor that grows as the recording runs, and no fixed "
                "half-width can describe that. The stream is still placed at %.3f s; its "
                "alignment is NOT_ESTABLISHED.",
                stream,
                offset_accuracy_seconds,
                offset_seconds,
            )
            self._uncertainties[stream] = TimeUncertainty.unknown()
            return
        self._uncertainties[stream] = TimeUncertainty.declared(offset_accuracy_seconds)

    @property
    def streams(self) -> tuple[StreamRef, ...]:
        """Every registered stream, in canonical order — never registration order."""
        return tuple(sorted(self._offsets, key=stream_sort_key))

    def is_registered(self, stream: StreamRef) -> bool:
        return stream in self._offsets

    def offset_seconds(self, stream: StreamRef) -> float:
        """Where this stream's zero sits on the venue timeline."""
        return self._require(stream)

    def uncertainty(self, stream: StreamRef) -> TimeUncertainty:
        """How well this stream is aligned to the venue timeline."""
        self._require(stream)
        return self._uncertainties[stream]

    def venue_seconds(self, stream: StreamRef, source_seconds: float) -> float:
        """This stream's elapsed seconds, expressed on the venue timeline.

        A single float addition, on purpose. With one camera and no declared
        offset the offset is ``0.0`` and the result is bit-identical to the
        source time — which is what makes the single-camera path a degenerate
        case of the venue path rather than a separate mode.
        """
        offset = self._require(stream)
        if not math.isfinite(source_seconds) or source_seconds < 0:
            raise VenueSessionError(
                f"Stream {stream}: source_seconds must be finite and non-negative, got "
                f"{source_seconds!r}"
            )
        return source_seconds + offset

    def _require(self, stream: StreamRef) -> float:
        try:
            return self._offsets[stream]
        except KeyError:
            known = ", ".join(str(registered) for registered in self.streams) or "none"
            raise VenueSessionError(
                f"Stream {stream} is not registered on this timebase, so it has no place on "
                f"the venue timeline. Registered streams: {known}."
            ) from None

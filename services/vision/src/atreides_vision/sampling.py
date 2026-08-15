"""Deterministic interval sampling over source time.

Three things in the pipeline need the same decision — "has enough *source* time
passed to do this again?": the zone metrics stream, behaviour output emission,
and calibration-integrity checks. They share this one implementation rather
than each growing its own modulo arithmetic.

Source time, not frame count, is what the interval is measured in. A 25 FPS and
a 60 FPS recording of the same scene therefore sample at the same points, and
no user-facing setting has to be re-tuned when footage changes frame rate.
"""

from __future__ import annotations

__all__ = ["IntervalSampler"]

#: Fraction of an interval within which a timestamp counts as having reached a
#: boundary. Guards against binary representation putting an exactly-on-time
#: sample a hair below its own deadline.
_BOUNDARY_TOLERANCE = 1e-9


class IntervalSampler:
    """Decides which moments in source time are due, at a fixed interval.

    The first moment offered is always due; afterwards a moment is due once the
    interval has elapsed since the last one taken. Sampling from the very first
    frame matters because a run that is shorter than one interval must still
    produce something.

    Timestamps are expected to advance, which they do for any decoded stream.
    A timestamp that goes backwards — a caller resetting or replaying — is
    treated as not yet due rather than as an error, so a misuse degrades to
    "sampled less often" rather than to a crash mid-run.
    """

    def __init__(self, interval_seconds: float = 1.0) -> None:
        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be positive, got {interval_seconds}")
        self._interval = interval_seconds
        self._origin: float | None = None
        self._last_interval_index = 0

    @property
    def interval_seconds(self) -> float:
        return self._interval

    def should_sample(self, timestamp_seconds: float) -> bool:
        """Whether this moment is due, advancing the schedule if it is.

        The schedule is computed from the **first** timestamp each time rather
        than accumulated from the last one taken. Accumulating adds each
        sample's floating-point error to the next deadline, which over a few
        hundred frames slides the schedule by whole frames and makes the output
        depend on rounding. Anchoring to the origin keeps every deadline exact.

        A gap in the stream also resolves cleanly: the schedule jumps straight
        to the interval the current moment falls in, so a stall produces one
        sample on resumption rather than a burst catching up on missed ones.
        """
        if self._origin is None:
            self._origin = timestamp_seconds
            self._last_interval_index = 0
            return True

        # The tolerance matters more than it looks. A frame landing exactly on a
        # boundary — 0.6 s into a 0.2 s schedule — divides to 2.9999999999999996
        # in binary, which truncates to the *previous* interval and silently
        # shifts that sample a frame later. Nudging by one part in a billion
        # treats "indistinguishable from the boundary" as being on it, and is far
        # too small to promote a timestamp that genuinely has not arrived.
        interval_index = int(
            (timestamp_seconds - self._origin) / self._interval + _BOUNDARY_TOLERANCE
        )
        if interval_index > self._last_interval_index:
            self._last_interval_index = interval_index
            return True
        return False

    def reset(self) -> None:
        """Forget the schedule, so the next moment offered is due again."""
        self._origin = None
        self._last_interval_index = 0

"""WILDTRACK frame identifiers and their mapping onto validation instants.

Every image and annotation filename is an eight-digit, zero-padded integer
(``00000000.png``, ``00000005.png``, ...). WILDTRACK's own paper documents
the two facts needed to interpret that number rather than guess at it:

* the seven cameras were synchronized and frames extracted at **10 fps**;
* full ground-truth annotation is provided for 400 frames at **2 fps**.

The released files step by exactly 5 (``FRAME_STEP``) between entries, and
5 frames at 10 fps is exactly 0.5 s, i.e. 2 fps — the documented annotation
cadence, not a number invented from the step alone. The filename is therefore
a frame count at the 10 fps synchronization rate, and the wall-clock-free
venue-relative instant it names is ``frame_id / FRAME_RATE_HZ`` seconds.

Nothing here reads a system clock. ``venue_seconds`` is derived once from a
documented dataset property and is otherwise ordinary arithmetic on the
frame identifier already present in the filename.
"""

from __future__ import annotations

from dataclasses import dataclass

from atreides_vision.errors import WildtrackDatasetError

__all__ = [
    "FRAME_RATE_HZ",
    "FRAME_STEP",
    "WildtrackFrameId",
]

#: The synchronized-extraction frame rate the released frame identifiers are
#: counted in. Documented in the WILDTRACK paper, not inferred.
FRAME_RATE_HZ = 10.0

#: The step, in frame-identifier units, between consecutive released frames
#: (images and, where present, annotations). ``FRAME_STEP / FRAME_RATE_HZ``
#: is exactly the documented 2 fps annotation cadence.
FRAME_STEP = 5


@dataclass(frozen=True, slots=True)
class WildtrackFrameId:
    """One WILDTRACK frame identifier, and the venue instant it names.

    ``value`` is the raw integer encoded in the eight-digit filename. It is
    not itself a count of released frames — consecutive released frames
    differ by :data:`FRAME_STEP`, not by 1.
    """

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise WildtrackDatasetError(f"WILDTRACK frame id must be >= 0, got {self.value!r}")
        if self.value % FRAME_STEP != 0:
            raise WildtrackDatasetError(
                f"WILDTRACK frame id {self.value!r} is not a multiple of the released frame "
                f"step ({FRAME_STEP}); this is not one of the dataset's released frames."
            )

    @classmethod
    def from_filename(cls, filename: str) -> WildtrackFrameId:
        """Parse a WILDTRACK filename stem such as ``"00000005"`` or ``"00000005.png"``."""
        stem = filename.split(".", 1)[0]
        if not stem.isdigit():
            raise WildtrackDatasetError(
                f"WILDTRACK filename {filename!r} does not encode an integer frame id"
            )
        return cls(int(stem))

    @property
    def filename_stem(self) -> str:
        """The canonical eight-digit filename stem, e.g. ``"00000005"``."""
        return f"{self.value:08d}"

    @property
    def venue_seconds(self) -> float:
        """This frame's instant on the venue-relative timeline, in seconds."""
        return self.value / FRAME_RATE_HZ

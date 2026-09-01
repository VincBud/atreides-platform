"""WILDTRACK person-level ground truth, held strictly downstream of the runtime.

Nothing in this module is read by the venue-validation runner, the observer
seam, or any association / fusion / identity / occupancy algorithm. It is loaded
by the offline evaluator *after* the runtime has produced its snapshots, and
compared against them. ``WildtrackPersonId`` never crosses back into runtime
code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from atreides_vision.errors import ValidationError
from atreides_vision.venue_validation.datasets.wildtrack.annotations import (
    WildtrackFrameAnnotations,
    load_frame_annotations,
)
from atreides_vision.venue_validation.datasets.wildtrack.layout import WildtrackDatasetLayout
from atreides_vision.venue_validation.datasets.wildtrack.time import WildtrackFrameId

__all__ = ["WildtrackGroundTruth"]


@dataclass(frozen=True, slots=True)
class WildtrackGroundTruth:
    """The annotated frames a WILDTRACK evaluation run scores against.

    ``frames[k]`` is the ground truth for the ``k``-th scored venue instant —
    the same ordinal the evaluator receives from the observer seam and the same
    ordinal as ``WildtrackRuntimeSession.scored_frame_ids[k]``.
    """

    frames: tuple[WildtrackFrameAnnotations, ...]

    @classmethod
    def from_layout(
        cls,
        layout: WildtrackDatasetLayout,
        scored_frame_ids: Sequence[WildtrackFrameId],
    ) -> WildtrackGroundTruth:
        """Load the annotation file for every scored frame id, in order."""
        if not scored_frame_ids:
            raise ValidationError(
                "WildtrackGroundTruth.from_layout needs at least one scored frame id."
            )
        frames: list[WildtrackFrameAnnotations] = []
        for frame_id in scored_frame_ids:
            path: Path = layout.annotation_path(frame_id)
            frames.append(load_frame_annotations(path, frame_id))
        return cls(frames=tuple(frames))

    def __len__(self) -> int:
        return len(self.frames)

    def frame(self, instant_index: int) -> WildtrackFrameAnnotations:
        if not 0 <= instant_index < len(self.frames):
            raise ValidationError(
                f"WILDTRACK ground truth has {len(self.frames)} scored frames; "
                f"instant index {instant_index} is out of range."
            )
        return self.frames[instant_index]

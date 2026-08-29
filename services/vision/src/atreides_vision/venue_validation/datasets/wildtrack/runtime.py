"""Feeding WILDTRACK's synchronized frames into the real Atreides runtime.

This is the one WILDTRACK module that produces something the *runtime* consumes
rather than something the *evaluator* consumes, and it is scoped accordingly: it
knows about images, frame identifiers, cameras and time, and nothing else. It
does not import — and must never import — WILDTRACK's person annotations, its
cross-camera correspondence or its ground-truth occupancy. The only ground-truth
fact it uses is structural: *which frame identifiers carry an annotation file*,
taken from :attr:`WildtrackDatasetLayout.annotation_frame_ids`. That is what
selects the 400 scored instants and, by the same stroke, excludes the single
trailing extracted image that has no annotation.

A :class:`WildtrackStreamSpec` is a deterministic, path-carrying description of
one camera's contribution — canonical camera order, canonical frame order, the
declared 2 fps rate derived from the dataset's own documented timing, and the
venue-timeline offset (exactly zero: the seven cameras are synchronized and
frame ``00000000`` is every camera's instant zero). Opening it yields an
ordinary :class:`ImageSequenceVideoSource`. Registration on a
:class:`~atreides_vision.venue.timebase.VenueTimebase` belongs to the runner
seam that consumes these specs, not here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from atreides_vision.errors import WildtrackDatasetError
from atreides_vision.venue_validation.datasets.wildtrack.cameras import CAMERA_IDS
from atreides_vision.venue_validation.datasets.wildtrack.layout import WildtrackDatasetLayout
from atreides_vision.venue_validation.datasets.wildtrack.time import (
    FRAME_RATE_HZ,
    FRAME_STEP,
    WildtrackFrameId,
)
from atreides_vision.video import ImageSequenceVideoSource

__all__ = [
    "WILDTRACK_DECLARED_FPS",
    "WILDTRACK_STREAM_OFFSET_ACCURACY_SECONDS",
    "WILDTRACK_STREAM_OFFSET_SECONDS",
    "WildtrackStreamSpec",
    "build_wildtrack_stream_specs",
]

#: The annotation cadence, derived from WILDTRACK's own documented timing
#: (:data:`FRAME_RATE_HZ` synchronized extraction, one annotation every
#: :data:`FRAME_STEP` frames) — never a bare literal ``2.0``.
WILDTRACK_DECLARED_FPS = FRAME_RATE_HZ / FRAME_STEP

#: WILDTRACK's seven cameras are hardware-synchronized and its frame identifiers
#: are a shared frame count: frame ``00000000`` is one instant across all seven
#: views. Every stream therefore sits on the venue timeline with no offset...
WILDTRACK_STREAM_OFFSET_SECONDS = 0.0
#: ...and, because that offset is a property of the published synchronization
#: rather than an operator's estimate, with no offset uncertainty either.
WILDTRACK_STREAM_OFFSET_ACCURACY_SECONDS = 0.0


@dataclass(frozen=True, slots=True)
class WildtrackStreamSpec:
    """One WILDTRACK camera's runtime contribution, fully resolved and ordered.

    ``image_paths`` is aligned index-for-index with ``frame_ids``: both are in
    ascending frame-identifier order, which for WILDTRACK is also ascending
    venue time.
    """

    camera_id: str
    image_paths: tuple[Path, ...]
    frame_ids: tuple[WildtrackFrameId, ...]
    declared_fps: float = WILDTRACK_DECLARED_FPS
    offset_seconds: float = WILDTRACK_STREAM_OFFSET_SECONDS
    offset_accuracy_seconds: float = WILDTRACK_STREAM_OFFSET_ACCURACY_SECONDS

    def open(self) -> ImageSequenceVideoSource:
        """Build the image-sequence source for this camera."""
        return ImageSequenceVideoSource(self.image_paths, declared_fps=self.declared_fps)


def build_wildtrack_stream_specs(
    layout: WildtrackDatasetLayout,
    *,
    cameras: Iterable[str] | None = None,
) -> tuple[WildtrackStreamSpec, ...]:
    """Resolve one :class:`WildtrackStreamSpec` per selected camera.

    The scored frame set is :attr:`WildtrackDatasetLayout.annotation_frame_ids`
    verbatim — every camera contributes exactly those instants, so the venue
    session is perfectly synchronized and the lone un-annotated trailing image
    is structurally absent.

    Args:
        layout: A validated dataset layout.
        cameras: The camera subset to include, in any order. ``None`` selects
            all seven. The result is always in canonical ``C1..C7`` order
            regardless of this argument's order.

    Raises:
        WildtrackDatasetError: if ``cameras`` names an unknown or duplicate
            camera, if no cameras are selected, if the dataset carries no
            annotation frames, or if any selected camera is missing an image
            for any required annotation frame.
    """
    selected = _canonical_camera_selection(cameras)

    frame_ids = layout.annotation_frame_ids
    if not frame_ids:
        raise WildtrackDatasetError(
            f"WILDTRACK dataset at {layout.root} has no annotation frames; there is no scored "
            "instant set to build runtime streams from."
        )

    specs: list[WildtrackStreamSpec] = []
    for camera_id in selected:
        image_paths: list[Path] = []
        for frame_id in frame_ids:
            image_path = layout.image_path(camera_id, frame_id)
            if not image_path.is_file():
                raise WildtrackDatasetError(
                    f"WILDTRACK camera {camera_id} is missing the image for annotation frame "
                    f"{frame_id.filename_stem} ({image_path}); the selected cameras are not "
                    "synchronized over the scored frame set."
                )
            image_paths.append(image_path)
        specs.append(
            WildtrackStreamSpec(
                camera_id=camera_id,
                image_paths=tuple(image_paths),
                frame_ids=frame_ids,
            )
        )
    return tuple(specs)


def _canonical_camera_selection(cameras: Iterable[str] | None) -> tuple[str, ...]:
    if cameras is None:
        return CAMERA_IDS
    requested = list(cameras)
    unknown = sorted(set(requested) - set(CAMERA_IDS))
    if unknown:
        raise WildtrackDatasetError(
            f"Unknown WILDTRACK camera id(s) {unknown}; expected a subset of {CAMERA_IDS}."
        )
    seen = set(requested)
    if len(seen) != len(requested):
        duplicates = sorted({c for c in requested if requested.count(c) > 1})
        raise WildtrackDatasetError(
            f"WILDTRACK camera selection lists {duplicates} more than once."
        )
    if not seen:
        raise WildtrackDatasetError("WILDTRACK camera selection is empty; select at least one.")
    return tuple(camera_id for camera_id in CAMERA_IDS if camera_id in seen)

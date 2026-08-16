"""The deterministic human-review sample: which frames, and which short windows.

Two properties matter more than any detail of the layout.

**Deterministic.** The same clip always yields the same sample points, so an
annotation made today still describes the frame a re-run produces next month.
Sample points are computed from the clip's frame count alone — no randomness, no
clock, no dependence on how the run went.

**Spread.** Points sit at the midpoints of ``n`` equal slices — ``(i + 0.5) / n``
of the clip — rather than at its start. Sampling the first seconds would measure
the tracker's warm-up and nothing else, and sampling exact fractions would put a
point on the first and last frames, which are the two least representative
frames in any clip.

Frames are extracted from the **rendered tracking video**, not the source, so a
reviewer sees the boxes, track ids, confidences and trails the run actually
produced. Nothing new is drawn: this is a consumer of the existing overlay, and
adding a validation-only annotation layer would mean reviewing something the
production pipeline never emits.

Nothing here labels anything. The sample plan says which frames to look at; what
is in them is for a human to write down.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from atreides_vision.errors import ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CONTINUITY_WINDOWS",
    "DEFAULT_CONTINUITY_WINDOW_SECONDS",
    "DEFAULT_SAMPLE_FRAMES",
    "ContinuityWindow",
    "ReviewSamplePlan",
    "SampleFrame",
    "build_sample_plan",
    "extract_sample_frames",
    "write_contact_sheet",
]

#: Still frames sampled per clip for detection review.
DEFAULT_SAMPLE_FRAMES = 10
#: Short windows sampled per clip for track-continuity review.
DEFAULT_CONTINUITY_WINDOWS = 3
#: Length of each continuity window, in source seconds.
DEFAULT_CONTINUITY_WINDOW_SECONDS = 8.0

#: Contact-sheet geometry. Wide enough to read a track id at a glance, small
#: enough that ten 1080p frames fit in a file worth opening.
_CONTACT_SHEET_COLUMNS = 5
_CONTACT_SHEET_TILE_WIDTH = 640
_CONTACT_SHEET_MARGIN = 8
_CONTACT_SHEET_LABEL_HEIGHT = 28


@dataclass(frozen=True, slots=True)
class SampleFrame:
    """One deterministic still to review."""

    index: int
    frame_index: int
    timestamp_seconds: float

    @property
    def filename(self) -> str:
        """Stable filename: sortable, and self-describing without the plan file."""
        return (
            f"sample_{self.index:02d}_f{self.frame_index:06d}_t{self.timestamp_seconds:07.3f}.jpg"
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "filename": self.filename,
        }


@dataclass(frozen=True, slots=True)
class ContinuityWindow:
    """One short window to follow individual subjects through by hand."""

    sequence_id: str
    start_seconds: float
    end_seconds: float
    start_frame_index: int
    end_frame_index: int

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def to_record(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "start_frame_index": self.start_frame_index,
            "end_frame_index": self.end_frame_index,
            "duration_seconds": round(self.duration_seconds, 3),
        }


@dataclass(frozen=True, slots=True)
class ReviewSamplePlan:
    """Every review point for one clip, and the geometry it was derived from."""

    clip_id: str
    frames_processed: int
    source_fps: float
    frames: tuple[SampleFrame, ...]
    windows: tuple[ContinuityWindow, ...]

    @property
    def duration_seconds(self) -> float:
        if self.source_fps <= 0:
            return 0.0
        return self.frames_processed / self.source_fps

    def frame_rows(self) -> tuple[tuple[float, int], ...]:
        """Annotation-template rows for the sampled stills."""
        return tuple((frame.timestamp_seconds, frame.frame_index) for frame in self.frames)

    def continuity_rows(self) -> tuple[tuple[str, float, float], ...]:
        """Annotation-template rows for the continuity windows."""
        return tuple(
            (window.sequence_id, window.start_seconds, window.end_seconds)
            for window in self.windows
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "frames_processed": self.frames_processed,
            "source_fps": round(self.source_fps, 4),
            "duration_seconds": round(self.duration_seconds, 3),
            "determinism": (
                "sample points are a pure function of frame count and frame rate: the same "
                "clip always yields the same points, so an annotation stays valid across runs."
            ),
            "sample_frames": [frame.to_record() for frame in self.frames],
            "continuity_windows": [window.to_record() for window in self.windows],
        }


def build_sample_plan(
    clip_id: str,
    frames_processed: int,
    source_fps: float,
    sample_frames: int = DEFAULT_SAMPLE_FRAMES,
    continuity_windows: int = DEFAULT_CONTINUITY_WINDOWS,
    window_seconds: float = DEFAULT_CONTINUITY_WINDOW_SECONDS,
) -> ReviewSamplePlan:
    """Compute the deterministic review sample for one clip.

    Short clips are handled by shrinking the plan rather than by producing
    overlapping or out-of-range points: a clip with 4 frames yields at most 4
    stills, and a clip shorter than one window yields one window covering it.

    Args:
        clip_id: The clip these points belong to.
        frames_processed: Frames the run actually processed.
        source_fps: Source frame rate, used to convert indices to timestamps.
        sample_frames: Stills to sample.
        continuity_windows: Windows to sample.
        window_seconds: Length of each window in source seconds.

    Raises:
        ValidationError: for a non-positive frame rate or a negative count,
            either of which would make the sample points meaningless.
    """
    if source_fps <= 0:
        raise ValidationError(
            f"Cannot build a review sample for '{clip_id}': source fps must be positive, "
            f"got {source_fps}."
        )
    if sample_frames < 0 or continuity_windows < 0 or window_seconds <= 0:
        raise ValidationError(
            f"Cannot build a review sample for '{clip_id}': counts must be non-negative and "
            f"window_seconds positive, got {sample_frames}, {continuity_windows}, "
            f"{window_seconds}."
        )
    if frames_processed <= 0:
        return ReviewSamplePlan(
            clip_id=clip_id,
            frames_processed=0,
            source_fps=source_fps,
            frames=(),
            windows=(),
        )

    wanted = min(sample_frames, frames_processed)
    frames: list[SampleFrame] = []
    seen: set[int] = set()
    for index in range(wanted):
        fraction = (index + 0.5) / wanted
        # Frame indices are 1-based, matching the pipeline's own numbering, so a
        # sampled index can be looked up directly in the output streams.
        frame_index = min(frames_processed, max(1, math.floor(fraction * frames_processed) + 1))
        if frame_index in seen:
            continue
        seen.add(frame_index)
        frames.append(
            SampleFrame(
                index=len(frames) + 1,
                frame_index=frame_index,
                timestamp_seconds=(frame_index - 1) / source_fps,
            )
        )

    duration = frames_processed / source_fps
    windows = _build_windows(
        continuity_windows, window_seconds, duration, source_fps, frames_processed
    )

    return ReviewSamplePlan(
        clip_id=clip_id,
        frames_processed=frames_processed,
        source_fps=source_fps,
        frames=tuple(frames),
        windows=tuple(windows),
    )


def _build_windows(
    count: int,
    window_seconds: float,
    duration: float,
    source_fps: float,
    frames_processed: int,
) -> list[ContinuityWindow]:
    if count == 0 or duration <= 0:
        return []

    length = min(window_seconds, duration)
    # Window starts are spread over the room the clip leaves after the window
    # length is reserved, so the last window ends at the clip's end rather than
    # running past it.
    room = max(duration - length, 0.0)
    windows: list[ContinuityWindow] = []
    for index in range(count):
        fraction = 0.0 if count == 1 else index / (count - 1)
        start = room * fraction
        end = min(start + length, duration)
        start_frame = min(frames_processed, max(1, math.floor(start * source_fps) + 1))
        end_frame = min(frames_processed, max(start_frame, math.floor(end * source_fps)))
        window = ContinuityWindow(
            sequence_id=f"w{index + 1:02d}",
            start_seconds=start,
            end_seconds=end,
            start_frame_index=start_frame,
            end_frame_index=end_frame,
        )
        if windows and window.start_frame_index == windows[-1].start_frame_index:
            continue
        windows.append(window)
    return windows


def extract_sample_frames(
    video_path: Path | str, plan: ReviewSamplePlan, output_dir: Path | str
) -> tuple[Path, ...]:
    """Write each planned still out of the rendered tracking video as a JPG.

    Frames are read sequentially and kept when their index is wanted, rather
    than seeking per sample. Seeking a compressed stream lands on the nearest
    keyframe, which would silently write a different frame than the plan names
    and the annotation would describe footage nobody reviewed.

    Returns:
        The written paths, in plan order. Empty when the plan has no frames.
    """
    if not plan.frames:
        return ()

    source = Path(video_path)
    if not source.exists():
        raise ValidationError(f"Rendered video not found for sampling: {source}")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    wanted = {frame.frame_index: frame for frame in plan.frames}
    written: dict[int, Path] = {}

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValidationError(f"Could not open rendered video for sampling: {source}")
    try:
        frame_index = 0
        remaining = len(wanted)
        while remaining > 0:
            ok, image = capture.read()
            if not ok or image is None:
                break
            frame_index += 1
            sample = wanted.get(frame_index)
            if sample is None:
                continue
            path = target / sample.filename
            if not cv2.imwrite(str(path), image):
                raise ValidationError(f"Could not write review frame {path}")
            written[sample.index] = path
            remaining -= 1
    finally:
        capture.release()

    if len(written) != len(plan.frames):
        missing = sorted(frame.frame_index for frame in plan.frames if frame.index not in written)
        logger.warning(
            "Rendered video %s ended before %d planned sample frame(s): %s",
            source,
            len(missing),
            missing,
        )
    return tuple(written[index] for index in sorted(written))


def write_contact_sheet(
    frame_paths: Sequence[Path], output_path: Path | str, plan: ReviewSamplePlan
) -> Path | None:
    """Tile the sampled stills into one reviewable image.

    A convenience for a first pass, never a replacement for the full-resolution
    stills: a tile is too small to count people in a dense scene, and the
    annotation workflow points at the individual JPGs.

    Returns:
        The contact sheet's path, or ``None`` when there was nothing to tile.
    """
    if not frame_paths:
        return None

    images: list[np.ndarray] = []
    labels: list[str] = []
    by_index = {frame.filename: frame for frame in plan.frames}
    for path in frame_paths:
        image = cv2.imread(str(path))
        if image is None:
            logger.warning("Skipping unreadable review frame %s", path)
            continue
        images.append(image)
        sample = by_index.get(path.name)
        labels.append(
            f"{plan.clip_id}  t={sample.timestamp_seconds:.2f}s  f={sample.frame_index}"
            if sample is not None
            else path.name
        )
    if not images:
        return None

    height, width = images[0].shape[:2]
    scale = _CONTACT_SHEET_TILE_WIDTH / width
    tile_width = _CONTACT_SHEET_TILE_WIDTH
    tile_height = max(1, round(height * scale))

    columns = min(_CONTACT_SHEET_COLUMNS, len(images))
    rows = math.ceil(len(images) / columns)
    cell_height = tile_height + _CONTACT_SHEET_LABEL_HEIGHT
    sheet_width = columns * tile_width + (columns + 1) * _CONTACT_SHEET_MARGIN
    sheet_height = rows * cell_height + (rows + 1) * _CONTACT_SHEET_MARGIN
    sheet = np.full((sheet_height, sheet_width, 3), 24, dtype=np.uint8)

    for position, (image, label) in enumerate(zip(images, labels, strict=True)):
        row, column = divmod(position, columns)
        x = _CONTACT_SHEET_MARGIN + column * (tile_width + _CONTACT_SHEET_MARGIN)
        y = _CONTACT_SHEET_MARGIN + row * (cell_height + _CONTACT_SHEET_MARGIN)
        tile = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        sheet[y : y + tile_height, x : x + tile_width] = tile
        cv2.putText(
            sheet,
            label,
            (x + 4, y + tile_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), sheet):
        raise ValidationError(f"Could not write contact sheet {destination}")
    return destination

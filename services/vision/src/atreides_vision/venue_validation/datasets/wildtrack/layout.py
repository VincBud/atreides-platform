"""WILDTRACK's on-disk layout: locating and validating one dataset root.

The dataset root itself is never hardcoded anywhere in this package — callers
(tests, an optional local diagnostic script, or a future validation runner)
supply it as a :class:`pathlib.Path`. This module only knows the *shape* the
real, locally-extracted WILDTRACK release has:

.. code-block:: text

    <root>/
      Image_subsets/
        C1/00000000.png  C1/00000005.png  ...
        C2/...  ...  C7/...
      annotations_positions/
        00000000.json  00000005.json  ...
      calibrations/
        intrinsic_zero/intr_CVLab1.xml  ...  intr_IDIAP3.xml
        extrinsic/extr_CVLab1.xml  ...  extr_IDIAP3.xml

confirmed by inspecting the locally extracted dataset directly, not assumed
from an online description. ``Image_subsets/C<n>`` holds every synchronized
extraction frame (401 of them per camera); ``annotations_positions`` holds
one JSON file per *annotated* frame, a strict subset (400 of the 401,
stepping by :data:`~atreides_vision.venue_validation.datasets.wildtrack.time.FRAME_STEP`)
— the trailing extracted frame simply has no annotation file. This module
checks that containment structurally (every annotation frame id is also an
image frame id) rather than hardcoding the counts 401 and 400, so it keeps
working if a differently-sized WILDTRACK-shaped release is ever pointed at
it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from atreides_vision.errors import WildtrackDatasetError
from atreides_vision.venue_validation.datasets.wildtrack.cameras import (
    CAMERA_IDS,
    calibration_name_for,
)
from atreides_vision.venue_validation.datasets.wildtrack.time import WildtrackFrameId

__all__ = [
    "IMAGE_SUFFIX",
    "WildtrackDatasetLayout",
]

#: The image file suffix used throughout the released ``Image_subsets`` trees.
IMAGE_SUFFIX = ".png"

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True, slots=True)
class WildtrackDatasetLayout:
    """A validated view of one WILDTRACK dataset root's directory structure.

    Construction (:meth:`load`) is the only place this package touches the
    filesystem to discover *what exists*; every other module in this package
    is handed already-known paths or already-parsed content.
    """

    root: Path
    image_frame_ids: tuple[WildtrackFrameId, ...]
    annotation_frame_ids: tuple[WildtrackFrameId, ...]
    image_width: int
    image_height: int

    @property
    def image_subsets_dir(self) -> Path:
        return self.root / "Image_subsets"

    @property
    def annotations_dir(self) -> Path:
        return self.root / "annotations_positions"

    @property
    def intrinsic_dir(self) -> Path:
        return self.root / "calibrations" / "intrinsic_zero"

    @property
    def extrinsic_dir(self) -> Path:
        return self.root / "calibrations" / "extrinsic"

    def image_path(self, camera_id: str, frame_id: WildtrackFrameId) -> Path:
        return self.image_subsets_dir / camera_id / f"{frame_id.filename_stem}{IMAGE_SUFFIX}"

    def annotation_path(self, frame_id: WildtrackFrameId) -> Path:
        return self.annotations_dir / f"{frame_id.filename_stem}.json"

    def intrinsic_path(self, camera_id: str) -> Path:
        return self.intrinsic_dir / f"intr_{calibration_name_for(camera_id)}.xml"

    def extrinsic_path(self, camera_id: str) -> Path:
        return self.extrinsic_dir / f"extr_{calibration_name_for(camera_id)}.xml"

    @classmethod
    def load(cls, root: Path) -> WildtrackDatasetLayout:
        """Validate ``root`` against the expected WILDTRACK layout.

        Raises:
            WildtrackDatasetError: if the root, its camera subdirectories, its
                annotation directory or its calibration files are missing,
                empty, mismatched across cameras, or contain an annotated
                frame with no corresponding image frame.
        """
        if not root.is_dir():
            raise WildtrackDatasetError(f"WILDTRACK dataset root does not exist: {root}")

        image_subsets_dir = root / "Image_subsets"
        if not image_subsets_dir.is_dir():
            raise WildtrackDatasetError(
                f"WILDTRACK dataset root is missing 'Image_subsets': {root}"
            )

        per_camera_frame_ids: dict[str, frozenset[int]] = {}
        for camera_id in CAMERA_IDS:
            camera_dir = image_subsets_dir / camera_id
            if not camera_dir.is_dir():
                raise WildtrackDatasetError(
                    f"WILDTRACK dataset root is missing camera directory "
                    f"'Image_subsets/{camera_id}': {root}"
                )
            frame_ids = _read_frame_ids(camera_dir, suffix=IMAGE_SUFFIX)
            if not frame_ids:
                raise WildtrackDatasetError(
                    f"WILDTRACK camera directory has no images: {camera_dir}"
                )
            per_camera_frame_ids[camera_id] = frame_ids

        reference_camera = CAMERA_IDS[0]
        reference_frame_ids = per_camera_frame_ids[reference_camera]
        for camera_id, frame_ids in per_camera_frame_ids.items():
            if frame_ids != reference_frame_ids:
                raise WildtrackDatasetError(
                    "WILDTRACK cameras are not synchronized on disk: "
                    f"{camera_id} has {len(frame_ids)} image frames but "
                    f"{reference_camera} has {len(reference_frame_ids)}"
                )
        image_frame_ids = tuple(WildtrackFrameId(value) for value in sorted(reference_frame_ids))

        annotations_dir = root / "annotations_positions"
        if not annotations_dir.is_dir():
            raise WildtrackDatasetError(
                f"WILDTRACK dataset root is missing 'annotations_positions': {root}"
            )
        annotation_ids = _read_frame_ids(annotations_dir, suffix=".json")
        if not annotation_ids:
            raise WildtrackDatasetError(
                f"WILDTRACK annotation directory has no annotation files: {annotations_dir}"
            )
        unmatched = annotation_ids - reference_frame_ids
        if unmatched:
            raise WildtrackDatasetError(
                "WILDTRACK has annotation files with no corresponding image frame: "
                f"{sorted(unmatched)}"
            )
        annotation_frame_ids = tuple(WildtrackFrameId(value) for value in sorted(annotation_ids))

        for camera_id in CAMERA_IDS:
            intr_path = (
                annotations_dir.parent
                / "calibrations"
                / "intrinsic_zero"
                / (f"intr_{calibration_name_for(camera_id)}.xml")
            )
            extr_path = (
                annotations_dir.parent
                / "calibrations"
                / "extrinsic"
                / (f"extr_{calibration_name_for(camera_id)}.xml")
            )
            if not intr_path.is_file():
                raise WildtrackDatasetError(f"Missing WILDTRACK intrinsic calibration: {intr_path}")
            if not extr_path.is_file():
                raise WildtrackDatasetError(f"Missing WILDTRACK extrinsic calibration: {extr_path}")

        first_image = (
            image_subsets_dir
            / reference_camera
            / (f"{image_frame_ids[0].filename_stem}{IMAGE_SUFFIX}")
        )
        width, height = _read_png_dimensions(first_image)

        return cls(
            root=root,
            image_frame_ids=image_frame_ids,
            annotation_frame_ids=annotation_frame_ids,
            image_width=width,
            image_height=height,
        )


def _read_frame_ids(directory: Path, *, suffix: str) -> frozenset[int]:
    frame_ids: set[int] = set()
    for entry in directory.iterdir():
        if entry.is_dir() or entry.suffix != suffix:
            continue
        stem = entry.stem
        if not stem.isdigit():
            raise WildtrackDatasetError(
                f"WILDTRACK file name does not encode an integer frame id: {entry}"
            )
        frame_ids.add(int(stem))
    return frozenset(frame_ids)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Read width/height from a PNG's ``IHDR`` chunk without a new dependency.

    Deliberately minimal: this package needs image extent only to describe a
    :class:`~atreides_vision.calibration.models.ReferenceFrame`, never pixel
    content, so a full PNG decoder would be unjustified scope.
    """
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != _PNG_SIGNATURE:
        raise WildtrackDatasetError(f"Not a valid PNG file (bad signature): {path}")
    if header[12:16] != b"IHDR":
        raise WildtrackDatasetError(f"Not a valid PNG file (missing IHDR): {path}")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)

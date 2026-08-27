"""Tiny synthetic WILDTRACK-shaped dataset roots, for adapter tests.

Reproduces the real dataset's directory layout, filename convention,
annotation JSON schema and OpenCV ``FileStorage`` calibration XML shape at a
scale suitable for a unit test — a handful of frames and a 2x2 pixel image —
never the real 7.1 GB release. Nothing here reads or references the real
dataset's on-disk location.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from atreides_vision.venue_validation.datasets.wildtrack.cameras import (
    CALIBRATION_NAME_BY_CAMERA_ID,
    CAMERA_IDS,
)
from atreides_vision.venue_validation.datasets.wildtrack.layout import (
    IMAGE_SUFFIX,
    WildtrackDatasetLayout,
)

__all__ = [
    "DEFAULT_CAMERA_MATRIX",
    "DEFAULT_DISTORTION",
    "DEFAULT_RVEC",
    "DEFAULT_TVEC_CM",
    "build_layout",
    "make_view",
    "not_visible_view",
    "png_bytes",
    "visible_view",
    "write_annotation_json",
    "write_extrinsic_xml",
    "write_intrinsic_xml",
    "write_minimal_dataset",
]

#: A simple, well-conditioned pinhole intrinsic: unit-scale focal length,
#: principal point at the centre of a 100x80 image, square pixels, no skew.
DEFAULT_CAMERA_MATRIX = (500.0, 0.0, 50.0, 0.0, 500.0, 40.0, 0.0, 0.0, 1.0)
DEFAULT_DISTORTION = (0.0, 0.0, 0.0, 0.0, 0.0)
#: Identity rotation (camera axis-aligned with the world) by default.
DEFAULT_RVEC = (0.0, 0.0, 0.0)
#: 5 m back and 3 m up from the world origin, in centimetres (WILDTRACK's own unit).
DEFAULT_TVEC_CM = (0.0, -300.0, 500.0)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_bytes(width: int, height: int, *, valid_signature: bool = True) -> bytes:
    """The first 24 bytes of a PNG file: signature plus an ``IHDR`` chunk header.

    ``layout._read_png_dimensions`` reads only these 24 bytes, so nothing
    beyond them (compressed pixel data, CRCs, other chunks) is needed for a
    test fixture.
    """
    signature = _PNG_SIGNATURE if valid_signature else b"NOTAPNG!"
    ihdr_header = struct.pack(">I4s", 13, b"IHDR")
    dimensions = struct.pack(">II", width, height)
    return signature + ihdr_header + dimensions


def write_intrinsic_xml(
    path: Path,
    *,
    camera_matrix: tuple[float, ...] = DEFAULT_CAMERA_MATRIX,
    distortion: tuple[float, ...] = DEFAULT_DISTORTION,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix_text = " ".join(str(v) for v in camera_matrix)
    distortion_text = " ".join(str(v) for v in distortion)
    path.write_text(
        '<?xml version="1.0"?>\n'
        "<opencv_storage>\n"
        "<camera_matrix>\n"
        f"<data>\n{matrix_text}\n</data>\n"
        "</camera_matrix>\n"
        "<distortion_coefficients>\n"
        f"<data>\n{distortion_text}\n</data>\n"
        "</distortion_coefficients>\n"
        "</opencv_storage>\n",
        encoding="utf-8",
    )


def write_extrinsic_xml(
    path: Path,
    *,
    rvec: tuple[float, float, float] = DEFAULT_RVEC,
    tvec_cm: tuple[float, float, float] = DEFAULT_TVEC_CM,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rvec_text = " ".join(str(v) for v in rvec)
    tvec_text = " ".join(str(v) for v in tvec_cm)
    path.write_text(
        '<?xml version="1.0"?>\n'
        "<opencv_storage>\n"
        f"<rvec>{rvec_text}</rvec>\n"
        f"<tvec>{tvec_text}</tvec>\n"
        "</opencv_storage>\n",
        encoding="utf-8",
    )


def visible_view(
    view_num: int, *, xmin: float, ymin: float, xmax: float, ymax: float
) -> dict[str, Any]:
    return {"viewNum": view_num, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def not_visible_view(view_num: int) -> dict[str, Any]:
    return {"viewNum": view_num, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1}


def make_view(view_num: int, bbox: tuple[float, float, float, float] | None) -> dict[str, Any]:
    if bbox is None:
        return not_visible_view(view_num)
    xmin, ymin, xmax, ymax = bbox
    return visible_view(view_num, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


def write_annotation_json(path: Path, persons: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(persons), encoding="utf-8")


def write_minimal_dataset(
    root: Path,
    *,
    image_frame_values: tuple[int, ...] = (0, 5, 10),
    annotation_frame_values: tuple[int, ...] = (0, 5),
    image_width: int = 100,
    image_height: int = 80,
    camera_ids: tuple[str, ...] = CAMERA_IDS,
    person_records_by_frame: dict[int, list[dict[str, Any]]] | None = None,
) -> None:
    """Write a complete, valid, tiny WILDTRACK-shaped dataset root.

    By default every annotated person is visible in no camera (an empty
    frame), which is enough to validate layout and calibration. Pass
    ``person_records_by_frame`` (already-shaped JSON records) to exercise
    annotation parsing.
    """
    image_bytes = png_bytes(image_width, image_height)
    for camera_id in camera_ids:
        camera_dir = root / "Image_subsets" / camera_id
        camera_dir.mkdir(parents=True, exist_ok=True)
        for value in image_frame_values:
            (camera_dir / f"{value:08d}{IMAGE_SUFFIX}").write_bytes(image_bytes)

    annotations_dir = root / "annotations_positions"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    records_by_frame = person_records_by_frame or {}
    for value in annotation_frame_values:
        records = records_by_frame.get(value, [])
        write_annotation_json(annotations_dir / f"{value:08d}.json", records)

    for camera_id in camera_ids:
        name = CALIBRATION_NAME_BY_CAMERA_ID[camera_id]
        write_intrinsic_xml(root / "calibrations" / "intrinsic_zero" / f"intr_{name}.xml")
        write_extrinsic_xml(root / "calibrations" / "extrinsic" / f"extr_{name}.xml")


def build_layout(
    root: Path,
    *,
    image_frame_values: tuple[int, ...] = (0, 5, 10),
    annotation_frame_values: tuple[int, ...] = (0, 5),
    image_width: int = 100,
    image_height: int = 80,
    person_records_by_frame: dict[int, list[dict[str, Any]]] | None = None,
) -> WildtrackDatasetLayout:
    write_minimal_dataset(
        root,
        image_frame_values=image_frame_values,
        annotation_frame_values=annotation_frame_values,
        image_width=image_width,
        image_height=image_height,
        person_records_by_frame=person_records_by_frame,
    )
    return WildtrackDatasetLayout.load(root)

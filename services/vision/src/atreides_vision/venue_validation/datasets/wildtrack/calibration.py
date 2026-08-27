"""WILDTRACK's OpenCV-pinhole calibration, and its conversion into Atreides'.

Each camera ships two small XML files in OpenCV's ``FileStorage`` format:

``calibrations/intrinsic_zero/intr_<name>.xml``
    A 3x3 ``camera_matrix`` and 5 ``distortion_coefficients``. This package
    deliberately reads the ``intrinsic_zero`` set (not ``intrinsic_original``)
    because every one of its five distortion coefficients is exactly ``0`` —
    confirmed by reading all seven shipped files directly — so projecting
    through it needs no distortion model at all.

``calibrations/extrinsic/extr_<name>.xml``
    A Rodrigues rotation vector ``rvec`` (radians) and a translation vector
    ``tvec``, together the standard OpenCV extrinsic convention
    ``X_cam = R(rvec) @ X_world + tvec``.

Units: empirically confirmed, from this dataset's own files, not from
external documentation. WILDTRACK's ``positionID`` -> ground-plane formula
(``README.txt``) and Atreides' own :class:`~atreides_vision.calibration.models.VenuePoint`
are both in **metres**. Projecting a ``positionID``-derived point through the
raw ``tvec`` as printed does not land anywhere near the matching bounding box
in ``rectangles.pom``; dividing ``tvec`` by 100 does, closely and
consistently, on two independently-oriented cameras (``CVLab1`` and
``IDIAP1``) checked directly against ``rectangles.pom``. ``tvec`` is
therefore in **centimetres** and is scaled by :data:`TVEC_CM_TO_M` before use;
no axis flip or matrix inversion is needed beyond that.

This module is the only place in the whole codebase that touches a WILDTRACK
extrinsic/intrinsic matrix. It hands off to
:func:`atreides_vision.calibration.fitting.fit_calibration` for the actual
homography solve — this module's job is only to produce
:class:`~atreides_vision.calibration.models.PointCorrespondence` pairs, never
to re-implement calibration fitting.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from atreides_vision.calibration.fitting import CalibrationFit, fit_calibration
from atreides_vision.calibration.models import (
    GroundPlaneCalibration,
    PointCorrespondence,
    ReferenceFrame,
    VenuePoint,
)
from atreides_vision.errors import WildtrackDatasetError
from atreides_vision.models import Point
from atreides_vision.venue_validation.datasets.wildtrack.annotations import (
    GRID_HEIGHT,
    GRID_WIDTH,
    ground_position_from_position_id,
)
from atreides_vision.venue_validation.datasets.wildtrack.cameras import CAMERA_IDS
from atreides_vision.venue_validation.datasets.wildtrack.layout import WildtrackDatasetLayout

__all__ = [
    "TVEC_CM_TO_M",
    "WildtrackCameraCalibration",
    "WildtrackExtrinsic",
    "WildtrackIntrinsic",
    "load_camera_calibration",
    "parse_extrinsic_xml",
    "parse_intrinsic_xml",
    "project_world_point",
    "rotation_matrix_from_rvec",
    "to_ground_plane_calibration",
]

#: WILDTRACK extrinsic ``tvec`` is in centimetres; Atreides' ground plane and
#: WILDTRACK's own ``positionID`` formula are both in metres. See module
#: docstring for how this was established.
TVEC_CM_TO_M = 0.01

#: WILDTRACK's own ``positionID`` grid describes one fixed physical plaza — the
#: same venue-local frame for every one of its seven cameras. This is not a
#: general-purpose default; it names one specific published dataset's own
#: coordinate frame, which is exactly what :attr:`GroundPlaneCalibration.venue_id`
#: exists to record.
VENUE_ID = "wildtrack"

#: Deterministic ground-point sampling grid for building calibration
#: correspondences: every this-many-th cell along each grid axis, always in
#: the same personID-independent order, so a rebuild reproduces byte-identical
#: correspondences.
_SAMPLE_STRIDE = 24


@dataclass(frozen=True, slots=True)
class WildtrackIntrinsic:
    """A camera's intrinsic matrix and (zero, for ``intrinsic_zero``) distortion."""

    camera_matrix: tuple[float, ...]
    distortion_coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.camera_matrix) != 9:
            raise WildtrackDatasetError(
                f"WILDTRACK camera_matrix must have 9 entries, got {len(self.camera_matrix)}"
            )
        if not all(math.isfinite(v) for v in self.camera_matrix):
            raise WildtrackDatasetError("WILDTRACK camera_matrix has a non-finite entry")
        if not all(math.isfinite(v) for v in self.distortion_coefficients):
            raise WildtrackDatasetError("WILDTRACK distortion_coefficients has a non-finite entry")
        if self.camera_matrix[0] <= 0 or self.camera_matrix[4] <= 0:
            raise WildtrackDatasetError(
                f"WILDTRACK camera_matrix has a non-positive focal length: {self.camera_matrix!r}"
            )

    @property
    def focal_length_x(self) -> float:
        return self.camera_matrix[0]

    @property
    def focal_length_y(self) -> float:
        return self.camera_matrix[4]

    @property
    def principal_point(self) -> tuple[float, float]:
        return (self.camera_matrix[2], self.camera_matrix[5])


@dataclass(frozen=True, slots=True)
class WildtrackExtrinsic:
    """A camera's Rodrigues rotation vector and raw (centimetre) translation."""

    rvec: tuple[float, float, float]
    tvec_cm: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (*self.rvec, *self.tvec_cm)):
            raise WildtrackDatasetError("WILDTRACK extrinsic has a non-finite entry")

    @property
    def tvec_m(self) -> tuple[float, float, float]:
        x_cm, y_cm, z_cm = self.tvec_cm
        return (x_cm * TVEC_CM_TO_M, y_cm * TVEC_CM_TO_M, z_cm * TVEC_CM_TO_M)


@dataclass(frozen=True, slots=True)
class WildtrackCameraCalibration:
    """One camera's full raw WILDTRACK calibration."""

    camera_id: str
    intrinsic: WildtrackIntrinsic
    extrinsic: WildtrackExtrinsic


def parse_intrinsic_xml(path: Path) -> WildtrackIntrinsic:
    """Parse an ``intr_<name>.xml`` OpenCV ``FileStorage`` document.

    Raises:
        WildtrackDatasetError: if the file cannot be parsed or the expected
            ``camera_matrix``/``distortion_coefficients`` nodes are missing or
            the wrong shape.
    """
    root = _parse_xml(path)
    camera_matrix = _read_matrix(root, "camera_matrix", expected_size=9, path=path)
    distortion = _read_matrix(root, "distortion_coefficients", expected_size=5, path=path)
    return WildtrackIntrinsic(camera_matrix=camera_matrix, distortion_coefficients=distortion)


def parse_extrinsic_xml(path: Path) -> WildtrackExtrinsic:
    """Parse an ``extr_<name>.xml`` OpenCV ``FileStorage`` document.

    Raises:
        WildtrackDatasetError: if the file cannot be parsed or the expected
            ``rvec``/``tvec`` nodes are missing or the wrong shape.
    """
    root = _parse_xml(path)
    rvec = _read_vector(root, "rvec", expected_size=3, path=path)
    tvec = _read_vector(root, "tvec", expected_size=3, path=path)
    return WildtrackExtrinsic(rvec=(rvec[0], rvec[1], rvec[2]), tvec_cm=(tvec[0], tvec[1], tvec[2]))


def load_camera_calibration(
    layout: WildtrackDatasetLayout, camera_id: str
) -> WildtrackCameraCalibration:
    """Load and parse both XML files for ``camera_id`` from ``layout``.

    Raises:
        WildtrackDatasetError: if ``camera_id`` is unknown or either file
            cannot be parsed.
    """
    if camera_id not in CAMERA_IDS:
        raise WildtrackDatasetError(f"Unknown WILDTRACK camera id {camera_id!r}")
    intrinsic = parse_intrinsic_xml(layout.intrinsic_path(camera_id))
    extrinsic = parse_extrinsic_xml(layout.extrinsic_path(camera_id))
    return WildtrackCameraCalibration(camera_id=camera_id, intrinsic=intrinsic, extrinsic=extrinsic)


def rotation_matrix_from_rvec(rvec: tuple[float, float, float]) -> np.ndarray:
    """Rodrigues' rotation formula: axis-angle vector -> 3x3 rotation matrix."""
    theta = math.sqrt(sum(v * v for v in rvec))
    if theta < 1e-12:
        return np.eye(3)
    axis = np.array(rvec, dtype=np.float64) / theta
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.asarray(np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k))


def project_world_point(
    calibration: WildtrackCameraCalibration, world_point_m: tuple[float, float, float]
) -> tuple[float, float] | None:
    """Project a ground-plane point (metres) into this camera's image pixels.

    Returns ``None`` when the point is behind the camera, where a pinhole
    model has no finite image position rather than a physically meaningless
    one.
    """
    rotation = rotation_matrix_from_rvec(calibration.extrinsic.rvec)
    translation = np.array(calibration.extrinsic.tvec_m, dtype=np.float64)
    world = np.array(world_point_m, dtype=np.float64)
    camera_point = rotation @ world + translation
    if camera_point[2] <= 1e-6:
        return None
    x_norm = camera_point[0] / camera_point[2]
    y_norm = camera_point[1] / camera_point[2]
    intrinsic = calibration.intrinsic
    u = intrinsic.focal_length_x * x_norm + intrinsic.principal_point[0]
    v = intrinsic.focal_length_y * y_norm + intrinsic.principal_point[1]
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return (float(u), float(v))


def to_ground_plane_calibration(
    calibration: WildtrackCameraCalibration, layout: WildtrackDatasetLayout
) -> CalibrationFit:
    """Convert one WILDTRACK camera's calibration into an Atreides ``CalibrationFit``.

    Samples a deterministic grid of ``positionID``-derived ground points,
    projects each through ``calibration``'s pinhole model, keeps only points
    that land in front of the camera and inside the released image bounds,
    and hands the resulting image/ground correspondences to
    :func:`atreides_vision.calibration.fitting.fit_calibration` — the same
    homography solver every other Atreides calibration path uses.

    Raises:
        WildtrackDatasetError: if too few sample points project inside the
            image to fit a calibration.
        CalibrationError: propagated unchanged if
            :func:`~atreides_vision.calibration.fitting.fit_calibration`
            itself rejects the resulting correspondences (e.g. as too
            collinear or too poor a fit).
    """
    correspondences: list[PointCorrespondence] = []
    for row in range(0, GRID_HEIGHT, _SAMPLE_STRIDE):
        for col in range(0, GRID_WIDTH, _SAMPLE_STRIDE):
            position_id = row * GRID_WIDTH + col
            x_m, y_m = ground_position_from_position_id(position_id)
            pixel = project_world_point(calibration, (x_m, y_m, 0.0))
            if pixel is None:
                continue
            u, v = pixel
            if not (0.0 <= u < layout.image_width and 0.0 <= v < layout.image_height):
                continue
            correspondences.append(
                PointCorrespondence(image=Point(u, v), venue=VenuePoint(x=x_m, y=y_m))
            )

    if len(correspondences) < 4:
        raise WildtrackDatasetError(
            f"Only {len(correspondences)} WILDTRACK ground samples project inside camera "
            f"{calibration.camera_id}'s image bounds; need at least 4 to fit a calibration"
        )

    reference_frame = ReferenceFrame(width=layout.image_width, height=layout.image_height)
    ground_plane_calibration = GroundPlaneCalibration(
        camera_id=calibration.camera_id,
        venue_id=VENUE_ID,
        reference_frame=reference_frame,
        correspondences=tuple(correspondences),
        description=(
            "Derived from WILDTRACK's own OpenCV pinhole calibration by projecting a "
            "deterministic sample of positionID-derived ground points; see "
            "atreides_vision.venue_validation.datasets.wildtrack.calibration."
        ),
    )
    return fit_calibration(ground_plane_calibration)


def _parse_xml(path: Path) -> ElementTree.Element:
    if not path.is_file():
        raise WildtrackDatasetError(f"Missing WILDTRACK calibration file: {path}")
    try:
        return ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise WildtrackDatasetError(f"Malformed WILDTRACK calibration XML: {path}") from exc


def _read_matrix(
    root: ElementTree.Element, tag: str, *, expected_size: int, path: Path
) -> tuple[float, ...]:
    node = root.find(tag)
    if node is None:
        raise WildtrackDatasetError(f"WILDTRACK calibration file is missing <{tag}>: {path}")
    data_node = node.find("data")
    if data_node is None or data_node.text is None:
        raise WildtrackDatasetError(f"WILDTRACK calibration file's <{tag}> has no <data>: {path}")
    values = _parse_floats(data_node.text, path=path, tag=tag)
    if len(values) != expected_size:
        raise WildtrackDatasetError(
            f"WILDTRACK calibration file's <{tag}> has {len(values)} values, "
            f"expected {expected_size}: {path}"
        )
    return values


def _read_vector(
    root: ElementTree.Element, tag: str, *, expected_size: int, path: Path
) -> tuple[float, ...]:
    node = root.find(tag)
    if node is None or node.text is None:
        raise WildtrackDatasetError(f"WILDTRACK calibration file is missing <{tag}>: {path}")
    values = _parse_floats(node.text, path=path, tag=tag)
    if len(values) != expected_size:
        raise WildtrackDatasetError(
            f"WILDTRACK calibration file's <{tag}> has {len(values)} values, "
            f"expected {expected_size}: {path}"
        )
    return values


def _parse_floats(text: str, *, path: Path, tag: str) -> tuple[float, ...]:
    try:
        return tuple(float(token) for token in text.split())
    except ValueError as exc:
        raise WildtrackDatasetError(
            f"WILDTRACK calibration file's <{tag}> is not all numeric: {path}"
        ) from exc

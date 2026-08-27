"""WILDTRACK's fixed seven-camera layout: naming and canonical ordering.

WILDTRACK is not a topology-agnostic venue: it is one specific, published
recording of seven synchronized GoPro cameras on the plaza in front of the
ETH Zurich main building. This module names that fixed set once, in one
place, so nothing downstream in this adapter re-derives or guesses it.

Camera identity and the calibration-file mapping
--------------------------------------------------
The dataset exposes two different names for the same camera:

* ``C1``..``C7`` — the "view" identity used by ``annotations_positions``
  (``views[i].viewNum``, 0-indexed) and by ``rectangles.pom`` (``camIdx``,
  also 0-indexed), and the suffix of each ``Image_subsets/C<n>`` directory.
- ``CVLab1``..``CVLab4``, ``IDIAP1``..``IDIAP3`` — the stem of each
  calibration file (``calibrations/extrinsic/extr_<name>.xml`` and
  ``calibrations/intrinsic_zero/intr_<name>.xml``).

The mapping between them is not alphabetical and not guessed: it is fixed by
the calibration directory contents themselves (exactly four ``CVLab*`` files
and three ``IDIAP*`` files, i.e. one per camera, in the numeric order already
present in their own filenames) and corroborated by WILDTRACK's own paper,
which lists cameras 1-4 as GoPro Hero 3 units and cameras 5-7 as GoPro Hero 4
units — matching the CVLab/IDIAP split exactly. ``C1`` is ``CVLab1``, ``C2``
is ``CVLab2``, and so on through ``C7`` is ``IDIAP3``.
"""

from __future__ import annotations

from atreides_vision.errors import WildtrackDatasetError

__all__ = [
    "CALIBRATION_NAME_BY_CAMERA_ID",
    "CAMERA_COUNT",
    "CAMERA_IDS",
    "calibration_name_for",
    "camera_id_for_view_num",
    "view_num_for_camera_id",
]

#: Canonical camera ids, in ``viewNum`` / ``camIdx`` order. This is the one
#: ordering every other module in this package sorts by, so two runs over the
#: same content produce identical output regardless of filesystem enumeration
#: order.
CAMERA_IDS: tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")

CAMERA_COUNT = len(CAMERA_IDS)

#: ``C<n>`` -> calibration file stem, fixed by the dataset's own calibration
#: directory contents and corroborated by the WILDTRACK paper's camera
#: hardware listing. See the module docstring.
CALIBRATION_NAME_BY_CAMERA_ID: dict[str, str] = {
    "C1": "CVLab1",
    "C2": "CVLab2",
    "C3": "CVLab3",
    "C4": "CVLab4",
    "C5": "IDIAP1",
    "C6": "IDIAP2",
    "C7": "IDIAP3",
}


def calibration_name_for(camera_id: str) -> str:
    """The calibration file stem for ``camera_id`` (e.g. ``"C5" -> "IDIAP1"``).

    Raises:
        WildtrackDatasetError: if ``camera_id`` is not one of :data:`CAMERA_IDS`.
    """
    try:
        return CALIBRATION_NAME_BY_CAMERA_ID[camera_id]
    except KeyError as exc:
        raise WildtrackDatasetError(
            f"Unknown WILDTRACK camera id {camera_id!r}; expected one of {CAMERA_IDS}"
        ) from exc


def camera_id_for_view_num(view_num: int) -> str:
    """The canonical camera id for a 0-indexed ``viewNum``/``camIdx``.

    Raises:
        WildtrackDatasetError: if ``view_num`` is outside ``0..6``.
    """
    if not (0 <= view_num < CAMERA_COUNT):
        raise WildtrackDatasetError(
            f"WILDTRACK viewNum must be in 0..{CAMERA_COUNT - 1}, got {view_num!r}"
        )
    return CAMERA_IDS[view_num]


def view_num_for_camera_id(camera_id: str) -> int:
    """The 0-indexed ``viewNum``/``camIdx`` for a canonical camera id.

    Raises:
        WildtrackDatasetError: if ``camera_id`` is not one of :data:`CAMERA_IDS`.
    """
    try:
        return CAMERA_IDS.index(camera_id)
    except ValueError as exc:
        raise WildtrackDatasetError(
            f"Unknown WILDTRACK camera id {camera_id!r}; expected one of {CAMERA_IDS}"
        ) from exc

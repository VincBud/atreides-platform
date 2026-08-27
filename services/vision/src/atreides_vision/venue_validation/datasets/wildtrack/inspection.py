"""A deterministic, non-identifying summary of one WILDTRACK dataset root.

Meant for a human or a validation report to sanity-check "is this the
dataset I think it is" without printing an absolute filesystem path, a
wall-clock timestamp, or any accuracy figure — this package never scores
Atreides' own output, so nothing here can look like a benchmark result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atreides_vision.venue_validation.datasets.wildtrack.annotations import (
    load_frame_annotations,
)
from atreides_vision.venue_validation.datasets.wildtrack.cameras import CAMERA_IDS
from atreides_vision.venue_validation.datasets.wildtrack.layout import WildtrackDatasetLayout
from atreides_vision.venue_validation.datasets.wildtrack.time import FRAME_RATE_HZ, FRAME_STEP

__all__ = ["WildtrackDatasetInspection", "inspect_dataset"]

#: Bumped whenever the shape of :meth:`WildtrackDatasetInspection.to_record`
#: changes. Purely descriptive of this adapter's own output, unrelated to any
#: runtime JSONL contract's ``schema_version``.
ADAPTER_SCHEMA_VERSION = 1

_DATASET_NAME = "WILDTRACK"


@dataclass(frozen=True, slots=True)
class WildtrackDatasetInspection:
    """A dataset root's shape, with no path, timestamp or accuracy figure in it."""

    camera_ids: tuple[str, ...]
    image_frame_count: int
    annotated_frame_count: int
    first_annotated_frame_value: int
    last_annotated_frame_value: int
    annotation_cadence_hz: float
    calibrated_camera_count: int
    total_person_observation_count: int
    distinct_person_id_count: int

    def to_record(self) -> dict[str, Any]:
        return {
            "dataset_name": _DATASET_NAME,
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "camera_ids": list(self.camera_ids),
            "image_frame_count": self.image_frame_count,
            "annotated_frame_count": self.annotated_frame_count,
            "first_annotated_frame_value": self.first_annotated_frame_value,
            "last_annotated_frame_value": self.last_annotated_frame_value,
            "annotation_cadence_hz": self.annotation_cadence_hz,
            "calibrated_camera_count": self.calibrated_camera_count,
            "total_person_observation_count": self.total_person_observation_count,
            "distinct_person_id_count": self.distinct_person_id_count,
        }


def inspect_dataset(layout: WildtrackDatasetLayout) -> WildtrackDatasetInspection:
    """Summarize ``layout``, reading every annotation file exactly once.

    Deterministic: iterates ``layout.annotation_frame_ids`` (already
    canonically sorted) and, within each frame, the already
    personID-sorted :attr:`~...annotations.WildtrackFrameAnnotations.persons`
    — never a filesystem or dict enumeration order.
    """
    distinct_person_ids: set[int] = set()
    total_observations = 0
    for frame_id in layout.annotation_frame_ids:
        frame = load_frame_annotations(layout.annotation_path(frame_id), frame_id)
        total_observations += frame.person_count
        distinct_person_ids.update(person.person_id.value for person in frame.persons)

    return WildtrackDatasetInspection(
        camera_ids=CAMERA_IDS,
        image_frame_count=len(layout.image_frame_ids),
        annotated_frame_count=len(layout.annotation_frame_ids),
        first_annotated_frame_value=layout.annotation_frame_ids[0].value,
        last_annotated_frame_value=layout.annotation_frame_ids[-1].value,
        annotation_cadence_hz=FRAME_RATE_HZ / FRAME_STEP,
        calibrated_camera_count=len(CAMERA_IDS),
        total_person_observation_count=total_observations,
        distinct_person_id_count=len(distinct_person_ids),
    )

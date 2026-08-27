"""Ground-truth-only cross-camera correspondence and occupancy for WILDTRACK.

Everything here is derived purely from WILDTRACK's own annotation, never from
Atreides' detector, tracker, association or fusion output. It answers "what
does WILDTRACK itself say happened", so a later, separate evaluator can
compare it against what Atreides said happened. This module creates no
:class:`~atreides_vision.identity.models.VenueSubjectId` and calls nothing in
:mod:`atreides_vision.association`, :mod:`atreides_vision.fusion` or
:mod:`atreides_vision.occupancy` — those layers must stay blind to which
dataset, if any, produced the frames they are scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

from atreides_vision.venue_validation.datasets.wildtrack.annotations import (
    WildtrackFrameAnnotations,
    WildtrackPersonAnnotation,
    WildtrackPersonId,
)
from atreides_vision.venue_validation.datasets.wildtrack.cameras import CAMERA_IDS

__all__ = [
    "WildtrackCrossCameraCorrespondence",
    "WildtrackGroundTruthOccupancy",
    "cross_camera_correspondences",
    "ground_truth_occupancy",
]


@dataclass(frozen=True, slots=True)
class WildtrackCrossCameraCorrespondence:
    """The set of cameras that, according to WILDTRACK, saw the same person at once.

    This is ground truth about *which observations belong to the same real
    person*, the same question :mod:`atreides_vision.association` answers
    from Atreides' own tracks — but computed here from WILDTRACK's
    ``personID``, never from anything Atreides produced. ``camera_ids`` is
    always in :data:`~atreides_vision.venue_validation.datasets.wildtrack.cameras.CAMERA_IDS`
    order, independent of annotation file ordering.
    """

    person_id: WildtrackPersonId
    camera_ids: tuple[str, ...]

    @property
    def camera_count(self) -> int:
        return len(self.camera_ids)

    @property
    def is_single_camera(self) -> bool:
        """Whether WILDTRACK observed this person from exactly one camera.

        A true, and expected, outcome: not every visible person is seen by
        every camera, and single-camera observation must remain a normal case
        rather than a degenerate one — including here, in ground truth.
        """
        return self.camera_count == 1


def cross_camera_correspondences(
    frame: WildtrackFrameAnnotations,
) -> tuple[WildtrackCrossCameraCorrespondence, ...]:
    """One correspondence per annotated person, in canonical personID order."""
    return tuple(
        WildtrackCrossCameraCorrespondence(
            person_id=person.person_id, camera_ids=person.visible_camera_ids
        )
        for person in frame.persons
    )


@dataclass(frozen=True, slots=True)
class WildtrackGroundTruthOccupancy:
    """WILDTRACK's own answer to "how many people were present" at one frame.

    Two distinct counts, kept apart deliberately:

    * :attr:`full_scene_person_count` — every annotated person at this frame,
      regardless of which (if any) cameras saw them. WILDTRACK annotates from
      the full multi-camera reconstruction, so this can include a person seen
      by zero of a restricted camera subset.
    * :attr:`observed_person_count` — people visible in at least one camera of
      ``considered_camera_ids``. Equal to the full count only when every
      camera is considered and every annotated person is visible somewhere.

    Neither is a prediction or a score of Atreides' own output — this module
    contains no comparison logic. A separate evaluator uses these as the
    truth side of such a comparison.
    """

    frame_id_value: int
    considered_camera_ids: tuple[str, ...]
    full_scene_person_count: int
    observed_person_count: int


def ground_truth_occupancy(
    frame: WildtrackFrameAnnotations,
    *,
    considered_camera_ids: tuple[str, ...] | None = None,
) -> WildtrackGroundTruthOccupancy:
    """Ground-truth occupancy for one frame, optionally restricted to some cameras.

    Args:
        frame: The parsed annotations for one frame.
        considered_camera_ids: Which cameras' visibility counts as "observed".
            Defaults to all of :data:`~...cameras.CAMERA_IDS`. Order does not
            affect the result: only membership matters.
    """
    considered = considered_camera_ids if considered_camera_ids is not None else CAMERA_IDS
    considered_set = set(considered)
    observed = sum(1 for person in frame.persons if _visible_in_any(person, considered_set))
    return WildtrackGroundTruthOccupancy(
        frame_id_value=frame.frame_id.value,
        considered_camera_ids=tuple(
            camera_id for camera_id in CAMERA_IDS if camera_id in considered_set
        ),
        full_scene_person_count=frame.person_count,
        observed_person_count=observed,
    )


def _visible_in_any(person: WildtrackPersonAnnotation, camera_ids: set[str]) -> bool:
    return any(camera_id in camera_ids for camera_id in person.visible_camera_ids)

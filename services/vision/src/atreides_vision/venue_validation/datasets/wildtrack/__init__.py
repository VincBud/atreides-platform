"""WILDTRACK dataset ground-truth adapter — offline validation infrastructure only.

`WILDTRACK <https://www.epfl.ch/labs/cvlab/data/data-wildtrack/>`_ is a public
seven-camera pedestrian dataset. This package teaches Atreides' offline
validation harness to read WILDTRACK's own file formats — images, per-frame
JSON annotations, and OpenCV-style camera calibration — and turn them into
typed, dataset-scoped ground truth for comparing against Atreides' own
output.

**This is validation infrastructure, never a runtime dependency.** Nothing
under :mod:`atreides_vision.behavior`, :mod:`atreides_vision.interaction`,
:mod:`atreides_vision.crowd`, :mod:`atreides_vision.zones`,
:mod:`atreides_vision.association`, :mod:`atreides_vision.fusion`,
:mod:`atreides_vision.identity` or :mod:`atreides_vision.occupancy` imports
this package, and nothing here creates a
:class:`~atreides_vision.identity.models.VenueSubjectId` or otherwise feeds
WILDTRACK's own ``personID`` into anything Atreides treats as identity.
WILDTRACK ground truth is offline evaluation truth only.

The real, locally-extracted 7.1 GB dataset lives outside this repository.
Nothing in this package hardcodes its path — every entry point accepts a
:class:`pathlib.Path` supplied by the caller — and it is never committed to
version control. Unit tests exercise this package against tiny synthetic
fixtures that reproduce WILDTRACK's own file layout and schema, not the real
dataset.

Public surface
--------------
* :mod:`.cameras` — the fixed seven-camera naming and ``C<n>``/calibration
  mapping.
* :mod:`.time` — frame identifier and venue-time semantics.
* :mod:`.annotations` — ground-truth domain models
  (:class:`~.annotations.WildtrackPersonId`,
  :class:`~.annotations.WildtrackPersonAnnotation`,
  :class:`~.annotations.WildtrackFrameAnnotations`) and JSON parsing.
* :mod:`.layout` — locating and validating one dataset root
  (:class:`~.layout.WildtrackDatasetLayout`).
* :mod:`.calibration` — parsing WILDTRACK's own OpenCV calibration and
  converting it into an Atreides :class:`~atreides_vision.calibration.fitting.CalibrationFit`.
* :mod:`.correspondence` — ground-truth-only cross-camera correspondence and
  occupancy.
* :mod:`.inspection` — a deterministic, non-identifying dataset summary.
* :mod:`.runtime` — the one runtime-facing seam: per-camera
  :class:`~.runtime.WildtrackStreamSpec` frame feeds
  (:func:`~.runtime.build_wildtrack_stream_specs`) that carry images, frame
  identifiers, cameras and declared timing and nothing from the person-level
  ground truth.
"""

from __future__ import annotations

from atreides_vision.venue_validation.datasets.wildtrack.annotations import (
    WildtrackBoundingBox,
    WildtrackFrameAnnotations,
    WildtrackGroundPosition,
    WildtrackPersonAnnotation,
    WildtrackPersonId,
    load_frame_annotations,
)
from atreides_vision.venue_validation.datasets.wildtrack.calibration import (
    WildtrackCameraCalibration,
    WildtrackExtrinsic,
    WildtrackIntrinsic,
    load_camera_calibration,
    to_ground_plane_calibration,
)
from atreides_vision.venue_validation.datasets.wildtrack.cameras import (
    CAMERA_IDS,
    calibration_name_for,
    camera_id_for_view_num,
    view_num_for_camera_id,
)
from atreides_vision.venue_validation.datasets.wildtrack.correspondence import (
    WildtrackCrossCameraCorrespondence,
    WildtrackGroundTruthOccupancy,
    cross_camera_correspondences,
    ground_truth_occupancy,
)
from atreides_vision.venue_validation.datasets.wildtrack.inspection import (
    WildtrackDatasetInspection,
    inspect_dataset,
)
from atreides_vision.venue_validation.datasets.wildtrack.layout import WildtrackDatasetLayout
from atreides_vision.venue_validation.datasets.wildtrack.runtime import (
    WILDTRACK_DECLARED_FPS,
    WildtrackStreamSpec,
    build_wildtrack_stream_specs,
)
from atreides_vision.venue_validation.datasets.wildtrack.session import (
    WildtrackRuntimeSession,
    build_wildtrack_runtime_session,
)
from atreides_vision.venue_validation.datasets.wildtrack.time import WildtrackFrameId

__all__ = [
    "CAMERA_IDS",
    "WILDTRACK_DECLARED_FPS",
    "WildtrackBoundingBox",
    "WildtrackCameraCalibration",
    "WildtrackCrossCameraCorrespondence",
    "WildtrackDatasetInspection",
    "WildtrackDatasetLayout",
    "WildtrackExtrinsic",
    "WildtrackFrameAnnotations",
    "WildtrackFrameId",
    "WildtrackGroundPosition",
    "WildtrackGroundTruthOccupancy",
    "WildtrackIntrinsic",
    "WildtrackPersonAnnotation",
    "WildtrackPersonId",
    "WildtrackRuntimeSession",
    "WildtrackStreamSpec",
    "build_wildtrack_runtime_session",
    "build_wildtrack_stream_specs",
    "calibration_name_for",
    "camera_id_for_view_num",
    "cross_camera_correspondences",
    "ground_truth_occupancy",
    "inspect_dataset",
    "load_camera_calibration",
    "load_frame_annotations",
    "to_ground_plane_calibration",
    "view_num_for_camera_id",
]

"""Per-instant occupancy scoring for the offline WILDTRACK evaluator.

Ground-truth occupancy is restricted to exactly the cameras this run scored:
a Stage-A subset run is compared against the people its subset could actually
observe, never against the full-scene count. The raw count is the runtime's
pre-fusion active-track total; the fused count is the venue-level occupancy
after duplicate suppression.
"""

from __future__ import annotations

from collections.abc import Sequence

from atreides_vision.venue_validation.datasets.wildtrack.annotations import (
    WildtrackFrameAnnotations,
)
from atreides_vision.venue_validation.datasets.wildtrack.correspondence import (
    ground_truth_occupancy,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.models import (
    InstantOccupancyEvaluation,
)
from atreides_vision.venue_validation.observer import VenueValidationInstantSnapshot

__all__ = ["score_instant_occupancy"]


def score_instant_occupancy(
    snapshot: VenueValidationInstantSnapshot,
    frame: WildtrackFrameAnnotations,
    *,
    selected_camera_ids: Sequence[str],
) -> InstantOccupancyEvaluation:
    """Score one venue instant's occupancy against WILDTRACK ground truth."""
    observable = ground_truth_occupancy(
        frame, considered_camera_ids=tuple(selected_camera_ids)
    ).observed_person_count

    occupancy = snapshot.occupancy
    return InstantOccupancyEvaluation(
        ground_truth_observable_count=observable,
        raw_active_track_count=occupancy.raw_active_track_count,
        fused_occupancy_count=occupancy.occupancy_count,
        suppressed_duplicate_count=occupancy.suppressed_duplicate_count,
    )

"""Offline WILDTRACK evaluator — scores runtime venue instants against ground truth.

This package is loaded *after* a WILDTRACK runtime session has produced its
per-instant snapshots. It never feeds anything back into the runtime: WILDTRACK
``personID`` values and :class:`~.ground_truth.WildtrackGroundTruth` stay
evaluator-only, and nothing here imports the association engine, candidate
builder, gates or matcher — only the frozen
:mod:`atreides_vision.association.models` domain types.

:class:`~.evaluator.WildtrackEvaluator` is a
:class:`~atreides_vision.venue_validation.observer.VenueValidationObserver`: the
venue-validation runner calls it once per venue instant, in order, and
:meth:`~.evaluator.WildtrackEvaluator.finalize` returns a whole-run
:class:`~.models.WildtrackEvaluationSummary`. The two artifacts
(:data:`~.reporting.WILDTRACK_EVALUATION_REPORT_FILENAME`,
:data:`~.reporting.WILDTRACK_EVALUATION_SUMMARY_FILENAME`) are deterministic and
byte-for-byte reproducible.
"""

from __future__ import annotations

from atreides_vision.venue_validation.datasets.wildtrack.evaluation.config import (
    MAX_GROUND_PLANE_DISTANCE_M,
    VENUE_SECONDS_TOLERANCE,
    EvaluationConfig,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.evaluator import (
    WildtrackEvaluator,
    aggregate_evaluation,
    evaluate_instant,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.ground_truth import (
    WildtrackGroundTruth,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.models import (
    CameraCoverage,
    CameraCoverageSummary,
    CameraMatch,
    InstantAssociationEvaluation,
    InstantEvaluation,
    InstantFusionEvaluation,
    InstantOccupancyEvaluation,
    OccupancyEvaluationSummary,
    Ratio,
    WildtrackEvaluationSummary,
    build_camera_matches,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.reporting import (
    WILDTRACK_EVALUATION_REPORT_FILENAME,
    WILDTRACK_EVALUATION_SCHEMA_VERSION,
    WILDTRACK_EVALUATION_SUMMARY_FILENAME,
    write_wildtrack_evaluation_report,
    write_wildtrack_evaluation_summary,
)

__all__ = [
    "MAX_GROUND_PLANE_DISTANCE_M",
    "VENUE_SECONDS_TOLERANCE",
    "WILDTRACK_EVALUATION_REPORT_FILENAME",
    "WILDTRACK_EVALUATION_SCHEMA_VERSION",
    "WILDTRACK_EVALUATION_SUMMARY_FILENAME",
    "CameraCoverage",
    "CameraCoverageSummary",
    "CameraMatch",
    "EvaluationConfig",
    "InstantAssociationEvaluation",
    "InstantEvaluation",
    "InstantFusionEvaluation",
    "InstantOccupancyEvaluation",
    "OccupancyEvaluationSummary",
    "Ratio",
    "WildtrackEvaluationSummary",
    "WildtrackEvaluator",
    "WildtrackGroundTruth",
    "aggregate_evaluation",
    "build_camera_matches",
    "evaluate_instant",
    "write_wildtrack_evaluation_report",
    "write_wildtrack_evaluation_summary",
]

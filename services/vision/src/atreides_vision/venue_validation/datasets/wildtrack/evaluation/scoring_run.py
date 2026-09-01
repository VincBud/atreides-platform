"""One programmatic entry point: run a WILDTRACK session and score it (V0.4.9b-D2).

This is the smallest orchestration that joins three things that already exist and
already have their own tests:

* :func:`~...wildtrack.session.build_wildtrack_runtime_session` — compose the real,
  WILDTRACK-agnostic venue-validation runtime for a camera subset and a contiguous
  window of annotated instants;
* :func:`~atreides_vision.venue_validation.runner.run_venue_validation` — walk that
  session once, calling an observer after association, fusion and occupancy for
  every venue instant;
* :class:`~...wildtrack.evaluation.evaluator.WildtrackEvaluator` — the frozen D1
  offline evaluator, attached as that observer.

It builds no second runtime loop, tunes nothing, and reads no person-level ground
truth outside the evaluator: WILDTRACK ``personID`` values enter strictly below the
observer seam, exactly as before. The detector and tracker arrive injected, so this
module never imports a model library and stays runnable in a test with fakes.

After the run it writes the three deterministic D1 artifacts into ``output_dir``:
``wildtrack_evaluation.jsonl``, ``wildtrack_validation_summary.json`` and — only
when the run produced at least one ground-truth-resolved false merge —
``false_merge_diagnostics.jsonl``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from atreides_vision.config import DetectorConfig, TrackerConfig
from atreides_vision.detection.base import PersonDetector
from atreides_vision.tracking.base import MultiObjectTracker
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.config import EvaluationConfig
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.evaluator import (
    WildtrackEvaluator,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.models import (
    WildtrackEvaluationSummary,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.reporting import (
    WILDTRACK_EVALUATION_REPORT_FILENAME,
    WILDTRACK_EVALUATION_SUMMARY_FILENAME,
    WILDTRACK_FALSE_MERGE_DIAGNOSTICS_FILENAME,
    write_false_merge_diagnostics,
    write_wildtrack_evaluation_report,
    write_wildtrack_evaluation_summary,
)
from atreides_vision.venue_validation.datasets.wildtrack.layout import WildtrackDatasetLayout
from atreides_vision.venue_validation.datasets.wildtrack.session import (
    build_wildtrack_runtime_session,
)
from atreides_vision.venue_validation.models import VenueValidationSummary
from atreides_vision.venue_validation.runner import run_venue_validation

__all__ = [
    "WildtrackScoringArtifacts",
    "run_wildtrack_scoring",
]


@dataclass(frozen=True, slots=True)
class WildtrackScoringArtifacts:
    """What one scored WILDTRACK run produced.

    ``false_merge_diagnostics_path`` is ``None`` when the run contained no
    ground-truth-resolved false merge — the absence of the file is meaningful and
    is not an error.
    """

    evaluation: WildtrackEvaluationSummary
    run_statistics: VenueValidationSummary
    evaluation_report_path: Path
    evaluation_summary_path: Path
    false_merge_diagnostics_path: Path | None


def run_wildtrack_scoring(
    layout: WildtrackDatasetLayout,
    *,
    output_dir: Path,
    detector: PersonDetector,
    tracker_factory: Callable[[TrackerConfig], MultiObjectTracker],
    detector_config: DetectorConfig,
    selected_camera_ids: Iterable[str] | None = None,
    start_instant_index: int = 0,
    instant_limit: int | None = None,
    calibrated: bool = True,
) -> WildtrackScoringArtifacts:
    """Run one WILDTRACK venue session end to end and score it against ground truth.

    Args:
        layout: A validated WILDTRACK dataset layout.
        output_dir: Where both the generic runner and the evaluator write their
            artifacts.
        detector: The person detector, already loaded. Injected so this module
            never imports a model library.
        tracker_factory: Builds a fresh camera-local tracker from a stream's
            frame rate, exactly as the generic runner expects.
        detector_config: The detector configuration to stamp onto the run config
            so the camera-local pipeline's detection settings match the injected
            detector. It must describe the same detector that was passed in.
        selected_camera_ids: The camera subset, in any order; ``None`` selects all
            seven. The run always uses canonical ``C1..C7`` order.
        start_instant_index: Index of the first annotated instant to score.
        instant_limit: Number of contiguous annotated instants from
            ``start_instant_index``; ``None`` scores every remaining one.
        calibrated: When true (the baseline), each camera runs with a
            :class:`CalibrationFit` derived from WILDTRACK's own calibration.

    Returns:
        The evaluation summary, the run's execution statistics, and the paths of
        the artifacts written.
    """
    session = build_wildtrack_runtime_session(
        layout,
        output_dir=output_dir,
        selected_camera_ids=selected_camera_ids,
        start_instant_index=start_instant_index,
        instant_limit=instant_limit,
        calibrated=calibrated,
    )
    config = session.config
    if config.detector != detector_config:
        config = dataclasses.replace(config, detector=detector_config)

    evaluator = WildtrackEvaluator.from_layout(
        EvaluationConfig(selected_camera_ids=session.camera_ids),
        layout,
        session.scored_frame_ids,
    )

    run_statistics = run_venue_validation(
        config,
        detector=detector,
        tracker_factory=tracker_factory,
        source_factory=session.source_factory,
        observer=evaluator,
    )
    evaluation = evaluator.finalize()

    report_path = write_wildtrack_evaluation_report(
        output_dir / WILDTRACK_EVALUATION_REPORT_FILENAME, evaluation, evaluator.instants
    )
    summary_path = write_wildtrack_evaluation_summary(
        output_dir / WILDTRACK_EVALUATION_SUMMARY_FILENAME, evaluation
    )
    false_merge_path = write_false_merge_diagnostics(
        output_dir / WILDTRACK_FALSE_MERGE_DIAGNOSTICS_FILENAME,
        evaluation.association.false_merges_detail,
    )

    return WildtrackScoringArtifacts(
        evaluation=evaluation,
        run_statistics=run_statistics,
        evaluation_report_path=report_path,
        evaluation_summary_path=summary_path,
        false_merge_diagnostics_path=false_merge_path,
    )

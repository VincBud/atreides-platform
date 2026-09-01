"""The offline WILDTRACK evaluator: a streaming observer over venue instants.

:class:`WildtrackEvaluator` implements the
:class:`~atreides_vision.venue_validation.observer.VenueValidationObserver`
protocol. The venue-validation runner calls it once per venue instant, in order,
after association, fusion and occupancy have completed for that instant. The
evaluator holds WILDTRACK ground truth (strictly downstream of the runtime),
scores each instant against it, and at :meth:`WildtrackEvaluator.finalize`
returns a whole-run :class:`WildtrackEvaluationSummary`.

Frame alignment is exact and checked: the ``k``-th snapshot the evaluator
receives is scored against ``ground_truth.frame(k)`` — the same ordinal as
``WildtrackRuntimeSession.scored_frame_ids[k]``. WILDTRACK's declared cadence is
:data:`~atreides_vision.venue_validation.datasets.wildtrack.runtime.WILDTRACK_DECLARED_FPS`,
so the snapshot's ``venue_seconds`` must equal ``k / WILDTRACK_DECLARED_FPS`` to
within :attr:`~...evaluation.config.EvaluationConfig.venue_seconds_tolerance`;
any drift, or too many / too few instants, raises
:class:`~atreides_vision.errors.ValidationError`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from atreides_vision.errors import ValidationError
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.association import (
    score_instant_association,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.config import EvaluationConfig
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.fusion import (
    score_instant_fusion,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.ground_truth import (
    WildtrackGroundTruth,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.models import (
    CameraCoverage,
    CameraCoverageSummary,
    InstantAssociationEvaluation,
    InstantEvaluation,
    InstantFusionEvaluation,
    OccupancyEvaluationSummary,
    WildtrackEvaluationSummary,
    build_camera_matches,
    mean,
    rmse,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.occupancy import (
    score_instant_occupancy,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.reporting import (
    WILDTRACK_EVALUATION_SCHEMA_VERSION,
)
from atreides_vision.venue_validation.datasets.wildtrack.layout import WildtrackDatasetLayout
from atreides_vision.venue_validation.datasets.wildtrack.runtime import WILDTRACK_DECLARED_FPS
from atreides_vision.venue_validation.datasets.wildtrack.time import WildtrackFrameId
from atreides_vision.venue_validation.observer import VenueValidationInstantSnapshot

__all__ = [
    "WildtrackEvaluator",
    "aggregate_evaluation",
    "evaluate_instant",
]


def evaluate_instant(
    snapshot: VenueValidationInstantSnapshot,
    ground_truth: WildtrackGroundTruth,
    config: EvaluationConfig,
    *,
    instant_index: int,
) -> InstantEvaluation:
    """Score one venue instant against its aligned WILDTRACK ground-truth frame."""
    frame = ground_truth.frame(instant_index)
    selected = config.selected_camera_ids
    camera_matches = build_camera_matches(
        snapshot,
        frame,
        selected_camera_ids=selected,
        radius_metres=config.max_ground_plane_distance_m,
    )

    cameras = tuple(
        CameraCoverage(
            camera_id=camera_id,
            runtime_observation_count=camera_matches[camera_id].runtime_observation_count,
            metric_observation_count=camera_matches[camera_id].metric_observation_count,
            non_metric_observation_count=camera_matches[camera_id].non_metric_observation_count,
            gt_person_count=camera_matches[camera_id].gt_person_count,
            matched_count=camera_matches[camera_id].matched_count,
            total_distance_micrometres=camera_matches[camera_id].total_distance_micrometres,
        )
        for camera_id in selected
    )

    association = score_instant_association(
        snapshot, frame, camera_matches, selected_camera_ids=selected
    )
    fusion = score_instant_fusion(snapshot, frame, camera_matches, selected_camera_ids=selected)
    occupancy = score_instant_occupancy(snapshot, frame, selected_camera_ids=selected)

    match_distances = tuple(
        sorted(
            distance
            for camera_id in selected
            for distance in camera_matches[camera_id].pair_distances_micrometres
        )
    )

    return InstantEvaluation(
        instant_index=instant_index,
        venue_seconds=snapshot.venue_seconds,
        expected_venue_seconds=instant_index / WILDTRACK_DECLARED_FPS,
        cameras=cameras,
        association=association,
        fusion=fusion,
        occupancy=occupancy,
        match_distances_micrometres=match_distances,
    )


def _aggregate_association(
    instants: Sequence[InstantEvaluation],
) -> InstantAssociationEvaluation:
    gate: Counter[str] = Counter()
    decision: Counter[str] = Counter()
    for instant in instants:
        gate.update(instant.association.gate_reason_counts)
        decision.update(instant.association.decision_reason_counts)
    return InstantAssociationEvaluation(
        true_positive_merges=sum(i.association.true_positive_merges for i in instants),
        false_merges=sum(i.association.false_merges for i in instants),
        unresolved_for_scoring=sum(i.association.unresolved_for_scoring for i in instants),
        evaluable_pair_count=sum(i.association.evaluable_pair_count for i in instants),
        evaluable_associated_count=sum(i.association.evaluable_associated_count for i in instants),
        missed_association_count=sum(i.association.missed_association_count for i in instants),
        ambiguous_pair_count=sum(i.association.ambiguous_pair_count for i in instants),
        support_not_sustained_count=sum(
            i.association.support_not_sustained_count for i in instants
        ),
        not_evaluable_pair_count=sum(i.association.not_evaluable_pair_count for i in instants),
        end_to_end_pair_count=sum(i.association.end_to_end_pair_count for i in instants),
        end_to_end_satisfied_count=sum(i.association.end_to_end_satisfied_count for i in instants),
        gate_reason_counts=dict(gate),
        decision_reason_counts=dict(decision),
        false_merges_detail=tuple(
            diagnostic
            for instant in instants
            for diagnostic in instant.association.false_merges_detail
        ),
    )


def _aggregate_fusion(instants: Sequence[InstantEvaluation]) -> InstantFusionEvaluation:
    outcomes: Counter[str] = Counter()
    for instant in instants:
        outcomes.update(instant.fusion.integration_outcome_counts)
    return InstantFusionEvaluation(
        evaluable_subject_count=sum(i.fusion.evaluable_subject_count for i in instants),
        impure_subject_count=sum(i.fusion.impure_subject_count for i in instants),
        persons_with_owned_match=sum(i.fusion.persons_with_owned_match for i in instants),
        fragmented_person_count=sum(i.fusion.fragmented_person_count for i in instants),
        matched_tracklet_count=sum(i.fusion.matched_tracklet_count for i in instants),
        matched_unowned_count=sum(i.fusion.matched_unowned_count for i in instants),
        matched_unowned_person_count=sum(i.fusion.matched_unowned_person_count for i in instants),
        integration_outcome_counts=dict(outcomes),
    )


def _aggregate_occupancy(instants: Sequence[InstantEvaluation]) -> OccupancyEvaluationSummary:
    raw_errors = [i.occupancy.raw_error for i in instants]
    fused_errors = [i.occupancy.fused_error for i in instants]
    abs_raw = [abs(error) for error in raw_errors]
    abs_fused = [abs(error) for error in fused_errors]

    improved = unchanged = worsened = 0
    for raw_error, fused_error in zip(abs_raw, abs_fused, strict=True):
        if fused_error < raw_error:
            improved += 1
        elif fused_error == raw_error:
            unchanged += 1
        else:
            worsened += 1

    return OccupancyEvaluationSummary(
        instant_count=len(instants),
        ground_truth_total=sum(i.occupancy.ground_truth_observable_count for i in instants),
        raw_total=sum(i.occupancy.raw_active_track_count for i in instants),
        fused_total=sum(i.occupancy.fused_occupancy_count for i in instants),
        raw_mae=mean(abs_raw),
        fused_mae=mean(abs_fused),
        raw_bias=mean(raw_errors),
        fused_bias=mean(fused_errors),
        raw_rmse=rmse(raw_errors),
        fused_rmse=rmse(fused_errors),
        exact_fused_count=sum(1 for error in fused_errors if error == 0),
        fused_over_count=sum(1 for error in fused_errors if error > 0),
        fused_under_count=sum(1 for error in fused_errors if error < 0),
        max_abs_raw_error=max(abs_raw, default=0),
        max_abs_fused_error=max(abs_fused, default=0),
        improved_count=improved,
        unchanged_count=unchanged,
        worsened_count=worsened,
        total_suppressed_duplicates=sum(i.occupancy.suppressed_duplicate_count for i in instants),
    )


def aggregate_evaluation(
    instants: Sequence[InstantEvaluation],
    config: EvaluationConfig,
) -> WildtrackEvaluationSummary:
    """Combine per-instant evaluations into one whole-run summary."""
    if not instants:
        raise ValidationError("A WILDTRACK evaluation summary needs at least one scored instant.")

    coverage: list[CameraCoverageSummary] = []
    for index, camera_id in enumerate(config.selected_camera_ids):
        coverage.append(
            CameraCoverageSummary(
                camera_id=camera_id,
                metric_observation_count=sum(
                    instant.cameras[index].metric_observation_count for instant in instants
                ),
                matched_count=sum(instant.cameras[index].matched_count for instant in instants),
                gt_person_count=sum(instant.cameras[index].gt_person_count for instant in instants),
            )
        )

    return WildtrackEvaluationSummary(
        schema_version=WILDTRACK_EVALUATION_SCHEMA_VERSION,
        selected_camera_ids=tuple(config.selected_camera_ids),
        instant_count=len(instants),
        max_ground_plane_distance_m=config.max_ground_plane_distance_m,
        camera_coverage=tuple(coverage),
        association=_aggregate_association(instants),
        fusion=_aggregate_fusion(instants),
        occupancy=_aggregate_occupancy(instants),
        match_distances_micrometres=tuple(
            sorted(
                distance for instant in instants for distance in instant.match_distances_micrometres
            )
        ),
    )


class WildtrackEvaluator:
    """A streaming :class:`VenueValidationObserver` that scores a WILDTRACK run."""

    def __init__(self, config: EvaluationConfig, ground_truth: WildtrackGroundTruth) -> None:
        self._config = config
        self._ground_truth = ground_truth
        self._instants: list[InstantEvaluation] = []
        self._finalized = False

    @classmethod
    def from_layout(
        cls,
        config: EvaluationConfig,
        layout: WildtrackDatasetLayout,
        scored_frame_ids: Sequence[WildtrackFrameId],
    ) -> WildtrackEvaluator:
        return cls(config, WildtrackGroundTruth.from_layout(layout, scored_frame_ids))

    @property
    def instants(self) -> tuple[InstantEvaluation, ...]:
        return tuple(self._instants)

    def __call__(self, snapshot: VenueValidationInstantSnapshot, /) -> None:
        if self._finalized:
            raise ValidationError(
                "WildtrackEvaluator received a venue instant after finalize(); the run is over."
            )
        index = len(self._instants)
        if index >= len(self._ground_truth):
            raise ValidationError(
                f"WildtrackEvaluator received venue instant {index} but WILDTRACK ground truth "
                f"only has {len(self._ground_truth)} scored frames; the run emitted too many "
                "instants."
            )
        expected = index / WILDTRACK_DECLARED_FPS
        if abs(snapshot.venue_seconds - expected) > self._config.venue_seconds_tolerance:
            raise ValidationError(
                f"WILDTRACK venue instant {index} arrived at venue_seconds="
                f"{snapshot.venue_seconds!r}, expected {expected!r} "
                f"(± {self._config.venue_seconds_tolerance}); the scored stream is misaligned."
            )
        self._instants.append(
            evaluate_instant(snapshot, self._ground_truth, self._config, instant_index=index)
        )

    def finalize(self) -> WildtrackEvaluationSummary:
        if self._finalized:
            raise ValidationError("WildtrackEvaluator.finalize() was already called.")
        if len(self._instants) != len(self._ground_truth):
            raise ValidationError(
                f"WildtrackEvaluator scored {len(self._instants)} venue instants but WILDTRACK "
                f"ground truth has {len(self._ground_truth)} scored frames; the run emitted too "
                "few instants."
            )
        self._finalized = True
        return aggregate_evaluation(self._instants, self._config)

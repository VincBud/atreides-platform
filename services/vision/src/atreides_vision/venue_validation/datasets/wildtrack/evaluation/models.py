"""Frozen result types for the offline WILDTRACK evaluator (V0.4.9b-D1).

Every type here is a factual measurement of one runtime venue instant (or an
aggregate of many) against WILDTRACK ground truth. Nothing is a grade: a ratio
carries its explicit numerator and denominator and reports ``value = None`` when
the denominator is zero, so "not measured" never collapses into a misleading
zero.

The one non-trivial piece of logic in this module is
:func:`build_camera_matches`: for one venue instant it decides, per camera,
which runtime observation stands in for which annotated person, using the
evaluator's own deterministic matcher (:mod:`.matching`) and the frozen
:data:`~atreides_vision.venue_validation.datasets.wildtrack.evaluation.config.MAX_GROUND_PLANE_DISTANCE_M`
radius. It imports nothing from :mod:`atreides_vision.association`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from atreides_vision.association.models import (
    TrackletKey,
    observation_order_key,
    tracklet_key,
)
from atreides_vision.behavior.coordinates import MotionSpace
from atreides_vision.venue_validation.datasets.wildtrack.annotations import (
    WildtrackFrameAnnotations,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.matching import (
    LocalMatch,
    distance_micrometres,
    match_points,
)
from atreides_vision.venue_validation.observer import VenueValidationInstantSnapshot

__all__ = [
    "CameraCoverage",
    "CameraCoverageSummary",
    "CameraMatch",
    "FalseMergeDiagnostic",
    "InstantAssociationEvaluation",
    "InstantEvaluation",
    "InstantFusionEvaluation",
    "InstantOccupancyEvaluation",
    "LocalCoverage",
    "MatchDistanceStats",
    "OccupancyEvaluationSummary",
    "Ratio",
    "WildtrackEvaluationSummary",
    "build_camera_matches",
]

_METRES_PLACES = 6
_MICROMETRES_PER_METRE = 1_000_000


def _round(value: float, places: int = _METRES_PLACES) -> float:
    return round(value, places)


@dataclass(frozen=True, slots=True)
class Ratio:
    """An explicit numerator/denominator pair with a null-safe value.

    ``value`` is ``numerator / denominator`` when the denominator is non-zero
    and ``None`` otherwise — an undefined ratio is reported as undefined, never
    as ``0.0``.
    """

    numerator: int
    denominator: int

    @classmethod
    def of(cls, numerator: int, denominator: int) -> Ratio:
        return cls(numerator=numerator, denominator=denominator)

    @property
    def value(self) -> float | None:
        if self.denominator == 0:
            return None
        return self.numerator / self.denominator

    def to_record(self) -> dict[str, object]:
        value = self.value
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": None if value is None else _round(value),
        }


class _CameraCounts(Protocol):
    """The camera-local counts :class:`LocalCoverage` aggregates.

    Satisfied structurally by both :class:`CameraCoverage` (per instant) and
    :class:`CameraCoverageSummary` (whole run).
    """

    @property
    def metric_observation_count(self) -> int: ...

    @property
    def matched_count(self) -> int: ...

    @property
    def gt_person_count(self) -> int: ...


@dataclass(frozen=True, slots=True)
class LocalCoverage:
    """Aggregate camera-local GT<->runtime coverage for one instant or a whole run.

    Every count is over ground-plane-metre observations only; non-metric runtime
    observations are excluded because they are never distance-matched. This is a
    camera-local observation-coverage measurement, not detector mAP or a standard
    detector recall.
    """

    gt_visible_observations: int
    runtime_observations: int
    matched_observations: int

    @classmethod
    def from_cameras(cls, cameras: Sequence[_CameraCounts]) -> LocalCoverage:
        return cls(
            gt_visible_observations=sum(camera.gt_person_count for camera in cameras),
            runtime_observations=sum(camera.metric_observation_count for camera in cameras),
            matched_observations=sum(camera.matched_count for camera in cameras),
        )

    @property
    def unmatched_gt_observations(self) -> int:
        return self.gt_visible_observations - self.matched_observations

    @property
    def unmatched_runtime_observations(self) -> int:
        return self.runtime_observations - self.matched_observations

    @property
    def local_observation_coverage(self) -> Ratio:
        return Ratio.of(self.matched_observations, self.gt_visible_observations)

    def to_record(self) -> dict[str, object]:
        return {
            "gt_visible_observations": self.gt_visible_observations,
            "runtime_observations": self.runtime_observations,
            "matched_observations": self.matched_observations,
            "unmatched_gt_observations": self.unmatched_gt_observations,
            "unmatched_runtime_observations": self.unmatched_runtime_observations,
            "local_observation_coverage": self.local_observation_coverage.to_record(),
        }


@dataclass(frozen=True, slots=True)
class MatchDistanceStats:
    """Distribution of every successful camera-local GT<->runtime match distance.

    Percentile convention: nearest-rank on the ascending-sorted sample, 1-indexed,
    ``rank = ceil(0.95 * n)`` clamped to ``[1, n]``; ``p95_m`` is the value at that
    rank. The median is the mean of the two central values for an even ``n``. All
    four fields are ``None`` when there are no matched observations.
    """

    mean_m: float | None
    median_m: float | None
    p95_m: float | None
    max_m: float | None

    @classmethod
    def from_micrometres(cls, distances: Sequence[int]) -> MatchDistanceStats:
        if not distances:
            return cls(mean_m=None, median_m=None, p95_m=None, max_m=None)
        ordered = sorted(distances)
        count = len(ordered)
        mean_um = sum(ordered) / count
        middle = count // 2
        if count % 2:
            median_um = float(ordered[middle])
        else:
            median_um = (ordered[middle - 1] + ordered[middle]) / 2
        rank = min(max(math.ceil(0.95 * count), 1), count)
        return cls(
            mean_m=_round(mean_um / _MICROMETRES_PER_METRE),
            median_m=_round(median_um / _MICROMETRES_PER_METRE),
            p95_m=_round(ordered[rank - 1] / _MICROMETRES_PER_METRE),
            max_m=_round(ordered[-1] / _MICROMETRES_PER_METRE),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "mean_m": self.mean_m,
            "median_m": self.median_m,
            "p95_m": self.p95_m,
            "max_m": self.max_m,
        }


@dataclass(frozen=True, slots=True)
class FalseMergeDiagnostic:
    """One GT-resolved ``ASSOCIATED`` pair that joined two different WILDTRACK persons.

    Endpoints are ordered so ``first_tracklet <= second_tracklet``. Carries only
    motion-space geometry and dataset person IDs — no images, no crops, no
    absolute paths, no wall-clock time, no external identity.
    """

    dataset_frame_id: int
    venue_seconds: float
    first_tracklet: TrackletKey
    second_tracklet: TrackletKey
    first_camera_id: str
    second_camera_id: str
    first_person_id: int
    second_person_id: int
    first_runtime_position: tuple[float, float]
    second_runtime_position: tuple[float, float]
    first_ground_truth_position: tuple[float, float]
    second_ground_truth_position: tuple[float, float]
    first_local_match_distance_m: float | None
    second_local_match_distance_m: float | None
    outcome: str
    reasons: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        def point(value: tuple[float, float]) -> list[float]:
            return [_round(value[0]), _round(value[1])]

        def opt(value: float | None) -> float | None:
            return None if value is None else _round(value)

        return {
            "dataset_frame_id": self.dataset_frame_id,
            "venue_seconds": round(self.venue_seconds, 3),
            "first_tracklet": list(self.first_tracklet),
            "second_tracklet": list(self.second_tracklet),
            "first_camera_id": self.first_camera_id,
            "second_camera_id": self.second_camera_id,
            "first_person_id": self.first_person_id,
            "second_person_id": self.second_person_id,
            "first_runtime_position": point(self.first_runtime_position),
            "second_runtime_position": point(self.second_runtime_position),
            "first_ground_truth_position": point(self.first_ground_truth_position),
            "second_ground_truth_position": point(self.second_ground_truth_position),
            "first_local_match_distance_m": opt(self.first_local_match_distance_m),
            "second_local_match_distance_m": opt(self.second_local_match_distance_m),
            "outcome": self.outcome,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class CameraMatch:
    """One camera's runtime-observation to ground-truth-person assignment for one instant.

    ``observation_keys`` are the camera's ground-plane-metre runtime tracklets in
    canonical observation order (the matcher rows); ``person_ids`` are the
    WILDTRACK ``personID`` values that camera saw, in ascending personID order
    (the matcher columns); ``pairs`` is ``(row, column)`` index pairs from the
    deterministic matcher.
    """

    camera_id: str
    observation_keys: tuple[TrackletKey, ...]
    person_ids: tuple[int, ...]
    non_metric_observation_count: int
    pairs: tuple[tuple[int, int], ...]
    total_distance_micrometres: int
    pair_distances_micrometres: tuple[int, ...]

    @property
    def matched_count(self) -> int:
        return len(self.pairs)

    @property
    def metric_observation_count(self) -> int:
        return len(self.observation_keys)

    @property
    def runtime_observation_count(self) -> int:
        return self.metric_observation_count + self.non_metric_observation_count

    @property
    def gt_person_count(self) -> int:
        return len(self.person_ids)

    def person_for_observation(self, key: TrackletKey) -> int | None:
        try:
            row = self.observation_keys.index(key)
        except ValueError:
            return None
        for pair_row, pair_column in self.pairs:
            if pair_row == row:
                return self.person_ids[pair_column]
        return None

    def observation_for_person(self, person_id: int) -> TrackletKey | None:
        try:
            column = self.person_ids.index(person_id)
        except ValueError:
            return None
        for pair_row, pair_column in self.pairs:
            if pair_column == column:
                return self.observation_keys[pair_row]
        return None

    def match_distance_micrometres_for_observation(self, key: TrackletKey) -> int | None:
        """The GT<->runtime match distance for ``key`` on this camera, if it matched."""
        try:
            row = self.observation_keys.index(key)
        except ValueError:
            return None
        for index, (pair_row, _pair_column) in enumerate(self.pairs):
            if pair_row == row:
                return self.pair_distances_micrometres[index]
        return None


def build_camera_matches(
    snapshot: VenueValidationInstantSnapshot,
    frame: WildtrackFrameAnnotations,
    *,
    selected_camera_ids: Sequence[str],
    radius_metres: float,
) -> dict[str, CameraMatch]:
    """Per-camera runtime/ground-truth matching for one venue instant.

    Only :attr:`~atreides_vision.behavior.coordinates.MotionSpace.GROUND_PLANE_METRES`
    observations are distance-matched; a non-metric observation is counted but
    never silently converted. Rows are that camera's metric observations in
    canonical :func:`observation_order_key` order; columns are the persons
    WILDTRACK records as visible in that camera, in ascending ``personID``
    order.
    """
    matches: dict[str, CameraMatch] = {}
    for camera_id in selected_camera_ids:
        metric_observations = sorted(
            (
                observation
                for observation in snapshot.observations
                if observation.camera_id == camera_id
                and observation.space is MotionSpace.GROUND_PLANE_METRES
            ),
            key=observation_order_key,
        )
        non_metric_count = sum(
            1
            for observation in snapshot.observations
            if observation.camera_id == camera_id
            and observation.space is not MotionSpace.GROUND_PLANE_METRES
        )
        persons = [person for person in frame.persons if person.visible_in(camera_id)]

        row_points = [
            (observation.position.x, observation.position.y) for observation in metric_observations
        ]
        column_points = [(person.position.x_m, person.position.y_m) for person in persons]
        local: LocalMatch = match_points(row_points, column_points, radius_metres=radius_metres)

        matches[camera_id] = CameraMatch(
            camera_id=camera_id,
            observation_keys=tuple(
                tracklet_key(observation) for observation in metric_observations
            ),
            person_ids=tuple(person.person_id.value for person in persons),
            non_metric_observation_count=non_metric_count,
            pairs=local.pairs,
            total_distance_micrometres=local.total_distance_micrometres,
            pair_distances_micrometres=tuple(
                distance_micrometres(row_points[row], column_points[column])
                for row, column in local.pairs
            ),
        )
    return matches


@dataclass(frozen=True, slots=True)
class CameraCoverage:
    """One camera's local coverage for one venue instant."""

    camera_id: str
    runtime_observation_count: int
    metric_observation_count: int
    non_metric_observation_count: int
    gt_person_count: int
    matched_count: int
    total_distance_micrometres: int

    @property
    def runtime_coverage(self) -> Ratio:
        return Ratio.of(self.matched_count, self.metric_observation_count)

    @property
    def ground_truth_coverage(self) -> Ratio:
        return Ratio.of(self.matched_count, self.gt_person_count)

    def to_record(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "runtime_observation_count": self.runtime_observation_count,
            "metric_observation_count": self.metric_observation_count,
            "non_metric_observation_count": self.non_metric_observation_count,
            "gt_person_count": self.gt_person_count,
            "matched_count": self.matched_count,
            "total_distance_micrometres": self.total_distance_micrometres,
            "runtime_coverage": self.runtime_coverage.to_record(),
            "ground_truth_coverage": self.ground_truth_coverage.to_record(),
        }


@dataclass(frozen=True, slots=True)
class InstantAssociationEvaluation:
    """Association scoring for one venue instant."""

    true_positive_merges: int
    false_merges: int
    unresolved_for_scoring: int
    evaluable_pair_count: int
    evaluable_associated_count: int
    missed_association_count: int
    ambiguous_pair_count: int
    support_not_sustained_count: int
    not_evaluable_pair_count: int
    end_to_end_pair_count: int
    end_to_end_satisfied_count: int
    gate_reason_counts: Mapping[str, int]
    decision_reason_counts: Mapping[str, int]
    #: The GT-resolved false merges behind :attr:`false_merges`, for the offline
    #: false-merge diagnostics artifact. Deliberately excluded from
    #: :meth:`to_record` — the summary/report record stays a pure count surface.
    false_merges_detail: tuple[FalseMergeDiagnostic, ...] = ()

    @property
    def association_precision(self) -> Ratio:
        return Ratio.of(self.true_positive_merges, self.true_positive_merges + self.false_merges)

    @property
    def evaluable_recall(self) -> Ratio:
        return Ratio.of(self.evaluable_associated_count, self.evaluable_pair_count)

    @property
    def missed_association_rate(self) -> Ratio:
        return Ratio.of(self.missed_association_count, self.evaluable_pair_count)

    @property
    def ambiguity_rate(self) -> Ratio:
        return Ratio.of(self.ambiguous_pair_count, self.evaluable_pair_count)

    @property
    def end_to_end_recall(self) -> Ratio:
        return Ratio.of(self.end_to_end_satisfied_count, self.end_to_end_pair_count)

    def to_record(self) -> dict[str, object]:
        return {
            "true_positive_merges": self.true_positive_merges,
            "false_merges": self.false_merges,
            "unresolved_for_scoring": self.unresolved_for_scoring,
            "association_precision": self.association_precision.to_record(),
            "evaluable_pair_count": self.evaluable_pair_count,
            "evaluable_associated_count": self.evaluable_associated_count,
            "evaluable_recall": self.evaluable_recall.to_record(),
            "missed_association_count": self.missed_association_count,
            "missed_association_rate": self.missed_association_rate.to_record(),
            "ambiguous_pair_count": self.ambiguous_pair_count,
            "ambiguity_rate": self.ambiguity_rate.to_record(),
            "support_not_sustained_count": self.support_not_sustained_count,
            "not_evaluable_pair_count": self.not_evaluable_pair_count,
            "end_to_end_pair_count": self.end_to_end_pair_count,
            "end_to_end_satisfied_count": self.end_to_end_satisfied_count,
            "end_to_end_recall": self.end_to_end_recall.to_record(),
            "gate_reason_counts": dict(sorted(self.gate_reason_counts.items())),
            "decision_reason_counts": dict(sorted(self.decision_reason_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class InstantFusionEvaluation:
    """Fusion / venue-subject purity scoring for one venue instant."""

    evaluable_subject_count: int
    impure_subject_count: int
    persons_with_owned_match: int
    fragmented_person_count: int
    matched_tracklet_count: int
    matched_unowned_count: int
    matched_unowned_person_count: int
    integration_outcome_counts: Mapping[str, int]

    @property
    def fusion_impurity(self) -> Ratio:
        return Ratio.of(self.impure_subject_count, self.evaluable_subject_count)

    @property
    def fragmentation_rate(self) -> Ratio:
        return Ratio.of(self.fragmented_person_count, self.persons_with_owned_match)

    @property
    def matched_unowned_rate(self) -> Ratio:
        return Ratio.of(self.matched_unowned_count, self.matched_tracklet_count)

    def to_record(self) -> dict[str, object]:
        return {
            "evaluable_subject_count": self.evaluable_subject_count,
            "impure_subject_count": self.impure_subject_count,
            "fusion_impurity": self.fusion_impurity.to_record(),
            "persons_with_owned_match": self.persons_with_owned_match,
            "fragmented_person_count": self.fragmented_person_count,
            "fragmentation_rate": self.fragmentation_rate.to_record(),
            "matched_tracklet_count": self.matched_tracklet_count,
            "matched_unowned_count": self.matched_unowned_count,
            "matched_unowned_person_count": self.matched_unowned_person_count,
            "matched_unowned_rate": self.matched_unowned_rate.to_record(),
            "integration_outcome_counts": dict(sorted(self.integration_outcome_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class InstantOccupancyEvaluation:
    """Occupancy scoring for one venue instant."""

    ground_truth_observable_count: int
    raw_active_track_count: int
    fused_occupancy_count: int
    suppressed_duplicate_count: int

    @property
    def raw_error(self) -> int:
        return self.raw_active_track_count - self.ground_truth_observable_count

    @property
    def fused_error(self) -> int:
        return self.fused_occupancy_count - self.ground_truth_observable_count

    def to_record(self) -> dict[str, object]:
        return {
            "ground_truth_observable_count": self.ground_truth_observable_count,
            "raw_active_track_count": self.raw_active_track_count,
            "fused_occupancy_count": self.fused_occupancy_count,
            "suppressed_duplicate_count": self.suppressed_duplicate_count,
            "raw_error": self.raw_error,
            "fused_error": self.fused_error,
        }


@dataclass(frozen=True, slots=True)
class InstantEvaluation:
    """The full evaluation of one venue instant."""

    instant_index: int
    venue_seconds: float
    expected_venue_seconds: float
    cameras: tuple[CameraCoverage, ...]
    association: InstantAssociationEvaluation
    fusion: InstantFusionEvaluation
    occupancy: InstantOccupancyEvaluation
    #: Every successful camera-local GT<->runtime match distance for this instant,
    #: ascending, in integer micrometres. Camera-order independent by construction.
    match_distances_micrometres: tuple[int, ...] = ()

    @property
    def local(self) -> LocalCoverage:
        return LocalCoverage.from_cameras(self.cameras)

    @property
    def match_distance_stats(self) -> MatchDistanceStats:
        return MatchDistanceStats.from_micrometres(self.match_distances_micrometres)

    def to_record(self) -> dict[str, object]:
        return {
            "instant_index": self.instant_index,
            "venue_seconds": round(self.venue_seconds, 3),
            "expected_venue_seconds": round(self.expected_venue_seconds, 3),
            "cameras": [camera.to_record() for camera in self.cameras],
            "local": self.local.to_record(),
            "match_distance_stats": self.match_distance_stats.to_record(),
            "association": self.association.to_record(),
            "fusion": self.fusion.to_record(),
            "occupancy": self.occupancy.to_record(),
        }


@dataclass(frozen=True, slots=True)
class CameraCoverageSummary:
    """One camera's coverage aggregated across every scored instant."""

    camera_id: str
    metric_observation_count: int
    matched_count: int
    gt_person_count: int

    @property
    def runtime_coverage(self) -> Ratio:
        return Ratio.of(self.matched_count, self.metric_observation_count)

    @property
    def ground_truth_coverage(self) -> Ratio:
        return Ratio.of(self.matched_count, self.gt_person_count)

    def to_record(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "metric_observation_count": self.metric_observation_count,
            "matched_count": self.matched_count,
            "gt_person_count": self.gt_person_count,
            "runtime_coverage": self.runtime_coverage.to_record(),
            "ground_truth_coverage": self.ground_truth_coverage.to_record(),
        }


@dataclass(frozen=True, slots=True)
class OccupancyEvaluationSummary:
    """Occupancy error aggregated across every scored instant."""

    instant_count: int
    ground_truth_total: int
    raw_total: int
    fused_total: int
    raw_mae: float | None
    fused_mae: float | None
    raw_bias: float | None
    fused_bias: float | None
    raw_rmse: float | None
    fused_rmse: float | None
    exact_fused_count: int
    fused_over_count: int
    fused_under_count: int
    max_abs_raw_error: int
    max_abs_fused_error: int
    improved_count: int
    unchanged_count: int
    worsened_count: int
    total_suppressed_duplicates: int

    @property
    def exact_fused_fraction(self) -> Ratio:
        return Ratio.of(self.exact_fused_count, self.instant_count)

    @property
    def mean_suppressed_duplicates(self) -> float | None:
        if self.instant_count == 0:
            return None
        return self.total_suppressed_duplicates / self.instant_count

    @property
    def fused_minus_raw_mae(self) -> float | None:
        if self.fused_mae is None or self.raw_mae is None:
            return None
        return self.fused_mae - self.raw_mae

    def to_record(self) -> dict[str, object]:
        def opt(value: float | None) -> float | None:
            return None if value is None else _round(value)

        return {
            "instant_count": self.instant_count,
            "ground_truth_total": self.ground_truth_total,
            "raw_total": self.raw_total,
            "fused_total": self.fused_total,
            "raw_mae": opt(self.raw_mae),
            "fused_mae": opt(self.fused_mae),
            "raw_bias": opt(self.raw_bias),
            "fused_bias": opt(self.fused_bias),
            "raw_rmse": opt(self.raw_rmse),
            "fused_rmse": opt(self.fused_rmse),
            "exact_fused_count": self.exact_fused_count,
            "exact_fused_fraction": self.exact_fused_fraction.to_record(),
            "fused_over_count": self.fused_over_count,
            "fused_under_count": self.fused_under_count,
            "max_abs_raw_error": self.max_abs_raw_error,
            "max_abs_fused_error": self.max_abs_fused_error,
            "improved_count": self.improved_count,
            "unchanged_count": self.unchanged_count,
            "worsened_count": self.worsened_count,
            "total_suppressed_duplicates": self.total_suppressed_duplicates,
            "mean_suppressed_duplicates": opt(self.mean_suppressed_duplicates),
            "fused_minus_raw_mae": opt(self.fused_minus_raw_mae),
        }


@dataclass(frozen=True, slots=True)
class WildtrackEvaluationSummary:
    """The whole-run WILDTRACK evaluation summary."""

    schema_version: str
    selected_camera_ids: tuple[str, ...]
    instant_count: int
    max_ground_plane_distance_m: float
    camera_coverage: tuple[CameraCoverageSummary, ...]
    association: InstantAssociationEvaluation
    fusion: InstantFusionEvaluation
    occupancy: OccupancyEvaluationSummary
    #: Every successful camera-local GT<->runtime match distance across the whole
    #: run, ascending, in integer micrometres.
    match_distances_micrometres: tuple[int, ...] = ()

    @property
    def local(self) -> LocalCoverage:
        return LocalCoverage.from_cameras(self.camera_coverage)

    @property
    def match_distance_stats(self) -> MatchDistanceStats:
        return MatchDistanceStats.from_micrometres(self.match_distances_micrometres)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "selected_camera_ids": list(self.selected_camera_ids),
            "instant_count": self.instant_count,
            "max_ground_plane_distance_m": self.max_ground_plane_distance_m,
            "camera_coverage": [camera.to_record() for camera in self.camera_coverage],
            "local": self.local.to_record(),
            "match_distance_stats": self.match_distance_stats.to_record(),
            "association": self.association.to_record(),
            "fusion": self.fusion.to_record(),
            "occupancy": self.occupancy.to_record(),
        }


def rmse(errors: Sequence[int]) -> float | None:
    """Root-mean-square of an integer error sequence, or ``None`` when empty."""
    if not errors:
        return None
    return math.sqrt(sum(error * error for error in errors) / len(errors))


def mean(values: Sequence[int]) -> float | None:
    """Arithmetic mean of an integer sequence, or ``None`` when empty."""
    if not values:
        return None
    return sum(values) / len(values)

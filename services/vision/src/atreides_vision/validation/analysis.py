"""Turning one clip's run artifacts into its validation result.

Kept separate from the runner so the expensive half and the cheap half can be
performed independently. Processing a clip costs minutes of GPU time; reading
its streams back costs seconds. That split is what makes the manual annotation
workflow practical:

1. ``validation run`` processes the clips and writes the annotation templates.
2. A human fills the templates in, over however long that takes.
3. ``validation report`` re-reads the same streams and the now-filled templates
   and regenerates every artifact — without re-processing a frame.

Because both paths go through this module, a report regenerated after annotation
is identical to a freshly-run one except for the ground-truth section. There is
no second implementation of any metric.

Every stream is read exactly once, streaming, in this order:

1. **behaviour** → track presence, lifetimes, churn proxies, signal coverage,
   warm-up. Presence is retained afterwards, because
2. **interaction** needs it to attribute pair-episode endings to member tracks
   disappearing, and finally
3. **crowd** → cluster occupancy and composition stability.

Behaviour is read once and fanned out to two accumulators rather than read
twice: the file is the largest artifact a run produces, and decompressing it
twice would double the analysis cost for no gain.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from atreides_vision.errors import ValidationError
from atreides_vision.validation.behavior_metrics import BehaviorMetricsAccumulator
from atreides_vision.validation.crowd_metrics import CrowdMetricsAccumulator
from atreides_vision.validation.ground_truth import (
    load_manual_ground_truth,
    write_annotation_templates,
)
from atreides_vision.validation.interaction_metrics import InteractionMetricsAccumulator
from atreides_vision.validation.manifest import ValidationClip
from atreides_vision.validation.models import (
    ClipValidationResult,
    OutputVolumeMetrics,
    PerformanceMetrics,
    RunProvenance,
    StreamVolume,
)
from atreides_vision.validation.review_samples import ReviewSamplePlan, SampleFrame
from atreides_vision.validation.runner import ClipRunOutcome, stream_paths
from atreides_vision.validation.storage import ValidationLayout, read_json
from atreides_vision.validation.streams import (
    iter_jsonl,
    stream_size_bytes,
    uncompressed_size_bytes,
)
from atreides_vision.validation.track_metrics import TrackPresenceAccumulator

logger = logging.getLogger(__name__)

__all__ = ["analyze_clip", "analyze_run_directory", "load_clip_outcome"]


def analyze_clip(layout: ValidationLayout, outcome: ClipRunOutcome) -> ClipValidationResult:
    """Measure one clip's output streams and gather any manual annotation."""
    run_dir = layout.run_dir(outcome.clip.clip_id)
    gt_dir = layout.ground_truth_dir(outcome.clip.clip_id)
    # Checked up front, before any of the three passes: a run missing one stream
    # would otherwise fail partway through with a bare path, and the operator
    # needs to know it is the crowd stream that is absent, not merely a file.
    _require_streams(run_dir, outcome.clip.clip_id)

    frames_processed = outcome.performance.frames_processed
    source_fps = outcome.performance.source_fps
    processed_duration = outcome.performance.processed_duration_seconds

    # Behaviour: one pass, two consumers. Track presence is kept afterwards for
    # the interaction attribution below.
    presence = TrackPresenceAccumulator(frames_processed, source_fps)
    behavior = BehaviorMetricsAccumulator()
    behavior_records = 0
    for record in iter_jsonl(outcome.behavior_path):
        behavior_records += 1
        presence.consume((record,))
        behavior.consume((record,))
    track_metrics = presence.finalize()
    behavior_metrics = behavior.finalize()

    interaction = InteractionMetricsAccumulator(
        frames_processed=frames_processed,
        source_fps=source_fps,
        track_presence=presence.presence,
        # Frame indices are 1-based, so the last processed frame is exactly
        # frames_processed. Episodes still open there had no ending to explain.
        final_frame_index=frames_processed,
    )
    interaction_records = 0
    for record in iter_jsonl(outcome.interaction_path):
        interaction_records += 1
        interaction.consume((record,))
    interaction_metrics = interaction.finalize()

    crowd = CrowdMetricsAccumulator(
        frames_processed=frames_processed,
        source_fps=source_fps,
        clip_duration_seconds=processed_duration,
    )
    crowd_records = 0
    for record in iter_jsonl(outcome.crowd_path):
        crowd_records += 1
        crowd.consume((record,))
    crowd_metrics = crowd.finalize()

    volumes = _stream_volumes(
        run_dir,
        {
            "behavior": behavior_records,
            "interaction": interaction_records,
            "crowd": crowd_records,
        },
    )
    rendered_bytes = (
        outcome.rendered_video_path.stat().st_size if outcome.rendered_video_path.exists() else None
    )

    # Templates are written here rather than by the runner so that regenerating
    # a report also repairs a template that was deleted. Existing files are
    # never overwritten, so an annotator's work survives any number of re-runs.
    write_annotation_templates(
        gt_dir,
        outcome.clip.clip_id,
        outcome.sample_plan.frame_rows(),
        outcome.sample_plan.continuity_rows(),
    )
    ground_truth = load_manual_ground_truth(gt_dir, outcome.clip.clip_id)

    return ClipValidationResult(
        clip=outcome.clip,
        provenance=outcome.provenance,
        performance=outcome.performance,
        tracks=track_metrics,
        behavior=behavior_metrics,
        interaction=interaction_metrics,
        crowd=crowd_metrics,
        output_volume=OutputVolumeMetrics(streams=volumes, rendered_video_bytes=rendered_bytes),
        ground_truth=ground_truth,
        review_sample={
            "sample_frames": len(outcome.sample_plan.frames),
            "continuity_windows": len(outcome.sample_plan.windows),
            "frames_dir": layout.relative(gt_dir / "frames"),
            "contact_sheet": layout.relative(gt_dir / "contact_sheet.jpg"),
            "frame_annotations": layout.relative(gt_dir / "frame_annotations.csv"),
            "continuity_annotations": layout.relative(gt_dir / "continuity_annotations.csv"),
        },
    )


def analyze_run_directory(layout: ValidationLayout, clip_id: str) -> ClipValidationResult:
    """Re-measure a clip from artifacts an earlier run left on disk.

    This is what ``validation report`` uses. It rebuilds the run outcome from
    ``run.json`` rather than re-deriving it, so a regenerated report describes
    the configuration the run actually used, not the configuration this build
    would use today.
    """
    return analyze_clip(layout, load_clip_outcome(layout, clip_id))


def load_clip_outcome(layout: ValidationLayout, clip_id: str) -> ClipRunOutcome:
    """Reconstruct a :class:`ClipRunOutcome` from a completed run directory."""
    run_dir = layout.run_dir(clip_id)
    record = read_json(run_dir / "run.json")

    clip_record = _require_mapping(record, "clip", run_dir)
    provenance_record = _require_mapping(record, "provenance", run_dir)
    performance_record = _require_mapping(record, "performance", run_dir)
    plan_record = _require_mapping(record, "sample_plan", run_dir)
    counters = record.get("pipeline_counters")
    counters = counters if isinstance(counters, dict) else {}

    calibration = clip_record.get("calibration_path")
    clip = ValidationClip(
        clip_id=str(clip_record["clip_id"]),
        input_path=Path(str(clip_record["input_path"])),
        scene_type=str(clip_record["scene_type"]),
        camera_id=str(clip_record["camera_id"]),
        notes=None if clip_record.get("notes") is None else str(clip_record["notes"]),
        calibration_path=None if calibration is None else Path(str(calibration)),
    )

    streams = provenance_record.get("schema_versions")
    streams = streams if isinstance(streams, dict) else {}

    provenance = RunProvenance(
        detector=str(provenance_record["detector"]),
        detector_confidence=float(provenance_record["detector_confidence"]),
        tracker=str(provenance_record["tracker"]),
        tracker_high_conf_threshold=float(provenance_record["tracker_high_conf_threshold"]),
        tracker_minimum_iou_threshold=float(provenance_record["tracker_minimum_iou_threshold"]),
        tracker_lost_track_buffer=int(provenance_record["tracker_lost_track_buffer"]),
        tracker_minimum_consecutive_frames=int(
            provenance_record["tracker_minimum_consecutive_frames"]
        ),
        tracker_direction_consistency_weight=float(
            provenance_record["tracker_direction_consistency_weight"]
        ),
        tracker_delta_t=int(provenance_record["tracker_delta_t"]),
        deduplicate_detections=bool(provenance_record["deduplicate_detections"]),
        deduplication_rule=str(provenance_record["deduplication_rule"]),
        device=str(provenance_record["device"]),
        coordinate_space=str(provenance_record["coordinate_space"]),
        distance_unit=str(provenance_record["distance_unit"]),
        is_physical_space=bool(provenance_record["is_physical_space"]),
        camera_id=str(provenance_record["camera_id"]),
        input_path=str(provenance_record["input_path"]),
        calibration_path=(
            None
            if provenance_record.get("calibration_path") is None
            else str(provenance_record["calibration_path"])
        ),
        behavior_config=dict(provenance_record.get("behavior_config") or {}),
        interaction_config=dict(provenance_record.get("interaction_config") or {}),
        crowd_config=dict(provenance_record.get("crowd_config") or {}),
        output_sample_interval_seconds=(
            None
            if provenance_record.get("output_sample_interval_seconds") is None
            else float(provenance_record["output_sample_interval_seconds"])
        ),
        behavior_schema_version=str(streams.get("behavior", "unknown")),
        interaction_schema_version=str(streams.get("interaction", "unknown")),
        crowd_schema_version=str(streams.get("crowd", "unknown")),
        package_version=str(provenance_record.get("package_version", "unknown")),
        at_v021_defaults=bool(provenance_record["at_v021_defaults"]),
    )

    performance = PerformanceMetrics(
        source_frames=(
            None
            if performance_record.get("source_frames") is None
            else int(performance_record["source_frames"])
        ),
        source_fps=float(performance_record["source_fps"]),
        source_duration_seconds=(
            None
            if performance_record.get("source_duration_seconds") is None
            else float(performance_record["source_duration_seconds"])
        ),
        frames_processed=int(performance_record["frames_processed"]),
        wall_clock_seconds=float(performance_record["wall_clock_seconds"]),
        device=str(performance_record["device"]),
    )

    plan = ReviewSamplePlan(
        clip_id=clip.clip_id,
        frames_processed=int(plan_record.get("frames_processed", performance.frames_processed)),
        source_fps=float(plan_record.get("source_fps", performance.source_fps)),
        frames=tuple(
            SampleFrame(
                index=int(entry["index"]),
                frame_index=int(entry["frame_index"]),
                timestamp_seconds=float(entry["timestamp_seconds"]),
            )
            for entry in plan_record.get("sample_frames", [])
        ),
        windows=tuple(_window(entry) for entry in plan_record.get("continuity_windows", [])),
    )

    behavior_path, interaction_path, crowd_path = (path for _, path in stream_paths(run_dir))
    return ClipRunOutcome(
        clip=clip,
        provenance=provenance,
        performance=performance,
        rendered_video_path=run_dir / "tracked.mp4",
        behavior_path=behavior_path,
        interaction_path=interaction_path,
        crowd_path=crowd_path,
        sample_plan=plan,
        unique_track_ids=int(counters.get("unique_track_ids", 0)),
        detections_removed_by_deduplication=int(
            counters.get("detections_removed_by_deduplication", 0)
        ),
    )


def _window(entry: Any) -> Any:
    from atreides_vision.validation.review_samples import ContinuityWindow

    return ContinuityWindow(
        sequence_id=str(entry["sequence_id"]),
        start_seconds=float(entry["start_seconds"]),
        end_seconds=float(entry["end_seconds"]),
        start_frame_index=int(entry["start_frame_index"]),
        end_frame_index=int(entry["end_frame_index"]),
    )


def _require_mapping(record: dict[str, Any], key: str, run_dir: Path) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict):
        raise ValidationError(
            f"{run_dir / 'run.json'} is missing its '{key}' section. Re-run "
            f"'atreides-vision validation run' for this clip; a report cannot be regenerated "
            "from an incomplete run record without guessing at the configuration."
        )
    return value


def _require_streams(run_dir: Path, clip_id: str) -> None:
    """Fail before measuring anything if a stream this clip needs is absent."""
    for name, path in stream_paths(run_dir):
        if not path.exists():
            raise ValidationError(
                f"Clip '{clip_id}' has no {name} stream at {path}. Re-run "
                "'atreides-vision validation run' for this clip; a report built from a "
                "partial run would describe less footage than it claims to."
            )


def _stream_volumes(run_dir: Path, record_counts: dict[str, int]) -> tuple[StreamVolume, ...]:
    """Size each structured stream from the filesystem, never by decoding it."""
    volumes: list[StreamVolume] = []
    for name, path in stream_paths(run_dir):
        volumes.append(
            StreamVolume(
                name=name,
                path=path.name,
                compressed=path.suffix.lower() == ".gz",
                size_bytes=stream_size_bytes(path),
                uncompressed_size_bytes=uncompressed_size_bytes(path),
                record_count=record_counts.get(name, 0),
            )
        )
    return tuple(volumes)

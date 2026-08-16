"""The three validation artifacts: the JSON summary, the CSV, and the report.

All three are deterministic functions of the results. Same inputs, byte-identical
outputs — which is what makes ``diff`` between two validation runs a usable tool
rather than a wall of reordered dictionaries.

The division of labour between them:

* **JSON** is complete. Every metric, every distribution, every definition and
  every limitation, nested as the metric types define it.
* **CSV** is one row per clip, flattened to scalars, for a spreadsheet. Nested
  structures that would flatten awkwardly — histograms, per-stream volumes —
  stay in the JSON rather than becoming forty columns nobody reads.
* **Markdown** is for a person. It leads with what the numbers are *not*.

Three rules hold across all three, and are covered by tests:

**Proxy and ground truth never share a label.** Automated metrics are stated as
proxy/system metrics wherever they appear. Manual metrics appear only where a
human actually annotated, in their own section, and are absent entirely
otherwise.

**No physical units for uncalibrated footage.** Distance-bearing figures carry
the coordinate space's own unit name, which for these clips is
``normalized_frame_unit``. No metre, m/s or subjects/m² figure is derived.

**No verdict.** There is no pass, no fail, no grade and no threshold band. Where
scenes differ, the difference is stated as a ratio between measurements, never
as a judgement about either.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from atreides_vision.behavior.models import SignalType
from atreides_vision.validation.crowd_metrics import PERSISTENCE_SURVIVAL_THRESHOLDS_SECONDS
from atreides_vision.validation.ground_truth import GroundTruthStatus
from atreides_vision.validation.models import ClipValidationResult, ValidationSummary
from atreides_vision.validation.storage import (
    ValidationLayout,
    write_csv,
    write_json,
    write_text,
)
from atreides_vision.validation.track_metrics import SHORT_TRACK_THRESHOLDS_SECONDS

__all__ = [
    "SUMMARY_CSV_COLUMNS",
    "build_summary",
    "render_report",
    "summary_csv_rows",
    "write_validation_artifacts",
]

_NA = "n/a"


@dataclass(frozen=True, slots=True)
class _Row:
    """One comparison-table line: a label, a unit, and one cell per clip."""

    label: str
    values: tuple[str, ...]


def build_summary(
    results: Sequence[ClipValidationResult], manifest_path: str | None = None
) -> ValidationSummary:
    """Assemble the machine-readable summary, in the results' own order."""
    return ValidationSummary(results=tuple(results), manifest_path=manifest_path)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

SUMMARY_CSV_COLUMNS: tuple[str, ...] = (
    # Identity and configuration
    "clip_id",
    "scene_type",
    "camera_id",
    "coordinate_space",
    "distance_unit",
    "is_physical_space",
    "at_v021_defaults",
    # Performance
    "source_frames",
    "source_fps",
    "source_duration_seconds",
    "frames_processed",
    "wall_clock_seconds",
    "processed_fps",
    "realtime_factor",
    "device",
    # Tracking proxies
    "proxy_unique_track_ids",
    "proxy_mean_active_tracks",
    "proxy_median_active_tracks",
    "proxy_max_active_tracks",
    "proxy_mean_track_lifetime_s",
    "proxy_median_track_lifetime_s",
    "proxy_p10_track_lifetime_s",
    "proxy_p90_track_lifetime_s",
    "proxy_min_track_lifetime_s",
    "proxy_max_track_lifetime_s",
    "proxy_tracks_under_0.5s_fraction",
    "proxy_tracks_under_1s_fraction",
    "proxy_tracks_under_2s_fraction",
    "proxy_track_births_per_minute",
    "proxy_unique_track_ids_per_minute",
    "proxy_tracks_with_gaps",
    "proxy_gap_count",
    "proxy_median_gap_s",
    "proxy_max_gap_s",
    # Behaviour coverage
    "behavior_observations",
    "coverage_speed",
    "coverage_acceleration",
    "coverage_heading_change_rate",
    "coverage_stationary_duration",
    "warmup_median_speed_s",
    "warmup_median_acceleration_s",
    "warmup_median_heading_change_rate_s",
    # Interaction
    "proxy_mean_pairs_per_frame",
    "proxy_median_pairs_per_frame",
    "proxy_max_pairs_per_frame",
    "proxy_unique_pair_references",
    "proxy_pair_episodes",
    "proxy_pair_births_per_minute",
    "proxy_median_pair_episode_s",
    "proxy_p10_pair_episode_s",
    "proxy_p90_pair_episode_s",
    "proxy_short_pair_episode_fraction",
    "proxy_pair_episodes_ended_before_clip_end",
    "proxy_pair_episodes_ending_with_member_track_end",
    "proxy_pair_member_track_end_fraction",
    # Crowd
    "proxy_mean_clusters_per_frame",
    "proxy_max_clusters_per_frame",
    "proxy_mean_clustered_subjects_per_frame",
    "proxy_mean_unclustered_subjects_per_frame",
    "proxy_largest_cluster_size",
    "proxy_unique_compositions",
    "proxy_composition_episodes",
    "proxy_composition_births_per_minute",
    "proxy_median_composition_persistence_s",
    "proxy_p90_composition_persistence_s",
    "proxy_composition_survival_over_1s",
    "proxy_composition_survival_over_2s",
    "proxy_composition_survival_over_5s",
    # Output volume
    "behavior_bytes",
    "interaction_bytes",
    "crowd_bytes",
    "total_structured_bytes",
    "structured_mib_per_source_minute",
    "behavior_records_per_source_second",
    "interaction_records_per_source_second",
    "crowd_records_per_source_second",
    "rendered_video_bytes",
    # Manual ground truth — always last, always its own block
    "manual_ground_truth_status",
    "manual_annotated_frames",
    "manual_detection_recall_sample",
    "manual_precision_sample",
    "manual_annotated_sequences",
    "manual_id_switches",
    "manual_fragments",
    "manual_lost_track_events",
)


def summary_csv_rows(results: Sequence[ClipValidationResult]) -> list[dict[str, Any]]:
    """One flat row per clip, in results order."""
    return [_csv_row(result) for result in results]


def _csv_row(result: ClipValidationResult) -> dict[str, Any]:
    tracks = result.tracks
    behavior = result.behavior
    interaction = result.interaction
    crowd = result.crowd
    performance = result.performance
    processed = performance.processed_duration_seconds
    ground_truth = result.ground_truth

    volumes = {stream.name: stream for stream in result.output_volume.streams}
    survival = crowd.survival_fractions
    short = tracks.short_track_fractions

    row: dict[str, Any] = {
        "clip_id": result.clip.clip_id,
        "scene_type": result.clip.scene_type,
        "camera_id": result.clip.camera_id,
        "coordinate_space": result.provenance.coordinate_space,
        "distance_unit": result.provenance.distance_unit,
        "is_physical_space": result.provenance.is_physical_space,
        "at_v021_defaults": result.provenance.at_v021_defaults,
        "source_frames": performance.source_frames,
        "source_fps": _r(performance.source_fps),
        "source_duration_seconds": _r(performance.source_duration_seconds),
        "frames_processed": performance.frames_processed,
        "wall_clock_seconds": _r(performance.wall_clock_seconds),
        "processed_fps": _r(performance.processed_fps),
        "realtime_factor": _r(performance.realtime_factor),
        "device": performance.device,
        "proxy_unique_track_ids": tracks.unique_track_count,
        "proxy_mean_active_tracks": _r(tracks.active_tracks.mean),
        "proxy_median_active_tracks": _r(tracks.active_tracks.median),
        "proxy_max_active_tracks": _r(tracks.active_tracks.maximum),
        "proxy_mean_track_lifetime_s": _r(tracks.lifetime_seconds.mean),
        "proxy_median_track_lifetime_s": _r(tracks.lifetime_seconds.median),
        "proxy_p10_track_lifetime_s": _r(tracks.lifetime_seconds.p10),
        "proxy_p90_track_lifetime_s": _r(tracks.lifetime_seconds.p90),
        "proxy_min_track_lifetime_s": _r(tracks.lifetime_seconds.minimum),
        "proxy_max_track_lifetime_s": _r(tracks.lifetime_seconds.maximum),
        "proxy_track_births_per_minute": _r(tracks.track_births_per_minute),
        "proxy_unique_track_ids_per_minute": _r(tracks.unique_track_ids_per_minute),
        "proxy_tracks_with_gaps": tracks.tracks_with_gaps,
        "proxy_gap_count": tracks.gap_count,
        "proxy_median_gap_s": _r(tracks.gap_seconds.median),
        "proxy_max_gap_s": _r(tracks.gap_seconds.maximum),
        "behavior_observations": behavior.total_observations,
        "warmup_median_speed_s": _r(_warm_up(result, SignalType.SPEED)),
        "warmup_median_acceleration_s": _r(_warm_up(result, SignalType.ACCELERATION)),
        "warmup_median_heading_change_rate_s": _r(_warm_up(result, SignalType.HEADING_CHANGE_RATE)),
        "proxy_mean_pairs_per_frame": _r(interaction.pairs_per_frame.mean),
        "proxy_median_pairs_per_frame": _r(interaction.pairs_per_frame.median),
        "proxy_max_pairs_per_frame": _r(interaction.pairs_per_frame.maximum),
        "proxy_unique_pair_references": interaction.unique_pair_references,
        "proxy_pair_episodes": interaction.pair_episodes,
        "proxy_pair_births_per_minute": _r(interaction.pair_births_per_minute),
        "proxy_median_pair_episode_s": _r(interaction.episode_seconds.median),
        "proxy_p10_pair_episode_s": _r(interaction.episode_seconds.p10),
        "proxy_p90_pair_episode_s": _r(interaction.episode_seconds.p90),
        "proxy_short_pair_episode_fraction": _r(interaction.short_episode_fraction),
        "proxy_pair_episodes_ended_before_clip_end": (interaction.episodes_ended_before_clip_end),
        "proxy_pair_episodes_ending_with_member_track_end": (
            interaction.episodes_ending_with_member_track
        ),
        "proxy_pair_member_track_end_fraction": _r(interaction.member_track_end_fraction),
        "proxy_mean_clusters_per_frame": _r(crowd.clusters_per_frame.mean),
        "proxy_max_clusters_per_frame": _r(crowd.clusters_per_frame.maximum),
        "proxy_mean_clustered_subjects_per_frame": _r(crowd.clustered_subjects_per_frame.mean),
        "proxy_mean_unclustered_subjects_per_frame": _r(crowd.unclustered_subjects_per_frame.mean),
        "proxy_largest_cluster_size": crowd.largest_cluster_size,
        "proxy_unique_compositions": crowd.unique_compositions,
        "proxy_composition_episodes": crowd.composition_episodes,
        "proxy_composition_births_per_minute": _r(crowd.composition_births_per_minute),
        "proxy_median_composition_persistence_s": _r(crowd.persistence_seconds.median),
        "proxy_p90_composition_persistence_s": _r(crowd.persistence_seconds.p90),
        "rendered_video_bytes": result.output_volume.rendered_video_bytes,
        "total_structured_bytes": result.output_volume.total_bytes,
        "structured_mib_per_source_minute": _r(
            result.output_volume.megabytes_per_source_minute(processed)
        ),
        "manual_ground_truth_status": ground_truth.status.value,
    }

    for threshold in SHORT_TRACK_THRESHOLDS_SECONDS:
        row[f"proxy_tracks_under_{threshold:g}s_fraction"] = _r(short.get(threshold))
    for signal in SignalType:
        row[f"coverage_{signal.value}"] = _r(behavior.coverage(signal.value))
    for threshold in PERSISTENCE_SURVIVAL_THRESHOLDS_SECONDS:
        row[f"proxy_composition_survival_over_{threshold:g}s"] = _r(survival.get(threshold))
    for name in ("behavior", "interaction", "crowd"):
        stream = volumes.get(name)
        row[f"{name}_bytes"] = None if stream is None else stream.size_bytes
        row[f"{name}_records_per_source_second"] = (
            None if stream is None else _r(stream.records_per_source_second(processed))
        )

    # Manual metrics are emitted only where a human actually annotated. An
    # un-annotated clip leaves these cells empty rather than zero, so nobody can
    # chart a fabricated recall of 0.0 against a real one.
    if ground_truth.frames:
        row["manual_annotated_frames"] = len(ground_truth.frames)
        row["manual_detection_recall_sample"] = _r(ground_truth.detection_recall_sample)
        row["manual_precision_sample"] = _r(ground_truth.detection_precision_sample)
    if ground_truth.continuity:
        row["manual_annotated_sequences"] = len(ground_truth.continuity)
        row["manual_id_switches"] = ground_truth.total_id_switches
        row["manual_fragments"] = ground_truth.total_fragments
        row["manual_lost_track_events"] = ground_truth.total_lost_track_events
    return row


def _warm_up(result: ClipValidationResult, signal: SignalType) -> float | None:
    distribution = result.behavior.warm_up_seconds.get(signal.value)
    return None if distribution is None else distribution.median


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def render_report(summary: ValidationSummary, layout: ValidationLayout) -> str:
    """Render the human-readable V0.3.6 report."""
    results = summary.results
    lines: list[str] = []
    add = lines.append

    add("# V0.3.6 — Real-World Validation")
    add("")
    add(
        "**Status:** automated system / proxy validation **complete** · manual ground truth "
        "**pending** · physical metric validation **pending** · production validation "
        "**not established** · final signal-stack acceptance **pending**"
    )
    add("")
    add(
        "This report measures the V0.3 stack on real footage. It changes no detector or "
        "tracker setting, adds no signal, and reaches no pass/fail conclusion — there is no "
        "validated threshold to compare any of these numbers against, and inventing one "
        "would be the least defensible thing in the document."
    )
    add("")
    add(
        "**It does not establish that the V0.3 signal stack is sufficient, accurate against "
        "reality, or valid in production.** It establishes what the stack *does* on four real "
        "clips, measured against itself. See the validation-status table at the end for what "
        "each remaining review would add."
    )
    add("")

    _read_this_first(add, summary)
    _scope(add)
    _inputs(add, results, summary)
    _provenance_section(add, results)
    _performance_section(add, results)
    _tracking_section(add, results)
    _behavior_section(add, results)
    _interaction_section(add, results)
    _crowd_section(add, results)
    _output_volume_section(add, results)
    _comparison_section(add, results)
    _ground_truth_section(add, results, layout)
    _limitations_section(add, results)
    _freeze_section(add)

    return "\n".join(lines).rstrip() + "\n"


def _read_this_first(add: Any, summary: ValidationSummary) -> None:
    add("## Read this first — what these numbers are")
    add("")
    add(
        "Every automated metric in this report is a **proxy / system metric**: it is derived "
        "from Atreides' own output streams, with no human annotation anywhere in the "
        "derivation. Specifically:"
    )
    add("")
    add(
        "- **`track_births_per_minute` is not an ID-switch rate.** A track birth is the first "
        "appearance of a `(camera_id, track_id)`. It may be a person entering the view, a "
        "re-detection after occlusion, tracking fragmentation, or a tracker lifecycle event. "
        "Nothing in the output stream distinguishes these."
    )
    add(
        "- **Short-track fractions are not proof of fragmentation.** They are consistent with "
        "fragmentation and equally consistent with somebody genuinely crossing a corner of "
        "the frame."
    )
    add(
        "- **Nothing here is detection recall.** Recall requires a human counting the people "
        "actually visible in a frame. Where that has not been done, this report says so and "
        "reports no recall figure at all."
    )
    add(
        "- **Unique track IDs are an upper bound on people, never a census.** One person may "
        "hold several track IDs across a clip."
    )
    add("")
    if not summary.any_physical_space:
        add(
            "**No physical units appear in this report.** None of the validation footage has a "
            "measured floor calibration, so every clip was processed in normalized image "
            "units. No metre, m/s or subjects/m² figure is derived, and no floor dimension is "
            "estimated from anything visible in the footage. Distances below are in "
            "`normalized_frame_unit` — camera-local frame fractions, not physical distance, "
            "and not comparable between cameras."
        )
        add("")
    if not summary.all_at_v021_defaults:
        add(
            "> **Warning:** not every clip ran under the V0.2.1 defaults. The scene comparison "
            "below is not valid across clips with different configurations — see the "
            "provenance table."
        )
        add("")


def _scope(add: Any) -> None:
    add("## Scope")
    add("")
    add("**In scope:** measuring the existing V0.3 stack on real footage, reproducibly.")
    add("")
    add(
        "**Explicitly out of scope:** any detector or tracker tuning, any new signal, any risk "
        "score, threat score, aggression/fight/intoxication/panic detection, crowd-danger "
        "classification, operator alerting, decision engine, LLM integration, ReID, appearance "
        "embeddings, face recognition, or cross-camera fusion. None of these exist in the "
        "codebase and none were added by this milestone."
    )
    add("")


def _inputs(add: Any, results: Sequence[ClipValidationResult], summary: ValidationSummary) -> None:
    add("## Input clips")
    add("")
    add("| clip_id | scene type | camera_id | resolution source | frames | source FPS | duration |")
    add("| --- | --- | --- | --- | ---: | ---: | ---: |")
    for result in results:
        performance = result.performance
        add(
            f"| `{result.clip.clip_id}` | {result.clip.scene_type} | "
            f"`{result.clip.camera_id}` | `{result.clip.input_path.name}` | "
            f"{performance.frames_processed} | {_fmt(performance.source_fps, 2)} | "
            f"{_fmt(performance.source_duration_seconds, 1)} s |"
        )
    add("")
    notes = [result for result in results if result.clip.notes]
    if notes:
        for result in notes:
            add(f"- `{result.clip.clip_id}` — {result.clip.notes}")
        add("")
    shared = summary.shared_camera_ids
    if shared:
        add("### Clips sharing a camera")
        add("")
        add(
            "One physical camera legitimately supplies many independent clips — different "
            "times, lighting, densities, days or test scenarios — so `camera_id` is **not** "
            "required to be unique across clips. Here it repeats:"
        )
        add("")
        for camera in shared:
            clip_ids = ", ".join(f"`{clip_id}`" for clip_id in summary.clip_ids_for_camera(camera))
            add(f"- `{camera}` — {clip_ids}")
        add("")
        add(
            "**A shared `camera_id` asserts only that the footage came from the same camera.** "
            "It creates no relationship between the clips' tracks. Each clip was processed by "
            "its own detector, tracker and analyzers, written to its own streams, and measured "
            "into its own result; no track id, history or metric crosses a clip boundary. "
            "`camera_01/clip_A/track_7` and `camera_01/clip_B/track_7` are two independent "
            "tracklets that happen to share an integer."
        )
        add("")
        add(
            "The consequence for a reader: **per-clip counts from clips sharing a camera must "
            "not be added together.** Two clips of one camera reporting 40 unique track ids "
            "each are not 80 of anything."
        )
        add("")
    else:
        add(
            "Each clip here has its own `camera_id`, though that is not required — one camera "
            "may supply many clips. Either way, nothing in this report combines counts or "
            "identities across clips."
        )
        add("")


def _provenance_section(add: Any, results: Sequence[ClipValidationResult]) -> None:
    add("## Runtime configuration (frozen)")
    add("")
    first = results[0].provenance if results else None
    if first is not None:
        add("```text")
        add(f"detector                       {first.detector}")
        add(f"detector confidence            {first.detector_confidence:.2f}")
        add(f"tracker                        {first.tracker}")
        add(f"  high-confidence threshold    {first.tracker_high_conf_threshold:.2f}")
        add(f"  minimum IoU threshold        {first.tracker_minimum_iou_threshold:.2f}")
        add(f"  lost track buffer            {first.tracker_lost_track_buffer}")
        add(f"  minimum consecutive frames   {first.tracker_minimum_consecutive_frames}")
        add(f"  direction consistency weight {first.tracker_direction_consistency_weight}")
        add(f"  delta_t                      {first.tracker_delta_t}")
        add(f"detection deduplication        {first.deduplication_rule}")
        add(f"device                         {first.device}")
        add(f"coordinate space               {first.coordinate_space} ({first.distance_unit})")
        add(f"output sampling                {first.output_sample_interval_seconds or 'none'}")
        add(
            f"schema versions                behaviour {first.behavior_schema_version} · "
            f"interaction {first.interaction_schema_version} · "
            f"crowd {first.crowd_schema_version}"
        )
        add(f"package version                {first.package_version}")
        add("```")
        add("")
    add(
        "**Output is unsampled.** Every analysed frame is written to all three streams. "
        "Track lifetimes, observation gaps, pair episodes and cluster-composition persistence "
        "are temporal metrics; a sampled stream cannot support any of them."
    )
    add("")
    add("Per-clip provenance, including every analysis threshold, is in `validation_summary.json`.")
    add("")


def _performance_section(add: Any, results: Sequence[ClipValidationResult]) -> None:
    add("## Local processing performance")
    add("")
    add(
        "**These are machine-specific development measurements.** They include annotated-video "
        "encoding and were taken on one development machine with one device backend. They do "
        "not establish production real-time capability."
    )
    add("")
    add("| clip | frames | source FPS | wall clock | processed FPS | realtime factor | device |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for result in results:
        performance = result.performance
        add(
            f"| `{result.clip.clip_id}` | {performance.frames_processed} | "
            f"{_fmt(performance.source_fps, 2)} | {_fmt(performance.wall_clock_seconds, 1)} s | "
            f"{_fmt(performance.processed_fps, 2)} | {_fmt(performance.realtime_factor, 3)}x | "
            f"`{performance.device}` |"
        )
    add("")
    add(
        "`realtime factor = processed FPS / source FPS`. Above 1.0 means the clip was processed "
        "faster than it plays."
    )
    add("")


def _tracking_section(add: Any, results: Sequence[ClipValidationResult]) -> None:
    add("## Tracking stability — proxy / system metrics")
    add("")
    add(
        "Reconstructed from the unsampled behaviour stream by grouping observations on "
        "`(camera_id, track_id)`. A track ID is a **tracklet**, not a person."
    )
    add("")
    add("### Track lifetime and occupancy")
    add("")
    header = "| metric | " + " | ".join(f"`{r.clip.clip_id}`" for r in results) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in results) + " |"
    add(header)
    add(divider)
    for row in _tracking_rows(results):
        add(f"| {row.label} | " + " | ".join(row.values) + " |")
    add("")
    add(
        "`track lifetime` is the span from a track's first to its last observation. A track "
        "seen on exactly one frame has a lifetime of 0.0 s — it is not padded to a frame "
        "period, because padding would understate the short-track fractions."
    )
    add("")
    add(
        "**`track births / min` is sensitive to clip length**, and these clips differ by more "
        "than 3x in duration. Every subject already visible on the first frame is a birth, so "
        "that fixed acquisition cost is amortised over less time on a short clip than on a "
        "long one. Compare the rates between clips of similar length with more confidence "
        "than across them."
    )
    add("")
    add("### Observation gaps")
    add("")
    add(
        "A gap is the same track ID being reported, absent for one or more frames, then "
        "reported again. **A track lost and re-acquired under a new ID leaves no gap here** — "
        "it leaves two tracks, and nothing in this harness stitches them together."
    )
    add("")
    add(header)
    add(divider)
    for row in _gap_rows(results):
        add(f"| {row.label} | " + " | ".join(row.values) + " |")
    add("")


def _tracking_rows(results: Sequence[ClipValidationResult]) -> list[_Row]:
    rows = [
        _Row("unique track IDs", tuple(str(r.tracks.unique_track_count) for r in results)),
        _Row(
            "mean active tracks / frame",
            tuple(_fmt(r.tracks.active_tracks.mean, 2) for r in results),
        ),
        _Row(
            "median active tracks / frame",
            tuple(_fmt(r.tracks.active_tracks.median, 2) for r in results),
        ),
        _Row("max active tracks", tuple(str(r.tracks.max_active_tracks) for r in results)),
        _Row(
            "mean track lifetime (s)",
            tuple(_fmt(r.tracks.lifetime_seconds.mean, 2) for r in results),
        ),
        _Row(
            "median track lifetime (s)",
            tuple(_fmt(r.tracks.lifetime_seconds.median, 2) for r in results),
        ),
        _Row("P10 lifetime (s)", tuple(_fmt(r.tracks.lifetime_seconds.p10, 2) for r in results)),
        _Row("P90 lifetime (s)", tuple(_fmt(r.tracks.lifetime_seconds.p90, 2) for r in results)),
        _Row(
            "shortest lifetime (s)",
            tuple(_fmt(r.tracks.lifetime_seconds.minimum, 2) for r in results),
        ),
        _Row(
            "longest lifetime (s)",
            tuple(_fmt(r.tracks.lifetime_seconds.maximum, 2) for r in results),
        ),
    ]
    for threshold in SHORT_TRACK_THRESHOLDS_SECONDS:
        rows.append(
            _Row(
                f"tracks < {threshold:g} s",
                tuple(_pct(r.tracks.short_track_fractions.get(threshold)) for r in results),
            )
        )
    rows.append(
        _Row(
            "track births / min *(not ID switches)*",
            tuple(_fmt(r.tracks.track_births_per_minute, 1) for r in results),
        )
    )
    rows.append(
        _Row(
            "unique track IDs / min",
            tuple(_fmt(r.tracks.unique_track_ids_per_minute, 1) for r in results),
        )
    )
    return rows


def _gap_rows(results: Sequence[ClipValidationResult]) -> list[_Row]:
    return [
        _Row("tracks with ≥1 gap", tuple(str(r.tracks.tracks_with_gaps) for r in results)),
        _Row("total gaps", tuple(str(r.tracks.gap_count) for r in results)),
        _Row("median gap (s)", tuple(_fmt(r.tracks.gap_seconds.median, 3) for r in results)),
        _Row("max gap (s)", tuple(_fmt(r.tracks.gap_seconds.maximum, 3) for r in results)),
    ]


def _behavior_section(add: Any, results: Sequence[ClipValidationResult]) -> None:
    add("## Behaviour signal coverage — proxy / system metrics")
    add("")
    add(
        "`coverage = observations containing the signal / total subject observations`. The "
        "denominator includes observations with no signal at all, which is the case coverage "
        "exists to quantify."
    )
    add("")
    add(
        "**An omitted signal means *not currently measurable*, never zero.** A subject standing "
        "still still reports a speed of 0.0; an absent speed is an absence of measurement."
    )
    add("")
    header = "| signal | " + " | ".join(f"`{r.clip.clip_id}`" for r in results) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in results) + " |"
    add(header)
    add(divider)
    add(
        "| subject observations | "
        + " | ".join(f"{r.behavior.total_observations:,}" for r in results)
        + " |"
    )
    for signal in SignalType:
        add(
            f"| {signal.value} coverage | "
            + " | ".join(_pct(r.behavior.coverage(signal.value)) for r in results)
            + " |"
        )
    add("")
    add("### Warm-up — time from track birth to first measurement")
    add("")
    add(
        "Median seconds from a track's first observation to its first record carrying that "
        "signal, over tracks that ever produced it. **Track fragmentation restarts this clock**, "
        "which is the mechanism by which upstream churn suppresses downstream signal "
        "availability."
    )
    add("")
    add(header)
    add(divider)
    for signal in (
        SignalType.SPEED,
        SignalType.ACCELERATION,
        SignalType.HEADING_CHANGE_RATE,
    ):
        add(
            f"| {signal.value} (s) | "
            + " | ".join(_fmt(_warm_up(r, signal), 2) for r in results)
            + " |"
        )
        add(
            f"| tracks never producing {signal.value} | "
            + " | ".join(
                _ratio(
                    r.behavior.tracks_never_producing.get(signal.value),
                    r.behavior.tracks_observed,
                )
                for r in results
            )
            + " |"
        )
    add("")
    add(
        "**`SubjectMotion` / usable motion heading is not reported.** It is computed in memory "
        "and consumed in process, but behaviour schema 0.3.3 does not serialise it, so there "
        "is no honest per-subject motion-heading coverage to derive from the streams. The "
        "nearest available observables are `heading_change_rate` coverage above and the "
        "pairwise `relative_motion_heading` availability below."
    )
    add("")


def _interaction_section(add: Any, results: Sequence[ClipValidationResult]) -> None:
    add("## Interaction (pairwise geometry) — proxy / system metrics")
    add("")
    add(
        "**These are geometric co-presence records.** Interaction frequency is not social "
        "interaction, and no record describes a relationship. A pair is two tracklets."
    )
    add("")
    add(
        "A **pair reference** is a canonical `(camera_id, low_track_id, high_track_id)`. A "
        "**pair episode** is a maximal run of consecutive frames in which that pair was "
        "measured; one pair reference can have several episodes."
    )
    add("")
    if _all_ungated(results):
        add(
            "> **Read the pair figures with the gate in mind.** These runs have "
            "`max_pair_distance = null`, so **every co-present pair of tracks is measured, at "
            "any separation**. Two consequences follow, and neither is a property of the "
            "people in the footage:"
        )
        add(">")
        add(
            "> - Pairs per frame is `N(N-1)/2` in the number of simultaneously tracked "
            "subjects. It is a combinatorial count, not a count of encounters."
        )
        add(
            "> - **A pair episode cannot end because two subjects separated** — there is no "
            "distance at which measurement stops. An episode ends only when one member is not "
            "observed on the next frame. Pair-episode churn on these clips therefore measures "
            "*observation continuity*, and nothing else."
        )
        add("")
    header = "| metric | " + " | ".join(f"`{r.clip.clip_id}`" for r in results) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in results) + " |"
    add(header)
    add(divider)
    rows = [
        _Row(
            "mean pairs / frame",
            tuple(_fmt(r.interaction.pairs_per_frame.mean, 2) for r in results),
        ),
        _Row(
            "median pairs / frame",
            tuple(_fmt(r.interaction.pairs_per_frame.median, 2) for r in results),
        ),
        _Row(
            "max pairs / frame",
            tuple(_fmt(r.interaction.pairs_per_frame.maximum, 0) for r in results),
        ),
        _Row(
            "unique pair references",
            tuple(f"{r.interaction.unique_pair_references:,}" for r in results),
        ),
        _Row("pair episodes", tuple(f"{r.interaction.pair_episodes:,}" for r in results)),
        _Row(
            "pair births / min *(episodes started)*",
            tuple(_fmt(r.interaction.pair_births_per_minute, 1) for r in results),
        ),
        _Row(
            "median pair episode (s)",
            tuple(_fmt(r.interaction.episode_seconds.median, 2) for r in results),
        ),
        _Row(
            "P10 pair episode (s)",
            tuple(_fmt(r.interaction.episode_seconds.p10, 2) for r in results),
        ),
        _Row(
            "P90 pair episode (s)",
            tuple(_fmt(r.interaction.episode_seconds.p90, 2) for r in results),
        ),
        _Row(
            "episodes < 0.5 s",
            tuple(_pct(r.interaction.short_episode_fraction) for r in results),
        ),
    ]
    for row in rows:
        add(f"| {row.label} | " + " | ".join(row.values) + " |")
    add("")
    add("### Track churn propagating into pair history")
    add("")
    add(
        "Pair episodes whose final frame is also the final frame one of their two member track "
        "IDs was ever observed on — **a member track stopped existing for the rest of the "
        "clip**. Episodes still open when the clip ended are excluded, because they have no "
        "ending to attribute."
    )
    add("")
    add(header)
    add(divider)
    add(
        "| episodes that ended before clip end | "
        + " | ".join(f"{r.interaction.episodes_ended_before_clip_end:,}" for r in results)
        + " |"
    )
    add(
        "| ...ending with a member track's disappearance | "
        + " | ".join(f"{r.interaction.episodes_ending_with_member_track:,}" for r in results)
        + " |"
    )
    add(
        "| ...as a fraction | "
        + " | ".join(_pct(r.interaction.member_track_end_fraction) for r in results)
        + " |"
    )
    add("")
    if _all_ungated(results):
        add(
            "The remainder ended for the only other reason available with no distance gate: "
            "a member was **momentarily** unobserved and later returned under the same track "
            "ID. Those are the observation gaps counted in the tracking section — the same "
            "mechanism, seen one layer downstream."
        )
        add("")
    add(
        "No identity stitching is attempted or implied: knowing an episode ended because a "
        "track vanished says nothing about which later track, if any, is the same person."
    )
    add("")


def _all_ungated(results: Sequence[ClipValidationResult]) -> bool:
    """Whether every clip ran with no pairwise distance gate.

    Read from provenance rather than assumed, because it changes what a pair
    episode's *ending* can possibly mean — and therefore what the churn figures
    describe.
    """
    return all(
        result.provenance.interaction_config.get("max_pair_distance") is None for result in results
    )


def _crowd_section(add: Any, results: Sequence[ClipValidationResult]) -> None:
    add("## Spatial clusters — proxy / system metrics")
    add("")
    add(
        "**A spatial cluster is a geometric arrangement at an instant, not a social group.** "
        "`member_count` counts tracklets, not unique people."
    )
    add("")
    add(
        "V0.3.5 addresses a cluster by its **exact member set**, so a **composition** is a "
        "canonical member set and a **composition episode** is a maximal run of consecutive "
        "frames reporting that exact set. Persistence is the analyser's own "
        "`persistence_duration` at the end of each episode, not a validation-only "
        "recomputation."
    )
    add("")
    header = "| metric | " + " | ".join(f"`{r.clip.clip_id}`" for r in results) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in results) + " |"
    add(header)
    add(divider)
    rows = [
        _Row(
            "mean clusters / frame",
            tuple(_fmt(r.crowd.clusters_per_frame.mean, 3) for r in results),
        ),
        _Row(
            "max clusters / frame",
            tuple(_fmt(r.crowd.clusters_per_frame.maximum, 0) for r in results),
        ),
        _Row(
            "mean clustered subjects / frame",
            tuple(_fmt(r.crowd.clustered_subjects_per_frame.mean, 2) for r in results),
        ),
        _Row(
            "mean unclustered subjects / frame",
            tuple(_fmt(r.crowd.unclustered_subjects_per_frame.mean, 2) for r in results),
        ),
        _Row(
            "largest observed cluster",
            tuple(str(r.crowd.largest_cluster_size) for r in results),
        ),
        _Row(
            "unique compositions",
            tuple(f"{r.crowd.unique_compositions:,}" for r in results),
        ),
        _Row(
            "composition episodes",
            tuple(f"{r.crowd.composition_episodes:,}" for r in results),
        ),
        _Row(
            "composition births / min",
            tuple(_fmt(r.crowd.composition_births_per_minute, 1) for r in results),
        ),
        _Row(
            "median composition persistence (s)",
            tuple(_fmt(r.crowd.persistence_seconds.median, 2) for r in results),
        ),
        _Row(
            "P90 composition persistence (s)",
            tuple(_fmt(r.crowd.persistence_seconds.p90, 2) for r in results),
        ),
    ]
    for threshold in PERSISTENCE_SURVIVAL_THRESHOLDS_SECONDS:
        rows.append(
            _Row(
                f"compositions surviving > {threshold:g} s",
                tuple(_pct(r.crowd.survival_fractions.get(threshold)) for r in results),
            )
        )
    for row in rows:
        add(f"| {row.label} | " + " | ".join(row.values) + " |")
    add("")
    add("### Cluster size distribution")
    add("")
    add("Cluster records by member count, across the whole clip.")
    add("")
    sizes = sorted({size for result in results for size in result.crowd.cluster_size_histogram})
    if sizes:
        add("| members | " + " | ".join(f"`{r.clip.clip_id}`" for r in results) + " |")
        add("| ---: | " + " | ".join("---:" for _ in results) + " |")
        for size in sizes:
            add(
                f"| {size} | "
                + " | ".join(f"{r.crowd.cluster_size_histogram.get(size, 0):,}" for r in results)
                + " |"
            )
    else:
        add("No cluster of the configured minimum size was observed in any clip.")
    add("")
    add(
        "A composition change is **not** automatically a real-world group event: it occurs on "
        "a join, on a leave, when a member's track ID fragments and is re-acquired under a new "
        "one, and when two members drift across the connectivity radius and back."
    )
    add("")
    _persistence_finding(add, results)


def _persistence_finding(add: Any, results: Sequence[ClipValidationResult]) -> None:
    """State the measured persistence result and what it does and does not show.

    Prominent rather than buried, because the figure is small enough that a
    reader skimming the table would reasonably conclude the metric is broken.
    It is not: it is measuring what it was built to measure.
    """
    medians = [
        result.crowd.persistence_seconds.median
        for result in results
        if result.crowd.persistence_seconds.median is not None
    ]
    if not medians:
        return

    add("### Finding — exact-composition persistence is very short")
    add("")
    add(
        f"Median exact-composition persistence measured **{min(medians):.2f} to "
        f"{max(medians):.2f} s** across these clips. Under a second, on every scene type."
    )
    add("")
    add(
        "**This is the metric behaving as specified, not a defect.** V0.3.5 addresses a cluster "
        "by its *exact member set*, so any change of membership ends the episode and starts a "
        "new one. With per-frame track churn at the levels reported above, the exact member set "
        "of a real gathering changes almost every frame — one track fragmenting, one "
        "re-acquired, one drifting a few hundredths of a normalized unit across the "
        "connectivity radius — even while the same people stand in the same place."
    )
    add("")
    add("What the figure therefore does and does not measure:")
    add("")
    add(
        "- **It measures** the stability of an *exact composition* — how long a specific set of "
        "track IDs stayed mutually connected without a single membership change."
    )
    add(
        "- **It does not measure** the continuity of a spatial crowd region. A cluster of "
        "roughly the same people, in roughly the same place, persisting for thirty seconds "
        "would still report a sequence of sub-second compositions."
    )
    add("")
    add(
        "No conclusion is drawn from this here, and **no algorithm was changed**: the V0.3.5 "
        "exact-member-set persistence rule is unmodified. It is recorded as a bounded review "
        "item — *evaluate whether a separate neutral spatial-cluster episode continuity "
        "primitive is needed after ground-truth review* — because whether region continuity is "
        "worth measuring separately depends on how much of the churn is genuine movement "
        "versus tracking fragmentation, and only human ground truth can separate those two. "
        "Nothing has been implemented for it, and no such primitive would infer social-group "
        "identity."
    )
    add("")


def _output_volume_section(add: Any, results: Sequence[ClipValidationResult]) -> None:
    add("## Output volume")
    add("")
    add(
        "Sizes are as stored — the streams are gzipped, so these are compressed sizes. "
        "Structured totals exclude the rendered review video. Record counts include each "
        "stream's `run_metadata` header line."
    )
    add("")
    add(
        "| clip | behaviour | interaction | crowd | total structured | MiB / source min "
        "| rendered video |"
    )
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        volumes = {stream.name: stream for stream in result.output_volume.streams}
        processed = result.performance.processed_duration_seconds
        add(
            f"| `{result.clip.clip_id}` | "
            f"{_mib(volumes.get('behavior'))} | "
            f"{_mib(volumes.get('interaction'))} | "
            f"{_mib(volumes.get('crowd'))} | "
            f"{result.output_volume.total_bytes / (1024 * 1024):.2f} MiB | "
            f"{_fmt(result.output_volume.megabytes_per_source_minute(processed), 2)} | "
            f"{_bytes_mib(result.output_volume.rendered_video_bytes)} |"
        )
    add("")
    add("| clip | behaviour rec/s | interaction rec/s | crowd rec/s |")
    add("| --- | ---: | ---: | ---: |")
    for result in results:
        volumes = {stream.name: stream for stream in result.output_volume.streams}
        processed = result.performance.processed_duration_seconds
        add(
            f"| `{result.clip.clip_id}` | "
            + " | ".join(
                _fmt(
                    (
                        None
                        if volumes.get(name) is None
                        else volumes[name].records_per_source_second(processed)
                    ),
                    1,
                )
                for name in ("behavior", "interaction", "crowd")
            )
            + " |"
        )
    add("")
    add("Rates are per second of **source** time, not per second of wall clock.")
    add("")


def _comparison_section(add: Any, results: Sequence[ClipValidationResult]) -> None:
    add("## Scene comparison")
    add("")
    add(
        "One table, every clip. Every row is a proxy / system metric except where marked. "
        "**No row is a pass/fail criterion**, and no threshold anywhere in this document "
        "carries an acceptance meaning."
    )
    add("")
    header = (
        "| metric | "
        + " | ".join(f"`{r.clip.clip_id}`<br>({r.clip.scene_type})" for r in results)
        + " |"
    )
    add(header)
    add("| --- | " + " | ".join("---:" for _ in results) + " |")
    for row in _comparison_rows(results):
        add(f"| {row.label} | " + " | ".join(row.values) + " |")
    add("")

    baseline = _baseline(results)
    if baseline is not None and len(results) > 1:
        add("### Relative differences")
        add("")
        add(
            f"Ratios against `{baseline.clip.clip_id}` ({baseline.clip.scene_type}), stated as "
            "measurements. A larger number is not a worse scene — it describes different "
            "footage under identical settings."
        )
        add("")
        for line in _relative_lines(results, baseline):
            add(f"- {line}")
        add("")


def _comparison_rows(results: Sequence[ClipValidationResult]) -> list[_Row]:
    rows = [
        _Row("mean active tracks", tuple(_fmt(r.tracks.active_tracks.mean, 2) for r in results)),
        _Row("max active tracks", tuple(str(r.tracks.max_active_tracks) for r in results)),
        _Row("unique tracks", tuple(f"{r.tracks.unique_track_count:,}" for r in results)),
        _Row(
            "track births / min",
            tuple(_fmt(r.tracks.track_births_per_minute, 1) for r in results),
        ),
        _Row(
            "median track lifetime (s)",
            tuple(_fmt(r.tracks.lifetime_seconds.median, 2) for r in results),
        ),
        _Row(
            "tracks < 1 s",
            tuple(_pct(r.tracks.short_track_fractions.get(1.0)) for r in results),
        ),
        _Row(
            "speed coverage",
            tuple(_pct(r.behavior.coverage(SignalType.SPEED.value)) for r in results),
        ),
        _Row(
            "acceleration coverage",
            tuple(_pct(r.behavior.coverage(SignalType.ACCELERATION.value)) for r in results),
        ),
        _Row(
            "heading-change coverage",
            tuple(_pct(r.behavior.coverage(SignalType.HEADING_CHANGE_RATE.value)) for r in results),
        ),
        _Row(
            "pair births / min",
            tuple(_fmt(r.interaction.pair_births_per_minute, 1) for r in results),
        ),
        _Row(
            "median pair episode (s)",
            tuple(_fmt(r.interaction.episode_seconds.median, 2) for r in results),
        ),
        _Row(
            "mean clusters / frame",
            tuple(_fmt(r.crowd.clusters_per_frame.mean, 3) for r in results),
        ),
        _Row(
            "median cluster persistence (s)",
            tuple(_fmt(r.crowd.persistence_seconds.median, 2) for r in results),
        ),
        _Row(
            "processing FPS *(machine-specific)*",
            tuple(_fmt(r.performance.processed_fps, 2) for r in results),
        ),
        _Row(
            "structured MiB / source min",
            tuple(
                _fmt(
                    r.output_volume.megabytes_per_source_minute(
                        r.performance.processed_duration_seconds
                    ),
                    2,
                )
                for r in results
            ),
        ),
        _Row(
            "manual ground truth *(human)*",
            tuple(r.ground_truth.status.value for r in results),
        ),
    ]
    return rows


def _baseline(results: Sequence[ClipValidationResult]) -> ClipValidationResult | None:
    """The clip to state ratios against: the control scene, else the first."""
    for result in results:
        if "control" in result.clip.scene_type.lower():
            return result
    return results[0] if results else None


def _relative_lines(
    results: Sequence[ClipValidationResult], baseline: ClipValidationResult
) -> list[str]:
    lines: list[str] = []
    for result in results:
        if result.clip.clip_id == baseline.clip.clip_id:
            continue
        parts: list[str] = []
        short_ratio = _ratio_of(
            result.tracks.short_track_fractions.get(1.0),
            baseline.tracks.short_track_fractions.get(1.0),
        )
        if short_ratio is not None:
            parts.append(f"{short_ratio:.2f}x the sub-1 s track fraction")
        birth_ratio = _ratio_of(
            result.tracks.track_births_per_minute, baseline.tracks.track_births_per_minute
        )
        if birth_ratio is not None:
            parts.append(f"{birth_ratio:.2f}x the track births per minute")
        lifetime_ratio = _ratio_of(
            result.tracks.lifetime_seconds.median, baseline.tracks.lifetime_seconds.median
        )
        if lifetime_ratio is not None:
            parts.append(f"{lifetime_ratio:.2f}x the median track lifetime")
        volume_ratio = _ratio_of(
            result.output_volume.megabytes_per_source_minute(
                result.performance.processed_duration_seconds
            ),
            baseline.output_volume.megabytes_per_source_minute(
                baseline.performance.processed_duration_seconds
            ),
        )
        if volume_ratio is not None:
            parts.append(f"{volume_ratio:.2f}x the structured output per source minute")
        if parts:
            lines.append(f"`{result.clip.clip_id}` has " + ", ".join(parts) + ".")
    return lines


def _ground_truth_section(
    add: Any, results: Sequence[ClipValidationResult], layout: ValidationLayout
) -> None:
    add("## Manual ground truth — human annotation")
    add("")
    add(
        "**A separate measurement class.** Nothing in this section is derived from the proxy "
        "metrics above, and nothing above is derived from this section."
    )
    add("")
    add("| clip | status | annotated frames | annotated sequences |")
    add("| --- | --- | ---: | ---: |")
    for result in results:
        ground_truth = result.ground_truth
        add(
            f"| `{result.clip.clip_id}` | `{ground_truth.status.value}` | "
            f"{len(ground_truth.frames)} / {ground_truth.frame_rows_available} | "
            f"{len(ground_truth.continuity)} / {ground_truth.continuity_rows_available} |"
        )
    add("")

    annotated = [
        result
        for result in results
        if result.ground_truth.status is not GroundTruthStatus.NOT_ANNOTATED
    ]
    if not annotated:
        add(
            "**No clip has been annotated.** Therefore this report contains **no detection "
            "recall, no precision, no true ID-switch count and no true fragmentation count**. "
            "Those quantities cannot be derived from the automated metrics above, and none of "
            "them has been estimated, inferred or substituted."
        )
        add("")
    else:
        add(
            "| clip | frames | visible people | correct | false pos. | missed | recall "
            "| precision |"
        )
        add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for result in annotated:
            ground_truth = result.ground_truth
            if not ground_truth.frames:
                continue
            add(
                f"| `{result.clip.clip_id}` | {len(ground_truth.frames)} | "
                f"{ground_truth.visible_people} | {ground_truth.correct_detections} | "
                f"{ground_truth.false_positives} | {ground_truth.missed_people} | "
                f"{_pct(ground_truth.detection_recall_sample)} | "
                f"{_pct(ground_truth.detection_precision_sample)} |"
            )
        add("")
        continuity = [result for result in annotated if result.ground_truth.continuity]
        if continuity:
            add("| clip | sequences | seconds | ID switches | fragments | lost-track events |")
            add("| --- | ---: | ---: | ---: | ---: | ---: |")
            for result in continuity:
                ground_truth = result.ground_truth
                add(
                    f"| `{result.clip.clip_id}` | {len(ground_truth.continuity)} | "
                    f"{_fmt(ground_truth.annotated_continuity_seconds, 1)} | "
                    f"{ground_truth.total_id_switches} | {ground_truth.total_fragments} | "
                    f"{ground_truth.total_lost_track_events} |"
                )
            add("")
        add(
            "These figures describe a small deterministic sample, not a dataset-level "
            "benchmark, and the sample size is stated next to every one of them."
        )
        add("")

    add("### The annotation workflow")
    add("")
    add("Per clip, the run generates a deterministic review sample:")
    add("")
    for result in results:
        sample = result.review_sample
        add(
            f"- `{result.clip.clip_id}` — {sample.get('sample_frames', 0)} still frames and "
            f"{sample.get('continuity_windows', 0)} continuity windows in "
            f"`{sample.get('frames_dir', _NA)}`"
        )
    add("")
    add(
        "Frames are extracted from the **rendered tracking video**, so boxes, track IDs, "
        "confidences and trails are visible. Sample points are a pure function of frame count "
        "and frame rate: the same clip always yields the same points, so an annotation stays "
        "valid across re-runs."
    )
    add("")
    add("To annotate:")
    add("")
    add("1. Open `manual-ground-truth/<clip_id>/contact_sheet.jpg` or the individual frames.")
    add(
        "2. Fill in `frame_annotations.csv` — `visible_people`, "
        "`correct_person_detections`, `false_positive_detections`, `missed_people`."
    )
    add(
        "3. Optionally follow individual subjects through the continuity windows and fill in "
        "`continuity_annotations.csv`."
    )
    add("4. Re-run `atreides-vision validation report` — no footage is re-processed.")
    add("")
    add(
        "`target_label` in the continuity file is a scratch label for keeping one subject "
        "straight within one window. **It is run-local annotation, never an identity**: "
        "nothing reads it back into the pipeline, and no ReID is implied. Existing annotation "
        "files are never overwritten by a re-run."
    )
    add("")
    add(f"Artifacts are under `{layout.relative(layout.ground_truth_root)}/`.")
    add("")


def _limitations_section(add: Any, results: Sequence[ClipValidationResult]) -> None:
    add("## Limitations")
    add("")
    add("### Camera-local track IDs are tracklets, not guaranteed unique people")
    add("")
    add(
        "This is the single most important limitation in the V0.3 stack, and every count in "
        "this report inherits it."
    )
    add("")
    add("Concretely:")
    add("")
    add("- Occlusion can fragment one person into several track IDs.")
    add(
        "- When it does, that person's **behaviour history resets** — speed, acceleration "
        "and heading windows all start again from empty."
    )
    add("- Their **pair history resets**: existing pair episodes end and new ones begin.")
    add("- The **exact cluster composition changes**, so cluster persistence restarts.")
    add(
        "- Unique track ID counts may therefore differ from the number of unique real "
        "people, in the direction of over-counting."
    )
    add("")
    add(
        "The behaviour warm-up figures above quantify what a reset costs: a re-acquired track "
        "produces no rate signal at all until its window refills."
    )
    add("")
    add("### No physical measurement ground truth")
    add("")
    add(
        "None of this footage has a measured floor calibration, so all of it was processed in "
        "normalized image units. **No metre, m/s or subjects/m² figure appears in this "
        "report**, and no floor dimension was estimated from doors, tiles, vehicles or people "
        "in the footage. Physical validation requires a scene with physically measured "
        "correspondences; the manifest can carry one (`calibration_path`), and none currently "
        "does."
    )
    add("")
    add("### Multi-camera")
    add("")
    add(
        "Each clip is its own camera. **Counts and identities are never combined across "
        "cameras**, and nothing in this report or the codebase performs cross-camera "
        "association. A future architecture may add camera-local tracks → multi-camera "
        "association → temporary venue-local IDs; that layer is not part of V0.3.6. There is "
        "no ReID, no appearance embedding and no face recognition anywhere in this system."
    )
    add("")
    add("### Sample size and scene coverage")
    add("")
    total = sum(r.performance.processed_duration_seconds for r in results)
    add(
        f"{len(results)} clips totalling {total / 60.0:.1f} minutes of footage. That is enough "
        "to characterise how the stack behaves on materially different scenes; it is not a "
        "dataset, and no figure here should be quoted as a general-purpose benchmark."
    )
    add("")
    add("### Harness scaling")
    add("")
    add(
        "The harness streams every input and never holds a whole file in memory, but three "
        "structures scale with content rather than with clip length: one presence record per "
        "unique track ID, one float per pair episode and per cluster-composition episode, and "
        "a set of every distinct cluster composition. On the validation footage these are "
        "small; on hours of dense footage the composition set would be the first to matter."
    )
    add("")
    add("### Performance figures")
    add("")
    add(
        "Machine-specific development measurements, including annotated-video encoding. Not a "
        "production capability claim, and not comparable to figures taken under a different "
        "configuration."
    )
    add("")


def _freeze_section(add: Any) -> None:
    add("## V0.3 signal stack — implementation scope and validation status")
    add("")
    add("### What is implemented, and now measured on real footage")
    add("")
    add("**Individual (V0.3.0)**")
    add("")
    add("- speed")
    add("- acceleration")
    add("- heading-change rate")
    add("- stationary duration")
    add("")
    add("**Pairwise (V0.3.4)**")
    add("")
    add("- separation distance")
    add("- closing / separation speed")
    add("- relative motion heading")
    add("- sustained proximity duration")
    add("")
    add("**Crowd (V0.3.5)**")
    add("")
    add("- radius-connectivity clustering")
    add("- cluster member count")
    add("- centroid")
    add("- RMS spatial spread")
    add("- mean / max pair distance")
    add("- local density")
    add("- motion-heading coherence")
    add("- exact-member-set persistence")
    add("")
    add("**Calibration (V0.3.1 - V0.3.3)**")
    add("")
    add("- ground-plane homography projection")
    add("- calibration fit reporting and visual diagnostics")
    add("- camera-drift integrity monitoring with fail-safe metric suppression")
    add("")
    add("### Known limitations of the implemented stack")
    add("")
    add(
        "1. **Camera-local track IDs are tracklets.** Occlusion fragments identity, and every "
        "downstream history — individual, pairwise and cluster — resets when it does."
    )
    add(
        "2. **No physically validated metric output.** The projection is implemented and "
        "tested against synthetic cameras with known geometry; no real scene has been "
        "validated against measured floor distances."
    )
    add("3. **Single camera.** No association, no fusion, no venue-local identity, no ReID.")
    add(
        "4. **Measurement only.** There is no risk engine, no decision engine, no "
        "classification of any behaviour, and no operator alerting."
    )
    add(
        "5. **Detection is the floor for everything.** Missed detections in distant or dense "
        "regions propagate into every signal above them, and this harness measures the "
        "consequences rather than the cause."
    )
    add(
        "6. **No production readiness claim.** These are development measurements on four "
        "clips, and production validation is not established."
    )
    add("")
    add("### Validation status")
    add("")
    add(
        "**Final signal-stack acceptance is pending.** V0.3.6 completes one of several "
        "reviews, and nothing in this report establishes that the V0.3 signal stack is "
        "sufficient, correct against reality, or valid in production."
    )
    add("")
    add("| Stage | Status |")
    add("| --- | --- |")
    add("| Implementation scope (V0.3.0 - V0.3.5 signals) | **complete** |")
    add("| Automated system / proxy validation on real footage | **complete** |")
    add("| Manual detection & tracking ground truth | **pending** |")
    add("| Controlled physical metric validation | **pending** |")
    add("| Production validation | **not established** |")
    add("| Final signal-stack acceptance / freeze | **pending the reviews above** |")
    add("")
    add(
        "What *is* settled: the implementation scope. No further measurement feature should be "
        "added to V0.3, and the detector and tracker defaults **remain frozen at their V0.2.1 "
        "values for the duration of validation** — they are not to be tuned in response to any "
        "figure in this report."
    )
    add("")
    add(
        "What is **not** settled: whether these signals are accurate against reality, whether "
        "their physical units mean anything on a real floor, and whether the stack behaves "
        "acceptably in production. Those are the pending rows, and none of them can be "
        "answered by the automated metrics above — which measure the system against itself."
    )
    add("")
    add("### Open review items")
    add("")
    add(
        "1. **Manual detection & tracking ground truth.** Annotate the generated review sample "
        "to establish detection recall, precision, true ID switches and true fragmentation. "
        "Templates and frames are already generated; see the ground-truth section above."
    )
    add(
        "2. **Controlled physical metric validation.** Validate the ground-plane projection "
        "against a scene with tape-measured floor correspondences, so metric output can be "
        "reported as physical distance rather than normalized frame fractions."
    )
    add(
        "3. **Evaluate whether a separate neutral spatial-cluster episode continuity primitive "
        "is needed after ground-truth review.** Scoped by the persistence finding above. This "
        "is a question to answer with evidence, not a decision already taken — and nothing has "
        "been implemented for it."
    )
    add("")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return _NA
    return f"{value:,.{digits}f}"


def _pct(value: float | None) -> str:
    if value is None:
        return _NA
    return f"{value * 100.0:.1f}%"


def _ratio(numerator: int | None, denominator: int | None) -> str:
    if numerator is None or not denominator:
        return _NA
    return f"{numerator} / {denominator}"


def _ratio_of(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline == 0:
        return None
    return value / baseline


def _mib(stream: Any) -> str:
    if stream is None:
        return _NA
    return f"{stream.size_bytes / (1024 * 1024):.2f} MiB"


def _bytes_mib(value: int | None) -> str:
    if value is None:
        return _NA
    return f"{value / (1024 * 1024):.1f} MiB"


def write_validation_artifacts(
    layout: ValidationLayout, summary: ValidationSummary
) -> dict[str, Any]:
    """Write the JSON summary, the CSV and the Markdown report."""
    json_path = write_json(layout.summary_json_path, summary.to_record())
    csv_path = write_csv(
        layout.summary_csv_path, SUMMARY_CSV_COLUMNS, summary_csv_rows(summary.results)
    )
    report_path = write_text(layout.report_path, render_report(summary, layout))
    return {"summary_json": json_path, "summary_csv": csv_path, "report": report_path}

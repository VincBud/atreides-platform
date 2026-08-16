"""The three report artifacts: determinism, labelling, and what must never appear.

Three properties are load-bearing enough to test explicitly:

* every artifact is a deterministic function of the results,
* proxy metrics and manual ground truth are never presented under one label, and
* no physical unit and no pass/fail verdict is ever produced for footage that
  supports neither.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from atreides_vision.validation.analysis import analyze_clip
from atreides_vision.validation.ground_truth import GroundTruthStatus
from atreides_vision.validation.manifest import ValidationClip
from atreides_vision.validation.models import ClipValidationResult
from atreides_vision.validation.reporting import (
    SUMMARY_CSV_COLUMNS,
    build_summary,
    render_report,
    summary_csv_rows,
    write_validation_artifacts,
)
from atreides_vision.validation.runner import ValidationRunner
from atreides_vision.validation.storage import ValidationLayout
from test_validation_runner import StableTracker, WalkingDetector

FPS = 20.0
FRAME_COUNT = 30
WIDTH, HEIGHT = 160, 120


def write_clip(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    for index in range(FRAME_COUNT):
        writer.write(np.full((HEIGHT, WIDTH, 3), 30 + index, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def results(tmp_path: Path) -> tuple[ValidationLayout, list[ClipValidationResult]]:
    """Two analysed clips, produced by the real pipeline with fakes."""
    layout = ValidationLayout.at(tmp_path / "out").ensure()
    runner = ValidationRunner(
        layout=layout,
        device="cpu",
        detector_factory=lambda config: WalkingDetector(),
        tracker_factory=lambda config: StableTracker(),
        sample_frames=3,
        continuity_windows=2,
    )
    analysed: list[ClipValidationResult] = []
    for name, scene in (("ctrl_01", "control"), ("dense_01", "dense")):
        clip = ValidationClip(
            clip_id=name,
            input_path=write_clip(tmp_path, name),
            scene_type=scene,
            camera_id=f"cam_{name}",
            notes=f"{scene} note",
        )
        analysed.append(analyze_clip(layout, runner.run_clip(clip)))
    return layout, analysed


# --- JSON ------------------------------------------------------------------


def test_summary_json_is_deterministic(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    layout, analysed = results
    summary = build_summary(analysed, manifest_path="config/validation/v0.3.6.json")

    first = write_validation_artifacts(layout, summary)["summary_json"].read_text(encoding="utf-8")
    second = write_validation_artifacts(layout, summary)["summary_json"].read_text(encoding="utf-8")

    assert first == second
    record = json.loads(first)
    assert list(record) == sorted(record)
    assert record["schema_version"] == "0.3.6"
    assert [clip["clip"]["clip_id"] for clip in record["clips"]] == ["ctrl_01", "dense_01"]


def test_summary_json_separates_proxy_metrics_from_ground_truth(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    _, analysed = results

    record = build_summary(analysed).to_record()
    clip = record["clips"][0]

    assert clip["proxy_metrics"]["measurement_class"] == "automated_proxy_system_metrics"
    assert clip["manual_ground_truth"]["measurement_class"] == "manual_ground_truth"
    # No proxy metric may live under the ground-truth key, or vice versa.
    assert "tracking" not in clip["manual_ground_truth"]
    assert "detection_recall_sample" not in json.dumps(clip["proxy_metrics"])


def test_summary_json_states_that_no_clip_is_physically_calibrated(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    _, analysed = results

    record = build_summary(analysed).to_record()

    assert record["physical_units"]["any_clip_physically_calibrated"] is False
    assert record["comparability"]["all_clips_at_v021_defaults"] is True


def test_summary_json_carries_full_provenance_per_clip(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    _, analysed = results

    clip = build_summary(analysed).to_record()["clips"][0]
    provenance = clip["provenance"]

    for key in (
        "detector",
        "detector_confidence",
        "tracker",
        "tracker_high_conf_threshold",
        "tracker_minimum_iou_threshold",
        "tracker_lost_track_buffer",
        "tracker_minimum_consecutive_frames",
        "tracker_direction_consistency_weight",
        "tracker_delta_t",
        "deduplicate_detections",
        "deduplication_rule",
        "coordinate_space",
        "camera_id",
        "input_path",
        "behavior_config",
        "interaction_config",
        "crowd_config",
        "output_sample_interval_seconds",
        "schema_versions",
        "package_version",
    ):
        assert key in provenance, key


# --- CSV -------------------------------------------------------------------


def test_summary_csv_has_one_row_per_clip_in_order(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    layout, analysed = results
    summary = build_summary(analysed)

    path = write_validation_artifacts(layout, summary)["summary_csv"]
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))

    assert [row["clip_id"] for row in rows] == ["ctrl_01", "dense_01"]
    assert list(rows[0]) == list(SUMMARY_CSV_COLUMNS)


def test_summary_csv_column_order_is_fixed(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    layout, analysed = results
    summary = build_summary(analysed)

    first = write_validation_artifacts(layout, summary)["summary_csv"].read_text(encoding="utf-8")
    second = write_validation_artifacts(layout, summary)["summary_csv"].read_text(encoding="utf-8")

    assert first == second
    assert first.splitlines()[0] == ",".join(SUMMARY_CSV_COLUMNS)


def test_unannotated_clips_leave_manual_columns_empty_not_zero(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    """A fabricated 0.0 recall would be charted next to a real one."""
    _, analysed = results

    rows = summary_csv_rows(analysed)

    for row in rows:
        assert row["manual_ground_truth_status"] == GroundTruthStatus.NOT_ANNOTATED.value
        assert row.get("manual_detection_recall_sample") is None
        assert row.get("manual_precision_sample") is None
        assert row.get("manual_id_switches") is None


def test_csv_renders_absent_values_as_empty_cells(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    layout, analysed = results

    path = write_validation_artifacts(layout, build_summary(analysed))["summary_csv"]
    text = path.read_text(encoding="utf-8")

    assert "None" not in text
    row = next(csv.DictReader(io.StringIO(text)))
    assert row["manual_detection_recall_sample"] == ""


def test_csv_marks_proxy_columns_as_proxies(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    proxy_columns = [name for name in SUMMARY_CSV_COLUMNS if name.startswith("proxy_")]
    manual_columns = [name for name in SUMMARY_CSV_COLUMNS if name.startswith("manual_")]

    assert "proxy_track_births_per_minute" in proxy_columns
    assert "proxy_unique_track_ids" in proxy_columns
    assert "manual_detection_recall_sample" in manual_columns
    # No column name may suggest a proxy is ground truth.
    assert not any("recall" in name for name in proxy_columns)
    assert not any("id_switch" in name for name in proxy_columns)


# --- Markdown --------------------------------------------------------------


def test_report_is_deterministic(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    layout, analysed = results
    summary = build_summary(analysed)

    assert render_report(summary, layout) == render_report(summary, layout)


def test_report_leads_with_the_proxy_limitation(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    layout, analysed = results

    report = render_report(build_summary(analysed), layout)

    assert "proxy / system metric" in report
    assert "is not an ID-switch rate" in report
    assert "not proof of fragmentation" in report
    assert "Nothing here is detection recall" in report
    assert "upper bound on people, never a census" in report


def test_report_states_the_absence_of_physical_units(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    layout, analysed = results

    report = render_report(build_summary(analysed), layout)

    assert "No physical units appear in this report" in report
    assert "normalized_frame_unit" in report
    # No *value* may carry a physical unit. The units are named in the caveats
    # that say they are absent, which is the opposite of quoting one.
    assert not re.search(r"\d+(\.\d+)?\s*m/s\b", report)
    assert not re.search(r"\d+(\.\d+)?\s*subjects/m", report)
    assert not re.search(r"\d+(\.\d+)?\s*(metres|meters|m\^2|m²)\b", report)


def test_report_contains_no_pass_fail_verdict(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    layout, analysed = results

    report = render_report(build_summary(analysed), layout).lower()

    assert "reaches no pass/fail conclusion" in report
    for verdict in ("**pass**", "**fail**", "verdict: pass", "verdict: fail", "acceptance band"):
        assert verdict not in report
    # Nor an invented threshold band presented as a criterion.
    assert "is bad" not in report
    assert "is good" not in report


def test_report_states_what_pair_churn_can_and_cannot_mean(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    """With no distance gate, an episode cannot end because two people separated."""
    layout, analysed = results
    assert all(r.provenance.interaction_config.get("max_pair_distance") is None for r in analysed)

    report = render_report(build_summary(analysed), layout)

    assert "max_pair_distance` = null" in report or "`max_pair_distance = null`" in report
    assert "cannot end because two subjects separated" in report
    assert "combinatorial count, not a count of encounters" in report
    assert "measures\n*observation continuity*" in report or "observation continuity" in report


def test_report_states_the_track_id_limitation_prominently(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    layout, analysed = results

    report = render_report(build_summary(analysed), layout)

    assert "tracklets, not guaranteed unique people" in report
    assert "behaviour history resets" in report
    assert "pair history resets" in report
    assert "exact cluster composition changes" in report


def test_report_states_the_multi_camera_limitation(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    layout, analysed = results

    report = render_report(build_summary(analysed), layout)

    assert "never combined across" in report
    assert "no reid" in report.lower()
    assert "face recognition" in report


def test_report_reports_ground_truth_as_not_annotated(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    layout, analysed = results

    report = render_report(build_summary(analysed), layout)

    assert "NOT_ANNOTATED" in report
    assert "**No clip has been annotated.**" in report
    assert "no detection recall" in report


def test_report_includes_a_scene_comparison_table(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    layout, analysed = results

    report = render_report(build_summary(analysed), layout)

    assert "## Scene comparison" in report
    for row in (
        "mean active tracks",
        "unique tracks",
        "track births / min",
        "median track lifetime",
        "speed coverage",
        "pair births / min",
        "mean clusters / frame",
        "processing FPS",
        "structured MiB / source min",
    ):
        assert row in report


def test_report_states_the_shared_camera_isolation_rule(tmp_path: Path) -> None:
    """When a camera supplies several clips, the report says what that does not mean."""
    layout = ValidationLayout.at(tmp_path / "out").ensure()
    runner = ValidationRunner(
        layout=layout,
        device="cpu",
        detector_factory=lambda config: WalkingDetector(),
        tracker_factory=lambda config: StableTracker(),
        sample_frames=2,
        continuity_windows=1,
        max_frames=10,
    )
    analysed = [
        analyze_clip(
            layout,
            runner.run_clip(
                ValidationClip(
                    clip_id=clip_id,
                    input_path=write_clip(tmp_path, clip_id),
                    scene_type="control",
                    camera_id="camera_01",
                )
            ),
        )
        for clip_id in ("clip_A", "clip_B")
    ]
    summary = build_summary(analysed)

    assert summary.shared_camera_ids == ("camera_01",)
    assert summary.clip_ids_for_camera("camera_01") == ("clip_A", "clip_B")

    report = render_report(summary, layout)
    assert "### Clips sharing a camera" in report
    assert "two independent" in report
    assert "must not be added together" in report

    record = summary.to_record()
    isolation = record["clip_isolation"]
    assert isolation["cameras_with_multiple_clips"] == {"camera_01": ["clip_A", "clip_B"]}
    assert "no track id, history or metric is carried across a clip" in isolation["note"]


def test_summary_reports_no_shared_cameras_when_each_clip_has_its_own(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    _, analysed = results

    summary = build_summary(analysed)

    assert summary.shared_camera_ids == ()
    assert summary.to_record()["clip_isolation"]["cameras_with_multiple_clips"] == {}


def test_report_states_the_cluster_persistence_finding(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    """The measured figure is stated, with what it does and does not measure."""
    layout, analysed = results

    report = render_report(build_summary(analysed), layout)

    assert "### Finding — exact-composition persistence is very short" in report
    assert "Median exact-composition persistence measured" in report
    assert "behaving as specified, not a defect" in report
    assert "does not measure** the continuity of a spatial crowd region" in report
    assert "no algorithm was changed" in report
    assert (
        "evaluate whether a separate neutral spatial-cluster episode continuity "
        "primitive is needed after ground-truth review" in report
    )
    assert "social-group identity" in report


def test_report_does_not_claim_the_stack_is_validated_or_sufficient(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    """Implementation scope is settled; acceptance explicitly is not."""
    layout, analysed = results

    report = render_report(build_summary(analysed), layout)

    assert "**Final signal-stack acceptance is pending.**" in report
    assert "| Manual detection & tracking ground truth | **pending** |" in report
    assert "| Controlled physical metric validation | **pending** |" in report
    assert "| Production validation | **not established** |" in report
    assert "| Implementation scope (V0.3.0 - V0.3.5 signals) | **complete** |" in report
    assert "sufficient, accurate against " in report
    assert "remain frozen at their V0.2.1 " in report
    # No wording that reads as an achieved freeze or a sufficiency claim.
    assert "Freeze recommendation" not in report
    assert "complete as specified and now measured on real footage" not in report


def test_report_ends_with_the_freeze_section(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    layout, analysed = results

    report = render_report(build_summary(analysed), layout)

    assert "## V0.3 signal stack — implementation scope and validation status" in report
    assert "### Validation status" in report
    assert "### Open review items" in report
    assert "heading-change rate" in report
    assert "exact-member-set persistence" in report
    assert "no risk engine" in report.lower()
    assert report.index("### Validation status") > report.index("## Scene comparison")


def test_report_marks_performance_as_machine_specific(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    layout, analysed = results

    report = render_report(build_summary(analysed), layout)

    assert "machine-specific development measurements" in report
    assert "do not establish production real-time capability" in report


def test_relative_differences_are_stated_as_ratios_not_judgements(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    layout, analysed = results

    report = render_report(build_summary(analysed), layout)

    if "### Relative differences" in report:
        assert "A larger number is not a worse scene" in report


def test_all_three_artifacts_are_written(
    results: tuple[ValidationLayout, list[ClipValidationResult]],
) -> None:
    layout, analysed = results

    written = write_validation_artifacts(layout, build_summary(analysed))

    assert written["summary_json"].name == "validation_summary.json"
    assert written["summary_csv"].name == "validation_summary.csv"
    assert written["report"].name == "V0.3.6_REPORT.md"
    assert all(path.exists() for path in written.values())

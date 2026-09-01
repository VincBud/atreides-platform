"""Deterministic artifacts for the offline WILDTRACK evaluator (V0.4.9b-D1).

Both files must be byte-for-byte reproducible from the same inputs, must carry
the schema version, and must preserve an undefined ratio as JSON ``null`` rather
than a misleading zero.
"""

from __future__ import annotations

import json
from pathlib import Path

from atreides_vision.venue_validation.datasets.wildtrack.evaluation.config import EvaluationConfig
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.evaluator import (
    WildtrackEvaluator,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.models import (
    FalseMergeDiagnostic,
    InstantEvaluation,
    WildtrackEvaluationSummary,
)
from atreides_vision.venue_validation.datasets.wildtrack.evaluation.reporting import (
    WILDTRACK_EVALUATION_SCHEMA_VERSION,
    write_false_merge_diagnostics,
    write_wildtrack_evaluation_report,
    write_wildtrack_evaluation_summary,
)
from fusion_builders import associated_decision
from wildtrack_evaluation_builders import (
    ground_truth_of,
    gt_frame,
    gt_person,
    metric_obs,
    snapshot_for,
)

CONFIG = EvaluationConfig(selected_camera_ids=("C1", "C2"))


def _run() -> tuple[WildtrackEvaluationSummary, tuple[InstantEvaluation, ...]]:
    a = metric_obs("C1", 1, x=0.0, y=0.0)
    frame = gt_frame(0, gt_person(1, x=0.0, y=0.0, cameras=("C1",)))
    evaluator = WildtrackEvaluator(CONFIG, ground_truth_of(frame))
    evaluator(snapshot_for(0, observations=[a]))
    summary = evaluator.finalize()
    return summary, evaluator.instants


def _false_merge_diagnostics() -> tuple[FalseMergeDiagnostic, ...]:
    a = metric_obs("C1", 7, x=0.0, y=0.0)
    b = metric_obs("C2", 3, x=0.0, y=0.0)
    frame = gt_frame(
        0,
        gt_person(1, x=0.0, y=0.0, cameras=("C1",)),
        gt_person(2, x=0.0, y=0.0, cameras=("C2",)),
    )
    evaluator = WildtrackEvaluator(CONFIG, ground_truth_of(frame))
    evaluator(snapshot_for(0, observations=[a, b], decisions=[associated_decision(a, b)]))
    return evaluator.finalize().association.false_merges_detail


def test_no_false_merge_diagnostics_file_is_written_when_there_are_none(tmp_path: Path) -> None:
    path = write_false_merge_diagnostics(tmp_path / "fm.jsonl", ())

    assert path is None
    assert not (tmp_path / "fm.jsonl").exists()


def test_a_false_merge_diagnostics_file_carries_a_header_and_one_record(tmp_path: Path) -> None:
    diagnostics = _false_merge_diagnostics()
    assert diagnostics  # a GT-resolved false merge was produced

    path = write_false_merge_diagnostics(tmp_path / "fm.jsonl", diagnostics)
    assert path is not None

    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["record"] == "header"
    assert header["schema_version"] == WILDTRACK_EVALUATION_SCHEMA_VERSION
    assert header["false_merge_count"] == 1
    record = json.loads(lines[1])
    assert record["record"] == "false_merge"
    assert record["first_tracklet"] == ["C1", 0, 7]
    assert record["second_tracklet"] == ["C2", 0, 3]
    assert record["outcome"] == "associated"


def test_the_false_merge_diagnostics_file_is_byte_identical_across_two_runs(tmp_path: Path) -> None:
    a = write_false_merge_diagnostics(tmp_path / "a.jsonl", _false_merge_diagnostics())
    b = write_false_merge_diagnostics(tmp_path / "b.jsonl", _false_merge_diagnostics())

    assert a is not None and b is not None
    assert a.read_bytes() == b.read_bytes()


def test_the_false_merge_diagnostics_file_carries_no_path_or_wall_clock_metadata(
    tmp_path: Path,
) -> None:
    path = write_false_merge_diagnostics(tmp_path / "fm.jsonl", _false_merge_diagnostics())
    assert path is not None

    text = path.read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    for banned in ("timestamp", "wall_clock", "hostname", "/", "\\"):
        assert banned not in text


def test_the_jsonl_report_leads_with_a_versioned_header(tmp_path: Path) -> None:
    summary, instants = _run()
    path = write_wildtrack_evaluation_report(tmp_path / "report.jsonl", summary, instants)

    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["record"] == "header"
    assert header["schema_version"] == WILDTRACK_EVALUATION_SCHEMA_VERSION
    assert header["instant_count"] == 1
    assert json.loads(lines[1])["record"] == "instant"


def test_the_summary_is_indented_json_with_a_trailing_newline(tmp_path: Path) -> None:
    summary, _ = _run()
    path = write_wildtrack_evaluation_summary(tmp_path / "summary.json", summary)

    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\n  " in text
    record = json.loads(text)
    assert record["schema_version"] == WILDTRACK_EVALUATION_SCHEMA_VERSION


def test_an_undefined_ratio_is_serialised_as_null(tmp_path: Path) -> None:
    summary, _ = _run()
    path = write_wildtrack_evaluation_summary(tmp_path / "summary.json", summary)

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["association"]["association_precision"]["value"] is None


def test_both_artifacts_are_byte_identical_across_two_runs(tmp_path: Path) -> None:
    summary_a, instants_a = _run()
    summary_b, instants_b = _run()

    report_a = write_wildtrack_evaluation_report(tmp_path / "a.jsonl", summary_a, instants_a)
    report_b = write_wildtrack_evaluation_report(tmp_path / "b.jsonl", summary_b, instants_b)
    sum_a = write_wildtrack_evaluation_summary(tmp_path / "a.json", summary_a)
    sum_b = write_wildtrack_evaluation_summary(tmp_path / "b.json", summary_b)

    assert report_a.read_bytes() == report_b.read_bytes()
    assert sum_a.read_bytes() == sum_b.read_bytes()

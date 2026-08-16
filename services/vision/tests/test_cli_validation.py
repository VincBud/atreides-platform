"""CLI exposure of the V0.3.6 validation harness.

No model weights are loaded: these parse arguments, or drive the real commands
with a fake detector and tracker patched into the runner.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import atreides_vision.validation.runner as runner_module
from atreides_vision.cli import EXIT_ERROR, EXIT_OK, build_parser, main
from atreides_vision.validation.manifest import MANIFEST_VERSION
from atreides_vision.validation.review_samples import DEFAULT_SAMPLE_FRAMES
from test_validation_runner import StableTracker, WalkingDetector

FPS = 20.0
FRAME_COUNT = 24
WIDTH, HEIGHT = 160, 120


def write_clip(path: Path) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    for index in range(FRAME_COUNT):
        writer.write(np.full((HEIGHT, WIDTH, 3), 20 + index, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    write_clip(tmp_path / "ctrl.mp4")
    write_clip(tmp_path / "dense.mp4")
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "manifest_version": MANIFEST_VERSION,
                "clips": [
                    {
                        "clip_id": "ctrl",
                        "input_path": str(tmp_path / "ctrl.mp4"),
                        "scene_type": "control",
                        "camera_id": "cam_ctrl",
                    },
                    {
                        "clip_id": "dense",
                        "input_path": str(tmp_path / "dense.mp4"),
                        "scene_type": "dense",
                        "camera_id": "cam_dense",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the runner's default factories so no weights are ever loaded.

    The runner resolves both at construction rather than as default argument
    values, which is what lets this substitution work without the CLI needing a
    test-only parameter.
    """
    monkeypatch.setattr(runner_module, "default_detector_factory", lambda config: WalkingDetector())
    monkeypatch.setattr(runner_module, "default_tracker_factory", lambda config: StableTracker())


# --- argument parsing ------------------------------------------------------


def test_validation_run_parses() -> None:
    args = build_parser().parse_args(
        ["validation", "run", "--manifest", "m.json", "--output", "out"]
    )

    assert args.command == "validation"
    assert args.validation_command == "run"
    assert args.manifest == Path("m.json")
    assert args.output == Path("out")
    assert args.clip is None
    assert args.max_frames is None
    assert args.sample_frames == DEFAULT_SAMPLE_FRAMES


def test_validation_report_parses() -> None:
    args = build_parser().parse_args(
        ["validation", "report", "--manifest", "m.json", "--output", "out", "--clip", "a"]
    )

    assert args.validation_command == "report"
    assert args.clip == ["a"]


def test_clip_is_repeatable() -> None:
    args = build_parser().parse_args(
        [
            "validation",
            "run",
            "--manifest",
            "m.json",
            "--output",
            "out",
            "--clip",
            "a",
            "--clip",
            "b",
        ]
    )

    assert args.clip == ["a", "b"]


def test_validation_run_requires_a_manifest() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["validation", "run", "--output", "out"])


def test_validation_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["validation"])


def test_no_detector_or_tracker_tuning_flag_is_exposed() -> None:
    """V0.3.6 measures the frozen defaults; a tuning flag would invite tuning."""
    parser = build_parser()
    for flag in (
        "--confidence",
        "--tracker",
        "--ocsort-high-conf-threshold",
        "--no-deduplicate-detections",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["validation", "run", "--manifest", "m.json", "--output", "out", flag, "0.5"]
            )


# --- run -------------------------------------------------------------------


def test_run_produces_every_artifact(tmp_path: Path, manifest: Path, fake_pipeline: None) -> None:
    out = tmp_path / "artifacts"

    code = main(
        ["validation", "run", "--manifest", str(manifest), "--output", str(out), "--device", "cpu"]
    )

    assert code == EXIT_OK
    assert (out / "validation_summary.json").exists()
    assert (out / "validation_summary.csv").exists()
    assert (out / "V0.3.6_REPORT.md").exists()
    for clip_id in ("ctrl", "dense"):
        run_dir = out / "runs" / clip_id
        assert (run_dir / "tracked.mp4").exists()
        assert (run_dir / "behavior.jsonl.gz").exists()
        assert (run_dir / "interaction.jsonl.gz").exists()
        assert (run_dir / "crowd.jsonl.gz").exists()
        assert (run_dir / "run.json").exists()
        gt_dir = out / "manual-ground-truth" / clip_id
        assert (gt_dir / "frame_annotations.csv").exists()
        assert (gt_dir / "continuity_annotations.csv").exists()


def test_run_summary_covers_every_manifest_clip(
    tmp_path: Path, manifest: Path, fake_pipeline: None
) -> None:
    out = tmp_path / "artifacts"
    main(
        ["validation", "run", "--manifest", str(manifest), "--output", str(out), "--device", "cpu"]
    )

    record = json.loads((out / "validation_summary.json").read_text(encoding="utf-8"))

    assert [clip["clip"]["clip_id"] for clip in record["clips"]] == ["ctrl", "dense"]
    assert record["physical_units"]["any_clip_physically_calibrated"] is False


def test_clip_filter_restricts_the_run(tmp_path: Path, manifest: Path, fake_pipeline: None) -> None:
    out = tmp_path / "artifacts"

    main(
        [
            "validation",
            "run",
            "--manifest",
            str(manifest),
            "--output",
            str(out),
            "--device",
            "cpu",
            "--clip",
            "dense",
        ]
    )

    assert (out / "runs" / "dense").is_dir()
    assert not (out / "runs" / "ctrl").exists()
    record = json.loads((out / "validation_summary.json").read_text(encoding="utf-8"))
    assert [clip["clip"]["clip_id"] for clip in record["clips"]] == ["dense"]


def test_max_frames_caps_processing(tmp_path: Path, manifest: Path, fake_pipeline: None) -> None:
    out = tmp_path / "artifacts"

    main(
        [
            "validation",
            "run",
            "--manifest",
            str(manifest),
            "--output",
            str(out),
            "--device",
            "cpu",
            "--max-frames",
            "6",
        ]
    )

    record = json.loads((out / "validation_summary.json").read_text(encoding="utf-8"))
    assert all(clip["performance"]["frames_processed"] == 6 for clip in record["clips"])


# --- report ----------------------------------------------------------------


def test_report_regenerates_without_reprocessing(
    tmp_path: Path, manifest: Path, fake_pipeline: None
) -> None:
    out = tmp_path / "artifacts"
    main(
        ["validation", "run", "--manifest", str(manifest), "--output", str(out), "--device", "cpu"]
    )
    before = (out / "runs" / "ctrl" / "tracked.mp4").stat().st_mtime_ns

    code = main(["validation", "report", "--manifest", str(manifest), "--output", str(out)])

    assert code == EXIT_OK
    assert (out / "runs" / "ctrl" / "tracked.mp4").stat().st_mtime_ns == before


def test_report_picks_up_annotation_added_after_the_run(
    tmp_path: Path, manifest: Path, fake_pipeline: None
) -> None:
    out = tmp_path / "artifacts"
    main(
        [
            "validation",
            "run",
            "--manifest",
            str(manifest),
            "--output",
            str(out),
            "--device",
            "cpu",
            "--sample-frames",
            "2",
        ]
    )

    annotations = out / "manual-ground-truth" / "ctrl" / "frame_annotations.csv"
    header, first, second = annotations.read_text(encoding="utf-8").strip().split("\n")
    prefix = first.split(",")[:3]
    annotations.write_text(
        "\n".join([header, ",".join([*prefix, "6", "5", "0", "1", ""]), second]) + "\n",
        encoding="utf-8",
    )

    main(["validation", "report", "--manifest", str(manifest), "--output", str(out)])

    record = json.loads((out / "validation_summary.json").read_text(encoding="utf-8"))
    ctrl = next(clip for clip in record["clips"] if clip["clip"]["clip_id"] == "ctrl")
    ground_truth = ctrl["manual_ground_truth"]
    assert ground_truth["manual_ground_truth_status"] == "PARTIALLY_ANNOTATED"
    # Serialised at four decimal places, which is the report's precision.
    assert ground_truth["detection_sample"]["detection_recall_sample"] == pytest.approx(
        5 / 6, abs=1e-4
    )

    dense = next(clip for clip in record["clips"] if clip["clip"]["clip_id"] == "dense")
    assert dense["manual_ground_truth"]["manual_ground_truth_status"] == "NOT_ANNOTATED"
    assert "detection_sample" not in dense["manual_ground_truth"]

    rows = list(
        csv.DictReader(io.StringIO((out / "validation_summary.csv").read_text(encoding="utf-8")))
    )
    by_clip = {row["clip_id"]: row for row in rows}
    assert by_clip["ctrl"]["manual_detection_recall_sample"] != ""
    assert by_clip["dense"]["manual_detection_recall_sample"] == ""


def test_report_before_any_run_fails_with_guidance(tmp_path: Path, manifest: Path) -> None:
    code = main(
        ["validation", "report", "--manifest", str(manifest), "--output", str(tmp_path / "none")]
    )

    assert code == EXIT_ERROR


# --- error handling --------------------------------------------------------


def test_a_malformed_manifest_exits_with_an_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")

    assert main(["validation", "run", "--manifest", str(bad), "--output", str(tmp_path)]) == (
        EXIT_ERROR
    )


def test_a_manifest_naming_a_missing_clip_exits_with_an_error(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "manifest_version": MANIFEST_VERSION,
                "clips": [
                    {
                        "clip_id": "gone",
                        "input_path": str(tmp_path / "absent.mp4"),
                        "scene_type": "control",
                        "camera_id": "cam_gone",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main(["validation", "run", "--manifest", str(path), "--output", str(tmp_path)]) == (
        EXIT_ERROR
    )


def test_an_unknown_clip_filter_exits_with_an_error(
    tmp_path: Path, manifest: Path, fake_pipeline: None
) -> None:
    code = main(
        [
            "validation",
            "run",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "out"),
            "--clip",
            "nope",
        ]
    )

    assert code == EXIT_ERROR


def test_run_logs_the_frozen_profile_and_the_absent_ground_truth(
    tmp_path: Path, manifest: Path, fake_pipeline: None, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="atreides_vision")
    out = tmp_path / "artifacts"

    main(
        ["validation", "run", "--manifest", str(manifest), "--output", str(out), "--device", "cpu"]
    )

    text = caplog.text
    assert "V0.2.1 validated defaults" in text
    assert "every analysed frame is written" in text
    assert "NOT_ANNOTATED" in text
    assert "normalized image units" in text

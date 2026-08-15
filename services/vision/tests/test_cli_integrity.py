"""CLI exposure of integrity monitoring, emission sampling and the integrity command."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from atreides_vision.behavior.reporting import EmitEveryFrame, IntervalEmissionPolicy
from atreides_vision.calibration.integrity.config import IntegrityPolicyMode
from atreides_vision.cli import (
    EXIT_ERROR,
    EXIT_OK,
    _build_emission_policy,
    _build_integrity_guard,
    build_parser,
    main,
)
from atreides_vision.errors import BehaviorOutputError
from synthetic_cameras import CAMERA_A, FRAME_A, calibration_for
from synthetic_scenes import rotate, synthetic_scene

CALIBRATION_REGION = [(0.0, 0.0), (12.0, 0.0), (12.0, 9.0), (0.0, 9.0)]


def write_calibration(tmp_path: Path, camera_id: str = "cam_01") -> Path:
    path = tmp_path / "cal.json"
    calibration = calibration_for(
        CAMERA_A, CALIBRATION_REGION, camera_id=camera_id, reference_frame=FRAME_A
    )
    path.write_text(json.dumps(calibration.to_record()), encoding="utf-8")
    return path


def write_scene(tmp_path: Path, name: str, transform: str = "none") -> Path:
    path = tmp_path / name
    scene = synthetic_scene(640, 360, seed=31)
    if transform == "rotate":
        scene = rotate(scene, 6.0)
    assert cv2.imwrite(str(path), scene)
    return path


class TestRunFlags:
    def test_the_new_flags_default_to_off(self) -> None:
        args = build_parser().parse_args(["run", "--input", "i.mp4", "--output", "o.mp4"])
        assert args.calibration_reference is None
        assert args.behavior_output_interval is None
        assert args.integrity_report_only is False

    def test_flags_parse(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--input",
                "i.mp4",
                "--output",
                "o.mp4",
                "--behavior-output",
                "b.jsonl",
                "--calibration",
                "c.json",
                "--calibration-reference",
                "r.png",
                "--integrity-report-only",
                "--behavior-output-interval",
                "0.25",
            ]
        )
        assert args.calibration_reference == Path("r.png")
        assert args.behavior_output_interval == 0.25
        assert args.integrity_report_only is True

    def test_run_help_lists_them(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "--help"])
        out = capsys.readouterr().out
        for flag in (
            "--calibration-reference",
            "--integrity-report-only",
            "--behavior-output-interval",
        ):
            assert flag in out

    def test_reference_without_calibration_is_rejected(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Normalized output makes no metric claim, so there is nothing to guard."""
        code = main(
            [
                "run",
                "--input",
                str(tmp_path / "in.mp4"),
                "--output",
                str(tmp_path / "out.mp4"),
                "--behavior-output",
                str(tmp_path / "b.jsonl"),
                "--calibration-reference",
                str(write_scene(tmp_path, "ref.png")),
                "--camera-id",
                "cam_01",
            ]
        )
        assert code == EXIT_ERROR
        assert "needs --calibration" in caplog.text

    def test_report_only_without_a_reference_is_rejected(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        code = main(
            [
                "run",
                "--input",
                str(tmp_path / "in.mp4"),
                "--output",
                str(tmp_path / "out.mp4"),
                "--behavior-output",
                str(tmp_path / "b.jsonl"),
                "--calibration",
                str(write_calibration(tmp_path)),
                "--integrity-report-only",
            ]
        )
        assert code == EXIT_ERROR
        assert "needs --calibration-reference" in caplog.text

    def test_a_missing_reference_fails_before_the_detector_loads(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Monitoring must not be quietly skipped on a run that asked for it."""
        code = main(
            [
                "run",
                "--input",
                str(tmp_path / "in.mp4"),
                "--output",
                str(tmp_path / "out.mp4"),
                "--behavior-output",
                str(tmp_path / "b.jsonl"),
                "--calibration",
                str(write_calibration(tmp_path)),
                "--calibration-reference",
                str(tmp_path / "absent.png"),
            ]
        )
        assert code == EXIT_ERROR
        assert "Reference frame not found" in caplog.text


class TestBuilders:
    def test_no_reference_means_no_guard(self) -> None:
        assert _build_integrity_guard(None, report_only=False) is None

    def test_a_reference_builds_a_guarded_monitor(self, tmp_path: Path) -> None:
        guard = _build_integrity_guard(write_scene(tmp_path, "ref.png"), report_only=False)
        assert guard is not None
        assert guard.mode is IntegrityPolicyMode.GUARDED
        assert guard.metric_output_allowed

    def test_report_only_is_selectable(self, tmp_path: Path) -> None:
        guard = _build_integrity_guard(write_scene(tmp_path, "ref.png"), report_only=True)
        assert guard is not None
        assert guard.mode is IntegrityPolicyMode.REPORT_ONLY

    def test_the_reference_id_records_the_path(self, tmp_path: Path) -> None:
        path = write_scene(tmp_path, "ref.png")
        guard = _build_integrity_guard(path, report_only=False)
        assert guard is not None
        assert guard.monitor.reference_id == str(path)  # type: ignore[attr-defined]

    def test_default_emission_is_every_frame(self) -> None:
        assert isinstance(_build_emission_policy(None), EmitEveryFrame)

    def test_an_interval_builds_an_interval_policy(self) -> None:
        policy = _build_emission_policy(0.5)
        assert isinstance(policy, IntervalEmissionPolicy)
        assert policy.interval_seconds == 0.5

    def test_a_non_positive_interval_is_rejected(self) -> None:
        with pytest.raises(BehaviorOutputError, match="must be positive"):
            _build_emission_policy(0.0)


class TestIntegrityCommand:
    def test_reports_a_consistent_view(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        calibration = write_calibration(tmp_path)
        reference = write_scene(tmp_path, "ref.png")
        with caplog.at_level(logging.INFO):
            code = main(
                [
                    "calibration",
                    "integrity",
                    "--calibration",
                    str(calibration),
                    "--reference",
                    str(reference),
                    "--current",
                    str(reference),
                ]
            )
        assert code == EXIT_OK
        text = caplog.text
        assert "State:       consistent" in text
        assert "Features:" in text
        assert "Matching:" in text
        assert "Drift:" in text

    def test_reports_detected_drift(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        calibration = write_calibration(tmp_path)
        reference = write_scene(tmp_path, "ref.png")
        moved = write_scene(tmp_path, "moved.png", transform="rotate")
        with caplog.at_level(logging.INFO):
            code = main(
                [
                    "calibration",
                    "integrity",
                    "--calibration",
                    str(calibration),
                    "--reference",
                    str(reference),
                    "--current",
                    str(moved),
                ]
            )
        assert code == EXIT_OK
        assert "State:       drift_detected" in caplog.text

    def test_it_does_not_claim_physical_accuracy(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        calibration = write_calibration(tmp_path)
        reference = write_scene(tmp_path, "ref.png")
        with caplog.at_level(logging.INFO):
            main(
                [
                    "calibration",
                    "integrity",
                    "--calibration",
                    str(calibration),
                    "--reference",
                    str(reference),
                    "--current",
                    str(reference),
                ]
            )
        text = caplog.text
        assert "not whether the calibration's metres are physically correct" in text
        for verdict in ("GOOD", "BAD", "SAFE", "PRODUCTION READY", "ACCURATE"):
            assert verdict not in text

    def test_it_writes_a_diagnostic_image(self, tmp_path: Path) -> None:
        calibration = write_calibration(tmp_path)
        reference = write_scene(tmp_path, "ref.png")
        moved = write_scene(tmp_path, "moved.png", transform="rotate")
        output = tmp_path / "integrity.png"

        assert (
            main(
                [
                    "calibration",
                    "integrity",
                    "--calibration",
                    str(calibration),
                    "--reference",
                    str(reference),
                    "--current",
                    str(moved),
                    "--output",
                    str(output),
                ]
            )
            == EXIT_OK
        )
        rendered = cv2.imread(str(output))
        original = cv2.imread(str(moved))
        assert rendered is not None and original is not None
        assert not np.array_equal(rendered, original)

    def test_a_missing_reference_exits_with_an_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        code = main(
            [
                "calibration",
                "integrity",
                "--calibration",
                str(write_calibration(tmp_path)),
                "--reference",
                str(tmp_path / "absent.png"),
                "--current",
                str(write_scene(tmp_path, "cur.png")),
            ]
        )
        assert code == EXIT_ERROR
        assert "Reference frame not found" in caplog.text

    def test_a_missing_calibration_exits_with_an_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        reference = write_scene(tmp_path, "ref.png")
        code = main(
            [
                "calibration",
                "integrity",
                "--calibration",
                str(tmp_path / "absent.json"),
                "--reference",
                str(reference),
                "--current",
                str(reference),
            ]
        )
        assert code == EXIT_ERROR
        assert "Calibration file not found" in caplog.text

    @pytest.mark.parametrize("missing", ["--calibration", "--reference", "--current"])
    def test_every_argument_is_required(self, missing: str) -> None:
        argv = [
            "calibration",
            "integrity",
            "--calibration",
            "c.json",
            "--reference",
            "r.png",
            "--current",
            "n.png",
        ]
        index = argv.index(missing)
        del argv[index : index + 2]
        with pytest.raises(SystemExit):
            build_parser().parse_args(argv)

    def test_a_resize_is_not_reported_as_drift(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The V0.3.1 resolution contract, through the command."""
        calibration = write_calibration(tmp_path)
        reference = write_scene(tmp_path, "ref.png")
        loaded = cv2.imread(str(reference))
        assert loaded is not None
        resized = tmp_path / "half.png"
        assert cv2.imwrite(str(resized), cv2.resize(loaded, (320, 180)))

        with caplog.at_level(logging.INFO):
            main(
                [
                    "calibration",
                    "integrity",
                    "--calibration",
                    str(calibration),
                    "--reference",
                    str(reference),
                    "--current",
                    str(resized),
                ]
            )
        assert "State:       consistent" in caplog.text

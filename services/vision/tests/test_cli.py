"""CLI argument parsing and error surfacing.

No test here loads a model; the ``run`` handler is replaced where the flow
itself is under test.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from atreides_vision.cli import (
    EXIT_ERROR,
    EXIT_OK,
    _build_tracker_config,
    build_parser,
    build_tracker,
    main,
)
from atreides_vision.config import (
    DEFAULT_DETECTION_CONFIDENCE,
    DEFAULT_OCSORT_HIGH_CONF_THRESHOLD,
    DEFAULT_TRACKER,
    Device,
    PipelineConfig,
    Tracker,
    TrackerConfig,
)
from atreides_vision.errors import (
    ConfigurationError,
    DeviceUnavailableError,
    VideoSourceError,
)
from atreides_vision.tracking.base import MultiObjectTracker
from atreides_vision.tracking.bytetrack import ByteTrackTracker
from atreides_vision.tracking.ocsort import OcSortTracker


class TestParser:
    def test_requires_a_subcommand(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_help_works(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args(["--help"])
        assert excinfo.value.code == 0
        assert "SEE + TRACK" in capsys.readouterr().out

    def test_run_help_lists_the_core_flags(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "--help"])
        out = capsys.readouterr().out
        for flag in ("--input", "--output", "--confidence", "--device"):
            assert flag in out

    def test_parses_the_documented_invocation(self) -> None:
        args = build_parser().parse_args(
            ["run", "--input", "data/input/test.mp4", "--output", "artifacts/tracked.mp4"]
        )
        assert args.input == Path("data/input/test.mp4")
        assert args.output == Path("artifacts/tracked.mp4")
        assert args.confidence == DEFAULT_DETECTION_CONFIDENCE
        assert args.device is Device.AUTO

    def test_input_and_output_are_required(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "--input", "a.mp4"])
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "--output", "b.mp4"])

    @pytest.mark.parametrize("device", ["auto", "cpu", "mps", "cuda"])
    def test_accepts_every_documented_device(self, device: str) -> None:
        args = build_parser().parse_args(
            ["run", "--input", "a.mp4", "--output", "b.mp4", "--device", device]
        )
        assert args.device is Device(device)

    def test_rejects_an_unknown_device(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["run", "--input", "a.mp4", "--output", "b.mp4", "--device", "tpu"]
            )
        assert "invalid" in capsys.readouterr().err.lower()

    def test_optional_flags_parse(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--input",
                "a.mp4",
                "--output",
                "b.mp4",
                "--confidence",
                "0.35",
                "--max-frames",
                "50",
                "--trail-length",
                "10",
                "--no-confidence-labels",
                "--log-every",
                "25",
                "--verbose",
            ]
        )
        assert args.confidence == 0.35
        assert args.max_frames == 50
        assert args.trail_length == 10
        assert args.no_confidence_labels is True
        assert args.log_every == 25
        assert args.verbose is True

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args(["--version"])
        assert excinfo.value.code == 0
        assert "atreides-vision" in capsys.readouterr().out


class TestZoneFlags:
    def test_zones_default_to_off(self) -> None:
        """V0.1 usage must keep working untouched."""
        args = build_parser().parse_args(["run", "--input", "a.mp4", "--output", "b.mp4"])
        assert args.zones is None
        assert args.metrics_output is None

    def test_parses_the_documented_v02_invocation(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--input",
                "data/input/test.mp4",
                "--output",
                "artifacts/zones.mp4",
                "--zones",
                "config/zones.example.json",
            ]
        )
        assert args.zones == Path("config/zones.example.json")
        assert args.output == Path("artifacts/zones.mp4")

    def test_parses_metrics_flags(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--input",
                "a.mp4",
                "--output",
                "b.mp4",
                "--zones",
                "z.json",
                "--metrics-output",
                "artifacts/zone_metrics.jsonl",
                "--metrics-interval",
                "0.5",
            ]
        )
        assert args.metrics_output == Path("artifacts/zone_metrics.jsonl")
        assert args.metrics_interval == 0.5

    def test_metrics_interval_defaults_to_one_second(self) -> None:
        args = build_parser().parse_args(["run", "--input", "a.mp4", "--output", "b.mp4"])
        assert args.metrics_interval == 1.0

    def test_run_help_lists_the_zone_flags(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "--help"])
        out = capsys.readouterr().out
        for flag in ("--zones", "--metrics-output", "--metrics-interval"):
            assert flag in out


class TestTrackerSelection:
    def test_default_is_ocsort(self) -> None:
        """V0.2.1 promoted OC-SORT to the default after three-scene validation."""
        args = build_parser().parse_args(["run", "--input", "a.mp4", "--output", "b.mp4"])
        assert args.tracker is Tracker.OCSORT
        assert args.tracker is DEFAULT_TRACKER

    def test_default_builds_the_ocsort_adapter(self) -> None:
        args = build_parser().parse_args(["run", "--input", "a.mp4", "--output", "b.mp4"])
        tracker = build_tracker(args.tracker, TrackerConfig())
        assert isinstance(tracker, OcSortTracker)
        assert tracker.name == "OC-SORT"

    def test_bytetrack_remains_selectable_and_builds(self) -> None:
        """Promoting OC-SORT must not remove the ByteTrack path."""
        args = build_parser().parse_args(
            ["run", "--input", "a.mp4", "--output", "b.mp4", "--tracker", "bytetrack"]
        )
        assert args.tracker is Tracker.BYTETRACK
        tracker = build_tracker(args.tracker, TrackerConfig())
        assert isinstance(tracker, ByteTrackTracker)
        assert tracker.name == "ByteTrack"

    def test_accepts_ocsort(self) -> None:
        args = build_parser().parse_args(
            ["run", "--input", "a.mp4", "--output", "b.mp4", "--tracker", "ocsort"]
        )
        assert args.tracker is Tracker.OCSORT

    def test_ocsort_builds_the_ocsort_adapter(self) -> None:
        tracker = build_tracker(Tracker.OCSORT, TrackerConfig())
        assert isinstance(tracker, OcSortTracker)
        assert tracker.name == "OC-SORT"

    def test_accepts_bytetrack_explicitly(self) -> None:
        args = build_parser().parse_args(
            ["run", "--input", "a.mp4", "--output", "b.mp4", "--tracker", "bytetrack"]
        )
        assert args.tracker is Tracker.BYTETRACK

    def test_rejects_an_unknown_tracker(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["run", "--input", "a.mp4", "--output", "b.mp4", "--tracker", "deepsort"]
            )
        assert "invalid" in capsys.readouterr().err.lower()

    def test_both_trackers_satisfy_the_same_interface(self) -> None:
        """The A/B swap is one argument precisely because of this."""
        for choice in (Tracker.BYTETRACK, Tracker.OCSORT):
            assert isinstance(build_tracker(choice, TrackerConfig()), MultiObjectTracker)

    def test_tracking_debug_defaults_to_off(self) -> None:
        args = build_parser().parse_args(["run", "--input", "a.mp4", "--output", "b.mp4"])
        assert args.tracking_debug_output is None

    def test_accepts_tracking_debug_output(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--input",
                "a.mp4",
                "--output",
                "b.mp4",
                "--tracker",
                "ocsort",
                "--tracking-debug-output",
                "artifacts/tracking-debug-ocsort.jsonl",
            ]
        )
        assert args.tracking_debug_output == Path("artifacts/tracking-debug-ocsort.jsonl")
        assert args.tracker is Tracker.OCSORT

    def test_run_help_lists_the_tracking_debug_flag(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "--help"])
        assert "--tracking-debug-output" in capsys.readouterr().out

    def test_ocsort_high_conf_override_defaults_to_off(self) -> None:
        """Omitting the flag must leave the permanent default in place."""
        args = build_parser().parse_args(["run", "--input", "a.mp4", "--output", "b.mp4"])
        assert args.ocsort_high_conf_threshold is None
        assert _build_tracker_config(30.0, args.ocsort_high_conf_threshold) == TrackerConfig(
            frame_rate=30.0
        )
        assert (
            _build_tracker_config(30.0, None).high_conf_det_threshold
            == DEFAULT_OCSORT_HIGH_CONF_THRESHOLD
        )

    def test_ocsort_high_conf_override_is_applied(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--input",
                "a.mp4",
                "--output",
                "b.mp4",
                "--tracker",
                "ocsort",
                "--ocsort-high-conf-threshold",
                "0.30",
            ]
        )
        assert args.ocsort_high_conf_threshold == 0.30
        config = _build_tracker_config(30.0, args.ocsort_high_conf_threshold)
        assert config.high_conf_det_threshold == 0.30

    def test_override_changes_nothing_else(self) -> None:
        """The experiment must move exactly one knob."""
        baseline = _build_tracker_config(30.0, None)
        overridden = _build_tracker_config(30.0, 0.55)
        assert overridden.high_conf_det_threshold == 0.55
        assert baseline.high_conf_det_threshold == DEFAULT_OCSORT_HIGH_CONF_THRESHOLD
        for field in (
            "frame_rate",
            "lost_track_buffer",
            "minimum_consecutive_frames",
            "minimum_iou_threshold",
            "ocsort_minimum_iou_threshold",
            "direction_consistency_weight",
            "delta_t",
            "track_activation_threshold",
        ):
            assert getattr(overridden, field) == getattr(baseline, field), field

    def test_override_is_rejected_when_out_of_range(self) -> None:
        with pytest.raises(ConfigurationError, match="high_conf_det_threshold"):
            _build_tracker_config(30.0, 1.5)

    def test_run_help_lists_the_ocsort_override(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "--help"])
        assert "--ocsort-high-conf-threshold" in capsys.readouterr().out

    def test_deduplication_defaults_to_on(self) -> None:
        """V0.2.1 promoted Rule-B3 deduplication to the default."""
        args = build_parser().parse_args(["run", "--input", "a.mp4", "--output", "b.mp4"])
        assert args.deduplicate_detections is True
        assert PipelineConfig(Path("a.mp4"), Path("b.mp4")).deduplicate_detections is True

    def test_deduplication_flag_still_accepted(self) -> None:
        """The pre-V0.2.1 opt-in flag must keep working for existing scripts."""
        args = build_parser().parse_args(
            ["run", "--input", "a.mp4", "--output", "b.mp4", "--deduplicate-detections"]
        )
        assert args.deduplicate_detections is True

    def test_deduplication_can_be_disabled(self) -> None:
        args = build_parser().parse_args(
            ["run", "--input", "a.mp4", "--output", "b.mp4", "--no-deduplicate-detections"]
        )
        assert args.deduplicate_detections is False

    def test_run_help_lists_the_deduplication_flag(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "--help"])
        assert "--deduplicate-detections" in capsys.readouterr().out

    def test_run_help_lists_the_tracker_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "--help"])
        out = capsys.readouterr().out
        assert "--tracker" in out
        assert "bytetrack" in out
        assert "ocsort" in out

    def test_documented_ocsort_invocation_parses(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--input",
                "data/input/validation/control_baseline_01.mp4",
                "--output",
                "artifacts/validation-baseline-ocsort.mp4",
                "--zones",
                "config/zones.example.json",
                "--metrics-output",
                "artifacts/validation-baseline-ocsort.jsonl",
                "--tracker",
                "ocsort",
            ]
        )
        assert args.tracker is Tracker.OCSORT
        assert args.zones == Path("config/zones.example.json")
        assert args.metrics_output == Path("artifacts/validation-baseline-ocsort.jsonl")


class TestErrorSurfacing:
    """Domain errors must reach the operator as a message, not a traceback."""

    def _run_with_handler(
        self, monkeypatch: pytest.MonkeyPatch, handler: Callable[[argparse.Namespace], int]
    ) -> int:
        # build_parser() binds the handler by name at call time, and main() calls
        # it after this patch lands — so swapping the module attribute is enough.
        monkeypatch.setattr("atreides_vision.cli._run_command", handler)
        return main(["run", "--input", "a.mp4", "--output", "b.mp4"])

    def test_missing_video_exits_cleanly(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def handler(_args: argparse.Namespace) -> int:
            raise VideoSourceError("Input video not found: a.mp4")

        with caplog.at_level(logging.ERROR, logger="atreides_vision"):
            code = self._run_with_handler(monkeypatch, handler)

        assert code == EXIT_ERROR
        assert "Input video not found" in caplog.text

    def test_unavailable_device_exits_cleanly(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def handler(_args: argparse.Namespace) -> int:
            raise DeviceUnavailableError("CUDA was requested but ... --device cpu")

        with caplog.at_level(logging.ERROR, logger="atreides_vision"):
            code = self._run_with_handler(monkeypatch, handler)

        assert code == EXIT_ERROR
        assert "--device cpu" in caplog.text

    def test_unexpected_errors_are_not_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(_args: argparse.Namespace) -> int:
            raise ZeroDivisionError("a real bug")

        with pytest.raises(ZeroDivisionError):
            self._run_with_handler(monkeypatch, handler)

    def test_successful_run_returns_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._run_with_handler(monkeypatch, lambda _args: EXIT_OK) == EXIT_OK


class TestRealMissingInput:
    """End-to-end through the real handler, failing before any model is loaded."""

    def test_nonexistent_input_reports_clearly(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR, logger="atreides_vision"):
            code = main(
                [
                    "run",
                    "--input",
                    str(tmp_path / "missing.mp4"),
                    "--output",
                    str(tmp_path / "out.mp4"),
                    "--device",
                    "cpu",
                ]
            )
        assert code == EXIT_ERROR
        assert "not found" in caplog.text


class TestZoneErrorsBeforeModelLoad:
    """Bad zone input must fail fast, before weights or video are touched."""

    def _run(self, caplog: pytest.LogCaptureFixture, *extra: str) -> int:
        with caplog.at_level(logging.ERROR, logger="atreides_vision"):
            return main(
                ["run", "--input", "missing.mp4", "--output", "out.mp4", "--device", "cpu", *extra]
            )

    def test_missing_zone_file_reports_clearly(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        code = self._run(caplog, "--zones", str(tmp_path / "nope.json"))
        assert code == EXIT_ERROR
        assert "Zone configuration not found" in caplog.text

    def test_invalid_zone_file_reports_the_field(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        bad = tmp_path / "zones.json"
        bad.write_text('{"camera_id": "cam_01"}', encoding="utf-8")
        code = self._run(caplog, "--zones", str(bad))
        assert code == EXIT_ERROR
        assert "missing required field 'zones'" in caplog.text

    def test_metrics_output_without_zones_is_rejected(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        code = self._run(caplog, "--metrics-output", str(tmp_path / "m.jsonl"))
        assert code == EXIT_ERROR
        assert "--metrics-output needs zones" in caplog.text

    def test_zone_config_is_validated_before_the_video_is_opened(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Both input and zones are bad; the zone error must be the one reported."""
        bad = tmp_path / "zones.json"
        bad.write_text("{ not json", encoding="utf-8")
        code = self._run(caplog, "--zones", str(bad))
        assert code == EXIT_ERROR
        assert "not valid JSON" in caplog.text
        assert "Input video not found" not in caplog.text

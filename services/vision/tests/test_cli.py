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

from atreides_vision.cli import EXIT_ERROR, EXIT_OK, build_parser, main
from atreides_vision.config import Device
from atreides_vision.errors import DeviceUnavailableError, VideoSourceError


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
        assert args.confidence == 0.5
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

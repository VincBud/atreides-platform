"""Loading ground-plane calibration from disk.

File I/O and JSON parsing live here so that
:mod:`atreides_vision.calibration.models` stays a pure domain module, mirroring
how :mod:`atreides_vision.zones.loader` relates to the zone models. Every
failure mode — missing file, unreadable file, broken JSON, invalid content,
unsolvable geometry — becomes a :class:`CalibrationError` whose message says
what to fix.

Loading is strict on purpose: it solves the homography rather than only parsing
the file, so a calibration that parses but cannot work is rejected at startup
instead of at the first frame.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from atreides_vision.calibration.fitting import CalibrationFit, fit_calibration
from atreides_vision.calibration.models import GroundPlaneCalibration
from atreides_vision.errors import CalibrationError

logger = logging.getLogger(__name__)

__all__ = ["load_calibration", "load_calibration_fit"]


def load_calibration(path: Path | str) -> GroundPlaneCalibration:
    """Read and validate a calibration file, without solving it.

    Args:
        path: Path to a JSON file shaped like ``config/calibration.example.json``.

    Returns:
        The validated calibration.

    Raises:
        CalibrationError: if the file cannot be read, is not valid JSON, or
            does not describe a valid set of correspondences.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise CalibrationError(
            f"Calibration file not found: {config_path}. "
            "Copy config/calibration.example.json and measure the reference points "
            "for your own camera."
        )
    if not config_path.is_file():
        raise CalibrationError(f"Calibration path is not a file: {config_path}")

    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CalibrationError(f"Could not read calibration {config_path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CalibrationError(
            f"Calibration {config_path} is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc

    try:
        return GroundPlaneCalibration.from_mapping(data)
    except CalibrationError as exc:
        # Re-raise with the file name attached; the inner message has the field path.
        raise CalibrationError(f"Invalid calibration {config_path}: {exc}") from exc


def load_calibration_fit(path: Path | str) -> CalibrationFit:
    """Read, validate **and solve** a calibration file.

    The form a pipeline wants: an unsolvable calibration fails here, at
    startup, rather than part-way through a run.

    Raises:
        CalibrationError: for any parse, validation or geometry failure.
    """
    calibration = load_calibration(path)
    try:
        fit = fit_calibration(calibration)
    except CalibrationError as exc:
        raise CalibrationError(f"Invalid calibration {Path(path)}: {exc}") from exc

    logger.debug(
        "Loaded calibration for camera '%s' in venue '%s': %d points, mean error %.4f m",
        fit.camera_id,
        fit.venue_id,
        fit.quality.point_count,
        fit.quality.mean_venue_error_m,
    )
    return fit

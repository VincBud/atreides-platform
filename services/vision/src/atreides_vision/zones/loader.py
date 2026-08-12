"""Loading zone configuration from disk.

File I/O and JSON parsing live here so that :mod:`atreides_vision.zones.models`
stays a pure domain module. Every failure mode — missing file, unreadable file,
broken JSON, invalid content — becomes a :class:`ZoneConfigurationError` whose
message tells the operator what to fix.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from atreides_vision.errors import ZoneConfigurationError
from atreides_vision.zones.models import ZoneLayout

logger = logging.getLogger(__name__)

__all__ = ["load_zone_layout"]


def load_zone_layout(path: Path | str) -> ZoneLayout:
    """Read and validate a zone configuration file.

    Args:
        path: Path to a JSON file shaped like ``config/zones.example.json``.

    Returns:
        The validated layout.

    Raises:
        ZoneConfigurationError: if the file cannot be read, is not valid JSON,
            or does not describe a valid set of zones.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise ZoneConfigurationError(
            f"Zone configuration not found: {config_path}. "
            "Copy config/zones.example.json and edit it for your camera."
        )
    if not config_path.is_file():
        raise ZoneConfigurationError(f"Zone configuration path is not a file: {config_path}")

    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ZoneConfigurationError(
            f"Could not read zone configuration {config_path}: {exc}"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ZoneConfigurationError(
            f"Zone configuration {config_path} is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc

    try:
        layout = ZoneLayout.from_mapping(data)
    except ZoneConfigurationError as exc:
        # Re-raise with the file name attached; the inner message has the field path.
        raise ZoneConfigurationError(f"Invalid zone configuration {config_path}: {exc}") from exc

    if not layout.zones:
        logger.warning(
            "Zone configuration %s defines no zones; nothing will be measured.", config_path
        )

    return layout

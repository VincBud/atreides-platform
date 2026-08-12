"""Zone intelligence — V0.2 SEE + TRACK + UNDERSTAND WHERE ACTIVITY OCCURS.

Operator-defined regions of a camera view, and descriptive crowd metrics for
each: occupancy, entries, exits, dwell time, image-plane movement speed and a
relative motion level.

This layer consumes :class:`~atreides_vision.models.Track` objects and nothing
else. It has no dependency on RF-DETR, ByteTrack, or any detector or tracker,
so those remain replaceable.

Scope boundaries carried forward from V0.1 and extended: these metrics describe
*where activity happens*, never *who* is involved. No identity, biometrics,
demographics, intent, or cross-camera matching. Metrics are derived from
anonymous box geometry and ephemeral, session-local track ids, and no
individual track history leaves the running process.
"""

from __future__ import annotations

from atreides_vision.zones.analyzer import ZoneAnalyzer
from atreides_vision.zones.config import ZoneAnalysisConfig
from atreides_vision.zones.geometry import (
    NormalizedPoint,
    bottom_center,
    point_in_polygon,
    to_normalized,
    track_floor_point,
)
from atreides_vision.zones.loader import load_zone_layout
from atreides_vision.zones.models import (
    Zone,
    ZoneLayout,
    ZoneMembership,
    ZonePolygon,
    ZoneReport,
    ZoneSnapshot,
)
from atreides_vision.zones.reporting import JsonlMetricsWriter, MetricsSampler, MetricsWriter

__all__ = [
    "JsonlMetricsWriter",
    "MetricsSampler",
    "MetricsWriter",
    "NormalizedPoint",
    "Zone",
    "ZoneAnalysisConfig",
    "ZoneAnalyzer",
    "ZoneLayout",
    "ZoneMembership",
    "ZonePolygon",
    "ZoneReport",
    "ZoneSnapshot",
    "bottom_center",
    "load_zone_layout",
    "point_in_polygon",
    "to_normalized",
    "track_floor_point",
]

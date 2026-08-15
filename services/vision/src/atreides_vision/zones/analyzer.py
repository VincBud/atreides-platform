"""The zone analyzer — the single component :class:`VideoProcessor` talks to.

It owns the whole zone pipeline for one session::

    tracks -> floor points -> zone assignment -> state update -> metrics

and hands back a :class:`ZoneReport` per frame. Keeping this behind one
interface is what stops ``VideoProcessor`` from growing a second job: the
processor calls :meth:`ZoneAnalyzer.update` and passes the result to the
annotator, and knows nothing about polygons, grace periods or smoothing.

The analyzer depends only on :class:`~atreides_vision.models.Track`. It has no
knowledge of RF-DETR, ByteTrack, or any detector or tracker, so replacing
either leaves zone intelligence untouched.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from atreides_vision.models import Track
from atreides_vision.zones.assignment import ZoneAssigner
from atreides_vision.zones.config import ZoneAnalysisConfig
from atreides_vision.zones.metrics import ZoneMetricsCalculator
from atreides_vision.zones.models import ZoneLayout, ZoneReport
from atreides_vision.zones.state import TrackMovementStore, ZoneStateStore

logger = logging.getLogger(__name__)

__all__ = ["ZoneAnalyzer"]


class ZoneAnalyzer:
    """Per-frame zone assignment, state and metrics for one camera session."""

    def __init__(
        self,
        layout: ZoneLayout,
        config: ZoneAnalysisConfig | None = None,
    ) -> None:
        self._layout = layout
        self._config = config or ZoneAnalysisConfig()
        self._assigner = ZoneAssigner(layout)
        self._state = ZoneStateStore(layout.zone_ids, self._config)
        self._movement = TrackMovementStore(self._config)
        self._metrics = ZoneMetricsCalculator(self._config)
        self._latest: ZoneReport | None = None
        self._latest_zone_ids: dict[int, tuple[str, ...]] = {}

    @property
    def layout(self) -> ZoneLayout:
        return self._layout

    @property
    def config(self) -> ZoneAnalysisConfig:
        return self._config

    @property
    def latest_report(self) -> ZoneReport | None:
        """The most recent report, or ``None`` before the first frame."""
        return self._latest

    @property
    def zone_ids_by_track(self) -> Mapping[int, tuple[str, ...]]:
        """Which zones each track was in on the most recent frame.

        Exposed so a caller can attach zone labels to something else — V0.3.3
        passes them to behaviour analysis as neutral context. Tracks in no zone
        are absent from the mapping rather than present with an empty tuple, so
        "outside every zone" needs no sentinel value.

        Ids are ordered as the zones are configured, so the same scene always
        produces the same ordering.

        These are **camera-local, image-space** zones: a zone is a polygon drawn
        on this camera's view. The id is a label, not a venue geometry, and
        attaching it to a metric position does not make it one.
        """
        return self._latest_zone_ids

    def update(
        self,
        tracks: Sequence[Track],
        frame_index: int,
        timestamp_seconds: float,
        frame_width: int,
        frame_height: int,
    ) -> ZoneReport:
        """Advance zone analysis by one frame.

        Args:
            tracks: Confirmed tracks for this frame.
            frame_index: 1-based frame number.
            timestamp_seconds: Source-time position of this frame. Derived from
                the source frame rate, never assumed — dwell and speed are only
                correct because this is a real source timestamp.
            frame_width: Source frame width in pixels.
            frame_height: Source frame height in pixels.

        Returns:
            Metrics for every configured zone.
        """
        assignment = self._assigner.assign(tracks, frame_width, frame_height, frame_index)
        self._latest_zone_ids = {
            track_id: zone_ids
            for track_id in assignment.floor_points
            if (zone_ids := assignment.zones_for_track(track_id))
        }

        for track_id, point in assignment.floor_points.items():
            self._movement.observe(track_id, point, frame_index, timestamp_seconds)

        self._state.update(assignment.memberships, frame_index, timestamp_seconds)
        self._movement.prune(frame_index)

        report = self._metrics.report(
            self._layout, self._state, self._movement, frame_index, timestamp_seconds
        )
        self._latest = report
        return report

    def log_summary(self) -> None:
        """Log a final per-zone summary. Called once at the end of a run."""
        report = self._latest
        if report is None:
            logger.info("Zones:               no frames analysed")
            return
        for snapshot in report.snapshots:
            logger.info(
                "Zone %-14s occupancy=%-3d entries=%-4d exits=%-4d avg dwell=%.1fs motion=%.2f",
                snapshot.zone_name,
                snapshot.occupancy,
                snapshot.entries,
                snapshot.exits,
                snapshot.average_dwell_seconds,
                snapshot.motion_level,
            )

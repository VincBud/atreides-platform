"""Turning zone state into the numbers an operator reads.

What the metrics are, precisely:

``occupancy``
    Tracks currently considered present in the zone (see
    :mod:`atreides_vision.zones.state` for what "currently" means).

``entries`` / ``exits``
    Cumulative transitions this session. A track continuously present is
    counted once, on the frame it first appears inside.

``average_dwell_seconds`` / ``max_dwell_seconds``
    Over the tracks currently in the zone, how long each has been there.
    Computed from source timestamps, so they are correct at any frame rate.

``mean_normalized_speed``
    Average image-plane speed of those tracks, in normalized frame units per
    second. **Not** metres per second — see :class:`TrackMovementStore`.

``motion_level``
    A smoothed 0..1 summary of how much movement there is, relative to a
    provisional reference speed. Descriptive only.
"""

from __future__ import annotations

from atreides_vision.zones.config import ZoneAnalysisConfig
from atreides_vision.zones.models import Zone, ZoneLayout, ZoneReport, ZoneSnapshot
from atreides_vision.zones.state import TrackMovementStore, ZoneStateStore

__all__ = ["MotionLevelEstimator", "ZoneMetricsCalculator"]


class MotionLevelEstimator:
    """A bounded, smoothed descriptor of movement activity in a zone.

    ``motion_level`` answers "how much are people in this zone moving, compared
    with brisk movement across this camera's view". It is deliberately simple
    and explainable: mean speed divided by a reference speed, clamped to
    ``[0, 1]``, then exponentially smoothed so the displayed number is stable.

    It is **not** a risk, threat, anomaly or safety score, and must not be
    presented as one. A busy dancefloor scores high because dancing is
    movement. An empty zone scores 0.0 because there is nobody in it.

    The reference speed is a provisional presentation constant, exposed through
    :class:`ZoneAnalysisConfig` so it can be re-tuned per camera.
    """

    def __init__(self, config: ZoneAnalysisConfig | None = None) -> None:
        self._config = config or ZoneAnalysisConfig()
        # Standard EMA coefficient for an N-frame span.
        self._alpha = 2.0 / (self._config.motion_smoothing_frames + 1.0)
        self._levels: dict[str, float] = {}

    def update(self, zone_id: str, mean_speed: float) -> float:
        """Fold this frame's mean speed in and return the smoothed level."""
        raw = _clamp(mean_speed / self._config.motion_reference_speed)
        previous = self._levels.get(zone_id)
        level = raw if previous is None else previous + self._alpha * (raw - previous)
        # Clamped again so floating-point drift can never escape [0, 1].
        level = _clamp(level)
        self._levels[zone_id] = level
        return level

    def level(self, zone_id: str) -> float:
        return self._levels.get(zone_id, 0.0)

    def forget(self, zone_id: str) -> None:
        self._levels.pop(zone_id, None)


class ZoneMetricsCalculator:
    """Builds a :class:`ZoneReport` from the current zone and movement state."""

    def __init__(
        self,
        config: ZoneAnalysisConfig | None = None,
        motion: MotionLevelEstimator | None = None,
    ) -> None:
        self._config = config or ZoneAnalysisConfig()
        self._motion = motion or MotionLevelEstimator(self._config)

    @property
    def motion(self) -> MotionLevelEstimator:
        return self._motion

    def snapshot(
        self,
        zone: Zone,
        state: ZoneStateStore,
        movement: TrackMovementStore,
        timestamp_seconds: float,
    ) -> ZoneSnapshot:
        """Metrics for one zone at ``timestamp_seconds``."""
        members = state.member_ids(zone.id)
        mean_speed = movement.mean_speed(members)
        motion_level = self._motion.update(zone.id, mean_speed)
        average_dwell, max_dwell = state.dwell_stats(zone.id, timestamp_seconds)

        return ZoneSnapshot(
            zone_id=zone.id,
            zone_name=zone.name,
            occupancy=len(members),
            entries=state.entries(zone.id),
            exits=state.exits(zone.id),
            mean_normalized_speed=mean_speed,
            motion_level=motion_level,
            average_dwell_seconds=average_dwell,
            max_dwell_seconds=max_dwell,
            completed_visits=state.completed_visits(zone.id),
        )

    def report(
        self,
        layout: ZoneLayout,
        state: ZoneStateStore,
        movement: TrackMovementStore,
        frame_index: int,
        timestamp_seconds: float,
    ) -> ZoneReport:
        """Metrics for every configured zone, in configuration order."""
        return ZoneReport(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            snapshots=tuple(
                self.snapshot(zone, state, movement, timestamp_seconds) for zone in layout
            ),
        )


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))

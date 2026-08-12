"""Tunable parameters for zone analysis.

Every value that shapes a metric lives here with a stated rationale, rather
than as an unexplained constant inside an algorithm. Defaults are deliberately
conservative: they smooth over tracker noise rather than chase responsiveness.
"""

from __future__ import annotations

from dataclasses import dataclass

from atreides_vision.errors import ConfigurationError

__all__ = ["ZoneAnalysisConfig"]


@dataclass(frozen=True, slots=True)
class ZoneAnalysisConfig:
    """Settings for zone membership, dwell, speed and motion level."""

    #: Consecutive frames a track may go unobserved inside a zone before it is
    #: counted as having exited. ByteTrack drops a box for a frame or two under
    #: partial occlusion, and a person standing on a zone edge flickers in and
    #: out; without tolerance both produce bursts of phantom entries and exits.
    #: The cost is that a genuine exit is reported this many frames late.
    exit_grace_frames: int = 8

    #: Time window over which a track's movement speed is averaged. Roughly one
    #: second smooths bounding-box jitter while still reacting to real movement.
    speed_window_seconds: float = 1.0

    #: PROVISIONAL. The mean normalized speed treated as "fully active" when
    #: mapping speed onto ``motion_level``'s 0..1 range. 0.15 frame-widths per
    #: second is brisk walking across a typical wide venue shot. It is a
    #: presentation constant chosen by eye, not a calibrated measurement, and
    #: should be re-tuned per camera once there is footage to tune against.
    motion_reference_speed: float = 0.15

    #: Frames over which motion level is exponentially smoothed, so the number
    #: on screen does not flicker frame to frame.
    motion_smoothing_frames: int = 25

    #: How often zone metrics are written to the metrics stream. Sampling once
    #: per second keeps the file small and readable; it does not affect the
    #: metrics themselves, which are computed every frame.
    metrics_sample_interval_seconds: float = 1.0

    #: Frames after which an unseen track's movement history is discarded.
    #: Bounds memory on long runs where track ids churn.
    stale_track_ttl_frames: int = 150

    #: Hard ceiling on tracks retained in movement history, as a second bound.
    max_tracked_movements: int = 1024

    #: Completed visits retained per zone for session statistics.
    max_completed_visits_per_zone: int = 512

    def __post_init__(self) -> None:
        if self.exit_grace_frames < 0:
            raise ConfigurationError(
                f"exit_grace_frames must be non-negative, got {self.exit_grace_frames}"
            )
        if self.speed_window_seconds <= 0:
            raise ConfigurationError(
                f"speed_window_seconds must be positive, got {self.speed_window_seconds}"
            )
        if self.motion_reference_speed <= 0:
            raise ConfigurationError(
                f"motion_reference_speed must be positive, got {self.motion_reference_speed}"
            )
        if self.motion_smoothing_frames < 1:
            raise ConfigurationError(
                f"motion_smoothing_frames must be >= 1, got {self.motion_smoothing_frames}"
            )
        if self.metrics_sample_interval_seconds <= 0:
            raise ConfigurationError(
                "metrics_sample_interval_seconds must be positive, "
                f"got {self.metrics_sample_interval_seconds}"
            )
        if self.stale_track_ttl_frames < 1:
            raise ConfigurationError(
                f"stale_track_ttl_frames must be >= 1, got {self.stale_track_ttl_frames}"
            )
        if self.max_tracked_movements < 1:
            raise ConfigurationError(
                f"max_tracked_movements must be >= 1, got {self.max_tracked_movements}"
            )
        if self.max_completed_visits_per_zone < 1:
            raise ConfigurationError(
                "max_completed_visits_per_zone must be >= 1, "
                f"got {self.max_completed_visits_per_zone}"
            )

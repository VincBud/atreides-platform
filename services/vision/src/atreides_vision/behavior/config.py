"""Tunable parameters for behaviour signal extraction.

Every number that shapes a signal lives here, with its unit and the reasoning
behind it. No algorithm in this package contains a bare constant.

**Units follow the active coordinate space.** Thresholds expressed in
"distance units" are in whatever :class:`~atreides_vision.behavior.coordinates.MotionSpace`
the projector emits — normalized frame units in V0.3.0, calibrated metres if a
ground-plane projector is added later. A configuration tuned against normalized
image motion is therefore **not** valid for a metric projector, and vice versa.

**These are DEVELOPMENT defaults.** They were chosen to be reasonable and to
make the signals testable, not by validation against real venue footage. Unlike
the V0.2.1 detector and tracker defaults, no measurement backs them. They are
not safety thresholds and must not be treated as any.
"""

from __future__ import annotations

from dataclasses import dataclass

from atreides_vision.errors import ConfigurationError

__all__ = ["BehaviorAnalysisConfig"]


@dataclass(frozen=True, slots=True)
class BehaviorAnalysisConfig:
    """Windows, thresholds and memory bounds for behaviour analysis."""

    #: Seconds of motion averaged into one speed estimate. Around a second
    #: smooths bounding-box jitter — which is large relative to real movement
    #: at these scales — while still reacting to a person starting to walk.
    #: Matches the zone layer's speed window so the two report comparable
    #: magnitudes on the same footage.
    speed_window_seconds: float = 1.0

    #: Seconds of history differenced to estimate acceleration. The window is
    #: split in half and the two halves' mean speeds are differenced, so this
    #: needs enough samples for both halves to be meaningful; at 25 FPS one
    #: second gives roughly twelve per half.
    acceleration_window_seconds: float = 1.0

    #: Seconds of history differenced to estimate turn rate, split the same way
    #: as acceleration.
    heading_window_seconds: float = 1.0

    #: Minimum net displacement, in the space's distance units, that a half
    #: window must show before its direction of travel is considered defined.
    #: Below this, direction is dominated by box jitter and a "turn" would be
    #: noise amplified into a signal, so no heading signal is emitted at all.
    #: 0.01 normalized frame units over half a window corresponds to the
    #: :attr:`stationary_speed_threshold`, so the two agree about what counts
    #: as "not really moving".
    min_heading_displacement: float = 0.01

    #: Speed below which a subject counts as stationary, in distance units per
    #: second. 0.02 normalized frame units per second is roughly a seventh of
    #: the zone layer's provisional "brisk movement" reference (0.15), which is
    #: about where a standing person's box jitter sits. Highly camera-dependent
    #: and the first thing to re-tune per camera.
    stationary_speed_threshold: float = 0.02

    #: Longest gap between consecutive observations of one subject that motion
    #: may still be measured across. A tracker drops a box for a frame or two
    #: under partial occlusion and the person really did keep moving, so short
    #: gaps are bridged. Beyond this the subject was unobserved for long enough
    #: that any inferred motion would be invention: the history is cleared and
    #: a new motion segment begins. Also resets stationary accumulation — we
    #: cannot claim someone stood still through an interval we could not see.
    max_observation_gap_seconds: float = 0.5

    #: How much history is retained per subject. Must cover the longest signal
    #: window; samples older than this are dropped as they age out.
    history_window_seconds: float = 2.0

    #: Hard ceiling on samples kept per subject, guarding against pathological
    #: timestamps. Time-based trimming is the real bound.
    max_samples_per_subject: int = 512

    #: Frames after which an unseen subject's history is discarded entirely.
    #: Bounds memory on long runs where track ids churn.
    stale_subject_ttl_frames: int = 150

    #: Hard ceiling on simultaneously retained subjects, as a second bound.
    #: The least-recently-seen are evicted first.
    max_tracked_subjects: int = 1024

    def __post_init__(self) -> None:
        for name, value in (
            ("speed_window_seconds", self.speed_window_seconds),
            ("acceleration_window_seconds", self.acceleration_window_seconds),
            ("heading_window_seconds", self.heading_window_seconds),
            ("max_observation_gap_seconds", self.max_observation_gap_seconds),
            ("history_window_seconds", self.history_window_seconds),
            ("stationary_speed_threshold", self.stationary_speed_threshold),
            ("min_heading_displacement", self.min_heading_displacement),
        ):
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive, got {value}")

        for name, count in (
            ("max_samples_per_subject", self.max_samples_per_subject),
            ("stale_subject_ttl_frames", self.stale_subject_ttl_frames),
            ("max_tracked_subjects", self.max_tracked_subjects),
        ):
            if count < 1:
                raise ConfigurationError(f"{name} must be >= 1, got {count}")

        # A signal cannot be computed over a window longer than the history
        # kept for it. Catching this at construction turns a silently truncated
        # measurement into an immediate, explicit failure.
        if self.history_window_seconds < self.longest_signal_window_seconds:
            raise ConfigurationError(
                f"history_window_seconds ({self.history_window_seconds}) must be at least the "
                f"longest signal window ({self.longest_signal_window_seconds}); otherwise that "
                "signal would silently be computed over less history than configured"
            )

    @property
    def longest_signal_window_seconds(self) -> float:
        """The widest window any signal asks for."""
        return max(
            self.speed_window_seconds,
            self.acceleration_window_seconds,
            self.heading_window_seconds,
        )

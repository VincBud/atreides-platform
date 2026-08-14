"""Tunable parameters for behaviour signal extraction.

Every number that shapes a signal lives here, with its unit and the reasoning
behind it. No algorithm in this package contains a bare constant.

**Units follow the active coordinate space.** Thresholds expressed in
"distance units" are in whatever :class:`~atreides_vision.behavior.coordinates.MotionSpace`
the projector emits — normalized frame units by default, calibrated metres
behind a ground-plane projector. A configuration tuned against normalized image
motion is **not** valid for a metric projector, and vice versa: 0.02 frame
widths per second and 0.02 metres per second are wildly different speeds.

That mistake is silent and severe — normalized thresholds under a metric
projector make nobody ever count as stationary — so two things guard it. The
class defaults are normalized, :meth:`BehaviorAnalysisConfig.for_ground_plane_metres`
supplies a metric set, and :attr:`BehaviorAnalysisConfig.intended_space` lets a
config declare which space it was written for so the analyzer can refuse a
mismatched projector.

**These are DEVELOPMENT defaults.** They were chosen to be reasonable and to
make the signals testable, not by validation against real venue footage. Unlike
the V0.2.1 detector and tracker defaults, no measurement backs them. They are
not safety thresholds and must not be treated as any.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from atreides_vision.behavior.coordinates import MotionSpace
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

    #: The coordinate space this configuration's distance thresholds were
    #: written for, or ``None`` to make no claim. When set, the analyzer
    #: refuses a projector emitting a different space rather than silently
    #: comparing metres against frame fractions. ``None`` by default so
    #: existing configurations behave exactly as before.
    intended_space: MotionSpace | None = None

    @classmethod
    def for_ground_plane_metres(cls) -> Self:
        """Development defaults for calibrated metric coordinates.

        The class defaults are normalized-image values and are meaningless
        under a ground-plane projector: ``0.02`` frame widths per second is a
        plausible standing-still threshold, ``0.02`` metres per second is 2 cm/s
        and nobody would ever exceed it.

        The metric values keep the same relationship the normalized pair has —
        the heading floor is the distance a subject at the stationary threshold
        covers in half a window — so the two sets behave alike, just in their
        own units. For scale, a walk is roughly 1.4 m/s.

        Like every other default here these are **development** values with no
        measurement behind them, and the stationary threshold in particular
        depends on how much a tracked box jitters on a given camera.
        """
        return cls(
            #: 0.20 m/s: slow enough that a standing person's box jitter stays
            #: under it, fast enough to be well below a walk.
            stationary_speed_threshold=0.20,
            #: 0.10 m of net travel before a direction is considered defined —
            #: what 0.20 m/s covers in half a one-second window.
            min_heading_displacement=0.10,
            intended_space=MotionSpace.GROUND_PLANE_METRES,
        )

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

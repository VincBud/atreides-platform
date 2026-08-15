"""Tunable parameters for pairwise interaction measurement.

Every number that shapes a pairwise signal lives here with its unit and
rationale. No algorithm in this package contains a bare constant.

**Distance thresholds follow the active coordinate space.** A proximity
threshold of ``1.5`` means 1.5 metres under a calibrated projector and 1.5
*frame widths* under a normalized one — the second being wider than the entire
image. Mixing them is silent and total, which is why there are two named
factories and an ``intended_space`` the analyzer enforces, exactly as the
behaviour config does.

**These are DEVELOPMENT defaults.** No measurement backs them: no personal-space
distance has been validated, no interaction threshold has been checked against
real venue footage, and none of them is a safety threshold. They were chosen to
be defensible and to make the signals testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from atreides_vision.behavior.coordinates import MotionSpace
from atreides_vision.errors import ConfigurationError

__all__ = ["InteractionAnalysisConfig"]


@dataclass(frozen=True, slots=True)
class InteractionAnalysisConfig:
    """Windows, thresholds and memory bounds for pairwise analysis.

    The class defaults are **normalized image space**, matching
    :class:`~atreides_vision.behavior.config.BehaviorAnalysisConfig`. Use
    :meth:`for_ground_plane_metres` behind a calibrated projector.
    """

    #: Seconds of pair history differenced to estimate closing speed. Around a
    #: second averages out the bounding-box jitter that dominates at these
    #: scales while still reacting to someone actually walking over.
    closing_speed_window_seconds: float = 1.0

    #: Minimum net displacement, in the space's distance units, before a
    #: subject is considered to have a direction of travel at all. Below it the
    #: heading is box noise, and a relative heading built from two noise vectors
    #: would be pure invention. Matches the behaviour layer's floor.
    min_heading_displacement: float = 0.01

    #: Separation at or below which the pair counts as in proximity, in the
    #: space's distance units. 0.15 frame-widths is a loose normalized-space
    #: placeholder with no physical meaning whatsoever.
    proximity_threshold: float = 0.15

    #: Longest gap between observations of a pair that history may be carried
    #: across. A tracker drops a box for a frame or two and both people really
    #: were still there; beyond this the pair was unobserved long enough that
    #: continuing a proximity duration would be a claim about time nobody
    #: watched.
    max_observation_gap_seconds: float = 0.5

    #: How much pair history is retained. Must cover the closing-speed window.
    history_window_seconds: float = 2.0

    #: Pairs further apart than this are not analysed at all, in the space's
    #: distance units. Bounds the quadratic cost in dense scenes: most pairs in
    #: a crowded room are metres apart and measuring them every frame is waste.
    #: ``None`` disables gating and analyses every compatible pair.
    max_pair_distance: float | None = None

    #: Seconds after which an unseen pair's history is discarded. Time-based so
    #: it does not change meaning with frame rate.
    stale_pair_ttl_seconds: float = 5.0

    #: Hard ceiling on retained pairs, as a second bound against quadratic
    #: growth in a very dense scene. Least-recently-seen pairs are evicted.
    max_tracked_pairs: int = 4096

    #: Hard ceiling on samples kept per pair, guarding against pathological
    #: timestamps. Time-based trimming is the real bound.
    max_samples_per_pair: int = 256

    #: The coordinate space these thresholds were written for, or ``None`` to
    #: make no claim. When set, the analyzer refuses a mismatched context.
    intended_space: MotionSpace | None = None

    @classmethod
    def for_normalized_image(cls) -> Self:
        """Explicitly-declared normalized defaults, identical to the class defaults."""
        return cls(intended_space=MotionSpace.NORMALIZED_IMAGE)

    @classmethod
    def for_ground_plane_metres(cls) -> Self:
        """Development defaults for calibrated metric coordinates.

        The normalized values are meaningless in metres: a 0.15 m proximity
        threshold is two people touching, and a 0.01 m heading floor is a
        centimetre of movement.

        For scale, an ordinary walking pace is roughly 1.4 m/s. The proximity
        threshold is set at 1.5 m — conversational distance — purely so the
        signal is reachable and testable. **It is not a validated personal-space
        measurement and carries no behavioural meaning.**
        """
        return cls(
            #: 1.5 m. Development placeholder, not a validated social distance.
            proximity_threshold=1.5,
            #: 0.10 m of net travel before a direction is considered defined,
            #: matching the behaviour layer's metric heading floor.
            min_heading_displacement=0.10,
            #: 8 m. Beyond conversational range by a wide margin, so gating
            #: cannot discard a pair any current signal would report on.
            max_pair_distance=8.0,
            intended_space=MotionSpace.GROUND_PLANE_METRES,
        )

    def __post_init__(self) -> None:
        for name, value in (
            ("closing_speed_window_seconds", self.closing_speed_window_seconds),
            ("min_heading_displacement", self.min_heading_displacement),
            ("proximity_threshold", self.proximity_threshold),
            ("max_observation_gap_seconds", self.max_observation_gap_seconds),
            ("history_window_seconds", self.history_window_seconds),
            ("stale_pair_ttl_seconds", self.stale_pair_ttl_seconds),
        ):
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive, got {value}")

        for name, count in (
            ("max_tracked_pairs", self.max_tracked_pairs),
            ("max_samples_per_pair", self.max_samples_per_pair),
        ):
            if count < 1:
                raise ConfigurationError(f"{name} must be >= 1, got {count}")

        if self.max_pair_distance is not None and self.max_pair_distance <= 0:
            raise ConfigurationError(
                f"max_pair_distance must be positive when set, got {self.max_pair_distance}. "
                "Use None to analyse every compatible pair."
            )
        if self.history_window_seconds < self.closing_speed_window_seconds:
            raise ConfigurationError(
                f"history_window_seconds ({self.history_window_seconds}) must be at least "
                f"closing_speed_window_seconds ({self.closing_speed_window_seconds}); otherwise "
                "closing speed would silently be computed over less history than configured"
            )
        if self.max_pair_distance is not None and self.max_pair_distance < self.proximity_threshold:
            raise ConfigurationError(
                f"max_pair_distance ({self.max_pair_distance}) is below proximity_threshold "
                f"({self.proximity_threshold}): pairs would be gated out before they could ever "
                "be reported as in proximity."
            )

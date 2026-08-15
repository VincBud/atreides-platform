"""Tunable parameters for spatial-cluster measurement.

Every number that shapes a cluster or a density lives here with its unit and the
reasoning behind it. No algorithm in this package contains a bare constant.

**Distance thresholds follow the active coordinate space.** A cluster radius of
``1.5`` means 1.5 metres under a calibrated projector and 1.5 *frame widths*
under a normalized one — the second being wider than the whole image. Mixing
them is silent and total, which is why there are two named factories and an
``intended_space`` the analyzer enforces, exactly as the behaviour and
interaction configs do.

**These are DEVELOPMENT defaults.** They are geometric algorithm parameters
chosen so the measurements are reachable and testable. None is a validated
crowding limit, safe density, social-distance threshold or safety figure of any
kind, and no measurement backs any of them.

**No threshold encodes a venue size.** A cluster radius is a local distance
between two people; it does not scale with how big the room is. A camera
covering a doorway and one covering a festival field use the same configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from atreides_vision.behavior.coordinates import MotionSpace
from atreides_vision.errors import ConfigurationError

__all__ = ["CrowdAnalysisConfig"]


@dataclass(frozen=True, slots=True)
class CrowdAnalysisConfig:
    """Connectivity, density, persistence and memory bounds.

    Class defaults are **normalized image space**, matching the behaviour and
    interaction configs. Use :meth:`for_ground_plane_metres` behind a calibrated
    projector.
    """

    #: Two subjects are connected when their separation is at or below this, in
    #: the space's distance units. This is a **connectivity** distance, not a
    #: maximum cluster diameter: chains of connected subjects can span much
    #: further. See :mod:`atreides_vision.crowd.clustering`.
    cluster_radius: float = 0.12

    #: Fewest connected subjects that count as a spatial cluster. Three, because
    #: a two-person relationship is already measured pairwise by V0.3.4 and
    #: reporting it again here would duplicate that contract with a worse model.
    minimum_cluster_size: int = 3

    #: Radius of the disk sampled for local density, in the space's distance
    #: units. Deliberately **separate from** :attr:`cluster_radius`: connectivity
    #: asks "is this person attached to that one", density asks "how concentrated
    #: is this neighbourhood". They answer different questions and a good value
    #: for one is rarely a good value for the other, so they are never tied.
    density_radius: float = 0.20

    #: Fewest cluster members that must have usable motion before heading
    #: coherence is reported. One moving member has nothing to be coherent
    #: *with*, so the signal is omitted rather than reported as perfectly
    #: aligned.
    minimum_moving_members: int = 2

    #: Net displacement below which a member is treated as having no direction
    #: of travel, in the space's distance units. Matches the behaviour and
    #: interaction layers' floor so all three agree about who is moving.
    min_motion_displacement: float = 0.01

    #: Longest gap between observations of a cluster that its persistence may be
    #: carried across. A tracker drops a box for a frame or two and everyone was
    #: still standing there; beyond this the cluster was unobserved long enough
    #: that continuing a duration would be a claim about time nobody watched.
    max_observation_gap_seconds: float = 0.5

    #: Seconds after which an unseen cluster composition is forgotten.
    #: Time-based so it does not change meaning with frame rate.
    stale_cluster_ttl_seconds: float = 5.0

    #: Hard ceiling on retained cluster compositions, as a second bound against
    #: churn in a dense scene. Least-recently-seen are evicted.
    max_tracked_clusters: int = 2048

    #: The coordinate space these thresholds were written for, or ``None`` to
    #: make no claim. When set, the analyzer refuses a mismatched context.
    intended_space: MotionSpace | None = None

    @classmethod
    def for_normalized_image(cls) -> Self:
        """Explicitly-declared normalized defaults, identical to the class defaults.

        Normalized cluster sizes and densities are perspective- and
        camera-dependent. They are useful for architecture and tests; they are
        not physical crowd measurements.
        """
        return cls(intended_space=MotionSpace.NORMALIZED_IMAGE)

    @classmethod
    def for_ground_plane_metres(cls) -> Self:
        """Development defaults for calibrated metric coordinates.

        The normalized values are meaningless in metres: a 0.12 m cluster radius
        would connect only people already touching.

        The metric values are geometric parameters chosen so clusters form at
        conversational spacing and densities are computable. **They are not a
        crowding limit, a safe density, or a social-distance figure**, and
        nothing about them has been validated against a real venue.
        """
        return cls(
            #: 1.5 m — roughly conversational spacing. A connectivity distance,
            #: chosen so clusters are reachable in tests, nothing more.
            cluster_radius=1.5,
            #: 2.0 m disk for the density neighbourhood: wider than the
            #: connectivity radius so density describes a neighbourhood rather
            #: than restating who is connected to whom.
            density_radius=2.0,
            #: 0.10 m of net travel before a member counts as moving, matching
            #: the behaviour layer's metric heading floor.
            min_motion_displacement=0.10,
            intended_space=MotionSpace.GROUND_PLANE_METRES,
        )

    def __post_init__(self) -> None:
        for name, value in (
            ("cluster_radius", self.cluster_radius),
            ("density_radius", self.density_radius),
            ("min_motion_displacement", self.min_motion_displacement),
            ("max_observation_gap_seconds", self.max_observation_gap_seconds),
            ("stale_cluster_ttl_seconds", self.stale_cluster_ttl_seconds),
        ):
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive, got {value}")

        if self.minimum_cluster_size < 2:
            raise ConfigurationError(
                f"minimum_cluster_size must be >= 2, got {self.minimum_cluster_size}: "
                "a cluster needs at least two connected subjects to exist at all."
            )
        if self.minimum_moving_members < 2:
            raise ConfigurationError(
                f"minimum_moving_members must be >= 2, got {self.minimum_moving_members}: "
                "heading coherence compares directions, and one direction has nothing "
                "to be coherent with."
            )
        if self.max_tracked_clusters < 1:
            raise ConfigurationError(
                f"max_tracked_clusters must be >= 1, got {self.max_tracked_clusters}"
            )

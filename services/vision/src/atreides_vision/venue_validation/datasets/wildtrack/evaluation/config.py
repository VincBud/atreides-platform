"""Frozen configuration for the offline WILDTRACK evaluator (V0.4.9b-D1).

Everything here is a property of the *evaluation*, never of the runtime. The
evaluator observes the runtime through the read-only
:class:`~atreides_vision.venue_validation.observer.VenueValidationInstantSnapshot`
seam and compares it against WILDTRACK ground truth; nothing in this module is
read by, or reachable from, any association / fusion / occupancy algorithm.

``max_ground_plane_distance_m`` is the one frozen baseline constant. It is the
radius, in the shared WILDTRACK ``positionID`` ground-plane metre frame, within
which a single runtime observation on one camera is accepted as *the same
physical person* as one WILDTRACK annotated person at the same instant. It is:

* **camera-local** — applied per camera, between that camera's runtime
  observations and the persons WILDTRACK says that camera saw;
* **instant-local** — applied within one venue instant, never across time;
* **ground-plane metres only** — an observation not in
  :attr:`~atreides_vision.behavior.coordinates.MotionSpace.GROUND_PLANE_METRES`
  is never distance-matched and is never silently converted;
* **inclusive** — a separation of exactly ``0.75`` m matches
  (``separation <= max_ground_plane_distance_m``);
* **not** :class:`~atreides_vision.association.AssociationConfig` — it does not
  gate, tune or influence the runtime in any way. It only decides which runtime
  observation stands in for which ground-truth person when scoring.

It was frozen before any real Stage A/B/C run and is not retuned in response to
a validation figure; changing it is a new milestone.
"""

from __future__ import annotations

from dataclasses import dataclass

from atreides_vision.errors import ValidationError

__all__ = [
    "MAX_GROUND_PLANE_DISTANCE_M",
    "VENUE_SECONDS_TOLERANCE",
    "EvaluationConfig",
]

#: Frozen baseline. See the module docstring for the full semantics.
MAX_GROUND_PLANE_DISTANCE_M = 0.75

#: How far a snapshot's ``venue_seconds`` may deviate from its expected value
#: (``instant_index / WILDTRACK_DECLARED_FPS``) before the evaluator treats the
#: stream as misaligned and refuses to score it. WILDTRACK's declared cadence is
#: exact in binary floating point, so this only absorbs representation noise.
VENUE_SECONDS_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Tunables for one WILDTRACK evaluation run.

    Args:
        selected_camera_ids: The cameras this run scored, in canonical order.
            Ground-truth occupancy and cross-camera correspondence are
            restricted to exactly this set, so a Stage-A subset run is scored
            against the subset it could actually observe.
        max_ground_plane_distance_m: The observation/ground-truth match radius.
            Defaults to the frozen baseline and is not expected to be
            overridden outside a dedicated experiment.
        venue_seconds_tolerance: Allowed representation noise on the venue
            clock; see :data:`VENUE_SECONDS_TOLERANCE`.
    """

    selected_camera_ids: tuple[str, ...]
    max_ground_plane_distance_m: float = MAX_GROUND_PLANE_DISTANCE_M
    venue_seconds_tolerance: float = VENUE_SECONDS_TOLERANCE

    def __post_init__(self) -> None:
        if not self.selected_camera_ids:
            raise ValidationError(
                "EvaluationConfig.selected_camera_ids is empty; an evaluation must score at "
                "least one camera."
            )
        if len(set(self.selected_camera_ids)) != len(self.selected_camera_ids):
            raise ValidationError(
                f"EvaluationConfig.selected_camera_ids repeats a camera: {self.selected_camera_ids}"
            )
        if not self.max_ground_plane_distance_m > 0:
            raise ValidationError(
                "EvaluationConfig.max_ground_plane_distance_m must be a positive metre distance, "
                f"got {self.max_ground_plane_distance_m!r}."
            )
        if not self.venue_seconds_tolerance >= 0:
            raise ValidationError(
                "EvaluationConfig.venue_seconds_tolerance must be non-negative, got "
                f"{self.venue_seconds_tolerance!r}."
            )

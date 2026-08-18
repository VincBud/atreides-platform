"""The hard comparability gates: whether two observations may be compared at all.

A gate answers a question about the *measurement*, never about the people. It
asks whether these two numbers are expressed in the same physical frame, whether
that frame is currently trustworthy, and whether the two clocks are related well
enough for "at the same moment" to mean anything. When the answer is no, the
result is :attr:`~atreides_vision.association.models.AssociationOutcome.NOT_COMPARABLE`
— which is not a weaker "no match". Nothing was measured, so nothing was found,
and a consumer that treats the two the same will report a blind spot as an
absence of people.

Order and completeness
----------------------
Gates are evaluated in one fixed order, :data:`HARD_GATE_ORDER`, and **every
applicable gate is evaluated**, not merely the first that fails. An operator
looking at two cameras that never associate needs the whole list: fixing the
venue frame is wasted work if the surfaces also differ.

Some gates cannot be evaluated once an earlier one has failed — there is no
venue frame to compare when a camera is not calibrated at all, and no radius to
compare a region against when the clocks are unrelated. Those are skipped rather
than guessed, so a rejection list is always a set of facts.

Uncertainty weakens a gate, never strengthens it
------------------------------------------------
Both time-dependent gates take the stated timing uncertainty in the candidate's
favour: a pair is refused as non-simultaneous only if even its best case is
outside the window, and the spatial radius is widened by the worst case. Two
badly aligned cameras therefore admit almost everything and say so in the
evidence, instead of quietly producing rejections derived from a precision
nobody established.
"""

from __future__ import annotations

from typing import Final

from atreides_vision.association.config import AssociationGatingConfig
from atreides_vision.association.geometry import (
    admissible_radius_m,
    earliest_possible_gap_seconds,
    latest_possible_gap_seconds,
    nominal_gap_seconds,
    timing_slack_seconds,
)
from atreides_vision.association.models import (
    CandidatePair,
    GateReason,
    GateRejection,
    TrackObservation,
)
from atreides_vision.behavior.coordinates import MotionSpace

__all__ = ["HARD_GATE_ORDER", "comparability_rejections"]

#: The order gates are evaluated and recorded in. Fixed and total, so a pair's
#: rejection list is byte-identical between runs and never depends on which
#: camera happened to be enumerated first.
HARD_GATE_ORDER: Final[tuple[GateReason, ...]] = (
    GateReason.SAME_CAMERA,
    GateReason.NOT_GROUND_PLANE_METRES,
    GateReason.DIFFERENT_VENUE_FRAME,
    GateReason.DIFFERENT_SURFACE,
    GateReason.CAMERA_GEOMETRY_UNTRUSTED,
    GateReason.VENUE_TIME_NOT_ESTABLISHED,
    GateReason.NOT_SIMULTANEOUS,
    GateReason.NO_SHARED_CALIBRATED_REGION,
)


def comparability_rejections(
    pair: CandidatePair, config: AssociationGatingConfig
) -> tuple[GateRejection, ...]:
    """Every hard gate that refuses this pair, in :data:`HARD_GATE_ORDER`.

    An empty tuple means the pair may be measured. It does **not** mean the two
    observations are the same person, and nothing in this module ever will.
    """
    first, second = pair.first, pair.second
    rejections: list[GateRejection] = []

    if first.camera_id == second.camera_id:
        rejections.append(
            GateRejection(
                GateReason.SAME_CAMERA,
                f"{first.subject} and {second.subject} are both on camera "
                f"'{first.camera_id}'. Association is cross-camera; joining two tracklets on one "
                "camera is fragment stitching, which is a different capability.",
            )
        )

    rejections.extend(_coordinate_frame_rejections(first, second))
    rejections.extend(_trust_rejections(first, second))
    rejections.extend(_time_and_region_rejections(first, second, config))
    return tuple(rejections)


def _coordinate_frame_rejections(
    first: TrackObservation, second: TrackObservation
) -> list[GateRejection]:
    """Whether the two positions are numbers in one physical coordinate frame."""
    uncalibrated = [
        observation
        for observation in (first, second)
        if observation.space is not MotionSpace.GROUND_PLANE_METRES
    ]
    if uncalibrated:
        described = ", ".join(
            f"{observation.subject} in {observation.space.value}" for observation in uncalibrated
        )
        return [
            GateRejection(
                GateReason.NOT_GROUND_PLANE_METRES,
                f"Cross-camera geometry requires calibrated ground-plane metres, but {described}. "
                "Image-plane coordinates are camera-local, and no physical distance may be "
                "derived from an uncalibrated scene.",
            )
        ]

    rejections: list[GateRejection] = []
    if first.spatial.venue_id != second.spatial.venue_id:
        rejections.append(
            GateRejection(
                GateReason.DIFFERENT_VENUE_FRAME,
                f"Camera '{first.camera_id}' measures in venue frame "
                f"'{first.spatial.venue_id}' and camera '{second.camera_id}' in "
                f"'{second.spatial.venue_id}'. Two unrelated coordinate systems that happen to "
                "share a unit.",
            )
        )
    if first.spatial.surface_id != second.spatial.surface_id:
        rejections.append(
            GateRejection(
                GateReason.DIFFERENT_SURFACE,
                f"Camera '{first.camera_id}' is calibrated to surface "
                f"'{first.spatial.surface_id}' and camera '{second.camera_id}' to "
                f"'{second.spatial.surface_id}'. One homography describes one plane: two points "
                "a storey apart can have near-identical coordinates.",
            )
        )
    return rejections


def _trust_rejections(first: TrackObservation, second: TrackObservation) -> list[GateRejection]:
    """Whether both cameras' metres may still be relied on at all."""
    untrusted = [
        observation
        for observation in (first, second)
        if not observation.spatial.trust.permits_comparison
    ]
    if not untrusted:
        return []
    cameras = ", ".join(f"'{observation.camera_id}'" for observation in untrusted)
    return [
        GateRejection(
            GateReason.CAMERA_GEOMETRY_UNTRUSTED,
            f"Camera {cameras} no longer matches the viewpoint its calibration was authored "
            "against, so its metres are not trusted. Degraded integrity lowers what geometry can "
            "conclude; it does not keep comparing quietly.",
        )
    ]


def _time_and_region_rejections(
    first: TrackObservation, second: TrackObservation, config: AssociationGatingConfig
) -> list[GateRejection]:
    """The three gates that need a common clock before they can be evaluated.

    Whether the clocks relate at all, whether the two instants can coincide, and
    whether the two calibrated regions are within reach of each other in the
    time that may separate them.
    """
    slack = timing_slack_seconds(first.timing, second.timing)
    if slack is None:
        unestablished = ", ".join(
            f"{observation.subject} ({observation.stream})"
            for observation in (first, second)
            if not observation.timing.is_established
        )
        return [
            GateRejection(
                GateReason.VENUE_TIME_NOT_ESTABLISHED,
                f"Nothing relates the venue timeline to {unestablished}. Without an established "
                "alignment the two observations share no clock, so 'at the same moment' has no "
                "meaning and a shared venue tick is only a scheduling unit.",
            )
        ]

    rejections: list[GateRejection] = []
    nominal = nominal_gap_seconds(first, second)
    earliest = earliest_possible_gap_seconds(nominal, slack)
    if earliest > config.max_pairing_gap_seconds:
        rejections.append(
            GateRejection(
                GateReason.NOT_SIMULTANEOUS,
                f"The observations are {nominal:.3f} s apart in venue time; even allowing the "
                f"full +/-{slack:.3f} s of timing uncertainty in their favour, at least "
                f"{earliest:.3f} s separates them, beyond the "
                f"{config.max_pairing_gap_seconds:.3f} s pairing window.",
            )
        )

    radius = admissible_radius_m(latest_possible_gap_seconds(nominal, slack), config)
    first_region = first.spatial.calibrated_region
    second_region = second.spatial.calibrated_region
    if first_region is not None and second_region is not None:
        region_gap = first_region.gap_to(second_region)
        if region_gap > radius:
            rejections.append(
                GateRejection(
                    GateReason.NO_SHARED_CALIBRATED_REGION,
                    f"The calibrated regions of '{first.camera_id}' and '{second.camera_id}' are "
                    f"{region_gap:.3f} m apart, further than the {radius:.3f} m a person could "
                    "bridge in the time that may separate these observations. Cameras that do "
                    "not overlap are an ordinary deployment, not an error.",
                )
            )
    return rejections

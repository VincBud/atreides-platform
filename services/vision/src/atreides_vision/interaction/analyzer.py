"""The interaction analyzer — the single component a pipeline talks to.

It owns the whole pairwise path for one camera session::

    BehaviorFrame -> compatible pairs -> separation history -> pairwise signals

and returns an :class:`~atreides_vision.interaction.models.InteractionFrame` per
frame, exactly as ``BehaviorAnalyzer`` and ``ZoneAnalyzer`` do for their layers.

Why it consumes behaviour snapshots
-----------------------------------
Its input is the **already-projected** output of behaviour analysis, not tracks
and not detections. That means there is one projection in the process, one
motion history, and no path by which pairwise code could acquire an opinion
about RF-DETR, OC-SORT or pixels. A second, independently-derived set of
positions would drift out of agreement with the individual signals and be
impossible to reconcile.

Pairwise state lives here, never in ``BehaviorAnalyzer``: individual behaviour
and pairwise interaction are separate analytical layers with different
cardinality, different history and different lifetimes.

Same camera only
----------------
One analyzer serves one camera, and every pair is two subjects from that
camera's own frame. Cross-camera pairing is rejected rather than merely absent —
see :class:`~atreides_vision.interaction.models.SubjectPairRef`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from itertools import combinations

from atreides_vision.behavior.coordinates import MotionSpace
from atreides_vision.behavior.models import BehaviorFrame, BehaviorSubjectSnapshot
from atreides_vision.errors import ConfigurationError, InteractionError
from atreides_vision.interaction.config import InteractionAnalysisConfig
from atreides_vision.interaction.geometry import (
    closing_speed,
    pair_distance,
    proximity_since,
    relative_motion_heading,
)
from atreides_vision.interaction.history import PairHistoryStore, PairSample
from atreides_vision.interaction.models import (
    InteractionFrame,
    InteractionSignal,
    InteractionSignalType,
    InteractionSnapshot,
    PairGeometry,
    SpatialContext,
    SubjectPairRef,
)

logger = logging.getLogger(__name__)

__all__ = ["InteractionAnalyzer"]


class InteractionAnalyzer:
    """Per-frame pairwise measurement for one camera session.

    Args:
        context: The coordinate frame these measurements are made in. Fixes the
            camera, the space, and — for metric analysis — the venue and
            surface.
        config: Windows, thresholds and memory bounds. Development defaults.

    Raises:
        ConfigurationError: if ``config`` declares an ``intended_space`` the
            context does not provide. Normalized thresholds interpreted as
            metres would put the proximity boundary 15 cm away; metric ones
            interpreted as frame fractions would put it ten frame-widths away.
            Both produce a full stream of confident, wrong signals.
    """

    def __init__(
        self,
        context: SpatialContext,
        config: InteractionAnalysisConfig | None = None,
    ) -> None:
        self._context = context
        self._config = config or InteractionAnalysisConfig()
        self._require_matching_space()

        self._history = PairHistoryStore(self._config)
        self._latest: InteractionFrame | None = None
        self._snapshots_emitted = 0
        self._candidate_pairs_seen = 0
        self._pairs_analysed = 0
        self._incompatible_pairs_skipped = 0

    def _require_matching_space(self) -> None:
        intended = self._config.intended_space
        actual = self._context.space
        if intended is not None and intended is not actual:
            raise ConfigurationError(
                f"Interaction configuration is written for {intended.value} coordinates but the "
                f"spatial context is {actual.value}. Distance thresholds would be interpreted in "
                "the wrong unit. Use InteractionAnalysisConfig.for_ground_plane_metres() with a "
                "calibrated context, or for_normalized_image() with a normalized one."
            )

    @property
    def context(self) -> SpatialContext:
        return self._context

    @property
    def config(self) -> InteractionAnalysisConfig:
        return self._config

    @property
    def camera_id(self) -> str:
        return self._context.camera_id

    @property
    def space(self) -> MotionSpace:
        return self._context.space

    @property
    def latest_frame(self) -> InteractionFrame | None:
        return self._latest

    @property
    def snapshots_emitted(self) -> int:
        """Pair measurements produced this session."""
        return self._snapshots_emitted

    @property
    def candidate_pairs_seen(self) -> int:
        """Pairs formed before distance gating, across the whole run."""
        return self._candidate_pairs_seen

    @property
    def pairs_analysed(self) -> int:
        """Pairs that survived gating and were measured."""
        return self._pairs_analysed

    @property
    def tracked_pairs(self) -> int:
        """Pairs currently holding history."""
        return len(self._history)

    def update(self, frame: BehaviorFrame) -> InteractionFrame:
        """Measure every compatible pair in one behaviour frame.

        Args:
            frame: The behaviour analyzer's output for this frame. Its
                snapshots carry projected positions and motion vectors, which is
                everything pairwise measurement needs.

        Returns:
            One snapshot per analysed pair, in deterministic pair order.

        Raises:
            InteractionError: if a snapshot belongs to a different camera or a
                different coordinate space than this analyzer's context. Both
                mean the caller wired something wrong, and measuring anyway
                would produce a plausible number for an invalid comparison.
        """
        usable = self._validated_snapshots(frame.snapshots)
        snapshots: list[InteractionSnapshot] = []
        candidates = 0

        # combinations() over a stably-ordered sequence yields each unordered
        # pair exactly once and never a self-pair, deterministically.
        for first, second in combinations(usable, 2):
            candidates += 1
            distance = pair_distance(first.position, second.position)

            if not self._within_candidate_range(distance):
                # Gated out: too far apart for any current signal to be about
                # them. History is deliberately left alone rather than deleted —
                # a pair drifting in and out of range keeps its proximity spell,
                # and time-based expiry collects it if they stay apart.
                continue

            pair = SubjectPairRef(first.subject, second.subject)
            snapshots.append(
                self._measure(
                    pair,
                    first,
                    second,
                    distance,
                    frame.frame_index,
                    frame.timestamp_seconds,
                )
            )

        self._history.forget_subjects_absent_from(
            frozenset(snapshot.subject for snapshot in usable)
        )
        self._history.prune(frame.timestamp_seconds)

        result = InteractionFrame.from_snapshots(
            frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds,
            snapshots=snapshots,
            candidate_pairs=candidates,
        )
        self._candidate_pairs_seen += candidates
        self._pairs_analysed += len(snapshots)
        self._snapshots_emitted += len(snapshots)
        self._latest = result
        return result

    def _validated_snapshots(
        self, snapshots: Sequence[BehaviorSubjectSnapshot]
    ) -> list[BehaviorSubjectSnapshot]:
        """Reject anything that cannot legitimately be compared in this context."""
        for snapshot in snapshots:
            if snapshot.subject.camera_id != self._context.camera_id:
                raise InteractionError(
                    f"{snapshot.subject} does not belong to camera "
                    f"'{self._context.camera_id}'. Interaction analysis is camera-local: "
                    "pairing subjects across cameras would risk pairing one real person "
                    "with themselves."
                )
            if snapshot.space is not self._context.space:
                raise InteractionError(
                    f"{snapshot.subject} is in {snapshot.space.value} but this analyzer "
                    f"measures in {self._context.space.value}."
                )
        return list(snapshots)

    def _within_candidate_range(self, distance: float) -> bool:
        limit = self._config.max_pair_distance
        return limit is None or distance <= limit

    def _measure(
        self,
        pair: SubjectPairRef,
        first: BehaviorSubjectSnapshot,
        second: BehaviorSubjectSnapshot,
        distance: float,
        frame_index: int,
        timestamp_seconds: float,
    ) -> InteractionSnapshot:
        """Everything measurable about one pair at this frame."""
        config = self._config
        history = self._history.observe(
            pair,
            PairSample(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                distance=distance,
            ),
        )

        signals: list[InteractionSignal] = []

        window = history.recent(config.closing_speed_window_seconds)
        rate = closing_speed(window)
        if rate is not None:
            signals.append(self._signal(InteractionSignalType.CLOSING_SPEED, rate))

        heading = relative_motion_heading(
            first.motion, second.motion, config.min_heading_displacement
        )
        if heading is not None:
            signals.append(self._signal(InteractionSignalType.RELATIVE_MOTION_HEADING, heading))

        # Proximity must outlive the retained window — two people can stand
        # together far longer — so it is carried on the pair's history as a
        # spell start and updated here from this frame's separation.
        window_start = window[0].timestamp_seconds if window else timestamp_seconds
        spell_start = proximity_since(
            history.proximity_since,
            distance=distance,
            threshold=config.proximity_threshold,
            window_start_seconds=window_start,
            continuous=history.is_continuous,
        )
        history.set_proximity_since(spell_start)
        if spell_start is not None:
            signals.append(
                self._signal(
                    InteractionSignalType.PROXIMITY_DURATION,
                    max(0.0, timestamp_seconds - spell_start),
                )
            )

        return InteractionSnapshot(
            pair=pair,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            geometry=PairGeometry(distance=distance, space=self._context.space),
            signals=tuple(signals),
            shared_zone_ids=_shared_zones(first, second),
        )

    def _signal(self, signal_type: InteractionSignalType, value: float) -> InteractionSignal:
        return InteractionSignal(signal_type=signal_type, value=value, space=self._context.space)

    def log_summary(self) -> None:
        """Log a final summary. Called once at the end of a run."""
        logger.info(
            "Interaction pairs:   %d measured of %d candidates in %s space",
            self._pairs_analysed,
            self._candidate_pairs_seen,
            self.space.value,
        )


def _shared_zones(
    first: BehaviorSubjectSnapshot, second: BehaviorSubjectSnapshot
) -> tuple[str, ...]:
    """Zone ids both subjects are inside, in the first subject's order.

    Context only. Two people sharing a zone label are in the same operator-drawn
    region, which says nothing about how close they are — a zone can be the
    whole dancefloor. Proximity is measured geometrically and never inferred
    from zone equality.
    """
    if not first.zone_ids or not second.zone_ids:
        return ()
    second_zones = set(second.zone_ids)
    return tuple(zone for zone in first.zone_ids if zone in second_zones)

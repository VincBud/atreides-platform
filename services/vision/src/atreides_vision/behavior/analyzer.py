"""The behaviour analyzer — the single component a pipeline talks to.

It owns the whole behaviour path for one camera session::

    tracks -> anchor -> projection -> history -> kinematics -> signals

and returns a :class:`~atreides_vision.behavior.models.BehaviorFrame` per
frame. Keeping it behind one interface is what stops ``VideoProcessor`` from
growing a third job, exactly as :class:`~atreides_vision.zones.analyzer.ZoneAnalyzer`
does for zones.

The analyzer depends only on :class:`~atreides_vision.models.Track` and on a
:class:`~atreides_vision.behavior.coordinates.MotionProjector`. It imports no
detector, no tracker, and not the zone layer either, so RF-DETR, OC-SORT and
ByteTrack all remain replaceable without touching behaviour analysis.

One analyzer, one camera
------------------------
``camera_id`` is fixed at construction and stamped onto every subject and
signal. Running two cameras means two analyzers with separate state — there is
no shared store to leak through, and identical track ids on different cameras
stay distinct because the subject key is the *(camera, track)* pair.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.behavior.coordinates import (
    MotionProjector,
    MotionSpace,
    PersonAnchor,
    bottom_center_anchor,
)
from atreides_vision.behavior.history import SubjectHistory, SubjectHistoryStore
from atreides_vision.behavior.kinematics import (
    acceleration,
    heading_change_rate,
    mean_speed,
    span_seconds,
    stationary_since,
)
from atreides_vision.behavior.models import (
    BehaviorFrame,
    BehaviorObservation,
    BehaviorSignal,
    SignalType,
    SubjectRef,
)
from atreides_vision.errors import ConfigurationError, ProjectionError
from atreides_vision.models import Track

logger = logging.getLogger(__name__)

__all__ = ["BehaviorAnalyzer"]


class BehaviorAnalyzer:
    """Per-frame behaviour signals for one camera session.

    Args:
        camera_id: Identifies the camera these signals came from. Stamped onto
            every subject, so downstream consumers can never mistake one
            camera's track ids for another's.
        projector: Converts image-space anchors into the analysis coordinate
            space. Pass a
            :class:`~atreides_vision.behavior.coordinates.NormalizedFrameProjector`
            built from the source resolution, or a calibrated
            ``GroundPlaneProjector`` for venue metres — the analyzer neither
            knows nor cares which, beyond checking the config agrees.
        config: Windows, thresholds and memory bounds. Development defaults.
        anchor: How a box becomes a point. Defaults to bottom-centre, the
            same floor convention the zone layer uses.

    Raises:
        ConfigurationError: if ``config`` declares an
            :attr:`~atreides_vision.behavior.config.BehaviorAnalysisConfig.intended_space`
            that ``projector`` does not emit.
    """

    def __init__(
        self,
        camera_id: str,
        projector: MotionProjector,
        config: BehaviorAnalysisConfig | None = None,
        anchor: PersonAnchor = bottom_center_anchor,
    ) -> None:
        if not camera_id or not camera_id.strip():
            raise ValueError("BehaviorAnalyzer requires a non-empty camera_id")
        self._camera_id = camera_id
        self._projector = projector
        self._config = config or BehaviorAnalysisConfig()
        self._anchor = anchor
        self._history = SubjectHistoryStore(self._config)
        self._latest: BehaviorFrame | None = None
        self._signals_emitted = 0
        self._unprojectable_observations = 0
        self._require_matching_space()

    def _require_matching_space(self) -> None:
        """Refuse a config whose distance thresholds are in the wrong space.

        Catching this at construction matters because the failure is otherwise
        silent and total: normalized thresholds under a metric projector mean
        nobody is ever stationary, and metric thresholds under a normalized one
        mean everybody always is. Both produce a full stream of confident,
        wrong signals rather than an error.
        """
        intended = self._config.intended_space
        actual = self._projector.space
        if intended is not None and intended is not actual:
            raise ConfigurationError(
                f"Behaviour configuration is written for {intended.value} coordinates but the "
                f"projector emits {actual.value}. Distance thresholds would be interpreted in "
                f"the wrong unit. Use BehaviorAnalysisConfig.for_ground_plane_metres() with a "
                f"calibrated projector, or the defaults with a normalized one."
            )

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def config(self) -> BehaviorAnalysisConfig:
        return self._config

    @property
    def space(self) -> MotionSpace:
        """The coordinate space every signal from this analyzer is measured in."""
        return self._projector.space

    @property
    def latest_frame(self) -> BehaviorFrame | None:
        """The most recent frame's signals, or ``None`` before the first frame."""
        return self._latest

    @property
    def signals_emitted(self) -> int:
        """Total signals produced this session."""
        return self._signals_emitted

    @property
    def tracked_subjects(self) -> int:
        """Subjects currently holding history."""
        return len(self._history)

    @property
    def unprojectable_observations(self) -> int:
        """Anchors skipped this session for having no ground-plane position.

        Always ``0`` for a normalized projector, which has no horizon. A
        persistently high count under a calibrated one means people are being
        detected above the calibrated floor area — often a sign the reference
        points cover only part of the view.
        """
        return self._unprojectable_observations

    def update(
        self,
        tracks: Sequence[Track],
        frame_index: int,
        timestamp_seconds: float,
        zone_ids_by_track: Mapping[int, Sequence[str]] | None = None,
    ) -> BehaviorFrame:
        """Advance behaviour analysis by one frame.

        Args:
            tracks: Confirmed tracks for this frame.
            frame_index: 1-based frame number, strictly increasing per subject.
            timestamp_seconds: Source-time position of this frame, derived from
                the real frame rate. Every signal is a rate with respect to
                this, so passing wall-clock or assumed-30-FPS time silently
                falsifies all of them.
            zone_ids_by_track: Optional zone context per track id, as opaque
                strings. Copied onto observations and signals, never
                interpreted — behaviour analysis stays independent of whether
                zones are configured at all.

        Returns:
            Every signal computable for this frame, grouped in track order.
        """
        signals: list[BehaviorSignal] = []

        for track in tracks:
            subject = SubjectRef(camera_id=self._camera_id, track_id=track.track_id)
            zone_ids = tuple(zone_ids_by_track.get(track.track_id, ())) if zone_ids_by_track else ()

            try:
                position = self._projector.project(self._anchor(track.bbox_xyxy))
            except ProjectionError:
                # A calibrated projector has a horizon: an anchor at or beyond
                # it has no ground position at all. That is a property of the
                # single point, not of the calibration, so one such detection
                # must not end a run of thousands of frames. The track is
                # skipped for this frame and no signal is invented for it —
                # the same rule the rest of the analyzer follows when it cannot
                # measure something. Caught narrowly: any other failure is a
                # real fault and still propagates.
                self._unprojectable_observations += 1
                logger.debug(
                    "Skipping %s at frame %d: anchor has no ground-plane position",
                    subject,
                    frame_index,
                )
                continue

            observation = BehaviorObservation(
                subject=subject,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                position=position,
                bbox=track.bbox_xyxy,
                confidence=track.confidence,
                zone_ids=zone_ids,
            )
            history = self._history.observe(observation)
            signals.extend(self._signals_for(history, observation))

        self._history.prune(frame_index)

        frame = BehaviorFrame(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            signals=tuple(signals),
        )
        self._signals_emitted += len(signals)
        self._latest = frame
        return frame

    def _signals_for(
        self,
        history: SubjectHistory,
        observation: BehaviorObservation,
    ) -> list[BehaviorSignal]:
        """Every signal computable for one subject at this frame.

        A signal that cannot be computed is omitted rather than emitted as
        zero. "Not enough history to say" and "measured as not moving" are
        different facts, and collapsing them would be the kind of quiet
        fabrication a downstream risk layer could never recover from.
        """
        config = self._config
        signals: list[BehaviorSignal] = []

        speed_samples = history.recent(config.speed_window_seconds)
        speed = mean_speed(speed_samples)
        speed_quality = self._quality(speed_samples, config.speed_window_seconds)

        if speed is not None:
            signals.append(self._signal(SignalType.SPEED, observation, speed, speed_quality))

        acceleration_samples = history.recent(config.acceleration_window_seconds)
        rate_of_change = acceleration(acceleration_samples)
        if rate_of_change is not None:
            signals.append(
                self._signal(
                    SignalType.ACCELERATION,
                    observation,
                    rate_of_change,
                    self._quality(acceleration_samples, config.acceleration_window_seconds),
                )
            )

        heading_samples = history.recent(config.heading_window_seconds)
        turn_rate = heading_change_rate(heading_samples, config.min_heading_displacement)
        if turn_rate is not None:
            signals.append(
                self._signal(
                    SignalType.HEADING_CHANGE_RATE,
                    observation,
                    turn_rate,
                    self._quality(heading_samples, config.heading_window_seconds),
                )
            )

        # Stationary duration is the one quantity that must outlive the
        # retained window — a person can stand still for far longer than the
        # history covers — so it is carried on the subject's history as a spell
        # start, updated here from this frame's speed.
        window_start = (
            speed_samples[0].timestamp_seconds if speed_samples else observation.timestamp_seconds
        )
        spell_start = stationary_since(
            history.stationary_since,
            speed=speed,
            window_start_seconds=window_start,
            threshold=config.stationary_speed_threshold,
            continuous=history.is_continuous,
        )
        history.set_stationary_since(spell_start)
        if spell_start is not None:
            signals.append(
                self._signal(
                    SignalType.STATIONARY_DURATION,
                    observation,
                    max(0.0, observation.timestamp_seconds - spell_start),
                    speed_quality,
                )
            )

        return signals

    def _signal(
        self,
        signal_type: SignalType,
        observation: BehaviorObservation,
        value: float,
        quality: float | None,
    ) -> BehaviorSignal:
        return BehaviorSignal(
            signal_type=signal_type,
            subject=observation.subject,
            frame_index=observation.frame_index,
            timestamp_seconds=observation.timestamp_seconds,
            value=value,
            space=observation.space,
            quality=quality,
            zone_ids=observation.zone_ids,
        )

    @staticmethod
    def _quality(samples: Sequence[BehaviorObservation], window_seconds: float) -> float | None:
        """How much of the configured window the samples actually cover.

        A newly acquired track has only a fraction of a window behind it, and
        its speed is correspondingly less well supported. Reporting that as a
        number lets a consumer weight or ignore thin estimates instead of
        treating every value as equally solid. It says nothing about whether
        the measurement is *correct*.
        """
        if len(samples) < 2 or window_seconds <= 0:
            return None
        return min(1.0, span_seconds(samples) / window_seconds)

    def log_summary(self) -> None:
        """Log a final summary. Called once at the end of a run."""
        logger.info(
            "Behavior signals:    %d emitted in %s space over %d tracked subject(s)",
            self._signals_emitted,
            self.space.value,
            self.tracked_subjects,
        )

"""Frozen result types for a venue validation run.

Everything here is an *execution* statistic: what ran, how much of it ran, and
what the runtime decided. None of it is an accuracy figure. Without human
annotations there is no recall, no precision, no false-merge rate — and a zero
must never stand in for a measurement nobody made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atreides_vision.association.models import AssociationOutcome
from atreides_vision.fusion.models import IntegrationOutcome
from atreides_vision.timing import FrameRateSource, StreamRef, TimeUncertainty
from atreides_vision.venue.models import stream_sort_key

__all__ = [
    "GROUND_TRUTH_STATUS",
    "VenueValidationStreamSummary",
    "VenueValidationSummary",
    "alignment_record",
    "stream_summary_sort_key",
]

#: No run this harness performs is annotated, so every summary says so in the
#: same words the validation manifests use.
GROUND_TRUTH_STATUS = "NOT_ANNOTATED"

_SECONDS_PRECISION = 6
_FPS_PRECISION = 4
_MIN_SERIALISED_HALF_WIDTH = 10.0**-_SECONDS_PRECISION


def alignment_record(timing: TimeUncertainty) -> dict[str, Any]:
    """Uncertainty as data, matching the V0.4.0 session stream's encoding."""
    half_width = timing.half_width_seconds
    if half_width is not None:
        half_width = max(round(half_width, _SECONDS_PRECISION), _MIN_SERIALISED_HALF_WIDTH)
    return {"basis": timing.basis.value, "half_width_seconds": half_width}


@dataclass(frozen=True, slots=True)
class VenueValidationStreamSummary:
    """What one ``(camera_id, stream_epoch)`` stream actually contributed.

    ``frames_processed`` of zero is a legitimate outcome, not a failure: an
    empty stream is a supported V0.4.0 case and the session continues without it.
    """

    stream: StreamRef
    fps: float
    fps_source: FrameRateSource
    frames_processed: int
    observations_emitted: int
    calibrated: bool
    integrity_monitored: bool

    @property
    def contributed(self) -> bool:
        return self.frames_processed > 0

    def to_record(self) -> dict[str, Any]:
        return {
            "camera_id": self.stream.camera_id,
            "stream_epoch": self.stream.stream_epoch,
            "fps": round(self.fps, _FPS_PRECISION),
            "fps_source": self.fps_source.value,
            "frames_processed": self.frames_processed,
            "observations_emitted": self.observations_emitted,
            "calibrated": self.calibrated,
            "integrity_monitored": self.integrity_monitored,
        }


def stream_summary_sort_key(summary: VenueValidationStreamSummary) -> tuple[str, int]:
    """Canonical ordering, identical to the venue layer's stream ordering."""
    return stream_sort_key(summary.stream)


@dataclass(frozen=True, slots=True)
class VenueValidationSummary:
    """One run, described in counts.

    Held in canonical order throughout, so two runs of equivalent inputs produce
    the same record whatever order the manifest listed its cameras in.
    """

    session_id: str
    streams: tuple[VenueValidationStreamSummary, ...]
    venue_instants: int
    source_frames_processed: int
    observations_emitted: int
    association_outcomes: dict[AssociationOutcome, int]
    integration_outcomes: dict[IntegrationOutcome, int]
    venue_subjects_created: int
    venue_subjects_current: int
    occupancy_snapshots: int
    min_occupancy: int | None
    max_occupancy: int | None
    mean_occupancy: float | None
    suppressed_duplicates: int
    first_venue_seconds: float | None
    last_venue_seconds: float | None
    session_alignment: TimeUncertainty

    @property
    def configured_camera_count(self) -> int:
        return len({summary.stream.camera_id for summary in self.streams})

    @property
    def contributing_camera_ids(self) -> tuple[str, ...]:
        contributing = {s.stream.camera_id for s in self.streams if s.contributed}
        return tuple(sorted(contributing))

    @property
    def zero_frame_streams(self) -> tuple[StreamRef, ...]:
        return tuple(s.stream for s in self.streams if not s.contributed)

    @property
    def venue_span_seconds(self) -> float | None:
        """Venue-time extent of the run, or ``None`` if nothing was processed."""
        if self.first_venue_seconds is None or self.last_venue_seconds is None:
            return None
        return round(self.last_venue_seconds - self.first_venue_seconds, _SECONDS_PRECISION)

    def to_record(self) -> dict[str, Any]:
        first = self.first_venue_seconds
        last = self.last_venue_seconds
        return {
            "session_id": self.session_id,
            "ground_truth": GROUND_TRUTH_STATUS,
            "configured_camera_count": self.configured_camera_count,
            "configured_stream_count": len(self.streams),
            "contributing_camera_ids": list(self.contributing_camera_ids),
            "zero_frame_streams": [
                {"camera_id": stream.camera_id, "stream_epoch": stream.stream_epoch}
                for stream in self.zero_frame_streams
            ],
            "streams": [summary.to_record() for summary in self.streams],
            "venue_instants": self.venue_instants,
            "source_frames_processed": self.source_frames_processed,
            "observations_emitted": self.observations_emitted,
            "association_outcome_counts": {
                outcome.value: self.association_outcomes.get(outcome, 0)
                for outcome in AssociationOutcome
            },
            "integration_outcome_counts": {
                outcome.value: self.integration_outcomes.get(outcome, 0)
                for outcome in IntegrationOutcome
            },
            "venue_subjects_created": self.venue_subjects_created,
            "venue_subjects_current": self.venue_subjects_current,
            "occupancy": {
                "snapshots": self.occupancy_snapshots,
                "min_current": self.min_occupancy,
                "max_current": self.max_occupancy,
                "mean_current": self.mean_occupancy,
                "suppressed_duplicates": self.suppressed_duplicates,
            },
            "venue_time": {
                "first_seconds": None if first is None else round(first, _SECONDS_PRECISION),
                "last_seconds": None if last is None else round(last, _SECONDS_PRECISION),
                "span_seconds": self.venue_span_seconds,
                "session_time_alignment": alignment_record(self.session_alignment),
            },
        }

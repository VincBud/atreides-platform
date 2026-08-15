"""The machine-readable behaviour output contract, as versioned JSON Lines.

This is the boundary a future intelligence or risk layer consumes. It must be
able to do so **without parsing logs, without importing this package, and
without knowing that RF-DETR or OC-SORT exist**. Everything it needs — units,
coordinate space, provenance, identity scope — travels in the data.

File shape
----------
One JSON object per line. The first line is a ``run_metadata`` record; every
line after it is an ``observation`` record, one per subject per frame::

    {"record_type":"run_metadata","schema_version":"0.3.2",...}
    {"record_type":"observation","schema_version":"0.3.2","frame_index":1,...}
    {"record_type":"observation","schema_version":"0.3.2","frame_index":1,...}

Every record repeats ``schema_version`` and ``camera_id``, and metric records
repeat ``venue_id`` and ``surface_id``. The redundancy is deliberate: JSONL is
consumed a line at a time, lines from several cameras may later be merged into
one stream, and a line that cannot be interpreted alone is a line that will
eventually be misinterpreted.

Position and signals are separate
---------------------------------
``position`` is an observation — where the subject's anchor projected to on
*this* frame. ``signals`` are derived from a window of frames. A record may
carry a position and no signals, which means "seen here, not yet enough history
to measure motion". That is not the same as a zero speed, and the format keeps
the two distinguishable.

Identity scope
--------------
Records are keyed by ``camera_id`` plus ``subject.track_id``. **A camera-local
track id is not a unique real-world person**, and two cameras reporting nearby
venue coordinates are two independent observations, not one person. Nothing in
this format expresses cross-camera identity, and a consumer must not infer it.

Nothing here is biometric: no appearance descriptor, no persistent identifier,
no name, no demographic attribute, and no semantic behaviour label.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self

from atreides_vision.behavior.coordinates import MotionSpace
from atreides_vision.behavior.models import (
    BehaviorFrame,
    BehaviorSignal,
    BehaviorSubjectSnapshot,
)
from atreides_vision.errors import BehaviorOutputError
from atreides_vision.sampling import IntervalSampler

logger = logging.getLogger(__name__)

__all__ = [
    "BEHAVIOR_SCHEMA_VERSION",
    "GZIP_SUFFIX",
    "BehaviorEmissionPolicy",
    "BehaviorRunMetadata",
    "BehaviorWriter",
    "CalibrationProvenance",
    "EmitEveryFrame",
    "IntegrityFinding",
    "IntervalEmissionPolicy",
    "JsonlBehaviorWriter",
    "SourceProvenance",
    "integrity_record",
    "observation_records",
]


class IntegrityFinding(Protocol):
    """A calibration-integrity result, as this module needs to see one.

    Structural rather than a concrete import: the behaviour package does not
    depend on the calibration package, and that invariant is what lets behaviour
    analysis run with no calibration at all. The calibration layer's
    ``IntegrityAssessment`` satisfies this shape, and the composition root is
    what brings the two together.
    """

    @property
    def allows_metric_output(self) -> bool: ...

    def to_record(self) -> dict[str, Any]: ...


#: Version of the record contract below. Bump when the shape changes in a way a
#: consumer would notice; the value is written onto every record so a reader can
#: reject a stream it does not understand rather than silently mis-parsing it.
#:
#: 0.3.2 -> 0.3.3: a third ``record_type`` was added (``calibration_integrity``).
#: Observation and run-metadata records are structurally unchanged, so a 0.3.2
#: consumer reading only those keeps working — but a consumer that asserts on the
#: set of record types will now see one it does not know, and that is a contract
#: change worth announcing rather than slipping through.
BEHAVIOR_SCHEMA_VERSION = "0.3.3"

#: Decimal places for positions. Six is a micrometre in metric mode and far
#: below one pixel in normalized mode — precision the measurement does not
#: have, but rounding is for file size, not for implying accuracy.
_POSITION_PRECISION = 6
_VALUE_PRECISION = 6
_TIMESTAMP_PRECISION = 3
_QUALITY_PRECISION = 3

#: Output paths ending in this are gzip-compressed automatically. One
#: convention, visible in the filename, so a reader can tell what a file is
#: without a flag having been recorded anywhere.
GZIP_SUFFIX = ".gz"


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """What was processed, so a record set can be traced back to its input."""

    path: str
    width: int
    height: int
    fps: float

    def to_record(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 4),
        }


@dataclass(frozen=True, slots=True)
class CalibrationProvenance:
    """Which calibration produced the metric coordinates in this stream.

    A flat snapshot of facts rather than the calibration itself: repeating every
    correspondence on every record would bloat the file, and a consumer needs to
    *identify* the calibration, not re-derive it. The full correspondences stay
    in the calibration file.

    Deliberately plain data. This module does not import the calibration
    package — the composition root fills these fields in — which keeps the
    behaviour layer usable with no calibration at all.

    **The error fields measure fit self-consistency, not real-world accuracy.**
    They state how well the homography reproduces the operator's own reference
    points. If those points were mismeasured on the floor, a small error here
    simply means the calibration faithfully reproduces the wrong answer.
    """

    venue_id: str
    surface_id: str
    reference_width: int
    reference_height: int
    correspondence_count: int
    mean_venue_error_m: float
    max_venue_error_m: float
    mean_image_error_px: float
    max_image_error_px: float
    relative_error: float
    venue_extent_m: float

    def to_record(self) -> dict[str, Any]:
        return {
            "venue_id": self.venue_id,
            "surface_id": self.surface_id,
            "reference_frame": {"width": self.reference_width, "height": self.reference_height},
            "correspondence_count": self.correspondence_count,
            "reprojection": {
                "mean_venue_error_m": round(self.mean_venue_error_m, _POSITION_PRECISION),
                "max_venue_error_m": round(self.max_venue_error_m, _POSITION_PRECISION),
                "mean_image_error_px": round(self.mean_image_error_px, 4),
                "max_image_error_px": round(self.max_image_error_px, 4),
                "relative_error": round(self.relative_error, _POSITION_PRECISION),
                "venue_extent_m": round(self.venue_extent_m, 4),
                "measures": "fit self-consistency, not validated real-world accuracy",
            },
        }


@dataclass(frozen=True, slots=True)
class BehaviorRunMetadata:
    """The header record: everything constant for the whole stream."""

    camera_id: str
    space: MotionSpace
    source: SourceProvenance | None = None
    #: Present only when a ground-plane calibration was in effect.
    calibration: CalibrationProvenance | None = None

    def __post_init__(self) -> None:
        if not self.camera_id or not self.camera_id.strip():
            raise BehaviorOutputError("Behaviour output requires a non-empty camera_id")
        if self.space is MotionSpace.GROUND_PLANE_METRES and self.calibration is None:
            raise BehaviorOutputError(
                "Metric behaviour output must carry calibration provenance; otherwise a "
                "consumer cannot tell which calibration produced the metres."
            )

    @property
    def position_unit(self) -> str:
        return self.space.distance_unit

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "record_type": "run_metadata",
            "schema_version": BEHAVIOR_SCHEMA_VERSION,
            "camera_id": self.camera_id,
            "space": self.space.value,
            "position_unit": self.position_unit,
            "is_physical": self.space.is_physical,
            "identity_scope": "camera_local_track",
        }
        if self.source is not None:
            record["source"] = self.source.to_record()
        record["calibration"] = None if self.calibration is None else self.calibration.to_record()
        return record


def _signal_record(signal: BehaviorSignal) -> dict[str, Any]:
    """One signal, nested inside an observation record.

    Leaner than :meth:`BehaviorSignal.to_record`, which is the standalone form:
    subject, frame and timestamp are already on the parent record and repeating
    them per signal would triple the file for no gain.
    """
    record: dict[str, Any] = {
        "type": signal.signal_type.value,
        "value": round(signal.value, _VALUE_PRECISION),
        "unit": signal.unit,
    }
    if signal.quality is not None:
        record["quality"] = round(signal.quality, _QUALITY_PRECISION)
    return record


def _observation_record(
    snapshot: BehaviorSubjectSnapshot, metadata: BehaviorRunMetadata
) -> dict[str, Any]:
    """One subject, on one frame."""
    record: dict[str, Any] = {
        "record_type": "observation",
        "schema_version": BEHAVIOR_SCHEMA_VERSION,
        "camera_id": snapshot.subject.camera_id,
        "frame_index": snapshot.frame_index,
        "timestamp_seconds": round(snapshot.timestamp_seconds, _TIMESTAMP_PRECISION),
        "subject": {"track_id": snapshot.subject.track_id},
        "position": {
            "x": round(snapshot.position.x, _POSITION_PRECISION),
            "y": round(snapshot.position.y, _POSITION_PRECISION),
            "space": snapshot.space.value,
            "unit": snapshot.space.distance_unit,
        },
        "signals": [_signal_record(signal) for signal in snapshot.signals],
    }
    if metadata.calibration is not None:
        record["venue_id"] = metadata.calibration.venue_id
        record["surface_id"] = metadata.calibration.surface_id
    if snapshot.zone_ids:
        record["zone_ids"] = list(snapshot.zone_ids)
    return record


def observation_records(
    frame: BehaviorFrame, metadata: BehaviorRunMetadata
) -> list[dict[str, Any]]:
    """Every observation record for one frame, in stable subject order.

    Order follows the analyzer's snapshot order, which follows the tracker's
    track order, so the same input always serialises identically.
    """
    return [_observation_record(snapshot, metadata) for snapshot in frame.snapshots]


class BehaviorEmissionPolicy(Protocol):
    """Decides which analysed frames get serialised.

    **Emission only.** The analyzer still ingests every processed frame
    regardless of what this returns, so speed, acceleration, heading and
    stationary duration are always computed from the complete history. Sampling
    here reduces file size, never signal quality — which is why the decision
    lives at the writer rather than anywhere upstream of the analyzer.
    """

    def should_emit(self, frame_index: int, timestamp_seconds: float) -> bool: ...


class EmitEveryFrame:
    """Serialise every analysed frame. The default, preserving V0.3.2 behaviour."""

    def should_emit(self, frame_index: int, timestamp_seconds: float) -> bool:
        return True


class IntervalEmissionPolicy:
    """Serialise at most one frame per interval of source time.

    Frame-level rather than per-subject: when a frame is emitted, all of its
    subjects are. That keeps a frame's observations together in the file and
    keeps subject ordering deterministic, which per-subject schedules would
    scatter.

    Measured in source seconds, so the same setting means the same thing on
    25 FPS and 60 FPS footage. The first analysed frame is always emitted, so a
    run shorter than one interval still produces observations.
    """

    def __init__(self, interval_seconds: float) -> None:
        if interval_seconds <= 0:
            raise BehaviorOutputError(
                f"Behaviour output interval must be positive, got {interval_seconds}. "
                "Omit it entirely to emit every frame."
            )
        self._sampler = IntervalSampler(interval_seconds)

    @property
    def interval_seconds(self) -> float:
        return self._sampler.interval_seconds

    def should_emit(self, frame_index: int, timestamp_seconds: float) -> bool:
        return self._sampler.should_sample(timestamp_seconds)


def integrity_record(assessment: IntegrityFinding, metadata: BehaviorRunMetadata) -> dict[str, Any]:
    """One calibration-integrity record.

    Camera- and run-level context: it describes the camera's geometry, never any
    individual. Emitted as its own record type rather than repeated on every
    observation, which would bloat the file and imply the finding is about a
    person.
    """
    record: dict[str, Any] = {
        "record_type": "calibration_integrity",
        "schema_version": BEHAVIOR_SCHEMA_VERSION,
        "camera_id": metadata.camera_id,
        **assessment.to_record(),
        "metric_output_allowed": assessment.allows_metric_output,
    }
    if metadata.calibration is not None:
        record["venue_id"] = metadata.calibration.venue_id
        record["surface_id"] = metadata.calibration.surface_id
    return record


class BehaviorWriter(Protocol):
    """A destination for behaviour records."""

    def write_frame(self, frame: BehaviorFrame) -> None: ...

    def write_integrity(self, assessment: IntegrityFinding) -> None: ...

    def release(self) -> None: ...


class JsonlBehaviorWriter:
    """Streams behaviour records to a ``.jsonl`` file, one line per record.

    Streaming, never accumulating: each frame is serialised and written as it
    arrives, so a long run costs no more memory than a short one and the file
    can be tailed while processing.

    Each line is built as a complete string and written in one call, so an
    interrupted run leaves whole records followed by nothing. Earlier lines
    remaining after a mid-run failure is expected and fine; a truncated JSON
    object would not be.

    The header record is written on construction, which means an unwritable
    destination fails immediately rather than after the detector has loaded and
    a thousand frames have been processed.

    Use as a context manager::

        with JsonlBehaviorWriter(path, metadata) as writer:
            writer.write_frame(frame)
    """

    def __init__(
        self,
        path: Path | str,
        metadata: BehaviorRunMetadata,
        emission_policy: BehaviorEmissionPolicy | None = None,
        compress: bool | None = None,
    ) -> None:
        self._path = Path(path)
        self._metadata = metadata
        self._policy = emission_policy or EmitEveryFrame()
        self._compressed = (
            self._path.suffix.lower() == GZIP_SUFFIX if compress is None else compress
        )
        self._records_written = 0
        self._frames_written = 0
        self._frames_analysed = 0
        self._integrity_records_written = 0
        self._released = False

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # gzip.open in text mode streams: each write is compressed and
            # flushed into the file, so memory does not grow with run length.
            # Deliberately long-lived: this class *is* the context manager, and
            # the handle is closed by release(). ruff cannot see that ownership.
            self._handle = (
                gzip.open(self._path, "wt", encoding="utf-8")  # noqa: SIM115
                if self._compressed
                else self._path.open("w", encoding="utf-8")
            )
        except OSError as exc:
            raise BehaviorOutputError(
                f"Could not open behaviour output {self._path}: {exc}"
            ) from exc

        self._write_line(metadata.to_record())

    @property
    def path(self) -> Path:
        return self._path

    @property
    def metadata(self) -> BehaviorRunMetadata:
        return self._metadata

    @property
    def records_written(self) -> int:
        """Observation records written. Excludes the header."""
        return self._records_written

    @property
    def frames_written(self) -> int:
        """Frames actually serialised. Differs from :attr:`frames_analysed`
        whenever an emission interval is configured."""
        return self._frames_written

    @property
    def frames_analysed(self) -> int:
        """Frames offered to the writer, emitted or not.

        Exposed so a run can show that analysis saw every frame even when only
        a sample of them was written.
        """
        return self._frames_analysed

    @property
    def integrity_records_written(self) -> int:
        return self._integrity_records_written

    @property
    def is_compressed(self) -> bool:
        return self._compressed

    def write_frame(self, frame: BehaviorFrame) -> None:
        """Append one line per subject observed in this frame, if it is due.

        Every frame offered here has already been analysed in full; the emission
        policy only decides whether this one is serialised. A frame with no
        subjects writes nothing, so an empty scene costs no file space.
        """
        if self._released:
            raise BehaviorOutputError(f"Cannot write to a released behaviour writer: {self._path}")
        self._frames_analysed += 1

        if not self._policy.should_emit(frame.frame_index, frame.timestamp_seconds):
            return

        for record in observation_records(frame, self._metadata):
            self._write_line(record)
            self._records_written += 1
        self._frames_written += 1

    def write_integrity(self, assessment: IntegrityFinding) -> None:
        """Append one calibration-integrity record.

        Never sampled: integrity findings are rare, and a suppressed one would
        leave a consumer unable to explain a gap in the observations.
        """
        if self._released:
            raise BehaviorOutputError(f"Cannot write to a released behaviour writer: {self._path}")
        self._write_line(integrity_record(assessment, self._metadata))
        self._integrity_records_written += 1

    def _write_line(self, record: dict[str, Any]) -> None:
        """Serialise fully, then write once, so no half-record can reach disk."""
        try:
            line = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
        except (TypeError, ValueError) as exc:
            raise BehaviorOutputError(
                f"Could not serialise a behaviour record for {self._path}: {exc}"
            ) from exc
        try:
            self._handle.write(line)
        except OSError as exc:
            raise BehaviorOutputError(
                f"Could not write behaviour output to {self._path}: {exc}"
            ) from exc

    def release(self) -> None:
        """Flush and close. Idempotent, and safe after an exception."""
        if not getattr(self, "_released", False):
            self._handle.close()
            self._released = True
            logger.debug(
                "Released behaviour writer %s after %d records across %d frames",
                self._path,
                self._records_written,
                self._frames_written,
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

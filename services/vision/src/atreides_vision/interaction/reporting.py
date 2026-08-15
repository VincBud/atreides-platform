"""The machine-readable interaction output contract, as versioned JSON Lines.

A **separate stream** from behaviour output, deliberately. Pairs have different
cardinality (quadratic rather than linear in subject count), different history,
different lifetimes, and a different schema version. Interleaving them would
make both harder to consume and would tie the pairwise contract's version to
the individual one's for no benefit.

File shape
----------
One JSON object per line. The first is a ``run_metadata`` record; every line
after it is an ``interaction`` record, one per measured pair per emitted frame::

    {"record_type":"run_metadata","schema_version":"0.3.4",...}
    {"record_type":"interaction","schema_version":"0.3.4","frame_index":1,...}

Geometry and signals are separate
---------------------------------
``current`` is an observation — how far apart the two were on *this* frame.
``signals`` are derived over a window. A record may carry a distance and no
signals, meaning "measured this far apart, not yet enough history for a rate".
That is not a zero closing speed, and the format keeps them distinguishable.

Identity scope
--------------
Records name two **camera-local** subjects on one camera. A camera-local track
id is not a unique real-world person, and a pair is therefore two tracklets, not
two people. Nothing here expresses cross-camera identity, and no consumer may
infer it.

Nothing here is semantic. There is no confrontation, aggression, threat,
conflict or escalation field, and no appearance descriptor, biometric or
persistent identifier of any kind.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self

from atreides_vision.behavior.coordinates import MotionSpace
from atreides_vision.behavior.reporting import BehaviorEmissionPolicy, EmitEveryFrame
from atreides_vision.errors import InteractionOutputError
from atreides_vision.interaction.models import InteractionFrame, InteractionSnapshot
from atreides_vision.jsonl import JsonlStreamWriter

logger = logging.getLogger(__name__)

__all__ = [
    "INTERACTION_SCHEMA_VERSION",
    "InteractionRunMetadata",
    "InteractionWriter",
    "JsonlInteractionWriter",
    "interaction_records",
]

#: Version of the record contract below. Owned independently of the behaviour
#: schema: the two streams evolve for different reasons, and tying them together
#: would force a version bump on one whenever the other changed.
INTERACTION_SCHEMA_VERSION = "0.3.4"

_VALUE_PRECISION = 6
_TIMESTAMP_PRECISION = 3


@dataclass(frozen=True, slots=True)
class InteractionRunMetadata:
    """The header record: everything constant for the whole stream."""

    camera_id: str
    space: MotionSpace
    #: Present only for metric analysis, where they establish which physical
    #: plane the distances describe.
    venue_id: str | None = None
    surface_id: str | None = None
    #: Proximity threshold in effect, in the space's distance units. Recorded so
    #: a consumer can tell what a proximity duration was measured against
    #: instead of assuming a value.
    proximity_threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.camera_id or not self.camera_id.strip():
            raise InteractionOutputError("Interaction output requires a non-empty camera_id")
        if self.space is MotionSpace.GROUND_PLANE_METRES and (
            not self.venue_id or not self.surface_id
        ):
            raise InteractionOutputError(
                "Metric interaction output must name its venue and surface; otherwise a "
                "consumer cannot tell which physical plane the distances describe."
            )

    @property
    def distance_unit(self) -> str:
        return self.space.distance_unit

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "record_type": "run_metadata",
            "schema_version": INTERACTION_SCHEMA_VERSION,
            "stream": "interaction",
            "camera_id": self.camera_id,
            "space": self.space.value,
            "distance_unit": self.distance_unit,
            "is_physical": self.space.is_physical,
            "identity_scope": "camera_local_track_pair",
            "pair_scope": "same_camera_only",
        }
        if self.venue_id is not None:
            record["venue_id"] = self.venue_id
        if self.surface_id is not None:
            record["surface_id"] = self.surface_id
        if self.proximity_threshold is not None:
            record["proximity_threshold"] = round(self.proximity_threshold, _VALUE_PRECISION)
        return record


def _interaction_record(
    snapshot: InteractionSnapshot, metadata: InteractionRunMetadata
) -> dict[str, Any]:
    """One pair, on one frame."""
    record: dict[str, Any] = {
        "record_type": "interaction",
        "schema_version": INTERACTION_SCHEMA_VERSION,
        "camera_id": snapshot.pair.camera_id,
        "frame_index": snapshot.frame_index,
        "timestamp_seconds": round(snapshot.timestamp_seconds, _TIMESTAMP_PRECISION),
        # Canonically ordered by the pair reference, so the same two subjects
        # always serialise the same way regardless of tracker output order.
        "subjects": [
            {"track_id": snapshot.pair.first.track_id},
            {"track_id": snapshot.pair.second.track_id},
        ],
        "space": snapshot.space.value,
        "current": snapshot.geometry.to_record(),
        "signals": [signal.to_record() for signal in snapshot.signals],
    }
    if metadata.venue_id is not None:
        record["venue_id"] = metadata.venue_id
    if metadata.surface_id is not None:
        record["surface_id"] = metadata.surface_id
    if snapshot.shared_zone_ids:
        record["shared_zone_ids"] = list(snapshot.shared_zone_ids)
    return record


def interaction_records(
    frame: InteractionFrame, metadata: InteractionRunMetadata
) -> list[dict[str, Any]]:
    """Every interaction record for one frame, in stable pair order."""
    return [_interaction_record(snapshot, metadata) for snapshot in frame.snapshots]


class InteractionWriter(Protocol):
    """A destination for interaction records."""

    def write_frame(self, frame: InteractionFrame) -> None: ...

    def release(self) -> None: ...


class JsonlInteractionWriter:
    """Streams interaction records to a ``.jsonl`` file, one line per pair.

    Shares :class:`~atreides_vision.jsonl.JsonlStreamWriter` with the behaviour
    stream, so gzip selection, whole-line writes and lifecycle behave
    identically across both outputs rather than being reimplemented.

    Use as a context manager::

        with JsonlInteractionWriter(path, metadata) as writer:
            writer.write_frame(interaction_frame)
    """

    def __init__(
        self,
        path: Path | str,
        metadata: InteractionRunMetadata,
        emission_policy: BehaviorEmissionPolicy | None = None,
        compress: bool | None = None,
    ) -> None:
        self._metadata = metadata
        self._policy = emission_policy or EmitEveryFrame()
        self._stream = JsonlStreamWriter(
            path, InteractionOutputError, "interaction output", compress
        )
        self._records_written = 0
        self._frames_written = 0
        self._frames_analysed = 0

        self._stream.write_record(metadata.to_record())

    @property
    def path(self) -> Path:
        return self._stream.path

    @property
    def metadata(self) -> InteractionRunMetadata:
        return self._metadata

    @property
    def records_written(self) -> int:
        """Pair records written. Excludes the header."""
        return self._records_written

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def frames_analysed(self) -> int:
        """Frames offered to the writer, emitted or not.

        Diverges from :attr:`frames_written` under an emission interval, which
        is what shows that pair history saw every frame while only a sample was
        serialised.
        """
        return self._frames_analysed

    @property
    def is_compressed(self) -> bool:
        return self._stream.is_compressed

    def write_frame(self, frame: InteractionFrame) -> None:
        """Append one line per measured pair, if this frame is due.

        Every frame offered here has already been analysed in full; the emission
        policy only decides whether it is serialised. A frame with no measured
        pairs writes nothing.
        """
        if self._stream.is_released:
            raise InteractionOutputError(
                f"Cannot write to a released interaction writer: {self._stream.path}"
            )
        self._frames_analysed += 1

        if not self._policy.should_emit(frame.frame_index, frame.timestamp_seconds):
            return

        for record in interaction_records(frame, self._metadata):
            self._stream.write_record(record)
            self._records_written += 1
        self._frames_written += 1

    def release(self) -> None:
        """Flush and close. Idempotent, and safe after an exception."""
        if not self._stream.is_released:
            self._stream.release()
            logger.debug(
                "Released interaction writer %s after %d records across %d frames",
                self._stream.path,
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

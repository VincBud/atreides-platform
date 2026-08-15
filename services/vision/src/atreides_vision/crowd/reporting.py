"""The machine-readable crowd output contract, as versioned JSON Lines.

A **separate stream** from behaviour and interaction, with its own schema
version. Clusters have different cardinality from both, different history and a
different lifetime; tying the versions together would force a bump on one
whenever another changed.

File shape
----------
One JSON object per line: a ``run_metadata`` header, then per emitted frame one
``crowd_frame`` summary followed by one ``spatial_cluster`` record per cluster::

    {"record_type":"run_metadata","schema_version":"0.3.5",...}
    {"record_type":"crowd_frame","schema_version":"0.3.5","frame_index":200,...}
    {"record_type":"spatial_cluster","schema_version":"0.3.5","frame_index":200,...}

The frame summary is emitted even when there are no clusters, because "four
people were observed and none were clustered" is a measurement, and a reader
that saw only cluster records could not distinguish it from a frame nobody was
in.

What a spatial cluster is not
-----------------------------
It records that several camera-local observations were spatially connected under
a geometric rule at one instant. It is **not** a social group, a friend group,
an affiliation, or any persistent entity, and no consumer may read it as one.

Members are camera-local tracklets, so ``member_count`` is not a count of unique
real-world people: occlusion can split one person into several tracks, and
overlapping cameras are never merged.

Nothing here is semantic. There is no crowd risk, density verdict, panic,
unrest, escalation or safety field, and no biometric, appearance or persistent
identifier of any kind.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self

from atreides_vision.behavior.coordinates import MotionSpace
from atreides_vision.behavior.reporting import BehaviorEmissionPolicy, EmitEveryFrame
from atreides_vision.crowd.models import CrowdFrame, SpatialClusterSnapshot
from atreides_vision.errors import CrowdOutputError
from atreides_vision.jsonl import JsonlStreamWriter

logger = logging.getLogger(__name__)

__all__ = [
    "CROWD_SCHEMA_VERSION",
    "CrowdRunMetadata",
    "CrowdWriter",
    "JsonlCrowdWriter",
    "crowd_records",
]

#: Version of the record contract below, owned independently of the behaviour
#: (0.3.3) and interaction (0.3.4) schemas.
CROWD_SCHEMA_VERSION = "0.3.5"

_VALUE_PRECISION = 6
_TIMESTAMP_PRECISION = 3


def _density_unit(space: MotionSpace) -> str:
    """Units for a density in this space.

    Normalized density is emphatically **not** ``subjects/m²``: it is a
    perspective-dependent, camera-local number whose unit area has no physical
    size. Naming it separately is what stops it from being read as a crowd
    density.
    """
    return "subjects/m^2" if space.is_physical else "subjects/normalized_area_unit"


@dataclass(frozen=True, slots=True)
class CrowdRunMetadata:
    """The header record: everything constant for the whole stream."""

    camera_id: str
    space: MotionSpace
    #: Present only for metric analysis, where they establish which physical
    #: plane the geometry describes.
    venue_id: str | None = None
    surface_id: str | None = None
    #: Geometric parameters in effect, in the space's distance units. Recorded so
    #: a consumer can tell what the clusters were built with instead of assuming.
    cluster_radius: float | None = None
    density_radius: float | None = None
    minimum_cluster_size: int | None = None

    def __post_init__(self) -> None:
        if not self.camera_id or not self.camera_id.strip():
            raise CrowdOutputError("Crowd output requires a non-empty camera_id")
        if self.space is MotionSpace.GROUND_PLANE_METRES and (
            not self.venue_id or not self.surface_id
        ):
            raise CrowdOutputError(
                "Metric crowd output must name its venue and surface; otherwise a consumer "
                "cannot tell which physical plane the cluster geometry describes."
            )

    @property
    def distance_unit(self) -> str:
        return self.space.distance_unit

    @property
    def density_unit(self) -> str:
        return _density_unit(self.space)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "record_type": "run_metadata",
            "schema_version": CROWD_SCHEMA_VERSION,
            "stream": "crowd",
            "analysis": "spatial_cluster_crowd_dynamics",
            "camera_id": self.camera_id,
            "space": self.space.value,
            "distance_unit": self.distance_unit,
            "density_unit": self.density_unit,
            "is_physical": self.space.is_physical,
            "identity_scope": "camera_local_tracks",
            "cluster_scope": "same_camera_only",
            "cluster_rule": "radius_connectivity_connected_components",
        }
        if self.venue_id is not None:
            record["venue_id"] = self.venue_id
        if self.surface_id is not None:
            record["surface_id"] = self.surface_id
        for name, value in (
            ("cluster_radius", self.cluster_radius),
            ("density_radius", self.density_radius),
        ):
            if value is not None:
                record[name] = round(value, _VALUE_PRECISION)
        if self.minimum_cluster_size is not None:
            record["minimum_cluster_size"] = self.minimum_cluster_size
        return record


def _measure(value: float | None, unit: str) -> dict[str, Any] | None:
    """A measurement with its unit attached, or ``None`` if not measured."""
    return None if value is None else {"value": round(value, _VALUE_PRECISION), "unit": unit}


def _cluster_record(cluster: SpatialClusterSnapshot, metadata: CrowdRunMetadata) -> dict[str, Any]:
    """One spatial cluster, on one frame."""
    distance_unit = cluster.space.distance_unit
    density_unit = _density_unit(cluster.space)

    metrics: dict[str, Any] = {
        "member_count": cluster.member_count,
        "spatial_spread": _measure(cluster.spatial_spread, distance_unit),
        "persistence_duration": _measure(cluster.persistence_duration, "s"),
    }
    for name, value, unit in (
        ("mean_pair_distance", cluster.mean_pair_distance, distance_unit),
        ("max_pair_distance", cluster.max_pair_distance, distance_unit),
        ("mean_local_density", cluster.mean_local_density, density_unit),
        ("max_local_density", cluster.max_local_density, density_unit),
    ):
        measured = _measure(value, unit)
        if measured is not None:
            metrics[name] = measured
    if cluster.motion_heading_coherence is not None:
        # Dimensionless by construction, so it carries no unit field.
        metrics["motion_heading_coherence"] = round(
            cluster.motion_heading_coherence, _VALUE_PRECISION
        )

    record: dict[str, Any] = {
        "record_type": "spatial_cluster",
        "schema_version": CROWD_SCHEMA_VERSION,
        "camera_id": cluster.camera_id,
        "frame_index": cluster.frame_index,
        "timestamp_seconds": round(cluster.timestamp_seconds, _TIMESTAMP_PRECISION),
        # Canonically ordered, so the same cluster always serialises identically.
        "members": [{"track_id": member.track_id} for member in cluster.members],
        "space": cluster.space.value,
        "centroid": {
            "x": round(cluster.centroid.x, _VALUE_PRECISION),
            "y": round(cluster.centroid.y, _VALUE_PRECISION),
            "unit": distance_unit,
        },
        "metrics": metrics,
    }
    if metadata.venue_id is not None:
        record["venue_id"] = metadata.venue_id
    if metadata.surface_id is not None:
        record["surface_id"] = metadata.surface_id
    if cluster.common_zone_ids:
        record["common_zone_ids"] = list(cluster.common_zone_ids)
    if cluster.member_zone_ids:
        record["member_zone_ids"] = list(cluster.member_zone_ids)
    return record


def _frame_record(frame: CrowdFrame, metadata: CrowdRunMetadata) -> dict[str, Any]:
    """The per-frame summary: counts only, no cluster detail repeated."""
    record: dict[str, Any] = {
        "record_type": "crowd_frame",
        "schema_version": CROWD_SCHEMA_VERSION,
        "camera_id": metadata.camera_id,
        "frame_index": frame.frame_index,
        "timestamp_seconds": round(frame.timestamp_seconds, _TIMESTAMP_PRECISION),
        "space": metadata.space.value,
        "summary": frame.summary.to_record(),
    }
    if metadata.venue_id is not None:
        record["venue_id"] = metadata.venue_id
    if metadata.surface_id is not None:
        record["surface_id"] = metadata.surface_id
    return record


def crowd_records(frame: CrowdFrame, metadata: CrowdRunMetadata) -> list[dict[str, Any]]:
    """The frame summary followed by every cluster, in stable order."""
    return [
        _frame_record(frame, metadata),
        *(_cluster_record(cluster, metadata) for cluster in frame.clusters),
    ]


class CrowdWriter(Protocol):
    """A destination for crowd records."""

    def write_frame(self, frame: CrowdFrame) -> None: ...

    def release(self) -> None: ...


class JsonlCrowdWriter:
    """Streams crowd records to a ``.jsonl`` file.

    Shares :class:`~atreides_vision.jsonl.JsonlStreamWriter` with the behaviour
    and interaction streams, so gzip selection, whole-line writes and lifecycle
    behave identically across all three rather than being reimplemented.
    """

    def __init__(
        self,
        path: Path | str,
        metadata: CrowdRunMetadata,
        emission_policy: BehaviorEmissionPolicy | None = None,
        compress: bool | None = None,
    ) -> None:
        self._metadata = metadata
        self._policy = emission_policy or EmitEveryFrame()
        self._stream = JsonlStreamWriter(path, CrowdOutputError, "crowd output", compress)
        self._cluster_records_written = 0
        self._frames_written = 0
        self._frames_analysed = 0

        self._stream.write_record(metadata.to_record())

    @property
    def path(self) -> Path:
        return self._stream.path

    @property
    def metadata(self) -> CrowdRunMetadata:
        return self._metadata

    @property
    def cluster_records_written(self) -> int:
        """Cluster records written. Excludes headers and frame summaries."""
        return self._cluster_records_written

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def frames_analysed(self) -> int:
        """Frames offered to the writer, emitted or not.

        Diverges from :attr:`frames_written` under an emission interval, which is
        what shows that cluster persistence saw every frame while only a sample
        was serialised.
        """
        return self._frames_analysed

    @property
    def is_compressed(self) -> bool:
        return self._stream.is_compressed

    def write_frame(self, frame: CrowdFrame) -> None:
        """Append this frame's summary and clusters, if the frame is due.

        Every frame offered here has already been analysed in full; the emission
        policy only decides whether it is serialised.
        """
        if self._stream.is_released:
            raise CrowdOutputError(f"Cannot write to a released crowd writer: {self._stream.path}")
        self._frames_analysed += 1

        if not self._policy.should_emit(frame.frame_index, frame.timestamp_seconds):
            return

        for record in crowd_records(frame, self._metadata):
            self._stream.write_record(record)
            if record["record_type"] == "spatial_cluster":
                self._cluster_records_written += 1
        self._frames_written += 1

    def release(self) -> None:
        """Flush and close. Idempotent, and safe after an exception."""
        if not self._stream.is_released:
            self._stream.release()
            logger.debug(
                "Released crowd writer %s after %d cluster records across %d frames",
                self._stream.path,
                self._cluster_records_written,
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

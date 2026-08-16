"""Builders for synthetic V0.3 output records, for validation-harness tests.

Every validation metric is checked against records built here rather than
against a pipeline run, so each test states its own analytic input: "this track
was observed on frames 1-30 at 10 FPS, therefore its lifetime is 2.9 s". A test
that had to run a detector could not make that kind of claim.

The shapes mirror the real writers' output exactly — behaviour schema 0.3.3,
interaction 0.3.4, crowd 0.3.5 — so a schema change breaks these tests, which is
the point.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from atreides_vision.behavior.reporting import BEHAVIOR_SCHEMA_VERSION
from atreides_vision.crowd.reporting import CROWD_SCHEMA_VERSION
from atreides_vision.interaction.reporting import INTERACTION_SCHEMA_VERSION

__all__ = [
    "behavior_header",
    "behavior_stream",
    "crowd_frame_record",
    "crowd_header",
    "interaction_header",
    "interaction_record",
    "observation",
    "spatial_cluster_record",
]

CAMERA = "cam_test"


def behavior_header(space: str = "normalized_image", camera_id: str = CAMERA) -> dict[str, Any]:
    return {
        "record_type": "run_metadata",
        "schema_version": BEHAVIOR_SCHEMA_VERSION,
        "camera_id": camera_id,
        "space": space,
        "position_unit": "normalized_frame_unit",
        "is_physical": False,
        "identity_scope": "camera_local_track",
    }


def observation(
    track_id: int,
    frame_index: int,
    timestamp_seconds: float,
    signals: Sequence[str] = (),
    camera_id: str = CAMERA,
) -> dict[str, Any]:
    """One subject observation, carrying the named signal types."""
    return {
        "record_type": "observation",
        "schema_version": BEHAVIOR_SCHEMA_VERSION,
        "camera_id": camera_id,
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "subject": {"track_id": track_id},
        "position": {
            "x": 0.5,
            "y": 0.5,
            "space": "normalized_image",
            "unit": "normalized_frame_unit",
        },
        "signals": [
            {"type": name, "value": 0.1, "unit": "normalized_frame_unit/s"} for name in signals
        ],
    }


def behavior_stream(
    tracks: Iterable[tuple[int, Sequence[int]]],
    fps: float = 10.0,
    signals_from_frame: int | None = None,
    signals: Sequence[str] = ("speed",),
    camera_id: str = CAMERA,
) -> list[dict[str, Any]]:
    """A behaviour stream for tracks described as ``(track_id, frame_indices)``.

    Records come out in frame order then track order, exactly as the writer
    serialises them, because the accumulators detect frame boundaries from
    ``frame_index`` changing.

    Args:
        tracks: Each track's id and the 1-based frames it was observed on.
        fps: Frame rate, used to derive timestamps.
        signals_from_frame: Frame index from which each track carries
            ``signals``, counted from that track's own first observation.
            ``None`` means no observation carries any signal.
    """
    by_frame: dict[int, list[tuple[int, int]]] = {}
    first_frame: dict[int, int] = {}
    for track_id, frames in tracks:
        ordered = sorted(frames)
        first_frame[track_id] = ordered[0]
        for frame_index in ordered:
            by_frame.setdefault(frame_index, []).append((track_id, frame_index))

    records: list[dict[str, Any]] = [behavior_header(camera_id=camera_id)]
    for frame_index in sorted(by_frame):
        for track_id, _ in sorted(by_frame[frame_index]):
            carries = (
                signals_from_frame is not None
                and (frame_index - first_frame[track_id]) >= signals_from_frame
            )
            records.append(
                observation(
                    track_id=track_id,
                    frame_index=frame_index,
                    timestamp_seconds=(frame_index - 1) / fps,
                    signals=signals if carries else (),
                    camera_id=camera_id,
                )
            )
    return records


def interaction_header(camera_id: str = CAMERA) -> dict[str, Any]:
    return {
        "record_type": "run_metadata",
        "schema_version": INTERACTION_SCHEMA_VERSION,
        "stream": "interaction",
        "camera_id": camera_id,
        "space": "normalized_image",
        "distance_unit": "normalized_frame_unit",
        "is_physical": False,
        "identity_scope": "camera_local_track_pair",
        "pair_scope": "same_camera_only",
    }


def interaction_record(
    first: int,
    second: int,
    frame_index: int,
    timestamp_seconds: float,
    signals: Sequence[str] = (),
    camera_id: str = CAMERA,
) -> dict[str, Any]:
    return {
        "record_type": "interaction",
        "schema_version": INTERACTION_SCHEMA_VERSION,
        "camera_id": camera_id,
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "subjects": [{"track_id": first}, {"track_id": second}],
        "space": "normalized_image",
        "current": {
            "distance": {
                "value": 0.1,
                "unit": "normalized_frame_unit",
                "space": "normalized_image",
            }
        },
        "signals": [
            {"type": name, "value": 0.05, "unit": "normalized_frame_unit/s"} for name in signals
        ],
    }


def crowd_header(camera_id: str = CAMERA) -> dict[str, Any]:
    return {
        "record_type": "run_metadata",
        "schema_version": CROWD_SCHEMA_VERSION,
        "stream": "crowd",
        "analysis": "spatial_cluster_crowd_dynamics",
        "camera_id": camera_id,
        "space": "normalized_image",
        "distance_unit": "normalized_frame_unit",
        "density_unit": "subjects/normalized_area_unit",
        "is_physical": False,
        "identity_scope": "camera_local_tracks",
        "cluster_scope": "same_camera_only",
        "cluster_rule": "radius_connectivity_connected_components",
    }


def crowd_frame_record(
    frame_index: int,
    timestamp_seconds: float,
    observed: int,
    cluster_count: int,
    clustered: int,
    largest: int = 0,
    camera_id: str = CAMERA,
) -> dict[str, Any]:
    return {
        "record_type": "crowd_frame",
        "schema_version": CROWD_SCHEMA_VERSION,
        "camera_id": camera_id,
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "space": "normalized_image",
        "summary": {
            "observed_subject_count": observed,
            "cluster_count": cluster_count,
            "clustered_subject_count": clustered,
            "isolated_subject_count": observed - clustered,
            "largest_cluster_size": largest,
        },
    }


def spatial_cluster_record(
    members: Sequence[int],
    frame_index: int,
    timestamp_seconds: float,
    persistence_seconds: float,
    camera_id: str = CAMERA,
) -> dict[str, Any]:
    return {
        "record_type": "spatial_cluster",
        "schema_version": CROWD_SCHEMA_VERSION,
        "camera_id": camera_id,
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "members": [{"track_id": track_id} for track_id in members],
        "space": "normalized_image",
        "centroid": {"x": 0.5, "y": 0.5, "unit": "normalized_frame_unit"},
        "metrics": {
            "member_count": len(members),
            "spatial_spread": {"value": 0.05, "unit": "normalized_frame_unit"},
            "persistence_duration": {"value": persistence_seconds, "unit": "s"},
        },
    }

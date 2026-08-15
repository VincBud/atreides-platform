"""The crowd output contract, its writer, sampling and compression."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.behavior.reporting import IntervalEmissionPolicy
from atreides_vision.crowd.models import CrowdFrame, SpatialClusterSnapshot, canonical_members
from atreides_vision.crowd.reporting import (
    CROWD_SCHEMA_VERSION,
    CrowdRunMetadata,
    JsonlCrowdWriter,
    crowd_records,
)
from atreides_vision.errors import CrowdOutputError

METRIC = MotionSpace.GROUND_PLANE_METRES
NORMALIZED = MotionSpace.NORMALIZED_IMAGE
DT = 1 / 25.0


def metadata(space: MotionSpace = METRIC, **overrides: object) -> CrowdRunMetadata:
    values: dict[str, Any] = {
        "camera_id": "cam_01",
        "space": space,
        "venue_id": "venue_01" if space is METRIC else None,
        "surface_id": "main_floor" if space is METRIC else None,
        "cluster_radius": 1.5,
        "density_radius": 2.0,
        "minimum_cluster_size": 3,
    }
    values.update(overrides)
    return CrowdRunMetadata(**values)


def cluster(
    track_ids: tuple[int, ...] = (3, 7, 11),
    space: MotionSpace = METRIC,
    frame_index: int = 200,
    timestamp: float = 8.0,
    coherence: float | None = 0.86,
    common_zone_ids: tuple[str, ...] = (),
    member_zone_ids: tuple[str, ...] = (),
) -> SpatialClusterSnapshot:
    return SpatialClusterSnapshot(
        members=canonical_members([SubjectRef("cam_01", tid) for tid in track_ids]),
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        centroid=MotionPoint(5.2, 8.1, space),
        spatial_spread=0.7,
        mean_pair_distance=1.1,
        max_pair_distance=1.9,
        mean_local_density=0.24,
        max_local_density=0.31,
        motion_heading_coherence=coherence,
        persistence_duration=4.2,
        common_zone_ids=common_zone_ids,
        member_zone_ids=member_zone_ids,
    )


def frame(
    *clusters: SpatialClusterSnapshot,
    observed: int = 5,
    index: int = 200,
    timestamp: float = 8.0,
) -> CrowdFrame:
    return CrowdFrame.from_clusters(index, timestamp, clusters, observed)


def read(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def of_type(records: list[dict[str, Any]], record_type: str) -> list[dict[str, Any]]:
    return [record for record in records if record["record_type"] == record_type]


class TestSchema:
    def test_version_is_owned_independently(self) -> None:
        from atreides_vision.behavior.reporting import BEHAVIOR_SCHEMA_VERSION
        from atreides_vision.interaction.reporting import INTERACTION_SCHEMA_VERSION

        assert CROWD_SCHEMA_VERSION == "0.3.5"
        assert CROWD_SCHEMA_VERSION not in {BEHAVIOR_SCHEMA_VERSION, INTERACTION_SCHEMA_VERSION}

    def test_it_appears_on_every_record(self) -> None:
        run_metadata = metadata()
        records = [run_metadata.to_record(), *crowd_records(frame(cluster()), run_metadata)]
        assert all(record["schema_version"] == CROWD_SCHEMA_VERSION for record in records)


class TestRunMetadata:
    def test_metric_header(self) -> None:
        record = metadata().to_record()
        assert record["record_type"] == "run_metadata"
        assert record["stream"] == "crowd"
        assert record["analysis"] == "spatial_cluster_crowd_dynamics"
        assert record["space"] == "ground_plane_metres"
        assert record["distance_unit"] == "m"
        assert record["density_unit"] == "subjects/m^2"
        assert record["venue_id"] == "venue_01"
        assert record["cluster_radius"] == pytest.approx(1.5)
        assert record["density_radius"] == pytest.approx(2.0)
        assert record["minimum_cluster_size"] == 3

    def test_normalized_header_never_claims_square_metres(self) -> None:
        """Normalized density has no physical unit area."""
        record = metadata(NORMALIZED).to_record()
        assert record["density_unit"] == "subjects/normalized_area_unit"
        assert record["distance_unit"] == "normalized_frame_unit"
        assert record["is_physical"] is False
        assert "venue_id" not in record

    def test_it_declares_the_identity_and_cluster_scope(self) -> None:
        record = metadata().to_record()
        assert record["identity_scope"] == "camera_local_tracks"
        assert record["cluster_scope"] == "same_camera_only"
        assert record["cluster_rule"] == "radius_connectivity_connected_components"

    def test_metric_output_requires_venue_and_surface(self) -> None:
        with pytest.raises(CrowdOutputError, match="must name its venue and surface"):
            CrowdRunMetadata(camera_id="cam_01", space=METRIC)

    @pytest.mark.parametrize("camera_id", ["", "   "])
    def test_requires_a_camera_id(self, camera_id: str) -> None:
        with pytest.raises(CrowdOutputError, match="non-empty camera_id"):
            CrowdRunMetadata(camera_id=camera_id, space=NORMALIZED)


class TestRecords:
    def test_a_frame_summary_precedes_the_clusters(self) -> None:
        records = crowd_records(frame(cluster(), cluster((20, 21, 22))), metadata())
        assert [record["record_type"] for record in records] == [
            "crowd_frame",
            "spatial_cluster",
            "spatial_cluster",
        ]

    def test_the_frame_summary_is_emitted_even_with_no_clusters(self) -> None:
        """Otherwise a reader cannot tell an empty scene from an unclustered one."""
        records = crowd_records(frame(observed=4), metadata())
        assert len(records) == 1
        summary = records[0]["summary"]
        assert summary["observed_subject_count"] == 4
        assert summary["cluster_count"] == 0
        assert summary["isolated_subject_count"] == 4

    def test_cluster_shape(self) -> None:
        record = of_type(crowd_records(frame(cluster()), metadata()), "spatial_cluster")[0]
        assert record["camera_id"] == "cam_01"
        assert record["frame_index"] == 200
        assert record["timestamp_seconds"] == pytest.approx(8.0)
        assert record["members"] == [{"track_id": 3}, {"track_id": 7}, {"track_id": 11}]
        assert record["space"] == "ground_plane_metres"
        assert record["centroid"] == {
            "x": pytest.approx(5.2),
            "y": pytest.approx(8.1),
            "unit": "m",
        }

    def test_members_are_canonically_ordered(self) -> None:
        forward = of_type(crowd_records(frame(cluster((3, 7, 11))), metadata()), "spatial_cluster")
        reverse = of_type(crowd_records(frame(cluster((11, 3, 7))), metadata()), "spatial_cluster")
        assert forward[0]["members"] == reverse[0]["members"]

    def test_metrics_carry_units(self) -> None:
        metrics = of_type(crowd_records(frame(cluster()), metadata()), "spatial_cluster")[0][
            "metrics"
        ]
        assert metrics["member_count"] == 3
        assert metrics["spatial_spread"] == {"value": pytest.approx(0.7), "unit": "m"}
        assert metrics["mean_pair_distance"]["unit"] == "m"
        assert metrics["max_pair_distance"]["unit"] == "m"
        assert metrics["mean_local_density"]["unit"] == "subjects/m^2"
        assert metrics["persistence_duration"] == {"value": pytest.approx(4.2), "unit": "s"}

    def test_coherence_is_dimensionless(self) -> None:
        metrics = of_type(crowd_records(frame(cluster()), metadata()), "spatial_cluster")[0][
            "metrics"
        ]
        assert metrics["motion_heading_coherence"] == pytest.approx(0.86)

    def test_coherence_is_omitted_when_unmeasured(self) -> None:
        """Absent, never a zero that would read as headings cancelling out."""
        metrics = of_type(
            crowd_records(frame(cluster(coherence=None)), metadata()), "spatial_cluster"
        )[0]["metrics"]
        assert "motion_heading_coherence" not in metrics

    def test_normalized_density_is_labelled_separately(self) -> None:
        record = of_type(
            crowd_records(frame(cluster(space=NORMALIZED)), metadata(NORMALIZED)),
            "spatial_cluster",
        )[0]
        assert record["metrics"]["mean_local_density"]["unit"] == "subjects/normalized_area_unit"
        assert record["centroid"]["unit"] == "normalized_frame_unit"

    def test_venue_context_only_in_metric_mode(self) -> None:
        metric = of_type(crowd_records(frame(cluster()), metadata()), "spatial_cluster")[0]
        assert metric["venue_id"] == "venue_01"
        assert metric["surface_id"] == "main_floor"

        normalized = of_type(
            crowd_records(frame(cluster(space=NORMALIZED)), metadata(NORMALIZED)),
            "spatial_cluster",
        )[0]
        assert "venue_id" not in normalized

    def test_zone_context_is_optional(self) -> None:
        with_zones = of_type(
            crowd_records(
                frame(cluster(common_zone_ids=("bar",), member_zone_ids=("bar", "walkway"))),
                metadata(),
            ),
            "spatial_cluster",
        )[0]
        assert with_zones["common_zone_ids"] == ["bar"]
        assert with_zones["member_zone_ids"] == ["bar", "walkway"]

        without = of_type(crowd_records(frame(cluster()), metadata()), "spatial_cluster")[0]
        assert "common_zone_ids" not in without

    def test_no_semantic_or_identity_fields_are_emitted(self) -> None:
        """The privacy and risk boundaries, asserted on the serialised form."""
        serialised = json.dumps(
            crowd_records(frame(cluster(common_zone_ids=("bar",))), metadata())
        ).lower()
        for forbidden in (
            "risk",
            "danger",
            "panic",
            "unrest",
            "escalat",
            "aggress",
            "threat",
            "suspicious",
            "overcrowd",
            "unsafe",
            "group_id",
            "person_id",
            "embedding",
            "biometric",
        ):
            assert forbidden not in serialised

    def test_ordering_is_deterministic(self) -> None:
        run_metadata = metadata()
        multi = frame(cluster((1, 2, 3)), cluster((7, 8, 9)))

        def serialise() -> str:
            return "\n".join(json.dumps(record) for record in crowd_records(multi, run_metadata))

        assert serialise() == serialise()


class TestWriter:
    def test_writes_a_header_then_frames_and_clusters(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        with JsonlCrowdWriter(path, metadata()) as writer:
            writer.write_frame(frame(cluster((1, 2, 3)), cluster((7, 8, 9))))

        records = read(path)
        assert [record["record_type"] for record in records] == [
            "run_metadata",
            "crowd_frame",
            "spatial_cluster",
            "spatial_cluster",
        ]

    def test_counts_cluster_records(self, tmp_path: Path) -> None:
        with JsonlCrowdWriter(tmp_path / "c.jsonl", metadata()) as writer:
            writer.write_frame(frame(cluster((1, 2, 3)), cluster((7, 8, 9))))
            writer.write_frame(frame(cluster((1, 2, 3))))
            assert writer.cluster_records_written == 3
            assert writer.frames_written == 2

    def test_a_frame_with_no_clusters_still_writes_its_summary(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        with JsonlCrowdWriter(path, metadata()) as writer:
            writer.write_frame(frame(observed=2))
            assert writer.cluster_records_written == 0
        assert len(of_type(read(path), "crowd_frame")) == 1

    def test_sampling_reduces_output_without_hiding_analysis(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        with JsonlCrowdWriter(
            path, metadata(), emission_policy=IntervalEmissionPolicy(0.2)
        ) as writer:
            for index in range(1, 51):
                writer.write_frame(
                    frame(cluster((1, 2, 3)), index=index, timestamp=(index - 1) * DT)
                )
            assert writer.frames_analysed == 50
            assert writer.frames_written == 10

        assert len(of_type(read(path), "spatial_cluster")) == 10

    def test_metadata_is_never_sampled_away(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        with JsonlCrowdWriter(
            path, metadata(), emission_policy=IntervalEmissionPolicy(1000.0)
        ) as writer:
            writer.write_frame(frame(cluster(), index=1, timestamp=0.0))
        assert read(path)[0]["record_type"] == "run_metadata"

    def test_gzip_by_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl.gz"
        with JsonlCrowdWriter(path, metadata()) as writer:
            assert writer.is_compressed
            for index in range(1, 21):
                writer.write_frame(frame(cluster(), index=index, timestamp=index * DT))

        assert path.read_bytes()[:2] == b"\x1f\x8b"
        assert len(of_type(read(path), "spatial_cluster")) == 20

    def test_compressed_content_matches_the_plain_form(self, tmp_path: Path) -> None:
        def write(name: str) -> list[dict[str, Any]]:
            path = tmp_path / name
            with JsonlCrowdWriter(path, metadata()) as writer:
                for index in range(1, 16):
                    writer.write_frame(
                        frame(
                            cluster((1, 2, 3)),
                            cluster((7, 8, 9)),
                            index=index,
                            timestamp=index * DT,
                        )
                    )
            return read(path)

        assert write("plain.jsonl") == write("compressed.jsonl.gz")

    def test_every_line_is_complete_after_a_failure(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        with pytest.raises(RuntimeError), JsonlCrowdWriter(path, metadata()) as writer:
            writer.write_frame(frame(cluster()))
            raise RuntimeError("processing blew up")

        records = read(path)
        assert records[0]["record_type"] == "run_metadata"
        assert len(of_type(records, "spatial_cluster")) == 1

    def test_writing_after_release_fails(self, tmp_path: Path) -> None:
        writer = JsonlCrowdWriter(tmp_path / "c.jsonl", metadata())
        writer.release()
        with pytest.raises(CrowdOutputError, match="released crowd writer"):
            writer.write_frame(frame(cluster()))

    def test_release_is_idempotent(self, tmp_path: Path) -> None:
        writer = JsonlCrowdWriter(tmp_path / "c.jsonl", metadata())
        writer.release()
        writer.release()

    def test_unwritable_destination_fails_clearly(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        with pytest.raises(CrowdOutputError, match="Could not open crowd output"):
            JsonlCrowdWriter(blocker / "c.jsonl", metadata())

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deeper" / "c.jsonl"
        with JsonlCrowdWriter(path, metadata()) as writer:
            writer.write_frame(frame(cluster()))
        assert path.is_file()

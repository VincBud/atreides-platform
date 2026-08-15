"""The interaction output contract, its writer, sampling and compression."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import pytest

from atreides_vision.behavior.coordinates import MotionSpace
from atreides_vision.behavior.models import SubjectRef
from atreides_vision.behavior.reporting import IntervalEmissionPolicy
from atreides_vision.errors import InteractionOutputError
from atreides_vision.interaction.models import (
    InteractionFrame,
    InteractionSignal,
    InteractionSignalType,
    InteractionSnapshot,
    PairGeometry,
    SubjectPairRef,
)
from atreides_vision.interaction.reporting import (
    INTERACTION_SCHEMA_VERSION,
    InteractionRunMetadata,
    JsonlInteractionWriter,
    interaction_records,
)

METRIC = MotionSpace.GROUND_PLANE_METRES
NORMALIZED = MotionSpace.NORMALIZED_IMAGE
DT = 1 / 25.0


def metadata(space: MotionSpace = METRIC, **overrides: object) -> InteractionRunMetadata:
    values: dict[str, Any] = {
        "camera_id": "cam_01",
        "space": space,
        "venue_id": "venue_01" if space is METRIC else None,
        "surface_id": "main_floor" if space is METRIC else None,
        "proximity_threshold": 1.5,
    }
    values.update(overrides)
    return InteractionRunMetadata(**values)


def pair(first: int = 7, second: int = 12) -> SubjectPairRef:
    return SubjectPairRef(SubjectRef("cam_01", first), SubjectRef("cam_01", second))


def snapshot(
    first: int = 7,
    second: int = 12,
    distance: float = 0.84,
    space: MotionSpace = METRIC,
    frame_index: int = 140,
    timestamp: float = 5.6,
    shared_zone_ids: tuple[str, ...] = (),
    signals: tuple[InteractionSignal, ...] | None = None,
) -> InteractionSnapshot:
    return InteractionSnapshot(
        pair=pair(first, second),
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        geometry=PairGeometry(distance, space),
        signals=signals
        if signals is not None
        else (
            InteractionSignal(InteractionSignalType.CLOSING_SPEED, 1.12, space),
            InteractionSignal(InteractionSignalType.PROXIMITY_DURATION, 3.4, space),
        ),
        shared_zone_ids=shared_zone_ids,
    )


def frame(
    *snapshots: InteractionSnapshot, index: int = 140, timestamp: float = 5.6
) -> InteractionFrame:
    return InteractionFrame.from_snapshots(index, timestamp, snapshots)


def read(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def interactions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record["record_type"] == "interaction"]


class TestSchema:
    def test_version_is_owned_independently(self) -> None:
        """The interaction contract versions separately from the behaviour one."""
        from atreides_vision.behavior.reporting import BEHAVIOR_SCHEMA_VERSION

        assert INTERACTION_SCHEMA_VERSION == "0.3.4"
        assert INTERACTION_SCHEMA_VERSION != BEHAVIOR_SCHEMA_VERSION

    def test_it_appears_on_every_record(self) -> None:
        run_metadata = metadata()
        records = [run_metadata.to_record(), *interaction_records(frame(snapshot()), run_metadata)]
        assert all(record["schema_version"] == INTERACTION_SCHEMA_VERSION for record in records)


class TestRunMetadata:
    def test_metric_header(self) -> None:
        record = metadata().to_record()
        assert record["record_type"] == "run_metadata"
        assert record["stream"] == "interaction"
        assert record["space"] == "ground_plane_metres"
        assert record["distance_unit"] == "m"
        assert record["is_physical"] is True
        assert record["venue_id"] == "venue_01"
        assert record["surface_id"] == "main_floor"
        assert record["proximity_threshold"] == pytest.approx(1.5)

    def test_normalized_header_omits_venue(self) -> None:
        record = metadata(NORMALIZED).to_record()
        assert record["distance_unit"] == "normalized_frame_unit"
        assert record["is_physical"] is False
        assert "venue_id" not in record
        assert "surface_id" not in record

    def test_it_declares_the_identity_and_pair_scope(self) -> None:
        """A consumer must not be able to mistake these for venue-level pairs."""
        record = metadata().to_record()
        assert record["identity_scope"] == "camera_local_track_pair"
        assert record["pair_scope"] == "same_camera_only"

    def test_metric_output_requires_venue_and_surface(self) -> None:
        with pytest.raises(InteractionOutputError, match="must name its venue and surface"):
            InteractionRunMetadata(camera_id="cam_01", space=METRIC)

    @pytest.mark.parametrize("camera_id", ["", "   "])
    def test_requires_a_camera_id(self, camera_id: str) -> None:
        with pytest.raises(InteractionOutputError, match="non-empty camera_id"):
            InteractionRunMetadata(camera_id=camera_id, space=NORMALIZED)


class TestInteractionRecords:
    def test_shape(self) -> None:
        record = interaction_records(frame(snapshot()), metadata())[0]
        assert record["record_type"] == "interaction"
        assert record["camera_id"] == "cam_01"
        assert record["frame_index"] == 140
        assert record["timestamp_seconds"] == pytest.approx(5.6)
        assert record["subjects"] == [{"track_id": 7}, {"track_id": 12}]
        assert record["space"] == "ground_plane_metres"

    def test_current_geometry_is_separate_from_signals(self) -> None:
        record = interaction_records(frame(snapshot()), metadata())[0]
        assert record["current"]["distance"] == {
            "value": pytest.approx(0.84),
            "unit": "m",
            "space": "ground_plane_metres",
        }
        assert [entry["type"] for entry in record["signals"]] == [
            "closing_speed",
            "proximity_duration",
        ]

    def test_subjects_are_canonically_ordered(self) -> None:
        """Both construction orders serialise identically."""
        forward = interaction_records(frame(snapshot(7, 12)), metadata())[0]
        reverse = interaction_records(frame(snapshot(12, 7)), metadata())[0]
        assert forward["subjects"] == reverse["subjects"] == [{"track_id": 7}, {"track_id": 12}]

    def test_signal_units(self) -> None:
        record = interaction_records(
            frame(
                snapshot(
                    signals=(
                        InteractionSignal(InteractionSignalType.CLOSING_SPEED, 1.1, METRIC),
                        InteractionSignal(
                            InteractionSignalType.RELATIVE_MOTION_HEADING, math.pi, METRIC
                        ),
                        InteractionSignal(InteractionSignalType.PROXIMITY_DURATION, 3.4, METRIC),
                    )
                )
            ),
            metadata(),
        )[0]
        assert [entry["unit"] for entry in record["signals"]] == ["m/s", "rad", "s"]

    def test_a_pair_with_no_signals_still_reports_geometry(self) -> None:
        """ "Measured this far apart, no rate yet" is not a zero closing speed."""
        record = interaction_records(frame(snapshot(signals=())), metadata())[0]
        assert record["signals"] == []
        assert record["current"]["distance"]["value"] == pytest.approx(0.84)

    def test_venue_context_only_in_metric_mode(self) -> None:
        metric = interaction_records(frame(snapshot()), metadata())[0]
        assert metric["venue_id"] == "venue_01"
        assert metric["surface_id"] == "main_floor"

        normalized = interaction_records(frame(snapshot(space=NORMALIZED)), metadata(NORMALIZED))[0]
        assert "venue_id" not in normalized

    def test_shared_zones_are_optional_context(self) -> None:
        with_zones = interaction_records(
            frame(snapshot(shared_zone_ids=("dancefloor",))), metadata()
        )[0]
        assert with_zones["shared_zone_ids"] == ["dancefloor"]
        assert "shared_zone_ids" not in interaction_records(frame(snapshot()), metadata())[0]

    def test_no_semantic_or_identity_fields_are_emitted(self) -> None:
        """The privacy boundary, asserted on the serialised form."""
        serialised = json.dumps(
            interaction_records(frame(snapshot(shared_zone_ids=("bar",))), metadata())[0]
        ).lower()
        for forbidden in (
            "aggress",
            "fight",
            "threat",
            "conflict",
            "escalat",
            "suspicious",
            "risk",
            "danger",
            "person_id",
            "embedding",
            "biometric",
        ):
            assert forbidden not in serialised

    def test_ordering_is_deterministic(self) -> None:
        run_metadata = metadata()
        multi = frame(snapshot(1, 2), snapshot(1, 3), snapshot(2, 3))

        def serialise() -> str:
            return "\n".join(
                json.dumps(record) for record in interaction_records(multi, run_metadata)
            )

        assert serialise() == serialise()
        assert [record["subjects"] for record in interaction_records(multi, run_metadata)] == [
            [{"track_id": 1}, {"track_id": 2}],
            [{"track_id": 1}, {"track_id": 3}],
            [{"track_id": 2}, {"track_id": 3}],
        ]


class TestWriter:
    def test_writes_a_header_then_pairs(self, tmp_path: Path) -> None:
        path = tmp_path / "i.jsonl"
        with JsonlInteractionWriter(path, metadata()) as writer:
            writer.write_frame(frame(snapshot(1, 2), snapshot(1, 3)))

        records = read(path)
        assert len(records) == 3
        assert records[0]["record_type"] == "run_metadata"
        assert [record["record_type"] for record in records[1:]] == ["interaction"] * 2

    def test_counts_records_and_frames(self, tmp_path: Path) -> None:
        with JsonlInteractionWriter(tmp_path / "i.jsonl", metadata()) as writer:
            writer.write_frame(frame(snapshot(1, 2), snapshot(1, 3)))
            writer.write_frame(frame(snapshot(1, 2)))
            assert writer.records_written == 3
            assert writer.frames_written == 2

    def test_an_empty_frame_writes_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "i.jsonl"
        with JsonlInteractionWriter(path, metadata()) as writer:
            writer.write_frame(frame())
            assert writer.records_written == 0
        assert len(read(path)) == 1

    def test_sampling_reduces_output_without_hiding_frames(self, tmp_path: Path) -> None:
        path = tmp_path / "i.jsonl"
        with JsonlInteractionWriter(
            path, metadata(), emission_policy=IntervalEmissionPolicy(0.2)
        ) as writer:
            for index in range(1, 51):
                writer.write_frame(frame(snapshot(1, 2), index=index, timestamp=(index - 1) * DT))
            assert writer.frames_analysed == 50
            assert writer.frames_written == 10

        assert len(interactions(read(path))) == 10

    def test_metadata_is_never_sampled_away(self, tmp_path: Path) -> None:
        path = tmp_path / "i.jsonl"
        with JsonlInteractionWriter(
            path, metadata(), emission_policy=IntervalEmissionPolicy(1000.0)
        ) as writer:
            writer.write_frame(frame(snapshot(), index=1, timestamp=0.0))
        assert read(path)[0]["record_type"] == "run_metadata"

    def test_all_pairs_of_an_emitted_frame_are_written(self, tmp_path: Path) -> None:
        """Sampling is per frame, so a frame is never half-written."""
        path = tmp_path / "i.jsonl"
        with JsonlInteractionWriter(
            path, metadata(), emission_policy=IntervalEmissionPolicy(0.2)
        ) as writer:
            for index in range(1, 11):
                writer.write_frame(
                    frame(
                        snapshot(1, 2),
                        snapshot(1, 3),
                        snapshot(2, 3),
                        index=index,
                        timestamp=(index - 1) * DT,
                    )
                )
        assert len(interactions(read(path))) == 6  # two emitted frames x three pairs

    def test_gzip_by_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "i.jsonl.gz"
        with JsonlInteractionWriter(path, metadata()) as writer:
            assert writer.is_compressed
            for index in range(1, 21):
                writer.write_frame(frame(snapshot(), index=index, timestamp=index * DT))

        assert path.read_bytes()[:2] == b"\x1f\x8b"
        assert len(interactions(read(path))) == 20

    def test_compressed_content_matches_the_plain_form(self, tmp_path: Path) -> None:
        def write(name: str) -> list[dict[str, Any]]:
            path = tmp_path / name
            with JsonlInteractionWriter(path, metadata()) as writer:
                for index in range(1, 16):
                    writer.write_frame(
                        frame(snapshot(1, 2), snapshot(1, 3), index=index, timestamp=index * DT)
                    )
            return read(path)

        assert write("plain.jsonl") == write("compressed.jsonl.gz")

    def test_every_line_is_complete_json_after_a_failure(self, tmp_path: Path) -> None:
        path = tmp_path / "i.jsonl"
        with pytest.raises(RuntimeError), JsonlInteractionWriter(path, metadata()) as writer:
            writer.write_frame(frame(snapshot()))
            raise RuntimeError("processing blew up")

        records = read(path)
        assert records[0]["record_type"] == "run_metadata"
        assert len(interactions(records)) == 1

    def test_writing_after_release_fails(self, tmp_path: Path) -> None:
        writer = JsonlInteractionWriter(tmp_path / "i.jsonl", metadata())
        writer.release()
        with pytest.raises(InteractionOutputError, match="released interaction writer"):
            writer.write_frame(frame(snapshot()))

    def test_release_is_idempotent(self, tmp_path: Path) -> None:
        writer = JsonlInteractionWriter(tmp_path / "i.jsonl", metadata())
        writer.release()
        writer.release()

    def test_unwritable_destination_fails_clearly(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        with pytest.raises(InteractionOutputError, match="Could not open interaction output"):
            JsonlInteractionWriter(blocker / "i.jsonl", metadata())

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deeper" / "i.jsonl"
        with JsonlInteractionWriter(path, metadata()) as writer:
            writer.write_frame(frame(snapshot()))
        assert path.is_file()

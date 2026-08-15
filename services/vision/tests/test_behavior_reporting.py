"""The versioned behaviour output contract and its streaming writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atreides_vision.behavior.coordinates import MotionPoint, MotionSpace
from atreides_vision.behavior.models import (
    BehaviorFrame,
    BehaviorSignal,
    BehaviorSubjectSnapshot,
    SignalType,
    SubjectRef,
)
from atreides_vision.behavior.reporting import (
    BEHAVIOR_SCHEMA_VERSION,
    BehaviorRunMetadata,
    CalibrationProvenance,
    JsonlBehaviorWriter,
    SourceProvenance,
    observation_records,
)
from atreides_vision.errors import BehaviorOutputError

NORMALIZED = MotionSpace.NORMALIZED_IMAGE
METRIC = MotionSpace.GROUND_PLANE_METRES


def provenance(**overrides: object) -> CalibrationProvenance:
    values: dict[str, object] = {
        "venue_id": "venue_x",
        "surface_id": "ground",
        "reference_width": 1920,
        "reference_height": 1080,
        "correspondence_count": 6,
        "mean_venue_error_m": 0.002,
        "max_venue_error_m": 0.004,
        "mean_image_error_px": 0.4,
        "max_image_error_px": 0.9,
        "relative_error": 0.0002,
        "venue_extent_m": 12.5,
    }
    values.update(overrides)
    return CalibrationProvenance(**values)  # type: ignore[arg-type]


def metadata(space: MotionSpace = NORMALIZED, **overrides: object) -> BehaviorRunMetadata:
    values: dict[str, object] = {
        "camera_id": "cam_01",
        "space": space,
        "source": SourceProvenance(path="in.mp4", width=1920, height=1080, fps=25.0),
        "calibration": provenance() if space is METRIC else None,
    }
    values.update(overrides)
    return BehaviorRunMetadata(**values)  # type: ignore[arg-type]


def signal(
    signal_type: SignalType = SignalType.SPEED,
    value: float = 1.25,
    space: MotionSpace = NORMALIZED,
    quality: float | None = 1.0,
    track_id: int = 7,
) -> BehaviorSignal:
    return BehaviorSignal(
        signal_type=signal_type,
        subject=SubjectRef("cam_01", track_id),
        frame_index=3,
        timestamp_seconds=0.08,
        value=value,
        space=space,
        quality=quality,
    )


def snapshot(
    track_id: int = 7,
    space: MotionSpace = NORMALIZED,
    signals: tuple[BehaviorSignal, ...] | None = None,
    zone_ids: tuple[str, ...] = (),
) -> BehaviorSubjectSnapshot:
    return BehaviorSubjectSnapshot(
        subject=SubjectRef("cam_01", track_id),
        frame_index=3,
        timestamp_seconds=0.08,
        position=MotionPoint(4.5, 8.25, space),
        signals=signals if signals is not None else (signal(space=space, track_id=track_id),),
        zone_ids=zone_ids,
    )


def frame(*snapshots: BehaviorSubjectSnapshot) -> BehaviorFrame:
    return BehaviorFrame.from_snapshots(3, 0.08, snapshots)


class TestSchemaVersion:
    def test_is_the_milestone_version(self) -> None:
        assert BEHAVIOR_SCHEMA_VERSION == "0.3.3"

    def test_observation_records_stay_structurally_compatible_with_0_3_2(self) -> None:
        """The bump announces a new record type, not a changed observation.

        V0.3.3 added ``calibration_integrity`` as a third ``record_type``. A
        0.3.2 consumer that reads observations keeps working, which is why this
        pins the observation keys rather than the version alone.
        """
        record = observation_records(frame(snapshot(space=METRIC)), metadata(METRIC))[0]
        required = {
            "record_type",
            "schema_version",
            "camera_id",
            "frame_index",
            "timestamp_seconds",
            "subject",
            "position",
            "signals",
        }
        assert required <= set(record)
        assert set(record["position"]) == {"x", "y", "space", "unit"}
        assert set(record["subject"]) == {"track_id"}

    def test_appears_on_every_record(self) -> None:
        """JSONL is read a line at a time; every line must be self-describing."""
        run_metadata = metadata()
        records = [run_metadata.to_record(), *observation_records(frame(snapshot()), run_metadata)]
        assert all(record["schema_version"] == BEHAVIOR_SCHEMA_VERSION for record in records)


class TestRunMetadata:
    def test_normalized_metadata_has_no_calibration(self) -> None:
        record = metadata().to_record()
        assert record["record_type"] == "run_metadata"
        assert record["space"] == "normalized_image"
        assert record["position_unit"] == "normalized_frame_unit"
        assert record["is_physical"] is False
        assert record["calibration"] is None

    def test_metric_metadata_carries_provenance(self) -> None:
        record = metadata(METRIC).to_record()
        assert record["space"] == "ground_plane_metres"
        assert record["position_unit"] == "m"
        assert record["is_physical"] is True
        calibration = record["calibration"]
        assert calibration["venue_id"] == "venue_x"
        assert calibration["surface_id"] == "ground"
        assert calibration["reference_frame"] == {"width": 1920, "height": 1080}
        assert calibration["correspondence_count"] == 6

    def test_provenance_states_what_the_error_means(self) -> None:
        """It must never be read as validated real-world accuracy."""
        reprojection = metadata(METRIC).to_record()["calibration"]["reprojection"]
        assert reprojection["mean_venue_error_m"] == pytest.approx(0.002)
        assert "not validated real-world accuracy" in reprojection["measures"]

    def test_declares_camera_local_identity_scope(self) -> None:
        assert metadata().to_record()["identity_scope"] == "camera_local_track"

    def test_records_the_source(self) -> None:
        source = metadata().to_record()["source"]
        assert source == {"path": "in.mp4", "width": 1920, "height": 1080, "fps": 25.0}

    def test_metric_output_requires_provenance(self) -> None:
        """Metres with no stated calibration cannot be traced or trusted."""
        with pytest.raises(BehaviorOutputError, match="must carry calibration provenance"):
            BehaviorRunMetadata(camera_id="cam_01", space=METRIC)

    @pytest.mark.parametrize("camera_id", ["", "   "])
    def test_requires_a_camera_id(self, camera_id: str) -> None:
        with pytest.raises(BehaviorOutputError, match="non-empty camera_id"):
            BehaviorRunMetadata(camera_id=camera_id, space=NORMALIZED)


class TestObservationRecords:
    def test_shape(self) -> None:
        record = observation_records(frame(snapshot()), metadata())[0]
        assert record["record_type"] == "observation"
        assert record["camera_id"] == "cam_01"
        assert record["frame_index"] == 3
        assert record["timestamp_seconds"] == pytest.approx(0.08)
        assert record["subject"] == {"track_id": 7}

    def test_position_carries_space_and_unit(self) -> None:
        record = observation_records(frame(snapshot(space=METRIC)), metadata(METRIC))[0]
        assert record["position"] == {
            "x": pytest.approx(4.5),
            "y": pytest.approx(8.25),
            "space": "ground_plane_metres",
            "unit": "m",
        }

    def test_normalized_position_is_labelled_as_such(self) -> None:
        record = observation_records(frame(snapshot()), metadata())[0]
        assert record["position"]["space"] == "normalized_image"
        assert record["position"]["unit"] == "normalized_frame_unit"

    def test_signals_are_nested_without_repeating_the_parent(self) -> None:
        record = observation_records(frame(snapshot()), metadata())[0]
        assert record["signals"] == [
            {
                "type": "speed",
                "value": pytest.approx(1.25),
                "unit": "normalized_frame_unit/s",
                "quality": 1.0,
            }
        ]
        assert "frame_index" not in record["signals"][0]
        assert "camera_id" not in record["signals"][0]

    def test_metric_signals_report_metric_units(self) -> None:
        signals = (
            signal(SignalType.SPEED, 1.4, METRIC),
            signal(SignalType.ACCELERATION, 0.5, METRIC),
            signal(SignalType.HEADING_CHANGE_RATE, 0.2, METRIC),
            signal(SignalType.STATIONARY_DURATION, 3.0, METRIC),
        )
        record = observation_records(
            frame(snapshot(space=METRIC, signals=signals)), metadata(METRIC)
        )[0]
        assert [entry["unit"] for entry in record["signals"]] == ["m/s", "m/s^2", "rad/s", "s"]

    def test_a_position_without_signals_is_still_recorded(self) -> None:
        """Seen-here-but-not-yet-measurable is not the same as a zero speed."""
        record = observation_records(frame(snapshot(signals=())), metadata())[0]
        assert record["signals"] == []
        assert record["position"]["x"] == pytest.approx(4.5)

    def test_venue_context_only_in_metric_mode(self) -> None:
        metric = observation_records(frame(snapshot(space=METRIC)), metadata(METRIC))[0]
        assert metric["venue_id"] == "venue_x"
        assert metric["surface_id"] == "ground"

        normalized = observation_records(frame(snapshot()), metadata())[0]
        assert "venue_id" not in normalized
        assert "surface_id" not in normalized

    def test_zone_context_is_optional_and_opaque(self) -> None:
        with_zones = observation_records(frame(snapshot(zone_ids=("entrance", "bar"))), metadata())[
            0
        ]
        assert with_zones["zone_ids"] == ["entrance", "bar"]
        assert "zone_ids" not in observation_records(frame(snapshot()), metadata())[0]

    def test_quality_is_omitted_when_absent(self) -> None:
        record = observation_records(frame(snapshot(signals=(signal(quality=None),))), metadata())[
            0
        ]
        assert "quality" not in record["signals"][0]

    def test_no_identity_or_semantic_fields_are_emitted(self) -> None:
        """The privacy boundary, asserted on the serialised form."""
        record = observation_records(frame(snapshot(space=METRIC)), metadata(METRIC))[0]
        serialised = json.dumps(record)
        for forbidden in ("person", "name", "embedding", "biometric", "risk", "aggress"):
            assert forbidden not in serialised.lower()


class TestOrdering:
    def test_subject_order_follows_the_frame(self) -> None:
        records = observation_records(
            frame(snapshot(track_id=9), snapshot(track_id=2), snapshot(track_id=5)), metadata()
        )
        assert [record["subject"]["track_id"] for record in records] == [9, 2, 5]

    def test_signal_order_is_preserved(self) -> None:
        signals = (
            signal(SignalType.SPEED),
            signal(SignalType.ACCELERATION),
            signal(SignalType.STATIONARY_DURATION),
        )
        record = observation_records(frame(snapshot(signals=signals)), metadata())[0]
        assert [entry["type"] for entry in record["signals"]] == [
            "speed",
            "acceleration",
            "stationary_duration",
        ]

    def test_serialisation_is_deterministic(self) -> None:
        run_metadata = metadata(METRIC)
        subject_frame = frame(
            snapshot(track_id=1, space=METRIC), snapshot(track_id=2, space=METRIC)
        )

        def serialise() -> str:
            return "\n".join(
                json.dumps(record, sort_keys=False)
                for record in observation_records(subject_frame, run_metadata)
            )

        assert serialise() == serialise()


class TestJsonlWriter:
    def test_writes_a_header_then_observations(self, tmp_path: Path) -> None:
        path = tmp_path / "behavior.jsonl"
        with JsonlBehaviorWriter(path, metadata()) as writer:
            writer.write_frame(frame(snapshot(track_id=1), snapshot(track_id=2)))

        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 3
        assert lines[0]["record_type"] == "run_metadata"
        assert [line["record_type"] for line in lines[1:]] == ["observation", "observation"]

    def test_header_is_written_immediately(self, tmp_path: Path) -> None:
        """An unwritable destination must fail before the detector loads."""
        path = tmp_path / "behavior.jsonl"
        writer = JsonlBehaviorWriter(path, metadata())
        writer.release()
        assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["record_type"] == (
            "run_metadata"
        )

    def test_counts_records_and_frames(self, tmp_path: Path) -> None:
        with JsonlBehaviorWriter(tmp_path / "b.jsonl", metadata()) as writer:
            writer.write_frame(frame(snapshot(track_id=1), snapshot(track_id=2)))
            writer.write_frame(frame(snapshot(track_id=1)))
            assert writer.records_written == 3
            assert writer.frames_written == 2

    def test_an_empty_frame_writes_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "b.jsonl"
        with JsonlBehaviorWriter(path, metadata()) as writer:
            writer.write_frame(frame())
            assert writer.records_written == 0
        assert len(path.read_text(encoding="utf-8").splitlines()) == 1

    def test_every_line_is_complete_json(self, tmp_path: Path) -> None:
        """Partial records must never reach disk."""
        path = tmp_path / "b.jsonl"
        with JsonlBehaviorWriter(path, metadata()) as writer:
            for index in range(20):
                writer.write_frame(frame(snapshot(track_id=index)))

        for line in path.read_text(encoding="utf-8").splitlines():
            assert json.loads(line)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deeper" / "b.jsonl"
        with JsonlBehaviorWriter(path, metadata()) as writer:
            writer.write_frame(frame(snapshot()))
        assert path.is_file()

    def test_release_is_idempotent(self, tmp_path: Path) -> None:
        writer = JsonlBehaviorWriter(tmp_path / "b.jsonl", metadata())
        writer.release()
        writer.release()

    def test_writing_after_release_fails(self, tmp_path: Path) -> None:
        writer = JsonlBehaviorWriter(tmp_path / "b.jsonl", metadata())
        writer.release()
        with pytest.raises(BehaviorOutputError, match="released behaviour writer"):
            writer.write_frame(frame(snapshot()))

    def test_unwritable_destination_fails_clearly(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        with pytest.raises(BehaviorOutputError, match="Could not open behaviour output"):
            JsonlBehaviorWriter(blocker / "b.jsonl", metadata())

    def test_closes_even_when_processing_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "b.jsonl"
        with (
            pytest.raises(RuntimeError),
            JsonlBehaviorWriter(path, metadata()) as writer,
        ):
            writer.write_frame(frame(snapshot()))
            raise RuntimeError("processing blew up")

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert all(json.loads(line) for line in lines)

    def test_exposes_its_path_and_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "b.jsonl"
        with JsonlBehaviorWriter(path, metadata()) as writer:
            assert writer.path == path
            assert writer.metadata.camera_id == "cam_01"

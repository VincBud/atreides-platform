"""V0.4.6: observability for the occupancy analyzer's snapshots.

Reuses the exact fixtures and scenarios ``test_venue_pipeline.py`` and
``test_occupancy_analyzer.py`` already establish, and checks only what this
milestone's reporting adds: that an ``OccupancySnapshot`` can be turned into
a deterministic, versioned JSONL record — as a separate, additive stream
from V0.4.5's ``venue_pipeline_reporting`` — without changing anything about
the runtime state that produced it.
"""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import pytest

from association_builders import DECIDING, GATING, observation
from atreides_vision.association import TrackObservation
from atreides_vision.errors import OccupancyError
from atreides_vision.occupancy import (
    OCCUPANCY_REPORT_SCHEMA_VERSION,
    JsonlOccupancyReportWriter,
    OccupancyReportMetadata,
    compute_occupancy,
    occupancy_instant_record,
)
from atreides_vision.venue_pipeline import VenueInstantResult, VenuePipeline
from identity_builders import CONFIG

STEP = 0.125
START = 10.0

ONE_PAIR = (("cam_a", 1, 0.0), ("cam_b", 1, 0.5))

SESSION_ID = "session-0001"


def pipeline() -> VenuePipeline:
    return VenuePipeline(gating=GATING, decisions=DECIDING, identity=CONFIG)


def placed(
    seconds: float, placements: tuple[tuple[str, int, float], ...]
) -> list[TrackObservation]:
    return [
        observation(camera_id, track_id, x=x, y=0.0, venue_seconds=seconds)
        for camera_id, track_id, x in placements
    ]


def ticks(
    pipe: VenuePipeline,
    placements: tuple[tuple[str, int, float], ...],
    count: int = 3,
    start: float = START,
) -> list[VenueInstantResult]:
    results = []
    for index in range(count):
        seconds = start + index * STEP
        results.append(pipe.process_instant(seconds, placed(seconds, placements)))
    return results


class TestHeaderRecord:
    def test_metadata_rejects_an_empty_session_id(self) -> None:
        with pytest.raises(OccupancyError, match="session_id"):
            OccupancyReportMetadata(session_id="")

    def test_metadata_rejects_a_whitespace_only_session_id(self) -> None:
        with pytest.raises(OccupancyError, match="session_id"):
            OccupancyReportMetadata(session_id="   ")

    def test_header_record_carries_schema_version_and_session_id(self) -> None:
        header = OccupancyReportMetadata(session_id=SESSION_ID).to_record()

        assert header["record_type"] == "occupancy_report"
        assert header["schema_version"] == OCCUPANCY_REPORT_SCHEMA_VERSION
        assert header["session_id"] == SESSION_ID

    def test_the_writer_writes_the_header_on_construction(self, tmp_path: Path) -> None:
        path = tmp_path / "occupancy.jsonl"

        writer = JsonlOccupancyReportWriter(path, OccupancyReportMetadata(SESSION_ID))
        writer.release()

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        header = json.loads(lines[0])
        assert header["record_type"] == "occupancy_report"

    def test_the_writer_raises_immediately_on_an_unwritable_destination(
        self, tmp_path: Path
    ) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")

        with pytest.raises(OccupancyError):
            JsonlOccupancyReportWriter(
                blocker / "occupancy.jsonl", OccupancyReportMetadata(SESSION_ID)
            )


class TestDuplicateSuppressionIsReported:
    def test_a_confirmed_fusion_is_reported_as_one_suppressed_duplicate(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, ONE_PAIR)[-1]
        snapshot = compute_occupancy(final, pipe.registry)

        record = occupancy_instant_record(snapshot, SESSION_ID)

        assert record["raw_active_track_count"] == 2
        assert record["occupancy_count"] == 1
        assert record["suppressed_duplicate_count"] == 1

    def test_ambiguous_evidence_is_reported_with_no_suppression(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, (("cam_a", 1, 0.0), ("cam_b", 1, 0.5), ("cam_b", 2, 0.75)), count=6)[-1]
        snapshot = compute_occupancy(final, pipe.registry)

        record = occupancy_instant_record(snapshot, SESSION_ID)

        assert record["suppressed_duplicate_count"] == 0
        assert record["occupancy_count"] == record["raw_active_track_count"]


class TestProvenanceSurvivesReporting:
    def test_a_contributing_subject_resolves_back_to_its_camera_tracklets(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, (("cam_a", 1, 0.0), ("cam_b", 1, 0.5), ("cam_c", 1, 1.0)))[-1]
        snapshot = compute_occupancy(final, pipe.registry)

        record = occupancy_instant_record(snapshot, SESSION_ID)

        assert len(record["contributing_subjects"]) == 1
        members = record["contributing_subjects"][0]["members"]
        assert len(members) == 3


class TestDeterminism:
    PLACEMENTS = (
        ("cam_a", 1, 0.0),
        ("cam_b", 1, 0.5),
        ("cam_c", 1, 1.0),
        ("cam_b", 2, 6.0),
    )

    def test_two_independent_runs_produce_byte_identical_records(self) -> None:
        pipe1 = pipeline()
        final1 = ticks(pipe1, self.PLACEMENTS)[-1]
        record1 = occupancy_instant_record(compute_occupancy(final1, pipe1.registry), SESSION_ID)

        pipe2 = pipeline()
        final2 = ticks(pipe2, self.PLACEMENTS)[-1]
        record2 = occupancy_instant_record(compute_occupancy(final2, pipe2.registry), SESSION_ID)

        assert json.dumps(record1, sort_keys=True) == json.dumps(record2, sort_keys=True)

    def test_observation_enumeration_order_does_not_change_the_record(self) -> None:
        expected_pipe = pipeline()
        expected_final = ticks(expected_pipe, self.PLACEMENTS)[-1]
        expected = occupancy_instant_record(
            compute_occupancy(expected_final, expected_pipe.registry), SESSION_ID
        )

        for order in permutations(self.PLACEMENTS):
            pipe = pipeline()
            final = ticks(pipe, order)[-1]
            got = occupancy_instant_record(compute_occupancy(final, pipe.registry), SESSION_ID)
            assert json.dumps(got, sort_keys=True) == json.dumps(expected, sort_keys=True)


class TestSingleCameraDegeneratePath:
    def test_a_single_camera_session_reports_cleanly_with_no_suppression(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, (("cam_a", 1, 0.0), ("cam_a", 2, 5.0)), count=5)[-1]
        snapshot = compute_occupancy(final, pipe.registry)

        record = occupancy_instant_record(snapshot, SESSION_ID)

        assert record["contributing_subjects"] == []
        assert record["suppressed_duplicate_count"] == 0
        assert record["occupancy_count"] == 2


class TestZeroObservationInstant:
    def test_an_empty_instant_produces_a_legal_minimal_record(self) -> None:
        pipe = pipeline()
        result = pipe.process_instant(START, [])
        snapshot = compute_occupancy(result, pipe.registry)

        record = occupancy_instant_record(snapshot, SESSION_ID)

        assert record["record_type"] == "occupancy_instant"
        assert record["raw_active_track_count"] == 0
        assert record["occupancy_count"] == 0
        assert record["contributing_subjects"] == []
        assert record["unresolved_tracklets"] == []


class TestWriterIntegration:
    def test_write_snapshot_appends_exactly_one_line_per_call(self, tmp_path: Path) -> None:
        pipe = pipeline()
        path = tmp_path / "occupancy.jsonl"

        with JsonlOccupancyReportWriter(path, OccupancyReportMetadata(SESSION_ID)) as writer:
            for result in ticks(pipe, ONE_PAIR):
                writer.write_snapshot(compute_occupancy(result, pipe.registry))

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 4  # header + 3 instants
        assert writer.snapshots_written == 3
        for line in lines[1:]:
            record = json.loads(line)
            assert record["record_type"] == "occupancy_instant"

    def test_writing_after_release_raises_and_does_not_touch_the_registry(
        self, tmp_path: Path
    ) -> None:
        pipe = pipeline()
        final = ticks(pipe, ONE_PAIR)[-1]
        snapshot = compute_occupancy(final, pipe.registry)
        writer = JsonlOccupancyReportWriter(
            tmp_path / "occupancy.jsonl", OccupancyReportMetadata(SESSION_ID)
        )
        writer.release()

        subjects_before = pipe.registry.active_subjects()
        with pytest.raises(OccupancyError, match="released"):
            writer.write_snapshot(snapshot)
        assert pipe.registry.active_subjects() == subjects_before

    def test_release_is_idempotent(self, tmp_path: Path) -> None:
        writer = JsonlOccupancyReportWriter(
            tmp_path / "occupancy.jsonl", OccupancyReportMetadata(SESSION_ID)
        )
        writer.release()
        writer.release()

    def test_a_writer_failure_never_raises_the_wrong_error_type(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")

        pipe = pipeline()
        with pytest.raises(OccupancyError):
            JsonlOccupancyReportWriter(
                blocker / "occupancy.jsonl", OccupancyReportMetadata(SESSION_ID)
            )
        # the pipeline itself was never touched; still fully usable afterward
        result = pipe.process_instant(START, placed(START, ONE_PAIR))
        assert result.active_subjects == ()

    def test_gzip_suffix_selects_compression_like_every_other_writer(self, tmp_path: Path) -> None:
        path = tmp_path / "occupancy.jsonl.gz"
        writer = JsonlOccupancyReportWriter(path, OccupancyReportMetadata(SESSION_ID))
        assert writer.is_compressed
        writer.release()
        assert path.exists()


class TestReportingNeverMutatesRuntimeState:
    def test_building_records_repeatedly_does_not_change_registry_state(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, ONE_PAIR)[-1]
        snapshot = compute_occupancy(final, pipe.registry)

        before = pipe.registry.active_subjects()
        occupancy_instant_record(snapshot, SESSION_ID)
        occupancy_instant_record(snapshot, SESSION_ID)
        after = pipe.registry.active_subjects()

        assert before == after

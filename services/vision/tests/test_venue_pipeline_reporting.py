"""V0.4.5: observability for the V0.4.4 association/fusion/identity runtime.

Reuses the exact fixtures and scenarios ``test_venue_pipeline.py`` already
establishes for V0.4.4, and checks only what this milestone adds: that a
``VenueInstantResult`` and the registry that produced it can be turned into a
deterministic, explainable, versioned JSONL record without changing anything
about the runtime state that produced them.
"""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import pytest

from association_builders import DECIDING, GATING, observation
from atreides_vision.association import AssociationOutcome, TrackObservation
from atreides_vision.errors import FusionError
from atreides_vision.fusion import IntegrationOutcome
from atreides_vision.identity import VenueSubjectId
from atreides_vision.venue_pipeline import VenueInstantResult, VenuePipeline
from atreides_vision.venue_pipeline_reporting import (
    VENUE_PIPELINE_REPORT_SCHEMA_VERSION,
    JsonlVenuePipelineReportWriter,
    VenuePipelineReportMetadata,
    instant_record,
)
from identity_builders import CONFIG

STEP = 0.125
START = 10.0

ONE_PAIR = (("cam_a", 1, 0.0), ("cam_b", 1, 0.5))


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


SESSION_ID = "session-0001"


class TestHeaderRecord:
    def test_metadata_rejects_an_empty_session_id(self) -> None:
        with pytest.raises(FusionError, match="session_id"):
            VenuePipelineReportMetadata(session_id="")

    def test_metadata_rejects_a_whitespace_only_session_id(self) -> None:
        with pytest.raises(FusionError, match="session_id"):
            VenuePipelineReportMetadata(session_id="   ")

    def test_header_record_carries_schema_version_and_session_id(self) -> None:
        header = VenuePipelineReportMetadata(session_id=SESSION_ID).to_record()

        assert header["record_type"] == "venue_pipeline_report"
        assert header["schema_version"] == VENUE_PIPELINE_REPORT_SCHEMA_VERSION
        assert header["session_id"] == SESSION_ID

    def test_the_writer_writes_the_header_on_construction(self, tmp_path: Path) -> None:
        path = tmp_path / "report.jsonl"

        writer = JsonlVenuePipelineReportWriter(path, VenuePipelineReportMetadata(SESSION_ID))
        writer.release()

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        header = json.loads(lines[0])
        assert header["record_type"] == "venue_pipeline_report"

    def test_the_writer_raises_immediately_on_an_unwritable_destination(
        self, tmp_path: Path
    ) -> None:
        # a plain file used as a parent directory is not a directory
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")

        with pytest.raises(FusionError):
            JsonlVenuePipelineReportWriter(
                blocker / "report.jsonl", VenuePipelineReportMetadata(SESSION_ID)
            )


class TestAssociatedOutcomeIsReported:
    def test_a_true_association_is_visible_as_associated_and_applied(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, ONE_PAIR)[-1]

        record = instant_record(final, pipe.registry, SESSION_ID)

        outcomes = {d["outcome"] for d in record["association"]["decisions"]}
        assert AssociationOutcome.ASSOCIATED.value in outcomes
        integration_outcomes = {o["outcome"] for o in record["integration"]["outcomes"]}
        assert IntegrationOutcome.APPLIED.value in integration_outcomes
        assert len(record["active_subjects"]) == 1


class TestUnassociatedOutcomeIsReported:
    def test_a_genuine_non_match_is_refused_association_and_ignored_by_fusion(self) -> None:
        pipe = pipeline()
        results = ticks(pipe, (("cam_a", 1, 0.0), ("cam_b", 1, 9.0)), count=6)
        final = results[-1]

        record = instant_record(final, pipe.registry, SESSION_ID)

        outcomes = {d["outcome"] for d in record["association"]["decisions"]}
        assert outcomes == {AssociationOutcome.UNASSOCIATED.value}
        integration_outcomes = {o["outcome"] for o in record["integration"]["outcomes"]}
        assert integration_outcomes == {IntegrationOutcome.IGNORED.value}
        assert record["active_subjects"] == []


class TestAmbiguousOutcomeIsReported:
    def test_an_ambiguous_case_is_left_unmerged_and_reported_as_such(self) -> None:
        pipe = pipeline()
        results = ticks(pipe, (("cam_a", 1, 0.0), ("cam_b", 1, 0.5), ("cam_b", 2, 0.75)), count=6)
        final = results[-1]

        record = instant_record(final, pipe.registry, SESSION_ID)

        outcomes = {d["outcome"] for d in record["association"]["decisions"]}
        assert AssociationOutcome.AMBIGUOUS.value in outcomes
        integration_outcomes = {o["outcome"] for o in record["integration"]["outcomes"]}
        assert integration_outcomes == {IntegrationOutcome.IGNORED.value}
        assert record["active_subjects"] == []


class TestNotComparableOutcomeIsReported:
    def test_a_single_camera_pair_is_reported_not_comparable_and_ignored(self) -> None:
        pipe = pipeline()
        results = ticks(pipe, (("cam_a", 1, 0.0), ("cam_a", 2, 5.0)), count=4)
        final = results[-1]

        record = instant_record(final, pipe.registry, SESSION_ID)

        outcomes = {d["outcome"] for d in record["association"]["decisions"]}
        assert outcomes == {AssociationOutcome.NOT_COMPARABLE.value}
        integration_outcomes = {o["outcome"] for o in record["integration"]["outcomes"]}
        assert integration_outcomes == {IntegrationOutcome.IGNORED.value}


class TestRefusedFusionOutcomeIsReported:
    def test_a_refused_cross_subject_conflict_is_reported_with_both_subject_ids(self) -> None:
        pipe = pipeline()
        first = ticks(pipe, ONE_PAIR)[-1]
        subject_one = first.active_subjects[0]

        second = ticks(
            pipe,
            (("cam_a", 2, 0.0), ("cam_c", 1, 0.5)),
            count=3,
            start=START + 3 * STEP,
        )[-1]
        subject_two = next(s for s in second.active_subjects if s != subject_one)

        result = None
        for index in range(6, 10):
            seconds = START + index * STEP
            result = pipe.process_instant(
                seconds, placed(seconds, (("cam_b", 1, 0.0), ("cam_c", 1, 0.5)))
            )
        assert result is not None

        record = instant_record(result, pipe.registry, SESSION_ID)
        refused = [o for o in record["integration"]["outcomes"] if o["outcome"] == "refused"]
        assert refused
        assert {str(subject_one), str(subject_two)} == set(refused[0]["subject_ids"])


class TestProvenanceSurvivesFusion:
    def test_a_venue_subject_resolves_back_to_its_contributing_camera_tracks(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, (("cam_a", 1, 0.0), ("cam_b", 1, 0.5), ("cam_c", 1, 1.0)))[-1]

        record = instant_record(final, pipe.registry, SESSION_ID)

        assert len(record["active_subjects"]) == 1
        subject = record["active_subjects"][0]
        cameras = {m["camera_id"] for m in subject["members"]}
        assert cameras == {"cam_a", "cam_b", "cam_c"}
        for member in subject["members"]:
            assert "track_id" in member
            assert "stream_epoch" in member
            assert "joined_at_seconds" in member

    def test_the_pair_key_in_a_membership_outcome_names_real_camera_tracklets(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, ONE_PAIR)[-1]

        record = instant_record(final, pipe.registry, SESSION_ID)

        applied = [o for o in record["integration"]["outcomes"] if o["outcome"] == "applied"]
        assert applied
        pair = applied[0]["pair"]
        cameras = {pair["first"]["camera_id"], pair["second"]["camera_id"]}
        assert cameras == {"cam_a", "cam_b"}


class TestDuplicateCountingIsNotProduced:
    def test_active_subjects_never_repeats_a_venue_subject_id(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, (("cam_a", 1, 0.0), ("cam_b", 1, 0.5), ("cam_c", 1, 1.0)))[-1]

        record = instant_record(final, pipe.registry, SESSION_ID)

        ids = [s["venue_subject_id"] for s in record["active_subjects"]]
        assert len(ids) == len(set(ids))

    def test_a_no_op_re_observation_does_not_grow_the_subject_count(self) -> None:
        pipe = pipeline()
        results = ticks(pipe, ONE_PAIR, count=6)

        first_count = len(instant_record(results[2], pipe.registry, SESSION_ID)["active_subjects"])
        last_count = len(instant_record(results[-1], pipe.registry, SESSION_ID)["active_subjects"])
        assert first_count == last_count == 1


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
        record1 = instant_record(final1, pipe1.registry, SESSION_ID)

        pipe2 = pipeline()
        final2 = ticks(pipe2, self.PLACEMENTS)[-1]
        record2 = instant_record(final2, pipe2.registry, SESSION_ID)

        assert json.dumps(record1, sort_keys=True) == json.dumps(record2, sort_keys=True)

    def test_observation_enumeration_order_does_not_change_the_record(self) -> None:
        expected_pipe = pipeline()
        expected = instant_record(
            ticks(expected_pipe, self.PLACEMENTS)[-1], expected_pipe.registry, SESSION_ID
        )

        for order in permutations(self.PLACEMENTS):
            pipe = pipeline()
            got = instant_record(ticks(pipe, order)[-1], pipe.registry, SESSION_ID)
            assert json.dumps(got, sort_keys=True) == json.dumps(expected, sort_keys=True)


class TestSingleCameraDegeneratePath:
    def test_a_single_camera_session_reports_cleanly_with_no_subjects(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, (("cam_a", 1, 0.0), ("cam_a", 2, 5.0)), count=5)[-1]

        record = instant_record(final, pipe.registry, SESSION_ID)

        assert record["active_subjects"] == []
        outcomes = {d["outcome"] for d in record["association"]["decisions"]}
        assert outcomes == {AssociationOutcome.NOT_COMPARABLE.value}


class TestZeroObservationInstant:
    def test_an_empty_instant_produces_a_legal_minimal_record(self) -> None:
        pipe = pipeline()
        result = pipe.process_instant(START, [])

        record = instant_record(result, pipe.registry, SESSION_ID)

        assert record["association"]["decisions"] == []
        assert record["integration"]["outcomes"] == []
        assert record["active_subjects"] == []
        assert record["record_type"] == "venue_instant"


class TestWriterIntegration:
    def test_write_instant_appends_exactly_one_line_per_call(self, tmp_path: Path) -> None:
        pipe = pipeline()
        path = tmp_path / "report.jsonl"

        with JsonlVenuePipelineReportWriter(
            path, VenuePipelineReportMetadata(SESSION_ID)
        ) as writer:
            for result in ticks(pipe, ONE_PAIR):
                writer.write_instant(result, pipe.registry)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 4  # header + 3 instants
        assert writer.instants_written == 3
        for line in lines[1:]:
            record = json.loads(line)
            assert record["record_type"] == "venue_instant"

    def test_writing_after_release_raises_and_does_not_touch_the_pipeline(
        self, tmp_path: Path
    ) -> None:
        pipe = pipeline()
        result = ticks(pipe, ONE_PAIR)[-1]
        writer = JsonlVenuePipelineReportWriter(
            tmp_path / "report.jsonl", VenuePipelineReportMetadata(SESSION_ID)
        )
        writer.release()

        subjects_before = pipe.registry.active_subjects()
        with pytest.raises(FusionError, match="released"):
            writer.write_instant(result, pipe.registry)
        assert pipe.registry.active_subjects() == subjects_before

    def test_release_is_idempotent(self, tmp_path: Path) -> None:
        writer = JsonlVenuePipelineReportWriter(
            tmp_path / "report.jsonl", VenuePipelineReportMetadata(SESSION_ID)
        )
        writer.release()
        writer.release()

    def test_a_writer_failure_never_raises_an_association_or_identity_error(
        self, tmp_path: Path
    ) -> None:
        # An unwritable destination fails at construction with FusionError,
        # distinctly typed from AssociationError/IdentityError/FusionError's
        # sibling domain errors, and before any pipeline instant is touched.
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")

        pipe = pipeline()
        with pytest.raises(FusionError):
            JsonlVenuePipelineReportWriter(
                blocker / "report.jsonl", VenuePipelineReportMetadata(SESSION_ID)
            )
        # the pipeline itself was never invoked; nothing to assert changed,
        # but confirm it is still fully usable afterward
        result = pipe.process_instant(START, placed(START, ONE_PAIR))
        assert result.active_subjects == ()

    def test_gzip_suffix_selects_compression_like_every_other_writer(self, tmp_path: Path) -> None:
        path = tmp_path / "report.jsonl.gz"
        writer = JsonlVenuePipelineReportWriter(path, VenuePipelineReportMetadata(SESSION_ID))
        assert writer.is_compressed
        writer.release()
        assert path.exists()


class TestReportingNeverMutatesRuntimeState:
    def test_building_records_repeatedly_does_not_change_registry_or_engine_state(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, ONE_PAIR)[-1]

        before = pipe.registry.active_subjects()
        instant_record(final, pipe.registry, SESSION_ID)
        instant_record(final, pipe.registry, SESSION_ID)
        after = pipe.registry.active_subjects()

        assert before == after

    def test_venue_subject_id_in_a_record_is_the_opaque_string_form_not_a_track_id(self) -> None:
        pipe = pipeline()
        final = ticks(pipe, ONE_PAIR)[-1]
        subject_id = final.active_subjects[0]
        assert isinstance(subject_id, VenueSubjectId)

        record = instant_record(final, pipe.registry, SESSION_ID)
        recorded_id = record["active_subjects"][0]["venue_subject_id"]
        assert recorded_id == str(subject_id)
        assert recorded_id != subject_id.sequence

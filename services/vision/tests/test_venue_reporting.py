"""The venue session record: shape, provenance, determinism, and what it omits."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from atreides_vision.errors import VenueSessionError
from atreides_vision.timing import (
    FrameRateSource,
    StreamRef,
    TimeAlignmentBasis,
    TimeUncertainty,
)
from atreides_vision.venue.models import StreamAbsence, StreamActivity, VenueTick
from atreides_vision.venue.reporting import (
    VENUE_SESSION_SCHEMA_VERSION,
    JsonlVenueSessionWriter,
    VenueSessionMetadata,
    VenueSessionSummary,
    VenueStreamProvenance,
    tick_record,
)
from venue_fakes import stream_frames

CAM_A = StreamRef("cam_a", 0)
CAM_B = StreamRef("cam_b", 0)


def provenance(
    stream: StreamRef,
    fps: float = 25.0,
    fps_source: FrameRateSource = FrameRateSource.CONTAINER,
    offset_seconds: float = 0.0,
    timing: TimeUncertainty | None = None,
) -> VenueStreamProvenance:
    return VenueStreamProvenance(
        stream=stream,
        input_path=f"/clips/{stream.camera_id}.mp4",
        width=640,
        height=360,
        fps=fps,
        fps_source=fps_source,
        offset_seconds=offset_seconds,
        timing=TimeUncertainty.unknown() if timing is None else timing,
        frame_count=100,
    )


def metadata(
    *streams: VenueStreamProvenance, session_id: str = "session_a"
) -> VenueSessionMetadata:
    return VenueSessionMetadata(
        session_id=session_id, streams=streams, manifest_path="/config/venue.json"
    )


#: The session-level figure a tick record repeats. Distinct from any tick's own
#: contributor-scoped alignment, which is what these tests hold it against.
SESSION_ALIGNMENT = TimeUncertainty.declared(2.5)


def a_tick(sequence: int = 0) -> VenueTick:
    return VenueTick(
        venue_micros=0,
        sequence=sequence,
        contributions=tuple(stream_frames(CAM_A, 1, fps=25.0)),
        absent=(StreamAbsence(CAM_B, StreamActivity.NOT_YET_STARTED),),
    )


def read_records(path: Path) -> list[dict[str, Any]]:
    text = (
        gzip.decompress(path.read_bytes()).decode("utf-8")
        if path.suffix == ".gz"
        else path.read_text(encoding="utf-8")
    )
    return [json.loads(line) for line in text.splitlines()]


class TestHeaderRecord:
    def test_carries_the_schema_version_and_session(self) -> None:
        record = metadata(provenance(CAM_A)).to_record()

        assert record["record_type"] == "venue_session"
        assert record["schema_version"] == VENUE_SESSION_SCHEMA_VERSION
        assert record["session_id"] == "session_a"
        assert record["manifest_path"] == "/config/venue.json"
        assert record["camera_count"] == 1
        assert record["stream_count"] == 1

    def test_states_that_no_association_was_performed(self) -> None:
        record = metadata(provenance(CAM_A)).to_record()

        assert record["identity_scope"] == "camera_local_streams"
        assert record["association"] == "not_performed"
        assert record["venue_local_identity"] == "none"

    def test_venue_time_is_session_relative_and_event_driven(self) -> None:
        record = metadata(provenance(CAM_A)).to_record()["venue_time"]

        assert record == {
            "origin": "session_relative",
            "unit": "s",
            "ordering_key": "venue_micros",
            "tick_schedule": "source_event_driven",
        }

    def test_each_stream_records_where_its_frame_rate_came_from(self) -> None:
        record = metadata(
            provenance(CAM_A, fps=25.0, fps_source=FrameRateSource.DECLARED),
            provenance(CAM_B, fps=25.0, fps_source=FrameRateSource.ASSUMED),
        ).to_record()

        declared, assumed = record["streams"]
        assert (declared["fps_source"], declared["fps_is_trustworthy"]) == ("declared", True)
        assert (assumed["fps_source"], assumed["fps_is_trustworthy"]) == ("assumed", False)

    def test_stream_records_are_canonical_whatever_order_they_arrive_in(self) -> None:
        forward = metadata(provenance(CAM_A), provenance(CAM_B)).to_record()
        reversed_ = metadata(provenance(CAM_B), provenance(CAM_A)).to_record()

        assert forward == reversed_
        assert [entry["stream_ref"] for entry in forward["streams"]] == ["cam_a#0", "cam_b#0"]

    def test_alignment_is_recorded_without_a_verdict(self) -> None:
        unknown, declared = metadata(
            provenance(CAM_A),
            provenance(CAM_B, offset_seconds=5.0, timing=TimeUncertainty.declared(0.25)),
        ).to_record()["streams"]

        assert unknown["time_alignment"] == {
            "basis": "not_established",
            "half_width_seconds": None,
        }
        assert declared["time_alignment"] == {
            "basis": "operator_declared",
            "half_width_seconds": 0.25,
        }
        assert "aligned" not in json.dumps(declared)

    def test_each_stream_records_the_frame_count_its_container_claims(self) -> None:
        """Provenance, never a schedule input: the merge reads to exhaustion.

        Recorded so a reader can see a container claiming more frames than it
        delivered, which the summary's ``frames_contributed`` will contradict.
        """
        entry = metadata(provenance(CAM_A)).to_record()["streams"][0]

        assert entry["frame_count"] == 100

    def test_an_unknown_frame_count_is_recorded_as_null(self) -> None:
        unknown = VenueStreamProvenance(
            stream=CAM_A,
            input_path="/clips/a.mp4",
            width=640,
            height=360,
            fps=25.0,
            fps_source=FrameRateSource.CONTAINER,
            offset_seconds=0.0,
            timing=TimeUncertainty.unknown(),
        )

        assert metadata(unknown).to_record()["streams"][0]["frame_count"] is None

    def test_the_session_alignment_is_the_worst_across_every_stream(self) -> None:
        """Not the worst across contributors — every registered stream counts."""
        record = metadata(
            provenance(CAM_A, timing=TimeUncertainty.declared(0.25)),
            provenance(CAM_B),
        ).to_record()

        assert record["session_time_alignment"] == {
            "basis": "not_established",
            "half_width_seconds": None,
        }

    def test_the_session_alignment_widens_to_the_worst_declared(self) -> None:
        record = metadata(
            provenance(CAM_A, timing=TimeUncertainty.declared(0.25)),
            provenance(CAM_B, timing=TimeUncertainty.declared(2.0)),
        ).to_record()

        assert record["session_time_alignment"] == {
            "basis": "operator_declared",
            "half_width_seconds": 2.0,
        }

    def test_a_sub_microsecond_half_width_stays_readable_back(self) -> None:
        """The writer must never emit a record its own reader would refuse.

        Rounding 1e-9 to six decimal places would write ``0.0`` beside an
        ``operator_declared`` basis, which ``TimeUncertainty`` rejects. It is
        widened to the narrowest value the format can express — a weaker claim,
        never a stronger one.
        """
        record = metadata(provenance(CAM_A, timing=TimeUncertainty.declared(1e-9))).to_record()
        written = record["streams"][0]["time_alignment"]

        assert written["half_width_seconds"] == 1e-6
        parsed = TimeUncertainty(
            basis=TimeAlignmentBasis(written["basis"]),
            half_width_seconds=written["half_width_seconds"],
        )
        assert parsed.half_width_seconds is not None
        assert parsed.half_width_seconds >= 1e-9

    def test_a_header_needs_a_session_and_a_stream(self) -> None:
        with pytest.raises(VenueSessionError, match="non-empty session_id"):
            VenueSessionMetadata(session_id=" ", streams=(provenance(CAM_A),))
        with pytest.raises(VenueSessionError, match="at least one stream"):
            VenueSessionMetadata(session_id="session_a", streams=())


class TestTickRecord:
    def test_contributions_carry_both_times_and_their_frame_index(self) -> None:
        record = tick_record(a_tick(), "session_a", SESSION_ALIGNMENT)

        assert record["record_type"] == "venue_tick"
        assert record["schema_version"] == VENUE_SESSION_SCHEMA_VERSION
        assert record["session_id"] == "session_a"
        assert record["sequence"] == 0
        assert record["venue_micros"] == 0
        assert record["contributions"] == [
            {
                "camera_id": "cam_a",
                "stream_epoch": 0,
                "frame_index": 1,
                "source_seconds": 0.0,
                "venue_seconds": 0.0,
                "time_alignment": {"basis": "not_established", "half_width_seconds": None},
            }
        ]

    def test_absent_streams_carry_their_reason(self) -> None:
        record = tick_record(a_tick(), "session_a", SESSION_ALIGNMENT)

        assert record["absent"] == [
            {"camera_id": "cam_b", "stream_epoch": 0, "activity": "not_yet_started"}
        ]

    def test_every_line_carries_the_session_alignment_as_well_as_its_own(self) -> None:
        """A tick read alone must not look better aligned than its session.

        ``time_alignment`` covers contributors, so it improves whenever the
        worst camera is between frames or ended. The session figure does not
        move, which is what makes the line interpretable by itself.
        """
        record = tick_record(a_tick(), "session_a", TimeUncertainty.unknown())

        assert record["time_alignment"] == {
            "basis": "not_established",
            "half_width_seconds": None,
        }
        assert record["session_time_alignment"] == {
            "basis": "not_established",
            "half_width_seconds": None,
        }

    def test_a_well_aligned_tick_still_reports_an_unaligned_session(self) -> None:
        aligned = stream_frames(CAM_A, 1, fps=25.0, timing=TimeUncertainty.declared(0.25))
        tick = VenueTick(
            venue_micros=0,
            sequence=7,
            contributions=tuple(aligned),
            absent=(StreamAbsence(CAM_B, StreamActivity.BETWEEN_FRAMES),),
        )

        record = tick_record(tick, "session_a", TimeUncertainty.unknown())

        assert record["time_alignment"]["basis"] == "operator_declared"
        assert record["session_time_alignment"]["basis"] == "not_established"

    def test_absent_is_written_even_when_empty(self) -> None:
        tick = VenueTick(
            venue_micros=0, sequence=3, contributions=tuple(stream_frames(CAM_A, 1, fps=25.0))
        )

        assert tick_record(tick, "session_a", SESSION_ALIGNMENT)["absent"] == []

    def test_venue_seconds_keeps_microsecond_precision(self) -> None:
        frames = stream_frames(CAM_A, 3, fps=30.0)
        tick = VenueTick(
            venue_micros=frames[2].venue_micros, sequence=2, contributions=(frames[2],)
        )

        record = tick_record(tick, "session_a", SESSION_ALIGNMENT)

        assert record["venue_micros"] == 66_667
        assert record["contributions"][0]["source_seconds"] == round(2 / 30.0, 6)


class TestSummaryRecord:
    def test_reports_frames_per_stream_and_the_span(self) -> None:
        record = VenueSessionSummary(
            session_id="session_a",
            tick_count=11,
            frames_by_stream=((CAM_B, 6), (CAM_A, 6)),
            first_venue_micros=0,
            last_venue_micros=200_000,
        ).to_record()

        assert record["record_type"] == "venue_session_summary"
        assert record["tick_count"] == 11
        assert record["venue_span_seconds"] == 0.2
        assert [entry["camera_id"] for entry in record["streams"]] == ["cam_a", "cam_b"]
        assert all(entry["frames_contributed"] == 6 for entry in record["streams"])

    def test_an_empty_session_reports_no_span(self) -> None:
        record = VenueSessionSummary(
            session_id="session_a", tick_count=0, frames_by_stream=((CAM_A, 0),)
        ).to_record()

        assert record["tick_count"] == 0
        assert record["venue_span_seconds"] is None
        assert record["first_venue_micros"] is None

    def test_a_negative_tick_count_is_refused(self) -> None:
        with pytest.raises(VenueSessionError, match="non-negative"):
            VenueSessionSummary(session_id="s", tick_count=-1, frames_by_stream=())


class TestWriter:
    def test_writes_a_header_then_ticks_then_a_summary(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        header = metadata(provenance(CAM_A), provenance(CAM_B))
        with JsonlVenueSessionWriter(path, header) as writer:
            writer.write_tick(a_tick(0))
            writer.write_tick(a_tick(1))
            writer.write_summary(
                VenueSessionSummary(
                    session_id="session_a", tick_count=2, frames_by_stream=((CAM_A, 2), (CAM_B, 0))
                )
            )
            assert writer.ticks_written == 2

        records = read_records(path)
        assert [record["record_type"] for record in records] == [
            "venue_session",
            "venue_tick",
            "venue_tick",
            "venue_session_summary",
        ]
        assert all(record["schema_version"] == VENUE_SESSION_SCHEMA_VERSION for record in records)
        assert all(record["session_id"] == "session_a" for record in records)

    def test_a_gz_suffix_compresses(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl.gz"
        with JsonlVenueSessionWriter(path, metadata(provenance(CAM_A))) as writer:
            assert writer.is_compressed
            writer.write_tick(a_tick())

        assert len(read_records(path)) == 2

    def test_writing_after_release_is_refused(self, tmp_path: Path) -> None:
        writer = JsonlVenueSessionWriter(tmp_path / "s.jsonl", metadata(provenance(CAM_A)))
        writer.release()
        writer.release()  # idempotent

        with pytest.raises(VenueSessionError, match="released"):
            writer.write_tick(a_tick())

    def test_a_tick_after_the_summary_is_refused(self, tmp_path: Path) -> None:
        summary = VenueSessionSummary(
            session_id="session_a", tick_count=0, frames_by_stream=((CAM_A, 0),)
        )
        with JsonlVenueSessionWriter(tmp_path / "s.jsonl", metadata(provenance(CAM_A))) as writer:
            writer.write_summary(summary)
            with pytest.raises(VenueSessionError, match="already been written"):
                writer.write_tick(a_tick())

    def test_two_summaries_are_refused(self, tmp_path: Path) -> None:
        summary = VenueSessionSummary(
            session_id="session_a", tick_count=0, frames_by_stream=((CAM_A, 0),)
        )
        with JsonlVenueSessionWriter(tmp_path / "s.jsonl", metadata(provenance(CAM_A))) as writer:
            writer.write_summary(summary)
            with pytest.raises(VenueSessionError, match="already has a summary"):
                writer.write_summary(summary)

    def test_a_summary_for_another_session_is_refused(self, tmp_path: Path) -> None:
        with (
            JsonlVenueSessionWriter(tmp_path / "s.jsonl", metadata(provenance(CAM_A))) as writer,
            pytest.raises(VenueSessionError, match="describes"),
        ):
            writer.write_summary(
                VenueSessionSummary(
                    session_id="other", tick_count=0, frames_by_stream=((CAM_A, 0),)
                )
            )

    def test_the_same_session_written_twice_is_byte_identical(self, tmp_path: Path) -> None:
        def write(name: str, reverse: bool) -> bytes:
            entries = [provenance(CAM_A), provenance(CAM_B)]
            if reverse:
                entries.reverse()
            path = tmp_path / name
            with JsonlVenueSessionWriter(path, metadata(*entries)) as writer:
                writer.write_tick(a_tick(0))
                writer.write_summary(
                    VenueSessionSummary(
                        session_id="session_a",
                        tick_count=1,
                        frames_by_stream=(
                            ((CAM_B, 0), (CAM_A, 1)) if reverse else ((CAM_A, 1), (CAM_B, 0))
                        ),
                        first_venue_micros=0,
                        last_venue_micros=0,
                    )
                )
            return path.read_bytes()

        assert write("forward.jsonl", reverse=False) == write("reversed.jsonl", reverse=True)

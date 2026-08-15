"""Behaviour output volume control: emission sampling, cadence and compression.

The load-bearing property is that **sampling never touches analysis**. The
analyzer ingests every processed frame regardless of what is written, so speed
and acceleration come from the complete history whether the file holds every
frame or one per second.
"""

from __future__ import annotations

import gzip
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
    BehaviorRunMetadata,
    EmitEveryFrame,
    IntervalEmissionPolicy,
    JsonlBehaviorWriter,
    SourceProvenance,
)
from atreides_vision.errors import BehaviorOutputError
from atreides_vision.sampling import IntervalSampler

NORMALIZED = MotionSpace.NORMALIZED_IMAGE
FPS = 25.0
DT = 1.0 / FPS


def metadata() -> BehaviorRunMetadata:
    return BehaviorRunMetadata(
        camera_id="cam_01",
        space=NORMALIZED,
        source=SourceProvenance(path="in.mp4", width=640, height=360, fps=FPS),
    )


def frame(index: int, track_ids: tuple[int, ...] = (1,)) -> BehaviorFrame:
    timestamp = (index - 1) * DT
    return BehaviorFrame.from_snapshots(
        index,
        timestamp,
        [
            BehaviorSubjectSnapshot(
                subject=SubjectRef("cam_01", track_id),
                frame_index=index,
                timestamp_seconds=timestamp,
                position=MotionPoint(0.1 * track_id, 0.2, NORMALIZED),
                signals=(
                    BehaviorSignal(
                        signal_type=SignalType.SPEED,
                        subject=SubjectRef("cam_01", track_id),
                        frame_index=index,
                        timestamp_seconds=timestamp,
                        value=1.0,
                        space=NORMALIZED,
                    ),
                ),
            )
            for track_id in track_ids
        ],
    )


def read(path: Path) -> list[dict[str, object]]:
    """Read a JSONL file, transparently handling the gzip variant."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def observations(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [record for record in records if record["record_type"] == "observation"]


class TestIntervalSampler:
    def test_the_first_moment_is_always_due(self) -> None:
        assert IntervalSampler(1.0).should_sample(0.0)

    def test_it_waits_the_interval(self) -> None:
        sampler = IntervalSampler(1.0)
        assert sampler.should_sample(0.0)
        assert not sampler.should_sample(0.5)
        assert sampler.should_sample(1.0)

    def test_a_backwards_timestamp_is_not_due(self) -> None:
        sampler = IntervalSampler(1.0)
        sampler.should_sample(10.0)
        assert not sampler.should_sample(2.0)

    def test_reset_makes_the_next_moment_due(self) -> None:
        sampler = IntervalSampler(5.0)
        sampler.should_sample(0.0)
        sampler.reset()
        assert sampler.should_sample(0.1)

    def test_rejects_a_non_positive_interval(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            IntervalSampler(0.0)


class TestEmissionPolicies:
    def test_default_emits_everything(self) -> None:
        policy = EmitEveryFrame()
        assert all(policy.should_emit(index, index * DT) for index in range(50))

    def test_interval_emits_on_schedule(self) -> None:
        policy = IntervalEmissionPolicy(0.2)
        emitted = [index for index in range(50) if policy.should_emit(index, index * DT)]
        # 0.2 s at 25 FPS is every fifth frame, starting at the first.
        assert emitted == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

    def test_interval_is_measured_in_source_time_not_frames(self) -> None:
        """The same setting means the same thing at any frame rate.

        Not that the timestamps match exactly — a 0.5 s deadline falls between
        frames at 25 FPS (0.48 and 0.52) but lands on one at 50 FPS. The
        guarantee is that both rates produce the same number of samples, each
        within a frame of the ideal schedule. A frame-count rule would instead
        have emitted twice as often at 50 FPS.
        """

        def emitted_timestamps(fps: float) -> list[float]:
            policy = IntervalEmissionPolicy(0.5)
            return [
                index / fps
                for index in range(int(fps * 2))
                if policy.should_emit(index, index / fps)
            ]

        slow = emitted_timestamps(25.0)
        fast = emitted_timestamps(50.0)
        assert len(slow) == len(fast) == 4

        for index, (slow_time, fast_time) in enumerate(zip(slow, fast, strict=True)):
            ideal = index * 0.5
            assert abs(slow_time - ideal) <= 1 / 25.0
            assert abs(fast_time - ideal) <= 1 / 50.0

    def test_rejects_a_non_positive_interval(self) -> None:
        with pytest.raises(BehaviorOutputError, match="must be positive"):
            IntervalEmissionPolicy(0.0)

    def test_exposes_its_interval(self) -> None:
        assert IntervalEmissionPolicy(0.25).interval_seconds == 0.25


class TestWriterSampling:
    def test_default_writes_every_frame(self, tmp_path: Path) -> None:
        """V0.3.2 behaviour must be unchanged unless an interval is configured."""
        path = tmp_path / "b.jsonl"
        with JsonlBehaviorWriter(path, metadata()) as writer:
            for index in range(1, 26):
                writer.write_frame(frame(index))
        assert len(observations(read(path))) == 25

    def test_an_interval_reduces_what_is_written(self, tmp_path: Path) -> None:
        path = tmp_path / "b.jsonl"
        with JsonlBehaviorWriter(
            path, metadata(), emission_policy=IntervalEmissionPolicy(0.2)
        ) as writer:
            for index in range(1, 51):
                writer.write_frame(frame(index))

        records = observations(read(path))
        assert len(records) == 10
        expected = [1, 6, 11, 16, 21, 26, 31, 36, 41, 46]
        assert [record["frame_index"] for record in records] == expected

    def test_every_frame_is_still_offered_to_the_writer(self, tmp_path: Path) -> None:
        """The count the analyzer saw, versus the count written."""
        with JsonlBehaviorWriter(
            tmp_path / "b.jsonl", metadata(), emission_policy=IntervalEmissionPolicy(0.4)
        ) as writer:
            for index in range(1, 51):
                writer.write_frame(frame(index))
            assert writer.frames_analysed == 50
            assert writer.frames_written == 5

    def test_the_first_frame_is_always_emitted(self, tmp_path: Path) -> None:
        """A run shorter than one interval must still produce observations."""
        path = tmp_path / "b.jsonl"
        with JsonlBehaviorWriter(
            path, metadata(), emission_policy=IntervalEmissionPolicy(60.0)
        ) as writer:
            for index in range(1, 6):
                writer.write_frame(frame(index))
        assert len(observations(read(path))) == 1

    def test_metadata_is_never_suppressed(self, tmp_path: Path) -> None:
        path = tmp_path / "b.jsonl"
        with JsonlBehaviorWriter(
            path, metadata(), emission_policy=IntervalEmissionPolicy(1000.0)
        ) as writer:
            writer.write_frame(frame(1))
        records = read(path)
        assert records[0]["record_type"] == "run_metadata"

    def test_all_subjects_of_an_emitted_frame_are_written(self, tmp_path: Path) -> None:
        """Sampling is per frame, so a frame is never half-written."""
        path = tmp_path / "b.jsonl"
        with JsonlBehaviorWriter(
            path, metadata(), emission_policy=IntervalEmissionPolicy(0.2)
        ) as writer:
            for index in range(1, 11):
                writer.write_frame(frame(index, track_ids=(1, 2, 3)))

        records = observations(read(path))
        assert len(records) == 6  # two emitted frames x three subjects
        assert [record["subject"]["track_id"] for record in records[:3]] == [1, 2, 3]  # type: ignore[index]

    def test_sampling_is_deterministic(self, tmp_path: Path) -> None:
        def run(name: str) -> str:
            path = tmp_path / name
            with JsonlBehaviorWriter(
                path, metadata(), emission_policy=IntervalEmissionPolicy(0.12)
            ) as writer:
                for index in range(1, 41):
                    writer.write_frame(frame(index))
            return path.read_text(encoding="utf-8")

        assert run("a.jsonl") == run("b.jsonl")

    def test_output_shrinks_measurably(self, tmp_path: Path) -> None:
        def size(name: str, policy: object) -> int:
            path = tmp_path / name
            with JsonlBehaviorWriter(path, metadata(), emission_policy=policy) as writer:  # type: ignore[arg-type]
                for index in range(1, 101):
                    writer.write_frame(frame(index))
            return path.stat().st_size

        full = size("full.jsonl", EmitEveryFrame())
        sampled = size("sampled.jsonl", IntervalEmissionPolicy(0.4))
        assert sampled < full / 5


class TestCompression:
    def test_a_gz_suffix_compresses_automatically(self, tmp_path: Path) -> None:
        path = tmp_path / "b.jsonl.gz"
        with JsonlBehaviorWriter(path, metadata()) as writer:
            assert writer.is_compressed
            for index in range(1, 21):
                writer.write_frame(frame(index))

        assert path.read_bytes()[:2] == b"\x1f\x8b", "not a gzip stream"
        assert len(observations(read(path))) == 20

    def test_plain_suffix_stays_uncompressed(self, tmp_path: Path) -> None:
        path = tmp_path / "b.jsonl"
        with JsonlBehaviorWriter(path, metadata()) as writer:
            assert not writer.is_compressed
            writer.write_frame(frame(1))
        assert path.read_bytes()[:2] != b"\x1f\x8b"

    def test_compression_can_be_forced(self, tmp_path: Path) -> None:
        path = tmp_path / "b.jsonl"
        with JsonlBehaviorWriter(path, metadata(), compress=True) as writer:
            writer.write_frame(frame(1))
        assert path.read_bytes()[:2] == b"\x1f\x8b"

    def test_compression_can_be_declined_despite_the_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "b.jsonl.gz"
        with JsonlBehaviorWriter(path, metadata(), compress=False) as writer:
            writer.write_frame(frame(1))
        assert path.read_bytes()[:2] != b"\x1f\x8b"

    def test_content_is_identical_to_the_uncompressed_form(self, tmp_path: Path) -> None:
        def write(name: str) -> list[dict[str, object]]:
            path = tmp_path / name
            with JsonlBehaviorWriter(path, metadata()) as writer:
                for index in range(1, 16):
                    writer.write_frame(frame(index, track_ids=(1, 2)))
            return read(path)

        assert write("plain.jsonl") == write("compressed.jsonl.gz")

    def test_it_actually_saves_space(self, tmp_path: Path) -> None:
        def size(name: str) -> int:
            path = tmp_path / name
            with JsonlBehaviorWriter(path, metadata()) as writer:
                for index in range(1, 201):
                    writer.write_frame(frame(index, track_ids=(1, 2, 3)))
            return path.stat().st_size

        assert size("b.jsonl.gz") < size("b.jsonl") / 3

    def test_the_stream_is_valid_after_a_mid_run_failure(self, tmp_path: Path) -> None:
        """An interrupted run must still leave a readable gzip file."""
        path = tmp_path / "b.jsonl.gz"
        with pytest.raises(RuntimeError), JsonlBehaviorWriter(path, metadata()) as writer:
            writer.write_frame(frame(1))
            raise RuntimeError("processing blew up")

        records = read(path)
        assert records[0]["record_type"] == "run_metadata"
        assert len(observations(records)) == 1

    def test_compression_combines_with_sampling(self, tmp_path: Path) -> None:
        path = tmp_path / "b.jsonl.gz"
        with JsonlBehaviorWriter(
            path, metadata(), emission_policy=IntervalEmissionPolicy(0.2)
        ) as writer:
            for index in range(1, 51):
                writer.write_frame(frame(index))
        assert len(observations(read(path))) == 10

    def test_writing_after_release_still_fails(self, tmp_path: Path) -> None:
        writer = JsonlBehaviorWriter(tmp_path / "b.jsonl.gz", metadata())
        writer.release()
        with pytest.raises(BehaviorOutputError, match="released"):
            writer.write_frame(frame(1))

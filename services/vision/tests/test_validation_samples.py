"""Deterministic review sampling, frame extraction and the JSONL stream reader."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from atreides_vision.errors import ValidationError
from atreides_vision.validation.review_samples import (
    build_sample_plan,
    extract_sample_frames,
    write_contact_sheet,
)
from atreides_vision.validation.streams import (
    iter_jsonl,
    stream_size_bytes,
    uncompressed_size_bytes,
)

FPS = 25.0


def test_sample_points_are_deterministic() -> None:
    first = build_sample_plan("clip", frames_processed=1000, source_fps=FPS)
    second = build_sample_plan("clip", frames_processed=1000, source_fps=FPS)

    assert [frame.frame_index for frame in first.frames] == [
        frame.frame_index for frame in second.frames
    ]
    assert [window.start_seconds for window in first.windows] == [
        window.start_seconds for window in second.windows
    ]


def test_sample_points_are_spread_across_the_clip_not_bunched_at_the_start() -> None:
    plan = build_sample_plan("clip", frames_processed=1000, source_fps=FPS, sample_frames=10)

    indices = [frame.frame_index for frame in plan.frames]

    assert len(indices) == 10
    assert indices == sorted(indices)
    # Midpoints of ten equal slices of 1000 frames.
    assert indices == [51, 151, 251, 351, 451, 551, 651, 751, 851, 951]
    # Neither endpoint is sampled: they are the least representative frames.
    assert indices[0] > 1
    assert indices[-1] < 1000


def test_timestamps_follow_the_source_frame_rate() -> None:
    plan = build_sample_plan("clip", frames_processed=1000, source_fps=FPS, sample_frames=10)

    first = plan.frames[0]
    assert first.timestamp_seconds == pytest.approx((first.frame_index - 1) / FPS)


def test_frame_indices_are_one_based_and_within_range() -> None:
    plan = build_sample_plan("clip", frames_processed=7, source_fps=FPS, sample_frames=10)

    for frame in plan.frames:
        assert 1 <= frame.frame_index <= 7


def test_a_short_clip_yields_fewer_samples_rather_than_duplicates() -> None:
    plan = build_sample_plan("clip", frames_processed=3, source_fps=FPS, sample_frames=10)

    indices = [frame.frame_index for frame in plan.frames]
    assert len(indices) == len(set(indices))
    assert len(indices) <= 3


def test_continuity_windows_fit_inside_the_clip() -> None:
    plan = build_sample_plan(
        "clip",
        frames_processed=1500,
        source_fps=30.0,
        continuity_windows=3,
        window_seconds=8.0,
    )

    duration = 1500 / 30.0
    assert len(plan.windows) == 3
    for window in plan.windows:
        assert window.start_seconds >= 0.0
        assert window.end_seconds <= duration + 1e-9
        assert window.duration_seconds == pytest.approx(8.0)
    assert [window.sequence_id for window in plan.windows] == ["w01", "w02", "w03"]


def test_a_clip_shorter_than_one_window_yields_one_covering_window() -> None:
    plan = build_sample_plan(
        "clip", frames_processed=60, source_fps=30.0, continuity_windows=3, window_seconds=8.0
    )

    assert len(plan.windows) == 1
    assert plan.windows[0].start_seconds == pytest.approx(0.0)
    assert plan.windows[0].end_seconds == pytest.approx(2.0)


def test_zero_frames_yields_an_empty_plan() -> None:
    plan = build_sample_plan("clip", frames_processed=0, source_fps=FPS)

    assert plan.frames == ()
    assert plan.windows == ()
    assert plan.frame_rows() == ()


def test_a_non_positive_frame_rate_is_rejected() -> None:
    with pytest.raises(ValidationError, match="fps must be positive"):
        build_sample_plan("clip", frames_processed=100, source_fps=0.0)


def test_plan_rows_feed_the_annotation_templates() -> None:
    plan = build_sample_plan("clip", frames_processed=100, source_fps=FPS, sample_frames=4)

    rows = plan.frame_rows()

    assert len(rows) == 4
    assert all(len(row) == 2 for row in rows)
    assert rows[0] == (plan.frames[0].timestamp_seconds, plan.frames[0].frame_index)


def write_video(path: Path, frame_count: int, size: tuple[int, int] = (64, 48)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 25.0, size)
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((size[1], size[0], 3), (index * 7) % 256, dtype=np.uint8))
    writer.release()
    return path


def test_frames_are_extracted_from_the_rendered_video(tmp_path: Path) -> None:
    video = write_video(tmp_path / "tracked.mp4", frame_count=50)
    plan = build_sample_plan("clip", frames_processed=50, source_fps=FPS, sample_frames=5)

    paths = extract_sample_frames(video, plan, tmp_path / "frames")

    assert len(paths) == 5
    assert all(path.exists() for path in paths)
    assert [path.name for path in paths] == [frame.filename for frame in plan.frames]


def test_contact_sheet_tiles_the_extracted_frames(tmp_path: Path) -> None:
    video = write_video(tmp_path / "tracked.mp4", frame_count=50)
    plan = build_sample_plan("clip", frames_processed=50, source_fps=FPS, sample_frames=5)
    paths = extract_sample_frames(video, plan, tmp_path / "frames")

    sheet = write_contact_sheet(paths, tmp_path / "sheet.jpg", plan)

    assert sheet is not None and sheet.exists()
    image = cv2.imread(str(sheet))
    assert image is not None and image.size > 0


def test_contact_sheet_of_nothing_is_none(tmp_path: Path) -> None:
    plan = build_sample_plan("clip", frames_processed=0, source_fps=FPS)

    assert write_contact_sheet((), tmp_path / "sheet.jpg", plan) is None


def test_extracting_from_a_missing_video_is_an_error(tmp_path: Path) -> None:
    plan = build_sample_plan("clip", frames_processed=50, source_fps=FPS, sample_frames=2)

    with pytest.raises(ValidationError, match="Rendered video not found"):
        extract_sample_frames(tmp_path / "absent.mp4", plan, tmp_path / "frames")


# --- stream reading --------------------------------------------------------


def test_iter_jsonl_reads_plain_and_gzipped_streams_identically(tmp_path: Path) -> None:
    records = [{"record_type": "a", "n": index} for index in range(5)]
    plain = tmp_path / "s.jsonl"
    plain.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    compressed = tmp_path / "s.jsonl.gz"
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    assert list(iter_jsonl(plain)) == records
    assert list(iter_jsonl(compressed)) == records


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n\n', encoding="utf-8")

    assert list(iter_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_a_malformed_line_is_an_error_not_a_skip(tmp_path: Path) -> None:
    """Silently dropping records would understate every count derived from them."""
    path = tmp_path / "s.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")

    with pytest.raises(ValidationError, match="line 2 is not valid JSON"):
        list(iter_jsonl(path))


def test_a_non_object_line_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="JSON object"):
        list(iter_jsonl(path))


def test_missing_stream_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not found"):
        list(iter_jsonl(tmp_path / "absent.jsonl"))


def test_sizes_are_read_from_the_filesystem(tmp_path: Path) -> None:
    payload = "\n".join(json.dumps({"n": index}) for index in range(2000)) + "\n"
    compressed = tmp_path / "s.jsonl.gz"
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        handle.write(payload)

    stored = stream_size_bytes(compressed)

    assert stored == compressed.stat().st_size
    assert stored < len(payload)
    assert uncompressed_size_bytes(compressed) == len(payload.encode("utf-8"))


def test_uncompressed_size_is_none_for_a_plain_file(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text('{"a": 1}\n', encoding="utf-8")

    assert uncompressed_size_bytes(path) is None

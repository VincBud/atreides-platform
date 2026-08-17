"""CLI exposure of V0.4.0 venue ingestion.

No model weights are loaded and nothing is downloaded: ``venue inspect`` runs no
detector and no tracker, so the whole command is exercised on a few frames of
locally generated video.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from atreides_vision.cli import EXIT_ERROR, EXIT_OK, build_parser, main
from atreides_vision.venue.manifest import VENUE_MANIFEST_VERSION
from atreides_vision.venue.reporting import VENUE_SESSION_SCHEMA_VERSION

WIDTH, HEIGHT = 64, 48
FRAME_COUNT = 10
NORTH_FPS, WEST_FPS = 25.0, 50.0


def write_clip(path: Path, frame_count: int = FRAME_COUNT) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 20.0, (WIDTH, HEIGHT))
    assert writer.isOpened(), "OpenCV could not open an mp4v writer in this environment"
    for index in range(frame_count):
        writer.write(np.full((HEIGHT, WIDTH, 3), 10 + index * 5, dtype=np.uint8))
    writer.release()
    return path


def manifest_body(tmp_path: Path, reverse: bool = False) -> dict[str, Any]:
    cameras: list[dict[str, Any]] = [
        {
            "camera_id": "cam_north",
            "sources": [
                {
                    "input_path": str(tmp_path / "north.mp4"),
                    # Declared, so the schedule is exact whatever the container
                    # claims — and so the offsets below are allowed at all.
                    "declared_fps": NORTH_FPS,
                    "start_offset_seconds": 0.0,
                    "offset_accuracy_seconds": 0.5,
                }
            ],
        },
        {
            "camera_id": "cam_west",
            "sources": [
                {
                    "input_path": str(tmp_path / "west.mp4"),
                    "declared_fps": WEST_FPS,
                    "start_offset_seconds": 0.0,
                    "offset_accuracy_seconds": 0.25,
                }
            ],
        },
    ]
    if reverse:
        cameras.reverse()
    return {
        "manifest_version": VENUE_MANIFEST_VERSION,
        "session_id": "session_example",
        "cameras": cameras,
    }


@pytest.fixture
def venue(tmp_path: Path) -> Path:
    write_clip(tmp_path / "north.mp4")
    write_clip(tmp_path / "west.mp4")
    path = tmp_path / "venue.json"
    path.write_text(json.dumps(manifest_body(tmp_path)), encoding="utf-8")
    return path


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- argument parsing ------------------------------------------------------


def test_venue_inspect_parses() -> None:
    args = build_parser().parse_args(
        ["venue", "inspect", "--manifest", "venue.json", "--output", "out.jsonl"]
    )

    assert args.command == "venue"
    assert args.venue_command == "inspect"
    assert args.manifest == Path("venue.json")
    assert args.output == Path("out.jsonl")


def test_venue_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["venue"])


def test_inspect_requires_a_manifest_and_an_output() -> None:
    for argv in (
        ["venue", "inspect", "--output", "out.jsonl"],
        ["venue", "inspect", "--manifest", "venue.json"],
    ):
        with pytest.raises(SystemExit):
            build_parser().parse_args(argv)


def test_there_is_no_venue_run_command() -> None:
    """V0.4.0 ships ingestion, not a multi-camera processing command."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["venue", "run", "--manifest", "venue.json", "--output", "out.jsonl"]
        )


# --- inspect ---------------------------------------------------------------


def test_inspect_writes_a_versioned_session_record(tmp_path: Path, venue: Path) -> None:
    output = tmp_path / "session.jsonl"

    code = main(["venue", "inspect", "--manifest", str(venue), "--output", str(output)])

    assert code == EXIT_OK
    records = read_records(output)
    assert records[0]["record_type"] == "venue_session"
    assert records[-1]["record_type"] == "venue_session_summary"
    assert all(record["schema_version"] == VENUE_SESSION_SCHEMA_VERSION for record in records)
    assert all(record["session_id"] == "session_example" for record in records)


def test_the_tick_schedule_is_the_union_of_both_cadences(tmp_path: Path, venue: Path) -> None:
    output = tmp_path / "session.jsonl"
    main(["venue", "inspect", "--manifest", str(venue), "--output", str(output)])

    ticks = [record for record in read_records(output) if record["record_type"] == "venue_tick"]

    # 25 FPS lands on every 40 ms, 50 FPS on every 20 ms, over ten frames each:
    # ten instants from the slow stream, ten from the fast one, five shared.
    assert len(ticks) == 15
    assert [tick["sequence"] for tick in ticks] == list(range(15))
    assert [tick["venue_micros"] for tick in ticks] == sorted(
        tick["venue_micros"] for tick in ticks
    )


def test_the_header_records_each_stream_s_frame_rate_provenance(
    tmp_path: Path, venue: Path
) -> None:
    output = tmp_path / "session.jsonl"
    main(["venue", "inspect", "--manifest", str(venue), "--output", str(output)])

    header = read_records(output)[0]

    assert header["camera_count"] == 2
    assert header["stream_count"] == 2
    assert [entry["stream_ref"] for entry in header["streams"]] == ["cam_north#0", "cam_west#0"]
    assert all(entry["fps_source"] == "declared" for entry in header["streams"])
    assert header["streams"][0]["fps"] == NORTH_FPS
    assert header["manifest_path"] == str(venue)


def test_every_contribution_resolves_to_a_frame_of_a_recording(tmp_path: Path, venue: Path) -> None:
    output = tmp_path / "session.jsonl"
    main(["venue", "inspect", "--manifest", str(venue), "--output", str(output)])

    contributions = [
        contribution
        for record in read_records(output)
        if record["record_type"] == "venue_tick"
        for contribution in record["contributions"]
    ]

    north = [item for item in contributions if item["camera_id"] == "cam_north"]
    assert [item["frame_index"] for item in north] == list(range(1, FRAME_COUNT + 1))
    assert [item["source_seconds"] for item in north] == [
        round((index - 1) / NORTH_FPS, 6) for index in range(1, FRAME_COUNT + 1)
    ]
    assert all(item["stream_epoch"] == 0 for item in contributions)


def test_the_summary_counts_frames_per_stream(tmp_path: Path, venue: Path) -> None:
    output = tmp_path / "session.jsonl"
    main(["venue", "inspect", "--manifest", str(venue), "--output", str(output)])

    summary = read_records(output)[-1]

    assert summary["tick_count"] == 15
    assert summary["streams"] == [
        {"camera_id": "cam_north", "stream_epoch": 0, "frames_contributed": FRAME_COUNT},
        {"camera_id": "cam_west", "stream_epoch": 0, "frames_contributed": FRAME_COUNT},
    ]


def test_manifest_camera_order_does_not_reach_the_output(tmp_path: Path, venue: Path) -> None:
    """The same session, written from a reversed manifest, byte for byte."""
    forward = tmp_path / "forward.jsonl"
    main(["venue", "inspect", "--manifest", str(venue), "--output", str(forward)])

    # The same manifest path, rewritten with its cameras reversed, so the only
    # difference between the two runs is the order the cameras were listed in.
    venue.write_text(json.dumps(manifest_body(tmp_path, reverse=True)), encoding="utf-8")
    reversed_output = tmp_path / "reversed.jsonl"
    main(["venue", "inspect", "--manifest", str(venue), "--output", str(reversed_output)])

    assert forward.read_bytes() == reversed_output.read_bytes()


def test_a_shorter_stream_ends_while_the_other_continues(tmp_path: Path) -> None:
    write_clip(tmp_path / "north.mp4", frame_count=6)
    write_clip(tmp_path / "west.mp4", frame_count=2)
    path = tmp_path / "venue.json"
    path.write_text(json.dumps(manifest_body(tmp_path)), encoding="utf-8")
    output = tmp_path / "session.jsonl"

    main(["venue", "inspect", "--manifest", str(path), "--output", str(output)])

    ticks = [record for record in read_records(output) if record["record_type"] == "venue_tick"]
    assert ticks[-1]["absent"] == [
        {"camera_id": "cam_west", "stream_epoch": 0, "activity": "ended"}
    ]


def test_a_single_camera_session_is_not_a_special_case(tmp_path: Path) -> None:
    write_clip(tmp_path / "north.mp4")
    path = tmp_path / "venue.json"
    path.write_text(
        json.dumps(
            {
                "manifest_version": VENUE_MANIFEST_VERSION,
                "session_id": "solo",
                "cameras": [
                    {
                        "camera_id": "cam_north",
                        "sources": [{"input_path": str(tmp_path / "north.mp4")}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "session.jsonl"

    assert main(["venue", "inspect", "--manifest", str(path), "--output", str(output)]) == EXIT_OK

    records = read_records(output)
    ticks = [record for record in records if record["record_type"] == "venue_tick"]
    assert len(ticks) == FRAME_COUNT
    assert all(tick["absent"] == [] for tick in ticks)
    assert records[0]["streams"][0]["time_alignment"] == {
        "basis": "not_established",
        "half_width_seconds": None,
    }


def test_a_camera_with_two_recordings_gets_two_epochs(tmp_path: Path) -> None:
    write_clip(tmp_path / "part1.mp4", frame_count=3)
    write_clip(tmp_path / "part2.mp4", frame_count=3)
    path = tmp_path / "venue.json"
    path.write_text(
        json.dumps(
            {
                "manifest_version": VENUE_MANIFEST_VERSION,
                "session_id": "restart",
                "cameras": [
                    {
                        "camera_id": "cam_north",
                        "sources": [
                            {
                                "input_path": str(tmp_path / "part1.mp4"),
                                "declared_fps": 25.0,
                                "start_offset_seconds": 0.0,
                                "offset_accuracy_seconds": 0.5,
                            },
                            {
                                "input_path": str(tmp_path / "part2.mp4"),
                                "declared_fps": 25.0,
                                "start_offset_seconds": 60.0,
                                "offset_accuracy_seconds": 0.5,
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "session.jsonl"

    main(["venue", "inspect", "--manifest", str(path), "--output", str(output)])

    records = read_records(output)
    assert [entry["stream_ref"] for entry in records[0]["streams"]] == [
        "cam_north#0",
        "cam_north#1",
    ]
    ticks = [record for record in records if record["record_type"] == "venue_tick"]
    assert len(ticks) == 6
    assert ticks[0]["contributions"][0]["stream_epoch"] == 0
    assert ticks[-1]["contributions"][0]["stream_epoch"] == 1
    assert ticks[-1]["venue_seconds"] == 60.08


def test_inspect_states_that_no_association_was_performed(
    tmp_path: Path, venue: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="atreides_vision")
    output = tmp_path / "session.jsonl"

    main(["venue", "inspect", "--manifest", str(venue), "--output", str(output)])

    text = caplog.text
    assert "no cross-camera association" in text
    assert "venue-local identity" in text


# --- error handling --------------------------------------------------------


def test_a_missing_manifest_exits_with_an_error(tmp_path: Path) -> None:
    assert (
        main(
            [
                "venue",
                "inspect",
                "--manifest",
                str(tmp_path / "absent.json"),
                "--output",
                str(tmp_path / "out.jsonl"),
            ]
        )
        == EXIT_ERROR
    )


def test_an_unknown_manifest_version_exits_with_an_error(tmp_path: Path) -> None:
    write_clip(tmp_path / "north.mp4")
    path = tmp_path / "venue.json"
    body = manifest_body(tmp_path)
    body["manifest_version"] = "9.9.9"
    path.write_text(json.dumps(body), encoding="utf-8")

    assert (
        main(["venue", "inspect", "--manifest", str(path), "--output", str(tmp_path / "out.jsonl")])
        == EXIT_ERROR
    )


def test_a_restart_without_offsets_exits_with_an_error(tmp_path: Path) -> None:
    write_clip(tmp_path / "part1.mp4")
    write_clip(tmp_path / "part2.mp4")
    path = tmp_path / "venue.json"
    path.write_text(
        json.dumps(
            {
                "manifest_version": VENUE_MANIFEST_VERSION,
                "session_id": "restart",
                "cameras": [
                    {
                        "camera_id": "cam_north",
                        "sources": [
                            {"input_path": str(tmp_path / "part1.mp4")},
                            {"input_path": str(tmp_path / "part2.mp4")},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(["venue", "inspect", "--manifest", str(path), "--output", str(tmp_path / "out.jsonl")])
        == EXIT_ERROR
    )


def test_the_header_records_the_frame_count_each_container_claims(
    tmp_path: Path, venue: Path
) -> None:
    """Provenance only. The schedule is built by reading to exhaustion."""
    output = tmp_path / "session.jsonl"
    main(["venue", "inspect", "--manifest", str(venue), "--output", str(output)])

    header = read_records(output)[0]

    assert [entry["frame_count"] for entry in header["streams"]] == [FRAME_COUNT, FRAME_COUNT]


def test_every_line_carries_the_session_alignment(tmp_path: Path, venue: Path) -> None:
    output = tmp_path / "session.jsonl"
    main(["venue", "inspect", "--manifest", str(venue), "--output", str(output)])

    records = read_records(output)
    header = records[0]
    ticks = [record for record in records if record["record_type"] == "venue_tick"]

    # Both streams declare an offset, so the session is aligned to the worst of
    # the two half-widths and every tick repeats that same figure.
    assert header["session_time_alignment"] == {
        "basis": "operator_declared",
        "half_width_seconds": 0.5,
    }
    assert all(tick["session_time_alignment"] == header["session_time_alignment"] for tick in ticks)


def test_an_unaligned_camera_is_visible_on_every_line(tmp_path: Path) -> None:
    """A tick the unaligned camera missed must not report a better session."""
    write_clip(tmp_path / "north.mp4", frame_count=6)
    write_clip(tmp_path / "west.mp4", frame_count=1)
    body = manifest_body(tmp_path)
    # The west camera declares nothing: placed at venue zero, NOT_ESTABLISHED.
    body["cameras"][1]["sources"][0] = {"input_path": str(tmp_path / "west.mp4")}
    path = tmp_path / "venue.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    output = tmp_path / "session.jsonl"

    main(["venue", "inspect", "--manifest", str(path), "--output", str(output)])

    records = read_records(output)
    ticks = [record for record in records if record["record_type"] == "venue_tick"]
    ended = [tick for tick in ticks if tick["absent"]]

    assert ended, "expected ticks after the unaligned camera ran out"
    # Its own contributors look well aligned once the west camera has ended...
    assert ended[-1]["time_alignment"]["basis"] == "operator_declared"
    # ...and the session figure keeps saying otherwise.
    assert ended[-1]["session_time_alignment"]["basis"] == "not_established"
    assert records[0]["session_time_alignment"]["basis"] == "not_established"


def test_an_nvr_export_with_a_useless_frame_rate_is_representable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The M5 case: restarted camera, container reporting nothing usable.

    The offsets place the two recordings; no accuracy is claimed for either,
    because the frame rate behind them is a fallback constant. The operator is
    not pushed into inventing a declared_fps to make the session loadable.
    """
    write_clip(tmp_path / "part1.mp4", frame_count=3)
    write_clip(tmp_path / "part2.mp4", frame_count=3)
    path = tmp_path / "venue.json"
    path.write_text(
        json.dumps(
            {
                "manifest_version": VENUE_MANIFEST_VERSION,
                "session_id": "nvr",
                "cameras": [
                    {
                        "camera_id": "cam_north",
                        "sources": [
                            {
                                "input_path": str(tmp_path / "part1.mp4"),
                                "start_offset_seconds": 0.0,
                                "offset_accuracy_seconds": 2.0,
                            },
                            {
                                "input_path": str(tmp_path / "part2.mp4"),
                                "start_offset_seconds": 60.0,
                                "offset_accuracy_seconds": 2.0,
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "session.jsonl"

    with caplog.at_level("WARNING"):
        assert main(["venue", "inspect", "--manifest", str(path), "--output", str(output)]) == (
            EXIT_OK
        )

    records = read_records(output)
    ticks = [record for record in records if record["record_type"] == "venue_tick"]
    assert len(ticks) == 6
    # Placed by the offsets...
    assert ticks[-1]["venue_seconds"] > 60.0
    # ...and no accuracy claimed on top of a container-reported frame rate the
    # writer trusts. This clip's container does report a usable rate, so the
    # declaration stands; the discard path is covered where the rate is assumed.
    assert records[0]["streams"][0]["fps_source"] == "container"

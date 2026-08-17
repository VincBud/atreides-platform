"""Venue manifest loading: identity, epochs, offsets, and what is rejected.

The manifest is where a deployment describes itself. Every rejection here is a
number that would otherwise have been invented on an operator's behalf.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from atreides_vision.errors import VenueSessionError
from atreides_vision.venue.manifest import (
    VENUE_MANIFEST_VERSION,
    VenueCamera,
    VenueSessionManifest,
    VenueStreamEntry,
    load_venue_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def touch(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(b"not really a video, but it exists")
    return path


def source_entry(filename: str, **extra: object) -> dict[str, object]:
    return {"input_path": filename, **extra}


def camera_entry(camera_id: str, *sources: dict[str, object]) -> dict[str, object]:
    return {"camera_id": camera_id, "sources": list(sources)}


def write_manifest(
    tmp_path: Path,
    cameras: list[dict[str, object]],
    version: str | None = None,
    session_id: str = "session_a",
    name: str = "venue.json",
) -> Path:
    path = tmp_path / name
    body: dict[str, Any] = {
        "manifest_version": version or VENUE_MANIFEST_VERSION,
        "session_id": session_id,
        "cameras": cameras,
    }
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


class TestLoading:
    def test_loads_a_single_camera_session(self, tmp_path: Path) -> None:
        touch(tmp_path, "a.mp4")
        path = write_manifest(tmp_path, [camera_entry("cam_a", source_entry("a.mp4"))])

        manifest = load_venue_manifest(path, base_dir=tmp_path)

        assert manifest.session_id == "session_a"
        assert len(manifest) == 1
        assert manifest.stream_count == 1
        entry = manifest.camera("cam_a").sources[0]
        assert entry.input_path == tmp_path / "a.mp4"
        assert entry.start_offset_seconds is None
        assert entry.offset_accuracy_seconds is None
        assert entry.declared_fps is None
        assert manifest.source_path == path

    def test_loads_many_cameras_and_keeps_manifest_order(self, tmp_path: Path) -> None:
        for name in ("z.mp4", "a.mp4", "m.mp4"):
            touch(tmp_path, name)
        path = write_manifest(
            tmp_path,
            [
                camera_entry("cam_z", source_entry("z.mp4")),
                camera_entry("cam_a", source_entry("a.mp4")),
                camera_entry("cam_m", source_entry("m.mp4")),
            ],
        )

        manifest = load_venue_manifest(path, base_dir=tmp_path)

        assert manifest.camera_ids == ("cam_z", "cam_a", "cam_m")
        assert manifest.stream_count == 3

    def test_resolves_relative_paths_against_the_base_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "clips"
        nested.mkdir()
        touch(nested, "a.mp4")
        path = write_manifest(tmp_path, [camera_entry("cam_a", source_entry("clips/a.mp4"))])

        manifest = load_venue_manifest(path, base_dir=tmp_path)

        assert manifest.camera("cam_a").sources[0].input_path == nested / "a.mp4"

    def test_reads_declared_timing(self, tmp_path: Path) -> None:
        touch(tmp_path, "a.mp4")
        path = write_manifest(
            tmp_path,
            [
                camera_entry(
                    "cam_a",
                    source_entry(
                        "a.mp4",
                        start_offset_seconds=12.5,
                        offset_accuracy_seconds=0.25,
                        declared_fps=25.0,
                    ),
                )
            ],
        )

        entry = load_venue_manifest(path, base_dir=tmp_path).camera("cam_a").sources[0]

        assert entry.start_offset_seconds == 12.5
        assert entry.offset_accuracy_seconds == 0.25
        assert entry.declared_fps == 25.0

    def test_several_sources_of_one_camera_keep_their_epoch_order(self, tmp_path: Path) -> None:
        touch(tmp_path, "part1.mp4")
        touch(tmp_path, "part2.mp4")
        path = write_manifest(
            tmp_path,
            [
                camera_entry(
                    "cam_a",
                    source_entry(
                        "part1.mp4", start_offset_seconds=0.0, offset_accuracy_seconds=0.5
                    ),
                    source_entry(
                        "part2.mp4", start_offset_seconds=600.0, offset_accuracy_seconds=0.5
                    ),
                )
            ],
        )

        camera = load_venue_manifest(path, base_dir=tmp_path).camera("cam_a")

        assert camera.is_multi_source
        assert [entry.input_path.name for entry in camera] == ["part1.mp4", "part2.mp4"]
        assert [entry.start_offset_seconds for entry in camera] == [0.0, 600.0]


class TestRejections:
    def test_missing_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(VenueSessionError, match="not found"):
            load_venue_manifest(tmp_path / "absent.json")

    def test_directory_instead_of_a_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(VenueSessionError, match="not a file"):
            load_venue_manifest(tmp_path)

    def test_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(VenueSessionError, match="not valid JSON"):
            load_venue_manifest(path)

    def test_a_json_list_is_not_a_manifest(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(VenueSessionError, match="JSON object"):
            load_venue_manifest(path)

    def test_an_unknown_version_is_not_guessed_at(self, tmp_path: Path) -> None:
        touch(tmp_path, "a.mp4")
        path = write_manifest(
            tmp_path, [camera_entry("cam_a", source_entry("a.mp4"))], version="0.9.9"
        )
        with pytest.raises(VenueSessionError, match="Refusing to guess"):
            load_venue_manifest(path, base_dir=tmp_path)

    def test_a_missing_session_id(self, tmp_path: Path) -> None:
        touch(tmp_path, "a.mp4")
        path = tmp_path / "venue.json"
        path.write_text(
            json.dumps(
                {
                    "manifest_version": VENUE_MANIFEST_VERSION,
                    "cameras": [camera_entry("cam_a", source_entry("a.mp4"))],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(VenueSessionError, match="session_id"):
            load_venue_manifest(path, base_dir=tmp_path)

    def test_no_cameras(self, tmp_path: Path) -> None:
        path = write_manifest(tmp_path, [])
        with pytest.raises(VenueSessionError, match="non-empty 'cameras'"):
            load_venue_manifest(path, base_dir=tmp_path)

    def test_a_camera_with_no_sources(self, tmp_path: Path) -> None:
        path = write_manifest(tmp_path, [camera_entry("cam_a")])
        with pytest.raises(VenueSessionError, match="non-empty 'sources'"):
            load_venue_manifest(path, base_dir=tmp_path)

    def test_duplicate_camera_id(self, tmp_path: Path) -> None:
        touch(tmp_path, "a.mp4")
        touch(tmp_path, "b.mp4")
        path = write_manifest(
            tmp_path,
            [
                camera_entry("cam_a", source_entry("a.mp4")),
                camera_entry("cam_a", source_entry("b.mp4")),
            ],
        )
        with pytest.raises(VenueSessionError, match="Duplicate camera_id"):
            load_venue_manifest(path, base_dir=tmp_path)

    def test_a_missing_input_file_stops_the_session_before_it_starts(self, tmp_path: Path) -> None:
        path = write_manifest(tmp_path, [camera_entry("cam_a", source_entry("gone.mp4"))])
        with pytest.raises(VenueSessionError, match="input file not found"):
            load_venue_manifest(path, base_dir=tmp_path)

    def test_several_sources_without_offsets(self, tmp_path: Path) -> None:
        touch(tmp_path, "part1.mp4")
        touch(tmp_path, "part2.mp4")
        path = write_manifest(
            tmp_path,
            [camera_entry("cam_a", source_entry("part1.mp4"), source_entry("part2.mp4"))],
        )
        with pytest.raises(VenueSessionError, match="start_offset_seconds"):
            load_venue_manifest(path, base_dir=tmp_path)

    def test_several_sources_sharing_one_offset(self, tmp_path: Path) -> None:
        """Presence is not enough: a camera cannot record two things at once.

        Two segments at venue zero would contribute two frames at every venue
        instant from one camera, and a per-camera accessor would report a single
        CONTRIBUTED — the duplication invisible at exactly the level that would
        have to notice it.
        """
        touch(tmp_path, "part1.mp4")
        touch(tmp_path, "part2.mp4")
        path = write_manifest(
            tmp_path,
            [
                camera_entry(
                    "cam_a",
                    source_entry(
                        "part1.mp4", start_offset_seconds=0.0, offset_accuracy_seconds=0.5
                    ),
                    source_entry(
                        "part2.mp4", start_offset_seconds=0.0, offset_accuracy_seconds=0.5
                    ),
                )
            ],
        )

        with pytest.raises(VenueSessionError, match="cannot record two things at once"):
            load_venue_manifest(path, base_dir=tmp_path)

    def test_several_sources_in_decreasing_order(self, tmp_path: Path) -> None:
        touch(tmp_path, "part1.mp4")
        touch(tmp_path, "part2.mp4")
        path = write_manifest(
            tmp_path,
            [
                camera_entry(
                    "cam_a",
                    source_entry(
                        "part1.mp4", start_offset_seconds=600.0, offset_accuracy_seconds=0.5
                    ),
                    source_entry(
                        "part2.mp4", start_offset_seconds=60.0, offset_accuracy_seconds=0.5
                    ),
                )
            ],
        )

        with pytest.raises(VenueSessionError, match="not after source #1"):
            load_venue_manifest(path, base_dir=tmp_path)

    def test_two_cameras_may_share_a_start_offset(self, tmp_path: Path) -> None:
        """The rule is per camera: two cameras recording at once is ordinary."""
        touch(tmp_path, "a.mp4")
        touch(tmp_path, "b.mp4")
        path = write_manifest(
            tmp_path,
            [
                camera_entry(
                    "cam_a",
                    source_entry("a.mp4", start_offset_seconds=0.0, offset_accuracy_seconds=0.5),
                ),
                camera_entry(
                    "cam_b",
                    source_entry("b.mp4", start_offset_seconds=0.0, offset_accuracy_seconds=0.5),
                ),
            ],
        )

        manifest = load_venue_manifest(path, base_dir=tmp_path)

        assert manifest.stream_count == 2

    def test_an_offset_without_an_accuracy(self, tmp_path: Path) -> None:
        touch(tmp_path, "a.mp4")
        path = write_manifest(
            tmp_path, [camera_entry("cam_a", source_entry("a.mp4", start_offset_seconds=4.0))]
        )
        with pytest.raises(VenueSessionError, match="offset_accuracy_seconds"):
            load_venue_manifest(path, base_dir=tmp_path)

    def test_an_accuracy_without_an_offset(self, tmp_path: Path) -> None:
        touch(tmp_path, "a.mp4")
        path = write_manifest(
            tmp_path, [camera_entry("cam_a", source_entry("a.mp4", offset_accuracy_seconds=0.5))]
        )
        with pytest.raises(VenueSessionError, match="was not declared"):
            load_venue_manifest(path, base_dir=tmp_path)

    def test_a_negative_offset(self, tmp_path: Path) -> None:
        touch(tmp_path, "a.mp4")
        path = write_manifest(
            tmp_path,
            [
                camera_entry(
                    "cam_a",
                    source_entry("a.mp4", start_offset_seconds=-1.0, offset_accuracy_seconds=0.5),
                )
            ],
        )
        with pytest.raises(VenueSessionError, match="non-negative"):
            load_venue_manifest(path, base_dir=tmp_path)

    def test_a_zero_accuracy(self, tmp_path: Path) -> None:
        touch(tmp_path, "a.mp4")
        path = write_manifest(
            tmp_path,
            [
                camera_entry(
                    "cam_a",
                    source_entry("a.mp4", start_offset_seconds=1.0, offset_accuracy_seconds=0.0),
                )
            ],
        )
        with pytest.raises(VenueSessionError, match="strictly positive"):
            load_venue_manifest(path, base_dir=tmp_path)

    @pytest.mark.parametrize("value", [0.0, -25.0])
    def test_an_unusable_declared_fps(self, tmp_path: Path, value: float) -> None:
        touch(tmp_path, "a.mp4")
        path = write_manifest(
            tmp_path, [camera_entry("cam_a", source_entry("a.mp4", declared_fps=value))]
        )
        with pytest.raises(VenueSessionError, match="declared_fps"):
            load_venue_manifest(path, base_dir=tmp_path)

    def test_a_boolean_is_not_a_number(self, tmp_path: Path) -> None:
        touch(tmp_path, "a.mp4")
        path = write_manifest(
            tmp_path, [camera_entry("cam_a", source_entry("a.mp4", declared_fps=True))]
        )
        with pytest.raises(VenueSessionError, match="must be a number"):
            load_venue_manifest(path, base_dir=tmp_path)


class TestDomainInvariants:
    """Constructed directly, for the values JSON cannot easily express."""

    @pytest.mark.parametrize("value", [math.nan, math.inf])
    def test_non_finite_offset(self, tmp_path: Path, value: float) -> None:
        with pytest.raises(VenueSessionError, match="finite"):
            VenueStreamEntry(
                input_path=tmp_path / "a.mp4",
                start_offset_seconds=value,
                offset_accuracy_seconds=0.5,
            )

    @pytest.mark.parametrize("value", [math.nan, math.inf])
    def test_non_finite_accuracy(self, tmp_path: Path, value: float) -> None:
        with pytest.raises(VenueSessionError, match="finite"):
            VenueStreamEntry(
                input_path=tmp_path / "a.mp4",
                start_offset_seconds=1.0,
                offset_accuracy_seconds=value,
            )

    def test_a_blank_camera_id(self, tmp_path: Path) -> None:
        with pytest.raises(VenueSessionError, match="non-empty camera_id"):
            VenueCamera(camera_id="  ", sources=(VenueStreamEntry(tmp_path / "a.mp4"),))

    def test_a_blank_session_id(self, tmp_path: Path) -> None:
        camera = VenueCamera(camera_id="cam_a", sources=(VenueStreamEntry(tmp_path / "a.mp4"),))
        with pytest.raises(VenueSessionError, match="non-empty session_id"):
            VenueSessionManifest(session_id=" ", cameras=(camera,))

    def test_the_ordering_rule_does_not_claim_non_overlap(self, tmp_path: Path) -> None:
        """Documented limit: overlapping segments with increasing starts pass.

        Establishing that segment N+1 begins after segment N *ends* would need
        each segment's duration, which means trusting the container's
        frame_count — which this layer deliberately never does.
        """
        camera = VenueCamera(
            camera_id="cam_a",
            sources=(
                VenueStreamEntry(
                    tmp_path / "part1.mp4",
                    start_offset_seconds=0.0,
                    offset_accuracy_seconds=0.5,
                ),
                VenueStreamEntry(
                    tmp_path / "part2.mp4",
                    start_offset_seconds=0.001,
                    offset_accuracy_seconds=0.5,
                ),
            ),
        )

        assert camera.is_multi_source

    def test_an_unknown_camera_is_named_in_the_error(self, tmp_path: Path) -> None:
        camera = VenueCamera(camera_id="cam_a", sources=(VenueStreamEntry(tmp_path / "a.mp4"),))
        manifest = VenueSessionManifest(session_id="s", cameras=(camera,))
        with pytest.raises(VenueSessionError, match="Known cameras: cam_a"):
            manifest.camera("cam_b")


def test_the_shipped_example_declares_the_current_contract() -> None:
    """The example manifest must not drift away from the loader."""
    example = REPO_ROOT / "config" / "venue" / "example.json"
    body = json.loads(example.read_text(encoding="utf-8"))

    assert body["manifest_version"] == VENUE_MANIFEST_VERSION
    assert body["session_id"]
    cameras = body["cameras"]
    assert len({camera["camera_id"] for camera in cameras}) == len(cameras)
    for camera in cameras:
        sources = camera["sources"]
        assert sources
        if len(sources) > 1:
            assert all("start_offset_seconds" in source for source in sources)
        for source in sources:
            if "start_offset_seconds" in source:
                assert "offset_accuracy_seconds" in source

"""Manifest loading: identity rules, path resolution, and what is rejected.

The manifest establishes the camera identity every output record is namespaced
by, so the rejections matter as much as the successes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atreides_vision.errors import ValidationError
from atreides_vision.validation.manifest import (
    MANIFEST_VERSION,
    ValidationClip,
    ValidationManifest,
    load_manifest,
)


def clip_entry(clip_id: str, camera_id: str, filename: str, **extra: object) -> dict[str, object]:
    return {
        "clip_id": clip_id,
        "input_path": filename,
        "scene_type": "control",
        "camera_id": camera_id,
        **extra,
    }


def write_manifest(
    tmp_path: Path, clips: list[dict[str, object]], version: str | None = None
) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"manifest_version": version or MANIFEST_VERSION, "clips": clips}),
        encoding="utf-8",
    )
    return path


def touch(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(b"not really a video, but it exists")
    return path


def test_loads_a_valid_manifest(tmp_path: Path) -> None:
    touch(tmp_path, "a.mp4")
    path = write_manifest(tmp_path, [clip_entry("a", "cam_a", "a.mp4", notes="a note")])

    manifest = load_manifest(path, base_dir=tmp_path)

    assert len(manifest) == 1
    clip = manifest.clip("a")
    assert clip.camera_id == "cam_a"
    assert clip.scene_type == "control"
    assert clip.notes == "a note"
    assert clip.input_path == tmp_path / "a.mp4"
    assert clip.calibration_path is None
    assert manifest.source_path == path


def test_preserves_manifest_order(tmp_path: Path) -> None:
    for name in ("z.mp4", "a.mp4", "m.mp4"):
        touch(tmp_path, name)
    path = write_manifest(
        tmp_path,
        [
            clip_entry("z", "cam_z", "z.mp4"),
            clip_entry("a", "cam_a", "a.mp4"),
            clip_entry("m", "cam_m", "m.mp4"),
        ],
    )

    manifest = load_manifest(path, base_dir=tmp_path)

    assert [clip.clip_id for clip in manifest] == ["z", "a", "m"]


def test_missing_input_file_is_rejected(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, [clip_entry("a", "cam_a", "nowhere.mp4")])

    with pytest.raises(ValidationError, match="input file not found"):
        load_manifest(path, base_dir=tmp_path)


def test_duplicate_clip_id_is_rejected(tmp_path: Path) -> None:
    touch(tmp_path, "a.mp4")
    touch(tmp_path, "b.mp4")
    path = write_manifest(
        tmp_path, [clip_entry("a", "cam_a", "a.mp4"), clip_entry("a", "cam_b", "b.mp4")]
    )

    with pytest.raises(ValidationError, match="Duplicate clip_id"):
        load_manifest(path, base_dir=tmp_path)


def test_several_clips_may_share_one_camera_id(tmp_path: Path) -> None:
    """One physical camera legitimately supplies many independent clips.

    Different times, lighting, densities, days or scenarios. Requiring a
    fabricated distinct id per clip would misdescribe the deployment.
    """
    for name in ("morning.mp4", "evening.mp4", "busy.mp4"):
        touch(tmp_path, name)
    path = write_manifest(
        tmp_path,
        [
            clip_entry("morning", "camera_01", "morning.mp4"),
            clip_entry("evening", "camera_01", "evening.mp4"),
            clip_entry("busy", "camera_02", "busy.mp4"),
        ],
    )

    manifest = load_manifest(path, base_dir=tmp_path)

    assert len(manifest) == 3
    assert manifest.clip("morning").camera_id == "camera_01"
    assert manifest.clip("evening").camera_id == "camera_01"
    assert manifest.camera_ids == ("camera_01", "camera_02")
    assert manifest.shared_camera_ids == ("camera_01",)
    assert [clip.clip_id for clip in manifest.clips_for_camera("camera_01")] == [
        "morning",
        "evening",
    ]


def test_clips_from_one_camera_stay_distinct_manifest_entries(tmp_path: Path) -> None:
    """Sharing a camera id merges nothing: they remain separate clips throughout."""
    touch(tmp_path, "a.mp4")
    touch(tmp_path, "b.mp4")
    path = write_manifest(
        tmp_path,
        [clip_entry("clip_a", "camera_01", "a.mp4"), clip_entry("clip_b", "camera_01", "b.mp4")],
    )

    manifest = load_manifest(path, base_dir=tmp_path)

    first, second = manifest.clips
    assert first is not second
    assert first.clip_id != second.clip_id
    assert first.input_path != second.input_path
    # Selecting one never drags the other along, even sharing a camera.
    assert [clip.clip_id for clip in manifest.selecting(["clip_b"])] == ["clip_b"]


def test_no_shared_cameras_reports_none(tmp_path: Path) -> None:
    touch(tmp_path, "a.mp4")
    touch(tmp_path, "b.mp4")
    path = write_manifest(
        tmp_path, [clip_entry("a", "cam_a", "a.mp4"), clip_entry("b", "cam_b", "b.mp4")]
    )

    manifest = load_manifest(path, base_dir=tmp_path)

    assert manifest.shared_camera_ids == ()
    assert manifest.camera_ids == ("cam_a", "cam_b")


def test_unknown_manifest_version_is_rejected(tmp_path: Path) -> None:
    touch(tmp_path, "a.mp4")
    path = write_manifest(tmp_path, [clip_entry("a", "cam_a", "a.mp4")], version="0.9.9")

    with pytest.raises(ValidationError, match="manifest_version"):
        load_manifest(path, base_dir=tmp_path)


def test_malformed_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValidationError, match="not valid JSON"):
        load_manifest(path, base_dir=tmp_path)


def test_missing_manifest_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not found"):
        load_manifest(tmp_path / "absent.json", base_dir=tmp_path)


def test_empty_clip_list_is_rejected(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, [])

    with pytest.raises(ValidationError, match="non-empty 'clips' list"):
        load_manifest(path, base_dir=tmp_path)


def test_blank_camera_id_is_rejected(tmp_path: Path) -> None:
    touch(tmp_path, "a.mp4")
    path = write_manifest(tmp_path, [clip_entry("a", "   ", "a.mp4")])

    with pytest.raises(ValidationError, match="camera_id"):
        load_manifest(path, base_dir=tmp_path)


def test_clip_id_with_a_path_separator_is_rejected(tmp_path: Path) -> None:
    """Clip ids become directory names, so a separator would escape the run dir."""
    with pytest.raises(ValidationError, match="directory name"):
        ValidationClip(
            clip_id="../escape",
            input_path=Path("a.mp4"),
            scene_type="control",
            camera_id="cam_a",
        )


def test_missing_calibration_file_is_rejected(tmp_path: Path) -> None:
    touch(tmp_path, "a.mp4")
    path = write_manifest(
        tmp_path, [clip_entry("a", "cam_a", "a.mp4", calibration_path="nowhere.json")]
    )

    with pytest.raises(ValidationError, match="calibration file not found"):
        load_manifest(path, base_dir=tmp_path)


def test_calibration_path_is_resolved_when_present(tmp_path: Path) -> None:
    touch(tmp_path, "a.mp4")
    calibration = tmp_path / "cal.json"
    calibration.write_text("{}", encoding="utf-8")
    path = write_manifest(
        tmp_path, [clip_entry("a", "cam_a", "a.mp4", calibration_path="cal.json")]
    )

    manifest = load_manifest(path, base_dir=tmp_path)

    assert manifest.clip("a").calibration_path == calibration
    assert manifest.clip("a").is_calibrated


def test_selecting_restricts_to_named_clips(tmp_path: Path) -> None:
    for name in ("a.mp4", "b.mp4"):
        touch(tmp_path, name)
    path = write_manifest(
        tmp_path, [clip_entry("a", "cam_a", "a.mp4"), clip_entry("b", "cam_b", "b.mp4")]
    )
    manifest = load_manifest(path, base_dir=tmp_path)

    assert [clip.clip_id for clip in manifest.selecting(["b"])] == ["b"]
    assert [clip.clip_id for clip in manifest.selecting([])] == ["a", "b"]


def test_selecting_an_unknown_clip_raises(tmp_path: Path) -> None:
    touch(tmp_path, "a.mp4")
    path = write_manifest(tmp_path, [clip_entry("a", "cam_a", "a.mp4")])
    manifest = load_manifest(path, base_dir=tmp_path)

    with pytest.raises(ValidationError, match="No clip 'nope'"):
        manifest.selecting(["nope"])


def test_empty_manifest_object_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one clip"):
        ValidationManifest(clips=())


def test_repository_manifest_is_loadable() -> None:
    """The committed manifest must parse and name every clip distinctly."""
    repo_root = Path(__file__).resolve().parents[3]
    manifest_path = repo_root / "config" / "validation" / "v0.3.6.json"
    if not manifest_path.exists():  # pragma: no cover - repository layout guard
        pytest.skip("repository manifest not present")

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert raw["manifest_version"] == MANIFEST_VERSION
    clip_ids = [entry["clip_id"] for entry in raw["clips"]]
    assert len(set(clip_ids)) == len(clip_ids)
    # Camera ids are deliberately NOT asserted unique: one camera may supply
    # many clips. Each entry must merely name one.
    assert all(entry["camera_id"].strip() for entry in raw["clips"])
    # No clip may claim a calibration until one is physically measured.
    assert all(entry.get("calibration_path") is None for entry in raw["clips"])

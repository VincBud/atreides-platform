"""Tests for the WILDTRACK runtime frame adapter — no real corpus, no GT identity."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from atreides_vision.errors import WildtrackDatasetError
from atreides_vision.venue_validation.datasets.wildtrack import (
    WILDTRACK_DECLARED_FPS,
    build_wildtrack_stream_specs,
)
from atreides_vision.venue_validation.datasets.wildtrack import (
    runtime as _runtime_module,
)
from atreides_vision.venue_validation.datasets.wildtrack.layout import WildtrackDatasetLayout
from wildtrack_fixtures import write_minimal_dataset

IMG_W, IMG_H = 100, 80


def _real_dataset(
    root: Path,
    *,
    image_frame_values: tuple[int, ...] = (0, 5, 10),
    annotation_frame_values: tuple[int, ...] = (0, 5),
    camera_ids: tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5", "C6", "C7"),
) -> WildtrackDatasetLayout:
    write_minimal_dataset(
        root,
        image_frame_values=image_frame_values,
        annotation_frame_values=annotation_frame_values,
        image_width=IMG_W,
        image_height=IMG_H,
        camera_ids=camera_ids,
    )
    layout = WildtrackDatasetLayout.load(root)
    for camera_id in camera_ids:
        for value in image_frame_values:
            path = root / "Image_subsets" / camera_id / f"{value:08d}.png"
            assert cv2.imwrite(str(path), np.full((IMG_H, IMG_W, 3), value + 1, dtype=np.uint8))
    return layout


def test_declared_fps_is_derived_two_hertz() -> None:
    assert WILDTRACK_DECLARED_FPS == 2.0


def test_specs_are_canonical_camera_order_regardless_of_request(tmp_path: Path) -> None:
    layout = _real_dataset(tmp_path)
    specs = build_wildtrack_stream_specs(layout, cameras=["C5", "C1", "C3"])
    assert [s.camera_id for s in specs] == ["C1", "C3", "C5"]


def test_default_selects_all_seven_cameras(tmp_path: Path) -> None:
    layout = _real_dataset(tmp_path)
    specs = build_wildtrack_stream_specs(layout)
    assert [s.camera_id for s in specs] == ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def test_only_annotation_backed_frames_are_included(tmp_path: Path) -> None:
    layout = _real_dataset(tmp_path, image_frame_values=(0, 5, 10), annotation_frame_values=(0, 5))
    spec = build_wildtrack_stream_specs(layout, cameras=["C1"])[0]
    assert [fid.value for fid in spec.frame_ids] == [0, 5]
    assert len(spec.image_paths) == 2
    assert all(p.name in {"00000000.png", "00000005.png"} for p in spec.image_paths)


def test_trailing_image_without_annotation_is_structurally_excluded(tmp_path: Path) -> None:
    # Mirrors the real corpus: a trailing extracted image (10) with no annotation.
    layout = _real_dataset(tmp_path, image_frame_values=(0, 5, 10), annotation_frame_values=(0, 5))
    spec = build_wildtrack_stream_specs(layout, cameras=["C1"])[0]
    assert 10 not in {fid.value for fid in spec.frame_ids}
    assert all("00000010" not in p.name for p in spec.image_paths)


def test_streams_carry_zero_nominal_offset_and_published_alignment_accuracy(
    tmp_path: Path,
) -> None:
    layout = _real_dataset(tmp_path)
    for spec in build_wildtrack_stream_specs(layout):
        # WILDTRACK frame ids name corresponding samples on one dataset timeline,
        # so the nominal/model offset is exactly zero...
        assert spec.offset_seconds == 0.0
        # ...but the published physical synchronization accuracy (50 ms, then
        # manually refined) is carried as the conservative alignment half-width,
        # never zero.
        assert spec.offset_accuracy_seconds == 0.05
        assert spec.declared_fps == 2.0


def test_missing_required_synchronized_frame_is_rejected(tmp_path: Path) -> None:
    layout = _real_dataset(tmp_path)
    (tmp_path / "Image_subsets" / "C4" / "00000005.png").unlink()
    with pytest.raises(WildtrackDatasetError, match="C4 is missing the image"):
        build_wildtrack_stream_specs(layout, cameras=["C1", "C4"])


def test_unknown_camera_is_rejected(tmp_path: Path) -> None:
    layout = _real_dataset(tmp_path)
    with pytest.raises(WildtrackDatasetError, match="Unknown WILDTRACK camera"):
        build_wildtrack_stream_specs(layout, cameras=["C1", "C9"])


def test_duplicate_camera_is_rejected(tmp_path: Path) -> None:
    layout = _real_dataset(tmp_path)
    with pytest.raises(WildtrackDatasetError, match="more than once"):
        build_wildtrack_stream_specs(layout, cameras=["C1", "C1"])


def test_spec_opens_a_working_image_sequence_source(tmp_path: Path) -> None:
    layout = _real_dataset(tmp_path)
    spec = build_wildtrack_stream_specs(layout, cameras=["C2"])[0]
    source = spec.open()
    assert source.metadata.fps == 2.0
    assert source.metadata.frame_count == 2
    frames = list(source.frames())
    assert [f.shape for f in frames] == [(IMG_H, IMG_W, 3), (IMG_H, IMG_W, 3)]


def test_adapter_does_not_touch_person_level_ground_truth() -> None:
    source_text = Path(_runtime_module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "WildtrackPersonId",
        "load_frame_annotations",
        "cross_camera_correspondences",
        "ground_truth_occupancy",
        "import atreides_vision.venue_validation.datasets.wildtrack.annotations",
        "import atreides_vision.venue_validation.datasets.wildtrack.correspondence",
    ):
        assert forbidden not in source_text

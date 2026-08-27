"""Locating and validating one WILDTRACK dataset root's directory structure."""

from __future__ import annotations

from pathlib import Path

import pytest

from atreides_vision.errors import WildtrackDatasetError
from atreides_vision.venue_validation.datasets.wildtrack.layout import WildtrackDatasetLayout
from atreides_vision.venue_validation.datasets.wildtrack.time import WildtrackFrameId
from wildtrack_fixtures import (
    build_layout,
    png_bytes,
    write_minimal_dataset,
)


class TestValidLayout:
    def test_loads_canonical_mini_layout(self, tmp_path: Path) -> None:
        layout = build_layout(
            tmp_path, image_frame_values=(0, 5, 10), annotation_frame_values=(0, 5)
        )
        assert layout.image_frame_ids == (
            WildtrackFrameId(0),
            WildtrackFrameId(5),
            WildtrackFrameId(10),
        )
        assert layout.annotation_frame_ids == (WildtrackFrameId(0), WildtrackFrameId(5))
        assert layout.image_width == 100
        assert layout.image_height == 80

    def test_annotation_frame_id_with_no_image_frame_is_the_only_disallowed_gap(
        self, tmp_path: Path
    ) -> None:
        # The real dataset has 401 image frames and 400 annotation frames: every
        # annotation frame id must be a subset of image frame ids, but not vice
        # versa (the trailing extracted frame has no annotation). This is the
        # structural relationship, tested without hardcoding either count.
        layout = build_layout(
            tmp_path, image_frame_values=(0, 5, 10, 15), annotation_frame_values=(0, 5, 10)
        )
        assert set(f.value for f in layout.annotation_frame_ids) <= set(
            f.value for f in layout.image_frame_ids
        )
        assert len(layout.image_frame_ids) > len(layout.annotation_frame_ids)


class TestPathBuilding:
    def test_image_annotation_and_calibration_paths(self, tmp_path: Path) -> None:
        layout = build_layout(tmp_path)
        frame = WildtrackFrameId(5)
        assert layout.image_path("C1", frame) == (
            tmp_path / "Image_subsets" / "C1" / "00000005.png"
        )
        assert layout.annotation_path(frame) == tmp_path / "annotations_positions" / "00000005.json"
        assert layout.intrinsic_path("C5") == (
            tmp_path / "calibrations" / "intrinsic_zero" / "intr_IDIAP1.xml"
        )
        assert layout.extrinsic_path("C5") == (
            tmp_path / "calibrations" / "extrinsic" / "extr_IDIAP1.xml"
        )


class TestMissingStructure:
    def test_missing_root(self, tmp_path: Path) -> None:
        with pytest.raises(WildtrackDatasetError):
            WildtrackDatasetLayout.load(tmp_path / "does_not_exist")

    def test_missing_image_subsets(self, tmp_path: Path) -> None:
        write_minimal_dataset(tmp_path)
        import shutil

        shutil.rmtree(tmp_path / "Image_subsets")
        with pytest.raises(WildtrackDatasetError):
            WildtrackDatasetLayout.load(tmp_path)

    def test_missing_camera_directory(self, tmp_path: Path) -> None:
        write_minimal_dataset(tmp_path)
        import shutil

        shutil.rmtree(tmp_path / "Image_subsets" / "C7")
        with pytest.raises(WildtrackDatasetError):
            WildtrackDatasetLayout.load(tmp_path)

    def test_empty_camera_directory(self, tmp_path: Path) -> None:
        write_minimal_dataset(tmp_path)
        for entry in (tmp_path / "Image_subsets" / "C7").iterdir():
            entry.unlink()
        with pytest.raises(WildtrackDatasetError):
            WildtrackDatasetLayout.load(tmp_path)

    def test_mismatched_frame_ids_across_cameras(self, tmp_path: Path) -> None:
        write_minimal_dataset(tmp_path)
        (tmp_path / "Image_subsets" / "C2" / "00000099.png").write_bytes(png_bytes(100, 80))
        with pytest.raises(WildtrackDatasetError):
            WildtrackDatasetLayout.load(tmp_path)

    def test_missing_annotations_dir(self, tmp_path: Path) -> None:
        write_minimal_dataset(tmp_path)
        import shutil

        shutil.rmtree(tmp_path / "annotations_positions")
        with pytest.raises(WildtrackDatasetError):
            WildtrackDatasetLayout.load(tmp_path)

    def test_empty_annotations_dir(self, tmp_path: Path) -> None:
        write_minimal_dataset(tmp_path)
        for entry in (tmp_path / "annotations_positions").iterdir():
            entry.unlink()
        with pytest.raises(WildtrackDatasetError):
            WildtrackDatasetLayout.load(tmp_path)

    def test_annotation_frame_with_no_image_frame_rejected(self, tmp_path: Path) -> None:
        write_minimal_dataset(
            tmp_path, image_frame_values=(0, 5), annotation_frame_values=(0, 5, 10)
        )
        with pytest.raises(WildtrackDatasetError):
            WildtrackDatasetLayout.load(tmp_path)

    def test_missing_intrinsic_calibration(self, tmp_path: Path) -> None:
        write_minimal_dataset(tmp_path)
        (tmp_path / "calibrations" / "intrinsic_zero" / "intr_IDIAP3.xml").unlink()
        with pytest.raises(WildtrackDatasetError):
            WildtrackDatasetLayout.load(tmp_path)

    def test_missing_extrinsic_calibration(self, tmp_path: Path) -> None:
        write_minimal_dataset(tmp_path)
        (tmp_path / "calibrations" / "extrinsic" / "extr_CVLab1.xml").unlink()
        with pytest.raises(WildtrackDatasetError):
            WildtrackDatasetLayout.load(tmp_path)


class TestPngDimensionReading:
    def test_bad_signature_rejected(self, tmp_path: Path) -> None:
        write_minimal_dataset(tmp_path)
        (tmp_path / "Image_subsets" / "C1" / "00000000.png").write_bytes(
            png_bytes(100, 80, valid_signature=False)
        )
        with pytest.raises(WildtrackDatasetError):
            WildtrackDatasetLayout.load(tmp_path)

    def test_short_header_rejected(self, tmp_path: Path) -> None:
        write_minimal_dataset(tmp_path)
        (tmp_path / "Image_subsets" / "C1" / "00000000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        with pytest.raises(WildtrackDatasetError):
            WildtrackDatasetLayout.load(tmp_path)

    def test_missing_ihdr_rejected(self, tmp_path: Path) -> None:
        write_minimal_dataset(tmp_path)
        garbage = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        (tmp_path / "Image_subsets" / "C1" / "00000000.png").write_bytes(garbage)
        with pytest.raises(WildtrackDatasetError):
            WildtrackDatasetLayout.load(tmp_path)

"""V0.4.9b-D2 — the one programmatic entry point that scores a WILDTRACK run.

:func:`run_wildtrack_scoring` joins the real
:func:`build_wildtrack_runtime_session`, the real :func:`run_venue_validation`
and the frozen D1 :class:`WildtrackEvaluator` and writes the three deterministic
artifacts. These tests drive that whole path with only the detector and tracker
seams faked, against a tiny synthetic dataset in WILDTRACK's own on-disk layout.
No real corpus, no weights, no GPU, no network, and no person-level ground truth
is read above the observer seam.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from atreides_vision.config import DetectorConfig
from atreides_vision.errors import WildtrackDatasetError
from atreides_vision.models import Detection
from atreides_vision.venue_validation.datasets.wildtrack.evaluation import (
    WILDTRACK_EVALUATION_REPORT_FILENAME,
    WILDTRACK_EVALUATION_SUMMARY_FILENAME,
    WILDTRACK_FALSE_MERGE_DIAGNOSTICS_FILENAME,
    WildtrackScoringArtifacts,
    run_wildtrack_scoring,
)
from atreides_vision.venue_validation.datasets.wildtrack.layout import WildtrackDatasetLayout
from fakes import FakeDetector, FakeTracker
from wildtrack_fixtures import write_minimal_dataset

ALL_CAMS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")


def _write_pngs(
    root: Path, frame_values: tuple[int, ...], width: int, height: int, cameras: tuple[str, ...]
) -> None:
    for camera_id in cameras:
        for value in frame_values:
            path = root / "Image_subsets" / camera_id / f"{value:08d}.png"
            assert cv2.imwrite(str(path), np.full((height, width, 3), value % 200 + 1, np.uint8))


def _dataset(
    root: Path,
    *,
    image_frame_values: tuple[int, ...] = (0, 5, 10, 15, 20),
    annotation_frame_values: tuple[int, ...] = (0, 5, 10, 15),
    width: int = 120,
    height: int = 90,
) -> WildtrackDatasetLayout:
    write_minimal_dataset(
        root,
        image_frame_values=image_frame_values,
        annotation_frame_values=annotation_frame_values,
        image_width=width,
        image_height=height,
        camera_ids=ALL_CAMS,
    )
    _write_pngs(root, image_frame_values, width, height, ALL_CAMS)
    return WildtrackDatasetLayout.load(root)


_DETECTOR_CONFIG = DetectorConfig()


def _score(
    layout: WildtrackDatasetLayout,
    output_dir: Path,
    *,
    selected_camera_ids: list[str] | None = None,
    instant_limit: int | None = None,
) -> WildtrackScoringArtifacts:
    return run_wildtrack_scoring(
        layout,
        output_dir=output_dir,
        detector=FakeDetector([[Detection.from_xyxy((10.0, 10.0, 40.0, 70.0), 0.9, 0)]]),
        tracker_factory=lambda _config: FakeTracker(),
        detector_config=_DETECTOR_CONFIG,
        selected_camera_ids=selected_camera_ids,
        instant_limit=instant_limit,
        calibrated=False,
    )


class TestArtifacts:
    def test_writes_mandatory_artifacts_and_no_false_merge_file(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        out = tmp_path / "out"
        result = _score(layout, out, selected_camera_ids=["C1", "C2"])

        assert (out / WILDTRACK_EVALUATION_REPORT_FILENAME).is_file()
        assert (out / WILDTRACK_EVALUATION_SUMMARY_FILENAME).is_file()
        assert not (out / WILDTRACK_FALSE_MERGE_DIAGNOSTICS_FILENAME).exists()
        assert result.false_merge_diagnostics_path is None

    def test_summary_counts_every_annotated_instant_in_the_window(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        result = _score(layout, tmp_path / "out", selected_camera_ids=["C1", "C2"], instant_limit=3)
        assert result.evaluation.instant_count == 3
        assert result.evaluation.selected_camera_ids == ("C1", "C2")
        assert result.evaluation.max_ground_plane_distance_m == 0.75

    def test_too_few_cameras_or_frames_still_scores_the_window(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        result = _score(layout, tmp_path / "out", selected_camera_ids=["C1"])
        assert result.evaluation.instant_count == 4


class TestDeterminism:
    def test_two_runs_are_byte_identical(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        for name in ("one", "two"):
            _score(layout, tmp_path / name, selected_camera_ids=["C1", "C2", "C3"])
        for filename in (
            WILDTRACK_EVALUATION_REPORT_FILENAME,
            WILDTRACK_EVALUATION_SUMMARY_FILENAME,
        ):
            assert (tmp_path / "one" / filename).read_bytes() == (
                tmp_path / "two" / filename
            ).read_bytes()

    def test_camera_selection_order_does_not_change_the_artifacts(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        _score(layout, tmp_path / "forward", selected_camera_ids=["C1", "C2", "C3"])
        _score(layout, tmp_path / "reverse", selected_camera_ids=["C3", "C1", "C2"])
        for filename in (
            WILDTRACK_EVALUATION_REPORT_FILENAME,
            WILDTRACK_EVALUATION_SUMMARY_FILENAME,
        ):
            assert (tmp_path / "forward" / filename).read_bytes() == (
                tmp_path / "reverse" / filename
            ).read_bytes()


class TestAlignment:
    def test_one_scored_instant_per_annotated_frame(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        result = _score(layout, tmp_path / "out", selected_camera_ids=["C1", "C2"])
        assert result.evaluation.instant_count == 4

    def test_missing_image_for_an_annotated_frame_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(WildtrackDatasetError):
            layout = _dataset(
                tmp_path, image_frame_values=(0, 5), annotation_frame_values=(0, 5, 10)
            )
            _score(layout, tmp_path / "out", selected_camera_ids=["C1", "C2"])

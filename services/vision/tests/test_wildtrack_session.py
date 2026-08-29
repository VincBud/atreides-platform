"""V0.4.9b-D — the WILDTRACK -> real venue-validation runtime composition layer.

These tests drive :func:`build_wildtrack_runtime_session` through the *real*
:func:`run_venue_validation`, with only the detector and tracker seams faked
(the same seams :mod:`test_venue_validation_runner` fakes). The image sources
are real ``ImageSequenceVideoSource`` instances reading real PNG files, so the
live :class:`~atreides_vision.venue.timebase.VenueTimebase` registration, the
same-sample tick merge and the stage window all exercise the production path.

No real corpus: a tiny synthetic dataset in WILDTRACK's own on-disk layout.
No person-level ground truth is read here — the composition layer never touches
it and neither do these tests.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from atreides_vision.errors import WildtrackDatasetError
from atreides_vision.models import Detection
from atreides_vision.venue.manifest import VenueStreamEntry
from atreides_vision.venue_validation import VenueValidationSummary, run_venue_validation
from atreides_vision.venue_validation.datasets.wildtrack import (
    build_wildtrack_runtime_session,
)
from atreides_vision.venue_validation.datasets.wildtrack.layout import WildtrackDatasetLayout
from atreides_vision.venue_validation.datasets.wildtrack.runtime import (
    WILDTRACK_STREAM_OFFSET_ACCURACY_SECONDS,
)
from atreides_vision.venue_validation.datasets.wildtrack.session import WildtrackRuntimeSession
from fakes import FakeDetector, FakeTracker
from wildtrack_fixtures import write_extrinsic_xml, write_intrinsic_xml, write_minimal_dataset

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
    cameras: tuple[str, ...] = ALL_CAMS,
) -> WildtrackDatasetLayout:
    write_minimal_dataset(
        root,
        image_frame_values=image_frame_values,
        annotation_frame_values=annotation_frame_values,
        image_width=width,
        image_height=height,
        camera_ids=cameras,
    )
    _write_pngs(root, image_frame_values, width, height, cameras)
    return WildtrackDatasetLayout.load(root)


def _one_person_detector() -> FakeDetector:
    return FakeDetector([[Detection.from_xyxy((10.0, 10.0, 40.0, 70.0), 0.9, 0)]])


def _run(session: WildtrackRuntimeSession) -> VenueValidationSummary:
    return run_venue_validation(
        session.config,
        detector=_one_person_detector(),
        tracker_factory=lambda _config: FakeTracker(),
        source_factory=session.source_factory,
    )


class TestScoredFrameSelection:
    def test_full_selection_covers_every_annotated_instant_in_canonical_order(
        self, tmp_path: Path
    ) -> None:
        layout = _dataset(tmp_path)
        session = build_wildtrack_runtime_session(
            layout, output_dir=tmp_path / "out", calibrated=False
        )
        assert session.camera_ids == ALL_CAMS
        assert [fid.value for fid in session.scored_frame_ids] == [0, 5, 10, 15]

    def test_a_contiguous_window_is_a_slice_of_the_canonical_annotation_list(
        self, tmp_path: Path
    ) -> None:
        layout = _dataset(tmp_path)
        session = build_wildtrack_runtime_session(
            layout,
            output_dir=tmp_path / "out",
            start_instant_index=1,
            instant_limit=2,
            calibrated=False,
        )
        assert [fid.value for fid in session.scored_frame_ids] == [5, 10]

    def test_camera_subset_selects_and_canonicalises(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        session = build_wildtrack_runtime_session(
            layout,
            output_dir=tmp_path / "out",
            selected_camera_ids=["C5", "C1", "C3"],
            calibrated=False,
        )
        assert session.camera_ids == ("C1", "C3", "C5")
        assert session.config.manifest.camera_ids == ("C1", "C3", "C5")


class TestLiveTimebaseWiring:
    def test_every_stream_entry_carries_zero_offset_and_005_accuracy(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        session = build_wildtrack_runtime_session(
            layout, output_dir=tmp_path / "out", calibrated=False
        )
        entries = [entry for camera in session.config.manifest.cameras for entry in camera.sources]
        assert len(entries) == 7
        for entry in entries:
            assert entry.start_offset_seconds == 0.0
            assert entry.offset_accuracy_seconds == WILDTRACK_STREAM_OFFSET_ACCURACY_SECONDS
            assert entry.declared_fps == 2.0

    def test_the_live_run_resolves_timing_to_established_005(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        session = build_wildtrack_runtime_session(
            layout, output_dir=tmp_path / "out", selected_camera_ids=["C1", "C2"], calibrated=False
        )
        summary = _run(session)
        assert summary.session_alignment.is_established
        assert summary.session_alignment.half_width_seconds == 0.05

    def test_same_sample_cameras_merge_into_one_venue_instant(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        session = build_wildtrack_runtime_session(
            layout,
            output_dir=tmp_path / "out",
            selected_camera_ids=["C1", "C2", "C3"],
            start_instant_index=0,
            instant_limit=4,
            calibrated=False,
        )
        summary = _run(session)
        # Four synchronized instants, three cameras: one venue tick per instant,
        # not one per (camera, instant).
        assert summary.venue_instants == 4
        assert summary.source_frames_processed == 12


class TestDeterminism:
    def test_camera_selection_order_does_not_change_the_run(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        forward = build_wildtrack_runtime_session(
            layout,
            output_dir=tmp_path / "forward",
            selected_camera_ids=["C1", "C2", "C3"],
            calibrated=False,
        )
        reverse = build_wildtrack_runtime_session(
            layout,
            output_dir=tmp_path / "reverse",
            selected_camera_ids=["C3", "C2", "C1"],
            calibrated=False,
        )
        _run(forward)
        _run(reverse)
        assert forward.scored_frame_ids == reverse.scored_frame_ids
        assert (forward.config.output_dir / "venue_validation_summary.json").read_bytes() == (
            reverse.config.output_dir / "venue_validation_summary.json"
        ).read_bytes()

    def test_two_runs_are_byte_identical(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        for name in ("one", "two"):
            session = build_wildtrack_runtime_session(
                layout,
                output_dir=tmp_path / name,
                selected_camera_ids=["C1", "C2"],
                calibrated=False,
            )
            _run(session)
        assert (tmp_path / "one" / "venue_validation_summary.json").read_bytes() == (
            tmp_path / "two" / "venue_validation_summary.json"
        ).read_bytes()


class TestBoundaries:
    def test_start_index_past_the_last_instant_is_rejected(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        with pytest.raises(WildtrackDatasetError, match="past the last"):
            build_wildtrack_runtime_session(
                layout, output_dir=tmp_path / "out", start_instant_index=4, calibrated=False
            )

    def test_zero_instant_limit_is_rejected(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        with pytest.raises(WildtrackDatasetError, match="positive count"):
            build_wildtrack_runtime_session(
                layout, output_dir=tmp_path / "out", instant_limit=0, calibrated=False
            )

    def test_negative_start_index_is_rejected(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        with pytest.raises(WildtrackDatasetError, match="non-negative"):
            build_wildtrack_runtime_session(
                layout, output_dir=tmp_path / "out", start_instant_index=-1, calibrated=False
            )

    def test_unknown_camera_is_rejected(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        with pytest.raises(WildtrackDatasetError, match="Unknown WILDTRACK camera"):
            build_wildtrack_runtime_session(
                layout,
                output_dir=tmp_path / "out",
                selected_camera_ids=["C1", "C9"],
                calibrated=False,
            )

    def test_source_factory_refuses_a_foreign_manifest_entry(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        session = build_wildtrack_runtime_session(
            layout, output_dir=tmp_path / "out", calibrated=False
        )
        alien = VenueStreamEntry(input_path=tmp_path / "nowhere.png", declared_fps=2.0)
        with pytest.raises(WildtrackDatasetError, match="No WILDTRACK stream spec"):
            session.source_factory(alien)

    def test_window_limit_past_the_end_is_clamped_not_an_error(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        session = build_wildtrack_runtime_session(
            layout,
            output_dir=tmp_path / "out",
            start_instant_index=2,
            instant_limit=999,
            calibrated=False,
        )
        assert [fid.value for fid in session.scored_frame_ids] == [10, 15]


class TestCalibratedComposition:
    def _wide_field_dataset(self, root: Path) -> WildtrackDatasetLayout:
        width, height = 1000, 2000
        layout = _dataset(
            root,
            image_frame_values=(0, 5),
            annotation_frame_values=(0, 5),
            width=width,
            height=height,
        )
        for camera_id in ALL_CAMS:
            write_intrinsic_xml(
                layout.intrinsic_path(camera_id),
                camera_matrix=(400.0, 0.0, 500.0, 0.0, 400.0, 500.0, 0.0, 0.0, 1.0),
            )
            write_extrinsic_xml(
                layout.extrinsic_path(camera_id),
                rvec=(0.0, 0.0, 0.0),
                tvec_cm=(300.0, 900.0, 1000.0),
            )
        return layout

    def test_calibrated_run_builds_one_ground_plane_fit_per_selected_camera(
        self, tmp_path: Path
    ) -> None:
        cameras = ("C1", "C2")
        layout = self._wide_field_dataset(tmp_path)
        session = build_wildtrack_runtime_session(
            layout, output_dir=tmp_path / "out", selected_camera_ids=list(cameras), calibrated=True
        )
        assert tuple(c.camera_id for c in session.config.cameras) == cameras
        for camera_config in session.config.cameras:
            assert camera_config.calibration_fit is not None
            assert camera_config.calibration_fit.camera_id == camera_config.camera_id
            assert camera_config.calibration_path is None

    def test_uncalibrated_composition_configures_no_camera_geometry(self, tmp_path: Path) -> None:
        layout = _dataset(tmp_path)
        session = build_wildtrack_runtime_session(
            layout, output_dir=tmp_path / "out", calibrated=False
        )
        assert session.config.cameras == ()

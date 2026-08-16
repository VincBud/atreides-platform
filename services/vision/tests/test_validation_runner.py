"""The runner and the analysis pass, end to end, with no model weights.

A fake detector and tracker drive the real pipeline over a synthetic MP4, so
these cover the composition the runner actually performs — unsampled output,
all three streams, gzip, provenance — rather than a mock of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from atreides_vision.config import (
    DEFAULT_DETECTION_CONFIDENCE,
    DEFAULT_OCSORT_HIGH_CONF_THRESHOLD,
)
from atreides_vision.detection.base import PersonDetector
from atreides_vision.errors import ValidationError
from atreides_vision.models import BoundingBox, Detection, Track
from atreides_vision.validation.analysis import analyze_clip, analyze_run_directory
from atreides_vision.validation.manifest import ValidationClip, ValidationManifest
from atreides_vision.validation.review_samples import (
    DEFAULT_CONTINUITY_WINDOWS,
    DEFAULT_SAMPLE_FRAMES,
)
from atreides_vision.validation.runner import ValidationRunner
from atreides_vision.validation.storage import ValidationLayout
from atreides_vision.validation.streams import iter_jsonl

FRAME_COUNT = 40
FPS = 20.0
WIDTH, HEIGHT = 160, 120


class WalkingDetector:
    """Two people crossing the frame, close enough to form pairs and a cluster."""

    def __init__(self, subjects: int = 3) -> None:
        self._subjects = subjects
        self.calls = 0

    @property
    def name(self) -> str:
        return "Fake Walking Detector"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        step = self.calls
        self.calls += 1
        detections = []
        for index in range(self._subjects):
            x = 10.0 + step * 1.5 + index * 6.0
            y = 40.0 + index * 2.0
            detections.append(
                Detection(
                    bbox_xyxy=BoundingBox(x, y, x + 12.0, y + 34.0),
                    confidence=0.9,
                    class_id=1,
                )
            )
        return detections


class StableTracker:
    """Assigns ids by detection order, so track identity is analytic."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "Fake Stable Tracker"

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        self.calls += 1
        return [
            Track(track_id=index + 1, bbox_xyxy=det.bbox_xyxy, confidence=det.confidence)
            for index, det in enumerate(detections)
        ]

    def reset(self) -> None:  # pragma: no cover - interface completeness
        pass


class FailingDetector:
    @property
    def name(self) -> str:
        return "Failing Detector"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        from atreides_vision.errors import DetectorError

        raise DetectorError("inference exploded")


@pytest.fixture
def clip_video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    for index in range(FRAME_COUNT):
        writer.write(np.full((HEIGHT, WIDTH, 3), 40 + index, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def clip(clip_video: Path) -> ValidationClip:
    return ValidationClip(
        clip_id="synthetic_01",
        input_path=clip_video,
        scene_type="control",
        camera_id="cam_synthetic_01",
        notes="synthetic",
    )


def build_runner(
    layout: ValidationLayout,
    detector: PersonDetector | None = None,
    max_frames: int | None = None,
    sample_frames: int = DEFAULT_SAMPLE_FRAMES,
    continuity_windows: int = DEFAULT_CONTINUITY_WINDOWS,
) -> ValidationRunner:
    return ValidationRunner(
        layout=layout,
        device="cpu",
        detector_factory=lambda config: detector or WalkingDetector(),
        tracker_factory=lambda config: StableTracker(),
        max_frames=max_frames,
        sample_frames=sample_frames,
        continuity_windows=continuity_windows,
    )


def test_run_writes_every_expected_artifact(tmp_path: Path, clip: ValidationClip) -> None:
    layout = ValidationLayout.at(tmp_path / "out").ensure()

    outcome = build_runner(layout).run_clip(clip)

    run_dir = layout.run_dir(clip.clip_id)
    assert (run_dir / "tracked.mp4").exists()
    assert (run_dir / "behavior.jsonl.gz").exists()
    assert (run_dir / "interaction.jsonl.gz").exists()
    assert (run_dir / "crowd.jsonl.gz").exists()
    assert (run_dir / "run.json").exists()
    assert outcome.performance.frames_processed == FRAME_COUNT


def test_output_is_unsampled_one_record_per_subject_per_frame(
    tmp_path: Path, clip: ValidationClip
) -> None:
    """Every temporal metric depends on this; a sampled stream cannot support them."""
    layout = ValidationLayout.at(tmp_path / "out").ensure()

    outcome = build_runner(layout).run_clip(clip)

    observations = [
        record
        for record in iter_jsonl(outcome.behavior_path)
        if record["record_type"] == "observation"
    ]
    frames = {record["frame_index"] for record in observations}
    assert len(frames) == FRAME_COUNT
    assert len(observations) == FRAME_COUNT * 3
    assert outcome.provenance.output_sample_interval_seconds is None


def test_crowd_stream_carries_a_summary_for_every_frame(
    tmp_path: Path, clip: ValidationClip
) -> None:
    layout = ValidationLayout.at(tmp_path / "out").ensure()

    outcome = build_runner(layout).run_clip(clip)

    summaries = [
        record
        for record in iter_jsonl(outcome.crowd_path)
        if record["record_type"] == "crowd_frame"
    ]
    assert len(summaries) == FRAME_COUNT


def test_provenance_records_the_frozen_defaults(tmp_path: Path, clip: ValidationClip) -> None:
    layout = ValidationLayout.at(tmp_path / "out").ensure()

    provenance = build_runner(layout).run_clip(clip).provenance

    assert provenance.detector_confidence == DEFAULT_DETECTION_CONFIDENCE
    assert provenance.tracker_high_conf_threshold == DEFAULT_OCSORT_HIGH_CONF_THRESHOLD
    assert provenance.tracker_minimum_iou_threshold == 0.30
    assert provenance.tracker_lost_track_buffer == 30
    assert provenance.tracker_minimum_consecutive_frames == 2
    assert provenance.tracker_direction_consistency_weight == 0.2
    assert provenance.tracker_delta_t == 3
    assert provenance.deduplicate_detections is True
    assert provenance.deduplication_rule == "RULE_B3"
    assert provenance.at_v021_defaults is True
    assert provenance.camera_id == "cam_synthetic_01"


def test_an_uncalibrated_clip_measures_in_normalized_units(
    tmp_path: Path, clip: ValidationClip
) -> None:
    """No calibration means no physical claim, anywhere in the output."""
    layout = ValidationLayout.at(tmp_path / "out").ensure()

    outcome = build_runner(layout).run_clip(clip)

    assert outcome.provenance.coordinate_space == "normalized_image"
    assert outcome.provenance.distance_unit == "normalized_frame_unit"
    assert outcome.provenance.is_physical_space is False
    assert outcome.provenance.calibration_path is None

    header = next(iter_jsonl(outcome.behavior_path))
    assert header["is_physical"] is False
    assert header["position_unit"] == "normalized_frame_unit"


def test_review_sample_frames_and_templates_are_generated(
    tmp_path: Path, clip: ValidationClip
) -> None:
    layout = ValidationLayout.at(tmp_path / "out").ensure()
    runner = build_runner(layout, sample_frames=4, continuity_windows=2)

    outcome = runner.run_clip(clip)
    analyze_clip(layout, outcome)

    gt_dir = layout.ground_truth_dir(clip.clip_id)
    assert (gt_dir / "sample_plan.json").exists()
    assert (gt_dir / "contact_sheet.jpg").exists()
    assert (gt_dir / "frame_annotations.csv").exists()
    assert (gt_dir / "continuity_annotations.csv").exists()
    assert len(list((gt_dir / "frames").glob("*.jpg"))) == 4


def test_max_frames_caps_the_run_and_is_recorded(tmp_path: Path, clip: ValidationClip) -> None:
    layout = ValidationLayout.at(tmp_path / "out").ensure()

    outcome = build_runner(layout, max_frames=10).run_clip(clip)

    assert outcome.performance.frames_processed == 10
    assert outcome.performance.source_frames == FRAME_COUNT


def test_a_pipeline_failure_names_the_clip(tmp_path: Path, clip: ValidationClip) -> None:
    layout = ValidationLayout.at(tmp_path / "out").ensure()
    runner = build_runner(layout, detector=FailingDetector())

    with pytest.raises(ValidationError, match="Clip 'synthetic_01' failed to process"):
        runner.run_clip(clip)


def test_a_missing_input_file_fails_clearly(tmp_path: Path) -> None:
    layout = ValidationLayout.at(tmp_path / "out").ensure()
    missing = ValidationClip(
        clip_id="absent",
        input_path=tmp_path / "nope.mp4",
        scene_type="control",
        camera_id="cam_absent",
    )

    with pytest.raises(ValidationError, match="Clip 'absent' failed to process"):
        build_runner(layout).run_clip(missing)


def test_run_manifest_processes_every_clip_in_order(tmp_path: Path, clip_video: Path) -> None:
    layout = ValidationLayout.at(tmp_path / "out").ensure()
    manifest = ValidationManifest(
        clips=(
            ValidationClip(
                clip_id="second",
                input_path=clip_video,
                scene_type="dense",
                camera_id="cam_second",
            ),
            ValidationClip(
                clip_id="first",
                input_path=clip_video,
                scene_type="control",
                camera_id="cam_first",
            ),
        )
    )

    outcomes = build_runner(layout, max_frames=5).run_manifest(manifest)

    assert [outcome.clip.clip_id for outcome in outcomes] == ["second", "first"]


def test_output_directories_are_created(tmp_path: Path, clip: ValidationClip) -> None:
    root = tmp_path / "nested" / "deeper" / "out"
    layout = ValidationLayout.at(root).ensure()

    build_runner(layout, max_frames=5).run_clip(clip)

    assert root.exists()
    assert (root / "runs" / "synthetic_01").is_dir()
    assert (root / "manual-ground-truth").is_dir()


# --- analysis --------------------------------------------------------------


def test_analysis_measures_the_streams_it_just_produced(
    tmp_path: Path, clip: ValidationClip
) -> None:
    layout = ValidationLayout.at(tmp_path / "out").ensure()
    outcome = build_runner(layout).run_clip(clip)

    result = analyze_clip(layout, outcome)

    assert result.tracks.unique_track_count == 3
    assert result.tracks.frames_processed == FRAME_COUNT
    assert result.behavior.total_observations == FRAME_COUNT * 3
    assert result.crowd.frames_with_summary == FRAME_COUNT
    assert result.performance.processed_fps is not None
    assert {stream.name for stream in result.output_volume.streams} == {
        "behavior",
        "interaction",
        "crowd",
    }
    assert all(stream.compressed for stream in result.output_volume.streams)
    assert all(stream.size_bytes > 0 for stream in result.output_volume.streams)


def test_analysis_reports_no_ground_truth_before_anyone_annotates(
    tmp_path: Path, clip: ValidationClip
) -> None:
    layout = ValidationLayout.at(tmp_path / "out").ensure()
    outcome = build_runner(layout).run_clip(clip)

    record = analyze_clip(layout, outcome).to_record()

    assert record["manual_ground_truth"]["manual_ground_truth_status"] == "NOT_ANNOTATED"
    assert "detection_sample" not in record["manual_ground_truth"]
    assert record["proxy_metrics"]["measurement_class"] == "automated_proxy_system_metrics"


def test_report_path_reproduces_the_run_without_reprocessing(
    tmp_path: Path, clip: ValidationClip
) -> None:
    """The whole point of the two-command split: annotate, then regenerate."""
    layout = ValidationLayout.at(tmp_path / "out").ensure()
    outcome = build_runner(layout).run_clip(clip)
    fresh = analyze_clip(layout, outcome)

    reloaded = analyze_run_directory(layout, clip.clip_id)

    assert reloaded.tracks.to_record() == fresh.tracks.to_record()
    assert reloaded.behavior.to_record() == fresh.behavior.to_record()
    assert reloaded.interaction.to_record() == fresh.interaction.to_record()
    assert reloaded.crowd.to_record() == fresh.crowd.to_record()
    assert reloaded.provenance.to_record() == fresh.provenance.to_record()
    assert reloaded.performance.wall_clock_seconds == pytest.approx(
        fresh.performance.wall_clock_seconds, abs=1e-3
    )


def test_annotation_added_after_a_run_appears_in_the_regenerated_result(
    tmp_path: Path, clip: ValidationClip
) -> None:
    layout = ValidationLayout.at(tmp_path / "out").ensure()
    outcome = build_runner(layout, sample_frames=2).run_clip(clip)
    analyze_clip(layout, outcome)

    annotations = layout.ground_truth_dir(clip.clip_id) / "frame_annotations.csv"
    header, first, second = annotations.read_text(encoding="utf-8").strip().split("\n")
    prefix = first.split(",")[:3]
    annotations.write_text(
        "\n".join([header, ",".join([*prefix, "9", "7", "1", "2", "dense corner"]), second]) + "\n",
        encoding="utf-8",
    )

    result = analyze_run_directory(layout, clip.clip_id)

    assert result.ground_truth.status.value == "PARTIALLY_ANNOTATED"
    assert result.ground_truth.detection_recall_sample == pytest.approx(7 / 9)
    assert result.ground_truth.detection_precision_sample == pytest.approx(7 / 8)


def test_reporting_from_an_incomplete_run_record_is_refused(
    tmp_path: Path, clip: ValidationClip
) -> None:
    layout = ValidationLayout.at(tmp_path / "out").ensure()
    build_runner(layout, max_frames=5).run_clip(clip)
    run_json = layout.run_dir(clip.clip_id) / "run.json"
    record = json.loads(run_json.read_text(encoding="utf-8"))
    del record["provenance"]
    run_json.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValidationError, match="missing its 'provenance' section"):
        analyze_run_directory(layout, clip.clip_id)


def test_reporting_from_a_missing_stream_is_refused(tmp_path: Path, clip: ValidationClip) -> None:
    layout = ValidationLayout.at(tmp_path / "out").ensure()
    build_runner(layout, max_frames=5).run_clip(clip)
    (layout.run_dir(clip.clip_id) / "crowd.jsonl.gz").unlink()

    with pytest.raises(ValidationError, match="crowd stream"):
        analyze_run_directory(layout, clip.clip_id)


def test_run_json_is_deterministic_json(tmp_path: Path, clip: ValidationClip) -> None:
    layout = ValidationLayout.at(tmp_path / "out").ensure()
    build_runner(layout, max_frames=5).run_clip(clip)

    text = (layout.run_dir(clip.clip_id) / "run.json").read_text(encoding="utf-8")

    assert text.endswith("\n")
    record = json.loads(text)
    assert list(record) == sorted(record)


# --- clip isolation under a shared camera id --------------------------------
#
# One physical camera legitimately supplies many independent clips, so a
# camera_id may repeat across a manifest. These cover the guarantee that buys:
# sharing a camera id relates the footage, never the tracks.


def shared_camera_clips(tmp_path: Path, clip_video: Path) -> tuple[ValidationClip, ValidationClip]:
    """Two distinct clips recorded by one camera, `camera_01`."""
    return (
        ValidationClip(
            clip_id="clip_A",
            input_path=clip_video,
            scene_type="control",
            camera_id="camera_01",
        ),
        ValidationClip(
            clip_id="clip_B",
            input_path=clip_video,
            scene_type="control",
            camera_id="camera_01",
        ),
    )


def test_same_camera_and_track_id_in_two_clips_stay_independent_tracklets(
    tmp_path: Path, clip_video: Path
) -> None:
    """camera_01/clip_A/track_7 and camera_01/clip_B/track_7 are two tracklets.

    The regression this milestone must not break: a shared camera_id must never
    become a shared identity. The fake tracker assigns the same ids in both
    clips, so track 7 genuinely exists in each — and must stay two things.
    """
    layout = ValidationLayout.at(tmp_path / "out").ensure()
    clip_a, clip_b = shared_camera_clips(tmp_path, clip_video)
    runner = build_runner(layout)

    # Different frame counts, so an identity leak would be visible as a wrong
    # observation count rather than as two coincidentally equal numbers.
    result_a = analyze_clip(layout, runner.run_clip(clip_a))
    result_b = analyze_clip(layout, build_runner(layout, max_frames=20).run_clip(clip_b))

    # Both clips really do contain a camera_01 track under the same id.
    for outcome_path, expected_frames in (
        (layout.run_dir("clip_A") / "behavior.jsonl.gz", FRAME_COUNT),
        (layout.run_dir("clip_B") / "behavior.jsonl.gz", 20),
    ):
        observations = [
            record
            for record in iter_jsonl(outcome_path)
            if record.get("record_type") == "observation"
            and record["camera_id"] == "camera_01"
            and record["subject"]["track_id"] == 1
        ]
        assert observations, f"expected camera_01/track_1 in {outcome_path}"
        assert len(observations) == expected_frames

    # And they are measured entirely separately.
    assert result_a.clip.clip_id != result_b.clip.clip_id
    assert result_a.performance.frames_processed == FRAME_COUNT
    assert result_b.performance.frames_processed == 20
    assert result_a.behavior.total_observations != result_b.behavior.total_observations
    # The per-clip presence tables are separate objects with separate contents.
    assert result_a.tracks is not result_b.tracks
    assert result_a.tracks.total_observations != result_b.tracks.total_observations


def test_shared_camera_clips_write_to_separate_run_directories(
    tmp_path: Path, clip_video: Path
) -> None:
    """Artifacts are keyed on clip_id, so one clip cannot overwrite the other."""
    layout = ValidationLayout.at(tmp_path / "out").ensure()
    clip_a, clip_b = shared_camera_clips(tmp_path, clip_video)
    runner = build_runner(layout, max_frames=10)

    outcome_a = runner.run_clip(clip_a)
    outcome_b = runner.run_clip(clip_b)

    assert outcome_a.behavior_path != outcome_b.behavior_path
    assert outcome_a.rendered_video_path != outcome_b.rendered_video_path
    for name in ("behavior.jsonl.gz", "interaction.jsonl.gz", "crowd.jsonl.gz", "run.json"):
        assert (layout.run_dir("clip_A") / name).exists()
        assert (layout.run_dir("clip_B") / name).exists()
    # Review material is per clip too, and the analysis pass adds the templates.
    for clip_id in ("clip_A", "clip_B"):
        assert (layout.ground_truth_dir(clip_id) / "sample_plan.json").exists()
        assert list((layout.ground_truth_dir(clip_id) / "frames").glob("*.jpg"))
    analyze_clip(layout, outcome_a)
    analyze_clip(layout, outcome_b)
    for clip_id in ("clip_A", "clip_B"):
        assert (layout.ground_truth_dir(clip_id) / "frame_annotations.csv").exists()


def test_processing_order_does_not_affect_a_clip_s_metrics(
    tmp_path: Path, clip_video: Path
) -> None:
    """No analyzer, tracker or accumulator state survives a clip boundary.

    clip_B measured after clip_A on one runner must equal clip_B measured alone.
    Any carried-over history — track ids, behaviour windows, pair or cluster
    state — would show up here as a difference.
    """
    clip_a, clip_b = shared_camera_clips(tmp_path, clip_video)

    together = ValidationLayout.at(tmp_path / "together").ensure()
    runner = build_runner(together, max_frames=15)
    runner.run_clip(clip_a)
    after_a = analyze_clip(together, runner.run_clip(clip_b))

    alone = ValidationLayout.at(tmp_path / "alone").ensure()
    solo = analyze_clip(alone, build_runner(alone, max_frames=15).run_clip(clip_b))

    assert after_a.tracks.to_record() == solo.tracks.to_record()
    assert after_a.behavior.to_record() == solo.behavior.to_record()
    assert after_a.interaction.to_record() == solo.interaction.to_record()
    assert after_a.crowd.to_record() == solo.crowd.to_record()


def test_manifest_run_accepts_clips_sharing_a_camera(tmp_path: Path, clip_video: Path) -> None:
    layout = ValidationLayout.at(tmp_path / "out").ensure()
    clip_a, clip_b = shared_camera_clips(tmp_path, clip_video)
    manifest = ValidationManifest(clips=(clip_a, clip_b))

    outcomes = build_runner(layout, max_frames=8).run_manifest(manifest)

    assert [outcome.clip.clip_id for outcome in outcomes] == ["clip_A", "clip_B"]
    assert {outcome.clip.camera_id for outcome in outcomes} == {"camera_01"}

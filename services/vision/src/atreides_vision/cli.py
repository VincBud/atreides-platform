"""Command-line entry point for Atreides Vision.

This module is the composition root: the one place that picks concrete
implementations (RF-DETR Nano, OC-SORT or ByteTrack) and wires them into the
pipeline. Everything below it depends only on Atreides interfaces.

A run with no tuning flags resolves to the V0.2.1 validated candidate — RF-DETR
at 0.30, OC-SORT with a 0.30 high-confidence threshold, and Rule-B3
deduplication on. The startup banner states which profile is in effect.

    uv run atreides-vision run --input data/input/test.mp4 --output artifacts/tracked.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import cv2

from atreides_vision import __version__
from atreides_vision.behavior.analyzer import BehaviorAnalyzer
from atreides_vision.behavior.config import BehaviorAnalysisConfig
from atreides_vision.behavior.coordinates import NormalizedFrameProjector
from atreides_vision.behavior.reporting import (
    BehaviorEmissionPolicy,
    BehaviorRunMetadata,
    CalibrationProvenance,
    EmitEveryFrame,
    IntervalEmissionPolicy,
    JsonlBehaviorWriter,
    SourceProvenance,
)
from atreides_vision.calibration.fitting import CalibrationFit
from atreides_vision.calibration.integrity import (
    CalibrationIntegrityConfig,
    CalibrationIntegrityGuard,
    CalibrationIntegrityMonitor,
    IntegrityAssessment,
    IntegrityPolicyMode,
)
from atreides_vision.calibration.loader import load_calibration_fit
from atreides_vision.calibration.projector import GroundPlaneProjector
from atreides_vision.config import (
    DEFAULT_DEDUPLICATE_DETECTIONS,
    DEFAULT_DETECTION_CONFIDENCE,
    DEFAULT_OCSORT_HIGH_CONF_THRESHOLD,
    DEFAULT_TRACKER,
    DetectorConfig,
    Device,
    PipelineConfig,
    Tracker,
    TrackerConfig,
    VisionConfig,
    VisualizationConfig,
    ZoneAnalysisConfig,
    resolve_camera_id,
    resolve_device,
)
from atreides_vision.detection.rfdetr import RFDetrPersonDetector
from atreides_vision.errors import AtreidesVisionError, ConfigurationError
from atreides_vision.interaction.analyzer import InteractionAnalyzer
from atreides_vision.interaction.config import InteractionAnalysisConfig
from atreides_vision.interaction.models import SpatialContext
from atreides_vision.interaction.reporting import (
    InteractionRunMetadata,
    JsonlInteractionWriter,
)
from atreides_vision.tracking.base import MultiObjectTracker
from atreides_vision.tracking.bytetrack import ByteTrackTracker
from atreides_vision.tracking.diagnostics import JsonlTrackingDebugWriter
from atreides_vision.tracking.ocsort import OcSortTracker
from atreides_vision.video.processor import ProcessingResult, VideoProcessor
from atreides_vision.video.sink import Mp4VideoSink
from atreides_vision.video.source import FileVideoSource, VideoMetadata, read_reference_frame
from atreides_vision.visualization.calibration_overlay import (
    CalibrationOverlayConfig,
    render_calibration_overlay,
    render_integrity_overlay,
)
from atreides_vision.zones.analyzer import ZoneAnalyzer
from atreides_vision.zones.loader import load_zone_layout
from atreides_vision.zones.models import ZoneLayout
from atreides_vision.zones.reporting import JsonlMetricsWriter

logger = logging.getLogger("atreides_vision")

__all__ = ["build_parser", "build_tracker", "main"]

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    """Build the ``atreides-vision`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="atreides-vision",
        description=(
            "Atreides Vision V0.1 — SEE + TRACK. Detects and tracks people in a local "
            "video file and writes an annotated copy. No identification, no recognition, "
            "no cloud."
        ),
    )
    parser.add_argument("--version", action="version", version=f"atreides-vision {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run",
        help="Process a video file into an annotated video file.",
        description="Read an MP4, detect and track people, write an annotated MP4.",
    )

    run_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to the input video (e.g. data/input/test.mp4).",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path for the annotated output MP4 (e.g. artifacts/tracked.mp4).",
    )
    run_parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_DETECTION_CONFIDENCE,
        metavar="FLOAT",
        help=(
            f"Minimum detection confidence, in (0, 1]. "
            f"Default: {DEFAULT_DETECTION_CONFIDENCE:.2f} (V0.2.1 validated)."
        ),
    )
    run_parser.add_argument(
        "--device",
        type=Device,
        choices=list(Device),
        default=Device.AUTO,
        help=(
            "Compute device: auto, cpu, mps (Apple Silicon) or cuda. "
            "auto prefers cuda, then mps, then cpu. Default: auto."
        ),
    )
    run_parser.add_argument(
        "--tracker",
        type=Tracker,
        choices=list(Tracker),
        default=DEFAULT_TRACKER,
        help=(
            "Multi-object tracker: ocsort (default, V0.2.1 validated) or bytetrack. "
            "OC-SORT scores association on direction consistency as well as IoU, "
            "which holds ids through short occlusions and large scale changes. "
            "ByteTrack remains fully supported."
        ),
    )
    run_parser.add_argument(
        "--ocsort-high-conf-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Override OC-SORT's high-confidence detection threshold for this run "
            "only, in [0, 1]. Detections below it cannot start a new track. Omit "
            f"to keep the default of {DEFAULT_OCSORT_HIGH_CONF_THRESHOLD:.2f}. "
            "Ignored by ByteTrack."
        ),
    )
    run_parser.add_argument(
        "--deduplicate-detections",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DEDUPLICATE_DETECTIONS,
        help=(
            "Suppress duplicate person detections before tracking using the fixed "
            "conservative geometry rule B3 (IoU >= 0.85, area similarity >= 0.95, "
            "centre distance <= 0.03). Single-frame box geometry only — no identity, "
            "appearance or history. On by default since V0.2.1; pass "
            "--no-deduplicate-detections to restore pre-V0.2.1 behaviour."
        ),
    )
    run_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N frames. Useful for a quick check on a long video.",
    )
    run_parser.add_argument(
        "--zones",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Zone configuration JSON (e.g. config/zones.example.json). "
            "Omit it to run plain detection and tracking."
        ),
    )
    run_parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write zone metrics as JSON Lines (e.g. artifacts/zone_metrics.jsonl). "
            "Requires --zones."
        ),
    )
    run_parser.add_argument(
        "--tracking-debug-output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Diagnostics only. Write per-frame tracker internals as JSON Lines "
            "(detections, reported tracks, and each internal tracklet's age, "
            "time since update, last observation and predicted box). "
            "Observational: it does not change tracking behaviour."
        ),
    )
    run_parser.add_argument(
        "--metrics-interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Seconds of source time between metric records. Default: 1.0.",
    )
    run_parser.add_argument(
        "--behavior-output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write per-subject behaviour signals as JSON Lines "
            "(e.g. artifacts/behavior.jsonl). This flag is what enables behaviour "
            "analysis at all; without it the run is unchanged. Neutral measurements "
            "only — speed, acceleration, turn rate, time spent stationary."
        ),
    )
    run_parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Ground-plane calibration JSON (e.g. config/calibration.example.json). "
            "Switches behaviour measurement from normalized image units to venue "
            "metres. Requires --behavior-output. An invalid calibration is an error, "
            "never a silent fallback to normalized units."
        ),
    )
    run_parser.add_argument(
        "--interaction-output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write pairwise interaction measurements as JSON Lines "
            "(e.g. artifacts/interactions.jsonl). Enables behaviour analysis "
            "internally, so --behavior-output is not required. Neutral geometry "
            "only — separation, closing speed, relative direction of travel, time "
            "spent close. Pairs are same-camera only."
        ),
    )
    run_parser.add_argument(
        "--calibration-reference",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Image (or video whose first frame is used) captured from the camera geometry "
            "this calibration was authored against. Enables calibration-integrity "
            "monitoring: if the camera view stops matching it, metric observations stop "
            "rather than continuing from a calibration that no longer applies. "
            "Requires --calibration."
        ),
    )
    run_parser.add_argument(
        "--integrity-report-only",
        action="store_true",
        help=(
            "Record calibration-integrity findings but keep emitting metric observations "
            "even after drift is detected. Diagnostic use only — the output may be wrong. "
            "Requires --calibration-reference."
        ),
    )
    run_parser.add_argument(
        "--behavior-output-interval",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Emit at most one behaviour snapshot per this many seconds of source time. "
            "Analysis still runs on every frame, so signals are unaffected; this only "
            "reduces how much is written. Omit to write every frame, as before."
        ),
    )
    run_parser.add_argument(
        "--camera-id",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "Camera identity that behaviour records are namespaced by. Defaults to the "
            "camera named by --calibration or --zones; required when neither is given. "
            "A value conflicting with either of those is an error."
        ),
    )
    run_parser.add_argument(
        "--trail-length",
        type=int,
        default=25,
        metavar="N",
        help="Number of recent centre points drawn per track. Default: 25.",
    )
    run_parser.add_argument(
        "--no-confidence-labels",
        action="store_true",
        help="Draw only 'ID <n>' labels, without confidence scores.",
    )
    run_parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        metavar="N",
        help="Emit a progress line every N processed frames. Default: 100.",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    run_parser.set_defaults(handler=_run_command)
    _add_calibration_commands(subparsers)
    return parser


def _add_calibration_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Attach the ``calibration`` inspection commands.

    Inspection only: these read a calibration and report or draw it. There is
    no editor, no interactive point picking and no automatic calibration.
    """
    calibration_parser = subparsers.add_parser(
        "calibration",
        help="Inspect a ground-plane calibration.",
        description=(
            "Check a calibration before trusting metres from it. "
            "'validate' reports the fit numerically; 'render' draws it on a frame."
        ),
    )
    calibration_sub = calibration_parser.add_subparsers(dest="calibration_command", required=True)

    validate_parser = calibration_sub.add_parser(
        "validate",
        help="Solve a calibration and report its fit.",
        description=(
            "Load a calibration, solve the homography, and report the correspondences "
            "and reprojection error. Reprojection error measures how well the fit "
            "reproduces the operator's own reference points — it is not a measurement "
            "of real-world accuracy."
        ),
    )
    validate_parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        metavar="PATH",
        help="Calibration JSON to inspect.",
    )
    validate_parser.set_defaults(handler=_calibration_validate_command)

    render_parser = calibration_sub.add_parser(
        "render",
        help="Draw a calibration onto a frame for visual inspection.",
        description=(
            "Draw the venue grid, axes and reference points onto a still, so the "
            "geometry can be checked by eye against the actual floor. This is the "
            "only check that can catch reference points measured against the wrong "
            "spot, which reprojection error cannot detect."
        ),
    )
    render_parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        metavar="PATH",
        help="Calibration JSON to draw.",
    )
    render_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="PATH",
        help="Image, or video whose first frame is used.",
    )
    render_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="PATH",
        help="Destination image (e.g. artifacts/calibration.png).",
    )
    render_parser.add_argument(
        "--grid-spacing",
        type=float,
        default=1.0,
        metavar="METRES",
        help="Venue grid spacing in metres. Default: 1.0.",
    )
    render_parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Omit the venue grid, leaving reference points and axes.",
    )
    render_parser.set_defaults(handler=_calibration_render_command)

    integrity_parser = calibration_sub.add_parser(
        "integrity",
        help="Check whether a camera's view still matches its calibration reference.",
        description=(
            "Register a current frame against the reference frame the calibration was "
            "authored against, and report whether the camera geometry still matches. "
            "This checks that the view has not moved. It cannot check whether the "
            "calibration's metres were ever correct."
        ),
    )
    integrity_parser.add_argument(
        "--calibration", type=Path, required=True, metavar="PATH", help="Calibration JSON."
    )
    integrity_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        metavar="PATH",
        help="Reference image or video from the calibrated camera geometry.",
    )
    integrity_parser.add_argument(
        "--current",
        type=Path,
        required=True,
        metavar="PATH",
        help="Current image, or video whose first frame is used.",
    )
    integrity_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional annotated image showing matched scene features and the drift.",
    )
    integrity_parser.set_defaults(handler=_calibration_integrity_command)


def build_tracker(choice: Tracker, config: TrackerConfig) -> MultiObjectTracker:
    """Construct the selected tracker adapter.

    The one place a tracker implementation is chosen. Everything downstream sees
    only the :class:`MultiObjectTracker` interface, which is what makes an A/B
    swap a single argument rather than a code change.
    """
    if choice is Tracker.OCSORT:
        return OcSortTracker(config)
    return ByteTrackTracker(config)


def _build_tracker_config(
    frame_rate: float, ocsort_high_conf_threshold: float | None = None
) -> TrackerConfig:
    """Tracker settings for one run.

    Everything comes from :class:`TrackerConfig`'s defaults except the frame
    rate, which follows the source. ``ocsort_high_conf_threshold`` is an
    experimental per-run override for OC-SORT's high-confidence gate; ``None``
    leaves the default in place, so omitting the flag changes nothing.
    """
    config = TrackerConfig(frame_rate=frame_rate)
    if ocsort_high_conf_threshold is None:
        return config
    return replace(config, high_conf_det_threshold=ocsort_high_conf_threshold)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _build_behavior_analyzer(
    camera_id: str,
    metadata: VideoMetadata,
    calibration_fit: CalibrationFit | None,
) -> BehaviorAnalyzer:
    """Choose the projector and the matching threshold set.

    The two travel together and must never be mixed: normalized distance
    thresholds interpreted as metres would mean nobody is ever stationary, and
    the reverse would mean everybody always is. ``intended_space`` on the config
    makes the analyzer enforce the pairing rather than leaving it to this
    function being correct.
    """
    if calibration_fit is None:
        return BehaviorAnalyzer(
            camera_id=camera_id,
            projector=NormalizedFrameProjector(metadata.width, metadata.height),
            config=BehaviorAnalysisConfig(),
        )

    # A calibration that cannot be solved has already raised by this point;
    # there is deliberately no fallback to normalized units, which would keep
    # the output looking valid while silently changing what it means.
    return BehaviorAnalyzer(
        camera_id=camera_id,
        projector=GroundPlaneProjector(
            calibration_fit, frame_width=metadata.width, frame_height=metadata.height
        ),
        config=BehaviorAnalysisConfig.for_ground_plane_metres(),
    )


def _build_integrity_guard(
    reference_path: Path | None,
    report_only: bool,
) -> CalibrationIntegrityGuard | None:
    """Load the integrity reference and build the guard, or return ``None``.

    Failing here — a missing or unreadable reference — stops the run before the
    detector loads, rather than leaving monitoring quietly disabled on a run the
    operator asked to have guarded.
    """
    if reference_path is None:
        return None

    reference = read_reference_frame(reference_path)
    config = CalibrationIntegrityConfig(
        policy_mode=(
            IntegrityPolicyMode.REPORT_ONLY if report_only else IntegrityPolicyMode.GUARDED
        )
    )
    monitor = CalibrationIntegrityMonitor(
        reference_frame=reference,
        config=config,
        reference_id=str(reference_path),
    )
    return CalibrationIntegrityGuard(monitor, config)


def _build_emission_policy(interval_seconds: float | None) -> BehaviorEmissionPolicy:
    """Every frame by default; on an interval when one is configured."""
    if interval_seconds is None:
        return EmitEveryFrame()
    return IntervalEmissionPolicy(interval_seconds)


def _build_interaction_analyzer(
    camera_id: str,
    analyzer: BehaviorAnalyzer,
    calibration_fit: CalibrationFit | None,
) -> InteractionAnalyzer:
    """Build the pairwise analyzer for the same space the behaviour layer uses.

    The context is taken from the behaviour analyzer rather than re-derived, so
    the two layers cannot disagree about which space, venue or surface they are
    measuring in.
    """
    if calibration_fit is None:
        context = SpatialContext(camera_id=camera_id, space=analyzer.space)
        config = InteractionAnalysisConfig.for_normalized_image()
    else:
        calibration = calibration_fit.calibration
        context = SpatialContext(
            camera_id=camera_id,
            space=analyzer.space,
            venue_id=calibration.venue_id,
            surface_id=calibration.surface_id,
        )
        config = InteractionAnalysisConfig.for_ground_plane_metres()
    return InteractionAnalyzer(context, config)


def _build_interaction_metadata(
    analyzer: InteractionAnalyzer,
) -> InteractionRunMetadata:
    """The interaction stream's header, from the analyzer's own context."""
    context = analyzer.context
    return InteractionRunMetadata(
        camera_id=context.camera_id,
        space=context.space,
        venue_id=context.venue_id,
        surface_id=context.surface_id,
        proximity_threshold=analyzer.config.proximity_threshold,
    )


def _build_run_metadata(
    camera_id: str,
    analyzer: BehaviorAnalyzer,
    input_path: Path,
    metadata: VideoMetadata,
    calibration_fit: CalibrationFit | None,
) -> BehaviorRunMetadata:
    """Assemble the header record, including calibration provenance.

    Provenance is flattened here, in the composition root, rather than by the
    behaviour layer reaching into the calibration package — that direction of
    dependency is what keeps behaviour analysis usable with no calibration.
    """
    provenance = None
    if calibration_fit is not None:
        quality = calibration_fit.quality
        calibration = calibration_fit.calibration
        provenance = CalibrationProvenance(
            venue_id=calibration.venue_id,
            surface_id=calibration.surface_id,
            reference_width=calibration.reference_frame.width,
            reference_height=calibration.reference_frame.height,
            correspondence_count=quality.point_count,
            mean_venue_error_m=quality.mean_venue_error_m,
            max_venue_error_m=quality.max_venue_error_m,
            mean_image_error_px=quality.mean_image_error_px,
            max_image_error_px=quality.max_image_error_px,
            relative_error=quality.relative_error,
            venue_extent_m=quality.venue_extent_m,
        )

    return BehaviorRunMetadata(
        camera_id=camera_id,
        space=analyzer.space,
        source=SourceProvenance(
            path=str(input_path),
            width=metadata.width,
            height=metadata.height,
            fps=metadata.fps,
        ),
        calibration=provenance,
    )


def _run_command(args: argparse.Namespace) -> int:
    _configure_logging(args.verbose)

    if args.metrics_output is not None and args.zones is None:
        raise ConfigurationError(
            "--metrics-output needs zones to measure. Add --zones config/zones.example.json, "
            "or drop --metrics-output."
        )
    if args.calibration_reference is not None and args.calibration is None:
        raise ConfigurationError(
            "--calibration-reference checks a calibration's camera geometry, so it needs "
            "--calibration. Integrity monitoring does not apply to normalized behaviour "
            "output, which makes no metric claim to invalidate."
        )
    if args.integrity_report_only and args.calibration_reference is None:
        raise ConfigurationError(
            "--integrity-report-only changes what happens when drift is detected, so it "
            "needs --calibration-reference to detect any."
        )
    if (
        args.calibration is not None
        and args.behavior_output is None
        and args.interaction_output is None
    ):
        raise ConfigurationError(
            "--calibration only affects behaviour and interaction measurement, so it needs "
            "--behavior-output or --interaction-output. Use 'atreides-vision calibration "
            "validate' to check a calibration on its own."
        )

    pipeline_config = PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        max_frames=args.max_frames,
        log_every_n_frames=args.log_every,
        deduplicate_detections=args.deduplicate_detections,
    )
    device = resolve_device(args.device)

    # Fail on a bad zone or calibration file before loading a detector or
    # touching the video. Loading the calibration also solves it, so unusable
    # geometry stops the run here rather than a thousand frames in.
    layout = None if args.zones is None else load_zone_layout(args.zones)
    calibration_fit = None if args.calibration is None else load_calibration_fit(args.calibration)

    # Fails before the detector loads if the reference is missing or unreadable.
    integrity_guard = _build_integrity_guard(args.calibration_reference, args.integrity_report_only)

    # Interaction measurement is built on behaviour snapshots, so asking for it
    # turns the behaviour analyzer on. Writing behaviour JSONL stays a separate
    # choice: nobody should have to produce one file to get another.
    behavior_enabled = args.behavior_output is not None or args.interaction_output is not None
    camera_id = (
        resolve_camera_id(
            explicit=args.camera_id,
            calibration_camera_id=None if calibration_fit is None else calibration_fit.camera_id,
            zone_camera_id=None if layout is None else layout.camera_id,
        )
        if behavior_enabled
        else None
    )

    with FileVideoSource(pipeline_config.input_path) as source:
        metadata = source.metadata

        config = VisionConfig(
            pipeline=pipeline_config,
            detector=DetectorConfig(confidence_threshold=args.confidence, device=device),
            # The tracker's motion model is frame-rate sensitive, so it follows the source.
            tracker=_build_tracker_config(metadata.fps, args.ocsort_high_conf_threshold),
            visualization=VisualizationConfig(
                trail_length=args.trail_length,
                draw_confidence=not args.no_confidence_labels,
            ),
            zones=ZoneAnalysisConfig(
                metrics_sample_interval_seconds=args.metrics_interval,
            ),
        )

        detector = RFDetrPersonDetector(config.detector)
        tracker = build_tracker(args.tracker, config.tracker)
        analyzer = None if layout is None else ZoneAnalyzer(layout, config.zones)
        behavior_analyzer = (
            None
            if camera_id is None
            else _build_behavior_analyzer(camera_id, metadata, calibration_fit)
        )
        interaction_analyzer = (
            None
            if behavior_analyzer is None or camera_id is None or args.interaction_output is None
            else _build_interaction_analyzer(camera_id, behavior_analyzer, calibration_fit)
        )

        _log_startup(config, metadata, detector.name, tracker.name, layout, args.zones)
        _log_behavior_startup(behavior_analyzer, calibration_fit, args.behavior_output)
        _log_integrity_startup(integrity_guard, args.calibration_reference)
        _log_interaction_startup(interaction_analyzer, args.interaction_output)
        if args.behavior_output_interval is not None:
            logger.info(
                "  emission:  one snapshot per %.2f s of source time "
                "(analysis still runs on every frame)",
                args.behavior_output_interval,
            )

        logger.info("Loading detector weights (first run may download them)...")
        detector.load()
        logger.info("Detector ready.")

        with ExitStack() as stack:
            sink = stack.enter_context(
                Mp4VideoSink(
                    path=config.pipeline.output_path,
                    width=metadata.width,
                    height=metadata.height,
                    fps=metadata.fps,
                )
            )
            metrics_writer = (
                None
                if args.metrics_output is None
                else stack.enter_context(JsonlMetricsWriter(args.metrics_output))
            )
            tracking_debug_writer = (
                None
                if args.tracking_debug_output is None
                else stack.enter_context(JsonlTrackingDebugWriter(args.tracking_debug_output))
            )
            behavior_writer = (
                None
                if behavior_analyzer is None or camera_id is None
                else stack.enter_context(
                    JsonlBehaviorWriter(
                        args.behavior_output,
                        _build_run_metadata(
                            camera_id,
                            behavior_analyzer,
                            pipeline_config.input_path,
                            metadata,
                            calibration_fit,
                        ),
                        emission_policy=_build_emission_policy(args.behavior_output_interval),
                    )
                )
            )

            interaction_writer = (
                None
                if interaction_analyzer is None
                else stack.enter_context(
                    JsonlInteractionWriter(
                        args.interaction_output,
                        _build_interaction_metadata(interaction_analyzer),
                        emission_policy=_build_emission_policy(args.behavior_output_interval),
                    )
                )
            )

            processor = VideoProcessor(
                detector=detector,
                tracker=tracker,
                config=config,
                zone_analyzer=analyzer,
                metrics_writer=metrics_writer,
                tracking_debug_writer=tracking_debug_writer,
                behavior_analyzer=behavior_analyzer,
                behavior_writer=behavior_writer,
                integrity_guard=integrity_guard,
                interaction_analyzer=interaction_analyzer,
                interaction_writer=interaction_writer,
            )
            result = processor.run(source, sink)

    _log_completion(result, analyzer, args.metrics_output)
    _log_behavior_completion(behavior_analyzer, args.behavior_output, result)
    _log_integrity_completion(integrity_guard, result)
    _log_interaction_completion(interaction_analyzer, args.interaction_output, result)
    return EXIT_OK


def _log_behavior_startup(
    analyzer: BehaviorAnalyzer | None,
    calibration_fit: CalibrationFit | None,
    output_path: Path | None,
) -> None:
    """State plainly which coordinate space the run will measure in."""
    if analyzer is None:
        logger.info("Behavior:    disabled (pass --behavior-output to measure signals)")
        return

    logger.info("Behavior:    %s -> %s", analyzer.camera_id, output_path)
    if calibration_fit is None:
        logger.info(
            "  space:     normalized image units (camera-local; NOT metres, "
            "not comparable between cameras)"
        )
        return

    quality = calibration_fit.quality
    logger.info(
        "  space:     venue metres (venue '%s', surface '%s')",
        calibration_fit.calibration.venue_id,
        calibration_fit.calibration.surface_id,
    )
    logger.info(
        "  fit:       %d points, mean reprojection error %.3f m (%.2f%% of a %.1f m region)",
        quality.point_count,
        quality.mean_venue_error_m,
        quality.relative_error * 100.0,
        quality.venue_extent_m,
    )


def _log_behavior_completion(
    analyzer: BehaviorAnalyzer | None,
    output_path: Path | None,
    result: ProcessingResult,
) -> None:
    if analyzer is None:
        return
    logger.info(
        "Behavior signals:    %d -> %s",
        result.behavior_signals_emitted,
        output_path,
    )
    if analyzer.unprojectable_observations:
        logger.info(
            "  skipped:   %d observation(s) with no ground-plane position "
            "(anchor at or beyond the calibration's horizon)",
            analyzer.unprojectable_observations,
        )


def _log_interaction_startup(
    analyzer: InteractionAnalyzer | None, output_path: Path | None
) -> None:
    if analyzer is None:
        logger.info(
            "Interaction: disabled (pass --interaction-output to measure pairwise geometry)"
        )
        return
    logger.info("Interaction: %s -> %s", analyzer.camera_id, output_path)
    logger.info(
        "  pairs:     same-camera only | proximity <= %.2f %s | %s",
        analyzer.config.proximity_threshold,
        analyzer.space.distance_unit,
        "no distance gate"
        if analyzer.config.max_pair_distance is None
        else f"gated beyond {analyzer.config.max_pair_distance:.1f} {analyzer.space.distance_unit}",
    )


def _log_interaction_completion(
    analyzer: InteractionAnalyzer | None,
    output_path: Path | None,
    result: ProcessingResult,
) -> None:
    if analyzer is None:
        return
    logger.info(
        "Interaction pairs:   %d measured of %d candidate(s) -> %s",
        result.interaction_pairs_measured,
        analyzer.candidate_pairs_seen,
        output_path,
    )


def _log_integrity_startup(
    guard: CalibrationIntegrityGuard | None, reference_path: Path | None
) -> None:
    if guard is None:
        logger.info(
            "Integrity:   not monitored (pass --calibration-reference to detect camera drift)"
        )
        return
    config = guard.monitor.config
    logger.info("Integrity:   monitored against %s", reference_path)
    logger.info(
        "  cadence:   every %.1f s of source time | drift limit %.2f%% of the frame diagonal",
        config.check_interval_seconds,
        config.maximum_normalized_drift * 100.0,
    )
    logger.info(
        "  on drift:  %s",
        "metric observations stop"
        if guard.mode is IntegrityPolicyMode.GUARDED
        else "REPORT ONLY — output continues and may be wrong",
    )


def _log_integrity_completion(
    guard: CalibrationIntegrityGuard | None, result: ProcessingResult
) -> None:
    if guard is None:
        return
    logger.info("Calibration integrity: final state %s", guard.state.value)
    if result.behavior_frames_suppressed:
        logger.warning(
            "  %d frame(s) of metric observations were withheld after drift was detected. "
            "Re-calibrate this camera before trusting its metres again.",
            result.behavior_frames_suppressed,
        )


def _log_integrity_assessment(assessment: IntegrityAssessment) -> None:
    """Print one assessment, the same way for the run and the standalone command."""
    logger.info("State:       %s", assessment.state.value)
    logger.info("Reason:      %s", assessment.reason)
    metrics = assessment.metrics
    if metrics is None:
        return
    logger.info(
        "Features:    reference %d, current %d",
        metrics.reference_keypoints,
        metrics.current_keypoints,
    )
    logger.info(
        "Matching:    %d candidates, %d inliers (%.0f%%)",
        metrics.matches,
        metrics.inliers,
        metrics.inlier_ratio * 100.0,
    )
    if metrics.mean_reprojection_error_px is not None:
        logger.info("Fit:         mean residual %.2f px", metrics.mean_reprojection_error_px)
    if metrics.normalized_drift is not None:
        logger.info(
            "Drift:       mean %.3f%% of the frame diagonal, worst corner %.3f%%",
            metrics.normalized_drift * 100.0,
            (metrics.max_normalized_drift or metrics.normalized_drift) * 100.0,
        )


def _calibration_integrity_command(args: argparse.Namespace) -> int:
    """Compare one current frame against a calibration's reference frame."""
    _configure_logging(getattr(args, "verbose", False))

    fit = load_calibration_fit(args.calibration)
    reference = read_reference_frame(args.reference)
    current = read_reference_frame(args.current)

    config = CalibrationIntegrityConfig()
    monitor = CalibrationIntegrityMonitor(
        reference_frame=reference, config=config, reference_id=str(args.reference)
    )
    assessment = monitor.check(current, frame_index=0, timestamp_seconds=0.0)

    reference_height, reference_width = reference.shape[:2]
    current_height, current_width = current.shape[:2]

    logger.info("Calibration: %s (camera '%s')", args.calibration, fit.camera_id)
    logger.info("Reference:   %s at %dx%d", args.reference, reference_width, reference_height)
    logger.info("Current:     %s at %dx%d", args.current, current_width, current_height)
    _log_integrity_assessment(assessment)
    logger.info(
        "Note:        this compares image geometry against the reference. It says whether "
        "the camera view has moved, not whether the calibration's metres are physically "
        "correct — only measurement on the floor can establish that."
    )

    if args.output is not None:
        evidence = monitor.latest_evidence
        annotated = render_integrity_overlay(
            current,
            reference,
            assessment,
            transform=None if evidence is None else evidence.transform,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotated):
            raise ConfigurationError(f"Could not write the integrity diagnostic to {output_path}.")
        logger.info("Written:     %s", output_path)

    return EXIT_OK


def _calibration_validate_command(args: argparse.Namespace) -> int:
    """Solve a calibration and report what is known about the fit."""
    _configure_logging(getattr(args, "verbose", False))

    fit = load_calibration_fit(args.calibration)
    calibration = fit.calibration
    quality = fit.quality

    logger.info("Calibration: %s", args.calibration)
    logger.info("Camera:      %s", calibration.camera_id)
    logger.info("Venue:       %s", calibration.venue_id)
    logger.info("Surface:     %s", calibration.surface_id)
    logger.info("Reference:   %s", calibration.reference_frame)
    logger.info(
        "Points:      %d%s",
        quality.point_count,
        "" if calibration.is_overdetermined else "  (the minimum; see note below)",
    )
    logger.info("Homography:  solved")
    logger.info("Reprojection error")
    logger.info(
        "  venue:     mean %.4f m, max %.4f m",
        quality.mean_venue_error_m,
        quality.max_venue_error_m,
    )
    logger.info(
        "  image:     mean %.2f px, max %.2f px",
        quality.mean_image_error_px,
        quality.max_image_error_px,
    )
    logger.info(
        "  relative:  %.4f (%.2f%% of the %.2f m calibrated region)",
        quality.relative_error,
        quality.relative_error * 100.0,
        quality.venue_extent_m,
    )
    logger.info(
        "Note:        reprojection error measures how well the fit reproduces its own "
        "reference points. It is not a measurement of real-world accuracy, and there is "
        "no validated threshold to compare it against."
    )
    if not calibration.is_overdetermined:
        logger.info(
            "             With exactly 4 points the fit is exact by construction, so a "
            "near-zero error here says nothing about whether the points were measured "
            "correctly. Add more points to get a meaningful number."
        )
    logger.info(
        "             Verify the geometry visually as well: "
        "atreides-vision calibration render --calibration %s --input <frame> --output <png>",
        args.calibration,
    )
    return EXIT_OK


def _calibration_render_command(args: argparse.Namespace) -> int:
    """Draw a calibration onto a still for visual inspection."""
    _configure_logging(getattr(args, "verbose", False))

    fit = load_calibration_fit(args.calibration)
    frame = read_reference_frame(args.input)
    height, width = frame.shape[:2]

    projector = GroundPlaneProjector(fit, frame_width=width, frame_height=height)
    annotated = render_calibration_overlay(
        frame,
        projector,
        CalibrationOverlayConfig(
            grid_spacing_metres=args.grid_spacing,
            draw_grid=not args.no_grid,
        ),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise ConfigurationError(
            f"Could not write the calibration diagnostic to {output_path}. Check the "
            "directory is writable and the extension is one OpenCV can encode (.png, .jpg)."
        )

    logger.info("Calibration: %s (camera '%s')", args.calibration, fit.camera_id)
    logger.info("Frame:       %s at %dx%d", args.input, width, height)
    if projector.is_rescaling:
        logger.info(
            "             rescaled from the calibration reference frame %s",
            fit.calibration.reference_frame,
        )
    logger.info("Grid:        %.2f m spacing", args.grid_spacing)
    logger.info("Written:     %s", output_path)
    logger.info(
        "Check:       do the grid lines follow the real floor? That is the one test "
        "reprojection error cannot perform."
    )
    return EXIT_OK


def _log_startup(
    config: VisionConfig,
    metadata: VideoMetadata,
    detector_name: str,
    tracker_name: str,
    layout: ZoneLayout | None = None,
    zones_path: Path | None = None,
) -> None:
    frames = str(metadata.frame_count) if metadata.frame_count is not None else "unknown"
    at_defaults = (
        config.detector.confidence_threshold == DEFAULT_DETECTION_CONFIDENCE
        and tracker_name == "OC-SORT"
        and config.tracker.high_conf_det_threshold == DEFAULT_OCSORT_HIGH_CONF_THRESHOLD
        and config.pipeline.deduplicate_detections is DEFAULT_DEDUPLICATE_DETECTIONS
    )
    logger.info("Atreides Vision %s", __version__)
    logger.info(
        "Profile:     %s",
        "V0.2.1 validated defaults"
        if at_defaults
        else "CUSTOM (one or more V0.2.1 defaults overridden)",
    )
    logger.info("Input:       %s", config.pipeline.input_path)
    logger.info("Output:      %s", config.pipeline.output_path)
    logger.info("Resolution:  %s", metadata.resolution)
    logger.info("Source FPS:  %.2f", metadata.fps)
    logger.info("Frames:      %s", frames)
    logger.info(
        "Detector:    %s (confidence >= %.2f)",
        detector_name,
        config.detector.confidence_threshold,
    )
    logger.info("Tracker:     %s", tracker_name)
    if tracker_name == "OC-SORT":
        logger.info("  high-conf detection threshold: %.2f", config.tracker.high_conf_det_threshold)
    logger.info("Device:      %s", config.detector.device)
    logger.info(
        "Dedup:       %s",
        "Rule B3 (IoU>=0.85, area>=0.95, centre<=0.03)"
        if config.pipeline.deduplicate_detections
        else "disabled",
    )
    if layout is None:
        logger.info("Zones:       disabled (pass --zones to enable zone intelligence)")
    else:
        logger.info(
            "Zones:       %d from %s (camera '%s'): %s",
            len(layout),
            zones_path,
            layout.camera_id,
            ", ".join(zone.name for zone in layout) or "none",
        )


def _log_completion(
    result: ProcessingResult,
    analyzer: ZoneAnalyzer | None = None,
    metrics_path: Path | None = None,
) -> None:
    logger.info("Done.")
    logger.info("Frames processed:    %d", result.frames_processed)
    logger.info("Elapsed time:        %.2f s", result.elapsed_seconds)
    logger.info("Effective FPS:       %.2f", result.effective_fps)
    logger.info("Unique track IDs:    %d", result.unique_track_ids)
    logger.info("Output:              %s", result.output_path)
    if result.detections_removed_by_deduplication:
        logger.info(
            "Deduplication:       %d detections suppressed before tracking",
            result.detections_removed_by_deduplication,
        )
    if analyzer is not None:
        analyzer.log_summary()
    if metrics_path is not None:
        logger.info(
            "Metrics:             %d records -> %s",
            result.metrics_records_written,
            metrics_path,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code: int = args.handler(args)
        return exit_code
    except AtreidesVisionError as exc:
        # Domain errors carry operator-readable messages; a traceback adds nothing.
        logging.getLogger("atreides_vision").error("%s", exc)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        logging.getLogger("atreides_vision").warning("Interrupted. Output may be incomplete.")
        return EXIT_INTERRUPTED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

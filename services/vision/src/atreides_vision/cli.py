"""Command-line entry point for Atreides Vision.

This module is the composition root: the one place that picks concrete
implementations (RF-DETR Nano, ByteTrack) and wires them into the pipeline.
Everything below it depends only on Atreides interfaces.

    uv run atreides-vision run --input data/input/test.mp4 --output artifacts/tracked.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from atreides_vision import __version__
from atreides_vision.config import (
    DetectorConfig,
    Device,
    PipelineConfig,
    TrackerConfig,
    VisionConfig,
    VisualizationConfig,
    resolve_device,
)
from atreides_vision.detection.rfdetr import RFDetrPersonDetector
from atreides_vision.errors import AtreidesVisionError
from atreides_vision.tracking.bytetrack import ByteTrackTracker
from atreides_vision.video.processor import ProcessingResult, VideoProcessor
from atreides_vision.video.sink import Mp4VideoSink
from atreides_vision.video.source import FileVideoSource, VideoMetadata

logger = logging.getLogger("atreides_vision")

__all__ = ["build_parser", "main"]

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
        default=0.5,
        metavar="FLOAT",
        help="Minimum detection confidence, in (0, 1]. Default: 0.5.",
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
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N frames. Useful for a quick check on a long video.",
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
    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _run_command(args: argparse.Namespace) -> int:
    _configure_logging(args.verbose)

    pipeline_config = PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        max_frames=args.max_frames,
        log_every_n_frames=args.log_every,
    )
    device = resolve_device(args.device)

    with FileVideoSource(pipeline_config.input_path) as source:
        metadata = source.metadata

        config = VisionConfig(
            pipeline=pipeline_config,
            detector=DetectorConfig(confidence_threshold=args.confidence, device=device),
            # The tracker's motion model is frame-rate sensitive, so it follows the source.
            tracker=TrackerConfig(frame_rate=metadata.fps),
            visualization=VisualizationConfig(
                trail_length=args.trail_length,
                draw_confidence=not args.no_confidence_labels,
            ),
        )

        detector = RFDetrPersonDetector(config.detector)
        tracker = ByteTrackTracker(config.tracker)

        _log_startup(config, metadata, detector.name, tracker.name)

        logger.info("Loading detector weights (first run may download them)...")
        detector.load()
        logger.info("Detector ready.")

        with Mp4VideoSink(
            path=config.pipeline.output_path,
            width=metadata.width,
            height=metadata.height,
            fps=metadata.fps,
        ) as sink:
            processor = VideoProcessor(detector=detector, tracker=tracker, config=config)
            result = processor.run(source, sink)

    _log_completion(result)
    return EXIT_OK


def _log_startup(
    config: VisionConfig,
    metadata: VideoMetadata,
    detector_name: str,
    tracker_name: str,
) -> None:
    frames = str(metadata.frame_count) if metadata.frame_count is not None else "unknown"
    logger.info("Atreides Vision %s", __version__)
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
    logger.info("Device:      %s", config.detector.device)


def _log_completion(result: ProcessingResult) -> None:
    logger.info("Done.")
    logger.info("Frames processed:    %d", result.frames_processed)
    logger.info("Elapsed time:        %.2f s", result.elapsed_seconds)
    logger.info("Effective FPS:       %.2f", result.effective_fps)
    logger.info("Unique track IDs:    %d", result.unique_track_ids)
    logger.info("Output:              %s", result.output_path)


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

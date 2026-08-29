"""Composing a WILDTRACK run of the *real* generic venue-validation runtime.

This is the one WILDTRACK module that sits **above**
:func:`~atreides_vision.venue_validation.runner.run_venue_validation` rather than
beside it. The generic runner knows nothing about WILDTRACK and must not: it
consumes a :class:`~atreides_vision.venue_validation.config.VenueValidationConfig`,
an injected detector and tracker factory, and a ``source_factory``. This module
builds exactly those, from a validated :class:`WildtrackDatasetLayout`, so that
WILDTRACK's synchronized PNG frames flow through the unmodified runtime:

    WILDTRACK PNG -> ImageSequenceVideoSource -> detector -> tracker
      -> BehaviorAnalyzer -> GroundPlaneProjector -> build_track_observation
      -> AssociationEngine -> fusion / VenueSubjectRegistry -> compute_occupancy
      -> VenueValidationObserver snapshot

Only structural ground-truth is used: *which* frame identifiers carry an
annotation file (:attr:`WildtrackDatasetLayout.annotation_frame_ids`, via
:func:`build_wildtrack_stream_specs`), which selects the scored instants. No
person-level annotation, correspondence or occupancy ground truth is read here
and none may ever be: those belong to the offline evaluator, strictly below the
observer seam.

Live timebase wiring
--------------------
Every WILDTRACK stream is placed on the live
:class:`~atreides_vision.venue.timebase.VenueTimebase` through the generic
manifest mechanism, carrying the dataset's declared timing verbatim:
``fps_source = DECLARED`` (an :class:`ImageSequenceVideoSource` always reports
that), ``start_offset_seconds = 0.0`` and ``offset_accuracy_seconds = 0.05``
(:data:`WILDTRACK_STREAM_OFFSET_SECONDS` /
:data:`WILDTRACK_STREAM_OFFSET_ACCURACY_SECONDS`). The runner registers each
stream with those values unchanged, so timing resolves to ESTABLISHED with a
0.05 s half-width and every camera's identical WILDTRACK sample lands on one
nominal venue instant. Nothing here touches the timing core.

Stage subsetting
----------------
Stage A, B and C differ only in :paramref:`selected_camera_ids` and a contiguous
window over the canonical annotated instants (:paramref:`start_instant_index` /
:paramref:`instant_limit`) — the same composition and the same runtime for all
three. There is no second runtime path and no resampling: a window is a slice of
the already-canonical annotation frame list.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from atreides_vision.errors import WildtrackDatasetError
from atreides_vision.venue.manifest import VenueCamera, VenueSessionManifest, VenueStreamEntry
from atreides_vision.venue_validation.config import (
    VenueValidationCameraConfig,
    VenueValidationConfig,
)
from atreides_vision.venue_validation.datasets.wildtrack.calibration import (
    load_camera_calibration,
    to_ground_plane_calibration,
)
from atreides_vision.venue_validation.datasets.wildtrack.layout import WildtrackDatasetLayout
from atreides_vision.venue_validation.datasets.wildtrack.runtime import (
    WildtrackStreamSpec,
    build_wildtrack_stream_specs,
)
from atreides_vision.venue_validation.datasets.wildtrack.time import WildtrackFrameId
from atreides_vision.video.source import VideoSource

__all__ = [
    "DEFAULT_WILDTRACK_SESSION_ID",
    "WildtrackRuntimeSession",
    "build_wildtrack_runtime_session",
]

#: The session identifier used for a WILDTRACK validation run. Fixed, because it
#: reaches the deterministic run artifacts and a per-run value would break their
#: byte-for-byte reproducibility.
DEFAULT_WILDTRACK_SESSION_ID = "wildtrack"


@dataclass(frozen=True, slots=True)
class WildtrackRuntimeSession:
    """Everything needed to run one WILDTRACK session on the generic runner.

    ``config``, ``source_factory`` and (for an observer that scores the result)
    ``scored_frame_ids`` are the three things
    :func:`~atreides_vision.venue_validation.runner.run_venue_validation` and a
    downstream evaluator consume. ``scored_frame_ids`` is the canonical, ordered
    list of WILDTRACK annotation frames this run covers — the k-th emitted venue
    instant corresponds to the k-th entry here.
    """

    config: VenueValidationConfig
    source_factory: Callable[[VenueStreamEntry], VideoSource]
    scored_frame_ids: tuple[WildtrackFrameId, ...]
    #: The selected cameras, in canonical ``C1..C7`` order.
    camera_ids: tuple[str, ...]


def build_wildtrack_runtime_session(
    layout: WildtrackDatasetLayout,
    *,
    output_dir: Path,
    selected_camera_ids: Iterable[str] | None = None,
    start_instant_index: int = 0,
    instant_limit: int | None = None,
    calibrated: bool = True,
    session_id: str = DEFAULT_WILDTRACK_SESSION_ID,
) -> WildtrackRuntimeSession:
    """Compose a WILDTRACK run of the real venue-validation runtime.

    Args:
        layout: A validated WILDTRACK dataset layout.
        output_dir: Where the runner writes its artifacts.
        selected_camera_ids: The camera subset, in any order; ``None`` selects
            all seven. The run always uses canonical ``C1..C7`` order.
        start_instant_index: Index of the first annotated instant to include,
            into :attr:`WildtrackDatasetLayout.annotation_frame_ids`.
        instant_limit: Number of contiguous annotated instants to include from
            ``start_instant_index``; ``None`` includes every remaining one.
        calibrated: When true (the baseline), each camera gets a
            :class:`~atreides_vision.calibration.fitting.CalibrationFit` derived
            from WILDTRACK's own calibration, so runtime observations land in
            the same ``positionID`` metre ground frame the ground truth uses.
            When false, cameras run uncalibrated.
        session_id: Session identifier; the fixed default keeps run artifacts
            reproducible.

    Raises:
        WildtrackDatasetError: for an unknown/duplicate/empty camera selection,
            a window that selects no instants, or a window outside the
            annotated range.
    """
    specs = build_wildtrack_stream_specs(layout, cameras=selected_camera_ids)
    windowed = _apply_instant_window(specs, start_instant_index, instant_limit)

    scored_frame_ids = windowed[0].frame_ids
    for spec in windowed:
        if spec.frame_ids != scored_frame_ids:
            raise WildtrackDatasetError(
                "WILDTRACK cameras disagree on the scored frame set after windowing; "
                f"{spec.camera_id} does not match {windowed[0].camera_id}."
            )

    specs_by_input: dict[Path, WildtrackStreamSpec] = {}
    cameras: list[VenueCamera] = []
    camera_configs: list[VenueValidationCameraConfig] = []
    for spec in windowed:
        input_path = spec.image_paths[0]
        specs_by_input[input_path] = spec
        cameras.append(
            VenueCamera(
                camera_id=spec.camera_id,
                sources=(
                    VenueStreamEntry(
                        input_path=input_path,
                        start_offset_seconds=spec.offset_seconds,
                        offset_accuracy_seconds=spec.offset_accuracy_seconds,
                        declared_fps=spec.declared_fps,
                    ),
                ),
            )
        )
        if calibrated:
            fit = to_ground_plane_calibration(
                load_camera_calibration(layout, spec.camera_id), layout
            )
            camera_configs.append(
                VenueValidationCameraConfig(camera_id=spec.camera_id, calibration_fit=fit)
            )

    manifest = VenueSessionManifest(session_id=session_id, cameras=tuple(cameras))
    config = VenueValidationConfig(
        manifest=manifest,
        output_dir=output_dir,
        cameras=tuple(camera_configs),
    )

    def source_factory(entry: VenueStreamEntry) -> VideoSource:
        try:
            spec = specs_by_input[entry.input_path]
        except KeyError:
            raise WildtrackDatasetError(
                f"No WILDTRACK stream spec for manifest entry {entry.input_path}; "
                "this source factory only serves the session it was built for."
            ) from None
        return spec.open()

    return WildtrackRuntimeSession(
        config=config,
        source_factory=source_factory,
        scored_frame_ids=scored_frame_ids,
        camera_ids=tuple(spec.camera_id for spec in windowed),
    )


def _apply_instant_window(
    specs: tuple[WildtrackStreamSpec, ...],
    start_instant_index: int,
    instant_limit: int | None,
) -> tuple[WildtrackStreamSpec, ...]:
    total = len(specs[0].frame_ids)
    if start_instant_index < 0:
        raise WildtrackDatasetError(
            f"start_instant_index must be non-negative, got {start_instant_index}."
        )
    if start_instant_index >= total:
        raise WildtrackDatasetError(
            f"start_instant_index {start_instant_index} is past the last of {total} "
            "annotated instants."
        )
    if instant_limit is not None and instant_limit <= 0:
        raise WildtrackDatasetError(
            f"instant_limit must be a positive count when given, got {instant_limit}."
        )
    end = total if instant_limit is None else min(total, start_instant_index + instant_limit)
    window = slice(start_instant_index, end)
    return tuple(
        dataclasses.replace(
            spec,
            image_paths=spec.image_paths[window],
            frame_ids=spec.frame_ids[window],
        )
        for spec in specs
    )

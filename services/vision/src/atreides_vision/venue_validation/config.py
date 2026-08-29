"""Configuration for one end-to-end venue validation run.

Deliberately the smallest surface that can drive the existing runtime: the
already-loaded V0.4.0 manifest says which cameras and streams exist, and this
module adds only what the manifest cannot carry — where artifacts go, and the
per-camera calibration and integrity references the camera-local runtime needs.

Association gating and decision thresholds are *not* exposed. Retuning them from
a validation harness is exactly the frozen-scope violation the project forbids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from atreides_vision.calibration.fitting import CalibrationFit
from atreides_vision.config import DEFAULT_TRACKER, DetectorConfig, Tracker
from atreides_vision.errors import ConfigurationError
from atreides_vision.identity.config import VenueIdentityConfig
from atreides_vision.venue.manifest import VenueSessionManifest

__all__ = [
    "ASSOCIATION_REPORT_FILENAME",
    "OCCUPANCY_REPORT_FILENAME",
    "SUMMARY_FILENAME",
    "VenueValidationCameraConfig",
    "VenueValidationConfig",
]

#: Fixed artifact names. Fixed, because a run summary that named an absolute
#: temporary path would not be byte-identical across two equivalent runs.
ASSOCIATION_REPORT_FILENAME = "venue_pipeline_report.jsonl"
OCCUPANCY_REPORT_FILENAME = "occupancy_report.jsonl"
SUMMARY_FILENAME = "venue_validation_summary.json"


@dataclass(frozen=True, slots=True)
class VenueValidationCameraConfig:
    """Camera-local spatial configuration for one manifest camera.

    Both references are optional and independent. A camera with no calibration
    runs uncalibrated — it still contributes tracks and occupancy, it simply
    produces no venue-frame geometry, so association gates it as
    ``NOT_COMPARABLE`` instead of inventing coordinates for it.

    Calibration may be supplied either as a document path (``calibration_path``,
    loaded and solved by the runner) or as an already-built
    :class:`~atreides_vision.calibration.fitting.CalibrationFit`
    (``calibration_fit``, a programmatic injection seam for validation adapters
    that build the fit upstream). At most one of the two may be set; supplying
    both is a configuration error rather than a silent precedence choice. Either
    way the fit flows through the same :class:`GroundPlaneProjector`.
    """

    camera_id: str
    #: Calibration document for this camera. ``None`` runs it uncalibrated.
    calibration_path: Path | None = None
    #: A calibration fit built upstream, injected directly. Mutually exclusive
    #: with ``calibration_path``. Its ``camera_id`` must match this camera.
    calibration_fit: CalibrationFit | None = None
    #: Still or clip the integrity monitor compares live frames against.
    integrity_reference_path: Path | None = None
    #: Report drift without withholding metric output. Only meaningful with a
    #: reference; guarded is the default because a moved camera's metres are
    #: wrong, and wrong metres are worse than absent ones.
    integrity_report_only: bool = False

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ConfigurationError("camera_id must not be empty.")
        if self.calibration_path is not None and self.calibration_fit is not None:
            raise ConfigurationError(
                f"Camera {self.camera_id!r} supplies both calibration_path and "
                "calibration_fit; supply exactly one calibration source."
            )
        if self.calibration_fit is not None and self.calibration_fit.camera_id != self.camera_id:
            raise ConfigurationError(
                f"Camera {self.camera_id!r} was given a calibration_fit built for camera "
                f"{self.calibration_fit.camera_id!r}."
            )
        if self.integrity_reference_path is None and self.integrity_report_only:
            raise ConfigurationError(
                f"Camera {self.camera_id!r} sets integrity_report_only without an "
                "integrity_reference_path; there is no monitor for it to apply to."
            )


@dataclass(frozen=True, slots=True)
class VenueValidationConfig:
    """Everything one validation run needs, and nothing it does not.

    Raises:
        ConfigurationError: if a camera entry names a camera the manifest does
            not contain, or names one twice.
    """

    manifest: VenueSessionManifest
    output_dir: Path
    #: Per-camera spatial configuration. Cameras absent from this tuple run
    #: uncalibrated and unmonitored; it need not cover every manifest camera.
    cameras: tuple[VenueValidationCameraConfig, ...] = ()
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: Tracker = DEFAULT_TRACKER
    identity: VenueIdentityConfig = field(default_factory=VenueIdentityConfig)
    #: Passed through to the camera-local pipeline config, unchanged.
    deduplicate_detections: bool = True

    def __post_init__(self) -> None:
        known = set(self.manifest.camera_ids)
        seen: set[str] = set()
        for camera in self.cameras:
            if camera.camera_id not in known:
                raise ConfigurationError(
                    f"Camera {camera.camera_id!r} is configured for validation but is not in "
                    f"session {self.manifest.session_id!r}. Known cameras: "
                    f"{', '.join(sorted(known))}."
                )
            if camera.camera_id in seen:
                raise ConfigurationError(
                    f"Camera {camera.camera_id!r} is configured more than once."
                )
            seen.add(camera.camera_id)

    @property
    def session_id(self) -> str:
        return self.manifest.session_id

    @property
    def association_report_path(self) -> Path:
        """Where the V0.4.5 association/fusion runtime stream is written."""
        return self.output_dir / ASSOCIATION_REPORT_FILENAME

    @property
    def occupancy_report_path(self) -> Path:
        """Where the V0.4.6 occupancy stream is written."""
        return self.output_dir / OCCUPANCY_REPORT_FILENAME

    @property
    def summary_path(self) -> Path:
        """Where the deterministic run summary is written."""
        return self.output_dir / SUMMARY_FILENAME

    def camera_config(self, camera_id: str) -> VenueValidationCameraConfig | None:
        """Spatial configuration for ``camera_id``, or ``None`` if unconfigured."""
        for camera in self.cameras:
            if camera.camera_id == camera_id:
                return camera
        return None

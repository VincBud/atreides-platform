"""End-to-end multi-camera validation runner — V0.4.8.

An engineering harness, not a product runtime. It wires the already-committed
layers together — V0.4.0 venue ingestion, the camera-local video/detection/
tracking/behaviour/calibration runtime, V0.4.7's observation adapter, V0.4.4's
:class:`~atreides_vision.venue_pipeline.VenuePipeline`, V0.4.6 occupancy and
the V0.4.5/V0.4.6 reporting streams — and records what actually happened.

It measures execution, never accuracy. Nothing here compares an output against
human ground truth, and nothing here may be read as an association or occupancy
accuracy figure.
"""

from __future__ import annotations

from atreides_vision.venue_validation.config import (
    VenueValidationCameraConfig,
    VenueValidationConfig,
)
from atreides_vision.venue_validation.models import (
    GROUND_TRUTH_STATUS,
    VenueValidationStreamSummary,
    VenueValidationSummary,
)
from atreides_vision.venue_validation.reporting import (
    VENUE_VALIDATION_SUMMARY_SCHEMA_VERSION,
    summary_record,
    write_validation_summary,
)
from atreides_vision.venue_validation.runner import open_stream_source, run_venue_validation

__all__ = [
    "GROUND_TRUTH_STATUS",
    "VENUE_VALIDATION_SUMMARY_SCHEMA_VERSION",
    "VenueValidationCameraConfig",
    "VenueValidationConfig",
    "VenueValidationStreamSummary",
    "VenueValidationSummary",
    "open_stream_source",
    "run_venue_validation",
    "summary_record",
    "write_validation_summary",
]

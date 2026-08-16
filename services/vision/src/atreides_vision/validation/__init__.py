"""V0.3.6 real-world validation harness.

A **consumer** of the V0.3 runtime, never a part of it. Nothing in this package
is imported by the measurement layers, no measurement algorithm imports anything
from here, and running validation cannot change what the pipeline produces.

What it does: process a manifest of real clips with the frozen V0.2.1
detector/tracker defaults, then measure the resulting output streams and report
what they say about tracking stability, signal availability, output volume and
local throughput.

What it does not do: tune anything, decide anything, or claim anything the
outputs cannot support. Two boundaries are held throughout and stated on every
report:

**Proxy metrics vs ground truth.** Everything derived from Atreides' own output
is a *proxy/system metric*. Track births per minute is not an ID-switch rate,
short-track fraction is not proof of fragmentation, and neither is detection
recall. True recall, true ID switches and true fragmentation require human
annotation, which lives in :mod:`~atreides_vision.validation.ground_truth` and
is reported separately or not at all.

**No fabricated physical units.** None of the current validation footage has a
measured floor calibration, so every clip runs in normalized image units and no
metre, m/s or subjects/m² figure is reported for it. The manifest can carry a
calibration path for a future physically measured scene; until one is supplied,
the harness reports normalized units and says so.
"""

from __future__ import annotations

from atreides_vision.validation.analysis import analyze_clip, analyze_run_directory
from atreides_vision.validation.behavior_metrics import (
    BehaviorCoverageMetrics,
    BehaviorMetricsAccumulator,
)
from atreides_vision.validation.crowd_metrics import CrowdMetrics, CrowdMetricsAccumulator
from atreides_vision.validation.distributions import Distribution
from atreides_vision.validation.ground_truth import (
    ContinuityAnnotation,
    FrameAnnotation,
    GroundTruthStatus,
    ManualGroundTruth,
    load_manual_ground_truth,
    write_annotation_templates,
)
from atreides_vision.validation.interaction_metrics import (
    InteractionMetrics,
    InteractionMetricsAccumulator,
)
from atreides_vision.validation.manifest import (
    MANIFEST_VERSION,
    ValidationClip,
    ValidationManifest,
    load_manifest,
)
from atreides_vision.validation.models import (
    VALIDATION_SUMMARY_SCHEMA_VERSION,
    ClipValidationResult,
    OutputVolumeMetrics,
    PerformanceMetrics,
    RunProvenance,
    StreamVolume,
    ValidationSummary,
)
from atreides_vision.validation.reporting import (
    build_summary,
    render_report,
    write_validation_artifacts,
)
from atreides_vision.validation.review_samples import (
    ContinuityWindow,
    ReviewSamplePlan,
    SampleFrame,
    build_sample_plan,
)
from atreides_vision.validation.runner import ClipRunOutcome, ValidationRunner
from atreides_vision.validation.storage import ValidationLayout
from atreides_vision.validation.streams import iter_jsonl, stream_size_bytes
from atreides_vision.validation.track_metrics import TrackMetrics, TrackPresenceAccumulator

__all__ = [
    "MANIFEST_VERSION",
    "VALIDATION_SUMMARY_SCHEMA_VERSION",
    "BehaviorCoverageMetrics",
    "BehaviorMetricsAccumulator",
    "ClipRunOutcome",
    "ClipValidationResult",
    "ContinuityAnnotation",
    "ContinuityWindow",
    "CrowdMetrics",
    "CrowdMetricsAccumulator",
    "Distribution",
    "FrameAnnotation",
    "GroundTruthStatus",
    "InteractionMetrics",
    "InteractionMetricsAccumulator",
    "ManualGroundTruth",
    "OutputVolumeMetrics",
    "PerformanceMetrics",
    "ReviewSamplePlan",
    "RunProvenance",
    "SampleFrame",
    "StreamVolume",
    "TrackMetrics",
    "TrackPresenceAccumulator",
    "ValidationClip",
    "ValidationLayout",
    "ValidationManifest",
    "ValidationRunner",
    "ValidationSummary",
    "analyze_clip",
    "analyze_run_directory",
    "build_sample_plan",
    "build_summary",
    "iter_jsonl",
    "load_manifest",
    "load_manual_ground_truth",
    "render_report",
    "stream_size_bytes",
    "write_annotation_templates",
    "write_validation_artifacts",
]

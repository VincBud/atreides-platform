"""Atreides Vision error hierarchy.

Every failure the operator can plausibly cause (bad path, unreadable video,
unavailable accelerator) surfaces as an :class:`AtreidesVisionError` with a
message written for a human, not a stack trace written for a library author.
The CLI catches this base class and prints the message without a traceback.
"""

from __future__ import annotations


class AtreidesVisionError(Exception):
    """Base class for all Atreides Vision domain errors."""


class ConfigurationError(AtreidesVisionError):
    """Raised when the supplied configuration is internally inconsistent."""


class DeviceUnavailableError(AtreidesVisionError):
    """Raised when a requested compute device cannot be used on this machine."""


class VideoSourceError(AtreidesVisionError):
    """Raised when a video source cannot be opened or read."""


class VideoSinkError(AtreidesVisionError):
    """Raised when a video sink cannot be created or written to."""


class ZoneConfigurationError(AtreidesVisionError):
    """Raised when a zone configuration file is missing, malformed or invalid."""


class MetricsOutputError(AtreidesVisionError):
    """Raised when the zone-metrics output stream cannot be created or written."""


class BehaviorOutputError(AtreidesVisionError):
    """Raised when the behaviour signal stream cannot be created or written."""


class InteractionError(AtreidesVisionError):
    """Raised when pairwise interaction measurement is asked for the impossible.

    Covers pairs that cannot exist (a subject with itself, two subjects on
    different cameras) and positions that cannot be compared (different
    coordinate spaces, venues or surfaces). Every one of these would otherwise
    produce a confident number describing a relationship that is not real.
    """


class InteractionOutputError(AtreidesVisionError):
    """Raised when the interaction stream cannot be created or written."""


class CrowdError(AtreidesVisionError):
    """Raised when spatial-cluster measurement is asked for the impossible.

    Covers subjects that cannot be clustered together (different cameras,
    different coordinate spaces, different venues or surfaces) and geometry that
    cannot be computed. Each would otherwise produce a confident cluster
    describing an arrangement that is not real.
    """


class CrowdOutputError(AtreidesVisionError):
    """Raised when the crowd stream cannot be created or written."""


class CalibrationError(AtreidesVisionError):
    """Raised when a ground-plane calibration is missing, malformed or unusable.

    Covers configuration problems (too few correspondences, mismatched counts,
    non-finite values) and geometric ones (collinear or duplicate points, a
    singular transform, a fit whose reprojection error is grossly wrong).

    A calibration that cannot be trusted is always an error. Falling back to
    normalized image coordinates would silently change the unit of every
    downstream signal from metres to frame fractions, which is far worse than
    stopping.
    """


class ProjectionError(CalibrationError):
    """Raised when a specific point cannot be projected onto the ground plane.

    Distinct from a bad calibration: the calibration may be perfectly good and
    the *point* still have no ground position — anchors at or beyond the
    horizon line project to infinity or to the mirrored side of it. Callers
    that process many points per frame catch this narrowly and skip the point
    rather than aborting.
    """


class ValidationError(AtreidesVisionError):
    """Raised when the V0.3.6 validation harness cannot proceed.

    Covers manifests that cannot be trusted (missing clips, duplicate clip or
    camera ids, malformed JSON), output streams that cannot be read, and manual
    annotation files that do not parse. Every one of these would otherwise
    produce a validation report whose numbers describe something other than what
    the report claims — which is worse than no report at all.
    """


class AssociationError(AtreidesVisionError):
    """Raised when cross-camera association is asked for the impossible.

    Covers candidates that cannot exist (an observation paired with itself),
    observations that contradict themselves (a subject, a stream and a spatial
    context naming different cameras, or a position in a space the context does
    not declare), and results that would over-claim (a gating verdict asserting
    an association geometry cannot establish).

    Every one of these would otherwise produce a confident statement that two
    camera-local tracklets are one person. A wrong merge corrupts every
    venue-level number above it and is harder to detect than a missed one, so
    the impossible cases raise rather than resolve.
    """


class VenueSessionError(AtreidesVisionError):
    """Raised when a multi-camera venue session cannot be described or walked.

    Covers venue manifests that cannot be trusted (unknown version, duplicate
    camera ids, an offset declared without an accuracy), time bases that would
    have to invent a number (a declared offset on a stream whose frame rate was
    only assumed), stream references and frames that violate their invariants,
    and the venue session output stream failing to open or write.

    Every one of these would otherwise put a confident time on a frame that is
    not where it claims to be on the venue timeline. Since venue time is what
    orders observations from different cameras, a wrong second here is worse
    than a stopped run.
    """


class IdentityError(AtreidesVisionError):
    """Raised when venue-local identity is asked for the impossible.

    Covers memberships that contradict their own provenance (a tracklet
    claimed on a camera that disagrees with its stream), operations on a
    subject that does not exist or is no longer active, a camera-local
    tracklet claimed twice — by the same venue subject or by two different
    ones — and venue time that runs backwards within one subject's history.

    Every one of these would otherwise let two different camera-local
    tracklets be silently reconciled into one identity, or one identity
    silently forget who it was made of. Venue-local identity has no
    biometric or appearance evidence to fall back on, so the only thing
    protecting it from a wrong merge is refusing the operation outright
    rather than guessing.
    """


class FusionError(AtreidesVisionError):
    """Raised when association-to-venue-identity integration is asked for the impossible.

    Covers an outcome record built without the explanation it requires. This
    layer coordinates :mod:`atreides_vision.association` and
    :mod:`atreides_vision.identity` without absorbing either's own failure
    modes: a decision that fails association's own invariants raises
    :class:`AssociationError`, and a registry operation this layer performs
    that violates identity's own invariants raises :class:`IdentityError`
    unchanged. This class exists only for the integration layer's own
    reporting contract.
    """


class DetectorError(AtreidesVisionError):
    """Raised when the detector cannot be initialised or fails at inference."""


class TrackerError(AtreidesVisionError):
    """Raised when the tracker cannot be initialised or fails during update."""

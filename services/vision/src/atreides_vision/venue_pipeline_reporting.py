"""Observability for :mod:`atreides_vision.venue_pipeline` — V0.4.5.

V0.4.4 wired association, fusion and venue-local identity into one runtime
seam, :class:`~atreides_vision.venue_pipeline.VenuePipeline`, but produced
nothing a process outside that object graph could inspect. This module makes
one venue instant's full decision trail — what was observed, what was
compared, what was decided, what fusion did with it, and what venue-local
identity looks like afterward — into a deterministic, versioned JSON Lines
stream.

**This is observability only.** It reflects runtime truth; it never produces
it. Nothing here recomputes an :class:`~atreides_vision.association.models.AssociationOutcome`,
an :class:`~atreides_vision.fusion.models.IntegrationOutcome`, or a venue
subject's membership — every record is built by reading already-produced
:class:`~atreides_vision.venue_pipeline.VenueInstantResult` and
:class:`~atreides_vision.identity.registry.VenueSubjectRegistry` state, using
each domain's own ``to_record()`` wherever one already exists. No reporting
output here is ever fed back into association or fusion.

File shape
----------
One JSON object per line. The first line is a ``venue_pipeline_report``
header; every line after it is one ``venue_instant`` record, one per call to
:meth:`JsonlVenuePipelineReportWriter.write_instant`::

    {"record_type":"venue_pipeline_report","schema_version":"0.4.5",...}
    {"record_type":"venue_instant","schema_version":"0.4.5","venue_seconds":10.0,...}

Every record repeats ``schema_version`` and ``session_id``, matching every
other stream in this repository: JSONL is read a line at a time, and a line
that cannot be interpreted alone will eventually be misinterpreted.

What one ``venue_instant`` record contains
--------------------------------------------
- ``association`` — :meth:`~atreides_vision.association.models.AssociationRound.to_record`
  unchanged: every observation considered, every candidate pair's outcome
  (``associated``, ``ambiguous``, ``unassociated`` or ``not_comparable``),
  with its reasons, support and confidence.
- ``integration`` — one entry per decision in the round, each carrying the
  :class:`~atreides_vision.fusion.models.IntegrationOutcome` fusion produced
  for it (``applied``, ``no_op``, ``refused`` or ``ignored``), the structured
  detail fusion itself already recorded, and which venue subject(s) it
  concerns. A decision that was never a positive association is present here
  as ``ignored``, never omitted.
- ``active_subjects`` — every venue subject active immediately after this
  instant's integration, each resolved back to its full
  :class:`~atreides_vision.identity.models.VenueSubject` record: creation and
  last-activity time, and the camera-local tracklets that currently make it
  up. This is the provenance chain: a ``venue_subject_id`` in this record
  always resolves to the ``(camera_id, stream_epoch, track_id)`` observations
  that produced it.

What this module does not do
-------------------------------
No occupancy, no duplicate-suppression or unique-visitor metric, no risk or
decision-engine output, no new association or fusion policy, no embeddings,
appearance or biometric evidence, no persistent identity store, no CLI
wiring. Association's own refusal to force a merge, and fusion's own refusal
to resolve a cross-subject conflict, are reported here exactly as each layer
already decided them — this module never resolves an ``ambiguous`` decision
or a ``refused`` integration on their behalf.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from atreides_vision.errors import FusionError
from atreides_vision.fusion.models import MembershipOutcome
from atreides_vision.identity.registry import VenueSubjectRegistry
from atreides_vision.jsonl import GZIP_SUFFIX, JsonlStreamWriter
from atreides_vision.venue_pipeline import VenueInstantResult

logger = logging.getLogger(__name__)

__all__ = [
    "GZIP_SUFFIX",
    "VENUE_PIPELINE_REPORT_SCHEMA_VERSION",
    "JsonlVenuePipelineReportWriter",
    "VenuePipelineReportMetadata",
    "instant_record",
]

#: Version of the record contract below, owned independently of every other
#: stream in this repository — behaviour, interaction, crowd and the V0.4.0
#: venue session stream each already version themselves separately, and this
#: stream describes association/fusion/identity, which has nothing in common
#: with any of their cardinalities or lifetimes.
VENUE_PIPELINE_REPORT_SCHEMA_VERSION = "0.4.5"

#: Matches the precision :class:`~atreides_vision.association.models.AssociationRound`
#: and :class:`~atreides_vision.association.models.TrackObservation` already
#: round ``venue_seconds`` to, so the top-level field and the nested
#: association record can never disagree about the same instant.
_SECONDS_PRECISION = 3


@dataclass(frozen=True, slots=True)
class VenuePipelineReportMetadata:
    """The header record: everything constant for the whole report stream.

    ``session_id`` is a plain, caller-supplied label — this module does not
    import :mod:`atreides_vision.venue` and does not know what a venue
    session manifest is. It exists so this stream's lines can be correlated
    with a V0.4.0 venue session stream describing the same run, without this
    module depending on that one.
    """

    session_id: str

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise FusionError("A venue pipeline report requires a non-empty session_id.")

    def to_record(self) -> dict[str, Any]:
        return {
            "record_type": "venue_pipeline_report",
            "schema_version": VENUE_PIPELINE_REPORT_SCHEMA_VERSION,
            "session_id": self.session_id,
            "identity_scope": "venue_local_temporary",
            "association_scope": "cross_camera_pairwise",
        }


def _tracklet_key_record(key: tuple[str, int, int]) -> dict[str, Any]:
    """One ``TrackletKey`` — ``(camera_id, stream_epoch, track_id)`` — as data.

    Association's and identity's ``TrackletKey`` aliases are independently
    declared but structurally identical (see
    :mod:`atreides_vision.fusion.integration`); this reporting module reads
    both meanings and is the natural place to serialise either without
    picking a side.
    """
    camera_id, stream_epoch, track_id = key
    return {"camera_id": camera_id, "stream_epoch": stream_epoch, "track_id": track_id}


def _membership_outcome_record(outcome: MembershipOutcome) -> dict[str, Any]:
    """One decision's fate against the venue identity registry.

    Deliberately references the decision by its stable pair key rather than
    duplicating the full :class:`~atreides_vision.association.models.PairDecision`:
    that decision is already present in full in the same instant's
    ``association.decisions``, and repeating it here would let the two copies
    drift out of one another's sight.
    """
    first_key, second_key = outcome.key
    return {
        "outcome": outcome.outcome.value,
        "detail": outcome.detail,
        "pair": {
            "first": _tracklet_key_record(first_key),
            "second": _tracklet_key_record(second_key),
        },
        "subject_ids": [subject_id.to_record() for subject_id in outcome.subject_ids],
    }


def instant_record(
    result: VenueInstantResult, registry: VenueSubjectRegistry, session_id: str
) -> dict[str, Any]:
    """One venue instant's full association/fusion/identity trail, as data.

    ``registry`` must reflect the state immediately after ``result`` was
    produced — the ordinary case of calling this right after
    :meth:`~atreides_vision.venue_pipeline.VenuePipeline.process_instant`,
    before any later instant or :meth:`~atreides_vision.venue_pipeline.VenuePipeline.expire`
    call has had a chance to retire or purge a subject this instant reports
    as active. This function performs no registry mutation of its own.

    An instant with no observations produces a record with an empty
    ``association.decisions``, an empty ``integration.outcomes`` and whatever
    ``active_subjects`` the registry already held — a legal, fully-formed
    record, not a special case.
    """
    return {
        "record_type": "venue_instant",
        "schema_version": VENUE_PIPELINE_REPORT_SCHEMA_VERSION,
        "session_id": session_id,
        "venue_seconds": round(result.venue_seconds, _SECONDS_PRECISION),
        "association": result.round.to_record(),
        "integration": {
            "venue_seconds": round(result.integration.venue_seconds, _SECONDS_PRECISION),
            "outcomes": [
                _membership_outcome_record(outcome) for outcome in result.integration.outcomes
            ],
        },
        "active_subjects": [
            registry.get(subject_id).to_record() for subject_id in result.active_subjects
        ],
    }


class JsonlVenuePipelineReportWriter:
    """Streams venue instant reports to a ``.jsonl`` file, one line per record.

    Shares :class:`~atreides_vision.jsonl.JsonlStreamWriter` with every other
    output stream in this repository, so gzip selection, whole-line writes
    and lifecycle behave identically rather than being reimplemented.
    Instants are written as they are produced and never accumulated: a long
    session costs no more memory than a short one.

    A write failure raises :class:`~atreides_vision.errors.FusionError` —
    already the integration layer's own reporting-contract exception (see
    its docstring) — and never anything association, fusion or identity
    themselves raise. That keeps a reporting failure distinguishable from an
    ordinary association or integration outcome at every call site, and this
    writer never mutates the pipeline, engine or registry it observes: a
    failed or successful write changes nothing about their state.

    Use as a context manager::

        with JsonlVenuePipelineReportWriter(path, metadata) as writer:
            result = pipeline.process_instant(venue_seconds, observations)
            writer.write_instant(result, pipeline.registry)
    """

    def __init__(
        self,
        path: Path | str,
        metadata: VenuePipelineReportMetadata,
        compress: bool | None = None,
    ) -> None:
        self._metadata = metadata
        self._stream = JsonlStreamWriter(path, FusionError, "venue pipeline report", compress)
        self._instants_written = 0

        self._stream.write_record(metadata.to_record())

    @property
    def path(self) -> Path:
        return self._stream.path

    @property
    def metadata(self) -> VenuePipelineReportMetadata:
        return self._metadata

    @property
    def instants_written(self) -> int:
        return self._instants_written

    @property
    def is_compressed(self) -> bool:
        return self._stream.is_compressed

    def write_instant(self, result: VenueInstantResult, registry: VenueSubjectRegistry) -> None:
        """Append one venue instant's association/fusion/identity record.

        Exactly one JSON line per call, matching one call per processed
        venue instant: this method performs no batching, filtering or
        sampling of its own, so a reader can always recover the same number
        of instants the pipeline itself processed.
        """
        if self._stream.is_released:
            raise FusionError(
                f"Cannot write to a released venue pipeline report writer: {self._stream.path}"
            )
        self._stream.write_record(instant_record(result, registry, self._metadata.session_id))
        self._instants_written += 1

    def release(self) -> None:
        """Flush and close. Idempotent, and safe after an exception."""
        if not self._stream.is_released:
            self._stream.release()
            logger.debug(
                "Released venue pipeline report writer %s after %d instant(s)",
                self._stream.path,
                self._instants_written,
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

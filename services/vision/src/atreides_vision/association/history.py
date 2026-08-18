"""Short-lived evidence about candidate pairs, carried between venue instants.

Geometry can only ever say "these two are not impossible right now". Turning
that into an association needs the one thing a single instant cannot supply:
whether the agreement *held*. This module keeps that, and nothing else.

What is kept, and for how long
------------------------------
Per candidate pair: the length of the run of agreement currently in progress,
how long it has lasted in venue seconds, the weakest geometry trust and the
worst separation seen inside it, and how much room the preferred partner had
over its nearest rival. Aggregates, not a list of observations — the memory a
pair costs does not grow with how long it is watched.

A run ends the moment the agreement does: a contradicting observation, or a gap
longer than the configured tolerance, and the counters start again from zero. A
claim that two tracklets agreed for three seconds must not span a second in
which they did not.

State here is **operational and temporary**, in the strict sense the
architecture requires. It is keyed by ``(camera_id, stream_epoch, track_id)``
pairs, bounded by a TTL and a hard ceiling, evicted least-recently-observed
first, and discarded with the process. It is not an identity, not a history of
people, not a record that outlives the session, and it says nothing about who
anybody is.
"""

from __future__ import annotations

from dataclasses import dataclass

from atreides_vision.association.config import AssociationDecisionConfig
from atreides_vision.association.models import (
    CandidateKey,
    GeometryTrust,
    SupportSummary,
)
from atreides_vision.errors import AssociationError

__all__ = ["EvidenceSample", "PairEvidence", "PairEvidenceStore"]


@dataclass(frozen=True, slots=True)
class EvidenceSample:
    """One venue instant's worth of evidence about one candidate pair.

    Only pairs that were actually *measured* produce a sample. A pair a hard
    gate refused was never compared, and recording it would put a zero where a
    measurement does not exist.
    """

    #: Session-relative venue time of the instant this describes.
    venue_seconds: float
    #: Whether this instant supports the two tracklets being one subject:
    #: geometry admitted them, each was the other's nearest candidate on the
    #: other's camera, and no rival was close enough to make that a coin flip.
    #: Decided by the caller, which is the only place that can see the rivals.
    agrees: bool
    #: Weaker of the two contributing cameras' geometry trust.
    trust: GeometryTrust
    #: Measured separation, in metres.
    separation_m: float
    #: How much nearer the preferred partner was than the nearest rival on the
    #: same counterpart camera, in metres. ``None`` when no rival was offered —
    #: which is not the same as a margin of zero.
    rival_margin_m: float | None

    def __post_init__(self) -> None:
        if self.venue_seconds < 0:
            raise AssociationError(
                f"EvidenceSample.venue_seconds must be non-negative, got {self.venue_seconds!r}"
            )
        if self.separation_m < 0:
            raise AssociationError(
                f"EvidenceSample.separation_m must be non-negative, got {self.separation_m!r}"
            )


@dataclass(slots=True)
class PairEvidence:
    """The run of agreement in progress for one candidate pair.

    Mutable by design and never shared: the store owns exactly one of these per
    pair and hands out :class:`SupportSummary` snapshots, which are frozen.
    """

    key: CandidateKey
    #: Venue time of the most recent observation, agreeing or not.
    last_seconds: float
    #: Every observation ever recorded for this pair.
    total_samples: int = 0
    #: Consecutive agreeing observations in the run in progress.
    samples: int = 0
    started_seconds: float | None = None
    #: Venue time of the most recent *agreeing* observation in the run.
    agreed_seconds: float = 0.0
    trust: GeometryTrust = GeometryTrust.UNTRUSTED
    max_separation_m: float = 0.0
    min_rival_margin_m: float | None = None

    def observe(self, sample: EvidenceSample, config: AssociationDecisionConfig) -> None:
        """Fold one instant into the run, ending it if the agreement broke.

        Raises:
            AssociationError: if instants do not advance. Evidence replayed or
                reordered would produce a different answer for the same venue,
                which is exactly what determinism forbids.
        """
        if self.total_samples and sample.venue_seconds <= self.last_seconds:
            raise AssociationError(
                f"Evidence for {self.key} went backwards: venue second "
                f"{sample.venue_seconds} follows {self.last_seconds}. Instants must advance."
            )
        gap = sample.venue_seconds - self.last_seconds
        if self.total_samples and gap > config.max_support_gap_seconds:
            self._end_run()
        if sample.agrees:
            self._extend_run(sample)
        else:
            self._end_run()
            self.trust = sample.trust
            self.max_separation_m = sample.separation_m
        self.total_samples += 1
        self.last_seconds = sample.venue_seconds

    def summary(self) -> SupportSummary:
        """A frozen snapshot of what this pair has established so far."""
        return SupportSummary(
            samples=self.samples,
            total_samples=self.total_samples,
            span_seconds=(
                0.0 if self.started_seconds is None else self.agreed_seconds - self.started_seconds
            ),
            started_seconds=self.started_seconds,
            last_seconds=self.last_seconds,
            trust=self.trust,
            max_separation_m=self.max_separation_m,
            min_rival_margin_m=self.min_rival_margin_m,
        )

    def is_sustained(self, config: AssociationDecisionConfig) -> bool:
        """Whether the run in progress has earned an association.

        Both a count and a duration, because either alone can be satisfied by
        evidence that means nothing: a count is uninterpretable without a frame
        rate, which is never assumed here, and a duration alone could be two
        observations far apart.
        """
        summary = self.summary()
        return (
            summary.samples >= config.min_support_samples
            and summary.span_seconds >= config.min_support_seconds
        )

    def _extend_run(self, sample: EvidenceSample) -> None:
        if self.started_seconds is None:
            self.started_seconds = sample.venue_seconds
            self.samples = 0
            self.trust = sample.trust
            self.max_separation_m = sample.separation_m
            self.min_rival_margin_m = sample.rival_margin_m
        else:
            self.trust = self.trust.worst_of(sample.trust)
            self.max_separation_m = max(self.max_separation_m, sample.separation_m)
            self.min_rival_margin_m = _smaller_margin(
                self.min_rival_margin_m, sample.rival_margin_m
            )
        self.samples += 1
        self.agreed_seconds = sample.venue_seconds

    def _end_run(self) -> None:
        self.samples = 0
        self.started_seconds = None
        self.agreed_seconds = 0.0
        self.min_rival_margin_m = None


def _smaller_margin(left: float | None, right: float | None) -> float | None:
    """The tighter of two rival margins, treating "no rival" as no constraint."""
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


class PairEvidenceStore:
    """Bounded, session-scoped evidence for every pair under consideration.

    Two limits, both required: a TTL so a pair nobody has seen for a while stops
    occupying anything, and a hard ceiling so a venue churning through
    thousands of tracklets cannot grow this without limit. Eviction is by
    least-recently-observed and ties break on the pair's canonical key, so a
    full store behaves identically on two runs of the same input.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[CandidateKey, PairEvidence] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: CandidateKey) -> bool:
        return key in self._entries

    def keys(self) -> tuple[CandidateKey, ...]:
        """Every tracked pair, in canonical order — never dictionary order."""
        return tuple(sorted(self._entries))

    def observe(
        self,
        key: CandidateKey,
        sample: EvidenceSample,
        config: AssociationDecisionConfig,
    ) -> PairEvidence:
        """Record one instant for one pair, creating its entry if it is new."""
        evidence = self._entries.get(key)
        if evidence is None:
            evidence = PairEvidence(key=key, last_seconds=sample.venue_seconds)
            self._entries[key] = evidence
        evidence.observe(sample, config)
        return evidence

    def discard(self, key: CandidateKey) -> None:
        """Forget a pair entirely, so any later run starts from nothing.

        Used when a pair stops being comparable at all: whatever it agreed on
        before happened under conditions that no longer hold, and continuing the
        old run across the gap would assert agreement nobody measured.
        """
        self._entries.pop(key, None)

    def prune(self, venue_seconds: float, config: AssociationDecisionConfig) -> int:
        """Drop stale entries, then the oldest, until the ceiling is respected.

        Returns the number of entries dropped.
        """
        stale = [
            key
            for key, evidence in sorted(self._entries.items())
            if venue_seconds - evidence.last_seconds > config.evidence_ttl_seconds
        ]
        for key in stale:
            del self._entries[key]

        dropped = len(stale)
        excess = len(self._entries) - config.max_tracked_pairs
        if excess > 0:
            oldest = sorted(
                self._entries.items(), key=lambda entry: (entry[1].last_seconds, entry[0])
            )[:excess]
            for key, _ in oldest:
                del self._entries[key]
            dropped += excess
        return dropped

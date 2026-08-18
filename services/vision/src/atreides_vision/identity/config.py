"""Tunable parameters for the venue-local identity registry.

Venue-local identity is operational and temporary by construction (see
``atreides_vision.identity``), and these are the only two knobs that give
"temporary" a number: how long an inactive identity is allowed to sit idle
before this layer stops considering it current, and how long its record is
kept afterward so provenance stays queryable for a little while rather than
disappearing the instant it stops being useful.

**These are DEVELOPMENT defaults.** Nothing here was measured against a real
deployment. Each states which direction it errs in when it is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

from atreides_vision.errors import ConfigurationError

__all__ = ["VenueIdentityConfig"]


@dataclass(frozen=True, slots=True)
class VenueIdentityConfig:
    """Thresholds governing when a venue subject goes idle and when it is forgotten."""

    #: Longest venue time, in seconds, an active venue subject may go without
    #: a membership change before :meth:`VenueSubjectRegistry.expire` retires
    #: it automatically.
    #:
    #: Too small retires identities that are still being actively tracked but
    #: happen to have a quiet membership history; too large keeps a subject
    #: "active" long after every tracklet that made it up has gone stale,
    #: which risks a later tracklet being folded into a claim that is no
    #: longer meaningfully current.
    idle_expiry_seconds: float = 30.0

    #: How long, in seconds of venue time, a retired subject's record is kept
    #: after retirement before :meth:`VenueSubjectRegistry.expire` purges it.
    #:
    #: Purging frees the entry entirely — its provenance is no longer
    #: queryable afterward. Set well above a typical operator glance so a
    #: just-retired identity is still explainable; kept finite because
    #: identity here is temporary and dies with the session, not archived.
    retention_seconds: float = 60.0

    #: Hard ceiling on how many venue subjects the registry holds at once
    #: (active and retired-but-not-yet-purged combined). Reached only when a
    #: venue produces sustained churn faster than :meth:`expire` is called;
    #: :meth:`create` refuses rather than silently evicting an active subject.
    #:
    #: A ceiling on *identities*, not on tracklets or cameras, so it is
    #: deliberately generous: 4096 covers a venue with far more concurrent
    #: subjects than any real deployment expects before anything is refused.
    max_venue_subjects: int = 4096

    def __post_init__(self) -> None:
        for name, value in (
            ("idle_expiry_seconds", self.idle_expiry_seconds),
            ("retention_seconds", self.retention_seconds),
        ):
            if value <= 0:
                raise ConfigurationError(f"{name} must be positive, got {value}")
        if self.max_venue_subjects < 1:
            raise ConfigurationError(
                f"max_venue_subjects must be at least 1, got {self.max_venue_subjects}"
            )

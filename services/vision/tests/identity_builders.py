"""Builders for venue-identity tests: memberships and a legible config.

Every registry operation takes a :class:`TrackletMembership`, so the one thing
worth making terse is building one with just the field a test cares about.
"""

from __future__ import annotations

from atreides_vision.behavior.models import SubjectRef
from atreides_vision.identity import TrackletMembership, VenueIdentityConfig
from atreides_vision.timing import StreamRef

#: Small, exact numbers so idle/retention boundaries can be checked by hand.
CONFIG = VenueIdentityConfig(
    idle_expiry_seconds=10.0,
    retention_seconds=20.0,
    max_venue_subjects=8,
)


def membership(
    camera_id: str = "cam_a",
    track_id: int = 1,
    *,
    stream_epoch: int = 0,
    joined_at_seconds: float = 10.0,
) -> TrackletMembership:
    return TrackletMembership(
        subject=SubjectRef(camera_id, track_id),
        stream=StreamRef(camera_id, stream_epoch),
        joined_at_seconds=joined_at_seconds,
    )

"""Multi-camera venue ingestion: many streams, one timeline, one tick schedule.

The layer between camera-local video sources and everything above them. It
decodes, times and orders frames from any number of cameras — one is the
degenerate case, not a special mode — and says, at every venue instant, which
streams had a frame and what the others were doing.

It performs **no association**. There is no venue-local person identity here, no
cross-camera correspondence and no duplicate suppression; those are V1
capabilities that attach above this loop when they are built, not inside it.
"""

from __future__ import annotations

from atreides_vision.venue.manifest import (
    VENUE_MANIFEST_VERSION,
    VenueCamera,
    VenueSessionManifest,
    VenueStreamEntry,
    load_venue_manifest,
)
from atreides_vision.venue.models import (
    StreamAbsence,
    StreamActivity,
    VenueTick,
    stream_sort_key,
)
from atreides_vision.venue.reporting import (
    VENUE_SESSION_SCHEMA_VERSION,
    JsonlVenueSessionWriter,
    VenueSessionMetadata,
    VenueSessionSummary,
    VenueStreamProvenance,
    tick_record,
)
from atreides_vision.venue.scheduler import StreamCursor, merge_streams
from atreides_vision.venue.session import VenueSession, VideoSourceCursor
from atreides_vision.venue.timebase import VenueTimebase

__all__ = [
    "VENUE_MANIFEST_VERSION",
    "VENUE_SESSION_SCHEMA_VERSION",
    "JsonlVenueSessionWriter",
    "StreamAbsence",
    "StreamActivity",
    "StreamCursor",
    "VenueCamera",
    "VenueSession",
    "VenueSessionManifest",
    "VenueSessionMetadata",
    "VenueSessionSummary",
    "VenueStreamEntry",
    "VenueStreamProvenance",
    "VenueTick",
    "VenueTimebase",
    "VideoSourceCursor",
    "load_venue_manifest",
    "merge_streams",
    "stream_sort_key",
    "tick_record",
]

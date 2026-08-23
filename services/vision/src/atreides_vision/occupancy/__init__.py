"""Venue-level current occupancy and duplicate suppression — V0.4.6.

V0.4.4 wired camera-local observations through association and fusion into
temporary venue-local identity (:mod:`atreides_vision.venue_pipeline`); V0.4.5
made that trail observable. Nothing before this milestone turned it into a
venue-level *count*. This package is the first product-facing consumer of
that identity stack: it answers "how many people does this venue instant's
observations currently represent", suppressing a duplicate only where
association and fusion have already, confidently, said two camera-local
tracklets are one venue subject.

Where this sits
----------------
::

    camera-local tracking -> association -> fusion -> venue-local identity
      (V0.4.2)              (V0.4.2)        (V0.4.3b)  (V0.4.3a/V0.4.4)
      -> current venue occupancy                        <- V0.4.6 builds this

This package sits *above* :mod:`atreides_vision.venue_pipeline` and reads
only its already-produced :class:`~atreides_vision.venue_pipeline.VenueInstantResult`
and :class:`~atreides_vision.identity.registry.VenueSubjectRegistry` state.
It performs no association of its own, mutates neither the registry nor any
:class:`~atreides_vision.identity.models.VenueSubjectId`, and never merges
two already-distinct venue subjects to make a count tidier.

What "current" means here
--------------------------
Presence is the set of camera-local tracklets appearing in *one* venue
instant's :attr:`~atreides_vision.venue_pipeline.VenueInstantResult.observations`
— the same atomic, venue-time-aligned sampling unit V0.4.0 already
established as "now". No new timeout, retention window or smoothing policy
is introduced: :class:`~atreides_vision.identity.registry.VenueSubjectRegistry`
membership is *retained* identity provenance, not current presence, and this
package never treats the two as the same thing. A subject's tracklet that
stopped appearing in the observations drops out of this instant's occupancy
immediately, even though the registry still remembers it happened.

Duplicate suppression
----------------------
For each currently observed tracklet, :func:`~atreides_vision.occupancy.analyzer.compute_occupancy`
asks the registry, and only the registry, who currently owns it
(:meth:`~atreides_vision.identity.registry.VenueSubjectRegistry.owner_of`).
Tracklets sharing one live owner count once; every other tracklet — no
owner, or an owner not shared with anything else currently observed — counts
independently. Because ``owner_of`` only ever reflects a positive,
already-applied association (see :mod:`atreides_vision.fusion.integration`),
an ``AMBIGUOUS``, ``UNASSOCIATED`` or ``NOT_COMPARABLE`` decision, and a
``REFUSED`` fusion outcome, can never cause a suppression: this package
re-derives nothing association or fusion already declined to decide.

What this milestone does not do
--------------------------------
No cumulative unique-visitor counting, no entry/exit accounting, no
historical footfall, no persistent or cross-session identity, no ReID or
appearance evidence, no automatic merge of two already-distinct venue
subjects, no occupancy forecasting or capacity alerting, no dashboard, no
risk or decision engine, no new association or fusion policy.
"""

from __future__ import annotations

from atreides_vision.occupancy.analyzer import compute_occupancy
from atreides_vision.occupancy.models import OccupancyContribution, OccupancySnapshot
from atreides_vision.occupancy.reporting import (
    OCCUPANCY_REPORT_SCHEMA_VERSION,
    JsonlOccupancyReportWriter,
    OccupancyReportMetadata,
    occupancy_instant_record,
)

__all__ = [
    "OCCUPANCY_REPORT_SCHEMA_VERSION",
    "JsonlOccupancyReportWriter",
    "OccupancyContribution",
    "OccupancyReportMetadata",
    "OccupancySnapshot",
    "compute_occupancy",
    "occupancy_instant_record",
]

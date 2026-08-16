"""Order statistics, shared by every metric family in the harness.

Kept in its own module so the metric modules and the result models can both use
it without importing each other. It is the only place a percentile is computed,
which is what keeps "median track lifetime" and "median pair lifetime"
comparable rather than two conventions that happen to share a word.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Distribution", "percentile"]

_PRECISION = 4


@dataclass(frozen=True, slots=True)
class Distribution:
    """Order statistics of a sample, with the sample size attached.

    The count travels with the percentiles on purpose. "Median pair lifetime
    0.27 s" reads very differently when it summarises 4 episodes than when it
    summarises 40,000, and a reader who cannot see which is which will assume
    the flattering one.

    Percentiles use linear interpolation between the two nearest order
    statistics — the convention ``numpy.percentile`` uses by default — so a
    two-element sample has a meaningful median rather than an arbitrary pick.

    **Memory:** the sample is materialised and sorted, so building one costs
    memory linear in the number of *values*, not in clip length. Every caller in
    this package feeds it one value per track, per pair episode or per cluster
    composition episode — never one per frame — and each documents that scaling.
    """

    count: int
    mean: float | None = None
    median: float | None = None
    p10: float | None = None
    p90: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    @classmethod
    def from_values(cls, values: Iterable[float]) -> Distribution:
        """Summarise a sample. An empty sample yields a count of 0 and no statistics.

        An empty sample is a real outcome — a clip in which no pair ever formed
        has no pair lifetimes — and reporting ``0.0`` for its median would claim
        a measurement that was never taken.
        """
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return cls(count=0)
        return cls(
            count=len(ordered),
            mean=math.fsum(ordered) / len(ordered),
            median=percentile(ordered, 50.0),
            p10=percentile(ordered, 10.0),
            p90=percentile(ordered, 90.0),
            minimum=ordered[0],
            maximum=ordered[-1],
        )

    def to_record(self, digits: int = _PRECISION) -> dict[str, Any]:
        def rounded(value: float | None) -> float | None:
            return None if value is None else round(value, digits)

        return {
            "count": self.count,
            "mean": rounded(self.mean),
            "median": rounded(self.median),
            "p10": rounded(self.p10),
            "p90": rounded(self.p90),
            "min": rounded(self.minimum),
            "max": rounded(self.maximum),
        }


def percentile(ordered: Sequence[float], value: float) -> float:
    """Linear-interpolated percentile of an already-sorted, non-empty sample.

    Raises:
        ValueError: on an empty sample. There is no defensible percentile of
            nothing, and returning 0.0 would put a fabricated measurement into
            a report.
    """
    if not ordered:
        raise ValueError("percentile() requires a non-empty sample")
    if len(ordered) == 1:
        return ordered[0]
    position = (value / 100.0) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

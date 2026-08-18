"""Short-lived pair evidence: what a run of agreement is, and what ends it.

The claim this module makes is narrow — "these two tracklets agreed, unbroken,
for this many observations and this much venue time" — and every test here
exists to stop that claim being stretched. A run must not span an instant that
contradicted it, must not span a gap nobody observed, and must not survive
longer than the store is allowed to remember it.
"""

from __future__ import annotations

import dataclasses

import pytest

from association_builders import DECIDING
from atreides_vision.association import (
    EvidenceSample,
    GeometryTrust,
    PairEvidence,
    PairEvidenceStore,
)
from atreides_vision.errors import AssociationError

#: A ceiling of one, so eviction can be exercised without building a crowd.
ONE_PAIR_ONLY = dataclasses.replace(DECIDING, max_tracked_pairs=1)

KEY = (("cam_a", 0, 1), ("cam_b", 0, 2))
OTHER = (("cam_a", 0, 1), ("cam_c", 0, 3))


def sample(
    venue_seconds: float,
    *,
    agrees: bool = True,
    trust: GeometryTrust = GeometryTrust.VERIFIED,
    separation_m: float = 0.5,
    rival_margin_m: float | None = None,
) -> EvidenceSample:
    return EvidenceSample(
        venue_seconds=venue_seconds,
        agrees=agrees,
        trust=trust,
        separation_m=separation_m,
        rival_margin_m=rival_margin_m,
    )


def evidence_over(*samples: EvidenceSample) -> PairEvidence:
    evidence = PairEvidence(key=KEY, last_seconds=samples[0].venue_seconds)
    for one in samples:
        evidence.observe(one, DECIDING)
    return evidence


class TestEvidenceSample:
    def test_a_sample_cannot_sit_before_the_session_began(self) -> None:
        with pytest.raises(AssociationError, match="venue_seconds"):
            sample(-0.5)

    def test_a_sample_cannot_carry_a_negative_separation(self) -> None:
        with pytest.raises(AssociationError, match="separation_m"):
            sample(10.0, separation_m=-0.5)

    def test_no_rival_is_recorded_as_absent_rather_than_as_zero(self) -> None:
        assert sample(10.0).rival_margin_m is None


class TestARunOfAgreement:
    def test_agreeing_instants_extend_one_run(self) -> None:
        evidence = evidence_over(sample(10.0), sample(10.125), sample(10.25))

        summary = evidence.summary()

        assert (summary.samples, summary.total_samples) == (3, 3)
        assert summary.span_seconds == 0.25
        assert summary.started_seconds == 10.0

    def test_a_contradiction_restarts_the_count_rather_than_pausing_it(self) -> None:
        evidence = evidence_over(sample(10.0), sample(10.125), sample(10.25, agrees=False))

        summary = evidence.summary()

        assert summary.samples == 0
        assert summary.span_seconds == 0.0
        assert summary.started_seconds is None
        assert summary.total_samples == 3

    def test_a_run_never_spans_an_instant_that_disagreed(self) -> None:
        evidence = evidence_over(
            sample(10.0), sample(10.125, agrees=False), sample(10.25), sample(10.375)
        )

        summary = evidence.summary()

        assert (summary.samples, summary.started_seconds) == (2, 10.25)
        assert summary.span_seconds == 0.125

    def test_a_gap_longer_than_the_tolerance_ends_the_run(self) -> None:
        """0.75 s of nobody looking is not 0.75 s of agreement."""
        evidence = evidence_over(sample(10.0), sample(10.125), sample(10.875))

        summary = evidence.summary()

        assert (summary.samples, summary.total_samples) == (1, 3)
        assert summary.started_seconds == 10.875

    def test_evidence_replayed_out_of_order_is_refused(self) -> None:
        evidence = evidence_over(sample(10.0), sample(10.125))

        with pytest.raises(AssociationError, match="went backwards"):
            evidence.observe(sample(10.0625), DECIDING)

    def test_the_same_instant_twice_is_refused(self) -> None:
        evidence = evidence_over(sample(10.0))

        with pytest.raises(AssociationError, match="Instants must advance"):
            evidence.observe(sample(10.0), DECIDING)


class TestWhatTheRunRemembers:
    def test_trust_is_the_weakest_the_run_ever_saw(self) -> None:
        evidence = evidence_over(
            sample(10.0),
            sample(10.125, trust=GeometryTrust.UNVERIFIED),
            sample(10.25),
        )

        assert evidence.summary().trust is GeometryTrust.UNVERIFIED

    def test_separation_is_the_worst_the_run_ever_saw(self) -> None:
        evidence = evidence_over(
            sample(10.0, separation_m=0.25),
            sample(10.125, separation_m=1.5),
            sample(10.25, separation_m=0.5),
        )

        assert evidence.summary().max_separation_m == 1.5

    def test_the_rival_margin_is_the_tightest_the_run_ever_saw(self) -> None:
        evidence = evidence_over(
            sample(10.0, rival_margin_m=2.0),
            sample(10.125, rival_margin_m=0.75),
            sample(10.25, rival_margin_m=None),
        )

        assert evidence.summary().min_rival_margin_m == 0.75

    def test_a_run_with_no_rival_at_all_reports_none(self) -> None:
        evidence = evidence_over(sample(10.0), sample(10.125))

        assert evidence.summary().min_rival_margin_m is None


class TestSustained:
    def test_enough_samples_over_enough_time_is_sustained(self) -> None:
        evidence = evidence_over(sample(10.0), sample(10.125), sample(10.25))

        assert evidence.is_sustained(DECIDING)

    def test_enough_samples_too_quickly_is_not_sustained(self) -> None:
        """A count alone would be a frame rate assumption, and none is made."""
        evidence = evidence_over(sample(10.0), sample(10.03125), sample(10.0625))

        assert evidence.summary().samples == 3
        assert not evidence.is_sustained(DECIDING)

    def test_enough_time_with_too_few_samples_is_not_sustained(self) -> None:
        evidence = evidence_over(sample(10.0), sample(10.375))

        assert evidence.summary().span_seconds == 0.375
        assert not evidence.is_sustained(DECIDING)

    def test_a_broken_run_stops_being_sustained_immediately(self) -> None:
        evidence = evidence_over(sample(10.0), sample(10.125), sample(10.25))
        evidence.observe(sample(10.375, agrees=False), DECIDING)

        assert not evidence.is_sustained(DECIDING)


class TestTheStoreIsBounded:
    def test_a_new_pair_is_created_on_first_sight(self) -> None:
        store = PairEvidenceStore()

        store.observe(KEY, sample(10.0), DECIDING)

        assert len(store) == 1
        assert KEY in store

    def test_keys_come_back_in_canonical_order_not_insertion_order(self) -> None:
        store = PairEvidenceStore()
        store.observe(OTHER, sample(10.0), DECIDING)
        store.observe(KEY, sample(10.0), DECIDING)

        assert store.keys() == (KEY, OTHER)

    def test_a_discarded_pair_starts_from_nothing_next_time(self) -> None:
        store = PairEvidenceStore()
        store.observe(KEY, sample(10.0), DECIDING)
        store.observe(KEY, sample(10.125), DECIDING)

        store.discard(KEY)
        revived = store.observe(KEY, sample(10.25), DECIDING)

        assert revived.summary().total_samples == 1

    def test_a_pair_nobody_has_seen_for_longer_than_the_ttl_is_dropped(self) -> None:
        store = PairEvidenceStore()
        store.observe(KEY, sample(10.0), DECIDING)

        dropped = store.prune(11.5, DECIDING)

        assert (dropped, len(store)) == (1, 0)

    def test_a_pair_seen_within_the_ttl_survives(self) -> None:
        store = PairEvidenceStore()
        store.observe(KEY, sample(10.0), DECIDING)

        assert store.prune(11.0, DECIDING) == 0
        assert store.keys() == (KEY,)

    def test_churning_thousands_of_tracklets_cannot_grow_the_store(self) -> None:
        """The ceiling holds under ID churn, and the survivors are the recent ones."""
        store = PairEvidenceStore()
        for index in range(500):
            key = (("cam_a", 0, index), ("cam_b", 0, index))
            store.observe(key, sample(10.0 + index / 1000), DECIDING)

        store.prune(10.5, DECIDING)

        assert len(store) == DECIDING.max_tracked_pairs
        tracked = store.keys()
        surviving = {key[0][2] for key in tracked}
        assert min(surviving) == 500 - DECIDING.max_tracked_pairs

    def test_eviction_order_does_not_depend_on_insertion_order(self) -> None:
        recent, older = KEY, OTHER
        stores = []
        for order in ((older, recent), (recent, older)):
            store = PairEvidenceStore()
            for key in order:
                store.observe(key, sample(10.0 if key == older else 10.5), DECIDING)
            store.prune(10.5, ONE_PAIR_ONLY)
            stores.append(store.keys())

        assert stores[0] == stores[1] == (recent,)

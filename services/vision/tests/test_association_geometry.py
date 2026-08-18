"""The comparability maths, checked against arithmetic rather than against itself.

Every expectation here is computed by hand from the formula in the docstring, so
a change in behaviour fails the test instead of silently updating it.
"""

from __future__ import annotations

import pytest

from association_builders import GATING, NORMALIZED, observation
from atreides_vision.association import (
    AssociationGatingConfig,
    admissible_radius_m,
    earliest_possible_gap_seconds,
    latest_possible_gap_seconds,
    motion_allowance_m,
    nominal_gap_seconds,
    separation_metres,
    timing_slack_seconds,
)
from atreides_vision.errors import AssociationError, ConfigurationError
from atreides_vision.timing import TimeUncertainty


class TestVenueTimeGap:
    def test_gap_is_symmetric_and_absolute(self) -> None:
        early = observation("cam_a", 1, venue_seconds=10.0)
        late = observation("cam_b", 1, venue_seconds=10.75)

        assert nominal_gap_seconds(early, late) == pytest.approx(0.75)
        assert nominal_gap_seconds(late, early) == pytest.approx(0.75)

    def test_two_observations_at_one_instant_have_no_gap(self) -> None:
        assert nominal_gap_seconds(observation("cam_a"), observation("cam_b")) == 0.0


class TestTimingSlack:
    def test_slack_is_the_sum_of_both_half_widths(self) -> None:
        slack = timing_slack_seconds(TimeUncertainty.declared(0.25), TimeUncertainty.declared(0.75))

        assert slack == pytest.approx(1.0)

    @pytest.mark.parametrize("position", [0, 1])
    def test_an_unestablished_stream_yields_no_slack_at_all(self, position: int) -> None:
        """``None`` is not zero: nobody related that stream to the venue clock."""
        uncertainties = [TimeUncertainty.declared(0.25), TimeUncertainty.declared(0.25)]
        uncertainties[position] = TimeUncertainty.unknown()

        assert timing_slack_seconds(uncertainties[0], uncertainties[1]) is None

    def test_earliest_possible_gap_never_goes_below_zero(self) -> None:
        assert earliest_possible_gap_seconds(0.2, 0.5) == 0.0

    def test_earliest_and_latest_bracket_the_stated_gap(self) -> None:
        assert earliest_possible_gap_seconds(2.0, 0.5) == pytest.approx(1.5)
        assert latest_possible_gap_seconds(2.0, 0.5) == pytest.approx(2.5)


class TestSeparation:
    def test_separation_is_euclidean_in_venue_metres(self) -> None:
        first = observation("cam_a", 1, x=1.0, y=2.0)
        second = observation("cam_b", 1, x=4.0, y=6.0)

        assert separation_metres(first, second) == pytest.approx(5.0)

    def test_image_plane_coordinates_are_refused(self) -> None:
        """A frame fraction on one camera has no relationship to one on another."""
        normalized = observation("cam_a", 1, space=NORMALIZED)
        metric = observation("cam_b", 1)

        with pytest.raises(AssociationError, match="not calibrated metres"):
            separation_metres(normalized, metric)


class TestAdmissibleRadius:
    def test_radius_is_tolerance_plus_speed_times_time(self) -> None:
        """1.0 m + 2.0 m/s * 0.5 s = 2.0 m under the test config."""
        assert motion_allowance_m(0.5, GATING) == pytest.approx(1.0)
        assert admissible_radius_m(0.5, GATING) == pytest.approx(2.0)

    def test_a_co_timed_pair_still_gets_the_position_tolerance(self) -> None:
        assert admissible_radius_m(0.0, GATING) == pytest.approx(1.0)

    def test_worse_timing_widens_the_radius_and_never_narrows_it(self) -> None:
        """A gate must not derive a rejection from precision nobody established."""
        radii = [admissible_radius_m(gap, GATING) for gap in (0.0, 0.5, 5.0, 60.0)]

        assert radii == sorted(radii)
        assert radii[-1] == pytest.approx(121.0)

    def test_a_negative_gap_is_refused(self) -> None:
        with pytest.raises(AssociationError, match="cannot be negative"):
            motion_allowance_m(-0.1, GATING)


class TestGatingConfig:
    @pytest.mark.parametrize(
        "field",
        [
            "max_pairing_gap_seconds",
            "max_subject_speed_m_per_s",
            "position_tolerance_m",
            "ambiguity_margin_m",
        ],
    )
    def test_every_tunable_must_be_positive(self, field: str) -> None:
        with pytest.raises(ConfigurationError, match=field):
            AssociationGatingConfig(**{field: 0.0})

    def test_the_shipped_defaults_are_metric_and_positive(self) -> None:
        """There is no normalized variant: cross-camera geometry needs metres."""
        config = AssociationGatingConfig()

        assert config.position_tolerance_m > 0
        assert config.max_subject_speed_m_per_s > 0
        assert admissible_radius_m(0.0, config) == config.position_tolerance_m

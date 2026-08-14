"""Frame-relative overlay sizing.

The property that matters: the overlay must stay legible on a 1080p frame and
must not swallow a small CCTV frame. Both are asserted numerically here so a
future style change cannot quietly reintroduce the 384x288 defect.
"""

from __future__ import annotations

import pytest

from atreides_vision.visualization.layout import (
    COMPACT_BELOW_FACTOR,
    MAX_PANEL_HEIGHT_FRACTION,
    MAX_PANEL_WIDTH_FRACTION,
    REFERENCE_HEIGHT,
    panel_layout,
    resolve_overlay_scale,
)
from atreides_vision.visualization.overlay import StatusOverlay

FULL_HD = (1920, 1080)
CIF = (384, 288)
STATUS = StatusOverlay(frame_index=1234, processing_fps=18.6, track_count=9)
ZONE_STATUS = StatusOverlay(frame_index=1234, processing_fps=18.6, track_count=9, zone_count=4)


class TestResolveOverlayScale:
    def test_reference_height_reproduces_the_authored_sizes(self) -> None:
        """1080p must look exactly as it did before V0.2.1."""
        scale = resolve_overlay_scale(*FULL_HD)
        assert scale.factor == pytest.approx(1.0)
        assert scale.font_scale == pytest.approx(0.5)
        assert scale.box_thickness == 2
        assert scale.trail_thickness == 2
        assert scale.outline_thickness == 2
        assert scale.padding == 8
        assert scale.compact is False

    def test_small_frame_shrinks_every_dimension(self) -> None:
        big = resolve_overlay_scale(*FULL_HD)
        small = resolve_overlay_scale(*CIF)
        assert small.factor < big.factor
        assert small.font_scale < big.font_scale
        assert small.padding < big.padding
        assert small.box_thickness <= big.box_thickness
        assert small.compact is True

    def test_thicknesses_never_vanish(self) -> None:
        """A zero-thickness rectangle draws nothing; 1 px is the floor."""
        scale = resolve_overlay_scale(64, 48)
        assert scale.box_thickness >= 1
        assert scale.trail_thickness >= 1
        assert scale.outline_thickness >= 1
        assert scale.padding >= 2

    def test_font_scale_has_a_readability_floor(self) -> None:
        assert resolve_overlay_scale(64, 48).font_scale >= 0.30

    def test_font_scale_has_a_ceiling(self) -> None:
        """4K must not render a billboard."""
        assert resolve_overlay_scale(3840, 2160).font_scale <= 1.10

    def test_scaling_is_monotonic_in_frame_height(self) -> None:
        heights = [144, 240, 288, 360, 480, 720, 1080, 1440, 2160]
        scales = [resolve_overlay_scale(round(h * 16 / 9), h).font_scale for h in heights]
        assert scales == sorted(scales)

    def test_is_driven_by_height_not_a_named_resolution(self) -> None:
        """Two different widths at one height scale identically."""
        assert (
            resolve_overlay_scale(384, 288).font_scale == resolve_overlay_scale(512, 288).font_scale
        )

    def test_compact_threshold_is_frame_relative(self) -> None:
        cutoff = REFERENCE_HEIGHT * COMPACT_BELOW_FACTOR
        assert resolve_overlay_scale(640, int(cutoff) - 1).compact is True
        assert resolve_overlay_scale(640, int(cutoff) + 1).compact is False

    def test_common_broadcast_sizes_keep_full_labels(self) -> None:
        """360p and above still have room for the full wording."""
        for width, height in ((640, 360), (854, 480), (1280, 720)):
            assert resolve_overlay_scale(width, height).compact is False

    def test_rejects_degenerate_dimensions(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            resolve_overlay_scale(0, 100)
        with pytest.raises(ValueError, match="positive"):
            resolve_overlay_scale(100, -1)

    def test_config_base_values_are_honoured(self) -> None:
        scale = resolve_overlay_scale(*FULL_HD, base_font_scale=0.8, base_box_thickness=4)
        assert scale.font_scale == pytest.approx(0.8)
        assert scale.box_thickness == 4


class TestPanelLayout:
    @pytest.mark.parametrize("size", [FULL_HD, CIF, (640, 360), (1280, 720)])
    @pytest.mark.parametrize("status", [STATUS, ZONE_STATUS])
    def test_panel_stays_inside_its_budget(
        self, size: tuple[int, int], status: StatusOverlay
    ) -> None:
        width, height = size
        scale = resolve_overlay_scale(width, height)
        layout = panel_layout(width, height, status.lines(scale.compact), scale)
        assert layout.width <= width * MAX_PANEL_WIDTH_FRACTION + 1
        assert layout.height <= height * MAX_PANEL_HEIGHT_FRACTION + 1

    @pytest.mark.parametrize("size", [FULL_HD, CIF, (160, 120), (64, 48)])
    def test_panel_never_exceeds_the_frame(self, size: tuple[int, int]) -> None:
        width, height = size
        scale = resolve_overlay_scale(width, height)
        layout = panel_layout(width, height, STATUS.lines(scale.compact), scale)
        assert 0 <= layout.width <= width
        assert 0 <= layout.height <= height

    def test_the_cif_defect_is_fixed(self) -> None:
        """The regression this work exists for.

        Pre-V0.2.1 the status panel covered ~19% of a 384x288 frame. The budget
        alone caps it well below that.
        """
        scale = resolve_overlay_scale(*CIF)
        layout = panel_layout(*CIF, STATUS.lines(scale.compact), scale)
        coverage = (layout.width * layout.height) / (CIF[0] * CIF[1])
        assert coverage < 0.10, f"panel covers {coverage:.1%} of the frame"

    def test_full_hd_panel_is_a_small_share_of_the_frame(self) -> None:
        scale = resolve_overlay_scale(*FULL_HD)
        layout = panel_layout(*FULL_HD, ZONE_STATUS.lines(scale.compact), scale)
        coverage = (layout.width * layout.height) / (FULL_HD[0] * FULL_HD[1])
        assert coverage < 0.05

    def test_baselines_are_ordered_and_inside_the_panel(self) -> None:
        scale = resolve_overlay_scale(*CIF)
        lines = STATUS.lines(scale.compact)
        layout = panel_layout(*CIF, lines, scale)
        baselines = [layout.baseline(i) for i in range(len(lines))]
        assert baselines == sorted(baselines)
        assert baselines[0] > 0
        assert baselines[-1] <= layout.height

    def test_lines_do_not_overlap(self) -> None:
        scale = resolve_overlay_scale(*CIF)
        layout = panel_layout(*CIF, STATUS.lines(scale.compact), scale)
        assert layout.line_height > 0
        assert layout.baseline(1) - layout.baseline(0) == layout.line_height

    def test_more_lines_makes_a_taller_panel(self) -> None:
        scale = resolve_overlay_scale(*FULL_HD)
        short = panel_layout(*FULL_HD, ("one",), scale)
        tall = panel_layout(*FULL_HD, ("one", "two", "three"), scale)
        assert tall.height > short.height

    def test_longer_text_never_overflows_the_width_budget(self) -> None:
        scale = resolve_overlay_scale(*CIF)
        layout = panel_layout(*CIF, ("X" * 200,), scale)
        assert layout.width <= CIF[0]

    def test_empty_lines_produce_an_empty_panel(self) -> None:
        scale = resolve_overlay_scale(*FULL_HD)
        layout = panel_layout(*FULL_HD, (), scale)
        assert layout.width == 0
        assert layout.height == 0

    def test_font_never_shrinks_below_the_floor(self) -> None:
        scale = resolve_overlay_scale(120, 90)
        layout = panel_layout(120, 90, ("ATREIDES // ZONE INTELLIGENCE",), scale)
        assert layout.font_scale >= 0.30


class TestCompactStatusLines:
    def test_compact_keeps_every_number(self) -> None:
        """Abbreviation may shorten labels; it must not drop data."""
        full = STATUS.lines(compact=False)
        compact = STATUS.lines(compact=True)
        assert len(compact) == len(full)
        for value in ("1234", "18.6", "9"):
            assert any(value in line for line in compact)

    def test_compact_is_narrower(self) -> None:
        full = STATUS.lines(compact=False)
        compact = STATUS.lines(compact=True)
        assert max(len(x) for x in compact) < max(len(x) for x in full)

    def test_zone_variant_keeps_the_zone_count(self) -> None:
        compact = ZONE_STATUS.lines(compact=True)
        assert len(compact) == 5
        assert any("4" in line for line in compact)

    def test_default_is_the_full_wording(self) -> None:
        assert STATUS.lines() == STATUS.lines(compact=False)
        assert STATUS.lines()[0] == "ATREIDES // SEE + TRACK"

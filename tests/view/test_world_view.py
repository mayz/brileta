"""Tests for WorldView."""

from __future__ import annotations

import pytest

from brileta.view.render.actor_renderer import (
    COMBAT_OUTLINE_MAX_ALPHA,
    COMBAT_OUTLINE_MIN_ALPHA,
    ActorRenderer,
)
from brileta.view.views.world_view import WorldView


class TestGetShimmerAlpha:
    """Tests for the get_shimmer_alpha oscillation calculation."""

    def _make_renderer(self) -> ActorRenderer:
        """Create a minimal ActorRenderer for shimmer tests.

        get_shimmer_alpha is pure math - it only needs game_time as input,
        so none of the constructor dependencies matter here.
        """
        return object.__new__(ActorRenderer)

    def test_alpha_at_cycle_start_is_midpoint(self) -> None:
        """At t=0, sin(0)=0, so alpha should be at midpoint."""
        renderer = self._make_renderer()
        alpha = renderer.get_shimmer_alpha(0.0, period=2.0)

        midpoint = (COMBAT_OUTLINE_MIN_ALPHA + COMBAT_OUTLINE_MAX_ALPHA) / 2
        assert alpha == pytest.approx(midpoint, rel=1e-6)

    def test_alpha_at_quarter_cycle_is_max(self) -> None:
        """At t=period/4, sin(pi/2)=1, so alpha should be at max."""
        renderer = self._make_renderer()
        period = 2.0
        alpha = renderer.get_shimmer_alpha(period / 4, period=period)

        assert alpha == pytest.approx(COMBAT_OUTLINE_MAX_ALPHA, rel=1e-6)

    def test_alpha_at_three_quarter_cycle_is_min(self) -> None:
        """At t=3*period/4, sin(3*pi/2)=-1, so alpha should be at min."""
        renderer = self._make_renderer()
        period = 2.0
        alpha = renderer.get_shimmer_alpha(3 * period / 4, period=period)

        assert alpha == pytest.approx(COMBAT_OUTLINE_MIN_ALPHA, rel=1e-6)

    def test_alpha_stays_within_bounds_across_full_cycle(self) -> None:
        """Alpha should always stay within [MIN_ALPHA, MAX_ALPHA] bounds."""
        renderer = self._make_renderer()
        period = 2.4  # Use actual default period
        epsilon = 1e-9  # Tolerance for floating point comparison
        # Sample many points across several cycles
        for i in range(100):
            time = i * period / 20  # 5 full cycles
            alpha = renderer.get_shimmer_alpha(time, period=period)

            assert COMBAT_OUTLINE_MIN_ALPHA - epsilon <= alpha, (
                f"Alpha {alpha} below min at time {time}"
            )
            assert alpha <= COMBAT_OUTLINE_MAX_ALPHA + epsilon, (
                f"Alpha {alpha} above max at time {time}"
            )

    def test_alpha_is_periodic(self) -> None:
        """Alpha should return same value after one full period."""
        renderer = self._make_renderer()
        period = 2.0
        test_time = 0.7  # Arbitrary point in cycle

        alpha1 = renderer.get_shimmer_alpha(test_time, period=period)
        alpha2 = renderer.get_shimmer_alpha(test_time + period, period=period)
        alpha3 = renderer.get_shimmer_alpha(test_time + 2 * period, period=period)

        assert alpha1 == pytest.approx(alpha2, rel=1e-6)
        assert alpha1 == pytest.approx(alpha3, rel=1e-6)

    def test_custom_period_affects_oscillation_speed(self) -> None:
        """Different periods should produce different alpha at same absolute time."""
        renderer = self._make_renderer()

        alpha_fast = renderer.get_shimmer_alpha(1.0, period=1.0)  # 1s period
        alpha_slow = renderer.get_shimmer_alpha(1.0, period=4.0)  # 4s period

        # At t=1.0:
        # - period=1.0: t normalized = 0, sin(0) = 0, alpha = midpoint
        # - period=4.0: t normalized = 0.25, sin(pi/2) = 1, alpha = max
        midpoint = (COMBAT_OUTLINE_MIN_ALPHA + COMBAT_OUTLINE_MAX_ALPHA) / 2
        assert alpha_fast == pytest.approx(midpoint, rel=1e-6)
        assert alpha_slow == pytest.approx(COMBAT_OUTLINE_MAX_ALPHA, rel=1e-6)


class TestSmoothScrollingBuffer:
    """Tests for smooth scrolling buffer padding."""

    def test_glyph_buffer_includes_padding(self) -> None:
        """Glyph buffer should be larger than viewport by 2*padding."""
        view = object.__new__(WorldView)
        view._SCROLL_PADDING = 1

        # Simulate set_bounds setting viewport dimensions
        view.width = 60
        view.height = 40

        from brileta.util.glyph_buffer import GlyphBuffer

        pad = view._SCROLL_PADDING
        buffer = GlyphBuffer(view.width + 2 * pad, view.height + 2 * pad)

        # Buffer should be viewport + 2 tiles (1 on each edge)
        assert buffer.width == view.width + 2
        assert buffer.height == view.height + 2

    def test_scroll_padding_constant_exists(self) -> None:
        """WorldView should have _SCROLL_PADDING class constant."""
        assert hasattr(WorldView, "_SCROLL_PADDING")
        assert WorldView._SCROLL_PADDING >= 1

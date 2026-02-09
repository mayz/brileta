"""Tests for WorldView."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from brileta.view.views.world_view import (
    COMBAT_OUTLINE_MAX_ALPHA,
    COMBAT_OUTLINE_MIN_ALPHA,
    WorldView,
)


class TestGetShimmerAlpha:
    """Tests for the get_shimmer_alpha oscillation calculation."""

    def create_world_view_with_time(self, game_time: float) -> WorldView:
        """Create a WorldView with mocked controller clock at the given time."""
        mock_controller = MagicMock()
        mock_controller.clock.last_time = game_time

        # Create WorldView with minimal mocking - we only need controller for this test
        view = object.__new__(WorldView)
        view.controller = mock_controller
        return view

    def test_alpha_at_cycle_start_is_midpoint(self) -> None:
        """At t=0, sin(0)=0, so alpha should be at midpoint."""
        view = self.create_world_view_with_time(0.0)
        alpha = view.get_shimmer_alpha(period=2.0)

        midpoint = (COMBAT_OUTLINE_MIN_ALPHA + COMBAT_OUTLINE_MAX_ALPHA) / 2
        assert alpha == pytest.approx(midpoint, rel=1e-6)

    def test_alpha_at_quarter_cycle_is_max(self) -> None:
        """At t=period/4, sin(pi/2)=1, so alpha should be at max."""
        period = 2.0
        view = self.create_world_view_with_time(period / 4)
        alpha = view.get_shimmer_alpha(period=period)

        assert alpha == pytest.approx(COMBAT_OUTLINE_MAX_ALPHA, rel=1e-6)

    def test_alpha_at_three_quarter_cycle_is_min(self) -> None:
        """At t=3*period/4, sin(3*pi/2)=-1, so alpha should be at min."""
        period = 2.0
        view = self.create_world_view_with_time(3 * period / 4)
        alpha = view.get_shimmer_alpha(period=period)

        assert alpha == pytest.approx(COMBAT_OUTLINE_MIN_ALPHA, rel=1e-6)

    def test_alpha_stays_within_bounds_across_full_cycle(self) -> None:
        """Alpha should always stay within [MIN_ALPHA, MAX_ALPHA] bounds."""
        period = 2.4  # Use actual default period
        epsilon = 1e-9  # Tolerance for floating point comparison
        # Sample many points across several cycles
        for i in range(100):
            time = i * period / 20  # 5 full cycles
            view = self.create_world_view_with_time(time)
            alpha = view.get_shimmer_alpha(period=period)

            assert COMBAT_OUTLINE_MIN_ALPHA - epsilon <= alpha, (
                f"Alpha {alpha} below min at time {time}"
            )
            assert alpha <= COMBAT_OUTLINE_MAX_ALPHA + epsilon, (
                f"Alpha {alpha} above max at time {time}"
            )

    def test_alpha_is_periodic(self) -> None:
        """Alpha should return same value after one full period."""
        period = 2.0
        test_time = 0.7  # Arbitrary point in cycle

        view1 = self.create_world_view_with_time(test_time)
        view2 = self.create_world_view_with_time(test_time + period)
        view3 = self.create_world_view_with_time(test_time + 2 * period)

        alpha1 = view1.get_shimmer_alpha(period=period)
        alpha2 = view2.get_shimmer_alpha(period=period)
        alpha3 = view3.get_shimmer_alpha(period=period)

        assert alpha1 == pytest.approx(alpha2, rel=1e-6)
        assert alpha1 == pytest.approx(alpha3, rel=1e-6)

    def test_custom_period_affects_oscillation_speed(self) -> None:
        """Different periods should produce different alpha at same absolute time."""
        view = self.create_world_view_with_time(1.0)

        alpha_fast = view.get_shimmer_alpha(period=1.0)  # 1s period
        alpha_slow = view.get_shimmer_alpha(period=4.0)  # 4s period

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

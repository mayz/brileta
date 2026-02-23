"""Tests for WorldView."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from brileta.view.render.actor_renderer import (
    COMBAT_OUTLINE_MAX_ALPHA,
    COMBAT_OUTLINE_MIN_ALPHA,
    ActorRenderer,
)
from brileta.view.views.world_view import (
    _EDGE_BLEND_CARDINAL_DIRECTIONS,
    WorldView,
    _compute_tile_edge_transition_metadata,
)


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


class TestSunDirectionPropagation:
    """Tests for WorldView sun-direction forwarding to graphics contexts."""

    def test_applies_direction_to_primary_graphics(self) -> None:
        view = object.__new__(WorldView)
        view.graphics = Mock()
        graphics_arg = view.graphics
        directional_light = SimpleNamespace(direction=SimpleNamespace(x=0.3, y=-0.7))

        view._apply_sun_direction_to_graphics(graphics_arg, directional_light)

        view.graphics.set_sun_direction.assert_called_once_with(0.3, -0.7)

    def test_applies_direction_to_both_contexts_when_distinct(self) -> None:
        view = object.__new__(WorldView)
        view.graphics = Mock()
        graphics_arg = Mock()
        directional_light = SimpleNamespace(direction=SimpleNamespace(x=-0.5, y=0.9))

        view._apply_sun_direction_to_graphics(graphics_arg, directional_light)

        view.graphics.set_sun_direction.assert_called_once_with(-0.5, 0.9)
        graphics_arg.set_sun_direction.assert_called_once_with(-0.5, 0.9)

    def test_uses_zero_vector_when_no_directional_light(self) -> None:
        view = object.__new__(WorldView)
        view.graphics = Mock()
        graphics_arg = view.graphics

        view._apply_sun_direction_to_graphics(graphics_arg, None)

        view.graphics.set_sun_direction.assert_called_once_with(0.0, 0.0)


class TestTileEdgeTransitionMetadata:
    """Tests for vectorized organic tile edge metadata generation."""

    def test_uses_cardinals_one_sided_ownership_and_no_wraparound(self) -> None:
        tile_ids = np.array(
            [
                [1, 5, 1],
                [5, 2, 2],
                [3, 5, 4],
            ],
            dtype=np.int32,
        )
        drawn_mask = np.ones((3, 3), dtype=np.bool_)
        drawn_mask[2, 2] = False  # Hidden neighbor should not contribute
        edge_blend = np.full((3, 3), 0.5, dtype=np.float32)

        bg_rgb = np.zeros((3, 3, 3), dtype=np.uint8)
        bg_rgb[0, 1] = (10, 20, 30)  # west of center
        bg_rgb[1, 0] = (40, 50, 60)  # north of center
        bg_rgb[2, 2] = (70, 80, 90)  # southeast (hidden)

        neighbor_mask, neighbor_bg = _compute_tile_edge_transition_metadata(
            tile_id_window=tile_ids,
            edge_blend_window=edge_blend,
            drawn_mask_window=drawn_mask,
            bg_rgb_window=bg_rgb,
        )

        west_idx = _EDGE_BLEND_CARDINAL_DIRECTIONS.index((-1, 0))
        north_idx = _EDGE_BLEND_CARDINAL_DIRECTIONS.index((0, -1))
        east_idx = _EDGE_BLEND_CARDINAL_DIRECTIONS.index((1, 0))
        west_bit = 1 << west_idx
        north_bit = 1 << north_idx
        east_bit = 1 << east_idx

        center_mask = int(neighbor_mask[1, 1])
        assert center_mask & west_bit
        assert center_mask & north_bit
        assert neighbor_bg.shape[2] == len(_EDGE_BLEND_CARDINAL_DIRECTIONS)

        np.testing.assert_array_equal(neighbor_bg[1, 1, west_idx], (10, 20, 30))
        np.testing.assert_array_equal(neighbor_bg[1, 1, north_idx], (40, 50, 60))

        # One-sided ownership: west tile (id=5) should not also blend toward the
        # center tile (id=2) on the same boundary because 5 !< 2.
        assert (int(neighbor_mask[0, 1]) & east_bit) == 0

        # Left-edge cells should not see wrapped neighbors from the right edge.
        assert (int(neighbor_mask[0, 1]) & west_bit) == 0

    def test_higher_edge_blend_owns_boundary_even_with_higher_tile_id(self) -> None:
        """Ownership should follow edge_blend strength, not enum ordering alone."""
        tile_ids = np.array([[2, 0], [5, 0]], dtype=np.int32)  # x,y indexing
        edge_blend = np.array([[0.0, 0.0], [0.45, 0.0]], dtype=np.float32)
        drawn_mask = np.array([[True, False], [True, False]], dtype=np.bool_)
        bg_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        bg_rgb[0, 0] = (10, 10, 10)  # rigid neighbor (west)
        neighbor_mask, neighbor_bg = _compute_tile_edge_transition_metadata(
            tile_id_window=tile_ids,
            edge_blend_window=edge_blend,
            drawn_mask_window=drawn_mask,
            bg_rgb_window=bg_rgb,
        )

        west_idx = _EDGE_BLEND_CARDINAL_DIRECTIONS.index((-1, 0))
        west_bit = 1 << west_idx

        # Grass-like tile (id 5) should own the boundary because it has higher
        # edge_blend than the rigid neighbor (id 2), despite its higher ID.
        assert int(neighbor_mask[1, 0]) & west_bit
        np.testing.assert_array_equal(neighbor_bg[1, 0, west_idx], (10, 10, 10))
        # Rigid tile remains non-owner.
        assert (
            int(neighbor_mask[0, 0])
            & (1 << _EDGE_BLEND_CARDINAL_DIRECTIONS.index((1, 0)))
        ) == 0


class TestActorParticleEmitterCache:
    """Tests for WorldView actor particle emitter caching."""

    def test_rebuilds_particle_emitter_cache_only_when_actor_revision_changes(
        self,
    ) -> None:
        emitter_effect_1 = Mock()
        emitter_effect_1.should_emit.return_value = True
        visuals_1 = Mock()
        visuals_1.has_continuous_effects.return_value = True
        visuals_1.continuous_effects = [emitter_effect_1]

        emitter_effect_2 = Mock()
        emitter_effect_2.should_emit.return_value = True
        visuals_2 = Mock()
        visuals_2.has_continuous_effects.return_value = True
        visuals_2.continuous_effects = [emitter_effect_2]

        emitter_actor_1 = Mock()
        emitter_actor_1.x = 2
        emitter_actor_1.y = 3
        emitter_actor_1.visual_effects = visuals_1

        emitter_actor_2 = Mock()
        emitter_actor_2.x = 4
        emitter_actor_2.y = 5
        emitter_actor_2.visual_effects = visuals_2

        static_actor = Mock()
        static_actor.visual_effects = None

        gw = SimpleNamespace(
            actors=[emitter_actor_1, static_actor],
            actors_revision=1,
        )
        view = object.__new__(WorldView)
        view.controller = SimpleNamespace(gw=gw)
        view.particle_system = Mock()
        view.environmental_system = Mock()
        view._particle_emitter_actors = set()
        view._particle_emitter_actors_revision = -1

        WorldView._update_actor_particles(view)

        assert view._particle_emitter_actors == {emitter_actor_1}
        emitter_effect_1.execute.assert_called_once()
        emitter_effect_2.execute.assert_not_called()

        # Actor list changed, so bump revision and rebuild cache to the new emitter.
        gw.actors = [emitter_actor_2]
        gw.actors_revision = 2

        WorldView._update_actor_particles(view)

        assert view._particle_emitter_actors == {emitter_actor_2}
        emitter_effect_2.execute.assert_called_once()

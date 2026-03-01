"""Tests for WorldView."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from brileta.environment import tile_types
from brileta.environment.generators.buildings import Building
from brileta.environment.tile_types import TileTypeID
from brileta.types import InterpolationAlpha
from brileta.util.coordinates import Rect
from brileta.view.render.actor_renderer import (
    COMBAT_OUTLINE_MAX_ALPHA,
    COMBAT_OUTLINE_MIN_ALPHA,
    ActorRenderer,
)
from brileta.view.views.world_view import (
    _EDGE_BLEND_CARDINAL_DIRECTIONS,
    WorldView,
    _compute_tile_edge_transition_metadata,
    _override_edge_neighbor_bg_with_self_darken,
    _RoofSubstitutionResult,
    _suppress_edge_blend_toward_hard_edges,
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


class TestMapUnlitDirtyTracking:
    """Tests for `_render_map_unlit()` GlyphBuffer dirty tracking."""

    def test_skips_rebuild_when_viewport_output_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from brileta.util.glyph_buffer import GlyphBuffer

        tiles = np.full((6, 6), TileTypeID.GRASS, dtype=np.uint8)
        explored = np.zeros((6, 6), dtype=bool)
        explored[0:3, 0:3] = True
        dark_appearance_map = tile_types.get_dark_appearance_map(tiles)
        game_map = SimpleNamespace(
            width=6,
            height=6,
            tiles=tiles,
            explored=explored,
            dark_appearance_map=dark_appearance_map,
            decoration_seed=123,
            structural_revision=7,
            exploration_revision=10,
        )

        view = object.__new__(WorldView)
        view._SCROLL_PADDING = 1
        view._map_unlit_buffer_cache_key = None
        view._viewport_zoom = 1.0
        view.map_glyph_buffer = GlyphBuffer(4, 4)
        view.controller = SimpleNamespace(
            gw=SimpleNamespace(
                game_map=game_map,
                player=SimpleNamespace(x=2, y=2),
            )
        )
        view.viewport_system = SimpleNamespace(
            offset_x=0,
            offset_y=0,
            camera=SimpleNamespace(world_x=1.25, world_y=1.75),
            viewport=SimpleNamespace(
                get_world_bounds=lambda _camera: SimpleNamespace(x1=0, y1=0, x2=2, y2=2)
            ),
        )
        view._compute_roof_state = lambda: (None, [])
        view._get_sun_direction_cache_key = lambda: None
        view._apply_roof_substitution = (
            lambda chars, fg_rgb, bg_rgb, tile_ids, *_args, **_kwargs: (
                _RoofSubstitutionResult(tile_ids)
            )
        )
        view._apply_tile_edge_transition_data = Mock()

        monkeypatch.setattr(
            tile_types, "apply_terrain_decoration", lambda *args, **kwargs: None
        )

        original_clear = view.map_glyph_buffer.clear
        clear_mock = Mock(wraps=original_clear)
        view.map_glyph_buffer.clear = clear_mock

        view._render_map_unlit()
        assert clear_mock.call_count == 1

        # Change exploration outside the rendered padded viewport slice only.
        game_map.explored[5, 5] = True
        game_map.exploration_revision += 1

        view._render_map_unlit()

        # The second call should return before clearing/refilling the buffer.
        assert clear_mock.call_count == 1


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

    @pytest.mark.parametrize(
        "architectural_tile_id",
        [TileTypeID.WALL, TileTypeID.ROOF_THATCH, TileTypeID.ROOF_SHINGLE],
    )
    def test_suppresses_blending_toward_architectural_neighbors_only(
        self, architectural_tile_id: TileTypeID
    ) -> None:
        """Grass-side roof/wall blending is suppressed without harming natural edges."""
        # x-major layout: [architectural][grass][dirt]
        tile_ids = np.array(
            [[architectural_tile_id], [TileTypeID.GRASS], [TileTypeID.DIRT]],
            dtype=np.int32,
        )
        edge_blend = tile_types.get_edge_blend_map(tile_ids.astype(np.uint8))
        drawn_mask = np.ones((3, 1), dtype=np.bool_)
        bg_rgb = np.zeros((3, 1, 3), dtype=np.uint8)
        bg_rgb[0, 0] = (5, 5, 5)
        bg_rgb[2, 0] = (40, 30, 20)

        neighbor_mask, _neighbor_bg = _compute_tile_edge_transition_metadata(
            tile_id_window=tile_ids,
            edge_blend_window=edge_blend,
            drawn_mask_window=drawn_mask,
            bg_rgb_window=bg_rgb,
        )

        west_idx = _EDGE_BLEND_CARDINAL_DIRECTIONS.index((-1, 0))
        east_idx = _EDGE_BLEND_CARDINAL_DIRECTIONS.index((1, 0))
        west_bit = 1 << west_idx
        east_bit = 1 << east_idx
        arch_east_bit = 1 << _EDGE_BLEND_CARDINAL_DIRECTIONS.index((1, 0))

        grass_mask_before = int(neighbor_mask[1, 0])
        arch_mask_before = int(neighbor_mask[0, 0])

        # Grass should still own its natural boundary with dirt.
        assert grass_mask_before & east_bit

        # Rigid architectural tiles (wall/shingle roof) force grass to own the
        # boundary pre-suppression. Thatch may own instead when tuned to higher
        # edge_blend so the roof can render a soft self-darkened edge.
        if architectural_tile_id in (TileTypeID.WALL, TileTypeID.ROOF_SHINGLE):
            assert grass_mask_before & west_bit

        _suppress_edge_blend_toward_hard_edges(
            edge_neighbor_mask=neighbor_mask, tile_id_window=tile_ids
        )

        # Grass-side architectural edge is always hard after suppression.
        assert (int(neighbor_mask[1, 0]) & west_bit) == 0
        assert int(neighbor_mask[1, 0]) & east_bit

        if architectural_tile_id == TileTypeID.ROOF_THATCH:
            # Suppression is one-sided: a thatch-owned roof edge remains available
            # so the roof can use self-darkened soft edges.
            assert arch_mask_before & arch_east_bit
            assert int(neighbor_mask[0, 0]) & arch_east_bit

    def test_thatch_edge_neighbor_colors_are_overridden_with_darkened_self_color(
        self,
    ) -> None:
        """Thatch roof edges should darken toward self-color, not neighbor color."""
        tile_ids = np.array(
            [[TileTypeID.GRASS], [TileTypeID.ROOF_THATCH], [TileTypeID.DIRT]],
            dtype=np.int32,
        )
        edge_blend = tile_types.get_edge_blend_map(tile_ids.astype(np.uint8))
        drawn_mask = np.ones((3, 1), dtype=np.bool_)
        bg_rgb = np.zeros((3, 1, 3), dtype=np.uint8)
        bg_rgb[0, 0] = (10, 90, 10)  # grass neighbor (green)
        bg_rgb[1, 0] = (120, 100, 70)  # thatch tile's own color
        bg_rgb[2, 0] = (80, 55, 35)  # dirt neighbor

        _neighbor_mask, neighbor_bg = _compute_tile_edge_transition_metadata(
            tile_id_window=tile_ids,
            edge_blend_window=edge_blend,
            drawn_mask_window=drawn_mask,
            bg_rgb_window=bg_rgb,
        )

        _override_edge_neighbor_bg_with_self_darken(
            edge_neighbor_bg=neighbor_bg,
            tile_id_window=tile_ids,
            bg_rgb_window=bg_rgb,
        )

        expected = np.array([85, 65, 35], dtype=np.uint8)  # 120/100/70 - 35
        for direction_index in range(len(_EDGE_BLEND_CARDINAL_DIRECTIONS)):
            np.testing.assert_array_equal(neighbor_bg[1, 0, direction_index], expected)

    def test_edge_neighbor_colors_are_unchanged_for_tiles_without_self_darken(
        self,
    ) -> None:
        """Tiles with edge_self_darken=0 keep their normal neighbor colors."""
        tile_ids = np.array([[TileTypeID.GRASS], [TileTypeID.DIRT]], dtype=np.int32)
        edge_blend = tile_types.get_edge_blend_map(tile_ids.astype(np.uint8))
        drawn_mask = np.ones((2, 1), dtype=np.bool_)
        bg_rgb = np.zeros((2, 1, 3), dtype=np.uint8)
        bg_rgb[0, 0] = (20, 60, 20)
        bg_rgb[1, 0] = (70, 50, 30)

        neighbor_mask, neighbor_bg = _compute_tile_edge_transition_metadata(
            tile_id_window=tile_ids,
            edge_blend_window=edge_blend,
            drawn_mask_window=drawn_mask,
            bg_rgb_window=bg_rgb,
        )
        neighbor_bg_before = neighbor_bg.copy()

        _override_edge_neighbor_bg_with_self_darken(
            edge_neighbor_bg=neighbor_bg,
            tile_id_window=tile_ids,
            bg_rgb_window=bg_rgb,
        )

        east_idx = _EDGE_BLEND_CARDINAL_DIRECTIONS.index((1, 0))
        east_bit = 1 << east_idx
        # Grass owns the grass->dirt boundary in this setup, so the slot is populated
        # and should remain the actual dirt color after the no-op override pass.
        assert int(neighbor_mask[0, 0]) & east_bit
        np.testing.assert_array_equal(neighbor_bg[0, 0, east_idx], bg_rgb[1, 0])
        np.testing.assert_array_equal(neighbor_bg, neighbor_bg_before)


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
        view.effect_library = Mock()
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


class TestRoofSubstitution:
    """Tests for WorldView virtual roof substitution helper."""

    def test_building_roof_color_offset_varies_by_position(self) -> None:
        building_a = Building(
            id=42,
            building_type="house",
            footprint=Rect(10, 10, 6, 6),
        )
        building_b = Building(
            id=42,
            building_type="house",
            footprint=Rect(18, 14, 6, 6),
        )

        offset_a = WorldView._building_roof_color_offset(
            building_a, decoration_seed=123
        )
        offset_b = WorldView._building_roof_color_offset(
            building_b, decoration_seed=123
        )

        assert offset_a != offset_b

    def test_building_roof_color_offset_is_deterministic(self) -> None:
        building = Building(
            id=7,
            building_type="house",
            footprint=Rect(3, 9, 5, 7),
        )

        first = WorldView._building_roof_color_offset(building, decoration_seed=99)
        second = WorldView._building_roof_color_offset(building, decoration_seed=99)

        assert first == second

    def test_building_roof_color_offset_is_bounded(self) -> None:
        for building_id in range(1, 8):
            for seed in (0, 1, 17, 999):
                building = Building(
                    id=building_id,
                    building_type="house",
                    footprint=Rect(2 * building_id, 3 * building_id, 6, 6),
                )
                offset = WorldView._building_roof_color_offset(
                    building, decoration_seed=seed
                )
                assert all(-8 <= channel <= 8 for channel in offset)

    def test_building_roof_color_offset_channels_can_differ(self) -> None:
        offsets = [
            WorldView._building_roof_color_offset(
                Building(
                    id=building_id,
                    building_type="house",
                    footprint=Rect(5 * building_id, 4 * building_id, 6, 6),
                ),
                decoration_seed=11,
            )
            for building_id in range(1, 8)
        ]

        assert any(len(set(offset)) > 1 for offset in offsets)

    def test_compute_roof_state_treats_regionless_archway_in_footprint_as_inside(
        self,
    ) -> None:
        view = object.__new__(WorldView)
        view._SCROLL_PADDING = 1
        view._roof_state_cache_key = None
        view._roof_state_cache_value = None

        building = Building(id=7, building_type="house", footprint=Rect(10, 10, 8, 8))
        view.controller = SimpleNamespace(
            gw=SimpleNamespace(
                player=SimpleNamespace(x=12, y=13),
                buildings=[building],
                game_map=SimpleNamespace(
                    structural_revision=0,
                    get_region_at=lambda _pos: None,
                ),
            )
        )
        view.viewport_system = SimpleNamespace(
            get_visible_bounds=lambda: Rect(0, 0, 40, 30)
        )

        player_building_id, viewport_buildings = view._compute_roof_state()

        assert player_building_id == 7
        assert viewport_buildings == [building]

    def test_applies_roof_only_to_building_footprint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Roof substitution covers the footprint interior and ignores tiles outside."""
        monkeypatch.setattr(tile_types, "apply_terrain_decoration", lambda *args: None)

        view = object.__new__(WorldView)
        view._roof_stamp_cache = {}
        view.controller = SimpleNamespace(
            gw=SimpleNamespace(player=SimpleNamespace(x=50, y=50))
        )
        view._get_directional_light = lambda: None
        view._compute_roof_state = lambda: (
            None,
            [Building(id=1, building_type="house", footprint=Rect(2, 2, 4, 4))],
        )

        # Two tiles inside footprint, two outside.
        world_x = np.array([3, 4, 1, 8], dtype=np.int32)
        world_y = np.array([3, 4, 2, 8], dtype=np.int32)
        tile_ids = np.array(
            [
                TileTypeID.FLOOR,  # interior -> roofed
                TileTypeID.FLOOR,  # interior -> roofed
                TileTypeID.GRASS,  # outside footprint -> untouched
                TileTypeID.DIRT,  # outside footprint -> untouched
            ],
            dtype=np.uint8,
        )
        chars = np.full(4, ord("?"), dtype=np.int32)
        fg_rgb = np.full((4, 3), 100, dtype=np.uint8)
        bg_rgb = np.full((4, 3), 120, dtype=np.uint8)

        effective_ids = view._apply_roof_substitution(
            chars,
            fg_rgb,
            bg_rgb,
            tile_ids,
            world_x,
            world_y,
            is_light=False,
            decoration_seed=0,
        ).effective_tile_ids

        roof_tid = TileTypeID.ROOF_THATCH
        roof_dark = tile_types.get_tile_type_data_by_id(roof_tid)["dark"]

        # Interior tiles get roof substitution.
        assert int(effective_ids[0]) == roof_tid
        assert int(effective_ids[1]) == roof_tid
        assert chars[0] == int(roof_dark["ch"])
        assert tuple(int(v) for v in fg_rgb[0]) != (100, 100, 100)
        assert tuple(int(v) for v in bg_rgb[0]) != (120, 120, 120)

        # Outside tiles remain untouched.
        assert int(effective_ids[2]) == TileTypeID.GRASS
        assert int(effective_ids[3]) == TileTypeID.DIRT
        assert chars[2] == ord("?")
        assert tuple(int(v) for v in fg_rgb[2]) == (100, 100, 100)
        assert tuple(int(v) for v in bg_rgb[2]) == (120, 120, 120)

    def test_roof_fast_path_with_buffer_lookup_matches_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Direct footprint indexing should preserve the legacy substitution output."""
        monkeypatch.setattr(tile_types, "apply_terrain_decoration", lambda *args: None)

        view = object.__new__(WorldView)
        view._roof_stamp_cache = {}
        view.controller = SimpleNamespace(
            gw=SimpleNamespace(player=SimpleNamespace(x=50, y=50))
        )
        view._get_directional_light = lambda: None
        building = Building(
            id=3,
            building_type="house",
            footprint=Rect(10, 20, 5, 4),
            door_positions=[(10, 21)],
        )
        view._compute_roof_state = lambda: (None, [building])

        world_origin_x = 8
        world_origin_y = 18
        buf_width = 12
        buf_height = 10

        # Sparse explored subset inside the visible buffer; includes roof tiles,
        # a doorway clearance tile, the chimney tile, and an outside tile.
        world_x = np.array([11, 12, 10, 13, 9], dtype=np.int32)
        world_y = np.array([21, 22, 21, 21, 19], dtype=np.int32)
        buf_x = world_x - world_origin_x
        buf_y = world_y - world_origin_y
        tile_ids = np.array(
            [
                TileTypeID.FLOOR,
                TileTypeID.FLOOR,
                TileTypeID.DOOR_CLOSED,
                TileTypeID.FLOOR,
                TileTypeID.GRASS,
            ],
            dtype=np.uint8,
        )
        chars = np.array(
            [ord("."), ord("."), ord("+"), ord("."), ord(",")], dtype=np.int32
        )
        fg_rgb = np.full((5, 3), 100, dtype=np.uint8)
        bg_rgb = np.full((5, 3), 120, dtype=np.uint8)
        building.chimney_offset = (3, 1)  # -> (13, 21), included above

        chars_fallback = chars.copy()
        fg_fallback = fg_rgb.copy()
        bg_fallback = bg_rgb.copy()
        ids_fallback = view._apply_roof_substitution(
            chars_fallback,
            fg_fallback,
            bg_fallback,
            tile_ids.copy(),
            world_x,
            world_y,
            is_light=False,
            decoration_seed=0,
        )

        chars_fast = chars.copy()
        fg_fast = fg_rgb.copy()
        bg_fast = bg_rgb.copy()
        ids_fast = view._apply_roof_substitution(
            chars_fast,
            fg_fast,
            bg_fast,
            tile_ids.copy(),
            world_x,
            world_y,
            is_light=False,
            decoration_seed=0,
            buf_x=buf_x,
            buf_y=buf_y,
            buf_width=buf_width,
            buf_height=buf_height,
            world_origin_x=world_origin_x,
            world_origin_y=world_origin_y,
        )

        np.testing.assert_array_equal(
            ids_fast.effective_tile_ids, ids_fallback.effective_tile_ids
        )
        np.testing.assert_array_equal(chars_fast, chars_fallback)
        np.testing.assert_array_equal(fg_fast, fg_fallback)
        np.testing.assert_array_equal(bg_fast, bg_fallback)

    def test_entry_doorway_remains_visible_through_roof(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Door tiles and their exterior approach tiles stay clear of roof."""
        monkeypatch.setattr(tile_types, "apply_terrain_decoration", lambda *args: None)

        view = object.__new__(WorldView)
        view._roof_stamp_cache = {}
        view.controller = SimpleNamespace(
            gw=SimpleNamespace(player=SimpleNamespace(x=50, y=50))
        )
        view._get_directional_light = lambda: None
        building = Building(
            id=1,
            building_type="house",
            footprint=Rect(4, 4, 4, 4),
            door_positions=[(4, 5)],  # West wall door
        )
        view._compute_roof_state = lambda: (
            None,
            [building],
        )

        # door tile (on footprint edge), interior roof tile
        world_x = np.array([4, 5], dtype=np.int32)
        world_y = np.array([5, 5], dtype=np.int32)
        tile_ids = np.array(
            [
                TileTypeID.DOOR_CLOSED,
                TileTypeID.FLOOR,
            ],
            dtype=np.uint8,
        )
        chars = np.array([ord("+"), ord(".")], dtype=np.int32)
        fg_rgb = np.array(
            [
                [210, 180, 120],
                [160, 140, 100],
            ],
            dtype=np.uint8,
        )
        bg_rgb = np.array(
            [
                [50, 30, 20],
                [60, 50, 35],
            ],
            dtype=np.uint8,
        )

        original_chars = chars.copy()
        original_fg = fg_rgb.copy()
        effective_ids = view._apply_roof_substitution(
            chars,
            fg_rgb,
            bg_rgb,
            tile_ids,
            world_x,
            world_y,
            is_light=True,
            decoration_seed=0,
        ).effective_tile_ids

        roof_light = tile_types.get_tile_type_data_by_id(TileTypeID.ROOF_THATCH)[
            "light"
        ]

        # Door tile stays clear (entrance clearance).
        assert int(effective_ids[0]) == TileTypeID.DOOR_CLOSED
        assert chars[0] == original_chars[0]
        np.testing.assert_array_equal(fg_rgb[0], original_fg[0])

        # Interior roof tile stays fully roofed.
        assert int(effective_ids[1]) == TileTypeID.ROOF_THATCH
        assert chars[1] == int(roof_light["ch"])
        assert tuple(int(v) for v in fg_rgb[1]) != tuple(int(v) for v in original_fg[1])
        assert tuple(int(v) for v in bg_rgb[1]) != tuple(int(v) for v in bg_rgb[0])

    def test_roof_covers_interior_tiles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Interior footprint tiles get roof substitution applied."""
        monkeypatch.setattr(tile_types, "apply_terrain_decoration", lambda *args: None)

        view = object.__new__(WorldView)
        view._roof_stamp_cache = {}
        view.controller = SimpleNamespace(
            gw=SimpleNamespace(player=SimpleNamespace(x=50, y=50))
        )
        view._get_directional_light = lambda: None
        building = Building(
            id=1,
            building_type="house",
            footprint=Rect(4, 4, 4, 4),
            door_positions=[(4, 5)],
        )
        view._compute_roof_state = lambda: (None, [building])

        world_x = np.array([5], dtype=np.int32)
        world_y = np.array([5], dtype=np.int32)
        tile_ids = np.array([TileTypeID.FLOOR], dtype=np.uint8)
        chars = np.array([ord(".")], dtype=np.int32)
        fg_rgb = np.array([[180, 160, 120]], dtype=np.uint8)
        bg_rgb = np.array([[70, 60, 45]], dtype=np.uint8)

        effective_ids = view._apply_roof_substitution(
            chars,
            fg_rgb,
            bg_rgb,
            tile_ids,
            world_x,
            world_y,
            is_light=True,
            decoration_seed=0,
        ).effective_tile_ids

        roof_light = tile_types.get_tile_type_data_by_id(TileTypeID.ROOF_THATCH)[
            "light"
        ]
        assert int(effective_ids[0]) == TileTypeID.ROOF_THATCH
        assert chars[0] == int(roof_light["ch"])

    @pytest.mark.parametrize("is_light", [False, True])
    def test_roof_perimeter_tiles_are_darkened_with_darker_corners(
        self, monkeypatch: pytest.MonkeyPatch, is_light: bool
    ) -> None:
        """Roof eaves darken perimeter tiles, with extra darkening on corners."""
        monkeypatch.setattr(tile_types, "apply_terrain_decoration", lambda *args: None)

        view = object.__new__(WorldView)
        view._roof_stamp_cache = {}
        view.controller = SimpleNamespace(
            gw=SimpleNamespace(player=SimpleNamespace(x=50, y=50))
        )
        view._get_directional_light = lambda: None
        building = Building(
            id=1,
            building_type="house",
            footprint=Rect(4, 4, 4, 4),
            door_positions=[(4, 5)],  # west wall door should remain clear
        )
        view._compute_roof_state = lambda: (None, [building])

        # corner roof, top-edge roof, left-edge roof, interior roof, door (clearance), outside
        # For this even-parity square footprint, ridge shading is horizontal. We pick:
        # - corner/top-edge on same row (same ridge brightness)
        # - left-edge/interior on same row (same ridge brightness)
        # so eave darkening deltas are isolated from ridge shading.
        building.door_positions = [(4, 6)]  # keep sampled left-edge tile roof-covered
        world_x = np.array([4, 5, 4, 5, 4, 9], dtype=np.int32)
        world_y = np.array([4, 4, 5, 5, 6, 9], dtype=np.int32)
        tile_ids = np.array(
            [
                TileTypeID.FLOOR,
                TileTypeID.FLOOR,
                TileTypeID.FLOOR,
                TileTypeID.FLOOR,
                TileTypeID.DOOR_CLOSED,
                TileTypeID.GRASS,
            ],
            dtype=np.uint8,
        )
        chars = np.array(
            [ord("."), ord("."), ord("."), ord("."), ord("+"), ord(",")], dtype=np.int32
        )
        fg_rgb = np.array(
            [
                [180, 170, 160],
                [181, 171, 161],
                [182, 172, 162],
                [183, 173, 163],
                [210, 180, 120],
                [90, 100, 110],
            ],
            dtype=np.uint8,
        )
        bg_rgb = np.array(
            [
                [70, 60, 50],
                [71, 61, 51],
                [72, 62, 52],
                [73, 63, 53],
                [50, 30, 20],
                [20, 30, 40],
            ],
            dtype=np.uint8,
        )

        original_chars = chars.copy()
        original_fg = fg_rgb.copy()
        original_bg = bg_rgb.copy()

        effective_ids = view._apply_roof_substitution(
            chars,
            fg_rgb,
            bg_rgb,
            tile_ids,
            world_x,
            world_y,
            is_light=is_light,
            decoration_seed=0,
        ).effective_tile_ids

        roof_appearance = tile_types.get_tile_type_data_by_id(TileTypeID.ROOF_THATCH)[
            "light" if is_light else "dark"
        ]

        # Roof substitution applied to all non-clearance footprint samples.
        assert int(effective_ids[0]) == TileTypeID.ROOF_THATCH
        assert int(effective_ids[1]) == TileTypeID.ROOF_THATCH
        assert int(effective_ids[2]) == TileTypeID.ROOF_THATCH
        assert int(effective_ids[3]) == TileTypeID.ROOF_THATCH
        assert chars[0] == int(roof_appearance["ch"])
        assert chars[1] == int(roof_appearance["ch"])
        assert chars[2] == int(roof_appearance["ch"])
        assert chars[3] == int(roof_appearance["ch"])

        # Only bg is eave-darkened. Same-row tiles share ridge shading, so fg should match.
        # With perspective shift (F=0.7), the visual roof north edge moves to y=3,
        # so y=4 tiles are no longer on the north edge - just west-edge vs interior.
        np.testing.assert_array_equal(fg_rgb[0], fg_rgb[1])  # west edge vs interior
        np.testing.assert_array_equal(fg_rgb[2], fg_rgb[3])  # west edge vs interior

        # Exact bg deltas for eave shading when ridge shading is held constant.
        # Both pairs are west-edge (-6 perimeter) vs interior (0).
        np.testing.assert_array_equal(
            bg_rgb[2].astype(np.int16) - bg_rgb[3].astype(np.int16),
            np.array([-6, -6, -6], dtype=np.int16),
        )
        np.testing.assert_array_equal(
            bg_rgb[0].astype(np.int16) - bg_rgb[1].astype(np.int16),
            np.array([-6, -6, -6], dtype=np.int16),
        )
        assert np.all(bg_rgb[2] < bg_rgb[3])  # perimeter darker than interior
        assert np.all(bg_rgb[0] < bg_rgb[1])  # perimeter darker than interior

        # Non-roof tiles are untouched, including perimeter door clearance.
        assert int(effective_ids[4]) == TileTypeID.DOOR_CLOSED
        assert int(effective_ids[5]) == TileTypeID.GRASS
        assert chars[4] == original_chars[4]
        assert chars[5] == original_chars[5]
        np.testing.assert_array_equal(fg_rgb[4], original_fg[4])
        np.testing.assert_array_equal(fg_rgb[5], original_fg[5])
        np.testing.assert_array_equal(bg_rgb[4], original_bg[4])
        np.testing.assert_array_equal(bg_rgb[5], original_bg[5])

    def test_ridge_shading_sun_facing_vs_shadow_vs_ridge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sun-facing tiles are brighter, shadow tiles darker, ridge tiles moderate."""
        monkeypatch.setattr(tile_types, "apply_terrain_decoration", lambda *args: None)

        view = object.__new__(WorldView)
        view._roof_stamp_cache = {}
        view.controller = SimpleNamespace(
            gw=SimpleNamespace(player=SimpleNamespace(x=50, y=50))
        )
        # Sun from north: sun_dy < 0 means north-facing slope catches the light.
        view._get_directional_light = lambda: SimpleNamespace(
            direction=SimpleNamespace(x=0.0, y=-1.0),
            elevation_degrees=45.0,
        )

        # Wide building with horizontal ridge. Footprint center_y = (10 + 18) / 2 = 14.
        # With perspective_north_offset=1.5, N=2: shifted_center = 14 - 2 = 12.0.
        building = Building(
            id=1,
            building_type="house",
            footprint=Rect(10, 10, 10, 8),
        )
        view._compute_roof_state = lambda: (None, [building])

        # Three interior tiles (none on perimeter) at different distances from
        # shifted ridge center at y=12.0:
        #   y=10: offset = -1.5, sun-facing (+18)
        #   y=12: offset = +0.5, ridge (+6)
        #   y=15: offset = +3.5, shadow (-18)
        world_x = np.array([12, 12, 12], dtype=np.int32)
        world_y = np.array([10, 12, 15], dtype=np.int32)
        tile_ids = np.full(3, TileTypeID.FLOOR, dtype=np.uint8)
        chars = np.full(3, ord("."), dtype=np.int32)
        fg_rgb = np.full((3, 3), 100, dtype=np.uint8)
        bg_rgb = np.full((3, 3), 100, dtype=np.uint8)

        view._apply_roof_substitution(
            chars,
            fg_rgb,
            bg_rgb,
            tile_ids,
            world_x,
            world_y,
            is_light=False,
            decoration_seed=0,
        )

        roof_dark = tile_types.get_tile_type_data_by_id(TileTypeID.ROOF_THATCH)["dark"]
        base_bg = np.asarray(roof_dark["bg"], dtype=np.int16)
        base_fg = np.asarray(roof_dark["fg"], dtype=np.int16)
        roof_tint = np.asarray(
            WorldView._building_roof_color_offset(building, decoration_seed=0),
            dtype=np.int16,
        )

        # Verify exact offsets for bg and fg.
        np.testing.assert_array_equal(
            bg_rgb[0], np.clip(base_bg + roof_tint + 18, 0, 255).astype(np.uint8)
        )
        np.testing.assert_array_equal(
            bg_rgb[1], np.clip(base_bg + roof_tint + 6, 0, 255).astype(np.uint8)
        )
        np.testing.assert_array_equal(
            bg_rgb[2], np.clip(base_bg + roof_tint - 18, 0, 255).astype(np.uint8)
        )
        np.testing.assert_array_equal(
            fg_rgb[0], np.clip(base_fg + 18, 0, 255).astype(np.uint8)
        )
        np.testing.assert_array_equal(
            fg_rgb[1], np.clip(base_fg + 6, 0, 255).astype(np.uint8)
        )
        np.testing.assert_array_equal(
            fg_rgb[2], np.clip(base_fg - 18, 0, 255).astype(np.uint8)
        )

        # Brightness ordering: sun > ridge > shadow
        assert np.all(bg_rgb[0] > bg_rgb[1])
        assert np.all(bg_rgb[1] > bg_rgb[2])

    def test_ridge_shading_vertical_ridge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Vertical ridge uses x-axis sun component for directional shading."""
        monkeypatch.setattr(tile_types, "apply_terrain_decoration", lambda *args: None)

        view = object.__new__(WorldView)
        view._roof_stamp_cache = {}
        view.controller = SimpleNamespace(
            gw=SimpleNamespace(player=SimpleNamespace(x=50, y=50))
        )
        # Sun from west: sun_dx < 0 means west-facing slope catches the light.
        view._get_directional_light = lambda: SimpleNamespace(
            direction=SimpleNamespace(x=-1.0, y=0.0),
            elevation_degrees=45.0,
        )

        # Tall building with vertical ridge. center_x = (10 + 16) / 2 = 13.0
        building = Building(
            id=1,
            building_type="house",
            footprint=Rect(10, 10, 6, 10),
        )
        view._compute_roof_state = lambda: (None, [building])

        # Interior tiles west and east of the vertical ridge (y=14, not on perimeter).
        #   x=11: offset = -1.5, sun-facing (+10)
        #   x=14: offset = +1.5, shadow (-10)
        world_x = np.array([11, 14], dtype=np.int32)
        world_y = np.array([14, 14], dtype=np.int32)
        tile_ids = np.full(2, TileTypeID.FLOOR, dtype=np.uint8)
        chars = np.full(2, ord("."), dtype=np.int32)
        fg_rgb = np.full((2, 3), 100, dtype=np.uint8)
        bg_rgb = np.full((2, 3), 100, dtype=np.uint8)

        view._apply_roof_substitution(
            chars,
            fg_rgb,
            bg_rgb,
            tile_ids,
            world_x,
            world_y,
            is_light=True,
            decoration_seed=0,
        )

        # West tile (sun-facing) should be brighter than east tile (shadow).
        assert np.all(bg_rgb[0] > bg_rgb[1])
        assert np.all(fg_rgb[0] > fg_rgb[1])

    def test_tags_occluded_actors_under_opaque_roof(self) -> None:
        """Actors under opaque roofs are tagged as occluded, not removed."""
        view = object.__new__(WorldView)

        # Use a simple hashable mock for actors (SimpleNamespace isn't hashable).
        class _MockActor:
            def __init__(self, x: int, y: int) -> None:
                self.x = x
                self.y = y

        player = _MockActor(3, 5)
        hidden_actor = _MockActor(5, 5)
        visible_actor = _MockActor(2, 2)

        building = Building(
            id=1,
            building_type="house",
            footprint=Rect(4, 4, 4, 4),
            door_positions=[(4, 5)],
        )

        tiles = np.full((12, 12), TileTypeID.GRASS, dtype=np.uint8)
        tiles[5, 5] = TileTypeID.FLOOR

        view.controller = SimpleNamespace(
            gw=SimpleNamespace(
                player=player,
                game_map=SimpleNamespace(tiles=tiles),
            )
        )
        view._compute_roof_state = lambda: (None, [building])

        all_actors = [player, hidden_actor, visible_actor]
        result_actors, occluded = view._filter_roof_occluded_actors(
            actors=all_actors,
            viewport_bounds_world=Rect(0, 0, 12, 12),
        )

        # All actors are kept in the list (none removed).
        assert result_actors is all_actors
        # The actor under the roof is tagged as occluded.
        assert hidden_actor in occluded
        # The player and visible actor are not tagged.
        assert player not in occluded
        assert visible_actor not in occluded

    def test_occlusion_covers_north_overhang_zone(self) -> None:
        """Actors in the north overhang zone (shifted roof) are tagged occluded."""
        view = object.__new__(WorldView)

        class _MockActor:
            def __init__(self, x: int, y: int) -> None:
                self.x = x
                self.y = y

        # Building with 2 floors: F=3.0, N=3, floor_offset=3, no frac.
        # Footprint y=[20, 30), roof shifts north by 3 tiles.
        # Visual roof covers y=[17, 27).
        building = Building(
            id=1,
            building_type="house",
            footprint=Rect(10, 20, 8, 10),
            floor_count=2,
        )

        actor_in_overhang = _MockActor(14, 18)  # y=18 is in overhang zone [17, 20)
        actor_in_wall_face = _MockActor(14, 28)  # y=28 is wall face zone [27, 30)
        actor_outside = _MockActor(14, 15)  # y=15 is north of visual roof

        view.controller = SimpleNamespace(
            gw=SimpleNamespace(
                player=_MockActor(50, 50),
                game_map=SimpleNamespace(
                    tiles=np.full((40, 40), TileTypeID.GRASS, dtype=np.uint8)
                ),
            )
        )
        view._compute_roof_state = lambda: (None, [building])

        actors = [actor_in_overhang, actor_in_wall_face, actor_outside]
        _, occluded = view._filter_roof_occluded_actors(
            actors=actors,
            viewport_bounds_world=Rect(0, 0, 40, 40),
        )

        # Actor in the north overhang is under the shifted roof.
        assert actor_in_overhang in occluded
        # Actor in the wall face zone is not occluded (wall is visible).
        assert actor_in_wall_face not in occluded
        # Actor outside the visual roof is not occluded.
        assert actor_outside not in occluded

    def test_integer_perspective_offset_no_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Integer perspective offset (frac=0) produces no south split tiles."""
        monkeypatch.setattr(tile_types, "apply_terrain_decoration", lambda *args: None)

        view = object.__new__(WorldView)
        view._roof_stamp_cache = {}
        view.controller = SimpleNamespace(
            gw=SimpleNamespace(player=SimpleNamespace(x=50, y=50))
        )
        view._get_directional_light = lambda: None

        # 2 floors * 1.5 = F=3.0, exactly integer. No fractional split.
        building = Building(
            id=1,
            building_type="house",
            footprint=Rect(10, 20, 6, 8),
            floor_count=2,
        )
        assert building.perspective_has_frac is False
        view._compute_roof_state = lambda: (None, [building])

        # South footprint row (y=27) should be a wall face tile, not a split.
        # Interior row (y=22) should be a roof tile.
        world_x = np.array([13, 13], dtype=np.int32)
        world_y = np.array([22, 27], dtype=np.int32)
        tile_ids = np.full(2, TileTypeID.FLOOR, dtype=np.uint8)
        chars = np.full(2, ord("."), dtype=np.int32)
        fg_rgb = np.full((2, 3), 100, dtype=np.uint8)
        bg_rgb = np.full((2, 3), 100, dtype=np.uint8)

        result = view._apply_roof_substitution(
            chars,
            fg_rgb,
            bg_rgb,
            tile_ids,
            world_x,
            world_y,
            is_light=False,
            decoration_seed=0,
        )

        # Interior tile gets roof substitution.
        assert int(result.effective_tile_ids[0]) == TileTypeID.ROOF_THATCH
        # No split data for integer offset (no fractional boundary tiles).
        assert result.split_y is None

    def test_split_data_flows_through_roof_substitution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fractional perspective offset produces split_y data for boundary tiles."""
        monkeypatch.setattr(tile_types, "apply_terrain_decoration", lambda *args: None)

        view = object.__new__(WorldView)
        view._roof_stamp_cache = {}
        view.controller = SimpleNamespace(
            gw=SimpleNamespace(player=SimpleNamespace(x=50, y=50))
        )
        view._get_directional_light = lambda: None

        # 1 floor * 1.5 = F=1.5, N=2, floor_offset=1, frac=0.5.
        # Footprint y=[10, 20). South split row = 20 - 1 - 1 = 18.
        # split_y_value = 1.0 - 0.5 = 0.5.
        building = Building(
            id=1,
            building_type="house",
            footprint=Rect(5, 10, 6, 10),
        )
        assert building.perspective_has_frac is True
        view._compute_roof_state = lambda: (None, [building])

        # Query the south split row (y=18) and a full roof row (y=12).
        world_x = np.array([8, 8], dtype=np.int32)
        world_y = np.array([18, 12], dtype=np.int32)
        tile_ids = np.full(2, TileTypeID.FLOOR, dtype=np.uint8)
        chars = np.full(2, ord("."), dtype=np.int32)
        fg_rgb = np.full((2, 3), 100, dtype=np.uint8)
        bg_rgb = np.full((2, 3), 100, dtype=np.uint8)

        result = view._apply_roof_substitution(
            chars,
            fg_rgb,
            bg_rgb,
            tile_ids,
            world_x,
            world_y,
            is_light=False,
            decoration_seed=0,
        )

        # Split data should be populated for the boundary tile.
        assert result.split_y is not None
        assert result.split_bg is not None

        # South split row (index 0) should have split_y = 0.5.
        assert abs(float(result.split_y[0]) - 0.5) < 0.01
        # Full roof row (index 1) has no split.
        assert float(result.split_y[1]) == 0.0

        # Split bg should have non-zero alpha (eave shadow color).
        assert int(result.split_bg[0][3]) == 255


class TestAtmosphericViewportSizing:
    """Atmospheric uniforms should use visible world-tile viewport size under zoom."""

    def test_present_passes_zoomed_world_viewport_size_to_atmospheric_layer(
        self,
    ) -> None:
        view = object.__new__(WorldView)
        view.visible = True
        view.x = 5
        view.y = 4
        view.width = 20
        view.height = 12
        view._SCROLL_PADDING = 1
        view._viewport_zoom = 1.0
        view._shake_offset = (0.0, 0.0)
        view._active_background_texture = None
        view._light_overlay_texture = None
        view._gpu_actor_lightmap_texture = None
        view._gpu_actor_lightmap_viewport_origin = None
        view._game_time = 0.0
        view.camera_frac_offset = (0.0, 0.0)
        view.rain_config = SimpleNamespace(enabled=False)
        view.rain_animation = SimpleNamespace(time=0.0, render_angle=0.0)

        active_layer = SimpleNamespace(
            blend_mode="darken",
            strength=0.5,
            disable_when_overcast=False,
            sky_exposure_threshold=0.3,
            noise_scale=1.0,
            noise_threshold_low=0.2,
            noise_threshold_high=0.8,
            tint_color=(0, 0, 0),
            turbulence_strength=0.0,
            turbulence_scale=1.0,
        )
        layer_state = SimpleNamespace(
            drift_offset_x=0.0,
            drift_offset_y=0.0,
            turbulence_offset=0.0,
        )
        view.atmospheric_system = Mock()
        view.atmospheric_system.get_active_layers.return_value = [
            (active_layer, layer_state)
        ]
        view.atmospheric_system.config = SimpleNamespace(cloud_coverage=0.0)

        view._get_directional_light = Mock(return_value=None)
        view._apply_sun_direction_to_graphics = Mock()
        view._render_light_overlay_gpu_compose = Mock(return_value=None)
        view._update_tile_animations = Mock()
        view._update_actor_particles = Mock()
        view._render_map_unlit = Mock()
        view._get_background_cache_key = Mock(return_value=("k",))
        view._texture_cache = Mock()
        view._texture_cache.get.return_value = True
        view.controller = SimpleNamespace(
            clock=SimpleNamespace(last_delta_time=0.016, last_time=1.0),
            gw=SimpleNamespace(
                player=SimpleNamespace(x=0, y=0),
                game_map=SimpleNamespace(
                    width=200,
                    height=200,
                    structural_revision=1,
                    exploration_revision=1,
                    decoration_seed=1,
                ),
                lights=[],
            ),
            mode_stack=[],
        )
        view.lighting_system = None
        view.screen_shake = Mock()
        view.screen_shake.update.return_value = (0.0, 0.0)
        view.shadow_renderer = Mock()
        view.actor_renderer = Mock()
        view.actor_renderer.get_sorted_visible_actors.return_value = []
        view.particle_system = Mock()
        view.environmental_system = Mock()
        view.environmental_system.render_effects = Mock()
        view.decal_system = Mock()
        view.floating_text_manager = Mock()
        view.map_glyph_buffer = Mock()
        view.light_source_glyph_buffer = Mock()

        viewport = SimpleNamespace(
            width_tiles=10,
            height_tiles=6,
            offset_x=0,
            offset_y=0,
            resize=Mock(),
        )
        camera = SimpleNamespace(world_x=0.0, world_y=0.0, set_position=Mock())
        vs = Mock()
        vs.viewport = viewport
        vs.camera = camera
        vs.update_camera = Mock()
        vs.get_visible_bounds.return_value = SimpleNamespace(x1=30, y1=40, x2=39, y2=45)
        vs.get_display_scale_factors.return_value = (2.0, 2.0)
        vs.get_display_camera_fractional_offset.return_value = (0.0, 0.0)
        view.viewport_system = vs

        graphics = Mock()
        graphics.console_to_screen_coords.side_effect = lambda x, y: (
            x * 10.0,
            y * 10.0,
        )
        graphics.set_noise_seed = Mock()
        graphics.set_noise_tile_offset = Mock()
        graphics.render_glyph_buffer_to_texture.return_value = object()
        graphics.draw_background = Mock()
        graphics.render_decals = Mock()
        graphics.render_particles = Mock()
        graphics.shadow_pass.return_value = nullcontext()
        graphics.set_atmospheric_layer = Mock()
        graphics.set_atmospheric_layer.return_value = None
        graphics.set_rain_effect = Mock()
        graphics.set_actor_lighting_gpu_context = Mock()

        # Call present() directly; draw() is not needed for the atmospheric sizing regression.
        view.present(graphics, alpha=InterpolationAlpha(1.0))

        assert graphics.set_atmospheric_layer.called
        atmospheric_call = graphics.set_atmospheric_layer.call_args
        assert atmospheric_call is not None
        # Arg 1 is viewport_size (after viewport_offset).
        assert atmospheric_call.args[1] == (10, 6)

    def test_present_queues_zoom_compensated_rain_effect_when_enabled(self) -> None:
        view = object.__new__(WorldView)
        view.visible = True
        view.x = 5
        view.y = 4
        view.width = 20
        view.height = 12
        view._SCROLL_PADDING = 1
        view._viewport_zoom = 0.5
        view._shake_offset = (0.0, 0.0)
        view._active_background_texture = None
        view._light_overlay_texture = None
        view._gpu_actor_lightmap_texture = None
        view._gpu_actor_lightmap_viewport_origin = None
        view._game_time = 0.0
        view.camera_frac_offset = (0.0, 0.0)
        view.rain_config = SimpleNamespace(
            enabled=True,
            intensity=0.6,
            angle=0.2,
            drop_length=0.9,
            drop_speed=7.0,
            drop_spacing=1.4,
            stream_spacing=0.25,
            color=(10, 20, 30),
        )
        view.rain_animation = SimpleNamespace(time=12.5, render_angle=0.2)
        rain_exclusion_mask = np.zeros((10, 6), dtype=np.bool_)
        view._build_rain_exclusion_mask = Mock(return_value=rain_exclusion_mask)

        view.atmospheric_system = Mock()
        view.atmospheric_system.get_active_layers.return_value = []
        view.atmospheric_system.config = SimpleNamespace(cloud_coverage=0.0)

        view._get_directional_light = Mock(return_value=None)
        view._apply_sun_direction_to_graphics = Mock()
        view._render_light_overlay_gpu_compose = Mock(return_value=None)
        view._update_tile_animations = Mock()
        view._update_actor_particles = Mock()
        view._render_map_unlit = Mock()
        view._get_background_cache_key = Mock(return_value=("k",))
        view._texture_cache = Mock()
        view._texture_cache.get.return_value = True
        view.controller = SimpleNamespace(
            clock=SimpleNamespace(last_delta_time=0.016, last_time=1.0),
            gw=SimpleNamespace(
                player=SimpleNamespace(x=0, y=0),
                game_map=SimpleNamespace(
                    width=200,
                    height=200,
                    structural_revision=1,
                    exploration_revision=1,
                    decoration_seed=1,
                ),
                lights=[],
            ),
            mode_stack=[],
        )
        view.lighting_system = None
        view.screen_shake = Mock()
        view.screen_shake.update.return_value = (0.0, 0.0)
        view.shadow_renderer = Mock()
        view.actor_renderer = Mock()
        view.actor_renderer.get_sorted_visible_actors.return_value = []
        view.particle_system = Mock()
        view.environmental_system = Mock()
        view.environmental_system.render_effects = Mock()
        view.decal_system = Mock()
        view.floating_text_manager = Mock()
        view.map_glyph_buffer = Mock()
        view.light_source_glyph_buffer = Mock()

        viewport = SimpleNamespace(
            width_tiles=10,
            height_tiles=6,
            offset_x=0,
            offset_y=0,
            resize=Mock(),
        )
        camera = SimpleNamespace(world_x=0.0, world_y=0.0, set_position=Mock())
        vs = Mock()
        vs.viewport = viewport
        vs.camera = camera
        vs.update_camera = Mock()
        vs.get_visible_bounds.return_value = SimpleNamespace(x1=30, y1=40, x2=39, y2=45)
        vs.get_display_scale_factors.return_value = (2.0, 2.0)
        vs.get_display_camera_fractional_offset.return_value = (0.0, 0.0)
        view.viewport_system = vs

        graphics = Mock()
        graphics.console_to_screen_coords.side_effect = lambda x, y: (
            x * 10.0,
            y * 10.0,
        )
        graphics.tile_dimensions = (16, 16)
        graphics.set_noise_seed = Mock()
        graphics.set_noise_tile_offset = Mock()
        graphics.render_glyph_buffer_to_texture.return_value = object()
        graphics.draw_background = Mock()
        graphics.render_decals = Mock()
        graphics.render_particles = Mock()
        graphics.shadow_pass.return_value = nullcontext()
        graphics.set_atmospheric_layer = Mock()
        graphics.set_rain_effect = Mock()
        graphics.set_actor_lighting_gpu_context = Mock()

        view.present(graphics, alpha=InterpolationAlpha(1.0))

        rain_call = graphics.set_rain_effect.call_args
        assert rain_call is not None
        assert rain_call.kwargs["viewport_size"] == (10, 6)
        assert rain_call.kwargs["time"] == 12.5
        assert rain_call.kwargs["drop_length"] == pytest.approx(1.8)
        assert rain_call.kwargs["drop_speed"] == pytest.approx(14.0)
        assert rain_call.kwargs["drop_spacing"] == pytest.approx(2.8)
        assert rain_call.kwargs["stream_spacing"] == pytest.approx(0.5)
        assert rain_call.kwargs["rain_exclusion_mask_buffer"] is rain_exclusion_mask

    def test_build_rain_exclusion_mask_marks_player_building_footprint(self) -> None:
        view = object.__new__(WorldView)
        building = SimpleNamespace(
            id=7,
            footprint=SimpleNamespace(x1=12, y1=22, x2=16, y2=25),
        )
        view._compute_roof_state = Mock(return_value=(7, [building]))

        mask = view._build_rain_exclusion_mask((10, 20), (8, 6))

        assert mask is not None
        expected = np.zeros((8, 6), dtype=np.bool_)
        expected[2:6, 2:5] = True
        assert np.array_equal(mask, expected)

    def test_build_rain_exclusion_mask_returns_none_when_player_not_in_building(
        self,
    ) -> None:
        """When the player is outdoors, no exclusion mask is needed."""
        view = object.__new__(WorldView)
        building = SimpleNamespace(
            id=7,
            footprint=SimpleNamespace(x1=12, y1=22, x2=16, y2=25),
        )
        view._compute_roof_state = Mock(return_value=(None, [building]))

        mask = view._build_rain_exclusion_mask((10, 20), (8, 6))

        assert mask is None

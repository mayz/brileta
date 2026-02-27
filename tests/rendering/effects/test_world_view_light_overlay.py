"""Tests for WorldView GPU light overlay composition logic."""

from unittest.mock import Mock

import numpy as np
import pytest

from brileta.environment import tile_types
from brileta.environment.generators.buildings.building import Building
from brileta.environment.map import TileAnimationState
from brileta.util.coordinates import Rect
from brileta.view.views.world_view import (
    _LIGHT_OVERLAY_MASK_ROOF_SUNLIT,
    WorldView,
    _RoofSubstitutionResult,
)


class TestWorldViewLightOverlay:
    """Test GPU light overlay composition behavior."""

    def test_gpu_compose_raises_when_no_lighting_system(self) -> None:
        """Composition should fail loudly when lighting is enabled but unavailable."""
        world_view = WorldView(Mock(), Mock(), lighting_system=None)
        with pytest.raises(RuntimeError, match="no lighting system"):
            world_view._render_light_overlay_gpu_compose(
                Mock(),
                dark_texture=object(),
            )

    def test_gpu_compose_uses_explored_tiles_and_visible_mask(self) -> None:
        """Explored tiles populate source data while visibility drives mask values."""
        mock_controller = Mock()
        mock_lighting_system = Mock()
        world_view = WorldView(mock_controller, Mock(), mock_lighting_system)
        world_view.set_bounds(0, 0, 10, 10)

        mock_gw = Mock()
        mock_controller.gw = mock_gw
        mock_gw.game_map.width = 3
        mock_gw.game_map.height = 3
        mock_gw.game_map.exploration_revision = 0

        explored = np.ones((3, 3), dtype=bool)
        visible = np.zeros((3, 3), dtype=bool)
        visible[1, 1] = True
        mock_gw.game_map.explored = explored
        mock_gw.game_map.visible = visible

        mock_gw.game_map.tiles = np.zeros((3, 3), dtype=np.uint8)
        mock_gw.game_map.decoration_seed = 0

        light_app = np.zeros((3, 3), dtype=tile_types.TileTypeAppearance)
        light_app[:, :] = (ord("#"), (200, 200, 200), (50, 50, 50))
        mock_gw.game_map.light_appearance_map = light_app
        mock_gw.game_map.animation_params = np.zeros(
            (3, 3),
            dtype=tile_types.TileAnimationParams,
        )
        mock_gw.game_map.animation_state = np.zeros((3, 3), dtype=TileAnimationState)

        mock_viewport_system = Mock()
        mock_viewport_system.get_visible_bounds.return_value = Rect(0, 0, 3, 3)
        mock_viewport_system.offset_x = 0
        mock_viewport_system.offset_y = 0
        world_view.viewport_system = mock_viewport_system

        lightmap_texture = object()
        composed_texture = object()
        mock_lighting_system.compute_lightmap_texture.return_value = lightmap_texture
        mock_lighting_system.compute_lightmap = Mock()

        mock_graphics = Mock()
        mock_graphics.render_glyph_buffer_to_texture.return_value = object()
        mock_graphics.compose_light_overlay_gpu.return_value = composed_texture

        result = world_view._render_light_overlay_gpu_compose(
            mock_graphics,
            dark_texture=object(),
        )

        assert result is composed_texture
        mock_lighting_system.compute_lightmap_texture.assert_called_once()
        mock_lighting_system.compute_lightmap.assert_not_called()

        compose_kwargs = mock_graphics.compose_light_overlay_gpu.call_args.kwargs
        visible_mask_buffer = compose_kwargs["visible_mask_buffer"]
        pad = world_view._SCROLL_PADDING

        # Corner tile is explored-but-not-visible. Terrain decoration replaces
        # the light appearance glyph with the tile's decoration glyph (space for WALL).
        corner_ch = world_view.light_source_glyph_buffer.data["ch"][pad, pad]
        assert corner_ch == ord(" ")
        assert bool(visible_mask_buffer[pad, pad]) is False
        # Center tile is visible.
        assert bool(visible_mask_buffer[pad + 1, pad + 1]) is True

        assert world_view._gpu_actor_lightmap_texture is lightmap_texture
        assert world_view._gpu_actor_lightmap_viewport_origin == (0, 0)

    def test_gpu_compose_handles_no_explored_tiles(self) -> None:
        """Composition should succeed with empty explored/visible masks."""
        mock_controller = Mock()
        mock_lighting_system = Mock()
        world_view = WorldView(mock_controller, Mock(), mock_lighting_system)
        world_view.set_bounds(0, 0, 10, 10)

        mock_gw = Mock()
        mock_controller.gw = mock_gw
        mock_gw.game_map.width = 3
        mock_gw.game_map.height = 3
        mock_gw.game_map.exploration_revision = 0
        mock_gw.game_map.explored = np.zeros((3, 3), dtype=bool)
        mock_gw.game_map.visible = np.zeros((3, 3), dtype=bool)
        mock_gw.game_map.light_appearance_map = np.zeros(
            (3, 3),
            dtype=tile_types.TileTypeAppearance,
        )
        mock_gw.game_map.animation_params = np.zeros(
            (3, 3),
            dtype=tile_types.TileAnimationParams,
        )
        mock_gw.game_map.animation_state = np.zeros((3, 3), dtype=TileAnimationState)

        mock_viewport_system = Mock()
        mock_viewport_system.get_visible_bounds.return_value = Rect(0, 0, 3, 3)
        mock_viewport_system.offset_x = 0
        mock_viewport_system.offset_y = 0
        world_view.viewport_system = mock_viewport_system

        mock_lighting_system.compute_lightmap_texture.return_value = object()

        mock_graphics = Mock()
        mock_graphics.render_glyph_buffer_to_texture.return_value = object()
        mock_graphics.compose_light_overlay_gpu.return_value = object()

        result = world_view._render_light_overlay_gpu_compose(
            mock_graphics,
            dark_texture=object(),
        )

        assert result is not None
        compose_kwargs = mock_graphics.compose_light_overlay_gpu.call_args.kwargs
        visible_mask_buffer = compose_kwargs["visible_mask_buffer"]
        assert not np.any(visible_mask_buffer)

    def test_gpu_compose_hides_visible_mask_for_roof_covered_tiles(self) -> None:
        """Roof-covered cells should not receive interior light/FOV compose."""
        mock_controller = Mock()
        mock_lighting_system = Mock()
        world_view = WorldView(mock_controller, Mock(), mock_lighting_system)
        world_view.set_bounds(0, 0, 10, 10)

        mock_gw = Mock()
        mock_controller.gw = mock_gw
        mock_gw.game_map.width = 1
        mock_gw.game_map.height = 1
        mock_gw.game_map.exploration_revision = 0
        mock_gw.game_map.structural_revision = 0
        mock_gw.game_map.tiles = np.array(
            [[tile_types.TileTypeID.FLOOR]], dtype=np.uint8
        )
        mock_gw.game_map.decoration_seed = 0
        mock_gw.game_map.explored = np.ones((1, 1), dtype=bool)
        mock_gw.game_map.visible = np.ones((1, 1), dtype=bool)
        mock_gw.game_map.light_appearance_map = np.zeros(
            (1, 1),
            dtype=tile_types.TileTypeAppearance,
        )
        mock_gw.game_map.animation_params = np.zeros(
            (1, 1),
            dtype=tile_types.TileAnimationParams,
        )
        mock_gw.game_map.animation_state = np.zeros((1, 1), dtype=TileAnimationState)
        mock_gw.player.x = 0
        mock_gw.player.y = 0
        mock_gw.buildings = []
        mock_gw.game_map.get_region_at.return_value = None

        mock_viewport_system = Mock()
        mock_viewport_system.get_visible_bounds.return_value = Rect(0, 0, 1, 1)
        mock_viewport_system.offset_x = 0
        mock_viewport_system.offset_y = 0
        world_view.viewport_system = mock_viewport_system

        mock_lighting_system.compute_lightmap_texture.return_value = object()

        mock_graphics = Mock()
        mock_graphics.render_glyph_buffer_to_texture.return_value = object()
        mock_graphics.compose_light_overlay_gpu.return_value = object()

        world_view._apply_roof_substitution = Mock(
            return_value=_RoofSubstitutionResult(
                np.array([tile_types.TileTypeID.ROOF_THATCH], dtype=np.uint8)
            )
        )

        world_view._render_light_overlay_gpu_compose(
            mock_graphics,
            dark_texture=object(),
        )

        compose_kwargs = mock_graphics.compose_light_overlay_gpu.call_args.kwargs
        visible_mask_buffer = compose_kwargs["visible_mask_buffer"]
        pad = world_view._SCROLL_PADDING
        assert int(visible_mask_buffer[pad, pad]) == int(
            _LIGHT_OVERLAY_MASK_ROOF_SUNLIT
        )

    def test_gpu_compose_raises_when_backend_returns_none(self) -> None:
        """Composition should fail loudly if backend compose unexpectedly returns None."""
        mock_controller = Mock()
        mock_lighting_system = Mock()
        world_view = WorldView(mock_controller, Mock(), mock_lighting_system)
        world_view.set_bounds(0, 0, 10, 10)

        mock_gw = Mock()
        mock_controller.gw = mock_gw
        mock_gw.game_map.width = 1
        mock_gw.game_map.height = 1
        mock_gw.game_map.exploration_revision = 0
        mock_gw.game_map.tiles = np.zeros((1, 1), dtype=np.uint8)
        mock_gw.game_map.decoration_seed = 0
        mock_gw.game_map.explored = np.ones((1, 1), dtype=bool)
        mock_gw.game_map.visible = np.ones((1, 1), dtype=bool)
        mock_gw.game_map.light_appearance_map = np.zeros(
            (1, 1),
            dtype=tile_types.TileTypeAppearance,
        )
        mock_gw.game_map.animation_params = np.zeros(
            (1, 1),
            dtype=tile_types.TileAnimationParams,
        )
        mock_gw.game_map.animation_state = np.zeros((1, 1), dtype=TileAnimationState)

        mock_viewport_system = Mock()
        mock_viewport_system.get_visible_bounds.return_value = Rect(0, 0, 1, 1)
        mock_viewport_system.offset_x = 0
        mock_viewport_system.offset_y = 0
        world_view.viewport_system = mock_viewport_system

        mock_lighting_system.compute_lightmap_texture.return_value = object()

        mock_graphics = Mock()
        mock_graphics.render_glyph_buffer_to_texture.return_value = object()
        mock_graphics.compose_light_overlay_gpu.return_value = None

        with pytest.raises(RuntimeError, match="produced no texture"):
            world_view._render_light_overlay_gpu_compose(
                mock_graphics,
                dark_texture=object(),
            )

    def test_gpu_compose_raises_when_lightmap_texture_missing(self) -> None:
        """Composition should fail loudly if lightmap texture generation fails."""
        mock_controller = Mock()
        mock_lighting_system = Mock()
        world_view = WorldView(mock_controller, Mock(), mock_lighting_system)
        world_view.set_bounds(0, 0, 10, 10)

        mock_gw = Mock()
        mock_controller.gw = mock_gw
        mock_gw.game_map.width = 1
        mock_gw.game_map.height = 1
        mock_gw.game_map.exploration_revision = 0
        mock_gw.game_map.tiles = np.zeros((1, 1), dtype=np.uint8)
        mock_gw.game_map.decoration_seed = 0
        mock_gw.game_map.explored = np.ones((1, 1), dtype=bool)
        mock_gw.game_map.visible = np.ones((1, 1), dtype=bool)
        mock_gw.game_map.light_appearance_map = np.zeros(
            (1, 1),
            dtype=tile_types.TileTypeAppearance,
        )
        mock_gw.game_map.animation_params = np.zeros(
            (1, 1),
            dtype=tile_types.TileAnimationParams,
        )
        mock_gw.game_map.animation_state = np.zeros((1, 1), dtype=TileAnimationState)

        mock_viewport_system = Mock()
        mock_viewport_system.get_visible_bounds.return_value = Rect(0, 0, 1, 1)
        mock_viewport_system.offset_x = 0
        mock_viewport_system.offset_y = 0
        world_view.viewport_system = mock_viewport_system

        mock_lighting_system.compute_lightmap_texture.return_value = None
        mock_graphics = Mock()
        mock_graphics.compose_light_overlay_gpu.return_value = object()

        with pytest.raises(RuntimeError, match="returned no lightmap texture"):
            world_view._render_light_overlay_gpu_compose(
                mock_graphics,
                dark_texture=object(),
            )

    def test_apply_tile_light_animations_modulates_only_animated_tiles(self) -> None:
        """Tile color modulation should affect only entries marked as animated."""
        world_view = object.__new__(WorldView)

        light_fg_rgb = np.array(
            [
                [100, 150, 200],
                [240, 240, 240],
            ],
            dtype=np.uint8,
        )
        light_bg_rgb = np.array(
            [
                [50, 60, 70],
                [10, 20, 30],
            ],
            dtype=np.uint8,
        )

        animation_params = np.zeros((2, 2), dtype=tile_types.TileAnimationParams)
        animation_state = np.zeros((2, 2), dtype=TileAnimationState)
        animation_params["animates"][1, 1] = True
        animation_params["fg_variation"][1, 1] = (100, 100, 100)
        animation_params["bg_variation"][1, 1] = (100, 100, 100)
        animation_state["fg_values"][1, 1] = (1000, 0, 500)
        animation_state["bg_values"][1, 1] = (1000, 0, 500)

        valid_exp_x = np.array([0, 1], dtype=np.int32)
        valid_exp_y = np.array([0, 1], dtype=np.int32)

        world_view._apply_tile_light_animations(
            light_fg_rgb,
            light_bg_rgb,
            animation_params,
            animation_state,
            valid_exp_x,
            valid_exp_y,
        )

        # Non-animated entry remains untouched.
        assert tuple(int(v) for v in light_fg_rgb[0]) == (100, 150, 200)
        assert tuple(int(v) for v in light_bg_rgb[0]) == (50, 60, 70)
        # Animated entry receives modulation and clamps per channel.
        assert tuple(int(v) for v in light_fg_rgb[1]) == (255, 190, 240)
        assert tuple(int(v) for v in light_bg_rgb[1]) == (60, 0, 30)

    def test_apply_tile_light_animations_skips_excluded_roof_covered_tiles(
        self,
    ) -> None:
        """Excluded cells should not animate after visual roof substitution."""
        world_view = object.__new__(WorldView)

        light_fg_rgb = np.array([[120, 130, 140]], dtype=np.uint8)
        light_bg_rgb = np.array([[40, 50, 60]], dtype=np.uint8)

        animation_params = np.zeros((1, 1), dtype=tile_types.TileAnimationParams)
        animation_state = np.zeros((1, 1), dtype=TileAnimationState)
        animation_params["animates"][0, 0] = True
        animation_params["fg_variation"][0, 0] = (200, 200, 200)
        animation_params["bg_variation"][0, 0] = (200, 200, 200)
        animation_state["fg_values"][0, 0] = (1000, 1000, 1000)
        animation_state["bg_values"][0, 0] = (0, 0, 0)

        valid_exp_x = np.array([0], dtype=np.int32)
        valid_exp_y = np.array([0], dtype=np.int32)
        roof_covered_mask = np.array([True], dtype=bool)

        world_view._apply_tile_light_animations(
            light_fg_rgb,
            light_bg_rgb,
            animation_params,
            animation_state,
            valid_exp_x,
            valid_exp_y,
            exclude_mask=roof_covered_mask,
        )

        # The underlying tile is animated, but roof substitution should keep the
        # visible roof colors unchanged.
        assert tuple(int(v) for v in light_fg_rgb[0]) == (120, 130, 140)
        assert tuple(int(v) for v in light_bg_rgb[0]) == (40, 50, 60)

    def test_build_atmospheric_roof_surface_mask_matches_roof_cutaways(self) -> None:
        """Atmospheric roof mask should mirror roof substitution visibility rules."""
        world_view = object.__new__(WorldView)
        building = Building(id=7, building_type="house", footprint=Rect(10, 20, 3, 3))

        world_view._compute_roof_state = Mock(return_value=(None, [building]))
        world_view._get_roof_entrance_clear_positions = Mock(
            return_value={(10, 20)}  # clear one footprint corner (e.g. door path)
        )

        mask = world_view._build_atmospheric_roof_surface_mask((10, 20), (4, 4))

        assert mask is not None
        assert mask.shape == (4, 4)
        # With perspective offset (F=1.5, N=2, floor_F=1), the visual roof
        # for this 3x3 building covers y=19..21 (shifted north from footprint
        # y=20..22). The viewport starts at y=20, so y=20 and y=21 are roofed.
        assert bool(mask[0, 0]) is False  # entrance clear at (10, 20)
        assert bool(mask[1, 1]) is True  # interior tile (11, 21) - roofed
        assert bool(mask[2, 0]) is True  # (12, 20) - still in visual roof
        # South footprint row (y=22) is now wall face, not roof.
        assert bool(mask[2, 2]) is False
        # Outside the footprint remains clear.
        assert bool(mask[3, 3]) is False

    def test_gpu_compose_reuses_cached_light_source_buffer_between_frames(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Second frame with stable viewport/exploration should skip full rebuild."""
        mock_controller = Mock()
        mock_lighting_system = Mock()
        world_view = WorldView(mock_controller, Mock(), mock_lighting_system)
        world_view.set_bounds(0, 0, 4, 4)

        mock_gw = Mock()
        mock_controller.gw = mock_gw
        mock_gw.game_map.width = 1
        mock_gw.game_map.height = 1
        mock_gw.game_map.exploration_revision = 0
        mock_gw.game_map.explored = np.ones((1, 1), dtype=bool)
        mock_gw.game_map.visible = np.ones((1, 1), dtype=bool)
        mock_gw.game_map.tiles = np.zeros((1, 1), dtype=np.uint8)
        mock_gw.game_map.decoration_seed = 0

        light_app = np.zeros((1, 1), dtype=tile_types.TileTypeAppearance)
        light_app[0, 0] = (ord("~"), (100, 100, 100), (20, 20, 20))
        mock_gw.game_map.light_appearance_map = light_app

        animation_params = np.zeros((1, 1), dtype=tile_types.TileAnimationParams)
        animation_params["animates"][0, 0] = True
        animation_params["fg_variation"][0, 0] = (100, 100, 100)
        animation_params["bg_variation"][0, 0] = (100, 100, 100)
        mock_gw.game_map.animation_params = animation_params

        animation_state = np.zeros((1, 1), dtype=TileAnimationState)
        animation_state["fg_values"][0, 0] = (0, 0, 0)
        animation_state["bg_values"][0, 0] = (0, 0, 0)
        mock_gw.game_map.animation_state = animation_state

        mock_viewport_system = Mock()
        mock_viewport_system.get_visible_bounds.return_value = Rect(0, 0, 1, 1)
        mock_viewport_system.offset_x = 0
        mock_viewport_system.offset_y = 0
        world_view.viewport_system = mock_viewport_system

        mock_lighting_system.compute_lightmap_texture.return_value = object()

        mock_graphics = Mock()
        mock_graphics.render_glyph_buffer_to_texture.return_value = object()
        mock_graphics.compose_light_overlay_gpu.return_value = object()

        # Keep base colors deterministic so only animation state drives the change.
        terrain_decoration = Mock()
        monkeypatch.setattr(tile_types, "apply_terrain_decoration", terrain_decoration)
        edge_transition = Mock()
        world_view._apply_tile_edge_transition_data = edge_transition
        clear_calls = Mock(wraps=world_view.light_source_glyph_buffer.clear)
        monkeypatch.setattr(world_view.light_source_glyph_buffer, "clear", clear_calls)

        world_view._render_light_overlay_gpu_compose(
            mock_graphics, dark_texture=object()
        )

        pad = world_view._SCROLL_PADDING
        first_compose_kwargs = mock_graphics.compose_light_overlay_gpu.call_args_list[
            0
        ].kwargs
        first_visible_mask = first_compose_kwargs["visible_mask_buffer"]
        first_fg_rgb = tuple(
            int(v)
            for v in world_view.light_source_glyph_buffer.data["fg"][pad, pad][:3]
        )
        assert bool(first_visible_mask[pad, pad]) is True

        animation_state["fg_values"][0, 0] = (1000, 1000, 1000)
        animation_state["bg_values"][0, 0] = (1000, 1000, 1000)
        world_view._render_light_overlay_gpu_compose(
            mock_graphics, dark_texture=object()
        )

        second_compose_kwargs = mock_graphics.compose_light_overlay_gpu.call_args_list[
            1
        ].kwargs
        second_visible_mask = second_compose_kwargs["visible_mask_buffer"]
        second_fg_rgb = tuple(
            int(v)
            for v in world_view.light_source_glyph_buffer.data["fg"][pad, pad][:3]
        )

        assert second_visible_mask is first_visible_mask
        assert bool(second_visible_mask[pad, pad]) is True
        assert clear_calls.call_count == 1
        assert terrain_decoration.call_count == 1
        assert edge_transition.call_count == 1
        assert first_fg_rgb != second_fg_rgb

    def test_gpu_compose_updates_visibility_mask_without_rebuilding_on_player_move(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Player/FOV changes should not force a full light-source buffer rebuild."""
        mock_controller = Mock()
        mock_lighting_system = Mock()
        world_view = WorldView(mock_controller, Mock(), mock_lighting_system)
        world_view.set_bounds(0, 0, 4, 4)

        mock_gw = Mock()
        mock_controller.gw = mock_gw
        mock_gw.game_map.width = 1
        mock_gw.game_map.height = 1
        mock_gw.game_map.exploration_revision = 0
        mock_gw.game_map.explored = np.ones((1, 1), dtype=bool)
        visible = np.ones((1, 1), dtype=bool)
        mock_gw.game_map.visible = visible
        mock_gw.game_map.tiles = np.zeros((1, 1), dtype=np.uint8)
        mock_gw.game_map.decoration_seed = 0
        mock_gw.game_map.get_region_at = None
        mock_gw.player = Mock()
        mock_gw.player.x = 0
        mock_gw.player.y = 0

        light_app = np.zeros((1, 1), dtype=tile_types.TileTypeAppearance)
        light_app[0, 0] = (ord("."), (100, 100, 100), (20, 20, 20))
        mock_gw.game_map.light_appearance_map = light_app
        mock_gw.game_map.animation_params = np.zeros(
            (1, 1),
            dtype=tile_types.TileAnimationParams,
        )
        mock_gw.game_map.animation_state = np.zeros((1, 1), dtype=TileAnimationState)

        mock_viewport_system = Mock()
        mock_viewport_system.get_visible_bounds.return_value = Rect(0, 0, 1, 1)
        mock_viewport_system.offset_x = 0
        mock_viewport_system.offset_y = 0
        world_view.viewport_system = mock_viewport_system

        mock_lighting_system.compute_lightmap_texture.return_value = object()
        mock_graphics = Mock()
        mock_graphics.render_glyph_buffer_to_texture.return_value = object()
        mock_graphics.compose_light_overlay_gpu.return_value = object()

        terrain_decoration = Mock()
        monkeypatch.setattr(tile_types, "apply_terrain_decoration", terrain_decoration)
        edge_transition = Mock()
        world_view._apply_tile_edge_transition_data = edge_transition
        clear_calls = Mock(wraps=world_view.light_source_glyph_buffer.clear)
        monkeypatch.setattr(world_view.light_source_glyph_buffer, "clear", clear_calls)

        world_view._render_light_overlay_gpu_compose(
            mock_graphics, dark_texture=object()
        )
        pad = world_view._SCROLL_PADDING
        first_visible_mask = mock_graphics.compose_light_overlay_gpu.call_args_list[
            0
        ].kwargs["visible_mask_buffer"]
        assert bool(first_visible_mask[pad, pad]) is True

        # Simulate player movement changing FOV visibility without any new
        # exploration in the viewport.
        mock_gw.player.x = 99
        mock_gw.player.y = 42
        visible[0, 0] = False

        world_view._render_light_overlay_gpu_compose(
            mock_graphics, dark_texture=object()
        )
        second_visible_mask = mock_graphics.compose_light_overlay_gpu.call_args_list[
            1
        ].kwargs["visible_mask_buffer"]

        assert second_visible_mask is first_visible_mask
        assert bool(second_visible_mask[pad, pad]) is False
        assert clear_calls.call_count == 1
        assert terrain_decoration.call_count == 1
        assert edge_transition.call_count == 1

    def test_gpu_compose_keeps_cache_when_exploration_changes_offscreen(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Global exploration revision should not invalidate cache for offscreen changes."""
        mock_controller = Mock()
        mock_lighting_system = Mock()
        world_view = WorldView(mock_controller, Mock(), mock_lighting_system)
        world_view.set_bounds(0, 0, 4, 4)

        mock_gw = Mock()
        mock_controller.gw = mock_gw
        mock_gw.game_map.width = 3
        mock_gw.game_map.height = 3
        mock_gw.game_map.exploration_revision = 0
        explored = np.zeros((3, 3), dtype=bool)
        explored[0, 0] = True
        mock_gw.game_map.explored = explored
        visible = np.zeros((3, 3), dtype=bool)
        visible[0, 0] = True
        mock_gw.game_map.visible = visible
        mock_gw.game_map.tiles = np.zeros((3, 3), dtype=np.uint8)
        mock_gw.game_map.decoration_seed = 0

        light_app = np.zeros((3, 3), dtype=tile_types.TileTypeAppearance)
        light_app[:, :] = (ord("."), (100, 100, 100), (20, 20, 20))
        mock_gw.game_map.light_appearance_map = light_app
        mock_gw.game_map.animation_params = np.zeros(
            (3, 3),
            dtype=tile_types.TileAnimationParams,
        )
        mock_gw.game_map.animation_state = np.zeros((3, 3), dtype=TileAnimationState)

        mock_viewport_system = Mock()
        mock_viewport_system.get_visible_bounds.return_value = Rect(0, 0, 1, 1)
        mock_viewport_system.offset_x = 0
        mock_viewport_system.offset_y = 0
        world_view.viewport_system = mock_viewport_system

        mock_lighting_system.compute_lightmap_texture.return_value = object()
        mock_graphics = Mock()
        mock_graphics.render_glyph_buffer_to_texture.return_value = object()
        mock_graphics.compose_light_overlay_gpu.return_value = object()

        terrain_decoration = Mock()
        monkeypatch.setattr(tile_types, "apply_terrain_decoration", terrain_decoration)
        edge_transition = Mock()
        world_view._apply_tile_edge_transition_data = edge_transition
        clear_calls = Mock(wraps=world_view.light_source_glyph_buffer.clear)
        monkeypatch.setattr(world_view.light_source_glyph_buffer, "clear", clear_calls)

        world_view._render_light_overlay_gpu_compose(
            mock_graphics, dark_texture=object()
        )

        # Explore a tile outside the current viewport and bump the global revision.
        explored[2, 2] = True
        mock_gw.game_map.exploration_revision = 1

        world_view._render_light_overlay_gpu_compose(
            mock_graphics, dark_texture=object()
        )

        assert clear_calls.call_count == 1
        assert terrain_decoration.call_count == 1
        assert edge_transition.call_count == 1

    def test_gpu_compose_rebuilds_when_exploration_changes_in_viewport(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Viewport-local exploration changes must still invalidate cached tile data."""
        mock_controller = Mock()
        mock_lighting_system = Mock()
        world_view = WorldView(mock_controller, Mock(), mock_lighting_system)
        world_view.set_bounds(0, 0, 4, 4)

        mock_gw = Mock()
        mock_controller.gw = mock_gw
        mock_gw.game_map.width = 2
        mock_gw.game_map.height = 1
        mock_gw.game_map.exploration_revision = 0
        explored = np.array([[True], [False]], dtype=bool)
        visible = np.array([[True], [False]], dtype=bool)
        mock_gw.game_map.explored = explored
        mock_gw.game_map.visible = visible
        mock_gw.game_map.tiles = np.zeros((2, 1), dtype=np.uint8)
        mock_gw.game_map.decoration_seed = 0

        light_app = np.zeros((2, 1), dtype=tile_types.TileTypeAppearance)
        light_app[:, :] = (ord("."), (100, 100, 100), (20, 20, 20))
        mock_gw.game_map.light_appearance_map = light_app
        mock_gw.game_map.animation_params = np.zeros(
            (2, 1),
            dtype=tile_types.TileAnimationParams,
        )
        mock_gw.game_map.animation_state = np.zeros((2, 1), dtype=TileAnimationState)

        mock_viewport_system = Mock()
        mock_viewport_system.get_visible_bounds.return_value = Rect(0, 0, 2, 1)
        mock_viewport_system.offset_x = 0
        mock_viewport_system.offset_y = 0
        world_view.viewport_system = mock_viewport_system

        mock_lighting_system.compute_lightmap_texture.return_value = object()
        mock_graphics = Mock()
        mock_graphics.render_glyph_buffer_to_texture.return_value = object()
        mock_graphics.compose_light_overlay_gpu.return_value = object()

        terrain_decoration = Mock()
        monkeypatch.setattr(tile_types, "apply_terrain_decoration", terrain_decoration)
        edge_transition = Mock()
        world_view._apply_tile_edge_transition_data = edge_transition
        clear_calls = Mock(wraps=world_view.light_source_glyph_buffer.clear)
        monkeypatch.setattr(world_view.light_source_glyph_buffer, "clear", clear_calls)

        world_view._render_light_overlay_gpu_compose(
            mock_graphics, dark_texture=object()
        )

        explored[1, 0] = True
        visible[1, 0] = True
        mock_gw.game_map.exploration_revision = 1

        world_view._render_light_overlay_gpu_compose(
            mock_graphics, dark_texture=object()
        )

        assert clear_calls.call_count == 2
        assert terrain_decoration.call_count == 2
        assert edge_transition.call_count == 2

    def test_actor_lighting_uses_shadow_receive_scale_only(self) -> None:
        """Actor lighting intensity is now a per-actor receive-scale multiplier."""
        mock_controller = Mock()
        world_view = WorldView(mock_controller, Mock(), lighting_system=Mock())

        actor = Mock()
        actor.x = 1
        actor.y = 1
        world_view.shadow_renderer.actor_shadow_receive_light_scale[actor] = 0.4
        world_view._gpu_actor_lightmap_texture = None
        world_view._gpu_actor_lightmap_viewport_origin = None

        light_rgb = world_view.actor_renderer._get_actor_lighting_intensity(
            actor, Rect(0, 0, 2, 2)
        )
        assert light_rgb == (0.4, 0.4, 0.4)

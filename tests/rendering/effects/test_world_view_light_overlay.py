"""Tests for WorldView GPU light overlay composition logic."""

from unittest.mock import Mock

import numpy as np
import pytest

from brileta.environment import tile_types
from brileta.environment.map import TileAnimationState
from brileta.util.coordinates import Rect
from brileta.view.views.world_view import WorldView


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

        light_rgb = world_view._get_actor_lighting_intensity(actor, Rect(0, 0, 2, 2))
        assert light_rgb == (0.4, 0.4, 0.4)

"""Tests for WorldView light overlay rendering logic."""

from unittest.mock import Mock

import numpy as np

from catley.view.views.world_view import WorldView


class TestWorldViewLightOverlay:
    """Test WorldView light overlay rendering handles explored vs visible tiles."""

    def test_light_overlay_returns_none_when_no_lighting_system(self):
        """Test that light overlay returns None when lighting system is unavailable."""

        # Create WorldView without lighting system
        mock_controller = Mock()
        mock_screen_shake = Mock()
        world_view = WorldView(mock_controller, mock_screen_shake, lighting_system=None)

        mock_renderer = Mock()
        result = world_view._render_light_overlay(mock_renderer)

        assert result is None

    def test_light_overlay_processes_explored_mask_not_visible_mask(self):
        """Test that the light overlay logic checks explored tiles, not visible ones.

        This is a regression test for the bug where light overlay only processed
        visible tiles, causing explored-but-not-visible areas to appear black.
        """
        from catley.environment import tile_types
        from catley.environment.map import TileAnimationState
        from catley.util.coordinates import Rect

        # Mock controller and dependencies
        mock_controller = Mock()
        mock_screen_shake = Mock()
        mock_lighting_system = Mock()

        # Create WorldView instance
        world_view = WorldView(mock_controller, mock_screen_shake, mock_lighting_system)
        world_view.set_bounds(0, 0, 10, 10)

        # Mock game world with maps where explored > visible
        mock_gw = Mock()
        mock_controller.gw = mock_gw
        mock_gw.game_map.width = 3
        mock_gw.game_map.height = 3

        # Explored: all tiles
        explored = np.ones((3, 3), dtype=bool)
        mock_gw.game_map.explored = explored

        # Visible: only center tile
        visible = np.zeros((3, 3), dtype=bool)
        visible[1, 1] = True
        mock_gw.game_map.visible = visible

        # Create proper appearance maps with correct dtypes
        light_app = np.zeros((3, 3), dtype=tile_types.TileTypeAppearance)
        dark_app = np.zeros((3, 3), dtype=tile_types.TileTypeAppearance)
        for x in range(3):
            for y in range(3):
                light_app[x, y] = (ord("#"), (200, 200, 200), (50, 50, 50))
                dark_app[x, y] = (ord("#"), (100, 100, 100), (20, 20, 20))
        mock_gw.game_map.light_appearance_map = light_app
        mock_gw.game_map.dark_appearance_map = dark_app

        # Create animation arrays
        animation_params = np.zeros((3, 3), dtype=tile_types.TileAnimationParams)
        animation_state = np.zeros((3, 3), dtype=TileAnimationState)
        mock_gw.game_map.animation_params = animation_params
        mock_gw.game_map.animation_state = animation_state

        # Mock viewport system
        mock_viewport_system = Mock()
        mock_viewport_system.get_visible_bounds.return_value = Rect(0, 0, 3, 3)
        mock_viewport_system.offset_x = 0
        mock_viewport_system.offset_y = 0
        world_view.viewport_system = mock_viewport_system

        # Mock lighting intensity - only center tile has light
        light_intensity = np.zeros((3, 3, 3), dtype=np.float32)
        light_intensity[1, 1] = [0.5, 0.5, 0.5]
        mock_lighting_system.compute_lightmap.return_value = light_intensity

        # Run the light overlay rendering
        result = world_view._render_light_overlay(Mock())

        # Verify we got a result (method completed successfully)
        assert result is not None

        # Key assertion: ALL explored tiles should have been written to the buffer,
        # not just the visible ones. Check that corners (explored but not visible)
        # have non-zero data in the glyph buffer.
        pad = world_view._SCROLL_PADDING
        # Corner tile (0,0) is explored but not visible
        corner_ch = world_view.light_overlay_glyph_buffer.data["ch"][pad, pad]
        assert corner_ch == ord("#"), "Explored-but-not-visible tile should be rendered"

    def test_light_overlay_handles_no_explored_tiles(self):
        """Test that light overlay handles case where no tiles are explored."""

        # Mock dependencies
        mock_controller = Mock()
        mock_screen_shake = Mock()
        mock_lighting_system = Mock()

        world_view = WorldView(mock_controller, mock_screen_shake, mock_lighting_system)
        world_view.set_bounds(0, 0, 10, 10)

        # Mock game world with no explored tiles
        mock_gw = Mock()
        mock_controller.gw = mock_gw
        mock_gw.game_map.width = 3
        mock_gw.game_map.height = 3
        mock_gw.game_map.explored = np.zeros((3, 3), dtype=bool)  # No explored tiles
        mock_gw.game_map.visible = np.zeros((3, 3), dtype=bool)  # No visible tiles

        # Mock viewport system
        from catley.util.coordinates import Rect

        mock_viewport_system = Mock()
        mock_viewport_system.get_visible_bounds.return_value = Rect(0, 0, 3, 3)
        world_view.viewport_system = mock_viewport_system

        # Mock lighting intensity
        world_view.current_light_intensity = np.zeros((3, 3, 3), dtype=np.float32)
        mock_lighting_system.compute_lightmap.return_value = (
            world_view.current_light_intensity
        )

        # Should complete without error and return a GlyphBuffer
        result = world_view._render_light_overlay(Mock())
        assert result is not None

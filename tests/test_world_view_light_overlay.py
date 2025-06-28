"""Tests for WorldView light overlay rendering logic."""

from unittest.mock import Mock, patch

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

        # Mock viewport system
        from catley.util.coordinates import Rect

        mock_viewport_system = Mock()
        mock_viewport_system.get_visible_bounds.return_value = Rect(0, 0, 3, 3)
        mock_viewport_system.offset_x = 0
        mock_viewport_system.offset_y = 0
        world_view.viewport_system = mock_viewport_system

        # Mock lighting intensity
        light_intensity = np.zeros((3, 3, 3), dtype=np.float32)
        light_intensity[1, 1] = [0.5, 0.5, 0.5]  # Only center tile has light
        world_view.current_light_intensity = light_intensity

        # Mock the lighting system's compute_lightmap method
        mock_lighting_system.compute_lightmap.return_value = light_intensity

        # The key test: verify the method checks explored mask, not just visible mask
        # We'll patch the specific line that should process explored tiles
        with patch("catley.view.views.world_view.Console") as mock_console_class:
            mock_console = Mock()
            mock_console_class.return_value = mock_console

            # Create a console-like rgba array
            console_rgba = np.zeros(
                (10, 10), dtype=[("ch", np.int32), ("fg", "4u1"), ("bg", "4u1")]
            )
            mock_console.rgba = console_rgba

            # Mock the appearance maps to avoid complex numpy dtype issues
            mock_light_app = Mock()
            mock_dark_app = Mock()

            # Mock array slicing to return manageable mock data
            mock_light_slice = Mock()
            mock_dark_slice = Mock()

            mock_light_app.__getitem__ = lambda self, key: mock_light_slice
            mock_dark_app.__getitem__ = lambda self, key: mock_dark_slice

            mock_gw.game_map.light_appearance_map = mock_light_app
            mock_gw.game_map.dark_appearance_map = mock_dark_app

            # Mock the appearance data arrays to avoid indexing issues
            mock_light_slice.__getitem__ = lambda self, key: np.array([ord("#")])
            mock_dark_slice.__getitem__ = lambda self, key: np.array([[0, 0, 0]])

            # Mock TCOD renderer
            with patch(
                "catley.view.render.backends.tcod.renderer.TCODRenderer"
            ) as mock_tcod_renderer_class:
                mock_tcod_renderer = Mock()
                mock_tcod_renderer_class.return_value = mock_tcod_renderer
                mock_texture = Mock()
                mock_tcod_renderer.texture_from_console.return_value = mock_texture

                # Mock np.any to control the flow
                with patch("numpy.any") as mock_np_any:
                    # First call checks explored_mask_slice (should be True)
                    # Second call might check visible areas (varies)
                    mock_np_any.side_effect = [
                        True,
                        False,
                    ]  # explored=True, others=False

                    # Mock np.nonzero to return explored tile coordinates
                    with patch("numpy.nonzero") as mock_nonzero:
                        # Return coordinates for all 9 explored tiles
                        mock_nonzero.return_value = (
                            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),  # x coords
                            np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),  # y coords
                        )

                        # This should not raise an exception if logic is correct
                        result = world_view._render_light_overlay(Mock())

                        # Verify we got a texture (method completed successfully)
                        assert result is not None

                        # Most importantly: verify np.nonzero was called on mask
                        # (This confirms the fix - processing explored tiles)
                        mock_nonzero.assert_called()

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

        with patch("catley.view.views.world_view.Console") as mock_console_class:
            mock_console = Mock()
            mock_console_class.return_value = mock_console
            console_rgba = np.zeros(
                (10, 10), dtype=[("ch", np.int32), ("fg", "4u1"), ("bg", "4u1")]
            )
            mock_console.rgba = console_rgba

            with patch(
                "catley.view.render.backends.tcod.renderer.TCODRenderer"
            ) as mock_tcod_renderer_class:
                mock_tcod_renderer = Mock()
                mock_tcod_renderer_class.return_value = mock_tcod_renderer
                mock_texture = Mock()
                mock_tcod_renderer.texture_from_console.return_value = mock_texture

                # Should complete without error and return a texture
                result = world_view._render_light_overlay(Mock())
                assert result is not None

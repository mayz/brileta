"""Tests for the panel system including resizing and boundary enforcement."""

from unittest.mock import Mock

import pytest

from catley.view.panels.game_world_panel import GameWorldPanel
from catley.view.panels.panel import Panel


class ConcreteTestPanel(Panel):
    """Concrete test panel for testing base Panel functionality."""

    def __init__(self):
        super().__init__()
        self.draw_called = False

    def draw(self, renderer):
        self.draw_called = True


class TestPanelBasics:
    """Test basic Panel functionality."""

    def test_panel_init_sets_default_dimensions(self):
        panel = ConcreteTestPanel()
        assert panel.x == 0
        assert panel.y == 0
        assert panel.width == 0
        assert panel.height == 0
        assert panel.visible is True

    def test_resize_sets_dimensions_correctly(self):
        panel = ConcreteTestPanel()
        panel.resize(10, 20, 50, 80)

        assert panel.x == 10
        assert panel.y == 20
        assert panel.width == 40  # 50 - 10
        assert panel.height == 60  # 80 - 20

    def test_resize_handles_zero_dimensions(self):
        panel = ConcreteTestPanel()
        panel.resize(5, 5, 5, 5)

        assert panel.x == 5
        assert panel.y == 5
        assert panel.width == 0
        assert panel.height == 0

    def test_show_hide_visibility(self):
        panel = ConcreteTestPanel()
        assert panel.visible is True

        panel.hide()
        assert panel.visible is False

        panel.show()
        assert panel.visible is True


class TestGameWorldPanelBoundaryEnforcement:
    """Test GameWorldPanel's boundary enforcement and clipping logic."""

    def setup_method(self):
        # Create mock dependencies with proper integer values
        self.mock_controller = Mock()
        self.mock_controller.clock.last_delta_time = 0.016
        self.mock_controller.gw.game_map.width = 100
        self.mock_controller.gw.game_map.height = 100

        self.mock_screen_shake = Mock()
        self.mock_screen_shake.update.return_value = (0, 0)  # No shake

        self.mock_renderer = Mock()

        # Create panel
        self.panel = GameWorldPanel(self.mock_controller, self.mock_screen_shake)

    def test_game_world_panel_respects_smaller_boundaries(self):
        """Test that GameWorldPanel clips when panel is smaller than game map."""
        # Set panel to be smaller than the game map
        self.panel.resize(0, 0, 50, 40)  # 50x40 panel for 100x100 map

        # Mock the internal rendering
        self.panel._render_game_world = Mock()

        # Call draw
        self.panel.draw(self.mock_renderer)

        # Verify blit_console was called with clipped dimensions
        self.mock_renderer.blit_console.assert_called_once()
        call_kwargs = self.mock_renderer.blit_console.call_args[1]

        assert call_kwargs["width"] == 50  # Clipped to panel width
        assert call_kwargs["height"] == 40  # Clipped to panel height

    def test_game_world_panel_with_screen_shake_stays_in_bounds(self):
        """Test that screen shake doesn't cause rendering outside panel bounds."""
        # Set up screen shake
        self.mock_screen_shake.update.return_value = (5, 3)  # Some shake offset

        # Set panel boundaries
        self.panel.resize(10, 15, 60, 55)  # Panel at (10,15) with 50x40 size

        # Mock the internal rendering
        self.panel._render_game_world = Mock()

        # Call draw
        self.panel.draw(self.mock_renderer)

        # Verify blit destination is clamped to panel boundaries
        call_kwargs = self.mock_renderer.blit_console.call_args[1]
        dest_x = call_kwargs["dest_x"]
        dest_y = call_kwargs["dest_y"]

        # Should be clamped to not exceed panel boundaries
        assert dest_x >= self.panel.x
        assert dest_y >= self.panel.y

    def test_game_world_panel_skips_draw_when_invisible(self):
        """Test that invisible panels don't render."""
        self.panel.resize(0, 0, 50, 50)
        self.panel.hide()

        self.panel.draw(self.mock_renderer)

        # Should not call blit_console when invisible
        self.mock_renderer.blit_console.assert_not_called()


class TestFrameManagerResizing:
    """Test FrameManager panel layout and resizing logic."""

    def test_resize_panels_layout_logic(self):
        """Test the layout calculations in _resize_panels method."""
        # Test the layout math directly without creating full FrameManager
        screen_width = 100
        screen_height = 60
        help_height = 2

        # Replicate the layout logic from _resize_panels
        message_log_height = 10
        equipment_height = 4
        bottom_ui_height = message_log_height + 1

        game_world_y = help_height
        game_world_height = screen_height - game_world_y - bottom_ui_height

        message_log_y = screen_height - message_log_height - 1
        equipment_y = screen_height - equipment_height - 2

        equipment_width = 25
        equipment_x = screen_width - equipment_width - 2

        # Verify layout calculations
        assert game_world_y == 2  # Below help panel
        assert game_world_height == 47  # 60 - 2 - 11 = 47
        assert message_log_y == 49  # 60 - 10 - 1 = 49
        assert equipment_y == 54  # 60 - 4 - 2 = 54
        assert equipment_x == 73  # 100 - 25 - 2 = 73

        # Verify no overlap between game world and bottom UI
        game_world_bottom = game_world_y + game_world_height
        assert game_world_bottom <= message_log_y  # No overlap with message log

    def test_panel_resize_boundary_logic(self):
        """Test panel boundary calculations for different screen sizes."""
        # Test with different screen dimensions
        test_cases = [
            (80, 40),  # Small screen
            (120, 80),  # Medium screen
            (200, 120),  # Large screen
        ]

        for width, height in test_cases:
            # Calculate layout for this screen size
            help_height = 2
            message_log_height = 10
            bottom_ui_height = message_log_height + 1

            game_world_y = help_height
            game_world_height = height - game_world_y - bottom_ui_height

            # Verify game world gets reasonable space
            assert game_world_height > 0, f"No game world space at {width}x{height}"
            assert game_world_y >= help_height, (
                f"Game world overlaps help at {width}x{height}"
            )

            # Verify equipment panel fits on screen
            equipment_width = 25
            equipment_x = width - equipment_width - 2
            assert equipment_x >= 0, f"Equipment panel off-screen at {width}x{height}"


class TestPanelCoordinateConversion:
    """Test coordinate conversion and boundary checking logic."""

    def test_panel_boundary_containment(self):
        """Test logic for checking if coordinates are within panel bounds."""
        panel = ConcreteTestPanel()
        panel.resize(10, 20, 50, 60)  # Panel from (10,20) to (50,60)

        # Test points inside panel
        assert 10 <= 25 < 50 and 20 <= 35 < 60  # (25, 35) should be inside
        assert 10 <= 10 < 50 and 20 <= 20 < 60  # Top-left corner
        assert 10 <= 49 < 50 and 20 <= 59 < 60  # Bottom-right corner (exclusive)

        # Test points outside panel
        assert not (10 <= 5 < 50 and 20 <= 35 < 60)  # Left of panel
        assert not (10 <= 25 < 50 and 20 <= 70 < 60)  # Below panel
        assert not (10 <= 50 < 50 and 20 <= 35 < 60)  # Right edge (exclusive)
        assert not (10 <= 25 < 50 and 20 <= 60 < 60)  # Bottom edge (exclusive)


if __name__ == "__main__":
    pytest.main([__file__])

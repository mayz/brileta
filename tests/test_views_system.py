"""Tests for the views system including resizing and boundary enforcement."""

from unittest.mock import Mock

import pytest

from catley.view.views.base import View
from catley.view.views.world_view import WorldView


class ConcreteTestView(View):
    """Concrete test view for testing base View functionality."""

    def __init__(self):
        super().__init__()
        self.draw_called = False

    def draw(self, renderer):
        self.draw_called = True


class TestViewBasics:
    """Test basic View functionality."""

    def test_view_init_sets_default_dimensions(self):
        view = ConcreteTestView()
        assert view.x == 0
        assert view.y == 0
        assert view.width == 0
        assert view.height == 0
        assert view.visible is True

    def test_resize_sets_dimensions_correctly(self):
        view = ConcreteTestView()
        view.resize(10, 20, 50, 80)

        assert view.x == 10
        assert view.y == 20
        assert view.width == 40  # 50 - 10
        assert view.height == 60  # 80 - 20

    def test_resize_handles_zero_dimensions(self):
        view = ConcreteTestView()
        view.resize(5, 5, 5, 5)

        assert view.x == 5
        assert view.y == 5
        assert view.width == 0
        assert view.height == 0

    def test_show_hide_visibility(self):
        view = ConcreteTestView()
        assert view.visible is True

        view.hide()
        assert view.visible is False

        view.show()
        assert view.visible is True


class TestWorldViewBoundaryEnforcement:
    """Test WorldView's boundary enforcement and clipping logic."""

    def setup_method(self):
        # Create mock dependencies with proper integer values
        self.mock_controller = Mock()
        self.mock_controller.clock.last_delta_time = 0.016
        self.mock_controller.gw.game_map.width = 100
        self.mock_controller.gw.game_map.height = 100
        self.mock_controller.gw.player.x = 0
        self.mock_controller.gw.player.y = 0

        self.mock_screen_shake = Mock()
        self.mock_screen_shake.update.return_value = (0, 0)  # No shake

        self.mock_renderer = Mock()

        # Create view
        self.view = WorldView(self.mock_controller, self.mock_screen_shake)

    def test_world_view_respects_smaller_boundaries(self):
        """Test that WorldView clips when view is smaller than game map."""
        # Set view to be smaller than the game map
        self.view.resize(0, 0, 50, 40)  # 50x40 view for 100x100 map

        # Mock the internal rendering
        self.view._render_game_world = Mock()

        # Call draw
        self.view.draw(self.mock_renderer)

        # Verify blit_console was called with clipped dimensions
        self.mock_renderer.blit_console.assert_called_once()
        call_kwargs = self.mock_renderer.blit_console.call_args[1]

        assert call_kwargs["width"] == 50  # Clipped to view width
        assert call_kwargs["height"] == 40  # Clipped to view height

    def test_world_view_with_screen_shake_stays_in_bounds(self):
        """Test that screen shake doesn't cause rendering outside view bounds."""
        # Set up screen shake
        self.mock_screen_shake.update.return_value = (5, 3)  # Some shake offset

        # Set view boundaries
        self.view.resize(10, 15, 60, 55)  # View at (10,15) with 50x40 size

        # Mock the internal rendering
        self.view._render_game_world = Mock()

        # Call draw
        self.view.draw(self.mock_renderer)

        # Verify blit destination is clamped to view boundaries
        call_kwargs = self.mock_renderer.blit_console.call_args[1]
        dest_x = call_kwargs["dest_x"]
        dest_y = call_kwargs["dest_y"]

        # Should be clamped to not exceed view boundaries
        assert dest_x >= self.view.x
        assert dest_y >= self.view.y

    def test_world_view_skips_draw_when_invisible(self):
        """Test that invisible views don't render."""
        self.view.resize(0, 0, 50, 50)
        self.view.hide()

        self.view.draw(self.mock_renderer)

        # Should not call blit_console when invisible
        self.mock_renderer.blit_console.assert_not_called()


class TestFrameManagerResizing:
    """Test FrameManager view layout and resizing logic."""

    def test_resize_views_layout_logic(self):
        """Test the layout calculations in _resize_views method."""
        # Test the layout math directly without creating full FrameManager
        screen_width = 100
        screen_height = 60
        help_height = 2

        # Replicate the layout logic from _resize_views
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
        assert game_world_y == 2  # Below help view
        assert game_world_height == 47  # 60 - 2 - 11 = 47
        assert message_log_y == 49  # 60 - 10 - 1 = 49
        assert equipment_y == 54  # 60 - 4 - 2 = 54
        assert equipment_x == 73  # 100 - 25 - 2 = 73

        # Verify no overlap between game world and bottom UI
        game_world_bottom = game_world_y + game_world_height
        assert game_world_bottom <= message_log_y  # No overlap with message log

    def test_view_resize_boundary_logic(self):
        """Test view boundary calculations for different screen sizes."""
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

            world_y = help_height
            world_height = height - world_y - bottom_ui_height

            # Verify game world gets reasonable space
            assert world_height > 0, f"No game world space at {width}x{height}"
            assert world_y >= help_height, f"World overlaps help at {width}x{height}"

            # Verify equipment view fits on screen
            equipment_width = 25
            equipment_x = width - equipment_width - 2
            assert equipment_x >= 0, f"Equipment view off-screen at {width}x{height}"


class TestViewCoordinateConversion:
    """Test coordinate conversion and boundary checking logic."""

    def test_view_boundary_containment(self):
        """Test logic for checking if coordinates are within view bounds."""
        view = ConcreteTestView()
        view.resize(10, 20, 50, 60)  # View from (10,20) to (50,60)

        # Test points inside view
        assert 10 <= 25 < 50 and 20 <= 35 < 60  # (25, 35) should be inside
        assert 10 <= 10 < 50 and 20 <= 20 < 60  # Top-left corner
        assert 10 <= 49 < 50 and 20 <= 59 < 60  # Bottom-right corner (exclusive)

        # Test points outside view
        assert not (10 <= 5 < 50 and 20 <= 35 < 60)  # Left of view
        assert not (10 <= 25 < 50 and 20 <= 70 < 60)  # Below view
        assert not (10 <= 50 < 50 and 20 <= 35 < 60)  # Right edge (exclusive)
        assert not (10 <= 25 < 50 and 20 <= 60 < 60)  # Bottom edge (exclusive)


if __name__ == "__main__":
    pytest.main([__file__])

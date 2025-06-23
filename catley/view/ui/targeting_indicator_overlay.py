"""
Targeting indicator overlay for displaying targeting mode status.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import tcod.event

from catley import colors
from catley.view.render.canvas import TCODConsoleCanvas
from catley.view.ui.overlays import TextOverlay

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.view.render.canvas import Canvas


class TargetingIndicatorOverlay(TextOverlay):
    """Non-interactive overlay that displays targeting mode status."""

    def __init__(self, controller: Controller) -> None:
        super().__init__(controller)
        self.is_interactive = False

    def _get_backend(self) -> Canvas:
        """Return a TCODConsoleCanvas backend."""
        return TCODConsoleCanvas(self.renderer, transparent=True)

    def _calculate_dimensions(self) -> None:
        """Calculate dimensions for the targeting indicator."""
        text = "[ TARGETING ]"

        # Calculate overlay dimensions
        self.width = len(text) + 2  # Add 2 for padding
        self.height = 1

        # Center horizontally, place just below top bar
        self.x_tiles = (self.renderer.console_width_tiles - self.width) // 2
        self.y_tiles = 1

        # Calculate pixel dimensions
        self.tile_dimensions = self.renderer.tile_dimensions
        self.pixel_width = self.width * self.tile_dimensions[0]
        self.pixel_height = self.height * self.tile_dimensions[1]

    def draw_content(self) -> None:
        """Draw the targeting indicator text."""
        assert self.canvas is not None

        text = "[ TARGETING ]"

        # Center the text within the overlay
        text_x_offset = (self.width - len(text)) // 2

        self.canvas.draw_text(
            pixel_x=text_x_offset * self.tile_dimensions[0],
            pixel_y=0,
            text=text,
            color=colors.RED,
        )

    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle input events. Since this is non-interactive, always return False."""
        return False

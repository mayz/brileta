from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.constants.view import ViewConstants as View
from catley.util.live_vars import live_variable_registry
from catley.util.misc import to_bool
from catley.view.render.canvas import TCODConsoleCanvas
from catley.view.ui.overlays import TextOverlay

if TYPE_CHECKING:
    from catley.util.clock import Clock


class FPSOverlay(TextOverlay):
    """Simple view that displays the current frames per second."""

    def __init__(
        self,
        controller,
        clock: Clock,
        update_interval: float = View.FPS_UPDATE_INTERVAL,
    ) -> None:
        super().__init__(controller)
        self.clock = clock
        self.update_interval = update_interval
        self.last_update = getattr(
            clock, "last_time", 0.0
        )  # Handle mock clocks in tests
        self.display_string = "FPS: 0.0"
        # FPS overlay doesn't handle input - it's just a display element
        self.is_interactive = False
        # Mark as non-interactive overlay
        self.is_interactive = False

        live_variable_registry.register(
            "dev.show_fps",
            getter=lambda: self.is_active,
            setter=lambda value: self.show() if to_bool(value) else self.hide(),
            description="Toggle the FPS overlay visibility.",
        )

    def _get_backend(self):
        """Return a TCODConsoleCanvas for FPS display."""
        return TCODConsoleCanvas(self.renderer)

    def _calculate_dimensions(self) -> None:
        """Calculate overlay dimensions based on FPS text."""
        # Position in top-right corner, small size
        self.width = 20  # Tiles wide enough for FPS text
        self.height = 1  # Single line
        self.x_tiles = self.renderer.root_console.width - self.width
        self.y_tiles = 1

        tile_width, tile_height = self.tile_dimensions
        self.pixel_width = self.width * tile_width
        self.pixel_height = self.height * tile_height

    def handle_input(self, event) -> bool:
        """FPS overlay doesn't handle input."""
        return False

    def draw_content(self) -> None:
        """Draw the current FPS in the top-right corner."""

        current_time = getattr(self.clock, "last_time", 0.0)
        if current_time - self.last_update >= self.update_interval:
            mean_fps = getattr(self.clock, "mean_fps", 0.0)
            self.display_string = f"FPS: {mean_fps:4.0f}"
            self.last_update = current_time

        tile_width, tile_height = self.tile_dimensions

        # Right-align within overlay bounds
        fps_width = len(self.display_string)
        x_position = max(0, self.width - fps_width)
        if self.canvas:
            self.canvas.draw_text(
                x_position * tile_width,
                0,
                self.display_string,
                colors.YELLOW,
            )

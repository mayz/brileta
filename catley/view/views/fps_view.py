from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.config import PERFORMANCE_PROFILING
from catley.constants.view import ViewConstants as View
from catley.view.render.text_backend import TCODTextBackend

from .base import TextView

if TYPE_CHECKING:
    from catley.util.clock import Clock
    from catley.view.render.renderer import Renderer


class FPSView(TextView):
    """Simple view that displays the current frames per second."""

    def __init__(
        self,
        clock: Clock,
        update_interval: float = View.FPS_UPDATE_INTERVAL,
        *,
        renderer: Renderer,
    ) -> None:
        super().__init__()
        self.clock = clock
        self.update_interval = update_interval
        self.last_update = clock.last_time
        self.display_string = "FPS: 0.0"
        self.text_backend = TCODTextBackend(renderer)

    def needs_redraw(self, renderer: Renderer) -> bool:
        _ = renderer
        return True

    def draw_content(self, renderer: Renderer) -> None:
        """Draw the current FPS in the top-right corner."""

        current_time = self.clock.last_time
        if current_time - self.last_update >= self.update_interval:
            if PERFORMANCE_PROFILING:
                self.display_string = (
                    f"FPS: mean {self.clock.mean_fps:.0f}, "
                    f"last {self.clock.last_fps:.0f}"
                )
            else:
                self.display_string = f"FPS: {self.clock.mean_fps:.1f}"
            self.last_update = current_time

        tile_width, tile_height = self.tile_dimensions

        # Right-align within view bounds
        fps_width = len(self.display_string)
        x_position = max(0, self.width - fps_width)
        self.text_backend.draw_text(
            x_position * tile_width,
            0,
            self.display_string,
            colors.YELLOW,
        )

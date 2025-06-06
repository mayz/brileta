from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.config import PERFORMANCE_TESTING

from .panel import Panel

if TYPE_CHECKING:
    from catley.util.clock import Clock
    from catley.view.render.renderer import Renderer


class FPSPanel(Panel):
    """Simple panel that displays the current frames per second."""

    def __init__(self, clock: Clock, update_interval: float = 0.5, y: int = 0) -> None:
        super().__init__(x=0, y=y, width=0, height=1)
        self.clock = clock
        self.update_interval = update_interval
        self.last_update = clock.last_time
        self.display_string = "FPS: 0.0"

    def draw(self, renderer: Renderer) -> None:
        """Draw the current FPS in the top-right corner."""
        if not self.visible:
            return

        current_time = self.clock.last_time
        if current_time - self.last_update >= self.update_interval:
            if PERFORMANCE_TESTING:
                self.display_string = (
                    f"FPS: mean {self.clock.mean_fps:.0f}, "
                    f"last {self.clock.last_fps:.0f}"
                )
            else:
                self.display_string = f"FPS: {self.clock.mean_fps:.1f}"
            self.last_update = current_time

        fps_width = len(self.display_string)
        x_position = max(0, renderer.root_console.width - fps_width - 1)
        renderer.draw_text(x_position, self.y, self.display_string, fg=colors.YELLOW)

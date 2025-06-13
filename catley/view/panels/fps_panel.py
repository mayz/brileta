from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.config import PERFORMANCE_TESTING
from catley.constants.view import ViewConstants as View
from catley.view.text_backend import TextBackend

from .panel import Panel

if TYPE_CHECKING:
    from catley.util.clock import Clock
    from catley.view.renderer import Renderer


class FPSPanel(Panel):
    """Simple panel that displays the current frames per second."""

    def __init__(
        self,
        clock: Clock,
        update_interval: float = View.FPS_UPDATE_INTERVAL,
        *,
        text_backend: TextBackend | None = None,
    ) -> None:
        super().__init__(text_backend)
        self.clock = clock
        self.update_interval = update_interval
        self.last_update = clock.last_time
        self.display_string = "FPS: 0.0"

    def draw_content(self, renderer: Renderer) -> None:
        """Draw the current FPS in the top-right corner."""
        assert self.text_backend is not None

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

        # Right-align within panel bounds
        fps_width = len(self.display_string)
        x_position = max(0, self.width - fps_width)
        self.text_backend.draw_text(
            x_position,
            0,
            self.display_string,
            colors.YELLOW,
        )

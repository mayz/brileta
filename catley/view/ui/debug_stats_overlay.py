from __future__ import annotations

from typing import TYPE_CHECKING

import tcod.event

from catley import colors
from catley.util.live_vars import live_variable_registry
from catley.view.render.backends.tcod.canvas import TCODConsoleCanvas
from catley.view.ui.overlays import TextOverlay

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from catley.controller import Controller
    from catley.view.render.canvas import Canvas


class DebugStatsOverlay(TextOverlay):
    """Overlay that displays watched live variables."""

    def __init__(self, controller: Controller) -> None:
        super().__init__(controller)
        self.is_interactive = False

    def _get_backend(self) -> Canvas:
        return TCODConsoleCanvas(self.renderer, transparent=True)

    def _calculate_dimensions(self) -> None:
        watched = live_variable_registry.get_watched_variables()
        self.tile_dimensions = self.renderer.tile_dimensions

        if not watched:
            self.width = 0
            self.height = 0
            self.pixel_width = 0
            self.pixel_height = 0
            self.x_tiles = 0
            self.y_tiles = 0
            return

        max_width = 0
        for var in watched:
            raw_value = var.get_value()
            display_value = (
                var.formatter(raw_value) if var.formatter else str(raw_value)
            )
            line = f"{var.name}: {display_value}"
            max_width = max(max_width, len(line))

        self.width = max_width + 2  # small padding
        self.height = len(watched)

        self.x_tiles = self.renderer.console_width_tiles - self.width
        self.y_tiles = 1

        self.pixel_width = self.width * self.tile_dimensions[0]
        self.pixel_height = self.height * self.tile_dimensions[1]

    def draw_content(self) -> None:
        watched = live_variable_registry.get_watched_variables()
        assert self.canvas is not None

        tile_width, tile_height = self.tile_dimensions
        for y_offset, var in enumerate(watched):
            raw_value = var.get_value()
            display_value = (
                var.formatter(raw_value) if var.formatter else str(raw_value)
            )
            text = f"{var.name}: {display_value}"
            self.canvas.draw_text(
                0,
                y_offset * tile_height,
                text,
                colors.YELLOW,
            )

    def handle_input(self, event: tcod.event.Event) -> bool:  # pragma: no cover
        return False

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.view.render.backends.tcod.canvas import TCODConsoleCanvas

from .base import TextView

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.view.render.renderer import Renderer


class HelpTextView(TextView):
    """View that displays basic control hints at the top of the screen."""

    def __init__(self, controller: Controller, renderer: Renderer) -> None:
        super().__init__()
        self.controller = controller
        self.canvas = TCODConsoleCanvas(renderer)

    def get_cache_key(self) -> bool:
        """The key is simply whether there are items to pick up."""
        player = self.controller.gw.player
        return self.controller.gw.has_pickable_items_at_location(player.x, player.y)

    def draw_content(self, renderer: Renderer, alpha: float) -> None:
        """Render a short string with helpful key bindings."""

        tile_w, tile_h = self.tile_dimensions
        pixel_width = self.width * tile_w
        pixel_height = self.height * tile_h
        self.canvas.draw_rect(0, 0, pixel_width, pixel_height, colors.BLACK, fill=True)

        help_items = ["?: Help", "I: Inventory", "Space: Actions"]

        player = self.controller.gw.player
        if self.controller.gw.has_pickable_items_at_location(player.x, player.y):
            help_items.append("G: Get items")

        text = " | ".join(help_items)
        self.canvas.draw_text(0, 0, text, colors.GREY)

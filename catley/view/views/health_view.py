from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.view.render.base_renderer import Renderer
from catley.view.render.canvas import TCODConsoleCanvas

from .base import TextView

if TYPE_CHECKING:
    from catley.controller import Controller


class HealthView(TextView):
    """View that displays the player's HP and AP status."""

    def __init__(self, controller: Controller, renderer: Renderer) -> None:
        """Initialize the view.

        Position and size will be set by FrameManager.resize().
        """

        super().__init__()
        self.controller = controller
        self.canvas = TCODConsoleCanvas(renderer)

    def get_cache_key(self) -> tuple[int, int, int]:
        """The key is a tuple of the health values. If it hasn't changed, no redraw."""
        player = self.controller.gw.player
        return (player.health.hp, player.health.max_hp, player.health.ap)

    def draw_content(self, renderer: Renderer) -> None:
        tile_width, tile_height = self.tile_dimensions
        pixel_width = self.width * tile_width
        pixel_height = self.height * tile_height
        self.canvas.draw_rect(0, 0, pixel_width, pixel_height, colors.BLACK, fill=True)

        player = self.controller.gw.player
        text = f"HP: {player.health.hp}/{player.health.max_hp} AP: {player.health.ap}"

        x_pos_tiles = self.width - len(text) - 1
        self.canvas.draw_text(x_pos_tiles * tile_width, 0, text, colors.WHITE)

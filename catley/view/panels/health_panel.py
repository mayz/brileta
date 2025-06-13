from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.view.render.renderer import Renderer
from catley.view.render.text_backend import TCODTextBackend

from .panel import TextPanel

if TYPE_CHECKING:
    from catley.controller import Controller


class HealthPanel(TextPanel):
    """Panel that displays the player's HP and AP status."""

    def __init__(self, controller: Controller, renderer: Renderer) -> None:
        """Initialize the panel.

        Position and size will be set by FrameManager.resize().
        """

        super().__init__()
        self.controller = controller
        self.text_backend = TCODTextBackend(renderer)

    def needs_redraw(self, renderer: Renderer) -> bool:
        _ = renderer
        return True

    def draw_content(self, renderer: Renderer) -> None:
        player = self.controller.gw.player
        tile_width, tile_height = self.tile_dimensions
        text = f"HP: {player.health.hp}/{player.health.max_hp} AP: {player.health.ap}"

        # Calculate x in tiles, then convert to pixels
        x_pos_tiles = self.width - len(text) - 1
        self.text_backend.draw_text(x_pos_tiles * tile_width, 0, text, colors.WHITE)

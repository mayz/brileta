from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.render.renderer import Renderer
from catley.ui.panel import Panel

if TYPE_CHECKING:
    from catley.controller import Controller


class HealthPanel(Panel):
    """Panel that displays the player's HP and AP status."""

    def __init__(self, controller: Controller, x: int, y: int) -> None:
        # Width is dynamic based on text length; height is a single line.
        super().__init__(x=x, y=y, width=0, height=1)
        self.controller = controller

    def draw(self, renderer: Renderer) -> None:
        if not self.visible:
            return

        player = self.controller.gw.player
        text = f"HP: {player.health.hp}/{player.health.max_hp} AP: {player.health.ap}"
        renderer.draw_text(self.x, self.y, text, fg=colors.WHITE)

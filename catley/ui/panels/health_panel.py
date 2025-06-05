from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.render.renderer import Renderer
from catley.ui.panel import Panel

if TYPE_CHECKING:
    from catley.controller import Controller


class HealthPanel(Panel):
    """Panel that displays the player's HP and AP status."""

    def __init__(self, controller: Controller, y: int = 0) -> None:
        """Initialize the panel.

        ``x`` is ignored during drawing since the text is right-aligned
        dynamically each frame. ``y`` defaults to the top row just below any
        help text.
        """

        super().__init__(x=0, y=y, width=0, height=1)
        self.controller = controller

    def draw(self, renderer: Renderer) -> None:
        if not self.visible:
            return

        player = self.controller.gw.player
        text = f"HP: {player.health.hp}/{player.health.max_hp} AP: {player.health.ap}"
        # Right align with a one tile margin.
        x_pos = renderer.root_console.width - len(text) - 1
        renderer.draw_text(x_pos, self.y, text, fg=colors.WHITE)

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.ui.panels.panel import Panel

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.render.renderer import Renderer


class HelpTextPanel(Panel):
    """Panel that displays basic control hints at the top of the screen."""

    def __init__(self, controller: Controller, y: int = 0) -> None:
        super().__init__(x=1, y=y, width=0, height=1)
        self.controller = controller

    def draw(self, renderer: Renderer) -> None:
        """Render a short string with helpful key bindings."""
        if not self.visible:
            return

        help_items = ["Arrows/HJKL: Move", "?: Help", "I: Inventory"]

        player = self.controller.gw.player
        if self.controller.gw.has_pickable_items_at_location(player.x, player.y):
            help_items.append("G: Get items")

        text = " | ".join(help_items)
        renderer.draw_text(self.x, self.y, text, fg=colors.GREY)

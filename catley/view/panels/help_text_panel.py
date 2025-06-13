from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.view.panels.panel import TextPanel
from catley.view.render.text_backend import TCODTextBackend

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.view.render.renderer import Renderer


class HelpTextPanel(TextPanel):
    """Panel that displays basic control hints at the top of the screen."""

    def __init__(self, controller: Controller, renderer: Renderer) -> None:
        super().__init__()
        self.controller = controller
        self.text_backend = TCODTextBackend(renderer)

    def needs_redraw(self, renderer: Renderer) -> bool:
        _ = renderer
        return True

    def draw_content(self, renderer: Renderer) -> None:
        """Render a short string with helpful key bindings."""

        help_items = ["?: Help", "I: Inventory", "Space: Actions"]

        player = self.controller.gw.player
        if self.controller.gw.has_pickable_items_at_location(player.x, player.y):
            help_items.append("G: Get items")

        text = " | ".join(help_items)
        self.text_backend.draw_text(0, 0, text, colors.GREY)

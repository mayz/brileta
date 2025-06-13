from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.view.renderer import Renderer
from catley.view.text_backend import TextBackend

from .panel import Panel

if TYPE_CHECKING:
    from catley.controller import Controller


class HealthPanel(Panel):
    """Panel that displays the player's HP and AP status."""

    def __init__(
        self, controller: Controller, *, text_backend: TextBackend | None = None
    ) -> None:
        """Initialize the panel.

        Position and size will be set by FrameManager.resize().
        """

        super().__init__(text_backend)
        self.controller = controller

    def draw_content(self, renderer: Renderer) -> None:
        assert self.text_backend is not None

        player = self.controller.gw.player
        text = f"HP: {player.health.hp}/{player.health.max_hp} AP: {player.health.ap}"
        # Right align with a one tile margin.
        x_pos = self.width - len(text) - 1
        self.text_backend.draw_text(x_pos, 0, text, colors.WHITE)

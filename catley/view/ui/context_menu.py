from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.game.actors import Actor, Character
from catley.view.ui.overlays import Menu, MenuOption

if TYPE_CHECKING:  # pragma: no cover - only for type checking
    from catley.controller import Controller


class ContextMenu(Menu):
    """Simple right-click context menu placeholder."""

    def __init__(
        self,
        controller: Controller,
        target: Actor | tuple[int, int] | None,
        click_position: tuple[int, int],
    ) -> None:
        self.target = target
        self.click_position = click_position

        title = self._get_title()
        super().__init__(title, controller, width=30, max_height=10)

    def _get_title(self) -> str:
        if isinstance(self.target, Actor):
            return f"Actions for {self.target.name}"
        if isinstance(self.target, tuple):
            return "Tile Actions"
        return "Context"

    def populate_options(self) -> None:  # pragma: no cover - simple placeholder
        self.options.clear()
        if isinstance(self.target, Actor):
            alive_str = (
                "alive"
                if not isinstance(self.target, Character)
                or self.target.health.is_alive()
                else "dead"
            )
            text = f"Target: {self.target.name} ({alive_str})"
        elif isinstance(self.target, tuple):
            x, y = self.target
            text = f"Empty tile ({x}, {y})"
        else:
            text = "No target"
        self.add_option(
            MenuOption(key=None, text=text, enabled=False, is_primary_action=False)
        )
        self.add_option(
            MenuOption(
                key=None,
                text="(actions will be implemented in Task 2)",
                enabled=False,
                color=colors.GREY,
                is_primary_action=False,
            )
        )

    def _calculate_dimensions(self) -> None:
        super()._calculate_dimensions()
        x, y = self.click_position
        x += 1
        y += 1
        max_x = self.renderer.root_console.width - self.width
        max_y = self.renderer.root_console.height - self.height
        self.x_tiles = max(0, min(x, max_x))
        self.y_tiles = max(0, min(y, max_y))

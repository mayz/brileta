from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors, config
from catley.backends.pillow.canvas import PillowImageCanvas
from catley.view.render.canvas import Canvas
from catley.view.ui.overlays import Menu, MenuOption

if TYPE_CHECKING:
    from catley.controller import Controller


class HelpMenu(Menu):
    """Menu showing all available commands."""

    FONT_SIZE = 36

    def __init__(self, controller: Controller) -> None:
        super().__init__("Help - Available Commands", controller, width=60)

    def _get_backend(self) -> Canvas:
        """Return a PillowImageCanvas with larger font for keycap-heavy content."""
        return PillowImageCanvas(
            self.controller.graphics,
            font_path=config.UI_FONT_PATH,
            font_size=self.FONT_SIZE,
            line_spacing=config.MENU_LINE_SPACING,
        )

    def populate_options(self) -> None:
        """Populate help options.

        Items with a single-key binding use MenuOption.key for keycap rendering.
        Items with multi-key or non-keyboard bindings leave key=None.
        """
        # Format: (key or None, command name, description)
        help_items: list[tuple[str | None, str, str]] = [
            (None, "Movement", "Arrow keys or numpad"),
            (None, "Select Actor", "Left-click to select"),
            ("T", "Target Menu", "Full targeting options for selected actor"),
            ("I", "Inventory", "Open inventory (pick up items when on ground)"),
            ("`", "Dev Console", "Open developer console"),
            ("Q", "Quit", "Quit game"),
            ("?", "Help", "Show this help menu"),
        ]

        for key, command, description in help_items:
            self.add_option(
                MenuOption(
                    key=key,
                    text=f"{command} - {description}",
                    enabled=False,
                    color=colors.WHITE,
                    is_primary_action=False,
                )
            )

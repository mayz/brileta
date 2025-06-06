from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.view.ui.menu_system import Menu, MenuOption

if TYPE_CHECKING:
    from catley.controller import Controller


class HelpMenu(Menu):
    """Menu showing all available commands."""

    def __init__(self, controller: Controller) -> None:
        super().__init__("Help - Available Commands", controller, width=60)

    def populate_options(self) -> None:
        """Populate help options."""
        help_items = [
            ("Movement", "Arrow keys or numpad"),
            ("Select Actor", "Left-click to select"),
            ("Target Menu", "T - Full targeting options for selected actor"),
            ("Inventory", "I - Open inventory menu"),
            ("Get Items", "G - Pick up items from ground/corpses"),
            ("Quit", "Q or Escape - Quit game"),
            ("Help", "? - Show this help menu"),
        ]

        # Add sections without keys (just for display)
        for command, description in help_items:
            self.add_option(
                MenuOption(
                    key=None,
                    text=f"{command:<15} {description}",
                    enabled=False,
                    color=colors.WHITE,
                )
            )

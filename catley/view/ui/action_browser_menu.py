from __future__ import annotations

import functools
import string
from typing import TYPE_CHECKING

from catley import colors
from catley.game.actions.discovery import ActionCategory, ActionDiscovery
from catley.view.ui.overlays import Menu, MenuOption

if TYPE_CHECKING:
    from catley.controller import Controller


class ActionBrowserMenu(Menu):
    """Menu for browsing and selecting available actions."""

    def __init__(self, controller: Controller) -> None:
        super().__init__("Available Actions", controller, width=60)
        self.action_discovery = ActionDiscovery()

    def populate_options(self) -> None:
        """Populate action options as menu choices."""
        self.options.clear()

        player = self.controller.gw.player
        action_options = self.action_discovery.get_available_options(
            self.controller, player
        )

        # Use letters a-z for action options
        letters = string.ascii_lowercase
        letter_idx = 0

        if not action_options:
            self.add_option(
                MenuOption(
                    key=None,
                    text="(no actions available)",
                    enabled=False,
                    color=colors.GREY,
                )
            )
            return

        for action_option in action_options:
            # Assign hotkey or use action's preferred hotkey
            key = action_option.hotkey
            if not key and letter_idx < len(letters):
                key = letters[letter_idx]
                letter_idx += 1

            # Get category color
            color = self._get_category_color(action_option.category)

            self.add_option(
                MenuOption(
                    key=key,
                    text=action_option.display_text,
                    action=functools.partial(
                        self._execute_action_option, action_option
                    ),
                    enabled=True,
                    color=color,
                    force_color=True,
                )
            )

    def _execute_action_option(self, action_option) -> None:
        """Execute an action option."""
        if action_option.execute:
            game_action = action_option.execute()
            if game_action:
                self.controller.queue_action(game_action)

    def _get_category_color(self, category: ActionCategory) -> colors.Color:
        """Get display color for action category."""
        color_map = {
            ActionCategory.COMBAT: colors.RED,
            ActionCategory.MOVEMENT: colors.BLUE,
            ActionCategory.ITEMS: colors.GREEN,
            ActionCategory.ENVIRONMENT: colors.ORANGE,
            ActionCategory.SOCIAL: colors.MAGENTA,
        }
        return color_map.get(category, colors.WHITE)

from __future__ import annotations

import functools
import string
from typing import TYPE_CHECKING

import tcod

from catley import colors
from catley.game import ranges
from catley.game.actions.base import GameAction
from catley.game.actions.combat import AttackAction
from catley.game.actions.discovery import (
    ActionCategory,
    ActionDiscovery,
    ActionOption,
    CombatIntentCache,
)
from catley.view.ui.overlays import REPEAT_PREFIX, Menu, MenuOption

if TYPE_CHECKING:
    from catley.controller import Controller


class ActionBrowserMenu(Menu):
    """Menu for browsing and selecting available actions."""

    def __init__(self, controller: Controller) -> None:
        super().__init__("Available Actions", controller, width=60)
        self.action_discovery = ActionDiscovery()

    def show(self) -> None:
        """Display the menu and reset discovery state."""
        # Reset state machine so each use starts from the main screen
        self.action_discovery.ui_state = "main"
        self.action_discovery.selected_target = None
        self.action_discovery.selected_weapon = None
        self.action_discovery.selected_attack_mode = None

        super().show()

    def populate_options(self) -> None:
        """Populate action options as menu choices."""
        self.options.clear()

        player = self.controller.gw.player

        action_options_for_display = self.action_discovery.get_available_options(
            self.controller,
            player,
        )

        cache = self.controller.combat_intent_cache
        if cache:
            gm = self.controller.gw.game_map
            repeat_option = None
            if (
                cache.target
                and cache.target.health.is_alive()
                and gm.visible[cache.target.x, cache.target.y]
                and ranges.has_line_of_sight(
                    gm,
                    player.x,
                    player.y,
                    cache.target.x,
                    cache.target.y,
                )
            ):
                all_possible_actions = (
                    self.action_discovery._get_all_terminal_combat_actions(
                        self.controller,
                        player,
                    )
                )
                for opt in all_possible_actions:
                    if not opt.execute:
                        continue
                    action = opt.execute()
                    if (
                        isinstance(action, AttackAction)
                        and action.weapon == cache.weapon
                        and action.attack_mode == cache.attack_mode
                        and action.defender == cache.target
                    ):
                        repeat_option = opt
                        break

            if repeat_option:
                self.add_option(
                    MenuOption(
                        key=None,
                        text=f"{REPEAT_PREFIX} {repeat_option.menu_text}",
                        action=functools.partial(
                            self._execute_action_option, repeat_option
                        ),
                        color=colors.WHITE,
                    )
                )
                self.add_option(MenuOption(key=None, text="-" * 40, enabled=False))
            else:
                # Always offer to continue with the cached weapon/mode.
                # The target selection screen will show if no targets remain.
                self.add_option(
                    MenuOption(
                        key=None,
                        text=(
                            f"[Enter] Continue: {cache.attack_mode.title()} "
                            f"with {cache.weapon.name}..."
                        ),
                        action=functools.partial(
                            self.action_discovery._set_ui_state,
                            "targets_for_weapon",
                            weapon=cache.weapon,
                            attack_mode=cache.attack_mode,
                        ),
                        color=colors.WHITE,
                    )
                )
                self.add_option(MenuOption(key=None, text="-" * 40, enabled=False))

        action_options = action_options_for_display

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

            # Get probability color if available, otherwise use category color
            if action_option.success_probability is not None:
                _, prob_color_name = ActionDiscovery.get_probability_descriptor(
                    action_option.success_probability
                )
                color_map = {
                    "red": colors.RED,
                    "orange": colors.ORANGE,
                    "yellow": colors.YELLOW,
                    "light_green": colors.LIGHT_GREEN,
                    "green": colors.GREEN,
                }
                prob_color = color_map.get(prob_color_name, colors.WHITE)
            else:
                prob_color = self._get_category_color(action_option.category)

            self.add_option(
                MenuOption(
                    key=key,
                    text=action_option.menu_text,
                    action=functools.partial(
                        self._execute_action_option, action_option
                    ),
                    enabled=True,
                    color=prob_color,
                    force_color=True,
                )
            )

    def _execute_action_option(self, action_option: ActionOption) -> bool:
        """Execute an action option. Returns True if menu should close."""
        execute_fn = action_option.execute
        if execute_fn is not None:
            result = execute_fn()
            if isinstance(result, GameAction):
                self.controller.queue_action(result)
                if isinstance(result, AttackAction):
                    weapon = result.weapon
                    attack_mode = result.attack_mode
                    if weapon is not None and attack_mode is not None:
                        # A combat action was taken. Create and set the cache.
                        self.controller.combat_intent_cache = CombatIntentCache(
                            weapon=weapon,
                            attack_mode=attack_mode,
                            target=result.defender,
                        )
                else:
                    # A non-combat action was taken. Clear the cache.
                    self.controller.combat_intent_cache = None
                return True
            # This was a state change (e.g., entering a sub-menu), not a final action
            return False
        # This was likely a "Back" button or similar UI-only action
        return True

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

    def handle_input(self, event) -> bool:
        """State-driven input handling for the hierarchical menu."""
        if not self.is_active:
            return False

        match event:
            case tcod.event.KeyDown(sym=tcod.event.KeySym.ESCAPE):
                # Navigate back in the state machine
                self.action_discovery._go_back(self.controller)
                if self.action_discovery.ui_state == "main":
                    self.hide()
                else:
                    self.populate_options()
                return True
            case tcod.event.KeyDown(sym=tcod.event.KeySym.RETURN):
                if (
                    self.options
                    and self.options[0].text.startswith("[Enter]")
                    and self.options[0].action
                ):
                    should_close = self.options[0].action()
                    if should_close:
                        self.hide()
                    else:
                        self.populate_options()
                    return True
                self.hide()
                return True
            case tcod.event.KeyDown() as key_event:
                key_char = (
                    chr(key_event.sym).lower() if 32 <= key_event.sym <= 126 else ""
                )
                for option in self.options:
                    if (
                        option.key is not None
                        and option.key.lower() == key_char
                        and option.enabled
                        and option.action
                    ):
                        should_close = option.action()
                        if should_close:
                            self.hide()
                        else:
                            self.populate_options()
                        return True
                return super().handle_input(event)

        return super().handle_input(event)

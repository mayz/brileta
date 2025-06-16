from __future__ import annotations

import functools
import string
from typing import TYPE_CHECKING

import tcod.event

from catley import colors
from catley.environment import tile_types
from catley.game.actions.base import GameIntent
from catley.game.actions.combat import AttackIntent
from catley.game.actions.discovery import (
    ActionCategory,
    ActionDiscovery,
    ActionFormatter,
    ActionOption,
    CombatIntentCache,
)
from catley.game.actions.environment import CloseDoorIntent, OpenDoorIntent
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
        self.discovery = ActionDiscovery()

        title = self._get_title()
        super().__init__(title, controller, width=30, max_height=10)

    def _get_title(self) -> str:
        if isinstance(self.target, Actor):
            return f"Actions for {self.target.name}"
        if isinstance(self.target, tuple):
            return "Tile Actions"
        return "Context"

    def populate_options(self) -> None:
        self.options.clear()

        if self.target is None:
            return

        player = self.controller.gw.player
        gm = self.controller.gw.game_map

        action_options: list[ActionOption] = []

        if isinstance(self.target, Character):
            action_options.extend(
                self.discovery.get_options_for_target(
                    self.controller, player, self.target
                )
            )
        elif isinstance(self.target, tuple):
            x, y = self.target
            tile = gm.tiles[x, y]
            distance = abs(player.x - x) + abs(player.y - y)

            # TODO: FUTURE ENHANCEMENT - Implement pathfinding system so players
            # can right-click distant doors and automatically walk over to interact
            # with them. This will require pathfinding, autonomous movement, and an
            # action queue. For now, only show door actions when adjacent.
            if distance <= 1:
                if tile == tile_types.TILE_TYPE_ID_DOOR_CLOSED:  # type: ignore[attr-defined]
                    action_options.append(
                        ActionOption(
                            id="open-door-specific",
                            name="Open Door",
                            description="Open the door",
                            category=ActionCategory.ENVIRONMENT,
                            action_class=OpenDoorIntent,
                            requirements=[],
                            static_params={"x": x, "y": y},
                        )
                    )
                elif tile == tile_types.TILE_TYPE_ID_DOOR_OPEN:  # type: ignore[attr-defined]
                    action_options.append(
                        ActionOption(
                            id="close-door-specific",
                            name="Close Door",
                            description="Close the door",
                            category=ActionCategory.ENVIRONMENT,
                            action_class=CloseDoorIntent,
                            requirements=[],
                            static_params={"x": x, "y": y},
                        )
                    )

        letters = string.ascii_lowercase
        letter_idx = 0

        for option in action_options:
            key = option.hotkey
            if not key and letter_idx < len(letters):
                key = letters[letter_idx]
                letter_idx += 1

            if option.success_probability is not None:
                _, color_name = ActionDiscovery.get_probability_descriptor(
                    option.success_probability
                )
                color_map = {
                    "red": colors.RED,
                    "orange": colors.ORANGE,
                    "yellow": colors.YELLOW,
                    "light_green": colors.LIGHT_GREEN,
                    "green": colors.GREEN,
                }
                display_color = color_map.get(color_name, colors.WHITE)
            else:
                display_color = ActionFormatter.get_category_color(option.category)

            self.add_option(
                MenuOption(
                    key=key,
                    text=option.menu_text,
                    action=functools.partial(self._execute_action_option, option),
                    enabled=True,
                    color=display_color,
                    force_color=True,
                    is_primary_action=option.id != "back",
                )
            )

        if not self.options:
            self.add_option(
                MenuOption(
                    key=None,
                    text="(no actions available)",
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

    def handle_input(self, event: tcod.event.Event) -> bool:
        """Close the menu if a mouse click occurs outside its bounds."""

        if not self.is_active:
            return False

        match event:
            case tcod.event.MouseButtonDown(position=position):
                tx, ty = self.controller.coordinate_converter.pixel_to_tile(
                    position[0], position[1]
                )
                if not (
                    self.x_tiles <= tx < self.x_tiles + self.width
                    and self.y_tiles <= ty < self.y_tiles + self.height
                ):
                    self.hide()
                    return True

        return super().handle_input(event)

    def _execute_action_option(self, action_option: ActionOption) -> bool:
        """Execute an action option and return True if the menu should close."""

        if hasattr(action_option, "execute") and action_option.execute:
            result = action_option.execute()
            if isinstance(result, GameIntent):
                self.controller.queue_action(result)
                if isinstance(result, AttackIntent):
                    weapon = result.weapon
                    attack_mode = result.attack_mode
                    if weapon and attack_mode:
                        self.controller.combat_intent_cache = CombatIntentCache(
                            weapon=weapon,
                            attack_mode=attack_mode,
                            target=result.defender,
                        )
                else:
                    self.controller.combat_intent_cache = None
                return True
            return bool(result)

        if action_option.action_class:
            try:
                action_instance = action_option.action_class(
                    self.controller,
                    self.controller.gw.player,
                    **action_option.static_params,
                )
            except TypeError:
                return True
            self.controller.queue_action(action_instance)
            if isinstance(action_instance, AttackIntent):
                weapon = action_instance.weapon
                attack_mode = action_instance.attack_mode
                if weapon and attack_mode:
                    self.controller.combat_intent_cache = CombatIntentCache(
                        weapon=weapon,
                        attack_mode=attack_mode,
                        target=action_instance.defender,
                    )
            else:
                self.controller.combat_intent_cache = None
            return True

        if hasattr(action_option, "execute") and action_option.execute:
            result = action_option.execute()
            return bool(result)

        return True

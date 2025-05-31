from __future__ import annotations

import functools
import string
from typing import TYPE_CHECKING

from catley import colors
from catley.ui.menu_core import Menu, MenuOption

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.items.item_core import Item


class PickupMenu(Menu):
    """Menu for picking up items from the ground or from dead bodies."""

    def __init__(self, controller: Controller, location: tuple[int, int]) -> None:
        super().__init__("Pick up items", controller)
        self.location = location

    def populate_options(self) -> None:
        """Populate pickup options based on items at the location."""
        items_here = self._get_items_at_location()

        if not items_here:
            self.add_option(
                MenuOption(
                    key=None, text="(no items here)", enabled=False, color=colors.GREY
                )
            )
            return

        # Use letters a-z for items
        letters = string.ascii_lowercase
        for i, item in enumerate(items_here):
            if i >= len(letters):
                break

            key = letters[i]
            self.add_option(
                MenuOption(
                    key=key,
                    text=item.name,
                    action=functools.partial(self._pickup_item, item),
                    color=colors.WHITE,
                )
            )

    def _get_items_at_location(self) -> list[Item]:
        """Get all items at the specified location."""
        return self.controller.gw.get_pickable_items_at_location(
            self.location[0], self.location[1]
        )

    def _pickup_item(self, item: Item) -> None:
        """Pickup an item and add it to player inventory."""
        player = self.controller.gw.player
        # Check if player has inventory space using the new size-aware method
        if not player.inventory.can_add_to_inventory(item):
            self.controller.message_log.add_message(
                "Your inventory is full!", colors.RED
            )
            return

        # Remove item from dead actor
        for actor in self.controller.gw.actors:
            if actor.health and not actor.health.is_alive():
                if item in actor.inventory:
                    actor.inventory.remove_from_inventory(item)
                    break

                if actor.inventory.equipped_weapon == item:
                    actor.inventory.equipped_weapon = None
                    break

        # Add to player inventory
        player.inventory.add_to_inventory(item)
        self.controller.message_log.add_message(
            f"You pick up {item.name}.", colors.WHITE
        )

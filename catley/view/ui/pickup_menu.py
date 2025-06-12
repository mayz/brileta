from __future__ import annotations

import functools
import string
from typing import TYPE_CHECKING

from catley import colors
from catley.events import MessageEvent, publish_event
from catley.game.actors import Character
from catley.game.items.item_core import Item
from catley.view.ui.overlays import Menu, MenuOption

if TYPE_CHECKING:
    from catley.controller import Controller


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
        if not player.inventory.can_add_voluntary_item(item):
            publish_event(MessageEvent("Your inventory is full!", colors.RED))
            return

        # Remove item from whichever actor at the location actually holds it
        actors_here = [
            a
            for a in self.controller.gw.actors
            if a.x == self.location[0] and a.y == self.location[1]
        ]
        for actor in actors_here:
            inv = getattr(actor, "inventory", None)
            if inv is None:
                continue

            removed = False
            if item in inv:
                inv.remove_from_inventory(item)
                removed = True

            for i, equipped_item in enumerate(inv.attack_slots):
                if equipped_item == item:
                    inv.unequip_slot(i)
                    removed = True
                    break

            if removed:
                if (
                    not isinstance(actor, Character)
                    and len(inv) == 0
                    and all(s is None for s in inv.attack_slots)
                ):
                    # Remove empty ground container
                    self.controller.gw.remove_actor(actor)
                break

        # Add to player inventory
        success, message, dropped_items = player.inventory.add_to_inventory(item)
        publish_event(MessageEvent(f"You pick up {item.name}.", colors.WHITE))
        for dropped in dropped_items:
            self.controller.gw._spawn_dropped_item(dropped, player.x, player.y)

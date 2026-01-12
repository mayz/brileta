from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.events import MessageEvent, publish_event
from catley.game.actions.base import GameActionResult
from catley.game.actions.executors.base import ActionExecutor
from catley.game.actors import Character
from catley.game.items.item_core import Item

if TYPE_CHECKING:
    from catley.game.actions.misc import (
        DropItemIntent,
        PickupIntent,
        PickupItemsAtLocationIntent,
        SwitchWeaponIntent,
    )


class PickupExecutor(ActionExecutor):
    """Executes pickup intents by adding items to actor inventory."""

    def __init__(self) -> None:
        """Create a PickupExecutor without requiring a controller."""
        pass

    def execute(self, intent: PickupIntent) -> GameActionResult | None:  # type: ignore[override]
        """Pick up items from the ground, removing them from source actors."""
        actor_inv = getattr(intent.actor, "inventory", None)
        if actor_inv is None:
            return GameActionResult(succeeded=False)

        picked_up: list[str] = []
        skipped: list[str] = []

        for item in list(intent.items):
            if not actor_inv.can_add_voluntary_item(item):
                skipped.append(item.name)
                continue

            # Remove item from source actor (dead body or ground container)
            removed = self._remove_item_from_ground(intent, item)
            if not removed:
                # Item may have already been picked up or moved
                continue

            # Add to actor's inventory
            success, _message, dropped_items = actor_inv.add_to_inventory(item)
            if success:
                picked_up.append(item.name)
                # Spawn any dropped items at actor's location
                for dropped in dropped_items:
                    intent.controller.gw.spawn_ground_item(
                        dropped, intent.actor.x, intent.actor.y
                    )

        # Provide feedback
        if picked_up:
            if len(picked_up) == 1:
                publish_event(
                    MessageEvent(f"You pick up {picked_up[0]}.", colors.WHITE)
                )
            else:
                publish_event(
                    MessageEvent(f"You pick up {len(picked_up)} items.", colors.WHITE)
                )

        if skipped:
            publish_event(MessageEvent("Your inventory is full!", colors.RED))

        return GameActionResult(succeeded=bool(picked_up))

    def _remove_item_from_ground(self, intent: PickupIntent, item: Item) -> bool:
        """Remove an item from its source actor at the pickup location.

        Returns True if the item was successfully removed.
        """
        # Find actors at the actor's location that might hold this item
        gw = intent.controller.gw
        actors_here = [
            a for a in gw.actors if a.x == intent.actor.x and a.y == intent.actor.y
        ]

        for actor in actors_here:
            inv = getattr(actor, "inventory", None)
            if inv is None:
                continue

            removed = False

            # Check stored inventory
            if item in inv:
                inv.remove_from_inventory(item)
                removed = True

            # Check equipped slots
            if not removed:
                for i, equipped_item in enumerate(inv.attack_slots):
                    if equipped_item == item:
                        inv.unequip_slot(i)
                        removed = True
                        break

            if removed:
                # Remove empty ground containers (non-Character actors)
                if (
                    not isinstance(actor, Character)
                    and len(inv) == 0
                    and all(s is None for s in inv.attack_slots)
                ):
                    gw.remove_actor(actor)
                return True

        return False


class PickupItemsAtLocationExecutor(ActionExecutor):
    """Executes pickup at location intents by opening the pickup menu."""

    def __init__(self) -> None:
        """Create a PickupItemsAtLocationExecutor without requiring a controller."""
        pass

    def execute(self, intent: PickupItemsAtLocationIntent) -> GameActionResult | None:  # type: ignore[override]
        # Open the pickup menu UI for items at the actor's location
        from catley.view.ui.commands import OpenPickupMenuUICommand

        # Check if there are items at the actor's location
        items = intent.controller.gw.get_pickable_items_at_location(
            intent.actor.x, intent.actor.y
        )
        if items:
            # Open the pickup menu
            pickup_cmd = OpenPickupMenuUICommand(intent.controller)
            pickup_cmd.execute()
        return GameActionResult()


class SwitchWeaponExecutor(ActionExecutor):
    """Executes weapon switching intents."""

    def __init__(self) -> None:
        """Create a SwitchWeaponExecutor without requiring a controller."""
        pass

    def execute(self, intent: SwitchWeaponIntent) -> GameActionResult | None:  # type: ignore[override]
        # Move logic from old SwitchWeaponAction.execute()
        inventory = getattr(intent.actor, "inventory", None)
        if inventory:
            inventory.switch_to_weapon_slot(intent.slot)
        return GameActionResult()


class DropItemExecutor(ActionExecutor):
    """Executes drop item intents.

    Removes items from inventory and places them on the ground.
    """

    def __init__(self) -> None:
        """Create a DropItemExecutor without requiring a controller."""
        pass

    def execute(self, intent: DropItemIntent) -> GameActionResult | None:  # type: ignore[override]
        """Drop an item from inventory to the ground at actor's location."""
        inventory = getattr(intent.actor, "inventory", None)
        if inventory is None:
            return GameActionResult(succeeded=False)

        item = intent.item
        removed = False

        # Check if item is equipped in an attack slot and unequip it
        for i, slot_item in enumerate(inventory.attack_slots):
            if slot_item is item:
                inventory.attack_slots[i] = None
                removed = True
                break

        # If not equipped, remove from stored inventory
        if not removed and not inventory.remove_from_inventory(item):
            publish_event(MessageEvent("Item not found in inventory.", colors.RED))
            return GameActionResult(succeeded=False)

        # Spawn the item on the ground at actor's location
        intent.controller.gw.spawn_ground_item(item, intent.actor.x, intent.actor.y)
        publish_event(MessageEvent(f"You drop {item.name}.", colors.WHITE))

        return GameActionResult()

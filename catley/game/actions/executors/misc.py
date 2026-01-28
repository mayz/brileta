from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.events import MessageEvent, publish_event
from catley.game.actions.base import GameActionResult
from catley.game.actions.executors.base import ActionExecutor
from catley.game.actors import Character, ItemPile
from catley.game.countables import get_countable_definition
from catley.game.items.item_core import Item
from catley.game.items.properties import WeaponProperty

if TYPE_CHECKING:
    from catley.game.actions.misc import (
        DropCountableIntent,
        DropItemIntent,
        PickupCountableIntent,
        PickupIntent,
        PickupItemsAtLocationIntent,
        SwitchWeaponIntent,
    )


class PickupExecutor(ActionExecutor):
    """Executes pickup intents by adding items to actor inventory."""

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
                # Reset ammo for thrown weapons so they can be used again
                if (
                    item.ranged_attack
                    and WeaponProperty.THROWN in item.ranged_attack.properties
                ):
                    item.ranged_attack.current_ammo = item.ranged_attack.max_ammo
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
        gw = intent.controller.gw
        actors_here = [
            a for a in gw.actors if a.x == intent.actor.x and a.y == intent.actor.y
        ]

        for actor in actors_here:
            # Handle item piles (use ContainerStorage API)
            if isinstance(actor, ItemPile):
                if actor.inventory.remove_item(item):
                    # Remove empty piles
                    if actor.is_empty():
                        gw.remove_actor(actor)
                    return True

            # Handle dead characters (use CharacterInventory API)
            elif isinstance(actor, Character):
                inv = actor.inventory
                if inv is not None and inv.try_remove_item(item):
                    return True

        return False


class PickupItemsAtLocationExecutor(ActionExecutor):
    """Executes pickup at location intents by opening the pickup menu."""

    def execute(self, intent: PickupItemsAtLocationIntent) -> GameActionResult | None:  # type: ignore[override]
        """Open the pickup menu UI for items at the actor's location."""
        from catley.game.actors.container import ItemPile
        from catley.view.ui.commands import OpenExistingMenuUICommand
        from catley.view.ui.dual_pane_menu import DualPaneMenu, ExternalInventory

        gw = intent.controller.gw
        actor_x, actor_y = intent.actor.x, intent.actor.y

        # Check if there are items at the actor's location
        items = gw.get_pickable_items_at_location(actor_x, actor_y)

        # Also check for countables in any ItemPile at this location
        # Must use spatial index since get_actor_at_location prioritizes blocking
        # actors (player) over non-blocking (ItemPile)
        has_countables = False
        for actor in gw.actor_spatial_index.get_at_point(actor_x, actor_y):
            if isinstance(actor, ItemPile) and actor.inventory.countables:
                has_countables = True
                break

        if items or has_countables:
            # Open the pickup menu with ground items as the source
            source = ExternalInventory((actor_x, actor_y), "On the ground")
            menu = DualPaneMenu(intent.controller, source=source)
            pickup_cmd = OpenExistingMenuUICommand(intent.controller, menu)
            pickup_cmd.execute()
        return GameActionResult()


class SwitchWeaponExecutor(ActionExecutor):
    """Executes weapon switching intents."""

    def execute(self, intent: SwitchWeaponIntent) -> GameActionResult | None:  # type: ignore[override]
        # Move logic from old SwitchWeaponAction.execute()
        inventory = getattr(intent.actor, "inventory", None)
        if inventory:
            inventory.switch_to_slot(intent.slot)
        return GameActionResult()


class DropItemExecutor(ActionExecutor):
    """Executes drop item intents.

    Removes items from inventory and places them on the ground.
    """

    def execute(self, intent: DropItemIntent) -> GameActionResult | None:  # type: ignore[override]
        """Drop an item from inventory to the ground at actor's location."""
        inventory = getattr(intent.actor, "inventory", None)
        if inventory is None:
            return GameActionResult(succeeded=False)

        item = intent.item
        if not inventory.try_remove_item(item):
            publish_event(MessageEvent("Item not found in inventory.", colors.RED))
            return GameActionResult(succeeded=False)

        # Spawn the item on the ground at actor's location
        intent.controller.gw.spawn_ground_item(item, intent.actor.x, intent.actor.y)
        publish_event(MessageEvent(f"You drop {item.name}.", colors.WHITE))

        return GameActionResult()


class PickupCountableExecutor(ActionExecutor):
    """Executes pickup countable intents."""

    def execute(self, intent: PickupCountableIntent) -> GameActionResult | None:  # type: ignore[override]
        """Pick up countables from the source inventory."""
        actor_inv = getattr(intent.actor, "inventory", None)
        if actor_inv is None:
            return GameActionResult(succeeded=False)

        amount = intent.amount
        if amount < 0:
            # -1 means "take all"
            amount = intent.source.get_countable(intent.countable_type)

        if amount <= 0:
            return GameActionResult(succeeded=False)

        if intent.source.transfer_countable_to(
            actor_inv, intent.countable_type, amount
        ):
            defn = get_countable_definition(intent.countable_type)
            name = defn.plural_name if amount > 1 else defn.name
            publish_event(MessageEvent(f"Picked up {amount} {name}.", colors.WHITE))
            return GameActionResult(succeeded=True)

        return GameActionResult(succeeded=False)


class DropCountableExecutor(ActionExecutor):
    """Executes drop countable intents by placing countables on the ground."""

    def execute(self, intent: DropCountableIntent) -> GameActionResult | None:  # type: ignore[override]
        """Drop countables from inventory to the ground at actor's location."""
        inventory = getattr(intent.actor, "inventory", None)
        if inventory is None:
            return GameActionResult(succeeded=False)

        if not inventory.remove_countable(intent.countable_type, intent.amount):
            return GameActionResult(succeeded=False)

        # Spawn countables on the ground using the item spawner
        intent.controller.gw.item_spawner.spawn_ground_countable(
            (intent.actor.x, intent.actor.y),
            intent.countable_type,
            intent.amount,
        )

        defn = get_countable_definition(intent.countable_type)
        name = defn.plural_name if intent.amount > 1 else defn.name
        publish_event(MessageEvent(f"Dropped {intent.amount} {name}.", colors.WHITE))

        return GameActionResult(succeeded=True)

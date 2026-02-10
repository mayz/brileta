"""Item transfer operations between player inventory, containers, and ground.

Pure game logic extracted from DualPaneMenu. Each method performs the transfer,
publishes appropriate MessageEvents for the message log, and returns a bool
indicating whether the menu should close (True) or stay open (False).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from brileta import colors
from brileta.events import MessageEvent, publish_event
from brileta.game.actors import Character, ItemPile
from brileta.game.countables import CountableType, get_countable_definition
from brileta.game.items.item_core import Item
from brileta.util.coordinates import WorldTilePos

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actors import Actor


@dataclass
class ExternalInventory:
    """Describes position-based source of items in the right pane.

    Used for ground items at a specific world location.
    """

    position: WorldTilePos
    label: str  # e.g., "On the ground"


@dataclass
class ActorInventorySource:
    """Source items from an actor's inventory (container, corpse, etc.).

    This allows the DualPaneMenu to display and transfer items from
    any actor with an inventory, including:
    - Containers (crates, lockers, etc.)
    - Dead characters (corpse looting)
    - Any other actor with an inventory component
    """

    actor: Actor
    label: str  # e.g., "Wooden Crate", "Dead Raider"


# Type alias for external sources
type ExternalSource = ExternalInventory | ActorInventorySource | None


class ItemTransferHandler:
    """Handles item transfer operations between player inventory, containers, and ground.

    Pure game logic - no UI state manipulation (no cursor updates, pane refreshes,
    or layout calculations). Methods return a bool: True to close the menu,
    False to keep it open.
    """

    def __init__(
        self,
        controller: Controller,
        player: Character,
        source: ExternalSource,
    ) -> None:
        self.controller = controller
        self.player = player
        self.source = source

    def get_items_from_source(self) -> list[Item]:
        """Get items from the external source."""
        if self.source is None:
            return []

        if isinstance(self.source, ActorInventorySource):
            inv = self.source.actor.inventory
            if inv is None:
                return []
            return inv.get_items()

        # ExternalInventory - position-based ground items
        world_x, world_y = self.source.position
        return self.controller.gw.get_pickable_items_at_location(world_x, world_y)

    def get_countables_from_source(self) -> dict[CountableType, int]:
        """Get countables from the external source."""
        if self.source is None:
            return {}

        if isinstance(self.source, ActorInventorySource):
            inv = self.source.actor.inventory
            if inv is None:
                return {}
            return inv.countables

        # ExternalInventory - check for item piles at location
        world_x, world_y = self.source.position
        for actor in self.controller.gw.actor_spatial_index.get_at_point(
            world_x, world_y
        ):
            if isinstance(actor, ItemPile):
                return actor.inventory.countables
        return {}

    def equip_item(self, item: Item) -> bool:
        """Equip or unequip any item from player inventory.

        Handles:
        - Armor/outfits: Toggle equip to outfit slot
        - Any other item: Toggle equip to active hand slot

        Any item can be held in a hand slot. Attack actions just won't work
        for non-weapons.

        Returns False to keep menu open.
        """
        from brileta.game.outfit import is_outfit_type

        player = self.player

        # Check if item is an outfit (armor)
        if is_outfit_type(item.item_type):
            # Check if this outfit is already equipped
            equipped = player.inventory.equipped_outfit
            if equipped is not None and equipped[0] is item:
                # Unequip it
                success, message = player.inventory.unequip_outfit()
                color = colors.WHITE if success else colors.YELLOW
                publish_event(MessageEvent(message, color))
            else:
                # Equip it (will auto-unequip current outfit if any)
                success, message = player.inventory.equip_outfit(item)
                color = colors.GREEN if success else colors.YELLOW
                publish_event(MessageEvent(message, color))
        else:
            # Any item can go in a hand slot
            # Check if item is already equipped in a hand slot
            currently_equipped_slot = None
            for i, equipped_item in enumerate(player.inventory.ready_slots):
                if equipped_item == item:
                    currently_equipped_slot = i
                    break

            if currently_equipped_slot is not None:
                # Unequip it
                success, message = player.inventory.unequip_to_inventory(
                    currently_equipped_slot
                )
                color = colors.WHITE if success else colors.YELLOW
                publish_event(MessageEvent(message, color))
            else:
                # Equip to first empty slot, or active slot if none empty
                target_slot = None
                for i, slot_item in enumerate(player.inventory.ready_slots):
                    if slot_item is None:
                        target_slot = i
                        break
                if target_slot is None:
                    target_slot = player.inventory.active_slot

                success, message = player.inventory.equip_from_inventory(
                    item, target_slot
                )
                color = colors.GREEN if success else colors.YELLOW
                publish_event(MessageEvent(message, color))

        return False  # Keep menu open

    def use_consumable(self, item: Item) -> bool:
        """Use a consumable item from player inventory.

        Returns True to close menu after using consumable, False otherwise.
        """
        from brileta.game.actions.recovery import UseConsumableIntent

        if item.consumable_effect:
            intent = UseConsumableIntent(self.controller, self.player, item)
            self.controller.queue_action(intent)
            return True  # Close menu after using consumable

        publish_event(MessageEvent(f"{item.name} cannot be used.", colors.YELLOW))
        return False

    def transfer_to_inventory(self, item: Item) -> bool:
        """Transfer item from source to player inventory.

        Returns False to keep menu open.
        """
        player = self.player

        if not player.inventory.can_add_voluntary_item(item):
            publish_event(MessageEvent("Your inventory is full!", colors.RED))
            return False

        # Remove item from source
        if isinstance(self.source, ActorInventorySource):
            # Remove from actor's inventory directly
            inv = self.source.actor.inventory
            if inv is not None:
                inv.remove_item(item)
        elif self.source is not None:
            # ExternalInventory - position-based ground items
            world_x, world_y = self.source.position
            for actor in self.controller.gw.actor_spatial_index.get_at_point(
                world_x, world_y
            ):
                removed = False

                # Handle item piles (ContainerStorage)
                if isinstance(actor, ItemPile):
                    if actor.inventory.remove_item(item):
                        removed = True
                        if actor.is_empty():
                            self.controller.gw.remove_actor(actor)

                # Handle characters (dead bodies with CharacterInventory)
                elif isinstance(actor, Character):
                    inv = actor.inventory
                    if inv is not None:
                        if item in inv:
                            inv.remove_from_inventory(item)
                            removed = True

                        for i, equipped_item in enumerate(inv.ready_slots):
                            if equipped_item == item:
                                inv.unequip_slot(i)
                                removed = True
                                break

                if removed:
                    break

        # Add to player inventory
        _success, _message, dropped_items = player.inventory.add_to_inventory(item)

        # Reset ammo for thrown weapons so they can be used again
        from brileta.game.items.properties import WeaponProperty

        if (
            item.ranged_attack
            and WeaponProperty.THROWN in item.ranged_attack.properties
        ):
            item.ranged_attack.current_ammo = item.ranged_attack.max_ammo

        publish_event(MessageEvent(f"You pick up {item.name}.", colors.WHITE))

        # Handle overflow drops
        for dropped in dropped_items:
            self.controller.gw.spawn_ground_item(dropped, player.x, player.y)

        return False  # Keep menu open

    def transfer_all_to_inventory(self, item: Item) -> bool:
        """Transfer all items of the same ItemType from source to player inventory.

        Returns False to keep menu open.
        """
        player = self.player

        # Find all items of the same type from source
        source_items = self.get_items_from_source()
        items_to_transfer = [i for i in source_items if i.item_type is item.item_type]

        if not items_to_transfer:
            return False

        # Check capacity
        if not player.inventory.can_add_voluntary_item(items_to_transfer[0]):
            publish_event(MessageEvent("Your inventory is full!", colors.RED))
            return False

        transferred_count = 0
        for transfer_item in items_to_transfer:
            if not player.inventory.can_add_voluntary_item(transfer_item):
                break  # Stop when full

            # Remove from source
            if isinstance(self.source, ActorInventorySource):
                inv = self.source.actor.inventory
                if inv is not None:
                    inv.remove_item(transfer_item)
            elif self.source is not None:
                world_x, world_y = self.source.position
                for actor in self.controller.gw.actor_spatial_index.get_at_point(
                    world_x, world_y
                ):
                    if isinstance(actor, ItemPile):
                        removed = actor.inventory.remove_item(transfer_item)
                        if removed and actor.is_empty():
                            self.controller.gw.remove_actor(actor)
                        break

            player.inventory.add_to_inventory(transfer_item)
            transferred_count += 1

        if transferred_count == 1:
            publish_event(MessageEvent(f"You pick up {item.name}.", colors.WHITE))
        else:
            publish_event(
                MessageEvent(
                    f"You pick up {transferred_count} {item.name}.", colors.WHITE
                )
            )

        return False  # Keep menu open

    def transfer_countable_to_inventory(self, countable_type: CountableType) -> bool:
        """Transfer all countables of a type from source to player inventory.

        Returns False to keep menu open.
        """
        player = self.player

        # Get quantity from source
        if isinstance(self.source, ActorInventorySource):
            inv = self.source.actor.inventory
            if inv is None:
                return False
            quantity = inv.get_countable(countable_type)
            if quantity <= 0:
                return False
            inv.remove_countable(countable_type, quantity)
        elif self.source is not None:
            # ExternalInventory - position-based ground items
            world_x, world_y = self.source.position
            quantity = 0
            for actor in self.controller.gw.actor_spatial_index.get_at_point(
                world_x, world_y
            ):
                if isinstance(actor, ItemPile):
                    quantity = actor.inventory.get_countable(countable_type)
                    if quantity > 0:
                        actor.inventory.remove_countable(countable_type, quantity)
                        if actor.is_empty():
                            self.controller.gw.remove_actor(actor)
                    break
            if quantity <= 0:
                return False
        else:
            return False

        # Add to player inventory
        player.inventory.add_countable(countable_type, quantity)

        defn = get_countable_definition(countable_type)
        name = defn.plural_name if quantity > 1 else defn.name
        publish_event(MessageEvent(f"You pick up {quantity} {name}.", colors.WHITE))

        return False  # Keep menu open

    def transfer_to_container(self, item: Item) -> bool:
        """Transfer item from player inventory to source container.

        Returns False to keep menu open.
        """
        if self.source is None:
            return False

        player = self.player

        # Check if container can accept the item (for ActorInventorySource)
        if isinstance(self.source, ActorInventorySource):
            inv = self.source.actor.inventory
            if inv is not None and len(inv) >= inv.capacity:
                publish_event(MessageEvent("Container is full!", colors.RED))
                return False

        # Unequip if equipped in ready slots
        for i, equipped_item in enumerate(player.inventory.ready_slots):
            if equipped_item == item:
                player.inventory.unequip_slot(i)
                break

        # Unequip if equipped as outfit (armor)
        if player.inventory.equipped_outfit is not None:
            outfit_item, _ = player.inventory.equipped_outfit
            if outfit_item == item:
                player.inventory.unequip_outfit()

        # Remove from inventory
        player.inventory.remove_from_inventory(item)

        # Add to container or spawn on ground
        if isinstance(self.source, ActorInventorySource):
            inv = self.source.actor.inventory
            if inv is not None:
                inv.add_item(item)
            publish_event(
                MessageEvent(
                    f"You put {item.name} in {self.source.label}.", colors.WHITE
                )
            )
        else:
            # ExternalInventory - spawn as ground item at source location
            world_x, world_y = self.source.position
            self.controller.gw.spawn_ground_item(item, world_x, world_y)
            publish_event(MessageEvent(f"You put down {item.name}.", colors.WHITE))

        return False  # Keep menu open

    def transfer_all_to_container(self, item: Item) -> bool:
        """Transfer all items of the same ItemType from inventory to container/ground.

        Returns False to keep menu open.
        """
        if self.source is None:
            return False

        player = self.player

        # Find all items of the same type
        items_to_transfer = [
            i for i in player.inventory.get_items() if i.item_type is item.item_type
        ]

        if not items_to_transfer:
            return False

        # For ActorInventorySource, check capacity
        if isinstance(self.source, ActorInventorySource):
            inv = self.source.actor.inventory
            if inv is not None:
                available = inv.capacity - len(inv)
                if available < len(items_to_transfer):
                    publish_event(MessageEvent("Container is full!", colors.RED))
                    return False

        # Unequip and transfer each item
        for transfer_item in items_to_transfer:
            for i, equipped_item in enumerate(player.inventory.ready_slots):
                if equipped_item == transfer_item:
                    player.inventory.unequip_slot(i)
                    break
            if player.inventory.equipped_outfit is not None:
                outfit_item, _ = player.inventory.equipped_outfit
                if outfit_item == transfer_item:
                    player.inventory.unequip_outfit()

            player.inventory.remove_from_inventory(transfer_item)

            if isinstance(self.source, ActorInventorySource):
                inv = self.source.actor.inventory
                if inv is not None:
                    inv.add_item(transfer_item)
            else:
                world_x, world_y = self.source.position
                self.controller.gw.spawn_ground_item(transfer_item, world_x, world_y)

        if len(items_to_transfer) == 1:
            publish_event(
                MessageEvent(
                    f"You put {items_to_transfer[0].name} in {self.source.label}.",
                    colors.WHITE,
                )
            )
        else:
            count = len(items_to_transfer)
            msg = f"You put {count} {item.name} in {self.source.label}."
            publish_event(MessageEvent(msg, colors.WHITE))

        return False  # Keep menu open

    def transfer_countable_to_ground(self, countable_type: CountableType) -> bool:
        """Transfer all countables of a type from inventory to ground/container.

        Returns False to keep menu open.
        """
        if self.source is None:
            return False

        player = self.player
        quantity = player.inventory.get_countable(countable_type)

        if quantity <= 0:
            return False

        # Remove from player inventory
        player.inventory.remove_countable(countable_type, quantity)

        # Add to container or spawn on ground
        if isinstance(self.source, ActorInventorySource):
            inv = self.source.actor.inventory
            if inv is not None:
                inv.add_countable(countable_type, quantity)
            defn = get_countable_definition(countable_type)
            name = defn.plural_name if quantity > 1 else defn.name
            publish_event(
                MessageEvent(
                    f"You put {quantity} {name} in {self.source.label}.", colors.WHITE
                )
            )
        else:
            # ExternalInventory - spawn on ground at source location
            world_x, world_y = self.source.position
            self.controller.gw.item_spawner.spawn_ground_countable(
                (world_x, world_y), countable_type, quantity
            )
            defn = get_countable_definition(countable_type)
            name = defn.plural_name if quantity > 1 else defn.name
            publish_event(
                MessageEvent(f"You put down {quantity} {name}.", colors.WHITE)
            )

        return False  # Keep menu open

    def drop_item(self, item: Item) -> bool:
        """Drop an item from inventory to the ground at player location.

        Returns False to keep menu open.
        """
        player = self.player

        # Unequip if equipped in a hand slot
        for i, equipped_item in enumerate(player.inventory.ready_slots):
            if equipped_item == item:
                player.inventory.unequip_slot(i)
                break

        # Unequip if equipped as outfit
        if player.inventory.equipped_outfit is not None:
            outfit_item, _ = player.inventory.equipped_outfit
            if outfit_item == item:
                player.inventory.unequip_outfit()

        # Remove from inventory
        player.inventory.remove_from_inventory(item)

        # Spawn on ground at player position
        self.controller.gw.spawn_ground_item(item, player.x, player.y)
        publish_event(MessageEvent(f"You drop {item.name}.", colors.WHITE))

        return False  # Keep menu open

    def drop_all_of_type(self, item: Item) -> bool:
        """Drop all items of the same ItemType from inventory.

        For Tiny items, finds and drops all with the same item_type.
        For other sizes, just drops the single item.
        Returns False to keep menu open.
        """
        player = self.player

        # Find all items of the same type
        items_to_drop = [
            i for i in player.inventory.get_items() if i.item_type is item.item_type
        ]

        if not items_to_drop:
            return False

        # Unequip any that are equipped
        for drop_item in items_to_drop:
            for i, equipped_item in enumerate(player.inventory.ready_slots):
                if equipped_item == drop_item:
                    player.inventory.unequip_slot(i)
                    break
            if player.inventory.equipped_outfit is not None:
                outfit_item, _ = player.inventory.equipped_outfit
                if outfit_item == drop_item:
                    player.inventory.unequip_outfit()

        # Remove all from inventory and spawn on ground
        for drop_item in items_to_drop:
            player.inventory.remove_from_inventory(drop_item)
            self.controller.gw.spawn_ground_item(drop_item, player.x, player.y)

        if len(items_to_drop) == 1:
            publish_event(
                MessageEvent(f"You drop {items_to_drop[0].name}.", colors.WHITE)
            )
        else:
            publish_event(
                MessageEvent(
                    f"You drop {len(items_to_drop)} {item.name}.", colors.WHITE
                )
            )

        return False  # Keep menu open

    def drop_countable(self, countable_type: CountableType) -> bool:
        """Drop all countables of a type from inventory to the ground.

        Returns False to keep menu open.
        """
        player = self.player
        quantity = player.inventory.get_countable(countable_type)

        if quantity <= 0:
            return False

        # Remove from inventory
        player.inventory.remove_countable(countable_type, quantity)

        # Spawn on ground at player position
        self.controller.gw.item_spawner.spawn_ground_countable(
            (player.x, player.y), countable_type, quantity
        )

        defn = get_countable_definition(countable_type)
        name = defn.plural_name if quantity > 1 else defn.name
        publish_event(MessageEvent(f"You drop {quantity} {name}.", colors.WHITE))

        return False  # Keep menu open

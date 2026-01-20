"""Dual-pane inventory/loot interface.

A unified Fallout-style inventory screen that shows player inventory on the left,
container/ground items on the right (when looting), and item details below.
Replaces the separate InventoryMenu and PickupMenu.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final

import tcod.event

from catley import colors
from catley.backends.pillow.canvas import PillowImageCanvas
from catley.events import MessageEvent, publish_event
from catley.game.actors import Actor, Character, Condition
from catley.game.enums import ItemCategory, ItemSize
from catley.game.items.item_core import Item
from catley.util.coordinates import WorldTilePos
from catley.view.ui.overlays import Menu, MenuOption

if TYPE_CHECKING:
    from catley.controller import Controller


def _get_category_prefix(item: Item) -> tuple[str, colors.Color | None]:
    """Return the category prefix symbol and color for an item.

    Returns a tuple of (prefix_string, color) where prefix_string is
    a colored dot indicator. Returns ("", None) for MISC category items.
    """
    category_info: dict[ItemCategory, tuple[str, colors.Color]] = {
        ItemCategory.WEAPON: ("\u25cf ", colors.CATEGORY_WEAPON),
        ItemCategory.ARMOR: ("\u25cf ", colors.CATEGORY_ARMOR),
        ItemCategory.CONSUMABLE: ("\u25cf ", colors.CATEGORY_CONSUMABLE),
        ItemCategory.JUNK: ("\u25cf ", colors.CATEGORY_JUNK),
        ItemCategory.MUNITIONS: ("\u25cf ", colors.CATEGORY_MUNITIONS),
    }

    return category_info.get(item.category, ("", None))


def _get_category_name(category: ItemCategory) -> str:
    """Return the display name for an item category."""
    names: dict[ItemCategory, str] = {
        ItemCategory.WEAPON: "Weapon",
        ItemCategory.ARMOR: "Armor",
        ItemCategory.CONSUMABLE: "Consumable",
        ItemCategory.JUNK: "Junk",
        ItemCategory.MUNITIONS: "Munition",
        ItemCategory.MISC: "Misc",
    }
    return names.get(category, "")


class _MenuKeys:
    """KeySym constants for menu navigation keys.

    Using a class with class attributes ensures pattern matching treats
    these as value patterns (comparison) rather than capture patterns.
    """

    KEY_D: Final = tcod.event.KeySym(ord("d"))
    KEY_I: Final = tcod.event.KeySym(ord("i"))
    KEY_J: Final = tcod.event.KeySym(ord("j"))
    KEY_K: Final = tcod.event.KeySym(ord("k"))
    KEY_P: Final = tcod.event.KeySym(ord("p"))
    KEY_Q: Final = tcod.event.KeySym(ord("q"))
    KEY_T: Final = tcod.event.KeySym(ord("t"))
    KEY_U: Final = tcod.event.KeySym(ord("u"))


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
ExternalSource = ExternalInventory | ActorInventorySource | None


class PaneId(Enum):
    """Identifies which pane is active."""

    LEFT = "left"
    RIGHT = "right"


class DualPaneMenu(Menu):
    """Unified inventory/loot menu with dual panes and detail panel.

    When source is None, operates in inventory-only mode (right pane hidden).
    When source is an ExternalInventory, shows items at that location for looting.
    """

    # Layout constants (in tiles)
    TOTAL_WIDTH = 80
    SINGLE_PANE_WIDTH = 40  # inventory-only mode
    TOTAL_HEIGHT = 30
    LEFT_PANE_WIDTH = 35
    DIVIDER_WIDTH = 7
    RIGHT_PANE_WIDTH = 35
    DETAIL_HEIGHT = 5
    HINT_HEIGHT = 2  # Extra row for whitespace above keycaps
    HEADER_HEIGHT = 1
    # Calculated: item list height = TOTAL_HEIGHT - HEADER - DETAIL - HINT - borders
    ITEM_LIST_HEIGHT = TOTAL_HEIGHT - HEADER_HEIGHT - DETAIL_HEIGHT - HINT_HEIGHT - 3

    def __init__(
        self,
        controller: Controller,
        source: ExternalSource = None,
    ) -> None:
        """Initialize the dual-pane menu.

        Args:
            controller: Game controller.
            source: External source to interact with - can be:
                - ExternalInventory: Position-based ground items
                - ActorInventorySource: Items from an actor (container/corpse)
                - None: Inventory-only mode (no right pane)
        """
        title = "Inventory" if source is None else "Loot"
        super().__init__(
            title, controller, width=self.TOTAL_WIDTH, max_height=self.TOTAL_HEIGHT
        )

        # Source for right pane (None = inventory-only mode)
        self.source = source

        # Pane state
        self.active_pane: PaneId = PaneId.LEFT
        self.left_cursor: int = 0
        self.right_cursor: int = 0
        self.left_options: list[MenuOption] = []
        self.right_options: list[MenuOption] = []

        # Detail panel
        self.detail_item: Item | Condition | None = None

        # Get inventory reference
        self.inventory = controller.gw.player.inventory

    def populate_options(self) -> None:
        """Populate both panes from their respective sources."""
        self._populate_left_pane()
        self._populate_right_pane()

        # Set initial detail item from active pane
        self._update_detail_from_cursor()

    def _populate_left_pane(self) -> None:
        """Populate left pane with player inventory items."""
        self.left_options.clear()

        # Collect equipped items separately from stored items
        equipped_items: list[tuple[Item | Condition, str]] = []

        # Add equipped weapons with their slot index
        for slot_index, equipped_item in enumerate(self.inventory.attack_slots):
            if equipped_item:
                equipped_items.append((equipped_item, f"slot:{slot_index}"))

        # Add equipped outfit
        if self.inventory.equipped_outfit is not None:
            outfit_item, _ = self.inventory.equipped_outfit
            equipped_items.append((outfit_item, "outfit"))

        # Add stored items as tuples with "stored" status
        stored_items: list[tuple[Item | Condition, str]] = [
            (item, "stored") for item in self.inventory._stored_items
        ]

        # Stored items first, then equipped items at the bottom
        all_items_to_display = stored_items + equipped_items

        if not all_items_to_display:
            self.left_options.append(
                MenuOption(
                    key=None,
                    text="(inventory empty)",
                    enabled=False,
                    color=colors.GREY,
                    is_primary_action=False,
                )
            )
            return

        # Track whether we need to insert a separator
        has_equipped = len(equipped_items) > 0
        has_stored = len(stored_items) > 0
        separator_inserted = False

        for entity, item_status in all_items_to_display:
            # Insert separator between stored and equipped items (empty line)
            if (
                has_equipped
                and has_stored
                and not separator_inserted
                and item_status != "stored"
            ):
                self.left_options.append(
                    MenuOption(
                        key=None,
                        text="",
                        enabled=False,
                        color=colors.GREY,
                        is_primary_action=False,
                    )
                )
                separator_inserted = True
            if isinstance(entity, Item):
                item = entity

                # Build prefix segments: slot indicator (if equipped) + category
                prefix_segments: list[tuple[str, colors.Color]] = []

                # Parse equipped slot number for prefix
                if item_status.startswith("slot:"):
                    slot_num = int(item_status.split(":")[1]) + 1
                    prefix_segments.append((f"[{slot_num}]  ", colors.YELLOW))
                elif item_status == "outfit":
                    # Show [A] for equipped armor/outfit
                    prefix_segments.append(("[A]  ", colors.CATEGORY_ARMOR))
                else:
                    # Add alignment spaces to match equipped items
                    prefix_segments.append(("     ", colors.WHITE))

                # Add category prefix
                cat_prefix, cat_color = _get_category_prefix(item)
                if cat_prefix and cat_color:
                    prefix_segments.append((cat_prefix, cat_color))

                self.left_options.append(
                    MenuOption(
                        key=None,
                        text=item.name,
                        action=functools.partial(self._equip_item, item),
                        enabled=True,
                        color=colors.WHITE,
                        data=item,
                        prefix_segments=prefix_segments,
                    )
                )

            elif isinstance(entity, Condition):
                condition = entity
                # Conditions show red block in icon position to indicate slot usage
                self.left_options.append(
                    MenuOption(
                        key=None,
                        text=condition.name,
                        action=None,
                        enabled=False,
                        color=condition.display_color,
                        force_color=True,
                        data=condition,
                        prefix_segments=[
                            ("     ", colors.WHITE),  # Alignment with equipped slots
                            ("\u2588 ", colors.RED),  # Red block as slot indicator
                        ],
                    )
                )

    def _populate_right_pane(self) -> None:
        """Populate right pane from source (container/ground/None)."""
        self.right_options.clear()

        if self.source is None:
            # Inventory-only mode - no right pane content
            return

        items = self._get_items_from_source()

        if not items:
            self.right_options.append(
                MenuOption(
                    key=None,
                    text="(no items)",
                    enabled=False,
                    color=colors.GREY,
                    is_primary_action=False,
                )
            )
            return

        for item in items:
            # Build prefix segments with category prefix
            prefix_segments: list[tuple[str, colors.Color]] = []
            cat_prefix, cat_color = _get_category_prefix(item)
            if cat_prefix and cat_color:
                prefix_segments.append((cat_prefix, cat_color))

            self.right_options.append(
                MenuOption(
                    key=None,
                    text=item.name,
                    action=functools.partial(self._transfer_to_inventory, item),
                    enabled=True,
                    color=colors.WHITE,
                    data=item,
                    prefix_segments=prefix_segments if prefix_segments else None,
                )
            )

    def _get_items_from_source(self) -> list[Item]:
        """Get items from the right pane source."""
        if self.source is None:
            return []

        if isinstance(self.source, ActorInventorySource):
            # Get items from actor's inventory
            inv = self.source.actor.inventory
            if inv is None:
                return []
            return inv.get_items()

        # ExternalInventory - position-based ground items
        world_x, world_y = self.source.position
        return self.controller.gw.get_pickable_items_at_location(world_x, world_y)

    def _update_detail_from_cursor(self) -> None:
        """Update detail_item based on current cursor position."""
        if self.active_pane == PaneId.LEFT:
            options = self.left_options
            cursor = self.left_cursor
        else:
            options = self.right_options
            cursor = self.right_cursor

        if 0 <= cursor < len(options) and options[cursor].data is not None:
            self.detail_item = options[cursor].data
        else:
            self.detail_item = None

    def _find_item_index(self, item: Item, pane: PaneId) -> int | None:
        """Find the index of an item in the specified pane's options."""
        options = self.left_options if pane == PaneId.LEFT else self.right_options
        for i, opt in enumerate(options):
            if opt.data is item:
                return i
        return None

    def _ensure_valid_cursor(self, pane: PaneId) -> None:
        """Ensure cursor in specified pane is on an enabled option.

        If the current position is valid and enabled, keeps it.
        Otherwise finds the nearest enabled option (preferring forward, then backward).
        Falls back to position 0 if no enabled options exist.
        """
        if pane == PaneId.LEFT:
            options = self.left_options
            cursor = self.left_cursor
        else:
            options = self.right_options
            cursor = self.right_cursor

        # If current position is valid and enabled, keep it
        if 0 <= cursor < len(options) and options[cursor].enabled:
            return

        # Find nearest enabled option (prefer forward, then backward)
        for i in range(cursor, len(options)):
            if options[i].enabled:
                if pane == PaneId.LEFT:
                    self.left_cursor = i
                else:
                    self.right_cursor = i
                return
        for i in range(cursor - 1, -1, -1):
            if options[i].enabled:
                if pane == PaneId.LEFT:
                    self.left_cursor = i
                else:
                    self.right_cursor = i
                return

        # No enabled options - set to 0
        if pane == PaneId.LEFT:
            self.left_cursor = 0
        else:
            self.right_cursor = 0

    def _equip_item(self, item: Item) -> bool:
        """Equip or unequip any item from player inventory.

        Handles:
        - Armor/outfits: Toggle equip to outfit slot
        - Any other item: Toggle equip to active hand slot

        Any item can be held in a hand slot. Attack actions just won't work
        for non-weapons.

        Returns False to keep menu open.
        """
        from catley.game.outfit import is_outfit_type

        player = self.controller.gw.player

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
            for i, equipped_item in enumerate(player.inventory.attack_slots):
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
                # Equip to active hand slot
                active_slot = player.inventory.active_weapon_slot
                success, message = player.inventory.equip_from_inventory(
                    item, active_slot
                )
                color = colors.GREEN if success else colors.YELLOW
                publish_event(MessageEvent(message, color))

        # Refresh panes after equip/unequip
        self._populate_left_pane()

        # Follow the item to its new position in the list
        new_index = self._find_item_index(item, PaneId.LEFT)
        if new_index is not None:
            self.left_cursor = new_index
        self._ensure_valid_cursor(PaneId.LEFT)

        self._update_detail_from_cursor()
        return False  # Keep menu open

    def _use_consumable(self, item: Item) -> bool:
        """Use a consumable item from player inventory.

        Returns True to close menu after using consumable, False otherwise.
        """
        from catley.game.actions.recovery import UseConsumableIntent

        player = self.controller.gw.player

        if item.consumable_effect:
            intent = UseConsumableIntent(self.controller, player, item)
            self.controller.queue_action(intent)
            return True  # Close menu after using consumable

        publish_event(MessageEvent(f"{item.name} cannot be used.", colors.YELLOW))
        return False

    def _transfer_to_inventory(self, item: Item) -> bool:
        """Transfer item from source to player inventory.

        Returns False to keep menu open.
        """
        player = self.controller.gw.player

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
            actors_here = [
                a
                for a in self.controller.gw.actors
                if a.x == world_x and a.y == world_y
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
                    # Remove empty temporary ground piles (not permanent furniture)
                    # Permanent containers like bookcases have blocks_movement=True
                    if (
                        not actor.blocks_movement
                        and not isinstance(actor, Character)
                        and len(inv) == 0
                        and all(s is None for s in inv.attack_slots)
                    ):
                        self.controller.gw.remove_actor(actor)
                    break

        # Add to player inventory
        _success, _message, dropped_items = player.inventory.add_to_inventory(item)

        # Reset ammo for thrown weapons so they can be used again
        from catley.game.items.properties import WeaponProperty

        if (
            item.ranged_attack
            and WeaponProperty.THROWN in item.ranged_attack.properties
        ):
            item.ranged_attack.current_ammo = item.ranged_attack.max_ammo

        publish_event(MessageEvent(f"You pick up {item.name}.", colors.WHITE))

        # Handle overflow drops
        for dropped in dropped_items:
            self.controller.gw.spawn_ground_item(dropped, player.x, player.y)

        # Refresh both panes
        self._populate_left_pane()
        self._populate_right_pane()

        # Check if right pane still has selectable items
        has_selectable_right = any(opt.enabled for opt in self.right_options)

        if has_selectable_right:
            # Stay in right pane, select next item
            self._ensure_valid_cursor(PaneId.RIGHT)
        else:
            # Right pane empty - switch to left pane and select transferred item
            self.active_pane = PaneId.LEFT
            new_index = self._find_item_index(item, PaneId.LEFT)
            if new_index is not None:
                self.left_cursor = new_index
            self._ensure_valid_cursor(PaneId.LEFT)

        self._update_detail_from_cursor()
        return False  # Keep menu open

    def _transfer_to_container(self, item: Item) -> bool:
        """Transfer item from player inventory to source container.

        Returns False to keep menu open.
        """
        if self.source is None:
            return False

        player = self.controller.gw.player

        # Check if container can accept the item (for ActorInventorySource)
        if isinstance(self.source, ActorInventorySource):
            inv = self.source.actor.inventory
            if inv is not None and len(inv) >= inv.capacity:
                publish_event(MessageEvent("Container is full!", colors.RED))
                return False

        # Unequip if equipped in attack slots
        for i, equipped_item in enumerate(player.inventory.attack_slots):
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

        # Refresh both panes
        self._populate_left_pane()
        self._populate_right_pane()

        # Check if left pane still has selectable items
        has_selectable_left = any(opt.enabled for opt in self.left_options)

        if has_selectable_left:
            # Stay in left pane, select next item
            self._ensure_valid_cursor(PaneId.LEFT)
        else:
            # Left pane empty - switch to right pane and select transferred item
            self.active_pane = PaneId.RIGHT
            new_index = self._find_item_index(item, PaneId.RIGHT)
            if new_index is not None:
                self.right_cursor = new_index
            self._ensure_valid_cursor(PaneId.RIGHT)

        self._update_detail_from_cursor()
        return False  # Keep menu open

    def _drop_item(self, item: Item) -> bool:
        """Drop an item from inventory to the ground at player location.

        Drops the item immediately and transitions to dual-pane view showing
        ground items. Returns False to keep menu open.
        """
        player = self.controller.gw.player

        # Unequip if equipped in a hand slot
        for i, equipped_item in enumerate(player.inventory.attack_slots):
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

        # Transition to dual-pane mode showing ground items
        self.source = ExternalInventory(
            position=(player.x, player.y),
            label="On the ground",
        )

        # Recalculate dimensions for dual-pane width
        self._calculate_dimensions()

        # Refresh both panes
        self._populate_left_pane()
        self._populate_right_pane()

        # Check if left pane still has selectable items
        has_selectable_left = any(opt.enabled for opt in self.left_options)

        if has_selectable_left:
            # Stay in left pane, select next item
            self._ensure_valid_cursor(PaneId.LEFT)
        else:
            # Left pane empty - switch to right pane and select dropped item
            self.active_pane = PaneId.RIGHT
            new_index = self._find_item_index(item, PaneId.RIGHT)
            if new_index is not None:
                self.right_cursor = new_index
            self._ensure_valid_cursor(PaneId.RIGHT)

        self._update_detail_from_cursor()
        return False  # Keep menu open

    def _handle_drop_or_transfer(self) -> None:
        """Handle D/T/P key press: drop (single-pane) or transfer (dual-pane).

        In single-pane mode (inventory only):
            - Drops the item at player's feet and transitions to dual-pane

        In dual-pane mode:
            - Left pane: transfers item to container/ground
            - Right pane: transfers item to player inventory
        """
        if self.source is None:
            # Single-pane mode: drop item
            if self.active_pane == PaneId.LEFT and 0 <= self.left_cursor < len(
                self.left_options
            ):
                option = self.left_options[self.left_cursor]
                if option.enabled and isinstance(option.data, Item):
                    self._drop_item(option.data)
        else:
            # Dual-pane mode: transfer between panes
            if self.active_pane == PaneId.LEFT:
                # Transfer from inventory to container/ground
                if 0 <= self.left_cursor < len(self.left_options):
                    option = self.left_options[self.left_cursor]
                    if option.enabled and isinstance(option.data, Item):
                        self._transfer_to_container(option.data)
            else:
                # Transfer from container/ground to inventory
                if 0 <= self.right_cursor < len(self.right_options):
                    option = self.right_options[self.right_cursor]
                    if option.enabled and isinstance(option.data, Item):
                        self._transfer_to_inventory(option.data)

    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle dual-pane navigation and actions.

        Key mappings:
        - Enter: Equip/unequip (left pane only, both modes)
        - U: Use consumable (left pane only, both modes)
        - D/T/P: Drop (single-pane) or Transfer (dual-pane) - all three keys
                 do the same action
        - Tab: Switch panes (dual-pane only)
        - Arrows/J/K: Navigate
        - Esc/I/Q/Space: Close menu
        """
        if not self.is_active:
            return False

        match event:
            # Tab switches panes (only if right pane exists)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.TAB):
                if self.source is not None and self.right_options:
                    # Only switch if right pane has content
                    has_selectable_right = any(
                        opt.enabled for opt in self.right_options
                    )
                    if has_selectable_right or self.active_pane == PaneId.RIGHT:
                        self.active_pane = (
                            PaneId.RIGHT
                            if self.active_pane == PaneId.LEFT
                            else PaneId.LEFT
                        )
                        self._update_detail_from_cursor()
                return True

            # Arrow keys navigate within active pane
            case tcod.event.KeyDown(sym=tcod.event.KeySym.UP):
                self._move_cursor(-1)
                return True
            case tcod.event.KeyDown(sym=tcod.event.KeySym.DOWN):
                self._move_cursor(1)
                return True

            # Vim keys for navigation (j=down, k=up)
            case tcod.event.KeyDown(sym=_MenuKeys.KEY_J):
                self._move_cursor(1)
                return True
            case tcod.event.KeyDown(sym=_MenuKeys.KEY_K):
                self._move_cursor(-1)
                return True

            # D/T/P keys: Drop (single-pane) or Transfer (dual-pane)
            # All three keys do the same action
            case tcod.event.KeyDown(
                sym=_MenuKeys.KEY_D | _MenuKeys.KEY_T | _MenuKeys.KEY_P
            ):
                self._handle_drop_or_transfer()
                return True

            # U key: Use consumable (left pane only)
            case tcod.event.KeyDown(sym=_MenuKeys.KEY_U):
                if self.active_pane == PaneId.LEFT and 0 <= self.left_cursor < len(
                    self.left_options
                ):
                    option = self.left_options[self.left_cursor]
                    if (
                        option.enabled
                        and isinstance(option.data, Item)
                        and self._use_consumable(option.data)
                    ):
                        self.hide()
                return True

            # Enter: Equip/unequip (left pane only)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.RETURN):
                if self.active_pane == PaneId.LEFT and 0 <= self.left_cursor < len(
                    self.left_options
                ):
                    option = self.left_options[self.left_cursor]
                    if option.enabled and isinstance(option.data, Item):
                        self._equip_item(option.data)
                # Enter does nothing on right pane
                return True

            # Escape closes menu
            case tcod.event.KeyDown(sym=tcod.event.KeySym.ESCAPE):
                self.hide()
                return True

            # I key closes menu (toggle behavior)
            case tcod.event.KeyDown(sym=_MenuKeys.KEY_I):
                self.hide()
                return True

            # Q key closes menu (instead of quitting)
            case tcod.event.KeyDown(sym=_MenuKeys.KEY_Q):
                self.hide()
                return True

            # Space closes menu
            case tcod.event.KeyDown(sym=tcod.event.KeySym.SPACE):
                self.hide()
                return True

            # Mouse motion updates hover state
            case tcod.event.MouseMotion():
                return self._handle_mouse_motion(event)

            # Mouse click
            case tcod.event.MouseButtonDown(button=tcod.event.MouseButton.LEFT):
                return self._handle_mouse_click(event)

            case tcod.event.KeyUp():
                return False  # Let KeyUp events pass through to modes

        return True  # Consume all other input while menu is active

    def _move_cursor(self, delta: int) -> None:
        """Move cursor within active pane."""
        if self.active_pane == PaneId.LEFT:
            options = self.left_options
            # Find next enabled option
            new_cursor = self.left_cursor + delta
            while 0 <= new_cursor < len(options):
                if options[new_cursor].enabled:
                    self.left_cursor = new_cursor
                    break
                new_cursor += delta
        else:
            options = self.right_options
            new_cursor = self.right_cursor + delta
            while 0 <= new_cursor < len(options):
                if options[new_cursor].enabled:
                    self.right_cursor = new_cursor
                    break
                new_cursor += delta

        self._update_detail_from_cursor()

    def _activate_current_item(self) -> bool:
        """Activate the currently selected item (used for mouse clicks).

        Left pane: equip the item
        Right pane: transfer to inventory (pick up)

        Returns True if menu should close.
        """
        if self.active_pane == PaneId.LEFT:
            if 0 <= self.left_cursor < len(self.left_options):
                option = self.left_options[self.left_cursor]
                if option.enabled and option.action:
                    # Left pane action is _equip_item
                    result = option.action()
                    return result is True
        else:
            if 0 <= self.right_cursor < len(self.right_options):
                option = self.right_options[self.right_cursor]
                if option.enabled and option.action:
                    # Right pane action is _transfer_to_inventory
                    result = option.action()
                    return result is True

        return False

    def _handle_mouse_motion(self, event: tcod.event.MouseMotion) -> bool:
        """Handle mouse motion to update hover state."""
        mouse_px_x, mouse_px_y = event.position

        # Convert to menu-relative coordinates
        tile_w, tile_h = self.tile_dimensions
        menu_px_x = self.x_tiles * tile_w
        menu_px_y = self.y_tiles * tile_h
        rel_px_x = mouse_px_x - menu_px_x
        rel_px_y = mouse_px_y - menu_px_y

        char_w = self._char_width
        line_h = self._line_height
        if char_w == 0 or line_h == 0:
            return True

        # Convert to character/line position
        rel_char_x = int(rel_px_x // char_w)
        rel_line_y = int(rel_px_y // line_h)

        # Determine which pane the mouse is over
        # Left pane: chars 1 to LEFT_PANE_WIDTH
        # Right pane: chars LEFT_PANE_WIDTH + DIVIDER_WIDTH to end
        # Items start at line 2: border (0) + header (1) + items start (2)
        item_start_y = self.HEADER_HEIGHT + 2

        if 1 <= rel_char_x < self.LEFT_PANE_WIDTH:
            # Over left pane
            option_line = rel_line_y - item_start_y
            if 0 <= option_line < len(self.left_options):
                self.active_pane = PaneId.LEFT
                self.left_cursor = option_line
                self._update_detail_from_cursor()
        elif (
            self.source is not None
            and self.LEFT_PANE_WIDTH + self.DIVIDER_WIDTH
            <= rel_char_x
            < self.TOTAL_WIDTH - 1
        ):
            # Over right pane
            option_line = rel_line_y - item_start_y
            if 0 <= option_line < len(self.right_options):
                self.active_pane = PaneId.RIGHT
                self.right_cursor = option_line
                self._update_detail_from_cursor()

        return True

    def _handle_mouse_click(self, event: tcod.event.MouseButtonDown) -> bool:
        """Handle mouse click to select/transfer items."""
        mouse_px_x, mouse_px_y = event.position

        # Check if click is outside menu using tile-based positioning
        tile_w, tile_h = self.tile_dimensions
        menu_px_x = self.x_tiles * tile_w
        menu_px_y = self.y_tiles * tile_h
        rel_px_x = mouse_px_x - menu_px_x
        rel_px_y = mouse_px_y - menu_px_y

        if not (0 <= rel_px_x < self.pixel_width and 0 <= rel_px_y < self.pixel_height):
            self.hide()
            return True

        # Activate whatever is under cursor (motion handler already updated cursor)
        result = self._activate_current_item()
        if result:
            self.hide()
        return True

    def _generate_item_detail(self, entity: Item | Condition) -> list[str]:
        """Generate detail lines for the selected item/condition.

        For items, the first line is the category name (rendered in category
        color by _draw_detail_panel). Remaining lines are white.
        """
        lines: list[str] = []

        if isinstance(entity, Condition):
            lines.append(f"Condition: {entity.name}")
            if hasattr(entity, "description") and entity.description:
                lines.append(entity.description)
            return lines

        item: Item = entity

        # First line: category name (will be rendered in category color)
        category_name = _get_category_name(item.category)
        if category_name:
            lines.append(category_name)

        # Size with slot cost explanation
        size_text = {
            ItemSize.TINY: "Tiny (shares slot)",
            ItemSize.NORMAL: "Normal (1 slot)",
            ItemSize.BIG: "Big (2 slots)",
            ItemSize.HUGE: "Huge (4 slots)",
        }
        lines.append(f"{item.name} - {size_text.get(item.size, 'Unknown size')}")

        # Description
        if item.description:
            # Truncate long descriptions
            desc = item.description
            if len(desc) > 70:
                desc = desc[:67] + "..."
            lines.append(desc)

        # Check if this is an outfit item - capability is now stored on the item
        from catley.game.outfit import get_outfit_spec

        outfit_spec = get_outfit_spec(item.item_type)
        if outfit_spec is not None:
            # Get capability from item (persists damage state across equip/unequip)
            outfit_cap = item.outfit_capability

            if outfit_spec.protection > 0:
                if outfit_cap is not None:
                    if outfit_cap.is_broken:
                        lines.append(
                            f"Protection: {outfit_spec.protection}  AP: BROKEN"
                        )
                    else:
                        lines.append(
                            f"Protection: {outfit_spec.protection}  "
                            f"AP: {outfit_cap.ap}/{outfit_cap.max_ap}"
                        )
                else:
                    # Fallback (shouldn't happen for outfit items)
                    lines.append(
                        f"Protection: {outfit_spec.protection}  "
                        f"AP: {outfit_spec.max_ap}"
                    )
            else:
                lines.append("No protection (clothing only)")
            return lines

        # Find the attack to display: prefer one with PREFERRED, else first available
        from catley.game.items.properties import (
            WEAPON_PROPERTY_DESCRIPTIONS,
            TacticalProperty,
            WeaponProperty,
        )

        attacks: list[tuple[str, object | None]] = [
            ("Melee", item.melee_attack),
            ("Ranged", item.ranged_attack),
            ("Area", item.area_effect),
        ]

        # Look for attack with PREFERRED
        chosen_attack: tuple[str, object] | None = None
        for label, attack in attacks:
            if attack and WeaponProperty.PREFERRED in attack.properties:
                chosen_attack = (label, attack)
                break

        # Fallback to first available attack
        if not chosen_attack:
            for label, attack in attacks:
                if attack:
                    chosen_attack = (label, attack)
                    break

        # Show stats for the chosen attack
        if chosen_attack:
            label, attack = chosen_attack
            if label == "Ranged":
                # Skip ammo display for THROWN weapons (single-use items)
                if WeaponProperty.THROWN in attack.properties:
                    lines.append(f"Ranged: {attack.damage_dice}")
                else:
                    ammo_str = (
                        f"{attack.current_ammo}/{attack.max_ammo}"
                        if attack.max_ammo
                        else "N/A"
                    )
                    lines.append(f"Ranged: {attack.damage_dice} [{ammo_str}]")
            else:
                lines.append(f"{label}: {attack.damage_dice}")

            # Show properties from the chosen attack only
            lines.extend(
                WEAPON_PROPERTY_DESCRIPTIONS[prop]
                for prop in attack.properties
                if isinstance(prop, WeaponProperty | TacticalProperty)
                and prop in WEAPON_PROPERTY_DESCRIPTIONS
            )

        return lines

    def _calculate_dimensions(self) -> None:
        """Calculate menu dimensions in pixels using font metrics."""
        # Get font metrics from the PillowImageCanvas
        assert isinstance(self.canvas, PillowImageCanvas)
        self._line_height = self.canvas.get_effective_line_height()
        self._char_width, _, _ = self.canvas.get_text_metrics(" ")

        if self._line_height == 0 or self._char_width == 0:
            return

        # Use narrower width for inventory-only mode (in characters)
        self.width = self.SINGLE_PANE_WIDTH if self.source is None else self.TOTAL_WIDTH
        self.height = self.TOTAL_HEIGHT

        # Calculate pixel dimensions from font metrics
        self.pixel_width = self.width * self._char_width
        self.pixel_height = self.height * self._line_height

        # Store tile dimensions for screen positioning
        self.tile_dimensions = self.controller.graphics.tile_dimensions
        tile_w, tile_h = self.tile_dimensions

        # Center on screen
        screen_pixel_w = self.controller.graphics.console_width_tiles * tile_w
        screen_pixel_h = self.controller.graphics.console_height_tiles * tile_h

        # Calculate centered pixel position, then convert to tiles
        center_px_x = (screen_pixel_w - self.pixel_width) // 2
        center_px_y = (screen_pixel_h - self.pixel_height) // 2

        self.x_tiles = center_px_x // tile_w if tile_w > 0 else 0
        self.y_tiles = center_px_y // tile_h if tile_h > 0 else 0

    # Visual constants for menu styling
    MENU_BG_COLOR: colors.Color = (15, 15, 15)  # subtle dark grey, not pure black

    def draw_content(self) -> None:
        """Render the complete dual-pane interface using pixel-based layout."""
        assert self.canvas is not None
        assert isinstance(self.canvas, PillowImageCanvas)

        char_w = self._char_width
        line_h = self._line_height

        # Fill entire background with black
        self.canvas.draw_rect(
            pixel_x=0,
            pixel_y=0,
            width=self.pixel_width,
            height=self.pixel_height,
            color=colors.BLACK,
            fill=True,
        )

        # Draw double-line box frame using inherited method
        self._draw_box_frame()

        # Fill interior with dark grey (slightly lighter than pure black)
        interior_px_x = char_w
        interior_px_y = line_h
        interior_width = (self.width - 2) * char_w
        interior_height = (self.height - 2) * line_h
        self.canvas.draw_rect(
            pixel_x=interior_px_x,
            pixel_y=interior_px_y,
            width=interior_width,
            height=interior_height,
            color=self.MENU_BG_COLOR,
            fill=True,
        )

        # Draw pane headers
        self._draw_pane_headers()

        # Draw left pane items
        self._draw_left_pane_items()

        # Draw divider and right pane (only in loot mode)
        if self.source is not None:
            self._draw_divider()
            self._draw_right_pane_items()

        # Draw detail panel
        self._draw_detail_panel()

        # Draw hint bar
        self._draw_hint_bar()

    def _draw_pane_headers(self) -> None:
        """Draw the capacity bar headers for both panes."""
        assert self.canvas is not None
        assert isinstance(self.canvas, PillowImageCanvas)
        char_w = self._char_width
        line_h = self._line_height

        player = self.controller.gw.player
        used_space = player.inventory.get_used_inventory_slots()
        total_slots = player.inventory.total_inventory_slots
        slots_over = player.inventory.get_slots_over_capacity()

        # Determine capacity text color based on encumbrance severity
        # Yellow = at/under capacity, Orange = 1-2 over, Red = 3+ over
        if slots_over >= 3:
            capacity_color = colors.RED
        elif slots_over >= 1:
            capacity_color = colors.ORANGE
        else:
            capacity_color = colors.YELLOW

        # Left pane header (line 1, inside the border - line 0 is the top border)
        current_x = 1
        header_y = 1

        # Title
        prefix_text = "Inventory:"
        title_color = colors.YELLOW

        self.canvas.draw_text(
            pixel_x=current_x * char_w,
            pixel_y=header_y * line_h,
            text=prefix_text,
            color=title_color,
        )
        prefix_width, _, _ = self.canvas.get_text_metrics(prefix_text + " ")
        current_px_x = current_x * char_w + prefix_width

        # Capacity bar - draw slots as colored rectangles
        slot_colors = player.inventory.get_inventory_slot_colors()
        bar_max_chars = self.LEFT_PANE_WIDTH - current_x - 8
        slot_width = char_w  # Each slot is one character wide
        for i in range(min(total_slots, bar_max_chars)):
            color = slot_colors[i] if i < len(slot_colors) else colors.GREY
            self.canvas.draw_rect(
                current_px_x, header_y * line_h, slot_width, line_h, color, fill=True
            )
            current_px_x += slot_width

        # Add a small gap before usage text
        current_px_x += char_w // 2

        # Usage text with severity-based color
        usage_text = f"{used_space}/{total_slots}"
        self.canvas.draw_text(
            current_px_x, header_y * line_h, usage_text, capacity_color
        )

        # Right pane header (only in loot mode)
        if self.source is not None:
            right_start = self.LEFT_PANE_WIDTH + self.DIVIDER_WIDTH
            self.canvas.draw_text(
                right_start * char_w,
                header_y * line_h,
                f"{self.source.label}:",
                colors.YELLOW,
            )

    def _draw_left_pane_items(self) -> None:
        """Draw items in the left pane."""
        assert self.canvas is not None
        assert isinstance(self.canvas, PillowImageCanvas)
        char_w = self._char_width
        line_h = self._line_height

        # Items start at line 2: border (0) + header (1) + items start (2)
        start_y = self.HEADER_HEIGHT + 2
        for i, option in enumerate(self.left_options):
            if i >= self.ITEM_LIST_HEIGHT:
                break

            y = start_y + i
            is_selected = self.active_pane == PaneId.LEFT and i == self.left_cursor

            # Draw highlight background for selected item
            if is_selected and option.enabled:
                self.canvas.draw_rect(
                    char_w,  # Start at column 1
                    y * line_h,
                    (self.LEFT_PANE_WIDTH - 1) * char_w,
                    line_h,
                    colors.DARK_GREY,
                    fill=True,
                )

            # Draw prefix segments (slot indicator + category prefix)
            current_px_x = char_w  # Start at column 1
            if option.prefix_segments:
                for segment_text, segment_color in option.prefix_segments:
                    draw_color = segment_color
                    if is_selected and option.enabled:
                        draw_color = self._brighten_color(draw_color)
                    self.canvas.draw_text(
                        current_px_x, y * line_h, segment_text, draw_color
                    )
                    segment_width, _, _ = self.canvas.get_text_metrics(segment_text)
                    current_px_x += segment_width
            elif option.prefix:
                # Fallback to legacy prefix/prefix_color
                prefix_color = option.prefix_color or option.color
                if is_selected and option.enabled:
                    prefix_color = self._brighten_color(prefix_color)
                self.canvas.draw_text(
                    current_px_x, y * line_h, option.prefix, prefix_color
                )
                prefix_width, _, _ = self.canvas.get_text_metrics(option.prefix)
                current_px_x += prefix_width

            # Draw key hint
            if option.key:
                key_text = f"({option.key}) "
                key_color = colors.YELLOW if option.enabled else colors.GREY
                if is_selected and option.enabled:
                    key_color = self._brighten_color(key_color)
                self.canvas.draw_text(current_px_x, y * line_h, key_text, key_color)
                key_width, _, _ = self.canvas.get_text_metrics(key_text)
                current_px_x += key_width

            # Draw item text
            text_color = (
                option.color if option.enabled or option.force_color else colors.GREY
            )
            if is_selected and option.enabled:
                text_color = self._brighten_color(text_color)

            # Calculate max width for text and truncate if needed
            max_text_width = self.LEFT_PANE_WIDTH * char_w - current_px_x - char_w
            text = option.text
            text_width, _, _ = self.canvas.get_text_metrics(text)
            while text_width > max_text_width and len(text) > 1:
                text = text[:-1]
                text_width, _, _ = self.canvas.get_text_metrics(text)

            self.canvas.draw_text(current_px_x, y * line_h, text, text_color)

    def _draw_divider(self) -> None:
        """Draw the center divider with transfer arrow.

        Uses Unicode box-drawing double vertical line (║) to match the outer
        frame, with proper T-junctions at the top border (╦).
        Triangle arrows (▶/◀) indicate transfer direction.
        """
        assert self.canvas is not None
        char_w = self._char_width
        line_h = self._line_height

        divider_x = self.LEFT_PANE_WIDTH + self.DIVIDER_WIDTH // 2
        # Arrow centered in item list area (which starts at line 2)
        mid_y = self.HEADER_HEIGHT + 2 + self.ITEM_LIST_HEIGHT // 2

        # Column positions for the two divider lines
        left_divider_col = self.LEFT_PANE_WIDTH + 1
        right_divider_col = self.LEFT_PANE_WIDTH + self.DIVIDER_WIDTH - 2

        # Draw T-junctions at top border (row 0) where dividers meet horizontal line
        self.canvas.draw_text(left_divider_col * char_w, 0, "╦", colors.WHITE)
        self.canvas.draw_text(right_divider_col * char_w, 0, "╦", colors.WHITE)

        # Draw vertical lines using double box-drawing character (║)
        # Lines start at row 1 (below the top border T-junction)
        for y in range(1, self.HEADER_HEIGHT + 2 + self.ITEM_LIST_HEIGHT):
            self.canvas.draw_text(
                left_divider_col * char_w,
                y * line_h,
                "║",
                colors.WHITE,
            )
            self.canvas.draw_text(
                right_divider_col * char_w,
                y * line_h,
                "║",
                colors.WHITE,
            )

        # Draw transfer arrow using black triangles (U+25B6, U+25C0)
        arrow = "▶" if self.active_pane == PaneId.LEFT else "◀"
        arrow_color = colors.WHITE
        self.canvas.draw_text(divider_x * char_w, mid_y * line_h, arrow, arrow_color)

    def _draw_right_pane_items(self) -> None:
        """Draw items in the right pane."""
        assert self.canvas is not None
        assert isinstance(self.canvas, PillowImageCanvas)
        char_w = self._char_width
        line_h = self._line_height

        # Items start at line 2: border (0) + header (1) + items start (2)
        start_y = self.HEADER_HEIGHT + 2
        right_start = self.LEFT_PANE_WIDTH + self.DIVIDER_WIDTH

        for i, option in enumerate(self.right_options):
            if i >= self.ITEM_LIST_HEIGHT:
                break

            y = start_y + i
            is_selected = self.active_pane == PaneId.RIGHT and i == self.right_cursor

            # Draw highlight background
            if is_selected and option.enabled:
                self.canvas.draw_rect(
                    right_start * char_w,
                    y * line_h,
                    (self.TOTAL_WIDTH - right_start - 1) * char_w,
                    line_h,
                    colors.DARK_GREY,
                    fill=True,
                )

            current_px_x = right_start * char_w

            # Draw prefix segments (category prefix)
            if option.prefix_segments:
                for segment_text, segment_color in option.prefix_segments:
                    draw_color = segment_color
                    if is_selected and option.enabled:
                        draw_color = self._brighten_color(draw_color)
                    self.canvas.draw_text(
                        current_px_x, y * line_h, segment_text, draw_color
                    )
                    segment_width, _, _ = self.canvas.get_text_metrics(segment_text)
                    current_px_x += segment_width

            # Draw key hint
            if option.key:
                key_text = f"({option.key}) "
                key_color = colors.YELLOW if option.enabled else colors.GREY
                if is_selected and option.enabled:
                    key_color = self._brighten_color(key_color)
                self.canvas.draw_text(current_px_x, y * line_h, key_text, key_color)
                key_width, _, _ = self.canvas.get_text_metrics(key_text)
                current_px_x += key_width

            # Draw item text
            text_color = option.color if option.enabled else colors.GREY
            if is_selected and option.enabled:
                text_color = self._brighten_color(text_color)

            # Calculate max width for text and truncate if needed
            max_text_width = (self.TOTAL_WIDTH - 1) * char_w - current_px_x
            text = option.text
            text_width, _, _ = self.canvas.get_text_metrics(text)
            while text_width > max_text_width and len(text) > 1:
                text = text[:-1]
                text_width, _, _ = self.canvas.get_text_metrics(text)

            self.canvas.draw_text(current_px_x, y * line_h, text, text_color)

    def _draw_detail_panel(self) -> None:
        """Draw the item detail panel at the bottom."""
        assert self.canvas is not None
        assert isinstance(self.canvas, PillowImageCanvas)
        char_w = self._char_width
        line_h = self._line_height

        # Separator line - keycap hints always fit on one line
        sep_y = self.TOTAL_HEIGHT - self.DETAIL_HEIGHT - self.HINT_HEIGHT - 1

        # Draw horizontal separator with T-junctions where it meets the border
        # Uses double-to-single T-junctions: ╟ (left) and ╢ (right)
        separator_px_y = sep_y * line_h

        # Column positions for divider lines (used in loot mode)
        left_divider_col = self.LEFT_PANE_WIDTH + 1
        right_divider_col = self.LEFT_PANE_WIDTH + self.DIVIDER_WIDTH - 2

        # Left junction (double-vertical to single-horizontal)
        self.canvas.draw_text(0, separator_px_y, "╟", colors.WHITE)

        # Horizontal line with junction characters where dividers meet
        for x in range(1, self.width - 1):
            # In loot mode, draw junctions where divider columns meet separator
            if self.source is not None and x in (left_divider_col, right_divider_col):
                # ╨ = double vertical up meeting single horizontal
                self.canvas.draw_text(x * char_w, separator_px_y, "╨", colors.WHITE)
            else:
                self.canvas.draw_text(x * char_w, separator_px_y, "─", colors.WHITE)

        # Right junction
        self.canvas.draw_text(
            (self.width - 1) * char_w, separator_px_y, "╢", colors.WHITE
        )

        detail_start_y = sep_y + 1

        if self.detail_item is None:
            self.canvas.draw_text(
                2 * char_w,
                detail_start_y * line_h,
                "(No item selected)",
                colors.GREY,
            )
            return

        # Generate and draw detail lines
        detail_lines = self._generate_item_detail(self.detail_item)

        # Determine if first line should be colored (item category)
        first_line_color = colors.WHITE
        if isinstance(self.detail_item, Item):
            _, cat_color = _get_category_prefix(self.detail_item)
            if cat_color:
                first_line_color = cat_color

        # Calculate max text width for truncation
        max_text_width = (self.width - 4) * char_w

        for i, line in enumerate(detail_lines[: self.DETAIL_HEIGHT - 1]):
            # Truncate text to fit
            text = line
            text_width, _, _ = self.canvas.get_text_metrics(text)
            while text_width > max_text_width and len(text) > 1:
                text = text[:-1]
                text_width, _, _ = self.canvas.get_text_metrics(text)

            # First line (category) uses category color, rest are white
            line_color = first_line_color if i == 0 else colors.WHITE
            self.canvas.draw_text(
                2 * char_w,
                (detail_start_y + i) * line_h,
                text,
                line_color,
            )

    def _get_hint_lines(self) -> list[str]:
        """Get hint text wrapped into lines that fit the current width.

        Footer commands vary by mode, active pane, and selected item:
        - [Enter] Equip: Always shown (any item can be held)
        - [U] Use: Only if selected item has consumable_effect
        - [D] Drop / [T] Transfer: Always shown
        - [Tab] Switch: Dual-pane mode only
        """
        # Get selected item to determine which hints to show
        selected_item: Item | None = None
        if self.active_pane == PaneId.LEFT:
            if 0 <= self.left_cursor < len(self.left_options):
                data = self.left_options[self.left_cursor].data
                if isinstance(data, Item):
                    selected_item = data
        elif 0 <= self.right_cursor < len(self.right_options):
            data = self.right_options[self.right_cursor].data
            if isinstance(data, Item):
                selected_item = data

        # Build hint parts based on context
        parts: list[str] = []

        if self.source is None:
            # Single-pane (inventory only)
            parts.append("[Enter] Equip")
            if selected_item and selected_item.consumable_effect:
                parts.append("[U] Use")
            parts.append("[D] Drop")
            parts.append("[Esc] Close")
        elif self.active_pane == PaneId.LEFT:
            # Dual-pane, left pane focused
            parts.append("[Enter] Equip")
            if selected_item and selected_item.consumable_effect:
                parts.append("[U] Use")
            parts.append("[T] Transfer")
            parts.append("[Tab] Switch")
            parts.append("[Esc]")
        else:
            # Dual-pane, right pane focused
            parts.append("[T] Transfer")
            parts.append("[Tab] Switch")
            parts.append("[Esc] Close")

        hint_text = "  ".join(parts)

        # Use pixel-based width checking if canvas is available
        if isinstance(self.canvas, PillowImageCanvas) and self._char_width > 0:
            max_width = (self.width - 4) * self._char_width
            hint_width, _, _ = self.canvas.get_text_metrics(hint_text)
            if hint_width <= max_width:
                return [hint_text]

            # Wrap using pixel widths
            lines: list[str] = []
            remaining = hint_text
            while remaining:
                remaining_width, _, _ = self.canvas.get_text_metrics(remaining)
                if remaining_width <= max_width:
                    lines.append(remaining)
                    break
                # Find last space before max_width
                split_at = -1
                for j in range(len(remaining) - 1, 0, -1):
                    test_text = remaining[:j]
                    test_width, _, _ = self.canvas.get_text_metrics(test_text)
                    if test_width <= max_width:
                        # Found a position that fits, now find a space
                        space_pos = test_text.rfind("  ")
                        if space_pos == -1:
                            space_pos = test_text.rfind(" ")
                        split_at = space_pos if space_pos > 0 else j
                        break
                if split_at <= 0:
                    split_at = len(remaining) // 2
                lines.append(remaining[:split_at].rstrip())
                remaining = remaining[split_at:].lstrip()
            return lines

        # Fallback to character-based wrapping
        max_len = self.width - 4
        if len(hint_text) <= max_len:
            return [hint_text]

        lines = []
        remaining = hint_text
        while remaining:
            if len(remaining) <= max_len:
                lines.append(remaining)
                break
            split_at = remaining.rfind("  ", 0, max_len)
            if split_at == -1:
                split_at = remaining.rfind(" ", 0, max_len)
            if split_at == -1:
                split_at = max_len
            lines.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip()
        return lines

    def _get_hint_items(self) -> list[tuple[str, str]]:
        """Get hint items as (key, label) pairs for keycap rendering.

        Returns a list of (key_text, label_text) tuples. The key_text is
        displayed inside a keycap, and label_text follows it.
        """
        # Get selected item to determine which hints to show
        selected_item: Item | None = None
        if self.active_pane == PaneId.LEFT:
            if 0 <= self.left_cursor < len(self.left_options):
                data = self.left_options[self.left_cursor].data
                if isinstance(data, Item):
                    selected_item = data
        elif 0 <= self.right_cursor < len(self.right_options):
            data = self.right_options[self.right_cursor].data
            if isinstance(data, Item):
                selected_item = data

        items: list[tuple[str, str]] = []

        if self.source is None:
            # Single-pane (inventory only)
            items.append(("ENTER", "Equip"))
            if selected_item and selected_item.consumable_effect:
                items.append(("U", "Use"))
            items.append(("D", "Drop"))
            items.append(("ESC", "Close"))
        elif self.active_pane == PaneId.LEFT:
            # Dual-pane, left pane focused
            items.append(("ENTER", "Equip"))
            if selected_item and selected_item.consumable_effect:
                items.append(("U", "Use"))
            items.append(("T", "Transfer"))
            items.append(("TAB", "Switch"))
            items.append(("ESC", ""))
        else:
            # Dual-pane, right pane focused
            items.append(("T", "Transfer"))
            items.append(("TAB", "Switch"))
            items.append(("ESC", "Close"))

        return items

    def _draw_hint_bar(self) -> None:
        """Draw the keyboard hints at the bottom using keycap-style rendering."""
        from catley.view.ui.drawing_utils import draw_keycap

        assert self.canvas is not None
        char_w = self._char_width
        line_h = self._line_height

        hint_items = self._get_hint_items()

        # Starting position: 2 chars from left edge, on second-to-last row
        current_x = 2 * char_w
        y = (self.TOTAL_HEIGHT - 2) * line_h
        spacing = 16  # Pixels between hint groups

        # Use larger keycaps than the default (which is 85% of line height)
        keycap_size = int(line_h * 1.15)

        for key, label in hint_items:
            # Draw keycap at larger size
            width = draw_keycap(self.canvas, current_x, y, key, keycap_size=keycap_size)
            current_x += width

            # Draw label if present
            if label:
                self.canvas.draw_text(current_x, y, label, colors.GREY)
                label_width, _, _ = self.canvas.get_text_metrics(label)
                current_x += label_width + spacing
            else:
                current_x += spacing

    def _brighten_color(self, color: colors.Color) -> colors.Color:
        """Brighten a color for hover/selection highlighting."""
        return (
            min(255, color[0] + 40),
            min(255, color[1] + 40),
            min(255, color[2] + 40),
        )

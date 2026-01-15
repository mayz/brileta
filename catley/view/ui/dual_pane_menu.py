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
        ItemCategory.WEAPON: ("\u2022 ", colors.CATEGORY_WEAPON),
        ItemCategory.ARMOR: ("\u2022 ", colors.CATEGORY_ARMOR),
        ItemCategory.CONSUMABLE: ("\u2022 ", colors.CATEGORY_CONSUMABLE),
        ItemCategory.JUNK: ("\u2022 ", colors.CATEGORY_JUNK),
        ItemCategory.MUNITIONS: ("\u2022 ", colors.CATEGORY_MUNITIONS),
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

    KEY_J: Final = tcod.event.KeySym(ord("j"))
    KEY_K: Final = tcod.event.KeySym(ord("k"))
    KEY_D: Final = tcod.event.KeySym(ord("d"))
    KEY_I: Final = tcod.event.KeySym(ord("i"))
    KEY_Q: Final = tcod.event.KeySym(ord("q"))


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
    HINT_HEIGHT = 1
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

        # Collect all items: stored + equipped
        stored_items = self.inventory._stored_items
        all_items_to_display: list[tuple[Item | Condition, str]] = [
            (item, "stored") for item in stored_items
        ]

        # Add equipped items with their slot index
        for slot_index, equipped_item in enumerate(self.inventory.attack_slots):
            if equipped_item:
                all_items_to_display.append((equipped_item, f"slot:{slot_index}"))

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

        for entity, item_status in all_items_to_display:
            if isinstance(entity, Item):
                item = entity

                # Build prefix segments: slot indicator (if equipped) + category
                prefix_segments: list[tuple[str, colors.Color]] = []

                # Parse equipped slot number for prefix
                if item_status.startswith("slot:"):
                    slot_num = int(item_status.split(":")[1]) + 1
                    prefix_segments.append((f"[{slot_num}]  ", colors.YELLOW))
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
                        action=functools.partial(self._use_item, item),
                        enabled=True,
                        color=colors.WHITE,
                        data=item,
                        prefix_segments=prefix_segments,
                    )
                )

            elif isinstance(entity, Condition):
                condition = entity
                # Conditions have no category, just alignment spaces
                self.left_options.append(
                    MenuOption(
                        key=None,
                        text=f"{condition.name} \u2588",  # Block character
                        action=None,
                        enabled=False,
                        color=condition.display_color,
                        force_color=True,
                        data=condition,
                        prefix_segments=[("     ", colors.WHITE)],  # Alignment only
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

    def _use_item(self, item: Item) -> bool:
        """Use or equip an item from player inventory.

        Returns False to keep menu open.
        """
        player = self.controller.gw.player

        if item.equippable:
            # Check if item is already equipped
            currently_equipped_slot = None
            for i, equipped_item in enumerate(player.inventory.attack_slots):
                if equipped_item == item:
                    currently_equipped_slot = i
                    break

            if currently_equipped_slot is not None:
                success, message = player.inventory.unequip_to_inventory(
                    currently_equipped_slot
                )
                color = colors.WHITE if success else colors.YELLOW
                publish_event(MessageEvent(message, color))
            else:
                active_slot = player.inventory.active_weapon_slot
                success, message = player.inventory.equip_from_inventory(
                    item, active_slot
                )
                color = colors.GREEN if success else colors.YELLOW
                publish_event(MessageEvent(message, color))
        else:
            from catley.game.actions.recovery import UseConsumableIntent

            if item.consumable_effect:
                intent = UseConsumableIntent(self.controller, player, item)
                self.controller.queue_action(intent)
                return True  # Close menu after using consumable
            publish_event(MessageEvent(f"{item.name} cannot be used", colors.YELLOW))

        # Refresh panes after equip/unequip
        self._populate_left_pane()
        self._update_detail_from_cursor()
        return False  # Keep menu open

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
                    # Remove empty ground container
                    if (
                        not isinstance(actor, Character)
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

        # Adjust cursor if needed
        if self.right_cursor >= len(self.right_options):
            self.right_cursor = max(0, len(self.right_options) - 1)

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

        # Unequip if equipped
        for i, equipped_item in enumerate(player.inventory.attack_slots):
            if equipped_item == item:
                player.inventory.unequip_slot(i)
                break

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

        # Adjust cursor if needed
        if self.left_cursor >= len(self.left_options):
            self.left_cursor = max(0, len(self.left_options) - 1)

        self._update_detail_from_cursor()
        return False  # Keep menu open

    def _drop_item(self, item: Item) -> bool:
        """Drop an item from inventory to the ground at player location.

        Returns True to close menu.
        """
        from catley.game.actions.misc import DropItemIntent

        player = self.controller.gw.player
        intent = DropItemIntent(self.controller, player, item)
        self.controller.queue_action(intent)
        return True  # Close menu after dropping

    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle dual-pane navigation and actions."""
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

            # D key drops highlighted item from left pane
            case tcod.event.KeyDown(sym=_MenuKeys.KEY_D):
                if self.active_pane == PaneId.LEFT and 0 <= self.left_cursor < len(
                    self.left_options
                ):
                    option = self.left_options[self.left_cursor]
                    if (
                        option.enabled
                        and isinstance(option.data, Item)
                        and self._drop_item(option.data)
                    ):
                        self.hide()
                return True

            # Enter activates current item
            case tcod.event.KeyDown(sym=tcod.event.KeySym.RETURN):
                result = self._activate_current_item()
                if result:
                    self.hide()
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

        return True  # Consume all input while menu is active

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
        """Activate the currently selected item.

        Returns True if menu should close.
        """
        if self.active_pane == PaneId.LEFT:
            options = self.left_options
            cursor = self.left_cursor

            if 0 <= cursor < len(options):
                option = options[cursor]
                if option.enabled:
                    # In loot mode, Enter on left pane transfers to container
                    if self.source is not None and isinstance(option.data, Item):
                        return self._transfer_to_container(option.data)
                    if option.action:
                        result = option.action()
                        return result is True
        else:
            options = self.right_options
            cursor = self.right_cursor

            if 0 <= cursor < len(options):
                option = options[cursor]
                if option.enabled and option.action:
                    result = option.action()
                    return result is True

        return False

    def _handle_mouse_motion(self, event: tcod.event.MouseMotion) -> bool:
        """Handle mouse motion to update hover state."""
        mouse_px_x, mouse_px_y = event.position

        # Convert to menu-relative coordinates
        menu_px_x = self.x_tiles * self.tile_dimensions[0]
        menu_px_y = self.y_tiles * self.tile_dimensions[1]
        rel_px_x = mouse_px_x - menu_px_x
        rel_px_y = mouse_px_y - menu_px_y

        tile_w, tile_h = self.tile_dimensions
        if tile_w == 0 or tile_h == 0:
            return True

        rel_tile_x = int(rel_px_x // tile_w)
        rel_tile_y = int(rel_px_y // tile_h)

        # Determine which pane the mouse is over
        # Left pane: tiles 1 to LEFT_PANE_WIDTH
        # Right pane: tiles LEFT_PANE_WIDTH + DIVIDER_WIDTH to end
        item_start_y = self.HEADER_HEIGHT + 1  # After header and border

        if 1 <= rel_tile_x < self.LEFT_PANE_WIDTH:
            # Over left pane
            option_line = rel_tile_y - item_start_y
            if 0 <= option_line < len(self.left_options):
                self.active_pane = PaneId.LEFT
                self.left_cursor = option_line
                self._update_detail_from_cursor()
        elif (
            self.source is not None
            and self.LEFT_PANE_WIDTH + self.DIVIDER_WIDTH
            <= rel_tile_x
            < self.TOTAL_WIDTH - 1
        ):
            # Over right pane
            option_line = rel_tile_y - item_start_y
            if 0 <= option_line < len(self.right_options):
                self.active_pane = PaneId.RIGHT
                self.right_cursor = option_line
                self._update_detail_from_cursor()

        return True

    def _handle_mouse_click(self, event: tcod.event.MouseButtonDown) -> bool:
        """Handle mouse click to select/transfer items."""
        mouse_px_x, mouse_px_y = event.position

        # Check if click is outside menu
        menu_px_x = self.x_tiles * self.tile_dimensions[0]
        menu_px_y = self.y_tiles * self.tile_dimensions[1]
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
        """Calculate menu dimensions and position."""
        tile_w, tile_h = self.tile_dimensions
        if tile_w == 0 or tile_h == 0:
            return

        # Use narrower width for inventory-only mode
        self.width = self.SINGLE_PANE_WIDTH if self.source is None else self.TOTAL_WIDTH
        self.height = self.TOTAL_HEIGHT

        self.pixel_width = self.width * tile_w
        self.pixel_height = self.height * tile_h

        # Center on screen
        screen_width = self.controller.graphics.console_width_tiles
        screen_height = self.controller.graphics.console_height_tiles

        self.x_tiles = max(0, (screen_width - self.width) // 2)
        self.y_tiles = max(0, (screen_height - self.height) // 2)

    # Visual constants for menu styling
    MENU_BG_COLOR: colors.Color = (15, 15, 15)  # subtle dark grey, not pure black

    def draw_content(self) -> None:
        """Render the complete dual-pane interface."""
        assert self.canvas is not None

        tile_w, tile_h = self.tile_dimensions

        # Draw outer frame
        self.canvas.draw_frame(
            0, 0, self.width, self.height, colors.WHITE, colors.BLACK
        )

        # Fill interior with dark grey (slightly lighter than pure black)
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                self.canvas.draw_rect(
                    x * tile_w,
                    y * tile_h,
                    tile_w,
                    tile_h,
                    self.MENU_BG_COLOR,
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
        tile_w, tile_h = self.tile_dimensions

        player = self.controller.gw.player
        used_space = player.inventory.get_used_inventory_slots()
        total_slots = player.inventory.total_inventory_slots

        # Left pane header
        current_x = 1
        header_y = 0

        # Title
        prefix_text = "Inventory:"
        title_color = colors.YELLOW

        self.canvas.draw_text(
            pixel_x=current_x * tile_w,
            pixel_y=header_y * tile_h,
            text=prefix_text,
            color=title_color,
        )
        current_x += len(prefix_text) + 1

        # Capacity bar
        slot_colors = player.inventory.get_inventory_slot_colors()
        bar_end = min(current_x + total_slots, self.LEFT_PANE_WIDTH - 8)
        for i in range(min(total_slots, bar_end - current_x)):
            color = slot_colors[i] if i < len(slot_colors) else colors.GREY
            self.canvas.draw_rect(
                current_x * tile_w, header_y * tile_h, tile_w, tile_h, color, fill=True
            )
            self.canvas.draw_text(
                current_x * tile_w, header_y * tile_h, " ", colors.WHITE
            )
            current_x += 1

        # Usage text
        usage_text = f" {used_space}/{total_slots}"
        self.canvas.draw_text(
            current_x * tile_w, header_y * tile_h, usage_text, colors.YELLOW
        )

        # Right pane header (only in loot mode)
        if self.source is not None:
            right_start = self.LEFT_PANE_WIDTH + self.DIVIDER_WIDTH
            self.canvas.draw_text(
                right_start * tile_w,
                header_y * tile_h,
                f"{self.source.label}:",
                colors.YELLOW,
            )

    def _draw_left_pane_items(self) -> None:
        """Draw items in the left pane."""
        assert self.canvas is not None
        tile_w, tile_h = self.tile_dimensions

        start_y = self.HEADER_HEIGHT + 1
        for i, option in enumerate(self.left_options):
            if i >= self.ITEM_LIST_HEIGHT:
                break

            y = start_y + i
            is_selected = self.active_pane == PaneId.LEFT and i == self.left_cursor

            # Draw highlight background for selected item
            if is_selected and option.enabled:
                for x in range(1, self.LEFT_PANE_WIDTH):
                    self.canvas.draw_rect(
                        x * tile_w,
                        y * tile_h,
                        tile_w,
                        tile_h,
                        colors.DARK_GREY,
                        fill=True,
                    )

            # Draw prefix segments (slot indicator + category prefix)
            x = 1
            if option.prefix_segments:
                for segment_text, segment_color in option.prefix_segments:
                    draw_color = segment_color
                    if is_selected and option.enabled:
                        draw_color = self._brighten_color(draw_color)
                    self.canvas.draw_text(
                        x * tile_w, y * tile_h, segment_text, draw_color
                    )
                    x += len(segment_text)
            elif option.prefix:
                # Fallback to legacy prefix/prefix_color
                prefix_color = option.prefix_color or option.color
                if is_selected and option.enabled:
                    prefix_color = self._brighten_color(prefix_color)
                self.canvas.draw_text(
                    x * tile_w, y * tile_h, option.prefix, prefix_color
                )
                x += len(option.prefix)

            # Draw key hint
            if option.key:
                key_text = f"({option.key}) "
                key_color = colors.YELLOW if option.enabled else colors.GREY
                if is_selected and option.enabled:
                    key_color = self._brighten_color(key_color)
                self.canvas.draw_text(x * tile_w, y * tile_h, key_text, key_color)
                x += len(key_text)

            # Draw item text
            text_color = (
                option.color if option.enabled or option.force_color else colors.GREY
            )
            if is_selected and option.enabled:
                text_color = self._brighten_color(text_color)

            # Truncate text to fit pane
            max_text_len = self.LEFT_PANE_WIDTH - x - 1
            text = (
                option.text[:max_text_len]
                if len(option.text) > max_text_len
                else option.text
            )
            self.canvas.draw_text(x * tile_w, y * tile_h, text, text_color)

    def _draw_divider(self) -> None:
        """Draw the center divider with transfer arrow."""
        assert self.canvas is not None
        tile_w, tile_h = self.tile_dimensions

        divider_x = self.LEFT_PANE_WIDTH + self.DIVIDER_WIDTH // 2
        mid_y = self.HEADER_HEIGHT + 1 + self.ITEM_LIST_HEIGHT // 2

        # Draw vertical line
        for y in range(
            self.HEADER_HEIGHT + 1, self.HEADER_HEIGHT + 1 + self.ITEM_LIST_HEIGHT
        ):
            self.canvas.draw_text(
                (self.LEFT_PANE_WIDTH + 1) * tile_w,
                y * tile_h,
                "|",
                colors.GREY,
            )
            self.canvas.draw_text(
                (self.LEFT_PANE_WIDTH + self.DIVIDER_WIDTH - 2) * tile_w,
                y * tile_h,
                "|",
                colors.GREY,
            )

        # Draw transfer arrow
        arrow = ">" if self.active_pane == PaneId.LEFT else "<"
        arrow_color = colors.WHITE
        self.canvas.draw_text(divider_x * tile_w, mid_y * tile_h, arrow, arrow_color)

    def _draw_right_pane_items(self) -> None:
        """Draw items in the right pane."""
        assert self.canvas is not None
        tile_w, tile_h = self.tile_dimensions

        start_y = self.HEADER_HEIGHT + 1
        right_start = self.LEFT_PANE_WIDTH + self.DIVIDER_WIDTH

        for i, option in enumerate(self.right_options):
            if i >= self.ITEM_LIST_HEIGHT:
                break

            y = start_y + i
            is_selected = self.active_pane == PaneId.RIGHT and i == self.right_cursor

            # Draw highlight background
            if is_selected and option.enabled:
                for x in range(right_start, self.TOTAL_WIDTH - 1):
                    self.canvas.draw_rect(
                        x * tile_w,
                        y * tile_h,
                        tile_w,
                        tile_h,
                        colors.DARK_GREY,
                        fill=True,
                    )

            x = right_start

            # Draw prefix segments (category prefix)
            if option.prefix_segments:
                for segment_text, segment_color in option.prefix_segments:
                    draw_color = segment_color
                    if is_selected and option.enabled:
                        draw_color = self._brighten_color(draw_color)
                    self.canvas.draw_text(
                        x * tile_w, y * tile_h, segment_text, draw_color
                    )
                    x += len(segment_text)

            # Draw key hint
            if option.key:
                key_text = f"({option.key}) "
                key_color = colors.YELLOW if option.enabled else colors.GREY
                if is_selected and option.enabled:
                    key_color = self._brighten_color(key_color)
                self.canvas.draw_text(x * tile_w, y * tile_h, key_text, key_color)
                x += len(key_text)

            # Draw item text
            text_color = option.color if option.enabled else colors.GREY
            if is_selected and option.enabled:
                text_color = self._brighten_color(text_color)

            max_text_len = self.TOTAL_WIDTH - x - 2
            text = (
                option.text[:max_text_len]
                if len(option.text) > max_text_len
                else option.text
            )
            self.canvas.draw_text(x * tile_w, y * tile_h, text, text_color)

    def _draw_detail_panel(self) -> None:
        """Draw the item detail panel at the bottom."""
        assert self.canvas is not None
        tile_w, tile_h = self.tile_dimensions

        # Separator line - account for dynamic hint line count
        hint_line_count = len(self._get_hint_lines())
        sep_y = self.TOTAL_HEIGHT - self.DETAIL_HEIGHT - hint_line_count - 1
        for x in range(1, self.width - 1):
            self.canvas.draw_text(
                x * tile_w,
                sep_y * tile_h,
                "\u2500",
                colors.GREY,  # Box drawing horizontal
            )

        detail_start_y = sep_y + 1

        if self.detail_item is None:
            self.canvas.draw_text(
                2 * tile_w,
                detail_start_y * tile_h,
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

        for i, line in enumerate(detail_lines[: self.DETAIL_HEIGHT - 1]):
            truncated = line[: self.width - 4]
            # First line (category) uses category color, rest are white
            line_color = first_line_color if i == 0 else colors.WHITE
            self.canvas.draw_text(
                2 * tile_w,
                (detail_start_y + i) * tile_h,
                truncated,
                line_color,
            )

    def _get_hint_lines(self) -> list[str]:
        """Get hint text wrapped into lines that fit the current width."""
        if self.source is not None:
            hint_text = (
                "[Tab] Switch  [Arrows/JK] Navigate  [Enter] Transfer  [D] Drop  [Esc]"
            )
        else:
            hint_text = "[Arrows/JK] Navigate  [Enter] Use/Equip  [D] Drop  [Esc] Close"

        max_len = self.width - 4
        if len(hint_text) <= max_len:
            return [hint_text]

        # Wrap into multiple lines
        lines: list[str] = []
        remaining = hint_text
        while remaining:
            if len(remaining) <= max_len:
                lines.append(remaining)
                break
            # Find last space before max_len
            split_at = remaining.rfind("  ", 0, max_len)
            if split_at == -1:
                split_at = remaining.rfind(" ", 0, max_len)
            if split_at == -1:
                split_at = max_len
            lines.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip()
        return lines

    def _draw_hint_bar(self) -> None:
        """Draw the keyboard hints at the bottom."""
        assert self.canvas is not None
        tile_w, tile_h = self.tile_dimensions

        hint_lines = self._get_hint_lines()
        hint_color = colors.GREY

        # Draw lines from bottom up
        for i, line in enumerate(reversed(hint_lines)):
            y = self.TOTAL_HEIGHT - 1 - i
            self.canvas.draw_text(2 * tile_w, y * tile_h, line, hint_color)

    def _brighten_color(self, color: colors.Color) -> colors.Color:
        """Brighten a color for hover/selection highlighting."""
        return (
            min(255, color[0] + 40),
            min(255, color[1] + 40),
            min(255, color[2] + 40),
        )

from __future__ import annotations

import functools
import string
from collections.abc import Callable
from typing import TYPE_CHECKING

from tcod.console import Console

from catley import colors
from catley.events import MessageEvent, publish_event
from catley.game.actors import Condition
from catley.game.enums import ItemSize
from catley.game.items.item_core import Item
from catley.view.ui.overlays import Menu, MenuOption

if TYPE_CHECKING:
    from catley.controller import Controller


class InventoryMenu(Menu):
    """Menu for managing the player's inventory."""

    def __init__(self, controller: Controller) -> None:
        super().__init__("Inventory", controller)
        self.inventory = controller.gw.player.inventory

    def populate_options(self) -> None:
        """Populate inventory options."""
        self.options.clear()  # Ensure options are cleared at the start

        # Use letters a-z for inventory items
        letters = string.ascii_lowercase
        letter_idx = 0

        # This list will hold MenuOption objects for each display line
        display_lines: list[MenuOption] = []

        # Collect all items to display (stored + equipped)
        # Start with all stored items
        stored_items = self.inventory._stored_items
        all_items_to_display = [(item, "stored") for item in stored_items]

        # Add all equipped items
        for slot_index, equipped_item in enumerate(self.inventory.attack_slots):
            if equipped_item:
                slot_name = self.inventory.get_slot_display_name(slot_index)
                all_items_to_display.append((equipped_item, f"equipped in {slot_name}"))

        if not all_items_to_display:
            display_lines.append(
                MenuOption(
                    key=None, text="(inventory empty)", enabled=False, color=colors.GREY
                )
            )
            self.options = display_lines
            return

        # Iterate through all items (stored and equipped)
        for entity_info in all_items_to_display:
            if isinstance(entity_info, tuple):
                entity_in_slot, item_status = entity_info
            else:
                entity_in_slot = entity_info
                item_status = "stored"

            text_to_display: str
            action_to_take: Callable[[], None] | None = None
            option_color: colors.Color
            is_enabled: bool
            force_color: bool = False
            current_key: str | None = None

            if isinstance(entity_in_slot, Item):
                item: Item = entity_in_slot

                if item_status.startswith("equipped"):
                    equipped_marker = f" ({item_status})"
                else:
                    equipped_marker = ""

                size_display = ""
                item_display_color = colors.WHITE

                if item.size == ItemSize.TINY:
                    size_display = " (Tiny)"
                elif item.size == ItemSize.NORMAL:
                    size_display = " █"
                elif item.size == ItemSize.BIG:
                    size_display = " █ █"
                elif item.size == ItemSize.HUGE:
                    size_display = " █ █ █ █"

                text_to_display = f"{item.name}{size_display}{equipped_marker}"
                option_color = item_display_color
                is_enabled = True

                if letter_idx < len(letters):
                    current_key = letters[letter_idx]
                    letter_idx += 1
                action_to_take = functools.partial(self._use_item, item)

            elif isinstance(entity_in_slot, Condition):
                condition: Condition = entity_in_slot
                text_to_display = f"{condition.name} █"
                option_color = condition.display_color
                force_color = True
                is_enabled = False
                action_to_take = None
            else:
                text_to_display = f"Unknown: {entity_in_slot}"
                option_color = colors.RED
                is_enabled = False

            display_lines.append(
                MenuOption(
                    key=current_key,
                    text=text_to_display,
                    action=action_to_take,
                    enabled=is_enabled,
                    color=option_color,
                    force_color=force_color,
                )
            )

        self.options = display_lines

    def _use_item(self, item: Item) -> None:
        """Use or equip an item."""
        player = self.controller.gw.player

        if item.equippable:
            # Check if item is already equipped in any slot
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
            from catley.game.actions.recovery import UseConsumableAction

            if item.consumable_effect:
                action = UseConsumableAction(self.controller, player, item)
                action.execute()
            else:
                publish_event(
                    MessageEvent(f"{item.name} cannot be used", colors.YELLOW)
                )

    def render_title(
        self,
        menu_console: Console,
        title_y: int,
        actual_width: int,
    ) -> None:
        """Render the menu title with an inventory usage bar."""
        player = self.controller.gw.player  # Player is an Actor
        used_space = player.inventory.get_used_inventory_slots()
        total_slots = player.inventory.total_inventory_slots

        current_x = 1  # Start drawing at column 1 (content area start)
        # Max content column index is actual_width - 2
        max_content_x = actual_width - 2

        # 1. Print "Inventory: " (self.title for InventoryMenu is "Inventory")
        prefix_text = f"{self.title}: "
        if current_x <= max_content_x:
            chars_to_print = min(len(prefix_text), (max_content_x - current_x + 1))
            menu_console.print(
                x=current_x,
                y=title_y,
                text=prefix_text[:chars_to_print],
                fg=colors.YELLOW,
            )  # type: ignore[no-matching-overload]
            current_x += chars_to_print

        # 2. Draw the color bar using individual slot colors
        slot_colors_list = player.inventory.get_inventory_slot_colors()
        num_colored_slots = len(slot_colors_list)

        if total_slots > 0 and current_x <= max_content_x:
            for i in range(total_slots):
                if (
                    current_x <= max_content_x
                ):  # Check if current cell is within content area
                    cell_x = current_x
                    if i < num_colored_slots:
                        menu_console.bg[cell_x, title_y] = slot_colors_list[i]
                    else:
                        menu_console.bg[cell_x, title_y] = colors.GREY
                    menu_console.ch[cell_x, title_y] = ord(" ")
                    current_x += 1
                else:
                    break  # No more space for bar cells
        # 3. Print " X/Y used"
        suffix_text = f" {used_space}/{total_slots} used"
        if current_x <= max_content_x:
            chars_to_print = min(len(suffix_text), (max_content_x - current_x + 1))
            menu_console.print(
                x=current_x,
                y=title_y,
                text=suffix_text[:chars_to_print],
                fg=colors.YELLOW,
            )  # type: ignore[no-matching-overload]

from __future__ import annotations

import functools
import string
from collections.abc import Callable
from typing import TYPE_CHECKING

import tcod.event

from catley import colors
from catley.events import MessageEvent, publish_event
from catley.game.actors import Condition
from catley.game.items.item_core import Item
from catley.view.ui.overlays import Menu, MenuOption

if TYPE_CHECKING:
    from catley.controller import Controller


class InventoryMenu(Menu):
    """Menu for managing the player's inventory."""

    def __init__(self, controller: Controller) -> None:
        super().__init__("Inventory", controller)
        self.inventory = controller.gw.player.inventory
        self.drop_mode = False  # True when Shift is held for dropping items

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

        # Add all equipped items with their slot index
        for slot_index, equipped_item in enumerate(self.inventory.attack_slots):
            if equipped_item:
                all_items_to_display.append((equipped_item, f"slot:{slot_index}"))

        if not all_items_to_display:
            display_lines.append(
                MenuOption(
                    key=None,
                    text="(inventory empty)",
                    enabled=False,
                    color=colors.GREY,
                    is_primary_action=False,
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

                # Parse equipped slot number for prefix display
                # Use fixed width (5 chars) for alignment: "[1]  " or "     "
                equipped_prefix = "     "  # 5 spaces for non-equipped
                equipped_prefix_color: colors.Color | None = None
                if item_status.startswith("slot:"):
                    slot_num = (
                        int(item_status.split(":")[1]) + 1
                    )  # 0-indexed to 1-indexed
                    equipped_prefix = f"[{slot_num}]  "  # 5 chars: [n] + 2 spaces
                    equipped_prefix_color = colors.YELLOW

                text_to_display = item.name
                option_color = colors.WHITE
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
                equipped_prefix = "     "  # 5 spaces for alignment
                equipped_prefix_color = None
            else:
                text_to_display = f"Unknown: {entity_in_slot}"
                option_color = colors.RED
                is_enabled = False
                equipped_prefix = "     "  # 5 spaces for alignment
                equipped_prefix_color = None

            # Store entity reference in data field for drop functionality
            # (Items can be dropped; Conditions cannot - handle_input filters by type)
            display_lines.append(
                MenuOption(
                    key=current_key,
                    text=text_to_display,
                    action=action_to_take,
                    enabled=is_enabled,
                    color=option_color,
                    force_color=force_color,
                    data=entity_in_slot,
                    prefix=equipped_prefix,
                    prefix_color=equipped_prefix_color,
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
            from catley.game.actions.recovery import UseConsumableIntent

            if item.consumable_effect:
                intent = UseConsumableIntent(self.controller, player, item)
                self.controller.queue_action(intent)
            else:
                publish_event(
                    MessageEvent(f"{item.name} cannot be used", colors.YELLOW)
                )

    def _drop_item(self, item: Item) -> None:
        """Drop an item from inventory to the ground."""
        from catley.game.actions.misc import DropItemIntent

        player = self.controller.gw.player
        intent = DropItemIntent(self.controller, player, item)
        self.controller.queue_action(intent)

    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle input events, including Shift modifier for drop mode."""
        if not self.is_active:
            return False

        # Track Shift key state for drop mode visual feedback
        match event:
            case tcod.event.KeyUp(
                sym=tcod.event.KeySym.LSHIFT | tcod.event.KeySym.RSHIFT
            ):
                self.drop_mode = False
                return True

            case tcod.event.KeyDown() as key_event:
                # Update drop mode based on Shift state
                shift_held = bool(key_event.mod & tcod.event.Modifier.SHIFT)
                self.drop_mode = shift_held

                # If Shift is held, handle letter key as drop action
                if shift_held:
                    key_char = (
                        chr(key_event.sym).lower() if 32 <= key_event.sym <= 126 else ""
                    )
                    for option in self.options:
                        if (
                            option.key is not None
                            and option.key.lower() == key_char
                            and option.enabled
                            and isinstance(option.data, Item)
                        ):
                            self._drop_item(option.data)
                            self.hide()
                            return True
                    # Consume the event even if no match (Shift+key pressed)
                    return True

        # Fall back to base menu handling for non-Shift actions
        return super().handle_input(event)

    def draw_content(self) -> None:
        """Draw the base menu, then overlay the inventory usage bar and hint."""
        super().draw_content()
        self.render_title()
        self.render_hint()

    def render_title(self) -> None:
        """Render the inventory header with prefix, bar, and suffix."""
        assert self.canvas is not None

        player = self.controller.gw.player
        used_space = player.inventory.get_used_inventory_slots()
        total_slots = player.inventory.total_inventory_slots

        tile_w, tile_h = self.tile_dimensions

        current_x_tile = 1  # Start drawing one tile in for padding
        title_y_tile = 0

        # Change title text and color based on drop mode
        if self.drop_mode:
            prefix_text = "DROP MODE:"
            title_color = colors.RED
        else:
            prefix_text = f"{self.title}:"
            title_color = colors.YELLOW

        self.canvas.draw_text(
            pixel_x=current_x_tile * tile_w,
            pixel_y=title_y_tile * tile_h,
            text=prefix_text,
            color=title_color,
        )
        current_x_tile += len(prefix_text) + 1  # Space after prefix

        max_content_tile = self.width - 2
        slot_colors_list = player.inventory.get_inventory_slot_colors()
        for i in range(total_slots):
            if current_x_tile <= max_content_tile:
                color = (
                    slot_colors_list[i] if i < len(slot_colors_list) else colors.GREY
                )
                self.canvas.draw_rect(
                    pixel_x=current_x_tile * tile_w,
                    pixel_y=title_y_tile * tile_h,
                    width=tile_w,
                    height=tile_h,
                    color=color,
                    fill=True,
                )
                # Set a space character so the cell is fully drawn
                self.canvas.draw_text(
                    pixel_x=current_x_tile * tile_w,
                    pixel_y=title_y_tile * tile_h,
                    text=" ",
                    color=colors.WHITE,
                )
                current_x_tile += 1
            else:
                break

        suffix_text = f" {used_space}/{total_slots} used"
        if current_x_tile <= max_content_tile:
            available_tiles = max_content_tile - current_x_tile + 1
            text_to_draw = suffix_text[:available_tiles]
            self.canvas.draw_text(
                pixel_x=current_x_tile * tile_w,
                pixel_y=title_y_tile * tile_h,
                text=text_to_draw,
                color=colors.YELLOW,
            )

    def render_hint(self) -> None:
        """Render a hint at the bottom of the menu for drop mode."""
        assert self.canvas is not None

        tile_w, tile_h = self.tile_dimensions

        # Position hint at the bottom row of the menu (inside the frame)
        hint_y_tile = self.height - 1
        hint_x_tile = 1

        if self.drop_mode:
            hint_text = "Press letter to drop item"
            hint_color = colors.RED
        else:
            hint_text = "Hold [Shift] to drop item"
            hint_color = colors.GREY

        self.canvas.draw_text(
            pixel_x=hint_x_tile * tile_w,
            pixel_y=hint_y_tile * tile_h,
            text=hint_text,
            color=hint_color,
        )

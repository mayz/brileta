"""
Menu system for the Catley game using TCOD overlays similar to Brogue.
"""

from __future__ import annotations

import abc
import functools
import string
from typing import TYPE_CHECKING

import colors
import tcod
import tcod.event
from conditions import Condition
from items import Item, ItemSize
from model import Actor
from tcod.console import Console

if TYPE_CHECKING:
    from collections.abc import Callable

    from engine import Controller


class MenuOption:
    """Represents a single option in a menu."""

    def __init__(
        self,
        key: str | None,
        text: str,
        action: Callable[[], None] | None = None,
        enabled: bool = True,
        color: colors.Color = colors.WHITE,
        force_color: bool = False,
    ) -> None:
        self.key = key
        self.text = text
        self.action = action
        self.enabled = enabled
        self.force_color = force_color
        self.color = color if enabled or force_color else colors.GREY


class Menu(abc.ABC):
    """Base class for all menus in the game."""

    def __init__(
        self,
        title: str,
        controller: Controller,
        width: int = 50,
        max_height: int = 30,
    ) -> None:
        self.title = title
        self.controller = controller
        self.width = width
        self.max_height = max_height
        self.options: list[MenuOption] = []
        self.is_active = False

    @abc.abstractmethod
    def populate_options(self) -> None:
        """Populate the menu options. Must be implemented by subclasses."""
        pass

    def add_option(self, option: MenuOption) -> None:
        """Add an option to the menu."""
        self.options.append(option)

    def show(self) -> None:
        """Show the menu."""
        self.is_active = True
        self.populate_options()

    def hide(self) -> None:
        """Hide the menu."""
        self.is_active = False
        self.options.clear()

    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle input events for the menu. Returns True if event was consumed."""
        if not self.is_active:
            return False

        match event:
            case tcod.event.KeyDown(sym=tcod.event.KeySym.ESCAPE):
                self.hide()
                return True
            case tcod.event.KeyDown(sym=tcod.event.KeySym.SPACE):
                self.hide()
                return True
            case tcod.event.KeyDown(sym=tcod.event.KeySym.RETURN):  # Enter key
                self.hide()
                return True
            case tcod.event.KeyDown() as key_event:
                # Convert key to character
                key_char = (
                    chr(key_event.sym).lower() if 32 <= key_event.sym <= 126 else ""
                )

                # Find matching option
                for option in self.options:
                    if (
                        option.key is not None
                        and option.key.lower() == key_char
                        and option.enabled
                        and option.action
                    ):
                        option.action()
                        self.hide()
                        return True
                return True  # Consume all keyboard input while menu is active

        return False

    def render(self, console: Console) -> None:
        """Render the menu as an overlay."""
        if not self.is_active:
            return

        # Calculate menu dimensions
        # Number of lines needed for content pane (inside borders):
        # 1 for title, 1 for separator, len(self.options) for options,
        # 1 for bottom padding inside content area.
        required_content_pane_height = 1 + 1 + len(self.options) + 1
        # This is equivalent to: len(self.options) + 3

        # Total console height needs to accommodate the content pane
        # plus top/bottom borders.
        calculated_total_height = required_content_pane_height + 2  # +2 for borders

        # Apply max_height constraint (to total height) and
        # ensure a minimum total height.
        menu_height = min(calculated_total_height, self.max_height)
        menu_height = max(menu_height, 5)  # Minimum total height for the console

        # Ensure minimum width based on title and content
        min_width = max(len(self.title) + 4, 20)  # Title + padding, or minimum 20
        actual_width = max(self.width, min_width)

        # Center the menu on screen
        menu_x = (console.width - actual_width) // 2
        menu_y = (console.height - menu_height) // 2

        # Create menu console with same order as main console
        menu_console = Console(actual_width, menu_height, order="F")

        # Fill background first
        menu_console.clear(fg=colors.WHITE, bg=colors.BLACK)

        # Draw border manually to avoid order issues
        # Top and bottom borders
        for x in range(actual_width):
            menu_console.ch[x, 0] = (
                ord("─")
                if 0 < x < actual_width - 1
                else (ord("┌") if x == 0 else ord("┐"))
            )
            menu_console.ch[x, menu_height - 1] = (
                ord("─")
                if 0 < x < actual_width - 1
                else (ord("└") if x == 0 else ord("┘"))
            )
            menu_console.fg[x, 0] = colors.WHITE
            menu_console.fg[x, menu_height - 1] = colors.WHITE

        # Left and right borders
        for y in range(1, menu_height - 1):
            menu_console.ch[0, y] = ord("│")
            menu_console.ch[actual_width - 1, y] = ord("│")
            menu_console.fg[0, y] = colors.WHITE
            menu_console.fg[actual_width - 1, y] = colors.WHITE

        # Draw title
        if menu_height > 3:
            title_y = 1  # Title is on the second line of the menu_console (index 1)

            if isinstance(self, InventoryMenu):
                player = (
                    self.controller.model.player
                )  # Known to be WastoidActor for player
                used_space = player.get_used_inventory_space()
                total_slots = player.inventory_slots

                current_x = 1  # Start drawing at column 1 (content area start)
                # Max content column index is actual_width - 2
                max_content_x = actual_width - 2

                # 1. Print "Inventory: " (self.title for InventoryMenu is "Inventory")
                prefix_text = f"{self.title}: "
                if current_x <= max_content_x:
                    chars_to_print = min(
                        len(prefix_text), (max_content_x - current_x + 1)
                    )
                    menu_console.print(
                        current_x,
                        title_y,
                        prefix_text[:chars_to_print],
                        fg=colors.YELLOW,
                    )
                    current_x += chars_to_print

                # 2. Draw the color bar using individual slot colors
                slot_colors_list = player.get_inventory_slot_colors()
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
                    chars_to_print = min(
                        len(suffix_text), (max_content_x - current_x + 1)
                    )
                    menu_console.print(
                        current_x,
                        title_y,
                        suffix_text[:chars_to_print],
                        fg=colors.YELLOW,
                    )
            else:
                # For other menus, use print_box for centered title within content area
                # Content area for title: x from 1 to actual_width-2
                menu_console.print_box(
                    1,
                    title_y,
                    actual_width - 2,
                    1,
                    self.title,
                    fg=colors.YELLOW,
                    alignment=2,
                )

        # Draw separator line
        if actual_width > 2 and menu_height > 3:
            for x in range(1, actual_width - 1):
                menu_console.ch[x, 2] = ord("─")
                menu_console.fg[x, 2] = colors.WHITE
        # Draw options
        y_offset = 3
        for option in self.options:
            if y_offset >= menu_height - 1:
                break  # Don't draw beyond menu bounds

            if option.key is not None:
                option_text = f"({option.key}) {option.text}"
            else:
                option_text = option.text

            # Truncate text if it's too long
            max_text_width = actual_width - 4  # Leave room for borders and padding
            if len(option_text) > max_text_width:
                option_text = option_text[: max_text_width - 3] + "..."

            menu_console.print(2, y_offset, option_text, fg=option.color)
            y_offset += 1

        # Blit to main console
        menu_console.blit(console, menu_x, menu_y)


class HelpMenu(Menu):
    """Menu showing all available commands."""

    def __init__(self, controller: Controller) -> None:
        super().__init__("Help - Available Commands", controller, width=60)

    def populate_options(self) -> None:
        """Populate help options."""
        help_items = [
            ("Movement", "Arrow keys or numpad"),
            ("Inventory", "I - Open inventory menu"),
            ("Get Items", "G - Pick up items from ground/corpses"),
            ("Quit", "Q or Escape - Quit game"),
            ("Help", "? or H - Show this help menu"),
        ]

        # Add sections without keys (just for display)
        for command, description in help_items:
            self.add_option(
                MenuOption(
                    key=None,
                    text=f"{command:<12} {description}",
                    enabled=False,
                    color=colors.WHITE,
                )
            )


class InventoryMenu(Menu):
    """Menu for managing the player's inventory."""

    def __init__(self, controller: Controller) -> None:
        super().__init__("Inventory", controller)

    def populate_options(self) -> None:
        """Populate inventory options."""
        player = self.controller.model.player
        self.options.clear()  # Ensure options are cleared at the start
        # The title "Inventory" is set in __init__.
        # The dynamic part (bar, X/Y used) is now handled in Menu.render.

        # Use letters a-z for inventory items
        letters = string.ascii_lowercase
        letter_idx = 0

        # This list will hold MenuOption objects for each display line
        # up to player.inventory_slots.
        display_lines: list[MenuOption] = []

        if not player.inventory:
            display_lines.append(
                MenuOption(
                    key=None, text="(inventory empty)", enabled=False, color=colors.GREY
                )
            )
            self.options = display_lines
            return

        # Iterate through the actual items/conditions in inventory
        for entity_in_slot in player.inventory:
            text_to_display: str
            action_to_take: Callable[[], None] | None = None
            option_color: colors.Color
            is_enabled: bool
            force_color: bool = False
            current_key: str | None = None

            if isinstance(entity_in_slot, Item):
                item: Item = entity_in_slot
                equipped_marker = ""
                if item.equippable and player.equipped_weapon == item:
                    equipped_marker = " (equipped)"

                size_display = (
                    ""  # This will hold the graphical slot indicators or (Tiny)
                )
                item_display_color = colors.WHITE  # Default for items

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
                is_enabled = True  # Assume items are generally usable/selectable

                # Only assign keys to actual items that can be interacted with
                # Conditions are listed but not keyed for action from this menu.

                if letter_idx < len(letters):
                    current_key = letters[letter_idx]
                    letter_idx += 1
                action_to_take = functools.partial(self._use_item, item)

            elif isinstance(entity_in_slot, Condition):
                condition: Condition = entity_in_slot
                text_to_display = (
                    f"{condition.name} █"  # Add slot indicator for conditions
                )
                option_color = condition.display_color
                force_color = True
                is_enabled = False  # Conditions are not directly "used"
                action_to_take = None
            else:
                # Should not happen with current game structure
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
        player = self.controller.model.player

        if hasattr(item, "equippable") and item.equippable:
            # Equip/unequip weapon
            if player.equipped_weapon == item:
                player.equipped_weapon = None
                self.controller.message_log.add_message(
                    f"You unequip {item.name}.", colors.WHITE
                )
            else:
                # Unequip current weapon first if any
                if player.equipped_weapon:
                    old_weapon = player.equipped_weapon.name
                    self.controller.message_log.add_message(
                        f"You unequip {old_weapon}.", colors.GREY
                    )

                player.equipped_weapon = item
                self.controller.message_log.add_message(
                    f"You equip {item.name}.", colors.GREEN
                )
        else:
            # Use consumable item (placeholder for now)
            self.controller.message_log.add_message(
                f"You use {item.name}.", colors.WHITE
            )


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
        return self.controller.model.get_pickable_items_at_location(
            self.location[0], self.location[1]
        )

    def _pickup_item(self, item: Item) -> None:
        """Pickup an item and add it to player inventory."""
        player = self.controller.model.player
        # Check if player has inventory space using the new size-aware method
        if not player.can_add_to_inventory(item):
            self.controller.message_log.add_message(
                "Your inventory is full!", colors.RED
            )
            return

        # Remove item from dead entity
        for entity in self.controller.model.entities:
            if isinstance(entity, Actor) and not entity.is_alive():
                if item in entity.inventory:
                    entity.inventory.remove(item)
                    break

                if entity.equipped_weapon == item:
                    entity.equipped_weapon = None
                    break

        # Add to player inventory
        player.inventory.append(item)
        self.controller.message_log.add_message(
            f"You pick up {item.name}.", colors.WHITE
        )


class MenuSystem:
    """Manages the menu system for the game."""

    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.active_menus: list[Menu] = []

    def show_menu(self, menu: Menu) -> None:
        """Show a menu, adding it to the active menu stack."""
        menu.show()
        self.active_menus.append(menu)

    def hide_current_menu(self) -> None:
        """Hide the currently active menu."""
        if self.active_menus:
            menu = self.active_menus.pop()
            menu.hide()

    def hide_all_menus(self) -> None:
        """Hide all active menus."""
        while self.active_menus:
            self.hide_current_menu()

    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle input for the active menu. Returns True if event was consumed."""
        if self.active_menus:
            # Handle input for the topmost menu
            menu = self.active_menus[-1]
            consumed = menu.handle_input(event)

            # Remove menu from stack if it was hidden
            if not menu.is_active and menu in self.active_menus:
                self.active_menus.remove(menu)

            return consumed
        return False

    def render(self, console: Console) -> None:
        """Render all active menus."""
        for menu in self.active_menus:
            menu.render(console)

    def has_active_menus(self) -> bool:
        """Check if there are any active menus."""
        return len(self.active_menus) > 0

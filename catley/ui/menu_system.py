"""
Menu system for the Catley game using overlays.
"""

from __future__ import annotations

import abc
import functools
import string
from typing import TYPE_CHECKING, cast

import tcod
import tcod.constants
import tcod.event
from tcod.console import Console

from catley import colors
from catley.game import range_system
from catley.game.actions import AttackAction
from catley.game.conditions import Condition
from catley.game.enums import ItemSize
from catley.game.items.item_core import Item

if TYPE_CHECKING:
    from collections.abc import Callable

    from catley.controller import Controller
    from catley.game.actors import Actor


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
        menu_console.clear(
            fg=cast("tuple[int, int, int]", colors.WHITE),
            bg=cast("tuple[int, int, int]", colors.BLACK),
        )

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
                player = self.controller.gw.player  # Player is an Actor
                used_space = player.inventory.get_used_inventory_slots()
                total_slots = player.inventory.total_inventory_slots

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
                    chars_to_print = min(
                        len(suffix_text), (max_content_x - current_x + 1)
                    )
                    menu_console.print(
                        x=current_x,
                        y=title_y,
                        text=suffix_text[:chars_to_print],
                        fg=colors.YELLOW,
                    )  # type: ignore[no-matching-overload]
            else:
                # For other menus, use print_box for centered title within content area
                # Content area for title: x from 1 to actual_width-2
                menu_console.print(
                    x=1,
                    y=title_y,
                    text=self.title,
                    width=actual_width - 2,
                    height=1,
                    fg=colors.YELLOW,
                    alignment=tcod.constants.CENTER,
                )  # type: ignore[no-matching-overload]

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
            ("Select Actor", "Left-click to select (shows quick actions)"),
            ("Target Menu", "T - Full targeting options for selected actor"),
            ("Inventory", "I - Open inventory menu"),
            ("Get Items", "G - Pick up items from ground/corpses"),
            ("Quit", "Q or Escape - Quit game"),
            ("Help", "? - Show this help menu"),
        ]

        # Add sections without keys (just for display)
        for command, description in help_items:
            self.add_option(
                MenuOption(
                    key=None,
                    text=f"{command:<15} {description}",
                    enabled=False,
                    color=colors.WHITE,
                )
            )


class InventoryMenu(Menu):
    """Menu for managing the player's inventory."""

    def __init__(self, controller: Controller) -> None:
        super().__init__("Inventory", controller)
        self.inventory = controller.gw.player.inventory

    def populate_options(self) -> None:
        """Populate inventory options."""
        self.options.clear()  # Ensure options are cleared at the start
        # The title "Inventory" is set in __init__.
        # The dynamic part (bar, X/Y used) is now handled in Menu.render.

        # Use letters a-z for inventory items
        letters = string.ascii_lowercase
        letter_idx = 0

        # This list will hold MenuOption objects for each display line
        # up to player.inventory_slots.
        display_lines: list[MenuOption] = []

        if self.inventory.is_empty():
            display_lines.append(
                MenuOption(
                    key=None, text="(inventory empty)", enabled=False, color=colors.GREY
                )
            )
            self.options = display_lines
            return

        # Iterate through the actual items/conditions in inventory
        for entity_in_slot in self.inventory:
            text_to_display: str
            action_to_take: Callable[[], None] | None = None
            option_color: colors.Color
            is_enabled: bool
            force_color: bool = False
            current_key: str | None = None

            if isinstance(entity_in_slot, Item):
                item: Item = entity_in_slot
                equipped_marker = ""
                if item.equippable and self.inventory.equipped_weapon == item:
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
        player = self.controller.gw.player

        if item.equippable:
            # Equip/unequip weapon
            if player.inventory.equipped_weapon == item:
                player.inventory.unequip_weapon()
                self.controller.message_log.add_message(
                    f"You unequip {item.name}.", colors.WHITE
                )
            else:
                # Unequip current weapon first if any
                if player.inventory.equipped_weapon:
                    old_weapon = player.inventory.equipped_weapon.name
                    self.controller.message_log.add_message(
                        f"You unequip {old_weapon}.", colors.GREY
                    )

                player.inventory.equip_weapon(item)
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


class QuickActionBar(Menu):
    """
    Small action bar that appears when selecting an actor,
    showing most common actions."""

    def __init__(self, controller: Controller, target_actor: Actor) -> None:
        super().__init__(f"{target_actor.name}", controller, width=35, max_height=10)
        self.target_actor = target_actor
        self.distance = range_system.calculate_distance(
            controller.gw.player.x,
            controller.gw.player.y,
            target_actor.x,
            target_actor.y,
        )

    def populate_options(self) -> None:
        """Populate quick actions - only the most common 2-3 actions."""
        player = self.controller.gw.player

        # Show basic stats first (no key, just info)
        if self.target_actor.health:
            hp_text = (
                f"HP: {self.target_actor.health.hp}/{self.target_actor.health.max_hp}"
            )
            if self.target_actor.health.ap > 0:
                hp_text += f" | AP: {self.target_actor.health.ap}"
            self.add_option(
                MenuOption(key=None, text=hp_text, enabled=False, color=colors.CYAN)
            )

        # Separator
        self.add_option(
            MenuOption(key=None, text="─" * 25, enabled=False, color=colors.GREY)
        )

        # Quick attack action
        if self.distance == 1:
            # Melee range - show primary melee attack
            primary_weapon = player.inventory.equipped_weapon
            if primary_weapon and primary_weapon.melee_attack:
                weapon_name = primary_weapon.name
            else:
                weapon_name = "fists"

            action_func = functools.partial(self._perform_quick_attack)
            self.add_option(
                MenuOption(
                    key="1",
                    text=f"Attack with {weapon_name}",
                    action=action_func,
                    color=colors.WHITE,
                )
            )
        else:
            # Ranged - show primary ranged attack if possible
            primary_weapon = player.inventory.equipped_weapon
            if (
                primary_weapon
                and primary_weapon.ranged_attack
                and primary_weapon.ranged_attack.current_ammo > 0
            ):
                # Check if we can actually shoot
                range_category = range_system.get_range_category(
                    self.distance, primary_weapon
                )
                range_mods = range_system.get_range_modifier(
                    primary_weapon, range_category
                )
                has_los = range_system.has_line_of_sight(
                    self.controller.gw.game_map,
                    player.x,
                    player.y,
                    self.target_actor.x,
                    self.target_actor.y,
                )

                if range_mods is not None and has_los:
                    ammo_display = (
                        f"[{primary_weapon.ranged_attack.current_ammo}/"
                        f"{primary_weapon.ranged_attack.max_ammo}]"
                    )
                    action_func = functools.partial(self._perform_quick_attack)
                    self.add_option(
                        MenuOption(
                            key="1",
                            text=f"Shoot {primary_weapon.name} {ammo_display}",
                            action=action_func,
                            color=colors.WHITE,
                        )
                    )
                else:
                    # Can't shoot - show why
                    reason = "no line of sight" if not has_los else "out of range"
                    self.add_option(
                        MenuOption(
                            key=None,
                            text=f"Can't shoot: {reason}",
                            enabled=False,
                            color=colors.RED,
                        )
                    )

        # Quick reload if needed and possible
        primary_weapon = player.inventory.equipped_weapon
        if (
            primary_weapon
            and primary_weapon.ranged_attack
            and primary_weapon.ranged_attack.current_ammo
            < primary_weapon.ranged_attack.max_ammo
        ):
            # Check if player has compatible ammo
            has_ammo = any(
                ammo_item.ammo
                and ammo_item.ammo.ammo_type == primary_weapon.ranged_attack.ammo_type
                for ammo_item in player.inventory
            )

            if has_ammo:
                action_func = functools.partial(self._perform_quick_reload)
                self.add_option(
                    MenuOption(
                        key="r",
                        text=f"Reload {primary_weapon.name}",
                        action=action_func,
                        color=colors.GREEN,
                    )
                )

        # Always show "More options" to open full target menu
        action_func = functools.partial(self._open_full_menu)
        self.add_option(
            MenuOption(
                key="t",
                text="More options... (T)",
                action=action_func,
                color=colors.YELLOW,
            )
        )

    def _perform_quick_attack(self) -> None:
        """Perform attack with primary weapon."""
        player = self.controller.gw.player
        attack_action = AttackAction(self.controller, player, self.target_actor)
        self.controller.queue_action(attack_action)

    def _perform_quick_reload(self) -> None:
        """Reload primary weapon."""
        player = self.controller.gw.player
        primary_weapon = player.inventory.equipped_weapon

        if primary_weapon and primary_weapon.ranged_attack:
            # Find compatible ammo
            ammo_item = None
            for item in player.inventory:
                if (
                    item.ammo
                    and item.ammo.ammo_type == primary_weapon.ranged_attack.ammo_type
                ):
                    ammo_item = item
                    break

            if ammo_item:
                player.inventory.remove_from_inventory(ammo_item)
                primary_weapon.ranged_attack.current_ammo = (
                    primary_weapon.ranged_attack.max_ammo
                )
                self.controller.message_log.add_message(
                    f"Reloaded {primary_weapon.name}", colors.GREEN
                )
                # Refresh the quick bar to show updated ammo
                self.populate_options()

    def _open_full_menu(self) -> None:
        """Open the full target menu."""
        # Hide the quick bar first
        self.hide()
        # Import here to avoid circular imports
        from catley.ui.ui_commands import OpenTargetMenuUICommand

        command = OpenTargetMenuUICommand(
            self.controller, target_actor=self.target_actor
        )
        command.execute()


class TargetMenu(Menu):
    """Menu for selecting actions to perform on a target actor or location."""

    def __init__(
        self,
        controller: Controller,
        target_actor: Actor | None = None,
        target_location: tuple[int, int] | None = None,
    ) -> None:
        if target_actor:
            title = f"Target: {target_actor.name}"
            self.target_actor = target_actor
            self.target_location = (target_actor.x, target_actor.y)
            self.distance = range_system.calculate_distance(
                controller.gw.player.x,
                controller.gw.player.y,
                target_actor.x,
                target_actor.y,
            )
        elif target_location:
            title = f"Target: Location ({target_location[0]}, {target_location[1]})"
            self.target_actor = None
            self.target_location = target_location
            self.distance = range_system.calculate_distance(
                controller.gw.player.x,
                controller.gw.player.y,
                target_location[0],
                target_location[1],
            )
        else:
            raise ValueError("Either target_actor or target_location must be provided")

        super().__init__(title, controller, width=50)

    def populate_options(self) -> None:
        """Populate targeting options based on the target and player capabilities."""
        player = self.controller.gw.player
        target_x, target_y = self.target_location

        # Check line of sight first for ranged options
        has_los = range_system.has_line_of_sight(
            self.controller.gw.game_map, player.x, player.y, target_x, target_y
        )

        # Melee actions (adjacent only)
        if self.distance == 1 and self.target_actor:  # Can only melee attack actors
            melee_added = False
            for item, slot_index in player.inventory.get_equipped_items():
                if item.melee_attack:
                    slot_name = player.inventory.get_slot_display_name(slot_index)
                    key = chr(ord("a") + len(self.options))  # Sequential letters
                    action_func = functools.partial(
                        self._perform_melee_attack, item, self.target_actor
                    )
                    self.add_option(
                        MenuOption(
                            key=key,
                            text=f"Attack with {item.name} ({slot_name})",
                            action=action_func,
                            color=colors.WHITE,
                        )
                    )
                    melee_added = True

            # Always allow unarmed attack if no melee weapons equipped
            if not melee_added:
                from catley.game.items.item_types import FISTS_TYPE

                fists = FISTS_TYPE.create()
                action_func = functools.partial(
                    self._perform_melee_attack, fists, self.target_actor
                )
                self.add_option(
                    MenuOption(
                        key="1",
                        text="Attack (unarmed)",
                        action=action_func,
                        color=colors.WHITE,
                    )
                )

        # Ranged actions (if line of sight)
        if has_los and self.target_actor:  # Can only shoot at actors for now
            for item, slot_index in player.inventory.get_equipped_items():
                if item.ranged_attack:
                    range_category = range_system.get_range_category(
                        self.distance, item
                    )
                    range_mods = range_system.get_range_modifier(item, range_category)

                    if range_mods is None:  # Out of range
                        continue

                    slot_name = player.inventory.get_slot_display_name(slot_index)
                    key = chr(
                        ord("a") + len(self.options)
                    )  # Sequential letters: a, b, c, d...
                    if item.ranged_attack.current_ammo > 0:
                        ammo_display = (
                            f"[{item.ranged_attack.current_ammo}/"
                            f"{item.ranged_attack.max_ammo}]"
                        )
                        action_func = functools.partial(
                            self._perform_ranged_attack, item, self.target_actor
                        )
                        self.add_option(
                            MenuOption(
                                key=key,
                                text=f"Shoot {item.name} ({slot_name}) {ammo_display}",
                                action=action_func,
                                color=colors.WHITE,
                            )
                        )
                    else:
                        # Show out of ammo option but disabled
                        self.add_option(
                            MenuOption(
                                key=None,
                                text=f"Shoot {item.name} ({slot_name}) [OUT OF AMMO]",
                                enabled=False,
                                color=colors.RED,
                            )
                        )

        # Reload actions
        self._add_reload_options()

        # If no actions available, show a message
        if not self.options:
            self.add_option(
                MenuOption(
                    key=None,
                    text="No actions available",
                    enabled=False,
                    color=colors.GREY,
                )
            )

    def _add_reload_options(self) -> None:
        """Add reload actions for partially loaded weapons."""
        player = self.controller.gw.player

        for item, slot_index in player.inventory.get_equipped_items():
            ranged_attack = item.ranged_attack
            if ranged_attack and ranged_attack.current_ammo < ranged_attack.max_ammo:
                # Check if player has compatible ammo
                has_ammo = any(
                    isinstance(ammo_item, Item)
                    and ammo_item.ammo
                    and ammo_item.ammo.ammo_type == ranged_attack.ammo_type
                    for ammo_item in player.inventory
                )
                if has_ammo:
                    slot_name = player.inventory.get_slot_display_name(slot_index)
                    reload_key = chr(
                        ord("a") + len(self.options)
                    )  # Continue sequential letters
                    action_func = functools.partial(
                        self._perform_reload, item, slot_index
                    )
                    self.add_option(
                        MenuOption(
                            key=reload_key,
                            text=f"Reload {item.name} ({slot_name})",
                            action=action_func,
                            color=colors.GREEN,
                        )
                    )

    def _perform_melee_attack(self, weapon_item, target_actor) -> None:
        """Perform a melee attack with the specified weapon."""
        # Temporarily equip the weapon if it's not the player's fists
        player = self.controller.gw.player

        if weapon_item.name != "Fists":
            # Find which slot this weapon is in and make sure it's equipped
            for item, slot_index in player.inventory.get_equipped_items():
                if item == weapon_item:
                    break

        # Perform the attack using existing AttackAction
        attack_action = AttackAction(self.controller, player, target_actor)
        self.controller.queue_action(attack_action)

    def _perform_ranged_attack(self, weapon_item, target_actor) -> None:
        """Perform a ranged attack with the specified weapon."""
        # For now, use the existing AttackAction which will automatically
        # choose ranged attack based on distance
        player = self.controller.gw.player
        attack_action = AttackAction(self.controller, player, target_actor)
        self.controller.queue_action(attack_action)

    def _perform_reload(self, weapon_item, slot_index) -> None:
        """Reload the specified weapon."""
        player = self.controller.gw.player
        ranged_attack = weapon_item.ranged_attack

        # Find compatible ammo in inventory
        ammo_item = None
        for item in player.inventory:
            if item.ammo and item.ammo.ammo_type == ranged_attack.ammo_type:
                ammo_item = item
                break

        if ammo_item:
            # Remove ammo from inventory and reload weapon
            player.inventory.remove_from_inventory(ammo_item)
            ranged_attack.current_ammo = ranged_attack.max_ammo
            slot_name = player.inventory.get_slot_display_name(slot_index)
            self.controller.message_log.add_message(
                f"Reloaded {weapon_item.name} ({slot_name})", colors.GREEN
            )
        else:
            self.controller.message_log.add_message(
                f"No {ranged_attack.ammo_type} ammo available!", colors.RED
            )

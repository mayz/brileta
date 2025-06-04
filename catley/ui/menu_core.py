import abc
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import tcod
import tcod.constants
from tcod.console import Console

from catley import colors

if TYPE_CHECKING:
    from catley.controller import Controller


class MenuSystem:
    """Manages the menu system for the game."""

    def __init__(self, controller: "Controller") -> None:
        self.controller = controller
        self.active_menus: list[Menu] = []

    def show_menu(self, menu: "Menu") -> None:
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
        controller: "Controller",
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
            self.render_title(menu_console, title_y, actual_width)

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

    def render_title(self, menu_console, title_y, actual_width):
        """Render the menu title. Override in subclasses for custom behavior."""
        # Default implementation - simple centered title
        menu_console.print(
            x=1,
            y=title_y,
            text=self.title,
            width=actual_width - 2,
            height=1,
            fg=colors.YELLOW,
            alignment=tcod.constants.CENTER,
        )

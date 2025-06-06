"""UI Overlay System - Temporary modal UI elements.

Overlays are temporary UI that appear over the game world:
- Menus (inventory, help, pickup)
- Tooltips and status displays
- Confirmation dialogs

Use overlays for temporary UI. Use panels for persistent UI. Use modes for behavioral changes.

Quick decisions:
- "Show tooltip" -> Create Tooltip(Overlay)
- "Ask confirmation" -> Create Dialog(Overlay)
- "Show inventory" -> Use InventoryMenu(Menu)
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import tcod
import tcod.constants
from tcod.console import Console

from catley import colors

if TYPE_CHECKING:
    from catley.controller import Controller


class Overlay(abc.ABC):
    """Base class for temporary UI overlays.

    Overlays appear over the game view and can stack. They consume input
    events while active and remove themselves when hidden.

    Lifecycle: show() -> handle input/render -> hide()
    """

    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.is_active = False

    @abc.abstractmethod
    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle input. Return True if consumed (stops further processing)."""
        pass

    @abc.abstractmethod
    def render(self, console: Console) -> None:
        """Render overlay content. Called every frame."""
        pass

    def show(self) -> None:
        """Activate overlay. Called by OverlaySystem.show_overlay()."""
        self.is_active = True

    def hide(self) -> None:
        """Deactivate overlay. Triggers automatic removal from stack."""
        self.is_active = False

    def can_stack_with(self, other: Overlay) -> bool:
        """Return True if can coexist with another overlay. Override for exclusivity."""
        return True


class OverlaySystem:
    """Manages overlay stacking and input priority.

    Topmost overlay gets first input chance. Renders bottom to top.
    Auto-removes overlays when they hide themselves.
    """

    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.active_overlays: list[Overlay] = []

    def show_overlay(self, overlay: Overlay) -> None:
        """Show an overlay, adding it to the active overlay stack."""
        overlay.show()
        self.active_overlays.append(overlay)

    def hide_current_overlay(self) -> None:
        """Hide the currently active overlay."""
        if self.active_overlays:
            overlay = self.active_overlays.pop()
            overlay.hide()

    def hide_all_overlays(self) -> None:
        """Hide all active overlays."""
        while self.active_overlays:
            self.hide_current_overlay()

    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle input for the active overlays. Returns True if event was consumed."""
        if self.active_overlays:
            # Handle input for the topmost overlay
            overlay = self.active_overlays[-1]
            consumed = overlay.handle_input(event)

            # Remove overlay from stack if it was hidden
            if not overlay.is_active and overlay in self.active_overlays:
                self.active_overlays.remove(overlay)

            return consumed
        return False

    def render(self, console: Console) -> None:
        """Render all active overlays."""
        for overlay in self.active_overlays:
            overlay.render(console)

    def has_active_overlays(self) -> bool:
        """Check if there are any active overlays."""
        return len(self.active_overlays) > 0

    # Legacy methods for backward compatibility
    def show_menu(self, menu: Menu) -> None:
        """Legacy method: Show a menu overlay."""
        self.show_overlay(menu)

    def hide_current_menu(self) -> None:
        """Legacy method: Hide the currently active overlay."""
        self.hide_current_overlay()

    def hide_all_menus(self) -> None:
        """Legacy method: Hide all active overlays."""
        self.hide_all_overlays()

    def has_active_menus(self) -> bool:
        """Legacy method: Check if there are any active overlays."""
        return self.has_active_overlays()


class MenuOption:
    """Single menu choice with keyboard shortcut and action."""

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


class Menu(Overlay):
    """Structured choice interface with keyboard shortcuts.

    Provides bordered rendering and keyboard selection (a, b, c).
    Auto-handles ESC/Enter/Space to close.

    Override populate_options() to add MenuOption instances.
    """

    def __init__(
        self,
        title: str,
        controller: Controller,
        width: int = 50,
        max_height: int = 30,
    ) -> None:
        super().__init__(controller)
        self.title = title
        self.width = width
        self.max_height = max_height
        self.options: list[MenuOption] = []

    @abc.abstractmethod
    def populate_options(self) -> None:
        """Populate the menu options. Must be implemented by subclasses."""
        pass

    def add_option(self, option: MenuOption) -> None:
        """Add an option to the menu."""
        self.options.append(option)

    def show(self) -> None:
        """Show the menu."""
        super().show()
        self.populate_options()

    def hide(self) -> None:
        """Hide the menu."""
        super().hide()
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

    def render_title(
        self,
        menu_console: Console,
        title_y: int,
        actual_width: int,
    ) -> None:
        """Render the menu title."""
        # Default implementation - simple centered title
        menu_console.print(
            x=1,
            y=title_y,
            text=self.title,
            width=actual_width - 2,
            height=1,
            fg=cast("tuple[int, int, int]", colors.YELLOW),
            alignment=tcod.constants.CENTER,
        )

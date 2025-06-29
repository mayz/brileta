"""
Defines the Overlay system for temporary, modal UI elements.

Overlays are UI components that appear on top of the main `View` hierarchy and
typically consume all input while active. They are used for transient interfaces
like menus, dialog boxes, and informational pop-ups.

Key Principles:
- Overlays are managed in a stack by the `FrameManager`. Only the top-most
  overlay is interactive.
- They are distinct from `Views`, which are persistent parts of the main layout.
- An overlay is shown, handles its lifecycle, and then hides itself, at which
  point it is removed from the active stack.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import tcod.event

from catley import colors
from catley.backends.tcod.canvas import TCODConsoleCanvas
from catley.types import PixelCoord, PixelPos, RootConsoleTileCoord
from catley.view.render.canvas import Canvas

if TYPE_CHECKING:
    from catley.controller import Controller

REPEAT_PREFIX = "[Enter] Repeat:"


class Overlay(abc.ABC):
    """Base class for temporary UI overlays.

    Overlays appear over the game view and can stack. They consume input
    events while active and remove themselves when hidden.

    Lifecycle: show() -> handle input/render -> hide()
    """

    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.is_active = False
        self.is_interactive = True  # Most overlays handle input by default

    @abc.abstractmethod
    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle input. Return True if consumed (stops further processing)."""
        pass

    @abc.abstractmethod
    def draw(self) -> None:
        """Orchestrate rendering. This is the main entry point."""

    @abc.abstractmethod
    def present(self) -> None:
        """Presents the cached texture if one was created by the backend."""

    def show(self) -> None:
        """Activate overlay. Called by OverlaySystem.show_overlay()."""
        self.is_active = True

    def hide(self) -> None:
        """Deactivate overlay. Triggers automatic removal from stack."""
        self.is_active = False

    def can_stack_with(self, other: Overlay) -> bool:
        """Return True if can coexist with another overlay. Override for exclusivity."""
        return True


class TextOverlay(Overlay):
    """An overlay that renders via a TextBackend."""

    canvas: Canvas | None

    def __init__(self, controller: Controller) -> None:
        super().__init__(controller)
        self.canvas = None
        self._cached_texture: Any | None = None
        self.x_tiles, self.y_tiles, self.width, self.height = 0, 0, 0, 0
        self.pixel_width, self.pixel_height = 0, 0
        self.tile_dimensions = (
            self.controller.graphics.tile_dimensions
            if self.controller.graphics
            else (0, 0)
        )

    @abc.abstractmethod
    def _get_backend(self) -> Canvas:
        """Subclasses must implement this to provide a specific backend."""

    @abc.abstractmethod
    def _calculate_dimensions(self) -> None:
        """Subclasses must implement this to set their dimensions."""

    @abc.abstractmethod
    def draw_content(self) -> None:
        """Subclasses implement the actual drawing commands using the canvas."""

    def draw(self) -> None:
        """Orchestrate rendering. This is the main entry point."""
        if not self.is_active:
            return

        self.canvas = self._get_backend()
        self._calculate_dimensions()

        if self.width <= 0 or self.height <= 0:
            # Nothing to render when dimensions are zero.
            self.canvas = None
            self._cached_texture = None
            return

        self.canvas.configure_drawing_offset(self.x_tiles, self.y_tiles)
        self.canvas.configure_dimensions(self.pixel_width, self.pixel_height)
        self.canvas.configure_scaling(self.tile_dimensions[1])

        self.canvas.begin_frame()
        self.draw_content()
        artifact = self.canvas.end_frame()
        if artifact is not None:
            texture = self.canvas.create_texture(self.controller.graphics, artifact)
            self._cached_texture = texture

    def present(self) -> None:
        """Presents the cached texture if one was created by the backend."""
        if not self.is_active or not self._cached_texture:
            return

        # Convert pixel position to tile position for the renderer
        tile_width, tile_height = self.tile_dimensions
        width_tiles = self.pixel_width // tile_width if tile_width > 0 else 0
        height_tiles = self.pixel_height // tile_height if tile_height > 0 else 0

        self.controller.graphics.present_texture(
            self._cached_texture, self.x_tiles, self.y_tiles, width_tiles, height_tiles
        )


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

    def toggle_overlay(self, overlay: Overlay) -> None:
        """Show the overlay if hidden or hide it if already active."""
        if overlay in self.active_overlays:
            overlay.hide()
            self.active_overlays.remove(overlay)
        else:
            self.show_overlay(overlay)

    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle input for the active overlays. Returns True if event was consumed."""
        if self.active_overlays:
            # Find the topmost interactive overlay
            for overlay in reversed(self.active_overlays):
                if not overlay.is_interactive:
                    continue

                consumed = overlay.handle_input(event)

                # Remove overlay from stack if it was hidden
                if not overlay.is_active and overlay in self.active_overlays:
                    self.active_overlays.remove(overlay)

                return consumed

            # If we get here, there were only non-interactive overlays
            # Return False so input passes through to the main game
            return False

        return False

    def draw_overlays(self) -> None:
        """Draw all active overlays to their internal textures/surfaces."""
        for overlay in self.active_overlays:
            overlay.draw()

    def present_overlays(self) -> None:
        """Present the rendered textures of all active overlays to the screen."""
        for overlay in self.active_overlays:
            overlay.present()

    def has_active_overlays(self) -> bool:
        """Check if there are any active overlays."""
        return len(self.active_overlays) > 0

    def has_interactive_overlays(self) -> bool:
        """Check if there are any active, interactive overlays."""
        return any(o.is_interactive for o in self.active_overlays)

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
        action: Callable[[], None | bool] | None = None,
        enabled: bool = True,
        color: colors.Color = colors.WHITE,
        force_color: bool = False,
        data: Any | None = None,
        is_primary_action: bool = True,
    ) -> None:
        self.key = key
        self.text = text
        self.action = action
        self.enabled = enabled
        self.force_color = force_color
        self.color = color if enabled or force_color else colors.GREY
        self.data = data
        self.is_primary_action = is_primary_action


class Menu(TextOverlay):
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
        self.width_tiles = width
        self.max_height_tiles = max_height
        self.options: list[MenuOption] = []
        self._content_revision = -1
        self.hovered_option_index: int | None = None
        self.mouse_px_x: PixelCoord = 0
        self.mouse_px_y: PixelCoord = 0

    @abc.abstractmethod
    def populate_options(self) -> None:
        """Populate the menu options. Must be implemented by subclasses."""
        pass

    def add_option(self, option: MenuOption) -> None:
        """Add an option to the menu."""
        self.options.append(option)
        self._content_revision += 1

    def show(self) -> None:
        """Show the menu."""
        self.options.clear()
        self._content_revision = 0
        super().show()
        self.populate_options()

    def hide(self) -> None:
        """Hide the menu."""
        super().hide()
        self.options.clear()

    def _convert_global_mouse_to_menu_relative(
        self, global_px_x: PixelCoord, global_px_y: PixelCoord
    ) -> tuple[PixelCoord, PixelCoord]:
        """Convert global pixel coordinates to menu-relative pixel coordinates."""
        menu_px_x: PixelCoord = self.x_tiles * self.tile_dimensions[0]
        menu_px_y: PixelCoord = self.y_tiles * self.tile_dimensions[1]

        relative_px_x: PixelCoord = global_px_x - menu_px_x
        relative_px_y: PixelCoord = global_px_y - menu_px_y

        return relative_px_x, relative_px_y

    def _get_hovered_option_index(
        self, menu_relative_px_x: PixelCoord, menu_relative_px_y: PixelCoord
    ) -> int | None:
        """Determine which menu option (if any) is under the mouse cursor."""
        if not (
            0 <= menu_relative_px_x < self.pixel_width
            and 0 <= menu_relative_px_y < self.pixel_height
        ):
            return None

        tile_x: RootConsoleTileCoord = int(
            menu_relative_px_x // self.tile_dimensions[0]
        )
        tile_y: RootConsoleTileCoord = int(
            menu_relative_px_y // self.tile_dimensions[1]
        )

        if not (1 <= tile_x < self.width - 1 and 2 <= tile_y < self.height - 1):
            return None

        option_line: int = tile_y - 2

        if 0 <= option_line < len(self.options):
            return option_line

        return None

    def _update_mouse_state(
        self, global_px_x: PixelCoord, global_px_y: PixelCoord
    ) -> None:
        """Update internal mouse tracking state."""
        self.mouse_px_x, self.mouse_px_y = self._convert_global_mouse_to_menu_relative(
            global_px_x, global_px_y
        )
        self.hovered_option_index = self._get_hovered_option_index(
            self.mouse_px_x, self.mouse_px_y
        )

    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle input events for the menu. Returns True if event was consumed."""
        if not self.is_active:
            return False

        match event:
            case tcod.event.MouseMotion():
                mouse_pixel_pos: PixelPos = event.position
                mouse_px_x: PixelCoord = mouse_pixel_pos[0]
                mouse_px_y: PixelCoord = mouse_pixel_pos[1]
                self._update_mouse_state(mouse_px_x, mouse_px_y)
                return True

            case tcod.event.MouseButtonDown(button=tcod.event.MouseButton.LEFT):
                mouse_pixel_pos: PixelPos = event.position
                mouse_px_x: PixelCoord = mouse_pixel_pos[0]
                mouse_px_y: PixelCoord = mouse_pixel_pos[1]
                self._update_mouse_state(mouse_px_x, mouse_px_y)

                if (
                    self.hovered_option_index is not None
                    and self.hovered_option_index < len(self.options)
                ):
                    option = self.options[self.hovered_option_index]
                    if option.enabled and option.action:
                        result = option.action()
                        if result is not False:
                            self.hide()
                        return True
                return True

            case tcod.event.KeyDown(sym=tcod.event.KeySym.ESCAPE):
                self.hide()
                return True
            case tcod.event.KeyDown(sym=tcod.event.KeySym.SPACE):
                self.hide()
                return True
            case tcod.event.KeyDown(sym=tcod.event.KeySym.RETURN):  # Enter key
                if (
                    self.options
                    and self.options[0].text.startswith(REPEAT_PREFIX)
                    and self.options[0].action
                ):
                    self.options[0].action()
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

    def _get_backend(self) -> Canvas:
        """Lazily initializes and returns a TCODConsoleCanvas for Phase 1."""
        if self.canvas is None:
            self.canvas = TCODConsoleCanvas(self.controller.graphics, transparent=False)
        return self.canvas

    def draw(self) -> None:
        super().draw()

    def present(self) -> None:
        super().present()

    def _calculate_dimensions(self) -> None:
        """Calculate and set the menu's final dimensions in tiles."""
        required_content_lines = 1 + 1 + len(self.options) + 1
        total_height_tiles = required_content_lines + 2
        menu_height_tiles = min(total_height_tiles, self.max_height_tiles)
        menu_height_tiles = max(menu_height_tiles, 5)

        min_width_tiles = max(len(self.title) + 4, 20)
        menu_width_tiles = max(self.width_tiles, min_width_tiles)

        self.tile_dimensions = self.controller.graphics.tile_dimensions
        self.width = menu_width_tiles
        self.height = menu_height_tiles

        self.pixel_width = self.width * self.tile_dimensions[0]
        self.pixel_height = self.height * self.tile_dimensions[1]

        self.x_tiles = (self.controller.graphics.console_width_tiles - self.width) // 2
        self.y_tiles = (
            self.controller.graphics.console_height_tiles - self.height
        ) // 2

    def draw_content(self) -> None:
        """Render the menu content using only the Canvas interface."""
        assert self.canvas is not None

        # Draw the frame without a title.  The backend's draw_frame method no
        # longer accepts a title parameter, so the render_title hook handles
        # drawing header text explicitly.
        self.canvas.draw_frame(
            tile_x=0,
            tile_y=0,
            width=self.width,
            height=self.height,
            fg=colors.WHITE,
            bg=colors.BLACK,
        )

        interior_px_x: PixelCoord = self.tile_dimensions[0]
        interior_px_y: PixelCoord = self.tile_dimensions[1]
        interior_width_px: PixelCoord = (self.width - 2) * self.tile_dimensions[0]
        interior_height_px: PixelCoord = (self.height - 2) * self.tile_dimensions[1]

        self.canvas.draw_rect(
            pixel_x=interior_px_x,
            pixel_y=interior_px_y,
            width=interior_width_px,
            height=interior_height_px,
            color=colors.BLACK,
            fill=True,
        )

        # Draw the title
        self.render_title()

        # Draw options with hover highlighting
        y_offset_tiles: RootConsoleTileCoord = 2
        for i, option in enumerate(self.options):
            if y_offset_tiles >= self.height - 1:
                break

            is_hovered = self.hovered_option_index == i and option.enabled

            if is_hovered:
                hover_bg_px_x: PixelCoord = self.tile_dimensions[0]
                hover_bg_px_y: PixelCoord = self.tile_dimensions[1] * y_offset_tiles
                hover_bg_width_px: PixelCoord = (self.width - 2) * self.tile_dimensions[
                    0
                ]
                hover_bg_height_px: PixelCoord = self.tile_dimensions[1]

                self.canvas.draw_rect(
                    pixel_x=hover_bg_px_x,
                    pixel_y=hover_bg_px_y,
                    width=hover_bg_width_px,
                    height=hover_bg_height_px,
                    color=colors.DARK_GREY,
                    fill=True,
                )

            option_text = f"({option.key}) {option.text}" if option.key else option.text

            text_color = option.color
            if is_hovered:
                r, g, b = text_color
                text_color = (
                    min(255, r + 40),
                    min(255, g + 40),
                    min(255, b + 40),
                )

            text_px_x: PixelCoord = self.tile_dimensions[0] * 2
            text_px_y: PixelCoord = self.tile_dimensions[1] * y_offset_tiles

            self.canvas.draw_text(
                pixel_x=text_px_x,
                pixel_y=text_px_y,
                text=option_text,
                color=text_color,
            )
            y_offset_tiles += 1

    def render_title(self) -> None:
        """Renders the menu's title, centered, in the header area."""
        assert self.canvas is not None

        title_width_tiles = len(self.title)
        center_x_tile = (self.width - title_width_tiles) // 2

        self.canvas.draw_text(
            pixel_x=center_x_tile * self.tile_dimensions[0],
            pixel_y=0,
            text=self.title,
            color=colors.YELLOW,
        )

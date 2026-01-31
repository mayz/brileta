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

from catley import colors, config
from catley.backends.pillow.canvas import PillowImageCanvas
from catley.types import PixelCoord, PixelPos
from catley.view.render.canvas import Canvas
from catley.view.ui.selectable_list import (
    LayoutMode,
    SelectableListRenderer,
    SelectableRow,
)

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

    def __init__(self, controller: Controller) -> None:
        super().__init__(controller)
        self.canvas = self._get_backend()
        self._cached_texture: Any | None = None
        self._dirty = True
        # Set True for overlays with content that changes every frame (e.g. live stats).
        self.always_dirty = False
        self.x_tiles, self.y_tiles, self.width, self.height = 0, 0, 0, 0
        self.pixel_width, self.pixel_height = 0, 0
        self.tile_dimensions = (
            self.controller.graphics.tile_dimensions
            if self.controller.graphics
            else (0, 0)
        )
        # Unique cache key for this overlay instance to prevent texture sharing
        self._cache_key = str(id(self))

    @abc.abstractmethod
    def _get_backend(self) -> Canvas:
        """Subclasses must implement this to provide a specific backend."""

    @abc.abstractmethod
    def _calculate_dimensions(self) -> None:
        """Subclasses must implement this to set their dimensions."""

    @abc.abstractmethod
    def draw_content(self) -> None:
        """Subclasses implement the actual drawing commands using the canvas."""

    def invalidate(self) -> None:
        """Mark the overlay as dirty so it will be re-rendered next frame.

        Note: We don't release the cached texture here because it's owned by
        the FBO cache and will be reused on the next render.
        """
        self._dirty = True

    def show(self) -> None:
        """Activate overlay and mark it for redraw."""
        super().show()
        self.invalidate()

    def draw(self) -> None:
        """Orchestrate rendering. This is the main entry point."""
        if not self.is_active:
            return

        if (
            not self.always_dirty
            and not self._dirty
            and self._cached_texture is not None
        ):
            return

        self._calculate_dimensions()

        if self.width <= 0 or self.height <= 0:
            # Nothing to render when dimensions are zero.
            # Don't release texture - it's owned by the FBO cache and may be
            # reused if dimensions become non-zero again.
            self._cached_texture = None
            self._dirty = False
            return

        self.canvas.configure_drawing_offset(self.x_tiles, self.y_tiles)
        self.canvas.configure_dimensions(self.pixel_width, self.pixel_height)
        self.canvas.configure_scaling(self.tile_dimensions[1])

        self.canvas.begin_frame()
        self.draw_content()
        artifact = self.canvas.end_frame()
        if artifact is not None:
            # Note: We don't release the old texture because it's owned by
            # the FBO cache. The same texture will be returned and reused.
            # Use cache key for unique texture caching per overlay
            create_with_key = getattr(
                self.canvas, "create_texture_with_cache_key", None
            )
            if callable(create_with_key):
                texture = create_with_key(
                    self.controller.graphics, artifact, self._cache_key
                )
            else:
                texture = self.canvas.create_texture(self.controller.graphics, artifact)
            self._cached_texture = texture
            self._dirty = False

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

                if consumed and isinstance(overlay, TextOverlay):
                    overlay.invalidate()

                return consumed

            # If we get here, there were only non-interactive overlays
            # Return False so input passes through to the main game
            return False

        return False

    def draw_overlays(self) -> None:
        """Draw all active overlays to their internal textures/surfaces."""
        # Remove any overlays that have been hidden
        self.active_overlays = [
            overlay for overlay in self.active_overlays if overlay.is_active
        ]

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

    def invalidate_all(self) -> None:
        """Invalidate all active text overlays."""
        for overlay in self.active_overlays:
            if isinstance(overlay, TextOverlay):
                overlay.invalidate()

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
        prefix: str = "",
        prefix_color: colors.Color | None = None,
        suffix: str = "",
        suffix_color: colors.Color | None = None,
        prefix_segments: list[tuple[str, colors.Color]] | None = None,
    ) -> None:
        self.key = key
        self.text = text
        self.action = action
        self.enabled = enabled
        self.force_color = force_color
        self.color = color if enabled or force_color else colors.GREY
        self.data = data
        self.is_primary_action = is_primary_action
        self.prefix = prefix
        self.prefix_color = prefix_color if prefix_color else self.color
        self.suffix = suffix
        self.suffix_color = suffix_color if suffix_color else self.color
        # Multi-segment prefix support for different colored segments.
        # If provided, takes precedence over prefix/prefix_color.
        # Format: [("[1]  ", YELLOW), ("[W] ", RED)]
        self.prefix_segments = prefix_segments


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
        self.width_chars = width  # Width in characters (not tiles)
        self.max_height_lines = max_height  # Max height in text lines
        self.options: list[MenuOption] = []
        self._content_revision = -1
        self.hovered_option_index: int | None = None
        self.mouse_px_x: PixelCoord = 0
        self.mouse_px_y: PixelCoord = 0
        # Font metrics - will be set in _calculate_dimensions
        self._line_height = 0
        self._char_width = 0
        # Create list renderer for option rendering (canvas created by super().__init__)
        assert isinstance(self.canvas, PillowImageCanvas)
        self._list_renderer = SelectableListRenderer(self.canvas, LayoutMode.KEYCAP)

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

    def _is_point_inside_menu(
        self, global_px_x: PixelCoord, global_px_y: PixelCoord
    ) -> bool:
        """Check if a global pixel position is inside the menu bounds."""
        relative_px_x, relative_px_y = self._convert_global_mouse_to_menu_relative(
            global_px_x, global_px_y
        )
        return (
            0 <= relative_px_x < self.pixel_width
            and 0 <= relative_px_y < self.pixel_height
        )

    def _update_mouse_state(
        self, global_px_x: PixelCoord, global_px_y: PixelCoord
    ) -> None:
        """Update internal mouse tracking state using the list renderer's hit areas."""
        self.mouse_px_x, self.mouse_px_y = self._convert_global_mouse_to_menu_relative(
            global_px_x, global_px_y
        )
        # Use the renderer's hit area tracking for hover detection.
        # The renderer's hit areas are in menu-relative coordinates.
        self._list_renderer.update_hover_from_pixel(
            int(self.mouse_px_x), int(self.mouse_px_y)
        )
        self.hovered_option_index = self._list_renderer.hovered_index

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

                # Close menu if clicking outside its bounds
                if not self._is_point_inside_menu(mouse_px_x, mouse_px_y):
                    self.hide()
                    return True

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
                return True  # Consume all KeyDown input while menu is active

            case tcod.event.KeyUp():
                return False  # Let KeyUp events pass through to modes

        return False

    def _get_backend(self) -> Canvas:
        """Return a PillowImageCanvas for crisp VGA font rendering."""
        return PillowImageCanvas(
            self.controller.graphics,
            font_path=config.UI_FONT_PATH,
            font_size=config.MENU_FONT_SIZE,
            line_spacing=config.MENU_LINE_SPACING,
        )

    def draw(self) -> None:
        super().draw()

    def present(self) -> None:
        super().present()

    def _calculate_dimensions(self) -> None:
        """Calculate menu dimensions in pixels using font metrics."""
        # Get font metrics from the PillowImageCanvas
        assert isinstance(self.canvas, PillowImageCanvas)
        # Reduce line height by 1px to close gaps between border glyphs
        self._line_height = self.canvas.get_effective_line_height() - 1
        # Get character width from a space character
        self._char_width, _, _ = self.canvas.get_text_metrics(" ")

        # Calculate dimensions in text lines
        # Layout: border(1) + title(1) + options + border(1)
        required_content_lines = 1 + 1 + len(self.options)
        total_lines = required_content_lines + 1  # +1 for bottom border
        menu_height_lines = min(total_lines, self.max_height_lines)
        menu_height_lines = max(menu_height_lines, 5)

        # Calculate minimum width needed for title with padding
        min_width_chars = max(len(self.title) + 4, 20)

        # Also ensure width accommodates the longest option text.
        # Account for: border + padding + key "(x) " + text + padding + border
        for option in self.options:
            option_text = f"({option.key}) {option.text}" if option.key else option.text
            # Add prefix if present
            full_text = option.prefix + option_text + option.suffix
            # Add 4 chars for borders and padding
            required_chars = len(full_text) + 4
            min_width_chars = max(min_width_chars, required_chars)

        menu_width_chars = max(self.width_chars, min_width_chars)

        # Store tile dimensions for positioning on screen
        self.tile_dimensions = self.controller.graphics.tile_dimensions
        # Store dimensions in lines/chars for layout calculations
        self.width = menu_width_chars
        self.height = menu_height_lines

        # Calculate pixel dimensions from font metrics
        self.pixel_width = self.width * self._char_width
        self.pixel_height = self.height * self._line_height

        # Center on screen - convert pixel dimensions to tiles for positioning
        tile_w, tile_h = self.tile_dimensions
        screen_pixel_w = self.controller.graphics.console_width_tiles * tile_w
        screen_pixel_h = self.controller.graphics.console_height_tiles * tile_h

        # Calculate centered pixel position, then convert to tiles
        center_px_x = (screen_pixel_w - self.pixel_width) // 2
        center_px_y = (screen_pixel_h - self.pixel_height) // 2

        # Store tile position for presentation (used by TextOverlay.present())
        self.x_tiles = center_px_x // tile_w if tile_w > 0 else 0
        self.y_tiles = center_px_y // tile_h if tile_h > 0 else 0

    def draw_content(self) -> None:
        """Render the menu content using pixel-based layout with font metrics."""
        assert self.canvas is not None
        assert isinstance(self.canvas, PillowImageCanvas)

        # Fill entire background with black
        self.canvas.draw_rect(
            pixel_x=0,
            pixel_y=0,
            width=self.pixel_width,
            height=self.pixel_height,
            color=colors.BLACK,
            fill=True,
        )

        # Draw double-line box frame using Unicode box-drawing characters
        self._draw_box_frame()

        # Draw the title
        self.render_title()

        # Convert MenuOptions to SelectableRows for the renderer
        rows = [
            SelectableRow(
                text=opt.text,
                key=opt.key,
                enabled=opt.enabled,
                color=opt.color,
                # Convert prefix/prefix_color to prefix_segments if present
                prefix_segments=(
                    opt.prefix_segments
                    if opt.prefix_segments
                    else [(opt.prefix, opt.prefix_color)]
                    if opt.prefix
                    else None
                ),
                suffix=opt.suffix,
                suffix_color=opt.suffix_color,
                force_color=opt.force_color,
                data=opt,  # Preserve MenuOption for click handling
            )
            for opt in self.options
        ]
        self._list_renderer.rows = rows
        self._list_renderer.hovered_index = self.hovered_option_index

        # Options start at line 2 (after border + title)
        # Content area is inset from borders with some margin
        content_margin = self._char_width + 8  # 1 char + 8px margin from border
        content_x = content_margin
        content_y = self._line_height * 2  # After border (line 0) + title (line 1)
        content_width = self.pixel_width - (content_margin * 2)  # Width minus margins

        # Get font ascent for proper vertical positioning
        _, ascent, _ = self.canvas.get_text_metrics("X")

        self._list_renderer.render(
            x_start=content_x,
            y_start=content_y + ascent,  # Baseline position
            max_width=content_width,
            line_height=self._line_height,
            ascent=ascent,
            row_gap=0,  # Menu options are tightly packed
        )

    def _draw_box_frame(self) -> None:
        """Draw a double-line box frame using Unicode box-drawing characters.

        Uses double-line characters for a clean, classic menu appearance:
        ╔═══╗
        ║   ║
        ╚═══╝
        """
        assert self.canvas is not None
        char_w = self._char_width
        line_h = self._line_height

        # Double-line corners
        self.canvas.draw_text(0, 0, "╔", colors.WHITE)
        self.canvas.draw_text((self.width - 1) * char_w, 0, "╗", colors.WHITE)
        self.canvas.draw_text(0, (self.height - 1) * line_h, "╚", colors.WHITE)
        self.canvas.draw_text(
            (self.width - 1) * char_w, (self.height - 1) * line_h, "╝", colors.WHITE
        )

        # Double horizontal lines (top and bottom)
        for x in range(1, self.width - 1):
            self.canvas.draw_text(x * char_w, 0, "═", colors.WHITE)
            self.canvas.draw_text(
                x * char_w, (self.height - 1) * line_h, "═", colors.WHITE
            )

        # Double vertical lines (left and right)
        for y in range(1, self.height - 1):
            self.canvas.draw_text(0, y * line_h, "║", colors.WHITE)
            self.canvas.draw_text(
                (self.width - 1) * char_w, y * line_h, "║", colors.WHITE
            )

    def render_title(self) -> None:
        """Renders the menu's title, centered, in the header area.

        Title is rendered on line 1 (after the top border on line 0).
        """
        assert self.canvas is not None
        assert isinstance(self.canvas, PillowImageCanvas)

        # Get actual title width in pixels for proper centering
        title_width_px, _, _ = self.canvas.get_text_metrics(self.title)
        center_px_x = (self.pixel_width - title_width_px) // 2

        self.canvas.draw_text(
            pixel_x=center_px_x,
            pixel_y=self._line_height,  # Line 1, after the top border
            text=self.title,
            color=colors.YELLOW,
        )

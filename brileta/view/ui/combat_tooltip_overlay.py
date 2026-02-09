"""Cursor-attached tooltip overlay for combat hit probabilities.

Displays the selected action's hit probability (e.g., "65%") in red text
next to the mouse cursor when hovering over valid combat targets. Inspired by
the Fallout 1/2 targeting UI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from brileta import colors, config, input_events
from brileta.backends.pillow.canvas import PillowImageCanvas
from brileta.game.actors import Character
from brileta.types import PixelCoord
from brileta.view.ui.overlays import TextOverlay

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.view.render.canvas import Canvas


# Pixel offset from cursor to tooltip text (negative Y for upper-right positioning)
TOOLTIP_OFFSET_X: PixelCoord = 20
TOOLTIP_OFFSET_Y: PixelCoord = -10

# Drop shadow offset for text readability
SHADOW_OFFSET: int = 2


class CombatTooltipOverlay(TextOverlay):
    """Cursor-attached tooltip showing combat hit probabilities.

    A non-interactive overlay that renders just the hit probability percentage
    (e.g., "65%") next to the mouse cursor when hovering over valid combat
    targets. The tooltip only appears when:
    - Combat mode is active
    - The cursor is over a valid target (visible, alive enemy)
    - An action is selected in combat mode

    Uses minimal red text (no background) for the classic Fallout aesthetic.
    """

    def __init__(self, controller: Controller) -> None:
        super().__init__(controller)
        self.is_interactive = False  # Passthrough overlay - doesn't consume input

        # Cache for tooltip text to avoid recalculating every frame
        self._cached_text: str = ""
        self._cached_target: Character | None = None

    def _get_backend(self) -> Canvas:
        """Return a PillowImageCanvas for crisp text rendering."""
        return PillowImageCanvas(
            self.controller.graphics,
            font_path=config.UI_FONT_PATH,
            font_size=config.ACTION_PANEL_FONT_SIZE,
            line_spacing=1.0,
        )

    def _calculate_dimensions(self) -> None:
        """Calculate tooltip dimensions based on current text.

        The tooltip is positioned near the cursor, so dimensions are based
        solely on the text content. Position is updated each frame.
        """
        assert isinstance(self.canvas, PillowImageCanvas)

        # Get tile dimensions for coordinate conversion
        self.tile_dimensions = self.controller.graphics.tile_dimensions
        tile_width, tile_height = self.tile_dimensions

        # Get current hovered target and calculate text
        self._update_tooltip_content()

        if not self._cached_text:
            # No valid target - hide the tooltip
            self.width = 0
            self.height = 0
            self.pixel_width = 0
            self.pixel_height = 0
            self.x_tiles = 0
            self.y_tiles = 0
            return

        # Calculate text dimensions with minimal padding to prevent clipping
        # (fonts can have negative y offsets that get clipped at canvas edges)
        # Include shadow offset in dimensions for drop shadow effect
        text_width, text_height, _ = self.canvas.get_text_metrics(self._cached_text)
        padding = 4
        self.pixel_width = text_width + padding * 2 + SHADOW_OFFSET
        self.pixel_height = text_height + padding * 2 + SHADOW_OFFSET

        # Get mouse position from cursor manager
        cursor_manager = self.controller.frame_manager.cursor_manager
        mouse_x: PixelCoord = cursor_manager.mouse_pixel_x
        mouse_y: PixelCoord = cursor_manager.mouse_pixel_y

        # Position tooltip relative to cursor (upper-right: offset Y is negative,
        # and we subtract the tooltip height so the bottom edge is above the cursor)
        tooltip_x = mouse_x + TOOLTIP_OFFSET_X
        tooltip_y = mouse_y + TOOLTIP_OFFSET_Y - self.pixel_height

        # Clamp to screen bounds
        screen_width = self.controller.graphics.console_width_tiles * tile_width
        screen_height = self.controller.graphics.console_height_tiles * tile_height

        if tooltip_x + self.pixel_width > screen_width:
            tooltip_x = mouse_x - TOOLTIP_OFFSET_X - self.pixel_width
        if tooltip_y + self.pixel_height > screen_height:
            tooltip_y = mouse_y - TOOLTIP_OFFSET_Y - self.pixel_height

        # Ensure tooltip stays on screen
        tooltip_x = max(0, tooltip_x)
        tooltip_y = max(0, tooltip_y)

        # Convert to tiles for presentation system
        self.x_tiles = tooltip_x // tile_width if tile_width > 0 else 0
        self.y_tiles = tooltip_y // tile_height if tile_height > 0 else 0

        # Store width/height in "tiles" for the base class (though we use pixels)
        self.width = self.pixel_width // tile_width if tile_width > 0 else 0
        self.height = self.pixel_height // tile_height if tile_height > 0 else 0

    def _update_tooltip_content(self) -> None:
        """Update the cached tooltip text based on current hover target.

        Checks if the cursor is over a valid combat target and calculates
        the hit probability for the currently selected action. Shows only
        the percentage (e.g., "65%"), not the action name.
        """
        self._cached_text = ""
        self._cached_target = None

        # Only show in combat mode
        if not self.controller.is_combat_mode():
            return

        combat_mode = self.controller.combat_mode
        if combat_mode.selected_action is None:
            return

        # Get the target under the cursor
        gw = self.controller.gw
        mouse_pos = gw.mouse_tile_location_on_map

        if mouse_pos is None:
            return

        mx, my = mouse_pos
        if not (0 <= mx < gw.game_map.width and 0 <= my < gw.game_map.height):
            return

        # Check visibility
        if not gw.game_map.visible[mx, my]:
            return

        # Get actor at position
        actor = gw.get_actor_at_location(mx, my)
        if actor is None:
            return

        # Validate it's a valid combat target
        if not isinstance(actor, Character):
            return
        if not actor.health or not actor.health.is_alive():
            return
        if actor is gw.player:
            return

        self._cached_target = actor

        # Get the selected action's probability for this target
        selected = combat_mode.selected_action

        # Get probability by fetching actions with this specific target
        actions = combat_mode.get_available_combat_actions(actor)

        # Find matching action by id
        prob: float | None = None
        for action in actions:
            if action.id == selected.id:
                prob = action.success_probability
                break

        # Build tooltip text - show only percentage (no action name)
        if prob is not None:
            prob_percent = int(prob * 100)
            self._cached_text = f"{prob_percent}%"
        # If no probability (action not valid for this target), hide tooltip

    def draw_content(self) -> None:
        """Render the tooltip text with drop shadow for readability."""
        assert self.canvas is not None
        assert isinstance(self.canvas, PillowImageCanvas)

        if not self._cached_text:
            return

        # Draw drop shadow first (black, offset), then red text on top
        padding = 4

        # Black shadow for readability on varying backgrounds
        self.canvas.draw_text(
            pixel_x=padding + SHADOW_OFFSET,
            pixel_y=padding + SHADOW_OFFSET,
            text=self._cached_text,
            color=colors.BLACK,
        )

        # Red text on top
        self.canvas.draw_text(
            pixel_x=padding,
            pixel_y=padding,
            text=self._cached_text,
            color=colors.RED,
        )

    def handle_input(self, event: input_events.InputEvent) -> bool:
        """Non-interactive overlay - always returns False to pass input through."""
        return False

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors, config
from catley.backends.pillow.canvas import PillowImageCanvas
from catley.constants.view import ViewConstants as View
from catley.events import MessageEvent, publish_event
from catley.game.items.properties import WeaponProperty
from catley.types import InterpolationAlpha
from catley.view.render.graphics import GraphicsContext

from .base import TextView

if TYPE_CHECKING:
    from catley.controller import Controller


class EquipmentView(TextView):
    """Displays the player's equipped weapons and ammo counts.

    Interactive equipment slots where clicking:
    - Inactive slot: switches to it
    - Active slot: uses the item (enters combat/targeting mode, or toggles off)

    Uses tile-based text rendering for visual consistency with the rest of the UI.
    """

    # Minimum panel width to prevent overly narrow display with short/empty names
    MIN_PANEL_WIDTH = 15

    def __init__(self, controller: Controller, graphics: GraphicsContext) -> None:
        super().__init__()
        self.controller = controller
        self._graphics = graphics
        self.canvas = PillowImageCanvas(
            graphics,
            font_path=config.UI_FONT_PATH,
            font_size=config.EQUIPMENT_FONT_SIZE,
            line_spacing=1.0,
        )
        # Track slot bounds for hit detection: slot_index -> (y_start_px, y_end_px)
        self._slot_pixel_bounds: dict[int, tuple[int, int]] = {}
        # Track which row is currently hovered (None if not hovering)
        self._hover_row: int | None = None
        # High-water mark for panel width in pixels (grow-only semantics)
        self._max_width_px_seen: int = 0

    def set_bounds(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Override set_bounds to configure scaling when view is resized."""
        super().set_bounds(x1, y1, x2, y2)
        self.canvas.configure_scaling(self.tile_dimensions[1])

    def get_cache_key(self) -> int:
        """Cache key for rendering invalidation.

        Includes inventory revision, combat mode, hover state, and tile dimensions.
        We need to re-render when combat mode changes to update the border color,
        when hover state changes to update text color, or when tile dimensions change
        (font scaling depends on tile height).
        """
        inventory_rev = self.controller.gw.player.inventory.revision
        in_combat = 1 if self.controller.is_combat_mode() else 0
        # Use -1 for None hover to distinguish from row 0
        hover_component = self._hover_row if self._hover_row is not None else -1
        # Include tile dimensions since font scaling depends on them
        tile_dims = self.tile_dimensions
        # Combine into a single hash-like value
        return hash((inventory_rev, in_combat, hover_component, tile_dims))

    def calculate_min_width(self) -> int:
        """Calculate minimum tile width needed, using grow-only semantics.

        Uses font metrics to measure actual pixel width, then converts to tiles.

        Returns the maximum of:
        - Current equipment width requirements (in tiles)
        - The largest width ever seen (high-water mark)
        - MIN_PANEL_WIDTH floor

        This ensures the panel only grows, never shrinks.
        """
        player = self.controller.gw.player
        current_width_px = 0

        for i, item in enumerate(player.inventory.attack_slots):
            if i >= View.EQUIPMENT_MAX_SLOTS:
                break

            # Build text exactly as draw_content() does
            indicator = "▶ " if i == player.inventory.active_weapon_slot else "  "
            slot_number = f"[{i + 1}]"

            if item:
                item_text = f"{indicator}{slot_number} {item.name}"
                # Include ammo for ranged weapons (except THROWN)
                if (
                    item.ranged_attack
                    and WeaponProperty.THROWN not in item.ranged_attack.properties
                ):
                    ammo = (
                        f"[{item.ranged_attack.current_ammo}/"
                        f"{item.ranged_attack.max_ammo}]"
                    )
                    item_text += f" {ammo}"
            else:
                item_text = f"{indicator}{slot_number} Empty"

            # Use font metrics for pixel width
            text_width, _, _ = self.canvas.get_text_metrics(item_text)
            current_width_px = max(current_width_px, text_width)

        # Update high-water mark if current exceeds it
        if current_width_px > self._max_width_px_seen:
            self._max_width_px_seen = current_width_px

        # Convert pixel width to tiles for layout compatibility
        tile_width = self.tile_dimensions[0]
        if tile_width > 0:
            # Add some padding (8px) and round up
            width_in_tiles = (
                self._max_width_px_seen + 8 + tile_width - 1
            ) // tile_width
        else:
            width_in_tiles = self.MIN_PANEL_WIDTH

        return max(width_in_tiles, self.MIN_PANEL_WIDTH)

    def set_hover_row(self, row: int | None) -> None:
        """Update which row is being hovered over.

        The cache key includes hover state, so changing this will cause
        a cache miss on the next draw, triggering a re-render.

        Args:
            row: The row being hovered, or None if not hovering.
        """
        self._hover_row = row

    def draw_content(
        self, graphics: GraphicsContext, alpha: InterpolationAlpha
    ) -> None:
        """Draw equipment slots with visual differentiation for active slot.

        Uses pixel-based rendering with Cozette TTF font:
        - Active slot: ">" indicator, WHITE text (RED when hovered)
        - Inactive slot: no indicator, GREY text
        """
        tile_width, tile_height = self.tile_dimensions

        pixel_width = self.width * tile_width
        pixel_height = self.height * tile_height
        self.canvas.draw_rect(0, 0, pixel_width, pixel_height, colors.BLACK, fill=True)

        player = self.controller.gw.player
        active_slot = player.inventory.active_weapon_slot

        # Get font metrics for proper line spacing
        ascent, descent = self.canvas.get_font_metrics()
        line_height = ascent + descent

        # Clear slot pixel bounds
        self._slot_pixel_bounds = {}

        for i, item in enumerate(player.inventory.attack_slots):
            if i >= View.EQUIPMENT_MAX_SLOTS:
                break

            is_active = i == active_slot

            # Calculate pixel position for this slot
            y_pixel = i * line_height

            # Record pixel bounds for this slot (for hit detection)
            self._slot_pixel_bounds[i] = (y_pixel, y_pixel + line_height)

            # Build slot text with "▶" indicator for active slot
            indicator = "▶ " if is_active else "  "
            slot_number = f"[{i + 1}]"
            if item:
                item_text = f"{indicator}{slot_number} {item.name}"
                # Show ammo for ranged weapons, but not for THROWN (single-use items)
                if (
                    item.ranged_attack
                    and WeaponProperty.THROWN not in item.ranged_attack.properties
                ):
                    ammo = (
                        f"[{item.ranged_attack.current_ammo}/"
                        f"{item.ranged_attack.max_ammo}]"
                    )
                    item_text += f" {ammo}"
            else:
                item_text = f"{indicator}{slot_number} Empty"

            # Color based on slot state, combat mode, and item type:
            # - In combat mode with active weapon: RED (always)
            # - Hovered active consumable (not in combat): GREEN (beneficial)
            # - Hovered active weapon (not in combat): RED (aggressive)
            # - Active (not hovered, not in combat): WHITE
            # - Inactive: GREY
            if is_active:
                in_combat = self.controller.is_combat_mode()
                is_hovered = self._is_row_in_slot_bounds(self._hover_row, i)

                if in_combat:
                    # In combat mode: active weapon always shows RED
                    color = colors.RED
                elif is_hovered:
                    # Not in combat but hovering: show intent
                    if item and item.consumable_effect:
                        color = colors.CATEGORY_CONSUMABLE
                    else:
                        color = colors.RED
                else:
                    color = colors.WHITE
            else:
                color = colors.GREY

            self.canvas.draw_text(0, y_pixel, item_text, color)

    def _is_row_in_slot_bounds(self, row: int | None, slot_index: int) -> bool:
        """Check if a row (in tile coordinates) falls within a slot's pixel bounds.

        Converts the row to pixel coordinates and checks against the slot's bounds.

        Args:
            row: The row in tile coordinates, or None.
            slot_index: The slot to check against.

        Returns:
            True if the row falls within the slot's bounds.
        """
        if row is None:
            return False
        bounds = self._slot_pixel_bounds.get(slot_index)
        if bounds is None:
            return False

        # Convert row to pixel position (center of the row)
        tile_height = self._graphics.tile_dimensions[1]
        row_center_px = row * tile_height + tile_height // 2

        return bounds[0] <= row_center_px < bounds[1]

    def switch_to_slot(self, slot_index: int) -> None:
        """Switch to the specified weapon slot and publish a message.

        Args:
            slot_index: The weapon slot to switch to (0 = primary, 1 = secondary).
        """
        player = self.controller.gw.player
        if player.inventory.switch_to_weapon_slot(slot_index):
            weapon = player.inventory.get_active_weapon()
            weapon_name = weapon.name if weapon else "Empty"
            slot_names = {0: "primary", 1: "secondary"}
            slot_label = slot_names.get(slot_index, "weapon")
            publish_event(
                MessageEvent(
                    f"Switched to {slot_label} weapon: {weapon_name}",
                    colors.GREEN,
                )
            )

    def handle_click(self, view_relative_row: int) -> bool:
        """Handle a click on this view at the given row.

        Click behavior:
        - Click inactive slot: switch to it
        - Click active slot: use the item (toggle combat/picker mode)

        Args:
            view_relative_row: The row within this view that was clicked (0-indexed).

        Returns:
            True if the click was handled, False otherwise.
        """
        slot_index = self._get_slot_at_row(view_relative_row)
        if slot_index is None:
            return False

        player = self.controller.gw.player

        if slot_index != player.inventory.active_weapon_slot:
            # Inactive slot: switch to it
            self.switch_to_slot(slot_index)
            return True

        # Active slot: use the item
        return self._use_active_slot_item()

    def _get_slot_at_row(self, view_relative_row: int) -> int | None:
        """Get the slot index for a given row, or None if not a slot row.

        Converts tile row to pixel position and checks against slot pixel bounds.

        Args:
            view_relative_row: The row within this view (0-indexed, in tile coords).

        Returns:
            Slot index (0 or 1) if the row corresponds to a weapon slot, None otherwise.
        """
        # Convert row to pixel position (center of the row)
        tile_height = self._graphics.tile_dimensions[1]
        row_center_px = view_relative_row * tile_height + tile_height // 2

        for slot_index, (start_px, end_px) in self._slot_pixel_bounds.items():
            if start_px <= row_center_px < end_px:
                return slot_index
        return None

    def is_row_in_active_slot(self, view_relative_row: int) -> bool:
        """Check if a row is within the active equipment slot.

        Used for hover cursor detection.

        Args:
            view_relative_row: The row within this view (0-indexed, in tile coords).

        Returns:
            True if the row is within the active slot bounds.
        """
        active_slot = self.controller.gw.player.inventory.active_weapon_slot
        return self._is_row_in_slot_bounds(view_relative_row, active_slot)

    def _use_active_slot_item(self) -> bool:
        """Use the item in the active slot based on its capabilities.

        - Toggle: if already in combat or picker mode, exit that mode
        - Weapons: enter combat mode
        - Consumables: enter picker mode for target selection

        Returns:
            True if action was taken, False otherwise.
        """
        player = self.controller.gw.player
        item = player.inventory.get_active_weapon()

        if item is None:
            return False

        # Toggle: if already in combat mode, exit
        if self.controller.is_combat_mode():
            if self.controller.has_visible_hostiles():
                publish_event(
                    MessageEvent(
                        "Standing down despite hostile presence.", colors.YELLOW
                    )
                )
            self.controller.exit_combat_mode("manual_exit")
            return True

        # Check if picker mode is active (consumable targeting) and exit if so
        # Note: combat mode check above handles combat+picker case
        if self.controller.picker_mode.active:
            self.controller.pop_mode()  # Pop PickerMode
            return True

        # Weapons enter combat mode
        if item.melee_attack or item.ranged_attack:
            self.controller.enter_combat_mode()
            return True

        # Consumables enter picker mode for target selection
        if item.consumable_effect:
            self.controller.start_consumable_targeting(item)
            return True

        return False

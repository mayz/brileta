from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
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
        self.canvas = graphics.create_canvas()
        # Track slot bounds for hit detection: slot_index -> (y_start_row, y_end_row)
        self._slot_row_bounds: dict[int, tuple[int, int]] = {}
        # Track which row is currently hovered (None if not hovering)
        self._hover_row: int | None = None
        # High-water mark for panel width (grow-only semantics)
        self._max_width_seen: int = 0

    def get_cache_key(self) -> int:
        """Cache key includes inventory revision, combat mode, and hover state.

        We need to re-render when combat mode changes to update the border color,
        or when hover state changes to update text color.
        """
        inventory_rev = self.controller.gw.player.inventory.revision
        in_combat = 1 if self.controller.is_combat_mode() else 0
        # Use -1 for None hover to distinguish from row 0
        hover_component = self._hover_row if self._hover_row is not None else -1
        # Combine into a single hash-like value
        return hash((inventory_rev, in_combat, hover_component))

    def calculate_min_width(self) -> int:
        """Calculate minimum tile width needed, using grow-only semantics.

        Returns the maximum of:
        - Current equipment width requirements
        - The largest width ever seen (high-water mark)
        - MIN_PANEL_WIDTH floor

        This ensures the panel only grows, never shrinks.
        """
        player = self.controller.gw.player
        current_width = 0

        for i, item in enumerate(player.inventory.attack_slots):
            if i >= View.EQUIPMENT_MAX_SLOTS:
                break

            # Build text exactly as draw_content() does
            indicator = "> " if i == player.inventory.active_weapon_slot else "  "
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

            current_width = max(current_width, len(item_text))

        # Update high-water mark if current exceeds it
        if current_width > self._max_width_seen:
            self._max_width_seen = current_width

        return max(self._max_width_seen, self.MIN_PANEL_WIDTH)

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

        Uses tile-based rendering with same size for both slots:
        - Active slot: ">" indicator, WHITE text (RED when hovered)
        - Inactive slot: no indicator, GREY text
        """
        tile_width, tile_height = self.tile_dimensions

        pixel_width = self.width * tile_width
        pixel_height = self.height * tile_height
        self.canvas.draw_rect(0, 0, pixel_width, pixel_height, colors.BLACK, fill=True)

        player = self.controller.gw.player
        active_slot = player.inventory.active_weapon_slot

        # Clear slot bounds
        self._slot_row_bounds = {}

        for i, item in enumerate(player.inventory.attack_slots):
            if i >= View.EQUIPMENT_MAX_SLOTS:
                break

            is_active = i == active_slot

            # Record row bounds for this slot (for hit detection)
            # Each slot takes exactly one row in tile-based rendering
            self._slot_row_bounds[i] = (i, i + 1)

            y_pixel = i * tile_height

            # Build slot text with ">" indicator for active (tile-friendly character)
            indicator = "> " if is_active else "  "
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

            # Color based on slot state and item type:
            # - Hovered active consumable: GREEN (beneficial)
            # - Hovered active weapon: RED (aggressive)
            # - Active (not hovered): WHITE
            # - Inactive: GREY
            if is_active:
                # Check if hovering this active slot
                bounds = self._slot_row_bounds.get(i, (-1, -1))
                is_hovered = (
                    self._hover_row is not None
                    and bounds[0] <= self._hover_row < bounds[1]
                )
                if is_hovered:
                    # Consumables show green (beneficial), weapons show red (aggressive)
                    if item and item.consumable_effect:
                        color = colors.CATEGORY_CONSUMABLE
                    else:
                        color = colors.RED
                else:
                    color = colors.WHITE
            else:
                color = colors.GREY

            self.canvas.draw_text(0, y_pixel, item_text, color)

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

        Uses the dynamically computed slot bounds from the last render.

        Args:
            view_relative_row: The row within this view (0-indexed).

        Returns:
            Slot index (0 or 1) if the row corresponds to a weapon slot, None otherwise.
        """
        for slot_index, (start_row, end_row) in self._slot_row_bounds.items():
            if start_row <= view_relative_row < end_row:
                return slot_index
        return None

    def is_row_in_active_slot(self, view_relative_row: int) -> bool:
        """Check if a row is within the active equipment slot.

        Used for hover cursor detection.

        Args:
            view_relative_row: The row within this view (0-indexed).

        Returns:
            True if the row is within the active slot bounds.
        """
        active_slot = self.controller.gw.player.inventory.active_weapon_slot
        bounds = self._slot_row_bounds.get(active_slot)
        if bounds is None:
            return False
        start_row, end_row = bounds
        return start_row <= view_relative_row < end_row

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
            self.controller.exit_combat_mode()
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

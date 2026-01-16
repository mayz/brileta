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
    """Displays the player's equipped weapons and ammo counts."""

    def __init__(self, controller: Controller, graphics: GraphicsContext) -> None:
        super().__init__()
        self.controller = controller
        self.canvas = graphics.create_canvas()

    def get_cache_key(self) -> int:
        """The key is the inventory's revision number."""
        return self.controller.gw.player.inventory.revision

    def draw_content(
        self, graphics: GraphicsContext, alpha: InterpolationAlpha
    ) -> None:
        tile_width, tile_height = self.tile_dimensions
        pixel_width = self.width * tile_width
        pixel_height = self.height * tile_height
        self.canvas.draw_rect(0, 0, pixel_width, pixel_height, colors.BLACK, fill=True)

        y_start = 1
        player = self.controller.gw.player

        # Add weapon switching hint
        hint_text = "Weapons: [1][2] to switch"
        self.canvas.draw_text(0, 0, hint_text, colors.GREY)

        # Check for reload hint and display on separate line
        # (skip for THROWN weapons - they don't have reloadable ammo)
        active_weapon = player.inventory.get_active_weapon()
        if (
            active_weapon
            and active_weapon.ranged_attack
            and WeaponProperty.THROWN not in active_weapon.ranged_attack.properties
            and active_weapon.ranged_attack.current_ammo
            < active_weapon.ranged_attack.max_ammo
        ):
            reload_text = "[R] to reload"
            self.canvas.draw_text(
                0,
                View.EQUIPMENT_RELOAD_HINT_OFFSET * tile_height,
                reload_text,
                colors.YELLOW,
            )

        for i, item in enumerate(player.inventory.attack_slots):
            if i >= View.EQUIPMENT_MAX_SLOTS:
                break
            active_marker = ">" if i == player.inventory.active_weapon_slot else " "
            slot_name = f"{active_marker}{i + 1}"
            if item:
                item_text = f"{slot_name}: {item.name}"
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
                color = (
                    colors.WHITE
                    if i == player.inventory.active_weapon_slot
                    else colors.LIGHT_GREY
                )
            else:
                item_text = f"{slot_name}: Empty"
                color = colors.GREY
            self.canvas.draw_text(
                pixel_x=0,
                pixel_y=(y_start + i) * tile_height,
                text=item_text,
                color=color,
            )

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

        Args:
            view_relative_row: The row within this view that was clicked (0-indexed).

        Returns:
            True if the click was handled (clicked a weapon slot), False otherwise.
        """
        # Weapon slots start at row 1 (row 0 is hint text)
        # Row 1 = slot 0 (primary), Row 2 = slot 1 (secondary)
        weapon_slot_start_row = 1
        weapon_slot_end_row = weapon_slot_start_row + View.EQUIPMENT_MAX_SLOTS

        if weapon_slot_start_row <= view_relative_row < weapon_slot_end_row:
            slot_index = view_relative_row - weapon_slot_start_row
            self.switch_to_slot(slot_index)
            return True

        return False

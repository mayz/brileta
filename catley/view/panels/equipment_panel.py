from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.constants.view import ViewConstants as View
from catley.view.renderer import Renderer
from catley.view.text_backend import TextBackend

from .panel import Panel

if TYPE_CHECKING:
    from catley.controller import Controller


class EquipmentPanel(Panel):
    """Displays the player's equipped weapons and ammo counts."""

    def __init__(
        self, controller: Controller, *, text_backend: TextBackend | None = None
    ) -> None:
        super().__init__(text_backend)
        self.controller = controller

    def draw_content(self, renderer: Renderer) -> None:
        assert self.text_backend is not None

        y_start = 1
        player = self.controller.gw.player

        # Add weapon switching hint
        hint_text = "Weapons: [1][2] to switch"
        self.text_backend.draw_text(0, 0, hint_text, colors.GREY)

        # Check for reload hint and display on separate line
        active_weapon = player.inventory.get_active_weapon()
        if (
            active_weapon
            and active_weapon.ranged_attack
            and active_weapon.ranged_attack.current_ammo
            < active_weapon.ranged_attack.max_ammo
        ):
            reload_text = "[R] to reload"
            self.text_backend.draw_text(
                0,
                View.EQUIPMENT_RELOAD_HINT_OFFSET,
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
                if item.ranged_attack:
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
            self.text_backend.draw_text(0, y_start + i, item_text, color)

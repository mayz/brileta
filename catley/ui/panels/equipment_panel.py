from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.render.renderer import Renderer

from .panel import Panel

if TYPE_CHECKING:
    from catley.controller import Controller


class EquipmentPanel(Panel):
    """Displays the player's equipped weapons and ammo counts."""

    def __init__(self, controller: Controller, x: int, y: int) -> None:
        # Height is two lines for weapons plus one for hint text.
        super().__init__(x=x, y=y, width=0, height=3)
        self.controller = controller

    def draw(self, renderer: Renderer) -> None:
        if not self.visible:
            return

        y_start = self.y + 1
        player = self.controller.gw.player

        # Add weapon switching and reload hints
        hint_text = "Weapons: [1][2] to switch"
        active_weapon = player.inventory.get_active_weapon()
        if (
            active_weapon
            and active_weapon.ranged_attack
            and active_weapon.ranged_attack.current_ammo
            < active_weapon.ranged_attack.max_ammo
        ):
            hint_text += " | [R] to reload"

        renderer.draw_text(self.x, self.y, hint_text, fg=colors.GREY)

        for i, item in enumerate(player.inventory.attack_slots):
            if i >= 2:
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
            renderer.draw_text(self.x, y_start + i, item_text, fg=color)

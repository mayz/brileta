from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from brileta import colors, config
from brileta.backends.pillow.canvas import PillowImageCanvas
from brileta.view.render.canvas import Canvas
from brileta.view.ui.overlays import Menu, MenuOption

if TYPE_CHECKING:
    from brileta.controller import Controller


class HelpMenu(Menu):
    """Menu showing all available commands.

    Clicking on an option performs that action (where applicable).
    """

    def __init__(self, controller: Controller) -> None:
        super().__init__("Help - Available Commands", controller, width=60)

    def _get_backend(self) -> Canvas:
        """Return a PillowImageCanvas matching the action panel font size."""
        return PillowImageCanvas(
            self.controller.graphics,
            font_path=config.UI_FONT_PATH,
            font_size=config.ACTION_PANEL_FONT_SIZE,
            line_spacing=config.MENU_LINE_SPACING,
        )

    def populate_options(self) -> None:
        """Populate help options with clickable actions."""
        # Clickable items: (key, text, action)
        clickable_items: list[tuple[str, str, Callable[[], None] | None]] = [
            ("Arrows / HJKL", "Movement", self._on_movement),
            ("L-Click", "Select actor", self._on_select_actor),
            ("R-Click", "Quick action", self._on_quick_action),
            ("I", "Inventory", self._on_inventory),
            ("R", "Reload weapon", self._on_reload),
            ("1", "Weapon slot 1", lambda: self._on_weapon_slot(0)),
            ("2", "Weapon slot 2", lambda: self._on_weapon_slot(1)),
            ("`", "Dev console", self._on_dev_console),
            ("Alt+Ret", "Toggle fullscreen", self._on_fullscreen),
            ("Esc", "Close menus", None),
            ("?", "This help", None),
        ]

        for key, text, action in clickable_items:
            self.add_option(
                MenuOption(
                    key=key,
                    text=text,
                    enabled=True,
                    action=action if action else self.hide,
                    color=colors.WHITE,
                    is_primary_action=False,
                )
            )

    # -------------------------------------------------------------------------
    # Action Callbacks
    # -------------------------------------------------------------------------

    def _on_movement(self) -> None:
        """Hint about movement controls."""
        from brileta.events import MessageEvent, publish_event

        publish_event(
            MessageEvent("Use arrow keys or HJKL to move around", colors.GREY)
        )

    def _on_select_actor(self) -> None:
        """Hint about L-Click behavior."""
        from brileta.events import MessageEvent, publish_event

        publish_event(MessageEvent("Left-click on an actor to select it", colors.GREY))

    def _on_quick_action(self) -> None:
        """Hint about R-Click behavior."""
        from brileta.events import MessageEvent, publish_event

        publish_event(
            MessageEvent("Right-click on a target for quick action", colors.GREY)
        )

    def _on_inventory(self) -> None:
        """Open inventory (loot mode if standing on items)."""
        from brileta.view.ui.commands import open_inventory_or_loot

        open_inventory_or_loot(self.controller)

    def _on_reload(self) -> None:
        """Reload the active weapon."""
        from brileta.events import MessageEvent, publish_event
        from brileta.game.actions.combat import ReloadIntent

        player = self.controller.gw.player
        active_weapon = player.inventory.get_active_item()
        if (
            active_weapon
            and active_weapon.ranged_attack
            and active_weapon.ranged_attack.current_ammo
            < active_weapon.ranged_attack.max_ammo
        ):
            reload_action = ReloadIntent(self.controller, player, active_weapon)
            self.controller.queue_action(reload_action)
        else:
            publish_event(MessageEvent("Nothing to reload!", colors.GREY))

    def _on_weapon_slot(self, slot: int) -> None:
        """Switch to a weapon slot via the intent system."""
        from brileta.game.actions.misc import SwitchWeaponIntent

        player = self.controller.gw.player
        switch_action = SwitchWeaponIntent(self.controller, player, slot)
        self.controller.queue_action(switch_action)

    def _on_dev_console(self) -> None:
        """Toggle the dev console overlay."""
        fm = self.controller.frame_manager
        self.controller.overlay_system.toggle_overlay(fm.dev_console_overlay)

    def _on_fullscreen(self) -> None:
        """Toggle fullscreen mode."""
        self.controller.app.toggle_fullscreen()

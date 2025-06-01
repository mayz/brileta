from __future__ import annotations

from typing import TYPE_CHECKING

import tcod.event

from catley import colors
from catley.ui.help_menu import HelpMenu
from catley.ui.inventory_menu import InventoryMenu

from .game.actions import GameAction, MoveAction
from .ui.ui_commands import (
    OpenMenuUICommand,
    OpenPickupMenuUICommand,
    OpenTargetMenuUICommand,
    QuitUICommand,
    SelectOrDeselectActorUICommand,
    ToggleFullscreenUICommand,
)

if TYPE_CHECKING:
    from .controller import Controller
    from .ui.ui_commands import UICommand


class EventHandler:
    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.gw = controller.gw
        self.game_map = self.gw.game_map
        self.p = self.gw.player

    def dispatch(self, event: tcod.event.Event) -> None:
        # First, try to handle the event with the menu system
        menu_consumed = self.controller.menu_system.handle_input(event)

        # If no menu handled it, check for normal game actions
        if not menu_consumed:
            action = self.handle_event(event)
            if action:
                self.controller.queue_action(action)

        # Process a round (the controller will decide if anything needs to happen)
        self.controller.process_unified_round()

    def handle_event(self, event: tcod.event.Event) -> GameAction | None:
        # Don't process game actions if menus are active
        if self.controller.menu_system.has_active_menus():
            return None

        match event:
            case tcod.event.MouseMotion():
                root_tile_pos = event.position
                map_coords = self._get_tile_map_coords_from_root_coords(root_tile_pos)

                if map_coords is not None:
                    self.gw.mouse_tile_location_on_map = map_coords
                else:
                    # Mouse is outside the game map area (e.g., on UI panels).
                    self.gw.mouse_tile_location_on_map = None

                return None

        # Handle UI commands.
        ui_command = self._check_for_ui_command(event)
        if ui_command:
            ui_command.execute()
            return None

        # Don't process game actions if player is dead
        if self.p.health and not self.p.health.is_alive():
            return None

        return self._check_for_game_action(event)

    def _check_for_ui_command(self, event: tcod.event.Event) -> UICommand | None:
        match event:
            case tcod.event.Quit():
                return QuitUICommand()
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.ESCAPE)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.q)
            ):
                return QuitUICommand()

            case tcod.event.KeyDown(sym=tcod.event.KeySym.i):
                return OpenMenuUICommand(self.controller, InventoryMenu)

            case tcod.event.KeyDown(sym=key_sym, mod=key_mod) if (
                key_sym == tcod.event.KeySym.QUESTION
                or (
                    key_sym == tcod.event.KeySym.SLASH
                    and (key_mod & tcod.event.Modifier.SHIFT)
                )
            ):
                return OpenMenuUICommand(self.controller, HelpMenu)

            case tcod.event.KeyDown(sym=tcod.event.KeySym.g):
                return OpenPickupMenuUICommand(self.controller)

            case tcod.event.KeyDown(sym=tcod.event.KeySym.t):
                if (
                    self.gw.selected_actor
                    and self.gw.selected_actor != self.gw.player
                    and self.gw.selected_actor.health
                    and self.gw.selected_actor.health.is_alive()
                ):
                    return OpenTargetMenuUICommand(
                        self.controller, target_actor=self.gw.selected_actor
                    )

                # Show message if no valid target selected
                self.controller.message_log.add_message(
                    "No valid target selected. Left-click an actor first.",
                    colors.GREY,
                )
                return None

            case tcod.event.KeyDown(sym=tcod.event.KeySym.RETURN, mod=mod) if (
                mod & tcod.event.Modifier.ALT
            ):
                return ToggleFullscreenUICommand(self.controller.renderer.context)

            case tcod.event.MouseButtonDown(button=button):
                return self._handle_mouse_button_down_event(event, button)

            case _:
                return None

    def _check_for_game_action(self, event: tcod.event.Event) -> GameAction | None:
        match event:
            # Movement keys (Arrows and VIM)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.UP)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.k)
            ):
                return MoveAction(self.controller, self.p, 0, -1)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.DOWN)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.j)
            ):
                return MoveAction(self.controller, self.p, 0, 1)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.LEFT)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.h)
            ):
                return MoveAction(self.controller, self.p, -1, 0)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.RIGHT)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.l)
            ):
                return MoveAction(self.controller, self.p, 1, 0)

            # Diagonal movement (numpad or vi-keys)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_7):  # Up-left
                return MoveAction(self.controller, self.p, -1, -1)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_9):  # Up-right
                return MoveAction(self.controller, self.p, 1, -1)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_1):  # Down-left
                return MoveAction(self.controller, self.p, -1, 1)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_3):  # Down-right
                return MoveAction(self.controller, self.p, 1, 1)

            case _:
                return None

    def _handle_mouse_button_down_event(
        self, event: tcod.event.MouseButtonDown, button: tcod.event.MouseButton
    ) -> UICommand | None:
        """
        Handle mouse button down events.
        Returns an action if the event is handled, otherwise None.
        """
        root_tile_pos = event.position
        map_coords = self._get_tile_map_coords_from_root_coords(root_tile_pos)

        if button != tcod.event.MouseButton.LEFT:
            return None

        if not map_coords:
            # Clicked outside the map area (e.g., on UI panels).
            return SelectOrDeselectActorUICommand(self.controller, None)

        mx, my = map_coords
        actor_at_click = self.gw.get_actor_at_location(mx, my)

        if self.gw.selected_actor == actor_at_click:
            # If the clicked actor is already selected, deselect it.
            return SelectOrDeselectActorUICommand(self.controller, None)

        return SelectOrDeselectActorUICommand(self.controller, actor_at_click)

    def _get_tile_map_coords_from_root_coords(
        self, root_tile_coords: tuple[int, int]
    ) -> tuple[int, int] | None:
        """
        Translates mouse coordinates from root_console space to game_map_console space.
        Returns (map_x, map_y) if the click is within the map area, otherwise None.
        """
        renderer = self.controller.renderer  # Get access to the renderer

        map_render_offset_x = 0
        map_render_offset_y = renderer.help_height + renderer.message_log_height

        # Translate to coordinates relative to game_map_console
        map_x = root_tile_coords[0] - map_render_offset_x
        map_y = root_tile_coords[1] - map_render_offset_y

        # Check if the translated coordinates are within the bounds of game_map_console
        if (
            0 <= map_x < renderer.game_map_console.width
            and 0 <= map_y < renderer.game_map_console.height
        ):
            return map_x, map_y

        return None  # Click was outside the map area

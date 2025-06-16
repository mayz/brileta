from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import tcod.event

from catley import colors
from catley.events import MessageEvent, publish_event
from catley.game.actors import Actor, Character
from catley.view.ui.action_browser_menu import ActionBrowserMenu
from catley.view.ui.context_menu import ContextMenu
from catley.view.ui.help_menu import HelpMenu
from catley.view.ui.inventory_menu import InventoryMenu

from .game.actions.base import GameIntent
from .game.actions.movement import MoveIntent

if TYPE_CHECKING:
    from .controller import Controller

from .view.ui.commands import (
    OpenExistingMenuUICommand,
    OpenMenuUICommand,
    OpenPickupMenuUICommand,
    QuitUICommand,
    SelectOrDeselectActorUICommand,
    ToggleFullscreenUICommand,
    UICommand,
)


class InputHandler:
    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.renderer = self.controller.renderer
        self.cursor_manager = controller.frame_manager.cursor_manager
        self.gw = controller.gw
        self.game_map = self.gw.game_map
        self.p = self.gw.player

    def dispatch(self, event: tcod.event.Event) -> None:
        if isinstance(event, tcod.event.MouseState):
            # Update mouse cursor position.
            position = event.position
            self.cursor_manager.update_mouse_position(position[0], position[1])

        # Handle window resize events
        if isinstance(event, tcod.event.WindowResized):
            # Update coordinate conversion when the window size changes
            self.controller.renderer.update_dimensions()
            self.controller.coordinate_converter = (
                self.controller.renderer.coordinate_converter
            )
            # Update panel layouts for new window size
            self.controller.frame_manager.on_window_resized()

        # Try to handle the event with the menu system
        menu_consumed = self.controller.overlay_system.handle_input(event)

        # If no menu handled it, check for normal game actions
        if not menu_consumed:
            action = self.handle_event(event)
            if action:
                self.controller.queue_action(action)

        # Process a round (the controller will decide if anything needs to happen)
        self.controller.process_unified_round()

    def handle_event(self, event: tcod.event.Event) -> GameIntent | None:
        # Don't process game actions if menus are active
        if self.controller.overlay_system.has_active_menus():
            return None

        match event:
            case tcod.event.MouseMotion():
                event = self.convert_mouse_coordinates(event)

                root_tile_pos = event.position
                map_coords = (
                    self.controller.frame_manager.get_tile_map_coords_from_root_coords(
                        root_tile_pos
                    )
                )

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
        # Check if a mode wants to handle this first
        if self.controller.active_mode and self.controller.active_mode.handle_input(
            event
        ):
            return None

        match event:
            case tcod.event.Quit():
                return QuitUICommand()
            case tcod.event.KeyDown(sym=tcod.event.KeySym.q):
                return QuitUICommand()
            case tcod.event.KeyDown(sym=tcod.event.KeySym.ESCAPE):
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

            case tcod.event.KeyDown(sym=tcod.event.KeySym.SPACE):
                return OpenMenuUICommand(self.controller, ActionBrowserMenu)

            case tcod.event.KeyDown(sym=tcod.event.KeySym.g):
                return OpenPickupMenuUICommand(self.controller)

            case tcod.event.KeyDown(sym=tcod.event.KeySym.t):
                # Toggle targeting mode
                if self.controller.is_targeting_mode():
                    self.controller.exit_targeting_mode()
                else:
                    self.controller.enter_targeting_mode()
                return None

            case tcod.event.KeyDown(sym=tcod.event.KeySym.r):
                # Reload active weapon
                active_weapon = self.p.inventory.get_active_weapon()
                if (
                    active_weapon
                    and active_weapon.ranged_attack
                    and active_weapon.ranged_attack.current_ammo
                    < active_weapon.ranged_attack.max_ammo
                ):
                    from .game.actions.combat import ReloadIntent

                    reload_action = ReloadIntent(self.controller, self.p, active_weapon)
                    self.controller.queue_action(reload_action)
                else:
                    publish_event(MessageEvent("Nothing to reload!", colors.GREY))
                return None

            case tcod.event.KeyDown(sym=tcod.event.KeySym.N1):
                # Switch to weapon slot 1
                if self.p.inventory.switch_to_weapon_slot(0):
                    weapon = self.p.inventory.get_active_weapon()
                    weapon_name = weapon.name if weapon else "Empty"
                    publish_event(
                        MessageEvent(
                            f"Switched to primary weapon: {weapon_name}", colors.GREEN
                        )
                    )
                return None

            case tcod.event.KeyDown(sym=tcod.event.KeySym.N2):
                # Switch to weapon slot 2
                if self.p.inventory.switch_to_weapon_slot(1):
                    weapon = self.p.inventory.get_active_weapon()
                    weapon_name = weapon.name if weapon else "Empty"
                    publish_event(
                        MessageEvent(
                            f"Switched to secondary weapon: {weapon_name}", colors.GREEN
                        )
                    )
                return None

            case tcod.event.KeyDown(sym=tcod.event.KeySym.RETURN, mod=mod) if (
                mod & tcod.event.Modifier.ALT
            ):
                return ToggleFullscreenUICommand(
                    self.renderer.context,
                    self.renderer.root_console,
                )

            case tcod.event.MouseButtonDown():
                return self._handle_mouse_button_down_event(event)

            case _:
                return None

    def _check_for_game_action(self, event: tcod.event.Event) -> GameIntent | None:
        match event:
            # Movement keys (Arrows and VIM)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.UP)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.k)
            ):
                return MoveIntent(self.controller, self.p, 0, -1)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.DOWN)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.j)
            ):
                return MoveIntent(self.controller, self.p, 0, 1)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.LEFT)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.h)
            ):
                return MoveIntent(self.controller, self.p, -1, 0)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.RIGHT)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.l)
            ):
                return MoveIntent(self.controller, self.p, 1, 0)

            # Diagonal movement (numpad or vi-keys)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_7):  # Up-left
                return MoveIntent(self.controller, self.p, -1, -1)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_9):  # Up-right
                return MoveIntent(self.controller, self.p, 1, -1)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_1):  # Down-left
                return MoveIntent(self.controller, self.p, -1, 1)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_3):  # Down-right
                return MoveIntent(self.controller, self.p, 1, 1)

            case _:
                return None

    def _handle_mouse_button_down_event(
        self, event: tcod.event.MouseButtonDown
    ) -> UICommand | None:
        """
        Handle mouse button down events.
        Returns an action if the event is handled, otherwise None.
        """
        event_with_tile_coords = self.convert_mouse_coordinates(event)
        root_tile_pos = event_with_tile_coords.position
        map_coords = self.controller.frame_manager.get_tile_map_coords_from_root_coords(
            root_tile_pos
        )

        if event.button == tcod.event.MouseButton.RIGHT:
            target: Actor | tuple[int, int] | None = None
            if map_coords:
                mx, my = map_coords
                if (
                    0 <= mx < self.game_map.width
                    and 0 <= my < self.game_map.height
                    and self.game_map.visible[mx, my]
                ):
                    actor_at_click = self.gw.get_actor_at_location(mx, my)
                    if actor_at_click is not None and (
                        not isinstance(actor_at_click, Character)
                        or actor_at_click.health.is_alive()
                    ):
                        target = actor_at_click
                    else:
                        target = (mx, my)
            if target and self._has_available_actions(target):
                context_menu = ContextMenu(self.controller, target, root_tile_pos)
                return OpenExistingMenuUICommand(self.controller, context_menu)
            return None

        if event.button != tcod.event.MouseButton.LEFT:
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

    def _has_available_actions(self, target: Actor | tuple[int, int]) -> bool:
        """Quickly check if any actions are available for a target."""
        from catley.environment import tile_types
        from catley.game.actions.discovery import ActionDiscovery

        disc = ActionDiscovery()
        player = self.p

        if isinstance(target, Character):
            options = disc.get_options_for_target(self.controller, player, target)
            return bool(options)
        if isinstance(target, Actor):
            return False

        x, y = target
        if not (0 <= x < self.game_map.width and 0 <= y < self.game_map.height):
            return False

        tile = self.game_map.tiles[x, y]
        distance = abs(player.x - x) + abs(player.y - y)

        return bool(
            distance <= 1
            and tile
            in (
                tile_types.TILE_TYPE_ID_DOOR_CLOSED,  # type: ignore[attr-defined]
                tile_types.TILE_TYPE_ID_DOOR_OPEN,  # type: ignore[attr-defined]
            )
        )

    def convert_mouse_coordinates(
        self, event: tcod.event.MouseState
    ) -> tcod.event.MouseState:
        """Convert event coordinates."""
        pixel_x, pixel_y = event.position
        tile_x, tile_y = self.controller.coordinate_converter.pixel_to_tile(
            pixel_x, pixel_y
        )

        event_copy = copy.copy(event)
        event_copy.position = tcod.event.Point(tile_x, tile_y)
        return event_copy

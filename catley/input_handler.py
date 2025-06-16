from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING

import tcod.event

from catley import colors
from catley.events import MessageEvent, publish_event
from catley.game.actors import Actor, Character
from catley.util.coordinates import (
    PixelCoord,
    PixelPos,
    RootConsoleTileCoord,
    RootConsoleTilePos,
    WorldTileCoord,
    WorldTilePos,
)
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
        self.fm = self.controller.frame_manager
        self.cursor_manager = self.fm.cursor_manager
        self.gw = controller.gw
        self.game_map = self.gw.game_map
        self.p = self.gw.player

    def dispatch(self, event: tcod.event.Event) -> None:
        if isinstance(event, tcod.event.MouseState):
            # Update mouse cursor position using pixel coordinates.
            px_pos: PixelPos = event.position
            px_x: PixelCoord = px_pos[0]
            px_y: PixelCoord = px_pos[1]
            self.cursor_manager.update_mouse_position(px_x, px_y)

        # Handle window resize events
        if isinstance(event, tcod.event.WindowResized):
            # Update coordinate conversion when the window size changes
            self.controller.renderer.update_dimensions()
            self.controller.coordinate_converter = (
                self.controller.renderer.coordinate_converter
            )
            # Update panel layouts for new window size
            self.fm.on_window_resized()

        # Try to handle the event with the menu system
        menu_consumed = self.controller.overlay_system.handle_input(event)

        # If no menu handled it, check for normal game actions
        if not menu_consumed:
            action = self.handle_event(event)
            if action:
                self.controller.queue_action(action)

    def handle_event(self, event: tcod.event.Event) -> GameIntent | None:
        # Don't process game actions if menus are active
        if self.controller.overlay_system.has_active_menus():
            return None

        match event:
            case tcod.event.MouseMotion():
                event = self.convert_mouse_coordinates(event)

                root_tile_pos: RootConsoleTilePos = event.position
                world_tile_pos: WorldTilePos | None = (
                    self.fm.get_world_coords_from_root_tile_coords(root_tile_pos)
                )

                if world_tile_pos is not None:
                    self.gw.mouse_tile_location_on_map = world_tile_pos
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
        move_intent = None
        match event:
            # Movement keys (Arrows and VIM)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.UP)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.k)
            ):
                move_intent = MoveIntent(self.controller, self.p, 0, -1)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.DOWN)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.j)
            ):
                move_intent = MoveIntent(self.controller, self.p, 0, 1)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.LEFT)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.h)
            ):
                move_intent = MoveIntent(self.controller, self.p, -1, 0)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.RIGHT)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.l)
            ):
                move_intent = MoveIntent(self.controller, self.p, 1, 0)

            # Diagonal movement (numpad or vi-keys)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_7):  # Up-left
                move_intent = MoveIntent(self.controller, self.p, -1, -1)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_9):  # Up-right
                move_intent = MoveIntent(self.controller, self.p, 1, -1)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_1):  # Down-left
                move_intent = MoveIntent(self.controller, self.p, -1, 1)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_3):  # Down-right
                move_intent = MoveIntent(self.controller, self.p, 1, 1)

            case _:
                move_intent = None

        if move_intent:
            # Only time the first input of a chain
            if self.controller.last_input_time is None:
                self.controller.last_input_time = time.perf_counter()
                self.controller.action_count_for_latency_metric = 0
            return move_intent

        return None

    def _handle_mouse_button_down_event(
        self, event: tcod.event.MouseButtonDown
    ) -> UICommand | None:
        """
        Handle mouse button down events.
        Returns an action if the event is handled, otherwise None.
        """
        event_with_tile_coords = self.convert_mouse_coordinates(event)
        root_tile_pos: RootConsoleTilePos = event_with_tile_coords.position
        world_tile_pos: WorldTilePos | None = (
            self.fm.get_world_coords_from_root_tile_coords(root_tile_pos)
        )

        if event.button == tcod.event.MouseButton.RIGHT:
            target: Actor | WorldTilePos | None = None
            if world_tile_pos:
                world_x: WorldTileCoord
                world_y: WorldTileCoord
                world_x, world_y = world_tile_pos
                if (
                    0 <= world_x < self.game_map.width
                    and 0 <= world_y < self.game_map.height
                    and self.game_map.visible[world_x, world_y]
                ):
                    actor_at_click = self.gw.get_actor_at_location(world_x, world_y)
                    if actor_at_click is not None and (
                        not isinstance(actor_at_click, Character)
                        or actor_at_click.health.is_alive()
                    ):
                        target = actor_at_click
                    else:
                        target = (world_x, world_y)
            if target and self._has_available_actions(target):
                context_menu = ContextMenu(self.controller, target, root_tile_pos)
                return OpenExistingMenuUICommand(self.controller, context_menu)
            return None

        if event.button != tcod.event.MouseButton.LEFT:
            return None

        if not world_tile_pos:
            # Clicked outside the map area (e.g., on UI panels).
            return SelectOrDeselectActorUICommand(self.controller, None)

        world_x, world_y = world_tile_pos
        actor_at_click = self.gw.get_actor_at_location(world_x, world_y)

        if self.gw.selected_actor == actor_at_click:
            # If the clicked actor is already selected, deselect it.
            return SelectOrDeselectActorUICommand(self.controller, None)

        return SelectOrDeselectActorUICommand(self.controller, actor_at_click)

    def _has_available_actions(self, target: Actor | WorldTilePos) -> bool:
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

        world_x, world_y = target
        if not (
            0 <= world_x < self.game_map.width and 0 <= world_y < self.game_map.height
        ):
            return False

        tile = self.game_map.tiles[world_x, world_y]
        distance = abs(player.x - world_x) + abs(player.y - world_y)

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
        """Convert event pixel coordinates to root console tile coordinates."""
        px_pos: PixelPos = event.position
        px_x: PixelCoord = px_pos[0]
        px_y: PixelCoord = px_pos[1]
        root_tile_x: RootConsoleTileCoord
        root_tile_y: RootConsoleTileCoord
        root_tile_x, root_tile_y = self.controller.coordinate_converter.pixel_to_tile(
            px_x, px_y
        )

        event_copy = copy.copy(event)
        event_copy.position = tcod.event.Point(root_tile_x, root_tile_y)
        return event_copy

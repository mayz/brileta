from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING

import tcod.event

from catley import colors
from catley.events import MessageEvent, publish_event
from catley.game.actors import Actor, Character
from catley.types import (
    PixelCoord,
    PixelPos,
    RootConsoleTilePos,
    WorldTileCoord,
    WorldTilePos,
)
from catley.view.ui.action_browser_menu import ActionBrowserMenu
from catley.view.ui.context_menu import ContextMenu
from catley.view.ui.help_menu import HelpMenu
from catley.view.ui.inventory_menu import InventoryMenu

from .game.actions.base import GameIntent

if TYPE_CHECKING:
    from .app import App
    from .controller import Controller

from typing import Final

from .view.ui.commands import (
    OpenExistingMenuUICommand,
    OpenMenuUICommand,
    OpenPickupMenuUICommand,
    QuitUICommand,
    ToggleFullscreenUICommand,
    UICommand,
)

# Define KeySym constants for alphabetic keys missing from type stubs.


class Keys:
    """Container for letter KeySym constants used in pattern matching."""

    KEY_H: Final = tcod.event.KeySym(ord("h"))
    KEY_J: Final = tcod.event.KeySym(ord("j"))
    KEY_K: Final = tcod.event.KeySym(ord("k"))
    KEY_L: Final = tcod.event.KeySym(ord("l"))
    KEY_Q: Final = tcod.event.KeySym(ord("q"))
    KEY_I: Final = tcod.event.KeySym(ord("i"))
    KEY_G: Final = tcod.event.KeySym(ord("g"))
    KEY_T: Final = tcod.event.KeySym(ord("t"))
    KEY_R: Final = tcod.event.KeySym(ord("r"))


class InputHandler:
    def __init__(self, app: App, controller: Controller) -> None:
        self.app = app
        self.controller = controller
        self.fm = self.controller.frame_manager
        assert self.fm is not None
        self.graphics = self.fm.graphics
        self.cursor_manager = self.fm.cursor_manager
        self.dev_console_overlay = getattr(self.fm, "dev_console_overlay", None)
        self.gw = controller.gw
        self.game_map = self.gw.game_map
        self.p = self.gw.player
        self.movement_keys: set[tcod.event.KeySym] = set()

    def dispatch(self, event: tcod.event.Event) -> None:
        if isinstance(event, tcod.event.KeyDown | tcod.event.KeyUp) and event.sym in {
            tcod.event.KeySym.UP,
            tcod.event.KeySym.DOWN,
            tcod.event.KeySym.LEFT,
            tcod.event.KeySym.RIGHT,
            Keys.KEY_H,
            Keys.KEY_J,
            Keys.KEY_K,
            Keys.KEY_L,
        }:
            if isinstance(event, tcod.event.KeyDown):
                if not self.movement_keys:
                    self.controller.last_input_time = time.perf_counter()
                    self.controller.action_count_for_latency_metric = 0
                self.movement_keys.add(event.sym)
            else:
                self.movement_keys.discard(event.sym)

        if isinstance(event, tcod.event.MouseState):
            # Update mouse cursor position using pixel coordinates.
            px_pos: PixelPos = event.position
            px_x: PixelCoord = px_pos[0]
            px_y: PixelCoord = px_pos[1]

            # Scale coordinates for cursor rendering to match framebuffer coordinates
            scale_x, scale_y = self.graphics.get_display_scale_factor()
            scaled_px_x: PixelCoord = px_x * scale_x
            scaled_px_y: PixelCoord = px_y * scale_y
            self.cursor_manager.update_mouse_position(scaled_px_x, scaled_px_y)

        # Try to handle the event with the menu system
        # For mouse events, we need to send scaled coordinates to match cursor position
        menu_event = event
        if isinstance(
            event,
            tcod.event.MouseButtonDown
            | tcod.event.MouseButtonUp
            | tcod.event.MouseMotion,
        ):
            # Create scaled version for menu system
            scale_x, scale_y = self.graphics.get_display_scale_factor()
            scaled_x = event.position.x * scale_x
            scaled_y = event.position.y * scale_y
            menu_event = copy.copy(event)
            menu_event.position = tcod.event.Point(int(scaled_x), int(scaled_y))

        assert self.controller.overlay_system is not None
        menu_consumed = self.controller.overlay_system.handle_input(menu_event)

        # If no menu handled it, check for normal game actions
        if not menu_consumed:
            action = self.handle_event(event)
            if action:
                self.controller.queue_action(action)

    def handle_event(self, event: tcod.event.Event) -> GameIntent | None:
        assert self.controller.overlay_system is not None
        assert self.fm is not None

        # Don't process game actions if INTERACTIVE menus are active
        if self.controller.overlay_system.has_interactive_overlays():
            return None

        match event:
            case tcod.event.MouseMotion():
                event = self.convert_mouse_coordinates(event)

                root_tile_pos: RootConsoleTilePos = (
                    int(event.position.x),
                    int(event.position.y),
                )
                world_tile_pos: WorldTilePos | None = (
                    self.fm.get_world_coords_from_root_tile_coords(root_tile_pos)
                )

                if world_tile_pos is not None:
                    self.gw.mouse_tile_location_on_map = world_tile_pos
                else:
                    # Mouse is outside the game map area (e.g., on UI views).
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

        return None

    def _check_for_ui_command(self, event: tcod.event.Event) -> UICommand | None:
        # Check if a mode wants to handle this first
        if self.controller.active_mode and self.controller.active_mode.handle_input(
            event
        ):
            return None

        assert self.controller.overlay_system is not None

        # Check action panel hotkeys first (for immediate action execution)
        if isinstance(event, tcod.event.KeyDown) and hasattr(
            self.fm, "action_panel_view"
        ):
            # Get the character from the key symbol
            key_char = None
            if 97 <= event.sym <= 122:  # a-z
                key_char = chr(event.sym)
            elif 65 <= event.sym <= 90:  # A-Z
                key_char = chr(event.sym).lower()

            if key_char:
                # Don't process hotkey actions if player is dead
                if not self.p.health.is_alive():
                    return None

                hotkeys = self.fm.action_panel_view.get_hotkeys()
                if key_char in hotkeys:
                    action_option = hotkeys[key_char]
                    # Execute the action using the same pattern as context menu
                    action_executed = False
                    if hasattr(action_option, "execute") and action_option.execute:
                        result = action_option.execute()
                        if isinstance(result, GameIntent):
                            self.controller.queue_action(result)
                            action_executed = True
                    elif action_option.action_class:
                        # Create action instance with static params
                        action_instance = action_option.action_class(
                            self.controller,
                            self.controller.gw.player,
                            **action_option.static_params,
                        )
                        self.controller.queue_action(action_instance)
                        action_executed = True

                    # If action was executed, check if it affects the hovered tile
                    if action_executed and self._action_affects_hovered_tile(
                        action_option
                    ):
                        # Invalidate action panel cache to refresh on next frame
                        self.fm.action_panel_view.invalidate_cache()

                    if action_executed:
                        return None  # Action handled

        match event:
            case tcod.event.Quit():
                return QuitUICommand(self.app)
            case tcod.event.KeyDown(sym=Keys.KEY_Q):
                return QuitUICommand(self.app)

            case tcod.event.KeyDown(sym=Keys.KEY_I):
                return OpenMenuUICommand(self.controller, InventoryMenu)

            case tcod.event.KeyDown(sym=tcod.event.KeySym.BACKQUOTE):
                if self.dev_console_overlay is not None:
                    self.controller.overlay_system.toggle_overlay(
                        self.dev_console_overlay
                    )
                return None

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

            case tcod.event.KeyDown(sym=Keys.KEY_G):
                return OpenPickupMenuUICommand(self.controller)

            case tcod.event.KeyDown(sym=Keys.KEY_T):
                # Toggle targeting mode
                if self.controller.is_targeting_mode():
                    self.controller.exit_targeting_mode()
                else:
                    self.controller.enter_targeting_mode()
                return None

            case tcod.event.KeyDown(sym=Keys.KEY_R):
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
                return ToggleFullscreenUICommand(self.app)

            case tcod.event.MouseButtonDown():
                return self._handle_mouse_button_down_event(event)

            case _:
                return None

    def _handle_mouse_button_down_event(
        self, event: tcod.event.MouseButtonDown
    ) -> UICommand | None:
        """
        Handle mouse button down events.
        Returns an action if the event is handled, otherwise None.
        """
        assert self.fm is not None

        event_with_tile_coords = self.convert_mouse_coordinates(event)
        root_tile_pos: RootConsoleTilePos = (
            int(event_with_tile_coords.position.x),
            int(event_with_tile_coords.position.y),
        )
        world_tile_pos: WorldTilePos | None = (
            self.fm.get_world_coords_from_root_tile_coords(root_tile_pos)
        )

        if event.button == tcod.event.MouseButton.LEFT and (
            tcod.event.get_modifier_state() & tcod.event.Modifier.SHIFT
        ):
            if world_tile_pos is None:
                return None
            self.controller.start_actor_pathfinding(self.p, world_tile_pos)
            return None

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

        # Left click does nothing - no tile/actor selection
        return None

    def _has_available_actions(self, target: Actor | WorldTilePos) -> bool:
        """Quickly check if any actions are available for a target."""
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

        return bool(self.game_map.visible[world_x, world_y])

    def convert_mouse_coordinates(
        self, event: tcod.event.MouseState
    ) -> tcod.event.MouseState:
        """Convert event pixel coordinates to root console tile coordinates."""
        px_pos: PixelPos = event.position
        px_x: PixelCoord = px_pos[0]
        px_y: PixelCoord = px_pos[1]

        # Scale coordinates to match framebuffer space
        # (pixel_to_tile expects framebuffer coords)
        scale_x, scale_y = self.graphics.get_display_scale_factor()
        scaled_px_x: PixelCoord = px_x * scale_x
        scaled_px_y: PixelCoord = px_y * scale_y
        root_tile_x, root_tile_y = self.graphics.pixel_to_tile(scaled_px_x, scaled_px_y)

        event_copy = copy.copy(event)
        event_copy.position = tcod.event.Point(root_tile_x, root_tile_y)
        return event_copy

    def _action_affects_hovered_tile(self, action_option) -> bool:
        """Check if an action affects the currently hovered tile."""
        from catley.game.actions.environment import CloseDoorIntent, OpenDoorIntent

        # Get current mouse position
        mouse_pos = self.controller.gw.mouse_tile_location_on_map
        if mouse_pos is None:
            return False

        mouse_x, mouse_y = mouse_pos

        # Check if it's a door action (either direct or pathfinding)
        # Direct actions have action_class set
        if action_option.action_class in (OpenDoorIntent, CloseDoorIntent):
            action_x = action_option.static_params.get("x")
            action_y = action_option.static_params.get("y")
            if action_x is not None and action_y is not None:
                # Check if this action affects the hovered tile
                return action_x == mouse_x and action_y == mouse_y

        # Don't invalidate for pathfinding actions (they use execute function)
        # Cache will be invalidated when the action actually completes
        return False

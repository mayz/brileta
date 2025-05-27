from __future__ import annotations

from typing import TYPE_CHECKING

import actions
import menu_system
import tcod.event

if TYPE_CHECKING:
    from controller import Controller


class EventHandler:
    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.m = controller.model
        self.game_map = self.m.game_map
        self.p = self.m.player

    def dispatch(self, event: tcod.event.Event) -> None:
        # First, try to handle the event with the menu system
        if self.controller.menu_system.handle_input(event):
            return  # Event was consumed by menu system

        # If no menu handled it, process normal game actions
        action = self.handle_event(event)
        if action:
            self.controller.execute_action(action)

    def handle_event(self, event: tcod.event.Event) -> actions.Action | None:
        # Don't process game actions if menus are active
        if self.controller.menu_system.has_active_menus():
            return None

        match event:
            case tcod.event.Quit():
                return actions.QuitAction()

            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.ESCAPE)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.q)
            ):
                return actions.QuitAction()

            # Movement keys (Arrows and VIM)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.UP)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.k)
            ):
                return actions.MoveAction(self.controller, self.p, 0, -1)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.DOWN)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.j)
            ):
                return actions.MoveAction(self.controller, self.p, 0, 1)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.LEFT)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.h)
            ):
                return actions.MoveAction(self.controller, self.p, -1, 0)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.RIGHT)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.l)
            ):
                return actions.MoveAction(self.controller, self.p, 1, 0)

            # Diagonal movement (numpad or vi-keys)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_7):  # Up-left
                return actions.MoveAction(self.controller, self.p, -1, -1)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_9):  # Up-right
                return actions.MoveAction(self.controller, self.p, 1, -1)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_1):  # Down-left
                return actions.MoveAction(self.controller, self.p, -1, 1)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_3):  # Down-right
                return actions.MoveAction(self.controller, self.p, 1, 1)

            # Menu keys
            case tcod.event.KeyDown(sym=tcod.event.KeySym.i):
                return actions.OpenMenuAction(
                    self.controller, menu_system.InventoryMenu
                )

            # Help Menu: '?'
            # tcod.event.KeySym.QUESTION is typically generated for '?'
            # This pattern handles '?' (if sym is KeySym.QUESTION),
            # or '?' (if sym is KeySym.SLASH and SHIFT modifier is present).
            case tcod.event.KeyDown(sym=key_sym, mod=key_mod) if (
                key_sym == tcod.event.KeySym.QUESTION
                or (
                    key_sym == tcod.event.KeySym.SLASH
                    and (key_mod & tcod.event.Modifier.SHIFT)
                )
            ):
                return actions.OpenMenuAction(self.controller, menu_system.HelpMenu)

            case tcod.event.KeyDown(sym=tcod.event.KeySym.g):
                return actions.OpenPickupMenuAction(self.controller)

            case tcod.event.KeyDown(sym=tcod.event.KeySym.RETURN, mod=mod) if (
                mod & tcod.event.Modifier.ALT
            ):
                return actions.ToggleFullscreenAction(self.controller.renderer.context)

            case tcod.event.MouseMotion():
                root_tile_pos = event.position
                map_coords = self._get_tile_map_coords_from_root_coords(root_tile_pos)

                if map_coords is not None:
                    self.m.mouse_tile_location_on_map = map_coords
                else:
                    # Mouse is outside the game map area (e.g., on UI panels).
                    self.m.mouse_tile_location_on_map = None

                return None

            case tcod.event.MouseButtonDown(button=button):
                return self._handle_mouse_button_down_event(event, button)

            case _:
                return None

    def _handle_mouse_button_down_event(
        self, event: tcod.event.Event, button: tcod.event.MouseButton
    ) -> actions.Action | None:
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
            return actions.SelectOrDeselectEntityAction(self.controller, None)

        mx, my = map_coords
        entity_at_click = self.m.get_entity_at_location(mx, my)

        if self.m.selected_entity == entity_at_click:
            # If the clicked entity is already selected, deselect it.
            return actions.SelectOrDeselectEntityAction(self.controller, None)

        return actions.SelectOrDeselectEntityAction(self.controller, entity_at_click)

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

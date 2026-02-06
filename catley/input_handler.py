"""InputHandler - Thin dispatcher that routes input to modes and overlays.

The InputHandler receives raw input events from the app and dispatches them
to the appropriate handler:
1. Window close events (OS-level, always work)
2. Overlay system (menus are modal - they consume input first)
3. Q key quit command (respects menus)
4. Active mode's handle_input()
5. Mouse motion for tile tracking

Movement key tracking and UI command handling have moved to ExploreMode.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from catley import input_events
from catley.input_events import Keys
from catley.types import (
    PixelCoord,
    PixelPos,
    RootConsoleTilePos,
    WorldTilePos,
)

if TYPE_CHECKING:
    from .app import App
    from .controller import Controller

from .view.ui.commands import (
    QuitUICommand,
)


class InputHandler:
    """Thin dispatcher that routes input events to modes and overlays.

    This class is responsible for:
    - Updating mouse cursor position
    - Handling window close events (always work)
    - Delegating input to overlays first (menus are modal)
    - Handling Q key quit (after overlays had their chance)
    - Delegating remaining input to the active mode
    - Updating mouse tile location for hover effects

    Movement key tracking and UI command handling are in ExploreMode.
    """

    def __init__(self, app: App, controller: Controller) -> None:
        self.app = app
        self.controller = controller
        self.fm = self.controller.frame_manager
        assert self.fm is not None
        self.graphics = self.fm.graphics
        self.cursor_manager = self.fm.cursor_manager
        self.gw = controller.gw

    def dispatch(self, event: input_events.InputEvent) -> None:
        """Main entry point for all input events.

        Priority order:
        1. Mouse position updates (always)
        2. Window close events (OS-level, always work)
        3. Overlay system (menus are modal - they consume input first)
        4. Q key quit command (respects menus)
        5. Active mode input handling
        6. Mouse motion for tile tracking
        """
        # Update mouse cursor position
        if isinstance(event, input_events.MouseState):
            px_pos: PixelPos = event.position
            px_x: PixelCoord = px_pos[0]
            px_y: PixelCoord = px_pos[1]

            scale_x, scale_y = self.graphics.get_display_scale_factor()
            scaled_px_x: PixelCoord = px_x * scale_x
            scaled_px_y: PixelCoord = px_y * scale_y
            self.cursor_manager.update_mouse_position(scaled_px_x, scaled_px_y)

        # Window close events always work (OS-level, can't prevent)
        if isinstance(event, input_events.Quit):
            QuitUICommand(self.app).execute()
            return

        # Scale mouse events for overlay system
        menu_event = event
        if isinstance(
            event,
            input_events.MouseButtonDown
            | input_events.MouseButtonUp
            | input_events.MouseMotion,
        ):
            scale_x, scale_y = self.graphics.get_display_scale_factor()
            scaled_x = event.position.x * scale_x
            scaled_y = event.position.y * scale_y
            menu_event = copy.copy(event)
            menu_event.position = input_events.Point(int(scaled_x), int(scaled_y))

        # Overlays first: Menus are modal - they consume input before anything else
        assert self.controller.overlay_system is not None
        if self.controller.overlay_system.handle_input(menu_event):
            return  # Overlay consumed the event

        # Q key quits the game (only checked after overlays had their chance)
        if self._is_quit_key(event):
            QuitUICommand(self.app).execute()
            return

        # Then delegate to mode stack (top-to-bottom until handled)
        # This allows modes higher in the stack to intercept input,
        # and unhandled input falls through to modes below.
        for mode in reversed(self.controller.mode_stack):
            if mode.handle_input(event):
                return  # Mode consumed the event

        # Handle mouse motion for tile tracking and hover effects
        if isinstance(event, input_events.MouseMotion):
            self._update_mouse_tile_location(event)
            self._update_hover_cursor(event)

    def _is_quit_key(self, event: input_events.InputEvent) -> bool:
        """Check if event is the Q key (quit hotkey)."""
        match event:
            case input_events.KeyDown(sym=Keys.KEY_Q):
                return True
        return False

    def _update_mouse_tile_location(self, event: input_events.MouseMotion) -> None:
        """Update the mouse tile location for hover effects."""
        assert self.fm is not None

        event_with_tile_coords = self._convert_mouse_coordinates(event)
        root_tile_pos: RootConsoleTilePos = (
            int(event_with_tile_coords.position.x),
            int(event_with_tile_coords.position.y),
        )
        world_tile_pos: WorldTilePos | None = (
            self.fm.get_world_coords_from_root_tile_coords(root_tile_pos)
        )

        if world_tile_pos is not None:
            self.gw.mouse_tile_location_on_map = world_tile_pos
        else:
            # Mouse is outside the game map area (e.g., on UI views).
            self.gw.mouse_tile_location_on_map = None

        self.controller.update_hovered_actor(self.gw.mouse_tile_location_on_map)

        if self.controller.is_combat_mode():
            fm = self.controller.frame_manager
            if fm is not None and hasattr(fm, "combat_tooltip_overlay"):
                tooltip = fm.combat_tooltip_overlay
                if tooltip.is_active:
                    tooltip.invalidate()

    def _update_hover_cursor(self, event: input_events.MouseMotion) -> None:
        """Update cursor based on what the mouse is hovering over.

        Changes cursor to hand when hovering over the active equipment slot
        to indicate it's clickable.
        """
        assert self.fm is not None

        # Only change cursor in explore mode (not combat/picker)
        from catley.modes.explore import ExploreMode

        if not isinstance(self.controller.active_mode, ExploreMode):
            return

        event_with_tile_coords = self._convert_mouse_coordinates(event)
        root_tile_pos: RootConsoleTilePos = (
            int(event_with_tile_coords.position.x),
            int(event_with_tile_coords.position.y),
        )

        # Check if hovering over equipment view and update hover state
        is_hovering_active = self._update_equipment_hover_state(root_tile_pos)
        if is_hovering_active:
            self.cursor_manager.set_active_cursor_type("crosshair")
        else:
            self.cursor_manager.set_active_cursor_type("arrow")

    def _update_equipment_hover_state(self, root_tile_pos: RootConsoleTilePos) -> bool:
        """Update equipment view hover state and check if hovering active slot.

        Sets the hover row on the equipment view for visual feedback (RED text
        when hovering active slot). Also returns whether we're hovering over
        the active slot for cursor updates.

        Args:
            root_tile_pos: The mouse position in root console tile coordinates.

        Returns:
            True if hovering over the active equipment slot, False otherwise.
        """
        assert self.fm is not None

        if not hasattr(self.fm, "equipment_view"):
            return False

        equipment_view = self.fm.equipment_view
        tile_x, tile_y = root_tile_pos

        # Check if within equipment view bounds
        if not (equipment_view.x <= tile_x < equipment_view.x + equipment_view.width):
            equipment_view.set_hover_row(None)
            return False

        clicked_row = tile_y - equipment_view.y
        if clicked_row < 0 or clicked_row >= equipment_view.height:
            equipment_view.set_hover_row(None)
            return False

        # Update hover state for visual feedback
        is_active = equipment_view.is_row_in_active_slot(clicked_row)
        if is_active:
            equipment_view.set_hover_row(clicked_row)
        else:
            equipment_view.set_hover_row(None)

        return is_active

    def _convert_mouse_coordinates(
        self, event: input_events.MouseState
    ) -> input_events.MouseState:
        """Convert event pixel coordinates to root console tile coordinates."""
        px_pos: PixelPos = event.position
        px_x: PixelCoord = px_pos[0]
        px_y: PixelCoord = px_pos[1]

        scale_x, scale_y = self.graphics.get_display_scale_factor()
        scaled_px_x: PixelCoord = px_x * scale_x
        scaled_px_y: PixelCoord = px_y * scale_y
        root_tile_x, root_tile_y = self.graphics.pixel_to_tile(scaled_px_x, scaled_px_y)

        event_copy = copy.copy(event)
        event_copy.position = input_events.Point(root_tile_x, root_tile_y)
        return event_copy

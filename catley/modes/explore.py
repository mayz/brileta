"""ExploreMode - Default gameplay mode for exploration and interaction.

ExploreMode is the base layer that the game starts in. Other modes (CombatMode,
future LockpickingMode, etc.) layer on top by falling back to ExploreMode for
common functionality like movement, inventory, and weapon switching.

Responsibilities:
- Movement: tracks held movement keys, generates intents via MoveIntentGenerator
- UI commands: I (inventory), R (reload), 1/2 (slots), ? (help), etc.
- Mouse interaction: hover over things to see options in ActionPanel, click to select
- Click weapon for combat mode
- Rendering: selected actor highlight when not in another mode
"""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING

from catley import colors, config, input_events
from catley.events import MessageEvent, publish_event
from catley.game.action_plan import WalkToPlan
from catley.game.actions.base import GameIntent
from catley.game.actors import Actor, Character
from catley.input_events import Keys
from catley.modes.base import Mode
from catley.move_intent_generator import MoveIntentGenerator
from catley.types import (
    PixelCoord,
    PixelPos,
    RootConsoleTilePos,
    WorldTilePos,
)
from catley.view.ui.commands import OpenMenuUICommand, ToggleFullscreenUICommand
from catley.view.ui.help_menu import HelpMenu

if TYPE_CHECKING:
    from catley.controller import Controller


class ExploreMode(Mode):
    """Default gameplay mode. Other modes layer on top of this.

    ExploreMode handles all "normal gameplay" input:
    - Movement key tracking and intent generation
    - UI commands (inventory, help, action browser, etc.)
    - Mouse clicks (context menu, equipment view, status view)
    - Action panel hotkey execution
    """

    # Movement keys that this mode tracks
    MOVEMENT_KEYS = frozenset(
        {
            input_events.KeySym.UP,
            input_events.KeySym.DOWN,
            input_events.KeySym.LEFT,
            input_events.KeySym.RIGHT,
            Keys.KEY_H,
            Keys.KEY_J,
            Keys.KEY_K,
            Keys.KEY_L,
        }
    )

    def __init__(self, controller: Controller) -> None:
        super().__init__(controller)

        # Movement state
        self.movement_keys: set[input_events.KeySym] = set()
        self.move_generator = MoveIntentGenerator(controller)

        # Cache references for convenience (may be None in tests)
        self._cursor_manager = None
        self._fm = None
        self._app = controller.app

        # These may not be available during __init__ (test environments)
        if controller.frame_manager is not None:
            self._fm = controller.frame_manager
            if hasattr(controller.frame_manager, "cursor_manager"):
                self._cursor_manager = controller.frame_manager.cursor_manager

    @property
    def gw(self):
        """Game world, accessed dynamically to avoid stale references after regen."""
        return self.controller.gw

    @property
    def player(self):
        """Player character, accessed dynamically to stay fresh after world regen."""
        return self.controller.gw.player

    def enter(self) -> None:
        """Enter explore mode - set default cursor."""
        super().enter()
        if self._cursor_manager is not None:
            self._cursor_manager.set_active_cursor_type("arrow")

    def _exit(self) -> None:
        """Exit explore mode - clear movement keys."""
        self.movement_keys.clear()
        super()._exit()

    def handle_input(self, event: input_events.InputEvent) -> bool:
        """Handle explore mode input.

        Returns True if the event was consumed.
        """
        # Track movement keys
        if isinstance(event, input_events.KeyDown) and event.sym in self.MOVEMENT_KEYS:
            if not self.movement_keys:
                # First key press - record time for latency metric
                self.controller.last_input_time = time.perf_counter()
                self.controller.action_count_for_latency_metric = 0
            self.movement_keys.add(event.sym)
            return True

        if isinstance(event, input_events.KeyUp) and event.sym in self.MOVEMENT_KEYS:
            self.movement_keys.discard(event.sym)
            return True

        # Check action panel hotkeys first (for immediate action execution)
        if self._handle_action_panel_hotkey(event):
            return True

        # Handle UI commands
        match event:
            case input_events.KeyDown(sym=Keys.KEY_I):
                self._open_inventory()
                return True

            case input_events.KeyDown(sym=input_events.KeySym.GRAVE):
                self._toggle_dev_console()
                return True

            case input_events.KeyDown(sym=key_sym, mod=key_mod) if (
                key_sym == input_events.KeySym.QUESTION
                or (
                    key_sym == input_events.KeySym.SLASH
                    and (key_mod & input_events.Modifier.SHIFT)
                )
            ):
                self._open_help()
                return True

            case input_events.KeyDown(sym=Keys.KEY_R):
                self._reload_weapon()
                return True

            case input_events.KeyDown(sym=input_events.KeySym.N1):
                self._switch_weapon_slot(0)
                return True

            case input_events.KeyDown(sym=input_events.KeySym.N2):
                self._switch_weapon_slot(1)
                return True

            case input_events.KeyDown(sym=input_events.KeySym.RETURN, mod=mod) if (
                mod & input_events.Modifier.ALT
            ):
                ToggleFullscreenUICommand(self._app).execute()
                return True

            case input_events.MouseButtonDown():
                return self._handle_mouse_click(event)

            case input_events.MouseMotion():
                return self._handle_mouse_motion(event)

        return False

    def update(self) -> None:
        """Generate movement intent if keys held."""
        # Don't generate movement if player is dead
        if self.player.health and not self.player.health.is_alive():
            return

        # Always call generate_intent to maintain timing state.
        # When no keys are held, generate_intent resets is_first_move_of_burst
        # so the next key press gets the initial delay instead of fast repeat.
        intent = self.move_generator.generate_intent(self.movement_keys)
        if intent:
            # Energy gate: player must afford the action
            if self.player.energy.can_afford(config.ACTION_COST):
                self.controller._execute_player_action_immediately(intent)
            else:
                # Player tried to move but can't afford it - time passes anyway
                self.controller.turn_manager.on_player_action()

    def render_world(self) -> None:
        """Render actor outlines for selection and hover feedback."""
        # This is the default rendering when no other mode is active
        if self._fm is not None and hasattr(self._fm, "world_view"):
            self._fm.world_view._render_selection_and_hover_outlines()

    # -------------------------------------------------------------------------
    # UI Command Helpers
    # -------------------------------------------------------------------------

    def _open_inventory(self) -> None:
        """Open inventory (loot mode if standing on items)."""
        from catley.view.ui.commands import open_inventory_or_loot

        open_inventory_or_loot(self.controller)

    def _toggle_dev_console(self) -> None:
        """Toggle the dev console overlay."""
        if self._fm is not None:
            dev_console = getattr(self._fm, "dev_console_overlay", None)
            if dev_console is not None:
                self.controller.overlay_system.toggle_overlay(dev_console)

    def _open_help(self) -> None:
        """Open the help menu."""
        OpenMenuUICommand(self.controller, HelpMenu).execute()

    def _reload_weapon(self) -> None:
        """Reload the active weapon."""
        active_weapon = self.player.inventory.get_active_item()
        if (
            active_weapon
            and active_weapon.ranged_attack
            and active_weapon.ranged_attack.current_ammo
            < active_weapon.ranged_attack.max_ammo
        ):
            from catley.game.actions.combat import ReloadIntent

            reload_action = ReloadIntent(self.controller, self.player, active_weapon)
            self.controller.queue_action(reload_action)
        else:
            publish_event(MessageEvent("Nothing to reload!", colors.GREY))

    def _switch_weapon_slot(self, slot: int) -> None:
        """Switch to a weapon slot via the intent system."""
        from catley.game.actions.misc import SwitchWeaponIntent

        switch_action = SwitchWeaponIntent(self.controller, self.player, slot)
        self.controller.queue_action(switch_action)

    # -------------------------------------------------------------------------
    # Action Panel Hotkey Handling
    # -------------------------------------------------------------------------

    def _handle_action_panel_hotkey(self, event: input_events.InputEvent) -> bool:
        """Check if a keypress matches an action panel hotkey."""
        if not isinstance(event, input_events.KeyDown):
            return False

        if self._fm is None or not hasattr(self._fm, "action_panel_view"):
            return False

        # Get the character from the key symbol
        key_char = None
        if 97 <= event.sym <= 122:  # a-z
            key_char = chr(event.sym)
        elif 65 <= event.sym <= 90:  # A-Z
            key_char = chr(event.sym).lower()

        if not key_char:
            return False

        # Don't process hotkey actions if player is dead
        if self.player.health and not self.player.health.is_alive():
            return False

        hotkeys = self._fm.action_panel_view.get_hotkeys()
        if key_char not in hotkeys:
            return False

        action_option = hotkeys[key_char]
        action_executed = False

        if hasattr(action_option, "execute") and action_option.execute:
            result = action_option.execute()
            if isinstance(result, GameIntent):
                self.controller.queue_action(result)
            # execute() handled the action (returns True/False or a GameIntent)
            action_executed = True
        elif action_option.action_class:
            # Create action instance with static params
            action_instance = action_option.action_class(
                self.controller,
                self.player,
                **action_option.static_params,
            )
            self.controller.queue_action(action_instance)
            action_executed = True

        # If action was executed, check if it affects the hovered tile
        if action_executed and self._action_affects_hovered_tile(action_option):
            self._fm.action_panel_view.invalidate_cache()

        return action_executed

    def _action_affects_hovered_tile(self, action_option) -> bool:
        """Check if an action affects the currently hovered tile."""
        from catley.game.actions.environment import CloseDoorIntent, OpenDoorIntent

        mouse_pos = self.gw.mouse_tile_location_on_map
        if mouse_pos is None:
            return False

        mouse_x, mouse_y = mouse_pos

        if action_option.action_class in (OpenDoorIntent, CloseDoorIntent):
            action_x = action_option.static_params.get("x")
            action_y = action_option.static_params.get("y")
            if action_x is not None and action_y is not None:
                return action_x == mouse_x and action_y == mouse_y

        return False

    # -------------------------------------------------------------------------
    # Mouse Click Handling
    # -------------------------------------------------------------------------

    def _get_panel_relative_coords(
        self, event: input_events.MouseState
    ) -> tuple[int, int, bool]:
        """Convert mouse event coordinates to action panel-relative pixel coords.

        Args:
            event: Mouse event with position in raw pixel coordinates.

        Returns:
            Tuple of (rel_px_x, rel_px_y, is_inside_panel). If the mouse is
            outside the panel bounds, rel_px_x and rel_px_y are set to -1.
        """
        if self._fm is None or not hasattr(self._fm, "action_panel_view"):
            return (-1, -1, False)

        action_panel_view = self._fm.action_panel_view
        graphics = self.controller.graphics

        # Convert raw pixel position to scaled pixel position
        scale_x, scale_y = graphics.get_display_scale_factor()
        scaled_px_x = int(event.position.x * scale_x)
        scaled_px_y = int(event.position.y * scale_y)

        # Calculate action panel's screen pixel bounds
        tile_width, tile_height = graphics.tile_dimensions
        panel_px_x = action_panel_view.x * tile_width
        panel_px_y = action_panel_view.y * tile_height
        panel_px_width = action_panel_view.width * tile_width
        panel_px_height = action_panel_view.height * tile_height

        # Check if mouse is within panel bounds
        if (
            panel_px_x <= scaled_px_x < panel_px_x + panel_px_width
            and panel_px_y <= scaled_px_y < panel_px_y + panel_px_height
        ):
            # Convert to panel-relative pixel coordinates
            rel_px_x = scaled_px_x - panel_px_x
            rel_px_y = scaled_px_y - panel_px_y
            return (rel_px_x, rel_px_y, True)

        return (-1, -1, False)

    def _handle_mouse_motion(self, event: input_events.MouseMotion) -> bool:
        """Handle mouse motion for hover state updates.

        Updates the action panel hover state when the mouse moves over it.
        """
        if self._fm is None or not hasattr(self._fm, "action_panel_view"):
            return False

        action_panel_view = self._fm.action_panel_view
        rel_px_x, rel_px_y, _ = self._get_panel_relative_coords(event)

        # Update hover state (invalidates cache if changed)
        # When outside panel, rel_px_x/rel_px_y are -1, which clears hover
        if action_panel_view.update_hover_from_pixel(rel_px_x, rel_px_y):
            action_panel_view.invalidate_cache()

        # Don't consume mouse motion - other handlers may need it
        return False

    def _handle_mouse_click(self, event: input_events.MouseButtonDown) -> bool:
        """Handle mouse button down events.

        Click behavior:
        - Left-click on actor/object: Select it (ActionPanel locks to target)
        - Left-click on empty ground: Deselect current target
        - Shift+left-click: Pathfind to that tile
        - Right-click on actor/object: Execute default action immediately
        """
        if self._fm is None:
            return False

        event_with_tile_coords = self._convert_mouse_coordinates(event)
        root_tile_pos: RootConsoleTilePos = (
            int(event_with_tile_coords.position.x),
            int(event_with_tile_coords.position.y),
        )
        world_tile_pos: WorldTilePos | None = (
            self._fm.get_world_coords_from_root_tile_coords(root_tile_pos)
        )

        # Shift+left click for pathfinding (regardless of what's there)
        if event.button == input_events.MouseButton.LEFT and (
            event.mod & input_events.Modifier.SHIFT
        ):
            if world_tile_pos is not None:
                self.controller.start_plan(
                    self.player, WalkToPlan, target_position=world_tile_pos
                )
            return True

        # Right click executes default action
        if event.button == input_events.MouseButton.RIGHT:
            return self._handle_right_click(world_tile_pos, root_tile_pos)

        # Left click for selection/deselection or action panel clicks
        if event.button == input_events.MouseButton.LEFT:
            # Check if click is on the action panel first
            if self._try_execute_action_panel_click(event):
                return True
            return self._handle_left_click(root_tile_pos, world_tile_pos)

        return False

    def _try_execute_action_panel_click(
        self, event: input_events.MouseButtonDown
    ) -> bool:
        """Try to execute an action by clicking on it in the action panel.

        Args:
            event: The mouse button down event.

        Returns:
            True if click was on action panel and action executed.
        """
        if self._fm is None or not hasattr(self._fm, "action_panel_view"):
            return False

        rel_px_x, rel_px_y, is_inside = self._get_panel_relative_coords(event)
        if not is_inside:
            return False

        action_panel_view = self._fm.action_panel_view

        # First check for control row clicks (handled via execute callbacks)
        if action_panel_view.execute_at_pixel(rel_px_x, rel_px_y):
            return True

        # Then check for action option clicks
        action_option = action_panel_view.get_action_at_pixel(rel_px_x, rel_px_y)
        if action_option is not None:
            # Execute the action (same logic as hotkey handling)
            if hasattr(action_option, "execute") and action_option.execute:
                result = action_option.execute()
                if isinstance(result, GameIntent):
                    self.controller.queue_action(result)
            elif action_option.action_class:
                action_instance = action_option.action_class(
                    self.controller,
                    self.player,
                    **action_option.static_params,
                )
                self.controller.queue_action(action_instance)
            return True

        return False

    def _handle_right_click(
        self,
        world_tile_pos: WorldTilePos | None,
        root_tile_pos: RootConsoleTilePos,
    ) -> bool:
        """Handle right click for default action execution.

        Right-click executes the contextually obvious action immediately:
        - NPC: Talk (pathfind to them first if needed)
        - Container: Search (pathfind to adjacent tile if needed)
        - Closed door: Open (pathfind to adjacent tile if needed)
        - Open door: Close (pathfind to adjacent tile if needed)
        - Item pile: Pick up (pathfind to items if needed)
        - Floor tile: Walk there
        """
        from catley.game.actions.discovery import execute_default_action

        if world_tile_pos is None:
            return True  # Consume right click even if nothing to do

        world_x, world_y = world_tile_pos
        game_map = self.gw.game_map

        # Check bounds
        if not (0 <= world_x < game_map.width and 0 <= world_y < game_map.height):
            return True

        # Must at least be explored (remembered) to interact
        if not game_map.explored[world_x, world_y]:
            return True  # Can't interact with unexplored tiles

        # For remembered but not currently visible tiles, only allow walking
        if not game_map.visible[world_x, world_y]:
            # Can only walk to remembered tiles - no interaction with unseen actors
            # Use start_plan directly rather than execute_default_action,
            # since the latter would try to classify actors at the tile.
            self.controller.start_plan(
                self.player, WalkToPlan, target_position=(world_x, world_y)
            )
            self.controller.deselect_target()
            return True

        # Visible tiles - full interaction (existing code continues here)
        # Determine the target (actor at location or tile position)
        target: Actor | WorldTilePos
        actor_at_click = self.gw.get_actor_at_location(world_x, world_y)
        if actor_at_click is not None and actor_at_click is not self.player:
            # Skip dead characters
            if (
                isinstance(actor_at_click, Character)
                and not actor_at_click.health.is_alive()
            ):
                target = (world_x, world_y)
            else:
                target = actor_at_click
        else:
            target = (world_x, world_y)

        # Execute the default action for this target
        execute_default_action(self.controller, target)

        # Clear selection after right-click action
        self.controller.deselect_target()
        return True

    def _handle_left_click(
        self, root_tile_pos: RootConsoleTilePos, world_tile_pos: WorldTilePos | None
    ) -> bool:
        """Handle left click for selection and UI interaction.

        Left-click behavior:
        - Click on actor/container: Toggle selection (select if not selected,
          deselect if already selected)
        - Click on empty ground: Deselect current target
        - Click on equipment view: Handle weapon slot switching
        """
        # First check equipment view (always takes priority for its region)
        if self._check_equipment_view_click(root_tile_pos):
            return True

        # If no world position, can't do selection
        if world_tile_pos is None:
            return False

        world_x, world_y = world_tile_pos
        game_map = self.gw.game_map

        # Check bounds and visibility
        if not (0 <= world_x < game_map.width and 0 <= world_y < game_map.height):
            return False

        if not game_map.visible[world_x, world_y]:
            # Clicked on non-visible tile - deselect
            self.controller.deselect_target()
            return True

        # Check for actor at click location
        # Use spatial index to find all actors, since get_actor_at_location
        # prioritizes blocking actors (player) over non-blocking (ItemPile)
        actor_at_click = None
        for actor in self.gw.actor_spatial_index.get_at_point(world_x, world_y):
            if actor is not self.player:
                actor_at_click = actor
                break
        if actor_at_click is not None:
            # Skip dead characters
            if (
                isinstance(actor_at_click, Character)
                and not actor_at_click.health.is_alive()
            ):
                self.controller.deselect_target()
                return True
            # Toggle selection: deselect if already selected, otherwise select
            if self.controller.selected_target is actor_at_click:
                self.controller.deselect_target()
            else:
                self.controller.select_target(actor_at_click)
            return True

        # Clicked on empty ground - deselect current target
        self.controller.deselect_target()
        return True

    def _check_equipment_view_click(self, root_tile_pos: RootConsoleTilePos) -> bool:
        """Check if click is within equipment view bounds."""
        if self._fm is None or not hasattr(self._fm, "equipment_view"):
            return False

        equipment_view = self._fm.equipment_view
        tile_x, tile_y = root_tile_pos

        if not (equipment_view.x <= tile_x < equipment_view.x + equipment_view.width):
            return False

        clicked_row = tile_y - equipment_view.y
        if clicked_row < 0 or clicked_row >= equipment_view.height:
            return False

        return equipment_view.handle_click(clicked_row)

    def _has_available_actions(self, target: Actor | WorldTilePos) -> bool:
        """Quickly check if any actions are available for a target."""
        if isinstance(target, Character):
            # Always show menu for visible characters; menu handles empty options.
            return True
        if isinstance(target, Actor):
            return False

        world_x, world_y = target
        game_map = self.gw.game_map
        if not (0 <= world_x < game_map.width and 0 <= world_y < game_map.height):
            return False

        return bool(game_map.visible[world_x, world_y])

    def _convert_mouse_coordinates(
        self, event: input_events.MouseState
    ) -> input_events.MouseState:
        """Convert event pixel coordinates to root console tile coordinates."""
        px_pos: PixelPos = event.position
        px_x: PixelCoord = px_pos[0]
        px_y: PixelCoord = px_pos[1]

        graphics = self.controller.graphics
        scale_x, scale_y = graphics.get_display_scale_factor()
        scaled_px_x: PixelCoord = px_x * scale_x
        scaled_px_y: PixelCoord = px_y * scale_y
        root_tile_x, root_tile_y = graphics.pixel_to_tile(scaled_px_x, scaled_px_y)

        event_copy = copy.copy(event)
        event_copy.position = input_events.Point(root_tile_x, root_tile_y)
        return event_copy

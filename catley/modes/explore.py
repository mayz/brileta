"""ExploreMode - Default gameplay mode for exploration and interaction.

ExploreMode is the base layer that the game starts in. Other modes (CombatMode,
future LockpickingMode, etc.) layer on top by falling back to ExploreMode for
common functionality like movement, inventory, and weapon switching.

Responsibilities:
- Movement: tracks held movement keys, generates intents via MoveIntentGenerator
- UI commands: I (inventory), Space (action browser), R (reload), 1/2 (slots), etc.
- Mouse interaction: context menus, equipment clicks, action panel hotkeys
- Rendering: selected actor highlight when not in another mode
"""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING

import tcod.event

from catley import colors, config
from catley.events import MessageEvent, publish_event
from catley.game.actions.base import GameIntent
from catley.game.actors import Actor, Character
from catley.input_handler import Keys
from catley.modes.base import Mode
from catley.move_intent_generator import MoveIntentGenerator
from catley.types import (
    PixelCoord,
    PixelPos,
    RootConsoleTilePos,
    WorldTileCoord,
    WorldTilePos,
)
from catley.view.ui.action_browser_menu import ActionBrowserMenu
from catley.view.ui.commands import (
    OpenExistingMenuUICommand,
    OpenMenuUICommand,
    ToggleFullscreenUICommand,
)
from catley.view.ui.context_menu import ContextMenu
from catley.view.ui.dual_pane_menu import DualPaneMenu, ExternalInventory
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
            tcod.event.KeySym.UP,
            tcod.event.KeySym.DOWN,
            tcod.event.KeySym.LEFT,
            tcod.event.KeySym.RIGHT,
            Keys.KEY_H,
            Keys.KEY_J,
            Keys.KEY_K,
            Keys.KEY_L,
        }
    )

    def __init__(self, controller: Controller) -> None:
        super().__init__(controller)

        # Movement state
        self.movement_keys: set[tcod.event.KeySym] = set()
        self.move_generator = MoveIntentGenerator(controller)

        # Cache references for convenience (may be None in tests)
        self._cursor_manager = None
        self._fm = None
        self._gw = controller.gw
        self._p = controller.gw.player
        self._app = controller.app

        # These may not be available during __init__ (test environments)
        # We'll access them lazily
        if controller.frame_manager is not None:
            self._fm = controller.frame_manager
            if hasattr(controller.frame_manager, "cursor_manager"):
                self._cursor_manager = controller.frame_manager.cursor_manager

    def enter(self) -> None:
        """Enter explore mode - set default cursor."""
        super().enter()
        if self._cursor_manager is not None:
            self._cursor_manager.set_active_cursor_type("arrow")

    def _exit(self) -> None:
        """Exit explore mode - clear movement keys."""
        self.movement_keys.clear()
        super()._exit()

    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle explore mode input.

        Returns True if the event was consumed.
        """
        # Track movement keys
        if isinstance(event, tcod.event.KeyDown) and event.sym in self.MOVEMENT_KEYS:
            if not self.movement_keys:
                # First key press - record time for latency metric
                self.controller.last_input_time = time.perf_counter()
                self.controller.action_count_for_latency_metric = 0
            self.movement_keys.add(event.sym)
            return True

        if isinstance(event, tcod.event.KeyUp) and event.sym in self.MOVEMENT_KEYS:
            self.movement_keys.discard(event.sym)
            return True

        # Check action panel hotkeys first (for immediate action execution)
        if self._handle_action_panel_hotkey(event):
            return True

        # Handle UI commands
        match event:
            case tcod.event.KeyDown(sym=Keys.KEY_I):
                self._open_inventory()
                return True

            case tcod.event.KeyDown(sym=tcod.event.KeySym.GRAVE):
                self._toggle_dev_console()
                return True

            case tcod.event.KeyDown(sym=key_sym, mod=key_mod) if (
                key_sym == tcod.event.KeySym.QUESTION
                or (
                    key_sym == tcod.event.KeySym.SLASH
                    and (key_mod & tcod.event.Modifier.SHIFT)
                )
            ):
                self._open_help()
                return True

            case tcod.event.KeyDown(sym=tcod.event.KeySym.SPACE):
                self._open_action_browser()
                return True

            case tcod.event.KeyDown(sym=Keys.KEY_T):
                self.controller.enter_combat_mode()
                return True

            case tcod.event.KeyDown(sym=Keys.KEY_R):
                self._reload_weapon()
                return True

            case tcod.event.KeyDown(sym=tcod.event.KeySym.N1):
                self._switch_weapon_slot(0)
                return True

            case tcod.event.KeyDown(sym=tcod.event.KeySym.N2):
                self._switch_weapon_slot(1)
                return True

            case tcod.event.KeyDown(sym=tcod.event.KeySym.RETURN, mod=mod) if (
                mod & tcod.event.Modifier.ALT
            ):
                ToggleFullscreenUICommand(self._app).execute()
                return True

            case tcod.event.MouseButtonDown():
                return self._handle_mouse_click(event)

        return False

    def update(self) -> None:
        """Generate movement intent if keys held."""
        # Don't generate movement if player is dead
        if self._p.health and not self._p.health.is_alive():
            return

        # Always call generate_intent to maintain timing state.
        # When no keys are held, generate_intent resets is_first_move_of_burst
        # so the next key press gets the initial delay instead of fast repeat.
        intent = self.move_generator.generate_intent(self.movement_keys)
        if intent:
            # Energy gate: player must afford the action
            if self._p.energy.can_afford(config.ACTION_COST):
                self.controller._execute_player_action_immediately(intent)
            else:
                # Player tried to move but can't afford it - time passes anyway
                self.controller.turn_manager.on_player_action()

    def render_world(self) -> None:
        """Render selected actor highlight."""
        # This is the default rendering when no other mode is active
        if self._fm is not None and hasattr(self._fm, "world_view"):
            self._fm.world_view._render_selected_actor_highlight()

    # -------------------------------------------------------------------------
    # UI Command Helpers
    # -------------------------------------------------------------------------

    def _open_inventory(self) -> None:
        """Open inventory (loot mode if standing on items)."""
        player = self._p
        if self._gw.has_pickable_items_at_location(player.x, player.y):
            source = ExternalInventory((player.x, player.y), "On the ground")
            menu = DualPaneMenu(self.controller, source=source)
            OpenExistingMenuUICommand(self.controller, menu).execute()
        else:
            OpenMenuUICommand(self.controller, DualPaneMenu).execute()

    def _toggle_dev_console(self) -> None:
        """Toggle the dev console overlay."""
        if self._fm is not None:
            dev_console = getattr(self._fm, "dev_console_overlay", None)
            if dev_console is not None:
                self.controller.overlay_system.toggle_overlay(dev_console)

    def _open_help(self) -> None:
        """Open the help menu."""
        OpenMenuUICommand(self.controller, HelpMenu).execute()

    def _open_action_browser(self) -> None:
        """Open the action browser menu."""
        OpenMenuUICommand(self.controller, ActionBrowserMenu).execute()

    def _reload_weapon(self) -> None:
        """Reload the active weapon."""
        active_weapon = self._p.inventory.get_active_weapon()
        if (
            active_weapon
            and active_weapon.ranged_attack
            and active_weapon.ranged_attack.current_ammo
            < active_weapon.ranged_attack.max_ammo
        ):
            from catley.game.actions.combat import ReloadIntent

            reload_action = ReloadIntent(self.controller, self._p, active_weapon)
            self.controller.queue_action(reload_action)
        else:
            publish_event(MessageEvent("Nothing to reload!", colors.GREY))

    def _switch_weapon_slot(self, slot: int) -> None:
        """Switch to a weapon slot."""
        if self._fm is not None and hasattr(self._fm, "equipment_view"):
            self._fm.equipment_view.switch_to_slot(slot)

    # -------------------------------------------------------------------------
    # Action Panel Hotkey Handling
    # -------------------------------------------------------------------------

    def _handle_action_panel_hotkey(self, event: tcod.event.Event) -> bool:
        """Check if a keypress matches an action panel hotkey."""
        if not isinstance(event, tcod.event.KeyDown):
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
        if self._p.health and not self._p.health.is_alive():
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
                action_executed = True
        elif action_option.action_class:
            # Create action instance with static params
            action_instance = action_option.action_class(
                self.controller,
                self._p,
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

        mouse_pos = self._gw.mouse_tile_location_on_map
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

    def _handle_mouse_click(self, event: tcod.event.MouseButtonDown) -> bool:
        """Handle mouse button down events."""
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

        # Shift+left click for pathfinding
        if event.button == tcod.event.MouseButton.LEFT and (
            tcod.event.get_modifier_state() & tcod.event.Modifier.SHIFT
        ):
            if world_tile_pos is not None:
                self.controller.start_actor_pathfinding(self._p, world_tile_pos)
            return True

        # Right click for context menu
        if event.button == tcod.event.MouseButton.RIGHT:
            return self._handle_right_click(world_tile_pos, root_tile_pos)

        # Left click handling
        if event.button == tcod.event.MouseButton.LEFT:
            return self._handle_left_click(root_tile_pos)

        return False

    def _handle_right_click(
        self,
        world_tile_pos: WorldTilePos | None,
        root_tile_pos: RootConsoleTilePos,
    ) -> bool:
        """Handle right click for context menu."""
        target: Actor | WorldTilePos | None = None

        if world_tile_pos:
            world_x: WorldTileCoord
            world_y: WorldTileCoord
            world_x, world_y = world_tile_pos
            game_map = self._gw.game_map

            if (
                0 <= world_x < game_map.width
                and 0 <= world_y < game_map.height
                and game_map.visible[world_x, world_y]
            ):
                actor_at_click = self._gw.get_actor_at_location(world_x, world_y)
                if actor_at_click is not None and (
                    not isinstance(actor_at_click, Character)
                    or actor_at_click.health.is_alive()
                ):
                    target = actor_at_click
                else:
                    target = (world_x, world_y)

        if target and self._has_available_actions(target):
            context_menu = ContextMenu(self.controller, target, root_tile_pos)
            OpenExistingMenuUICommand(self.controller, context_menu).execute()
            return True

        return True  # Consume right click even if no menu opened

    def _handle_left_click(self, root_tile_pos: RootConsoleTilePos) -> bool:
        """Handle left click on status view or equipment view."""
        # Check if click is on the status view (conditions display)
        if self._check_status_view_click(root_tile_pos):
            return True

        # Check if click is on the equipment view (weapon slots)
        return self._check_equipment_view_click(root_tile_pos)

    def _check_status_view_click(self, root_tile_pos: RootConsoleTilePos) -> bool:
        """Check if click is on a condition row in status view."""
        if self._fm is None or not hasattr(self._fm, "status_view"):
            return False

        status_view = self._fm.status_view
        tile_x, tile_y = root_tile_pos

        if not (status_view.x <= tile_x < status_view.x + status_view.width):
            return False

        clicked_row = tile_y - status_view.y
        if clicked_row < 0:
            return False

        num_effects = status_view._num_status_effects_displayed
        num_conditions = status_view._num_conditions_displayed
        condition_start_row = num_effects
        condition_end_row = num_effects + num_conditions

        if condition_start_row <= clicked_row < condition_end_row:
            OpenMenuUICommand(self.controller, DualPaneMenu).execute()
            return True

        return False

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
        from catley.game.actions.discovery import ActionDiscovery

        disc = ActionDiscovery()

        if isinstance(target, Character):
            options = disc.get_options_for_target(self.controller, self._p, target)
            return bool(options)
        if isinstance(target, Actor):
            return False

        world_x, world_y = target
        game_map = self._gw.game_map
        if not (0 <= world_x < game_map.width and 0 <= world_y < game_map.height):
            return False

        return bool(game_map.visible[world_x, world_y])

    def _convert_mouse_coordinates(
        self, event: tcod.event.MouseState
    ) -> tcod.event.MouseState:
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
        event_copy.position = tcod.event.Point(root_tile_x, root_tile_y)
        return event_copy

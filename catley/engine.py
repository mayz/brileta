import random

import actions
import colors
import conditions
import items
import tcod
import tcod.event
from clock import Clock
from fov import FieldOfView
from menu_system import HelpMenu, InventoryMenu, MenuSystem, PickupMenu
from message_log import MessageLog
from model import Actor, Model, WastoidActor
from play_mode import PlayMode
from render import Renderer
from tcod.console import Console
from tcod.context import Context


class Controller:
    def __init__(self, context: Context, root_console: Console) -> None:
        self.context = context
        self.root_console = root_console

        self.help_height = 1  # One line for help text

        self.map_width = 80
        self.map_height = 43

        self.max_room_size = 20
        self.min_room_size = 6
        self.max_num_rooms = 3

        self.model = Model(self.map_width, self.map_height)

        rooms = self.model.game_map.make_map(
            self.max_num_rooms,
            self.min_room_size,
            self.max_room_size,
            self.map_width,
            self.map_height,
        )
        first_room = rooms[0]
        self.model.player.x, self.model.player.y = first_room.center()

        # Initialize Message Log
        self.message_log = MessageLog()

        # Initialize FOV after map is created but before renderer
        self.fov = FieldOfView(self.model)

        # Initialize clock for frame timing
        self.clock = Clock()
        self.target_fps = 60

        # Initialize menu system
        self.menu_system = MenuSystem(self)

        # Create renderer after FOV is initialized
        self.renderer = Renderer(
            screen_width=self.root_console.width,
            screen_height=self.root_console.height,
            model=self.model,
            fov=self.fov,
            clock=self.clock,
            message_log=self.message_log,
            menu_system=self.menu_system,
            context=self.context,
            root_console=self.root_console,
        )

        self.model.player.inventory.append(
            conditions.Injury(injury_type="Sprained Ankle")
        )
        self.model.player.inventory.append(items.SLEDGEHAMMER)

        self._add_npc()

        # Ensure initial FOV computation
        self.fov.fov_needs_recomputing = True

        # For handling input events in run_game_loop().
        self.event_handler = EventHandler(self)

    def _add_npc(self) -> None:
        """Add an NPC to the map."""
        # Place NPC right next to the player.
        npc_x = self.model.player.x
        npc_y = self.model.player.y
        choice = random.randint(1, 4)
        match choice:
            case 1:
                npc_x += 2
            case 2:
                npc_x -= 2
            case 3:
                npc_y += 2
            case 4:
                npc_y -= 2
        npc = WastoidActor(
            x=npc_x,
            y=npc_y,
            ch="T",
            name="Trog",
            color=colors.DARK_GREY,
            model=self.model,
            blocks_movement=True,
            weirdness=3,
            strength=3,
            toughness=3,
            intelligence=-3,
        )
        npc.equipped_weapon = items.SLEDGEHAMMER
        self.model.entities.append(npc)

    def run_game_loop(self) -> None:
        while True:
            # Process any pending events
            for event in tcod.event.get():
                event_with_tile_coords = self.context.convert_event(event)
                self.event_handler.dispatch(event_with_tile_coords)

            # Update using clock's delta time
            delta_time = self.clock.sync(fps=self.target_fps)

            # Update animations and render
            self.model.lighting.update(delta_time)
            self.renderer.render_all()


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
            try:
                action.execute()
            except SystemExit:  # Allow SystemExit to propagate for quitting
                raise
            except Exception as e:
                error_message = f"Error: {str(e)}"
                self.controller.message_log.add_message(error_message, colors.RED)
                print(f"Unhandled exception during action execution: {e}")

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
                return self._handle_movement_input(0, -1)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.DOWN)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.j)
            ):
                return self._handle_movement_input(0, 1)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.LEFT)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.h)
            ):
                return self._handle_movement_input(-1, 0)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.RIGHT)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.l)
            ):
                return self._handle_movement_input(1, 0)

            # Menu keys
            case tcod.event.KeyDown(sym=tcod.event.KeySym.i):
                # Quick access to inventory
                inventory_menu = InventoryMenu(self.controller)
                self.controller.menu_system.show_menu(inventory_menu)
                return None

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
                help_menu = HelpMenu(self.controller)
                self.controller.menu_system.show_menu(help_menu)
                return None

            # Toggle Combat Mode
            case tcod.event.KeyDown(sym=tcod.event.KeySym.c):
                return actions.ToggleCombatModeAction(self.controller)

            case tcod.event.KeyDown(sym=tcod.event.KeySym.g):
                player_x, player_y = self.p.x, self.p.y
                if self.m.has_pickable_items_at_location(player_x, player_y):
                    pickup_menu = PickupMenu(self.controller, (player_x, player_y))
                    self.controller.menu_system.show_menu(pickup_menu)
                # If no items, pressing 'g' does nothing.
                return None

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

    def _handle_movement_input(self, dx: int, dy: int) -> actions.Action | None:
        """Handle directional movement input based on current game mode."""
        if self.m.play_mode == PlayMode.COMBAT and self.m.combat_manager:
            if self.m.combat_manager.is_turn_based_active():
                # During turn-based combat, only move if it's player's turn
                if self.m.combat_manager.is_player_turn():
                    if self.m.combat_manager.can_move():
                        # Use combat movement (deducts movement points)
                        self.m.combat_manager.move_current_combatant(dx, dy)
                        self.controller.fov.fov_needs_recomputing = True
                        return None

                    self.controller.message_log.add_message(
                        "No movement points left!", colors.YELLOW
                    )
                    return None

                # Not player's turn
                return None

            # Planning phase - normal movement works
            return actions.MoveAction(self.controller, self.p, dx, dy)

        # Normal exploration movement
        return actions.MoveAction(self.controller, self.p, dx, dy)

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
            self.m.selected_entity = None
            return None
        mx, my = map_coords
        entity_at_click = self.m.get_entity_at_location(mx, my)

        if self.m.play_mode == PlayMode.COMBAT:
            if self.m.combat_manager.is_planning_phase():
                if (
                    entity_at_click
                    and isinstance(entity_at_click, Actor)
                    and entity_at_click != self.m.player
                    and entity_at_click.is_alive()
                    and self.controller.fov.contains(mx, my)
                ):
                    # Start turn-based combat with this target
                    self.m.combat_manager.begin_turn_based_combat(entity_at_click)
                    return None  # This is handled by combat manager
            elif (
                self.m.combat_manager.is_turn_based_active()
                and self.m.combat_manager.is_player_turn()
            ) and (
                entity_at_click
                and isinstance(entity_at_click, Actor)
                and entity_at_click != self.m.player
                and entity_at_click.is_alive()
                and self.controller.fov.contains(mx, my)
            ):
                # Player's turn in active combat - clicking to attack

                # Player wants to attack this target
                self.m.combat_manager._perform_player_attack_action(entity_at_click)
                self.m.combat_manager.end_current_turn()
                return None

            # During any combat mode, clicking still selects entities for UI
            if entity_at_click:
                if self.m.selected_entity == entity_at_click:
                    self.m.selected_entity = None
                else:
                    self.m.selected_entity = entity_at_click
            else:
                self.m.selected_entity = None
            return None

        # Normal (non-combat) mode handling
        if entity_at_click:
            if self.m.selected_entity == entity_at_click:
                # Clicked on already selected entity. Deselect it.
                self.m.selected_entity = None
            else:
                self.m.selected_entity = entity_at_click
            return None  # UI action, not a game turn action
        # Clicked on an empty tile
        self.m.selected_entity = None
        return None  # UI action, not a game turn action

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


def main() -> None:
    screen_width = 80
    screen_height = 50
    title = "Catley Prototype"

    tileset = tcod.tileset.load_tilesheet(
        "Taffer_20x20.png",
        columns=16,
        rows=16,
        charmap=tcod.tileset.CHARMAP_CP437,
    )

    root_console = Console(screen_width, screen_height, order="F")

    with tcod.context.new(
        console=root_console,
        tileset=tileset,
        title=title,
        vsync=True,
    ) as context:
        controller = Controller(context, root_console)
        try:
            controller.run_game_loop()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise


if __name__ == "__main__":
    main()

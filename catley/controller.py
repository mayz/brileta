import random

import tcod.constants
import tcod.event
import tcod.map
from tcod.console import Console
from tcod.context import Context

from . import colors
from .event_handler import EventHandler
from .game import conditions, items
from .game.actions import GameAction
from .game.components import StatsComponent
from .game.entities import Actor, Disposition, Entity
from .render.render import Renderer
from .ui.menu_system import MenuSystem
from .ui.message_log import MessageLog
from .util.clock import Clock
from .world.game_state import GameWorld


class Controller:
    """
    Orchestrates the main game loop and connects all other systems
    (GameWorld, Renderer, EventHandler, MenuSystem, MessageLog).

    Holds instances of the major game components and provides a central point
    of access for them. Responsible for high-level game flow, such as processing
    player actions and then triggering updates for all other game entities.
    """

    def __init__(self, context: Context, root_console: Console) -> None:
        self.context = context
        self.root_console = root_console

        self.help_height = 1  # One line for help text

        self.map_width = 80
        self.map_height = 43

        self.max_room_size = 20
        self.min_room_size = 6
        self.max_num_rooms = 3

        # Field of view (FOV) configuration.
        self.fov_algorithm = tcod.constants.FOV_SYMMETRIC_SHADOWCAST
        self.fov_light_walls = True
        self.fov_radius = 15

        self.gw = GameWorld(self.map_width, self.map_height)

        rooms = self.gw.game_map.make_map(
            self.max_num_rooms,
            self.min_room_size,
            self.max_room_size,
            self.map_width,
            self.map_height,
        )
        first_room = rooms[0]
        self.gw.player.x, self.gw.player.y = first_room.center()

        # Initialize Message Log
        self.message_log = MessageLog()

        # Initialize clock for frame timing
        self.clock = Clock()
        self.target_fps = 60

        # Initialize menu system
        self.menu_system = MenuSystem(self)

        # Create renderer after FOV is initialized
        self.renderer = Renderer(
            screen_width=self.root_console.width,
            screen_height=self.root_console.height,
            game_world=self.gw,
            clock=self.clock,
            message_log=self.message_log,
            menu_system=self.menu_system,
            context=self.context,
            root_console=self.root_console,
        )

        self.gw.player.inventory.add_to_inventory(
            conditions.Injury(injury_type="Sprained Ankle")
        )
        self.gw.player.inventory.add_to_inventory(items.SLEDGEHAMMER.clone())

        self._add_npc()

        # Initial FOV computation.
        self.update_fov()

        # For handling input events in run_game_loop().
        self.event_handler = EventHandler(self)

        # The game alternates between two states: waiting for the player to take an
        # action and resolving the consequences of their last action (i.e., letting
        # all NPCs and other entities take their turns). When False, the game is busy
        # updating the game state, and player input is ignored.
        self.waiting_for_player_action: bool = True

    def update_fov(self) -> None:
        """Recompute the visible area based on the player's point of view."""
        self.gw.game_map.visible[:] = tcod.map.compute_fov(
            self.gw.game_map.tiles["transparent"],
            (self.gw.player.x, self.gw.player.y),
            radius=self.fov_radius,
            light_walls=self.fov_light_walls,
            algorithm=self.fov_algorithm,
        )

        # If a tile is "visible" it should be added to "explored"
        self.gw.game_map.explored |= self.gw.game_map.visible

    def _add_npc(self) -> None:
        """Add an NPC to the map."""
        # Place NPC right next to the player.
        npc_x = self.gw.player.x
        npc_y = self.gw.player.y

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

        npc_stats = StatsComponent(
            weirdness=3,
            strength=3,
            toughness=3,
            intelligence=-3,
            # Other abilities (agility, observation, demeanor) will default to 0
        )
        npc = Actor(
            x=npc_x,
            y=npc_y,
            ch="T",
            name="Trog",
            color=colors.DARK_GREY,
            game_world=self.gw,
            blocks_movement=True,
            disposition=Disposition.WARY,
            stats=npc_stats,
        )
        npc.inventory.equipped_weapon = items.SLEDGEHAMMER.clone()
        self.gw.entities.append(npc)

    def run_game_loop(self) -> None:
        while True:
            # Process any pending events
            for event in tcod.event.get():
                event_with_tile_coords = self.context.convert_event(event)
                self.event_handler.dispatch(event_with_tile_coords)

            # Update using clock's delta time
            delta_time = self.clock.sync(fps=self.target_fps)

            # Update animations and render
            self.gw.lighting.update(delta_time)
            self.renderer.render_all()

    def execute_action(self, action: GameAction | None) -> None:
        """
        Executes the given Action instance and manages the game's turn flow.

        This method is the central point for processing any action within the game,
        whether initiated by the player or an NPC.

        The game alternates between two states: waiting for the player to take an
        action and resolving the consequences of their last action (i.e., letting
        all NPCs and other entities take their turns). When False, the game is busy
        updating the game state, and player input is ignored.

        Args:
            action_instance: The Action to be performed. Can be None, in which
                             case the method does nothing.
        """
        if not action:
            return

        is_player_game_turn_action = False
        if action.entity == self.gw.player:
            # Check if the entity performing the action is the player
            is_player_game_turn_action = True

        if is_player_game_turn_action and not self.waiting_for_player_action:
            # Player trying to take an action while the game is busy processing.
            # Ignore it.
            return

        try:
            action.execute()
        except SystemExit:
            raise
        except Exception as e:
            error_message = (
                f"Error executing action '{type(action).__name__}': {str(e)}"
            )
            self.message_log.add_message(error_message, colors.RED)
            print(f"Unhandled exception during action execution: {e}")
            return

        if is_player_game_turn_action:
            # Update FOV after player actions.
            self.update_fov()

        if not is_player_game_turn_action:
            return

        # Now that the player's action has consumed their turn,
        # process the world turn.
        self.waiting_for_player_action = False
        self.process_world_turn()

        if self.gw.player.health.is_alive():
            self.waiting_for_player_action = True
        else:
            # Handle game over: player is dead
            self.message_log.add_message(
                f"{self.gw.player.name} has been defeated! Game Over.",
                colors.RED,
            )
            # Further game over logic would go here (e.g., prevent further input,
            # show game over screen). For now, just *not* setting
            # `waiting_for_player_action`` to True effectively stops player turns.

    def process_world_turn(self) -> None:
        """
        Processes a single turn for all non-player entities in the game world.
        This is typically called after the player has performed a GameTurnAction.
        """
        # Just in case entities get added or removed during iteration,
        # iterate over a copy of the entities list.
        for entity in self.gw.entities.copy():
            if entity == self.gw.player:
                # Player's active turn is handled by input.
                # Passive effects for player (e.g., poison damage, regeneration)
                # could be processed by `entity.update_turn`.
                entity.update_turn(self)
            elif isinstance(entity, Entity):
                entity.update_turn(self)

            if not self.gw.player.health.is_alive():
                # If player died, no need to continue processing other entities' turns
                # The game over state will be handled when execute_action resets
                # awaiting_player_game_action after this process_world_turn completes.
                break

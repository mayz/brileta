import random

import colors
import conditions
import items
import tcod.event
from actions import Action, GameTurnAction
from clock import Clock
from event_handler import EventHandler
from fov import FieldOfView
from menu_system import MenuSystem
from message_log import MessageLog
from model import CatleyActor, Disposition, Entity, Model
from render import Renderer
from tcod.console import Console
from tcod.context import Context


class Controller:
    """
    Orchestrates the main game loop and connects all other systems
    (Model, Renderer, EventHandler, MenuSystem, MessageLog).

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

        # The game alternates between two states: waiting for the player to take an
        # action and resolving the consequences of their last action (i.e., letting
        # all NPCs and other entities take their turns). When False, the game is busy
        # updating the game state, and player input is ignored.
        self.waiting_for_player_action: bool = True

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
        npc = CatleyActor(
            x=npc_x,
            y=npc_y,
            ch="T",
            name="Trog",
            color=colors.DARK_GREY,
            model=self.model,
            blocks_movement=True,
            disposition=Disposition.WARY,
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

    def execute_action(self, action: Action | None) -> None:
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
        if isinstance(action, GameTurnAction) and action.entity == self.model.player:
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

        if not is_player_game_turn_action:
            return

        # Now that the player's action has consumed their turn,
        # process the world turn.
        self.waiting_for_player_action = False
        self.process_world_turn()

        if self.model.player.is_alive():
            self.waiting_for_player_action = True
        else:
            # Handle game over: player is dead
            self.message_log.add_message(
                f"{self.model.player.name} has been defeated! Game Over.", colors.RED
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
        for entity in self.model.entities.copy():
            if entity == self.model.player:
                # Player's active turn is handled by input.
                # Passive effects for player (e.g., poison damage, regeneration)
                # could be processed by `entity.update_turn`.
                entity.update_turn(self)
            elif isinstance(entity, Entity):
                entity.update_turn(self)

            if not self.model.player.is_alive():
                # If player died, no need to continue processing other entities' turns
                # The game over state will be handled when execute_action resets
                # awaiting_player_game_action after this process_world_turn completes.
                break

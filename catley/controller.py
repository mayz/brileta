import random

import tcod.constants
import tcod.event
import tcod.map
from tcod.console import Console
from tcod.context import Context

from . import colors
from .event_handler import EventHandler
from .game import conditions, items
from .game.actions import GameAction, MoveAction
from .game.ai import DispositionBasedAI
from .game.components import StatsComponent
from .game.entities import Actor, Disposition
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

        self.action_cost = 100

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
            ai=DispositionBasedAI(),
            speed=80,
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
        Executes the given Action instance.

        Args:
            action_instance: The Action to be performed. Can be None, in which
                             case the method does nothing.
        """
        if not action:
            return  # Do nothing if no action is provided

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

        # Update FOV only if the player moved
        if action.entity == self.gw.player and isinstance(action, MoveAction):
            self.update_fov()

    def process_unified_round(self) -> None:
        """
        Processes a single round where all actors can act.
        Uses a frequency-based system where faster entities act more often.
        """
        # Let actors spend their accumulated points to take actions
        max_iterations = 50  # Prevent infinite loops
        iteration = 0

        # 1. Give all actors energy once per round
        for entity in self.gw.entities:
            if isinstance(entity, Actor):
                entity.regenerate_energy()

        while iteration < max_iterations:
            someone_acted = False
            iteration += 1

            # Check each entity to see if they can act
            for entity in self.gw.entities.copy():
                if isinstance(entity, Actor) and entity.can_afford_action(
                    self.action_cost
                ):
                    action = entity.get_next_action(self)
                    # Always spend energy if you can afford to act. No hoarding!
                    # This ensures speed relationships remain consistent.
                    entity.spend_energy(self.action_cost)

                    if action:
                        self.execute_action(action)
                        someone_acted = True

                    if not self.gw.player.health.is_alive():
                        # Player died, handle game over and stop processing
                        self.message_log.add_message(
                            f"{self.gw.player.name} has been defeated! Game Over.",
                            colors.RED,
                        )
                        # Further game over logic (e.g., prevent input)
                        # could be handled by setting a game over flag here
                        # or in the main loop. For now, returning stops this round.
                        return

            # If nobody acted this round, we're done
            if not someone_acted:
                break

        # After the round, call update_turn for any passive effects
        # (like poison, regeneration) for all actors.
        for entity in self.gw.entities:
            if isinstance(entity, Actor):
                entity.update_turn(self)

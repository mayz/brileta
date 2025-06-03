import tcod.event
import tcod.map
from tcod.console import Console

from catley.game.actors import Character

from . import config
from .event_handler import EventHandler
from .game import conditions
from .game.actions import GameAction
from .render.render import Renderer
from .turn_manager import TurnManager
from .ui.menu_core import MenuSystem
from .ui.message_log import MessageLog
from .util.clock import Clock
from .world.game_state import GameWorld


class Controller:
    """
    Orchestrates the main game loop and connects all other systems
    (GameWorld, Renderer, EventHandler, MenuSystem, MessageLog).

    Holds instances of the major game components and provides a central point
    of access for them. Responsible for high-level game flow, such as processing
    player actions and then triggering updates for all other game actors.
    """

    def __init__(
        self,
        context: tcod.context.Context,
        root_console: Console,
        tile_dimensions: tuple[int, int],
    ) -> None:
        self.context = context
        self.root_console = root_console

        self.help_height = config.HELP_HEIGHT

        self.map_width = config.MAP_WIDTH
        self.map_height = config.MAP_HEIGHT

        self.max_room_size = config.MAX_ROOM_SIZE
        self.min_room_size = config.MIN_ROOM_SIZE
        self.max_num_rooms = config.MAX_NUM_ROOMS

        # Field of view (FOV) configuration.
        self.fov_algorithm = config.FOV_ALGORITHM
        self.fov_light_walls = config.FOV_LIGHT_WALLS
        self.fov_radius = config.FOV_RADIUS

        self.action_cost = config.ACTION_COST

        self.gw = GameWorld(self.map_width, self.map_height)

        rooms = self.gw.game_map.make_map(
            self.max_num_rooms,
            self.min_room_size,
            self.max_room_size,
        )
        first_room = rooms[0]
        self.gw.player.x, self.gw.player.y = first_room.center()

        self.turn_manager = TurnManager(self)

        # Initialize Message Log
        self.message_log = MessageLog()

        # Initialize clock for frame timing
        self.clock = Clock()
        self.target_fps = config.TARGET_FPS

        # Initialize menu system
        self.menu_system = MenuSystem(self)

        # Create renderer after FOV is initialized
        self.renderer = Renderer(
            self,
            screen_width=self.root_console.width,
            screen_height=self.root_console.height,
            game_world=self.gw,
            clock=self.clock,
            message_log=self.message_log,
            menu_system=self.menu_system,
            context=self.context,
            root_console=self.root_console,
            tile_dimensions=tile_dimensions,
        )

        self.gw.player.inventory.add_to_inventory(
            conditions.Injury(injury_type="Sprained Ankle")
        )

        self.gw.populate_npcs(rooms)

        # Initial FOV computation.
        self.update_fov()

        # For handling input events in run_game_loop().
        self.event_handler = EventHandler(self)

    def update_fov(self) -> None:
        """Recompute the visible area based on the player's point of view."""
        self.gw.game_map.visible[:] = tcod.map.compute_fov(
            self.gw.game_map.transparent,
            (self.gw.player.x, self.gw.player.y),
            radius=self.fov_radius,
            light_walls=self.fov_light_walls,
            algorithm=self.fov_algorithm,
        )

        # If a tile is "visible" it should be added to "explored"
        self.gw.game_map.explored |= self.gw.game_map.visible

    def run_game_loop(self) -> None:
        while True:
            for event in tcod.event.get():
                self.event_handler.dispatch(event)

            # Update using clock's delta time
            delta_time = self.clock.sync(fps=self.target_fps)

            # Update animations and render
            self.gw.lighting.update(delta_time)
            self.renderer.render_all()

    def queue_action(self, action: GameAction) -> None:
        """
        Queue a game action to be processed on the next turn.

        Use this when you want an action to consume a turn and trigger
        the unified round system (giving all actors a chance to act).

        Args:
            action: The GameAction to queue for execution
        """
        self.turn_manager.queue_action(action)

    def process_unified_round(self) -> None:
        """
        Processes a single round where all actors can act.
        Uses a frequency-based system where faster actors act more often.
        """
        self.turn_manager.process_unified_round()

    ######
    # FIXME: Move the methods below into a dedicated TargetingMode class.
    #        And eventually a base Mode class for ConversationMode, PickpocketMode, etc.
    #        Right now, this logic is scattered across GameState, Controller,
    #        EventHandler, and Renderer.
    # # catley/modes/targeting.py
    # class TargetingMode:
    # def __init__(self, controller):
    #     self.controller = controller
    #     self.active = False
    #     self.candidates = []
    #     self.current_index = 0
    #     self.last_targeted = None

    # def enter(self): ...
    # def exit(self): ...
    # def cycle_target(self, direction): ...
    # def get_current_target(self): ...
    # def update_for_dead_actors(self): ...
    # def handle_attack(self): ...
    #####

    def enter_targeting_mode(self) -> None:
        """Enter targeting mode and find all valid targets."""
        self.gw.targeting_mode = True
        self.gw.targeting_candidates = []

        # Find all valid targets within reasonable range
        for actor in self.gw.actors:
            if (
                actor != self.gw.player
                and isinstance(actor, Character)
                and actor.health.is_alive()
            ):
                from catley.game import range_system

                distance = range_system.calculate_distance(
                    self.gw.player.x, self.gw.player.y, actor.x, actor.y
                )
                if distance <= 15:
                    self.gw.targeting_candidates.append(actor)

        # Sort by distance (closest first)
        self.gw.targeting_candidates.sort(
            key=lambda a: self._calculate_distance_to_player(a)
        )

        # Use last targeted actor if available, otherwise use current selection
        preferred_target = self.gw.last_targeted_actor or self.gw.selected_actor

        if preferred_target and preferred_target in self.gw.targeting_candidates:
            assert isinstance(preferred_target, Character)
            self.gw.targeting_index = self.gw.targeting_candidates.index(
                preferred_target
            )
            self.gw.selected_actor = preferred_target
        else:
            self.gw.targeting_index = 0
            if self.gw.targeting_candidates:
                self.gw.selected_actor = self.gw.targeting_candidates[0]

    def exit_targeting_mode(self) -> None:
        """Exit targeting mode."""
        # Remember who we were targeting
        if self.gw.selected_actor:
            selected = self.gw.selected_actor
            assert isinstance(selected, Character)
            self.gw.last_targeted_actor = selected

        self.gw.targeting_mode = False
        self.gw.targeting_candidates = []
        self.gw.targeting_index = 0
        self.gw.selected_actor = None

    def update_targeting_for_dead_actors(self) -> None:
        """Remove dead actors from targeting and update current target."""
        if not self.gw.targeting_mode:
            return

        # Remove dead actors from candidates
        self.gw.targeting_candidates = [
            actor for actor in self.gw.targeting_candidates if actor.health.is_alive()
        ]

        # If current target is dead or no candidates left, exit targeting
        current_target = self.get_current_target()
        if not self.gw.targeting_candidates or (
            current_target and not current_target.health.is_alive()
        ):
            self.exit_targeting_mode()
            return

        # Adjust index if needed
        if self.gw.targeting_index >= len(self.gw.targeting_candidates):
            self.gw.targeting_index = 0

        # Update selected actor
        if self.gw.targeting_candidates:
            self.gw.selected_actor = self.gw.targeting_candidates[
                self.gw.targeting_index
            ]

    def get_current_target(self) -> Character | None:
        """Get currently targeted actor."""
        if (
            self.gw.targeting_mode
            and self.gw.targeting_candidates
            and 0 <= self.gw.targeting_index < len(self.gw.targeting_candidates)
        ):
            return self.gw.targeting_candidates[self.gw.targeting_index]
        return None

    def cycle_target(self, direction: int = 1) -> None:
        """Cycle to next/previous target. direction: 1 for next, -1 for previous."""
        if not self.gw.targeting_candidates:
            return

        self.gw.targeting_index = (self.gw.targeting_index + direction) % len(
            self.gw.targeting_candidates
        )
        self.gw.selected_actor = self.gw.targeting_candidates[self.gw.targeting_index]

    def _calculate_distance_to_player(self, actor) -> int:
        from catley.game import range_system

        return range_system.calculate_distance(
            self.gw.player.x, self.gw.player.y, actor.x, actor.y
        )

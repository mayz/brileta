import tcod.event
import tcod.map
import tcod.sdl.mouse
from tcod.console import Console

from . import config
from .event_handler import EventHandler
from .game.actions import GameAction
from .game.actors import Character
from .modes.base import Mode
from .modes.targeting import TargetingMode
from .render.old_render import Renderer
from .turn_manager import TurnManager
from .ui.menu_core import MenuSystem
from .util.clock import Clock
from .util.message_log import MessageLog
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
        self.coordinate_converter = self.renderer.coordinate_converter

        self.gw.populate_npcs(rooms)

        # Initial FOV computation.
        self.update_fov()

        # For handling input events in run_game_loop().
        self.event_handler = EventHandler(self)

        # Initialize mode system
        self.targeting_mode = TargetingMode(self)
        self.active_mode: Mode | None = None

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
        try:
            # Hide the system mouse cursor. We'll draw our own.
            tcod.sdl.mouse.show(False)

            while True:
                for event in tcod.event.get():
                    self.event_handler.dispatch(event)

                # Update using clock's delta time
                delta_time = self.clock.sync(fps=self.target_fps)

                # Update animations and render
                self.gw.lighting.update(delta_time)
                self.renderer.render_all()
        finally:
            # Show the system mouse cursor again.
            tcod.sdl.mouse.show(True)

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

    def enter_targeting_mode(self) -> None:
        """Enter targeting mode"""
        self.active_mode = self.targeting_mode
        self.targeting_mode.enter()

    def exit_targeting_mode(self) -> None:
        """Exit targeting mode"""
        if self.active_mode == self.targeting_mode:
            self.targeting_mode.exit()
            self.active_mode = None

    def is_targeting_mode(self) -> bool:
        """Check if currently in targeting mode"""
        return self.active_mode == self.targeting_mode

    def notify_actor_death(self, actor: Character) -> None:
        """Notify active mode about actor death"""
        if self.active_mode:
            self.active_mode.on_actor_death(actor)

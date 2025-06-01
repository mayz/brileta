import copy

import tcod.event
import tcod.map
import tcod.render
import tcod.sdl.render
from tcod.console import Console

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
        sdl_renderer: tcod.sdl.render.Renderer,
        console_render: tcod.render.SDLConsoleRender,
        root_console: Console,
        tileset: tcod.tileset.Tileset,
    ) -> None:
        self.context = context
        self.sdl_renderer = sdl_renderer
        self.console_render = console_render
        self.root_console = root_console

        self.tile_width = tileset.tile_width
        self.tile_height = tileset.tile_height

        renderer_width, renderer_height = sdl_renderer.output_size
        expected_width = root_console.width * self.tile_width
        expected_height = root_console.height * self.tile_height

        self.scale_x = expected_width / renderer_width
        self.scale_y = expected_height / renderer_height

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
            screen_width=self.root_console.width,
            screen_height=self.root_console.height,
            game_world=self.gw,
            clock=self.clock,
            message_log=self.message_log,
            menu_system=self.menu_system,
            sdl_renderer=self.sdl_renderer,
            console_render=self.console_render,
            root_console=self.root_console,
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
                event_with_tile_coords = self._convert_pixel_to_tile_coords(event)
                self.event_handler.dispatch(event_with_tile_coords)

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

    def _convert_pixel_to_tile_coords(self, event):
        """Convert mouse pixel coordinates to tile coordinates
        with scaling correction."""
        if hasattr(event, "position"):
            pixel_x, pixel_y = event.position

            # Apply scaling correction using the calculated factors
            scaled_x = pixel_x * self.scale_x  # Uses real scaling from tileset
            scaled_y = pixel_y * self.scale_y

            # Convert to tile coordinates using actual tile size
            tile_x = int(scaled_x // self.tile_width)
            tile_y = int(scaled_y // self.tile_height)

            # Clamp to valid range
            tile_x = max(0, min(tile_x, self.root_console.width - 1))
            tile_y = max(0, min(tile_y, self.root_console.height - 1))

            # Create new event with converted coordinates
            event_copy = copy.copy(event)
            event_copy.position = (tile_x, tile_y)
            return event_copy

        return event

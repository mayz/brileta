from __future__ import annotations

from typing import TYPE_CHECKING

import tcod.event
import tcod.map
import tcod.sdl.mouse
from tcod.console import Console

from catley.game.resolution.base import ResolutionSystem
from catley.util.coordinates import TileDimensions

if TYPE_CHECKING:
    from catley.game.actions.discovery import CombatIntentCache

from . import config
from .game.actions.base import GameIntent
from .game.game_world import GameWorld
from .game.turn_manager import TurnManager
from .input_handler import InputHandler
from .modes.base import Mode
from .modes.targeting import TargetingMode
from .util.clock import Clock
from .util.message_log import MessageLog
from .view.frame_manager import FrameManager
from .view.render.renderer import Renderer
from .view.ui.overlays import OverlaySystem


class Controller:
    """
    Orchestrates the main game loop and connects all other systems
    (GameWorld, Renderer, InputHandler, OverlaySystem, MessageLog).

    Holds instances of the major game components and provides a central point
    of access for them. Responsible for high-level game flow, such as processing
    player actions and then triggering updates for all other game actors.
    """

    def __init__(
        self,
        context: tcod.context.Context,
        root_console: Console,
        tile_dimensions: TileDimensions,
    ) -> None:
        self.context = context
        self.root_console = root_console

        self.help_height = config.HELP_HEIGHT

        self.map_width = config.MAP_WIDTH
        self.map_height = config.MAP_HEIGHT

        # Field of view (FOV) configuration.
        self.fov_algorithm = config.FOV_ALGORITHM
        self.fov_light_walls = config.FOV_LIGHT_WALLS
        self.fov_radius = config.FOV_RADIUS

        self.action_cost = config.ACTION_COST

        # GameWorld now handles all its own setup internally
        self.gw = GameWorld(self.map_width, self.map_height)

        self.turn_manager = TurnManager(self)

        self.combat_intent_cache: CombatIntentCache | None = None

        # Initialize Message Log
        self.message_log = MessageLog()

        # Initialize clock for frame timing
        self.clock = Clock()
        self.target_fps = config.TARGET_FPS

        # Initialize overlay system
        self.overlay_system = OverlaySystem(self)

        # Create new low-level renderer
        self.renderer = Renderer(context, root_console, tile_dimensions)
        self.coordinate_converter = self.renderer.coordinate_converter

        # Create FrameManager to coordinate rendering
        self.frame_manager = FrameManager(self)

        # Initial FOV computation.
        self.update_fov()

        # For handling input events in run_game_loop().
        self.input_handler = InputHandler(self)

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
                    self.input_handler.dispatch(event)

                # Update using clock's delta time
                delta_time = self.clock.sync(fps=self.target_fps)

                # Update lighting and render using FrameManager
                self.gw.lighting.update(delta_time)
                self.frame_manager.render_frame(delta_time)
        finally:
            # Show the system mouse cursor again.
            tcod.sdl.mouse.show(True)

    def queue_action(self, action: GameIntent) -> None:
        """
        Queue a game action to be processed on the next turn.

        Use this when you want an action to consume a turn and trigger
        the unified round system (giving all actors a chance to act).

        Args:
            action: The intent to queue for execution
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
            self.targeting_mode._exit()
            self.active_mode = None

    def is_targeting_mode(self) -> bool:
        """Check if currently in targeting mode"""
        return self.active_mode == self.targeting_mode

    def create_resolver(self, **kwargs: object) -> ResolutionSystem:
        """Factory method for resolution systems.

        Currently returns a :class:`D20System` but allows future
        customization without changing action code.
        """
        from catley.game.resolution.d20_system import D20System

        return D20System(**kwargs)  # type: ignore[call-arg]

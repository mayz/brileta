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

from . import colors, config
from .events import MessageEvent, publish_event
from .game.actions.base import GameIntent
from .game.actions.movement import MoveIntent  # noqa: F401
from .game.actors.core import Character
from .game.game_world import GameWorld
from .game.pathfinding_goal import PathfindingGoal
from .game.turn_manager import TurnManager
from .input_handler import InputHandler
from .modes.base import Mode
from .modes.targeting import TargetingMode
from .movement_handler import MovementInputHandler
from .util.clock import Clock
from .util.coordinates import WorldTilePos
from .util.message_log import MessageLog
from .util.pathfinding import find_path
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
        self.movement_handler = MovementInputHandler(self)
        self.last_input_time: float | None = None
        self.action_count_for_latency_metric: int = 0

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
                move_intent = self.movement_handler.generate_intent(
                    self.input_handler.movement_keys
                )
                if move_intent:
                    self.queue_action(move_intent)

                # Now, collect all new input for the next frame.
                for event in tcod.event.get():
                    self.input_handler.dispatch(event)

                # Process all pending game turns as fast as possible before rendering.
                while self.turn_manager.has_pending_actions():
                    # Increment the counter for each action processed
                    self.action_count_for_latency_metric += 1
                    self.process_unified_round()

                # This part is paced by the clock.
                delta_time = self.clock.sync(fps=self.target_fps)
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

    def start_actor_pathfinding(
        self,
        actor: Character,
        target_pos: WorldTilePos,
        final_intent: GameIntent | None = None,
    ) -> bool:
        """Calculate a path and assign a PathfindingGoal to ``actor``.

        Returns ``True`` if a path to ``target_pos`` was found and the goal was
        assigned. Returns ``False`` if no path exists. Any existing goal is
        cleared on failure and the player receives feedback when applicable.
        """

        path = find_path(
            self.gw.game_map,
            self.gw.actor_spatial_index,
            actor,
            (actor.x, actor.y),
            target_pos,
        )

        if path:
            goal = PathfindingGoal(
                target_pos=target_pos,
                final_intent=final_intent,
                _cached_path=path,
            )
            actor.pathfinding_goal = goal
            return True

        self.stop_actor_pathfinding(actor)
        if actor is self.gw.player:
            publish_event(MessageEvent("Path is blocked.", colors.YELLOW))
        return False

    def stop_actor_pathfinding(self, actor: Character) -> None:
        """Immediately cancel ``actor``'s active pathfinding goal."""

        actor.pathfinding_goal = None

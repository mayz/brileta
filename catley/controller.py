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
from .game.actions.types import AnimationType
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
from .view.animation import AnimationManager
from .view.frame_manager import FrameManager
from .view.render.renderer import Renderer
from .view.ui.overlays import OverlaySystem


class Controller:
    """
    Orchestrates the main game loop using the Reactive Actor Framework (RAF).

    The Controller operates as a lightweight dispatcher with a simple 4-step loop:
    1. Process player input immediately (zero-latency responsiveness)
    2. Process one pending NPC action per frame (smooth world reactions)
    3. Update animations (simultaneous, interruptible visual feedback)
    4. Render frame (consistent visual output)

    This architecture eliminates the complex state machine in favor of immediate
    player action processing while maintaining fair energy economy for NPCs.
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

        # Animation manager for handling movement and effects
        self.animation_manager = AnimationManager()

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
            tcod.sdl.mouse.show(False)

            while True:
                delta_time = self.clock.sync(fps=self.target_fps)

                # Step 1: Update frame-independent systems
                self.animation_manager.update(delta_time)
                self.gw.lighting.update(delta_time)

                # Step 2: Process player input immediately (if any)
                for event in tcod.event.get():
                    self.input_handler.dispatch(event)

                # Check for player-initiated actions (movement or queued)
                move_intent = None
                if not self.overlay_system.has_interactive_overlays():
                    move_intent = self.movement_handler.generate_intent(
                        self.input_handler.movement_keys
                    )
                    if move_intent:
                        self._execute_player_action_immediately(move_intent)

                # Check for other queued player actions
                if self.turn_manager.has_pending_actions():
                    player_action = self.turn_manager.dequeue_player_action()
                    if player_action:
                        self._execute_player_action_immediately(player_action)

                # Check for autopilot actions if no manual input
                if (
                    not move_intent
                    and not self.turn_manager.has_pending_actions()
                    and self.turn_manager.is_player_turn_available()
                ):
                    autopilot_action = self.gw.player.get_next_action(self)
                    if autopilot_action:
                        self._execute_player_action_immediately(autopilot_action)

                # Step 3: Process all available NPC actions immediately
                self._process_all_available_npc_actions()

                # Step 4: Render frame
                self.frame_manager.render_frame(delta_time)

        finally:
            tcod.sdl.mouse.show(True)

    def _execute_player_action_immediately(self, action: GameIntent) -> None:
        """Execute a player action immediately on the current frame.

        This is the core of the RAF system - player actions are never queued
        or delayed. They are processed instantly when detected.

        Args:
            action: The player's GameIntent to execute immediately
        """
        # Handle wind-up actions specially (preserve existing PPIAS behavior)
        if action.animation_type == AnimationType.WIND_UP and action.windup_animation:
            self.animation_manager.add(action.windup_animation)
            # For wind-up actions, we still need to wait for completion
            # This preserves the existing wind-up behavior from PPIAS
            # TODO: This may be simplified further in future phases
            return

        # For INSTANT actions, execute immediately.
        # Process turn effects for the acting player only.
        self.gw.player.update_turn(self)

        # Execute the player's action
        self.turn_manager.execute_intent(action)
        self.gw.player.energy.spend(self.action_cost)

        # Update FOV after player action (important for movement)
        self.update_fov()

        # RAF: Trigger immediate NPC scheduling based on the world state change
        self.turn_manager.on_player_action()

    def _process_all_available_npc_actions(self) -> None:
        """Process all NPCs who can currently afford actions immediately.

        This eliminates the artificial frame-rate delay that was breaking speed ratios
        in RAF V1. All NPCs who gained sufficient energy from the last player action
        will act immediately, maintaining perfect proportional speed relationships.
        """
        # Process all available NPC actions from TurnManager
        self.turn_manager.process_all_npc_reactions()

    def _process_one_npc_action_if_available(self) -> None:
        """Process one pending NPC action using the RAF scheduling system.

        DEPRECATED: This method is deprecated in RAF V2.
        Use _process_all_available_npc_actions() instead.
        NPCs now act immediately rather than being queued,
        maintaining perfect speed ratios.

        This method is kept for backwards compatibility during transition.
        """
        # In RAF V2, this method does nothing since NPCs act immediately
        # Left as no-op for backwards compatibility
        pass

    def queue_action(self, action: GameIntent) -> None:
        """
        Queue a game action to be processed on the next turn.

        Use this when you want an action to consume a turn and trigger
        the unified round system (giving all actors a chance to act).

        Args:
            action: The intent to queue for execution
        """
        self.turn_manager.queue_action(action)

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
        """Calculate a path and assign a :class:`PathfindingGoal` to ``actor``.

        If the direct destination is blocked, the search falls back to any
        adjacent walkable tile.  ``False`` is returned if no path can be found.
        """

        gm = self.gw.game_map
        asi = self.gw.actor_spatial_index

        def _best_path(dest: WorldTilePos) -> list[WorldTilePos] | None:
            return find_path(gm, asi, actor, (actor.x, actor.y), dest)

        path = _best_path(target_pos)
        dest = target_pos

        if not path:
            best_path: list[WorldTilePos] | None = None
            best_dest: WorldTilePos | None = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    tx = target_pos[0] + dx
                    ty = target_pos[1] + dy
                    if not (0 <= tx < gm.width and 0 <= ty < gm.height):
                        continue
                    if not gm.walkable[tx, ty]:
                        continue
                    blocker = self.gw.get_actor_at_location(tx, ty)
                    if blocker and blocker.blocks_movement and blocker is not actor:
                        continue
                    candidate = _best_path((tx, ty))
                    if candidate and (
                        best_path is None or len(candidate) < len(best_path)
                    ):
                        best_path = candidate
                        best_dest = (tx, ty)
            if best_path is not None and best_dest is not None:
                path = best_path
                dest = best_dest

        if path:
            actor.pathfinding_goal = PathfindingGoal(
                target_pos=dest,
                final_intent=final_intent,
                _cached_path=path,
            )
            return True

        self.stop_actor_pathfinding(actor)
        if actor is self.gw.player:
            publish_event(MessageEvent("Path is blocked.", colors.YELLOW))
        return False

    def stop_actor_pathfinding(self, actor: Character) -> None:
        """Immediately cancel ``actor``'s active pathfinding goal."""

        actor.pathfinding_goal = None

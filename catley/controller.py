from __future__ import annotations

from typing import TYPE_CHECKING

import tcod.map

from catley.app import App
from catley.game.resolution.base import ResolutionSystem
from catley.sound import SoundSystem

if TYPE_CHECKING:
    from catley.game.actions.discovery import CombatIntentCache

from . import colors, config
from .events import MessageEvent, publish_event
from .game.actions.base import GameIntent
from .game.actions.types import AnimationType
from .game.actors.core import Character
from .game.game_world import GameWorld
from .game.lights import DirectionalLight, DynamicLight
from .game.pathfinding_goal import PathfindingGoal
from .game.turn_manager import TurnManager
from .input_handler import InputHandler
from .modes.base import Mode
from .modes.targeting import TargetingMode
from .movement_handler import MovementInputHandler
from .types import FixedTimestep, InterpolationAlpha
from .util.clock import Clock
from .util.coordinates import WorldTilePos
from .util.live_vars import live_variable_registry, record_time_live_variable
from .util.message_log import MessageLog
from .util.pathfinding import find_local_path, find_region_path
from .view.animation import AnimationManager
from .view.frame_manager import FrameManager
from .view.presentation import PresentationManager
from .view.render.graphics import GraphicsContext
from .view.render.lighting.cpu import CPULightingSystem
from .view.ui.overlays import OverlaySystem


class Controller:
    """
    The central coordinator of the game's logic, state, and subsystems.

    Orchestrates the main application flow by managing a fixed-timestep game loop with
    interpolation. This architecture ensures two key things:

    - Deterministic Game Logic: All game state updates (NPC actions, status effects,
      etc.) occur in discrete, fixed-duration steps (e.g., 60 times per second). This
      makes the simulation predictable and independent of the rendering speed or
      hardware performance.
    - Smooth Visuals: Rendering happens as fast as the hardware allows (up to a target
      FPS cap), and is decoupled from the logic rate. Smooth animations and movement
      are achieved by interpolating object positions and states between the last logic
      step and the current one.

    Key Responsibilities:
    - Loop Management: Owns the game clock and uses an "accumulator" to manage the
      fixed-timestep loop, dispatching calls to `update_logic_step`.
    - System Orchestration: Initializes and holds references to all major subsystems,
      including the `GameWorld` (model), `FrameManager` (view), `InputHandler`, and
      `TurnManager`.
    - Input Processing: Immediately processes player input intents each frame to ensure
      zero-latency responsiveness, separate from the fixed logic loop.
    - Mode Management: Controls the active game mode (e.g., normal, targeting) which
      alters player input and rendering.
    - Factory for Core Systems: Provides factory methods for creating essential game
      systems like `ResolutionSystem`.
    """

    def __init__(self, app: App, graphics: GraphicsContext) -> None:
        self.app = app
        self.gw = GameWorld(config.MAP_WIDTH, config.MAP_HEIGHT)

        # In test environments, use CPU lighting to avoid GPU initialization issues
        # with dummy graphics contexts. This preserves explicit GPU failure behavior
        # in production while allowing tests to run.
        lighting_backend = config.LIGHTING_BACKEND
        if config.IS_TEST_ENVIRONMENT:
            lighting_backend = "cpu"

        match lighting_backend:
            case "cpu":
                self.gw.lighting_system = CPULightingSystem(self.gw)
            case "moderngl":
                from .backends.moderngl.gpu_lighting import GPULightingSystem

                self.gw.lighting_system = GPULightingSystem(self.gw, graphics)
            case "wgpu":
                from .backends.wgpu.gpu_lighting import GPULightingSystem

                self.gw.lighting_system = GPULightingSystem(self.gw, graphics)
            case _:
                raise ValueError(f"Unknown lighting backend: {config.LIGHTING_BACKEND}")

        self.graphics = graphics
        self.coordinate_converter = graphics.coordinate_converter

        self.message_log = MessageLog()
        self.overlay_system = OverlaySystem(self)
        self.frame_manager = FrameManager(self, graphics)
        self.input_handler = InputHandler(app, self)

        # Create the player's light now that lighting system is connected
        player_torch = DynamicLight.create_player_torch(self.gw.player)
        self.gw.add_light(player_torch)

        # Create the sun if enabled
        if config.SUN_ENABLED:
            sun = DirectionalLight.create_sun(
                elevation_degrees=config.SUN_ELEVATION_DEGREES,
                azimuth_degrees=config.SUN_AZIMUTH_DEGREES,
                intensity=config.SUN_INTENSITY,
                color=config.SUN_COLOR,
            )
            self.gw.add_light(sun)

        self.turn_manager = TurnManager(self)

        # Animation manager for handling movement and effects
        self.animation_manager = AnimationManager()

        # Presentation manager for staggered combat feedback
        self.presentation_manager = PresentationManager()

        # Initialize sound system
        self._initialize_sound_system()

        self.combat_intent_cache: CombatIntentCache | None = None

        # Fixed timestep game loop implementation
        # This separates rendering (variable framerate) from game logic (fixed rate)
        self.clock = Clock()
        self.target_fps = config.TARGET_FPS  # Visual framerate cap

        # Game logic runs at exactly 60Hz regardless of visual framerate
        # This ensures consistent physics, movement, and game timing
        # 16.67ms per logic step
        self.fixed_timestep: FixedTimestep = FixedTimestep(1.0 / 60.0)

        # Accumulator tracks excess frame time to "catch up" logic steps
        # When accumulator >= fixed_timestep, we run one logic update
        self.accumulator = 0.0

        # Death spiral protection: Never run more than this many logic steps per frame
        # Prevents performance collapse when rendering framerate drops below timestep
        self.max_logic_steps_per_frame = 5

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
            radius=config.FOV_RADIUS,
            light_walls=config.FOV_LIGHT_WALLS,
            algorithm=config.FOV_ALGORITHM,
        )

        # If a tile is "visible" it should be added to "explored"
        self.gw.game_map.explored |= self.gw.game_map.visible
        self.gw.game_map.exploration_revision += 1

    def register_metrics(self) -> None:
        """Register all performance monitoring metrics with live variable registry."""
        live_variable_registry.register_metric(
            "cpu.logic_step_ms",
            description="CPU time for one fixed logic step",
            num_samples=500,
        )
        live_variable_registry.register_metric(
            "cpu.animation_update_ms",
            description="CPU time for animation updates",
            num_samples=500,
        )
        live_variable_registry.register_metric(
            "cpu.action_processing_ms",
            description="CPU time for action processing",
            num_samples=500,
        )

        # Cascade the call to the FrameManager
        self.frame_manager.register_metrics()

    def process_player_input(self) -> None:
        """
        Processes all pending player input. Should be called once per visual frame
        before the logic step loop to ensure responsiveness.
        """
        # Assert that dependencies have been injected
        assert self.overlay_system is not None and self.input_handler is not None

        # Check for player-initiated actions (movement or queued)
        move_intent = None
        if not self.overlay_system.has_interactive_overlays():
            # Check if player is trying to move
            potential_move = self.movement_handler.generate_intent(
                self.input_handler.movement_keys
            )
            if potential_move:
                # Energy gate: player must afford the action for speed penalties
                # to be perceptible. If they can't afford it, time still passes
                # (energy regenerates) but they don't move.
                if self.gw.player.energy.can_afford(config.ACTION_COST):
                    move_intent = potential_move
                    self._execute_player_action_immediately(move_intent)
                else:
                    # Player tried to move but can't afford it - time passes anyway.
                    # This prevents deadlock where low energy = no regen = stuck.
                    self.turn_manager.on_player_action()

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
                # Energy gate autopilot same as manual movement
                if self.gw.player.energy.can_afford(config.ACTION_COST):
                    self._execute_player_action_immediately(autopilot_action)
                else:
                    # Can't afford action - time passes but no movement
                    self.turn_manager.on_player_action()

    def update_logic_step(self) -> None:
        """
        Runs one fixed-step of game logic for animations and non-player actors.
        Called by the App's fixed-step loop.
        """
        with record_time_live_variable("cpu.logic_step_ms"):
            # SNAPSHOT PREVIOUS STATE: Save current positions for smooth
            # This allows rendering to blend smoothly between old/new
            for actor in self.gw.actors:
                actor.prev_x = actor.x
                actor.prev_y = actor.y

            with record_time_live_variable("cpu.animation_update_ms"):
                # ANIMATIONS: Update frame-independent systems
                # Ensures consistent animation timing regardless of FPS
                self.animation_manager.update(self.fixed_timestep)

            if self.gw.lighting_system is not None:
                self.gw.lighting_system.update(self.fixed_timestep)

            # Update sound system with player position
            if self.gw.player:
                self.sound_system.update(
                    self.gw.player.x,
                    self.gw.player.y,
                    self.gw.actor_spatial_index,
                    self.fixed_timestep,
                )

            # Update presentation manager (dispatches staggered combat feedback)
            self.presentation_manager.update(self.fixed_timestep)

            with record_time_live_variable("cpu.action_processing_ms"):
                # GAME LOGIC: Process all NPC actions for this step
                # This is where the core game simulation happens
                self._process_all_available_npc_actions()

    def render_visual_frame(self, alpha: InterpolationAlpha) -> None:
        """Renders one visual frame with interpolation. Called by the App."""
        assert self.frame_manager is not None
        with record_time_live_variable("cpu.render_ms"):
            # Uses alpha to smoothly blend between prev_* and current
            self.frame_manager.render_frame(alpha)

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
        # CRITICAL: Check prevention BEFORE update_turn() to match NPC behavior.
        # If checked after, 1-duration effects (Staggered) would already expire,
        # allowing the player to act when they should be blocked.
        if self.gw.player.status_effects.is_action_prevented():
            # Player's action is blocked - still counts as their turn
            self.gw.player.update_turn(self)
            self.gw.player.energy.spend(config.ACTION_COST)
            publish_event(MessageEvent("You cannot act!", colors.RED))
            return

        # Process turn effects for the acting player
        self.gw.player.update_turn(self)

        # Execute the player's action
        self.turn_manager.execute_intent(action)
        self.gw.player.energy.spend(config.ACTION_COST)

        # Check for terrain hazard damage after the player completes their action
        self.turn_manager._apply_terrain_hazard(self.gw.player)

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

        Uses hierarchical pathfinding (HPA*) when crossing regions. For paths
        within a single region, uses direct A*. If the direct destination is
        blocked, the search falls back to any adjacent walkable tile.

        Returns ``False`` if no path can be found.
        """
        gm = self.gw.game_map
        asi = self.gw.actor_spatial_index
        start_pos: WorldTilePos = (actor.x, actor.y)

        def _local_path(start: WorldTilePos, dest: WorldTilePos) -> list[WorldTilePos]:
            return find_local_path(gm, asi, actor, start, dest)

        # Try hierarchical pathfinding if regions are set up
        result = self._try_hierarchical_path(actor, start_pos, target_pos)
        if result is not None:
            path, high_level_path, dest = result
            if path:
                actor.pathfinding_goal = PathfindingGoal(
                    target_pos=dest,
                    final_intent=final_intent,
                    high_level_path=high_level_path,
                    _cached_path=path,
                )
                return True

        # Fall back to direct A* (same region or no region info)
        path = _local_path(start_pos, target_pos)
        dest = target_pos

        # If direct path fails, try adjacent tiles
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
                    candidate = _local_path(start_pos, (tx, ty))
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

    def _try_hierarchical_path(
        self,
        actor: Character,
        start_pos: WorldTilePos,
        target_pos: WorldTilePos,
    ) -> tuple[list[WorldTilePos], list[int] | None, WorldTilePos] | None:
        """Attempt hierarchical pathfinding across regions.

        Returns (local_path, high_level_path, effective_dest) if cross-region
        path is found, or None if regions aren't set up or positions are in
        the same region.
        """
        gm = self.gw.game_map
        asi = self.gw.actor_spatial_index

        # Check if regions are set up
        if not gm.regions:
            return None

        # Get region IDs for start and end positions
        start_region_id = gm.tile_to_region_id[start_pos[0], start_pos[1]]
        end_region_id = gm.tile_to_region_id[target_pos[0], target_pos[1]]

        # If either position isn't in a region, or same region, use direct path
        if start_region_id < 0 or end_region_id < 0:
            return None
        if start_region_id == end_region_id:
            return None

        # Find path through region graph
        region_path = find_region_path(gm, start_region_id, end_region_id)
        if region_path is None or len(region_path) < 2:
            return None

        # Get the next region we need to enter (index 1, since 0 is current)
        next_region_id = region_path[1]

        # Get the connection point to the next region
        current_region = gm.regions.get(start_region_id)
        if current_region is None:
            return None

        connection_point = current_region.connections.get(next_region_id)
        if connection_point is None:
            return None

        # Compute local path to the connection point
        local_path = find_local_path(gm, asi, actor, start_pos, connection_point)
        if not local_path:
            return None

        # Store remaining regions to traverse (excluding current region)
        high_level_path = region_path[1:]

        return (local_path, high_level_path, target_pos)

    def stop_actor_pathfinding(self, actor: Character) -> None:
        """Immediately cancel ``actor``'s active pathfinding goal."""

        actor.pathfinding_goal = None

    def _initialize_sound_system(self) -> None:
        """Initialize the sound system and audio backend."""
        # Sound system for managing audio playback
        self.sound_system = SoundSystem()

        # Initialize audio backend if enabled and not in test environment
        if config.AUDIO_ENABLED and not config.IS_TEST_ENVIRONMENT:
            try:
                from pathlib import Path

                from catley.backends.tcod.audio import TCODAudioBackend

                audio_backend = TCODAudioBackend()
                audio_backend.initialize()
                self.sound_system.set_audio_backend(audio_backend)

                # Set assets path for loading sounds
                assets_path = Path(__file__).parent.parent / "assets"
                self.sound_system.set_assets_path(assets_path)
            except Exception as e:
                # Log error but continue without audio
                print(f"Warning: Failed to initialize audio backend: {e}")

    def cleanup(self) -> None:
        """Clean up resources when the controller is being shut down."""
        # Shutdown audio backend if it exists
        if self.sound_system.audio_backend:
            self.sound_system.audio_backend.shutdown()

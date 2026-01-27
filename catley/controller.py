from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import tcod.map

from catley.app import App
from catley.game.resolution.base import ResolutionSystem
from catley.sound import SoundSystem

if TYPE_CHECKING:
    from catley.game.action_plan import ActionPlan
    from catley.game.actions.discovery import CombatIntentCache
    from catley.game.items.item_core import Item

from . import colors, config
from .events import (
    CombatEndedEvent,
    CombatInitiatedEvent,
    MessageEvent,
    SoundEvent,
    publish_event,
    subscribe_to_event,
)
from .game.actions.base import GameIntent
from .game.actions.types import AnimationType
from .game.actors import Actor
from .game.actors.core import Character
from .game.game_world import GameWorld
from .game.lights import DirectionalLight, DynamicLight
from .game.turn_manager import TurnManager
from .input_handler import InputHandler
from .modes.base import Mode
from .modes.combat import CombatMode
from .modes.explore import ExploreMode
from .modes.picker import PickerMode
from .types import FixedTimestep, InterpolationAlpha
from .util.clock import Clock
from .util.coordinates import WorldTilePos
from .util.live_vars import live_variable_registry, record_time_live_variable
from .util.message_log import MessageLog
from .view.animation import AnimationManager
from .view.frame_manager import FrameManager
from .view.presentation import PresentationManager
from .view.render.graphics import GraphicsContext
from .view.ui.overlays import OverlaySystem

logger = logging.getLogger(__name__)


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
    - Mode Management: Controls the active game mode (e.g., explore, combat) which
      alters player input and rendering.
    - Factory for Core Systems: Provides factory methods for creating essential game
      systems like `ResolutionSystem`.
    """

    def __init__(self, app: App, graphics: GraphicsContext) -> None:
        self.app = app
        self.gw = GameWorld(config.MAP_WIDTH, config.MAP_HEIGHT)

        lighting_backend = config.LIGHTING_BACKEND
        if (
            lighting_backend == "moderngl"
            and config.IS_TEST_ENVIRONMENT
            and getattr(graphics, "mgl_context", None) is None
        ):
            import moderngl

            graphics.mgl_context = moderngl.create_context(standalone=True)  # type: ignore[unresolved-attribute]

        match lighting_backend:
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

        self.last_input_time: float | None = None
        self.action_count_for_latency_metric: int = 0

        # Initialize mode system - game always has an active mode
        # Modes are organized in a stack. ExploreMode is always at the bottom.
        # Other modes (CombatMode, PickerMode, etc.) push on top and pop when done.
        self.explore_mode = ExploreMode(self)
        self.combat_mode = CombatMode(self)
        self.picker_mode = PickerMode(self)
        self.mode_stack: list[Mode] = [self.explore_mode]
        self.explore_mode.enter()

        # Subscribe to combat initiation events for auto-entry
        subscribe_to_event(CombatInitiatedEvent, self._on_combat_initiated)

        # Selection-based target tracking for click-to-select interaction.
        # selected_target is sticky (persists until explicitly deselected by
        # clicking empty ground). ActionPanel displays actions for selected_target.
        self.selected_target: Actor | None = None

        # Hover target tracking for visual feedback (outline only, no game state).
        # Updated each frame based on mouse position, used for subtle hover highlight.
        self.hovered_actor: Actor | None = None

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
        """Process all pending player input.

        Should be called once per visual frame before the logic step loop
        to ensure responsiveness. All active modes get update() called to
        support layered behavior (e.g., movement while in combat/picker mode).
        """
        assert self.overlay_system is not None and self.input_handler is not None

        # Let all modes in the stack update (bottom-to-top)
        # This allows ExploreMode to generate movement intents even when
        # CombatMode or PickerMode is on top.
        if not self.overlay_system.has_interactive_overlays():
            for mode in self.mode_stack:
                mode.update()

        # Check for other queued player actions (manual input from UI/keys)
        if self.turn_manager.has_pending_actions():
            player_action = self.turn_manager.dequeue_player_action()
            if player_action:
                # Manual action cancels any active plan and presentation timing.
                # Player input should feel immediately responsive.
                self.stop_plan(self.gw.player)
                self.turn_manager.clear_presentation_timing()
                self.animation_manager.interrupt_player_animations(self.gw.player)
                self._execute_player_action_immediately(player_action)

        # Check for autopilot actions if no manual input.
        # Autopilot respects presentation timing (duration_ms) to create paced movement.
        # No separate rate limiting needed - duration_ms controls the pacing.
        has_movement_keys = bool(self.explore_mode.movement_keys)
        presentation_complete = self.turn_manager.is_presentation_complete()
        if (
            not has_movement_keys
            and presentation_complete
            and not self.turn_manager.has_pending_actions()
            and self.turn_manager.is_player_turn_available()
        ):
            # Get autopilot action - check active_plan first, then pathfinding_goal
            if self.gw.player.active_plan is not None:
                autopilot_action = self.turn_manager._get_intent_from_plan(
                    self.gw.player
                )
            else:
                # Fall back to pathfinding_goal (legacy system)
                autopilot_action = self.gw.player.get_next_action(self)

            if autopilot_action:
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
            self.invalidate_combat_tooltip()
            return

        # Process turn effects for the acting player
        self.gw.player.update_turn(self)

        # Execute the player's action (using execute_player_intent for plan advancement)
        self.turn_manager.execute_player_intent(action)
        self.gw.player.energy.spend(config.ACTION_COST)

        # Check for terrain hazard damage after the player completes their action
        self.turn_manager._apply_terrain_hazard(self.gw.player)

        # Update FOV after player action (important for movement)
        self.update_fov()

        # RAF: Trigger immediate NPC scheduling based on the world state change
        self.turn_manager.on_player_action()
        self.invalidate_combat_tooltip()

        # Refresh hovered actor in case actors moved into/out of the mouse position
        self.update_hovered_actor(self.gw.mouse_tile_location_on_map)

    def update_hovered_actor(self, mouse_pos: WorldTilePos | None) -> None:
        """Update the hovered actor based on mouse position.

        This is used for visual feedback only (subtle hover outline).
        Does not affect game state or ActionPanel.
        """
        if mouse_pos is None:
            self.hovered_actor = None
            return

        mouse_x, mouse_y = mouse_pos
        self.hovered_actor = self._get_visible_actor_at_tile(mouse_x, mouse_y)

    def _get_visible_actor_at_tile(self, x: int, y: int) -> Actor | None:
        """Return the first visible actor at a tile, if any."""
        gm = self.gw.game_map
        if not (0 <= x < gm.width and 0 <= y < gm.height):
            return None
        if not gm.visible[x, y]:
            return None

        actors_at_tile = self.gw.actor_spatial_index.get_at_point(x, y)
        if not actors_at_tile:
            return None

        for actor in sorted(actors_at_tile, key=self._actor_sort_key):
            if actor is not self.gw.player:
                return actor
        return None

    @staticmethod
    def _actor_sort_key(actor: Actor) -> tuple[int, int, str]:
        """Deterministic ordering for multiple actors on the same tile."""
        return (actor.y, actor.x, actor.name)

    def _process_all_available_npc_actions(self) -> None:
        """Process all NPCs who can currently afford actions immediately.

        This eliminates the artificial frame-rate delay that was breaking speed ratios
        in RAF V1. All NPCs who gained sufficient energy from the last player action
        will act immediately, maintaining perfect proportional speed relationships.

        NPC actions wait for presentation timing to complete, creating readable
        sequencing where players can follow cause and effect.
        """
        # Wait for current action's presentation to complete before NPCs act
        if not self.turn_manager.is_presentation_complete():
            return

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

    @property
    def active_mode(self) -> Mode:
        """The mode currently handling input (top of the mode stack)."""
        return self.mode_stack[-1]

    def push_mode(self, mode: Mode) -> None:
        """Push a mode onto the stack and activate it.

        The pushed mode becomes the new active mode and receives all input.
        The mode below remains in the stack but is temporarily inactive.

        Raises:
            RuntimeError: If a mode of the same type is already in the stack.
        """
        if any(type(m) is type(mode) for m in self.mode_stack):
            raise RuntimeError(
                f"A mode of type {type(mode).__name__} is already in the stack"
            )
        # Append before enter() so nested push_mode calls work correctly
        # (e.g., CombatMode.enter() pushing PickerMode)
        self.mode_stack.append(mode)
        mode.enter()

    def pop_mode(self) -> None:
        """Pop the top mode and return to the mode below.

        ExploreMode (the base) cannot be popped - the stack always has at
        least one mode. Calling pop_mode when only ExploreMode remains is a no-op.
        """
        if len(self.mode_stack) > 1:
            old_mode = self.mode_stack.pop()
            old_mode._exit()

    def transition_to_mode(self, new_mode: Mode) -> None:
        """Replace the top mode with a new one (legacy compatibility).

        For stack-based modes (PickerMode, etc.), prefer push_mode/pop_mode.
        This method is kept for the ExploreMode <-> CombatMode transitions
        which conceptually replace rather than stack.

        If the new mode is already in the stack (e.g., transitioning back to
        ExploreMode), this will pop modes until that mode is on top.
        """
        if self.active_mode is new_mode:
            return  # Already in this mode

        # Pop the current top mode
        self.pop_mode()

        # Keep popping if the new mode is already in the stack below
        # (e.g., transitioning back to ExploreMode which is at the base)
        while self.active_mode is not new_mode and new_mode in self.mode_stack:
            self.pop_mode()

        # Only push if the new mode isn't already the active mode
        if self.active_mode is not new_mode:
            self.push_mode(new_mode)

    def enter_combat_mode(self) -> None:
        """Enter combat mode from current mode."""
        self.transition_to_mode(self.combat_mode)

    def exit_combat_mode(self, reason: str = "manual_exit") -> None:
        """Exit combat mode back to explore mode.

        Args:
            reason: Why combat ended - "all_enemies_dead", "manual_exit",
                    or "cancelled". Used for ceremony hooks.
        """
        publish_event(CombatEndedEvent(reason=reason))
        self.transition_to_mode(self.explore_mode)

    def is_combat_mode(self) -> bool:
        """Check if currently in combat mode (combat mode is in the stack).

        Note: When combat mode is entered, it pushes PickerMode on top for
        target selection. So we check if combat_mode is anywhere in the stack,
        not just if it's the active mode.
        """
        return self.combat_mode in self.mode_stack

    def invalidate_combat_tooltip(self) -> None:
        """Refresh the combat tooltip overlay if it is active."""
        if not self.is_combat_mode():
            return

        fm = self.frame_manager
        if fm is None or not hasattr(fm, "combat_tooltip_overlay"):
            return

        tooltip = fm.combat_tooltip_overlay
        if tooltip.is_active:
            tooltip.invalidate()

    def _on_combat_initiated(self, event: CombatInitiatedEvent) -> None:
        """Auto-enter combat mode when combat is initiated.

        Triggered by:
        - NPC attacks player (hit or miss)
        - Player attacks non-hostile NPC
        - Player pushes non-hostile NPC
        - Player's noise alerts nearby NPCs
        """
        if not self.is_combat_mode():
            self.enter_combat_mode()

    def has_visible_hostiles(self) -> bool:
        """Check if any visible NPCs are hostile toward the player.

        Used to warn when exiting combat mode with enemies still in sight.
        Uses spatial index for efficient radius query (FOV_RADIUS).
        """
        from catley.game.actors.ai import DispositionBasedAI
        from catley.game.enums import Disposition

        player = self.gw.player
        # Query actors within the configured FOV radius
        nearby_actors = self.gw.actor_spatial_index.get_in_radius(
            player.x, player.y, radius=config.FOV_RADIUS
        )

        for actor in nearby_actors:
            # Skip non-characters and the player
            if not isinstance(actor, Character) or actor == player:
                continue
            # Skip dead actors
            if not actor.health.is_alive():
                continue
            # Skip actors not visible to the player
            if not self.gw.game_map.visible[actor.x, actor.y]:
                continue
            # Check if hostile
            if (
                isinstance(actor.ai, DispositionBasedAI)
                and actor.ai.disposition == Disposition.HOSTILE
            ):
                return True
        return False

    def select_target(self, target: Actor | None) -> None:
        """Set the selected target for the ActionPanel.

        Selected targets are sticky - they persist until explicitly deselected
        (by clicking empty ground or calling deselect_target). This enables
        RTS-style click-to-select where the panel locks to a target.

        Args:
            target: The actor to select, or None to deselect.
        """
        # Play selection sound when selecting a new target
        if target is not None and target is not self.selected_target:
            # Use player position so UI sound plays at full volume
            publish_event(
                SoundEvent(sound_id="ui_select", x=self.gw.player.x, y=self.gw.player.y)
            )
        self.selected_target = target

    def deselect_target(self) -> None:
        """Clear the selected target.

        Call this when the player clicks on empty ground or otherwise wants
        to clear the selection.
        """
        self.selected_target = None

    def start_consumable_targeting(self, item: Item) -> None:
        """Start consumable targeting via PickerMode.

        Clicking on self uses the item on the player. Clicking on another
        character uses it on them (pathfinding if needed).

        Args:
            item: The consumable Item to use.
        """
        from catley import colors
        from catley.events import MessageEvent, publish_event
        from catley.game import ranges
        from catley.game.actions.recovery import (
            UseConsumableIntent,
            UseConsumableOnTargetIntent,
        )
        from catley.game.actors import Character
        from catley.modes.picker import PickerResult

        player = self.gw.player

        def valid_filter(x: int, y: int) -> bool:
            """Check if a tile is a valid consumable target."""
            # Player's own tile is valid (self-target)
            if x == player.x and y == player.y:
                return True
            # Check for a living character at the tile
            actor = self.gw.get_actor_at_location(x, y)
            if actor is None:
                return False
            return (
                isinstance(actor, Character)
                and actor.health.is_alive()
                and self.gw.game_map.visible[x, y]
            )

        def on_select(result: PickerResult) -> None:
            """Handle target selection."""
            # Self-targeting: use on self
            if result.tile[0] == player.x and result.tile[1] == player.y:
                intent = UseConsumableIntent(self, player, item)
                self.queue_action(intent)
                return

            # Target is another character
            target = result.actor
            if target is not None and isinstance(target, Character):
                distance = ranges.calculate_distance(
                    player.x, player.y, target.x, target.y
                )
                if distance == 1:
                    # Adjacent target: use immediately
                    intent = UseConsumableOnTargetIntent(self, player, item, target)
                    self.queue_action(intent)
                else:
                    # Distant target: use ActionPlan to pathfind then use
                    from catley.game.actions.recovery import UseConsumableOnTargetPlan

                    self.start_plan(
                        player,
                        UseConsumableOnTargetPlan,
                        target_actor=target,
                        target_position=(target.x, target.y),
                        item=item,
                    )

        # Show message about targeting
        publish_event(MessageEvent(f"Select target for {item.name}", colors.CYAN))

        self.picker_mode.start(
            on_select=on_select,
            on_cancel=lambda: None,
            valid_filter=valid_filter,
        )

    def create_resolver(self, **kwargs: object) -> ResolutionSystem:
        """Factory method for resolution systems.

        Currently returns a :class:`D20System` but allows future
        customization without changing action code.
        """
        from catley.game.resolution.d20_system import D20System

        return D20System(**kwargs)  # type: ignore[call-arg]

    def start_plan(
        self,
        actor: Character,
        plan: ActionPlan,
        target_actor: Character | Actor | None = None,
        target_position: WorldTilePos | None = None,
        weapon: Item | None = None,
        item: Item | None = None,
    ) -> bool:
        """Start an action plan for an actor.

        This is the unified method for starting any ActionPlan. It creates
        the PlanContext and ActivePlan, then assigns it to the actor.

        Args:
            actor: The character executing the plan.
            plan: The ActionPlan to execute.
            target_actor: Optional target actor for targeted actions.
            target_position: Optional target position. For actor targets,
                this is usually derived from (target_actor.x, target_actor.y).
            weapon: Optional weapon for combat actions.
            item: Optional item for consumable actions.

        Returns:
            True (always succeeds in creating the plan).
        """
        from catley.game.action_plan import ActivePlan, PlanContext

        context = PlanContext(
            actor=actor,
            controller=self,
            target_actor=target_actor,
            target_position=target_position,
            weapon=weapon,
            item=item,
        )
        actor.active_plan = ActivePlan(plan=plan, context=context)
        return True

    def stop_plan(self, actor: Character) -> None:
        """Cancel an actor's active plan."""
        actor.active_plan = None

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
                # Continue without audio; avoid user-visible warnings.
                logger.debug("Failed to initialize audio backend: %s", e)

    def cleanup(self) -> None:
        """Clean up resources when the controller is being shut down."""
        # Shutdown audio backend if it exists
        if self.sound_system.audio_backend:
            self.sound_system.audio_backend.shutdown()

"""
Manages the game's action scheduling using the Reactive Actor Framework (RAF).

The RAF TurnManager implements speed-based energy accumulation in a purely
turn-based context. When the player acts, the world reacts immediately:

1. All actors get energy proportional to their individual speeds
2. Faster actors accumulate more energy per player action
3. NPCs who can afford actions get those actions queued
4. The Controller smoothly executes these queued actions one per frame

This creates responsive player input (zero delay) while maintaining fair
energy economy (faster actors naturally get more actions over time) and
smooth world presentation (simultaneous NPC reactions).

Key Principles:
- Energy accumulates ONLY when the player acts (turn-based, not real-time)
- Faster actors get proportionally more energy per player action
- No time passes when player is idle - game state is stable
- Action execution is distributed smoothly over frames for visual appeal

Presentation Timing:
- Actions return `presentation_ms` in their GameActionResult
- The TurnManager delays the next action until presentation completes
- This creates readable sequencing where players can follow cause and effect
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

from catley import config
from catley.game.action_router import ActionRouter
from catley.game.actions.base import GameActionResult, GameIntent
from catley.util.coordinates import WorldTilePos

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.action_plan import ActivePlan, ApproachStep
    from catley.game.actors import Actor
    from catley.game.actors.core import Character


class TurnManager:
    """
    Autonomous action scheduler that manages energy and NPC reactions.

    The TurnManager operates on these core principles:

    1. **Event-Driven Scheduling**: When the player acts, all actors immediately
       get their energy updated and any NPCs who can now afford actions get
       those actions queued.

    2. **Independent Energy Economy**: Each actor accumulates energy based on
       their speed, ensuring faster actors get proportionally more actions
       over time without requiring synchronized rounds.

    3. **Smooth Action Distribution**: NPC actions are queued and executed
       one per frame, creating smooth visual flow without blocking the player.

    4. **Autonomous Operation**: The Controller simply calls `on_player_action()`
       when the player acts and `get_next_npc_action()` each frame. All
       scheduling logic is encapsulated here.
    """

    def __init__(self, controller: Controller) -> None:
        """Initialize the action scheduler with performance optimizations.

        Args:
            controller: The main game controller for accessing actors and world state
        """
        self.controller = controller
        self.player = self.controller.gw.player

        # Player action queue (preserved for compatibility)
        self._player_action_queue: deque[GameIntent] = deque()

        # Action execution router (unchanged)
        self.action_router = ActionRouter(self.controller)

        # Performance optimization: cache actors with energy components
        self._energy_actors_cache: list[Actor] = []
        self._cache_dirty: bool = True

        # Presentation timing state: tracks when actions complete and how long
        # to wait before processing the next action. This creates readable
        # sequencing where players can follow cause and effect.
        self._last_action_completed_time: float = 0.0
        self._pending_presentation_ms: int = 0

    def _update_energy_actors_cache(self) -> None:
        """Update cached list of actors with energy components for performance.

        Only actors with a non-None energy component are included. Static objects
        like containers don't have energy and are excluded from the action economy.
        """
        if self._cache_dirty:
            self._energy_actors_cache = [
                actor for actor in self.controller.gw.actors if actor.energy is not None
            ]
            self._cache_dirty = False

    def invalidate_cache(self) -> None:
        """Mark actor cache as dirty (call when actors are added/removed)."""
        self._cache_dirty = True

    def is_presentation_complete(self) -> bool:
        """Check if the current action's presentation time has elapsed.

        Returns True if either:
        - No presentation is pending (presentation_ms was 0)
        - Enough time has passed since the last action completed

        This method is non-blocking and should be called before processing
        the next action to create readable action sequencing.
        """
        if self._pending_presentation_ms == 0:
            return True

        elapsed_ms = (time.perf_counter() - self._last_action_completed_time) * 1000
        return elapsed_ms >= self._pending_presentation_ms

    def _record_action_timing(self, result: GameActionResult) -> None:
        """Record timing after an action completes for presentation delay.

        Args:
            result: The GameActionResult containing presentation_ms.
        """
        self._last_action_completed_time = time.perf_counter()
        self._pending_presentation_ms = result.presentation_ms

    def clear_presentation_timing(self) -> None:
        """Clear any pending presentation timing.

        Call this when manually interrupting action flow (e.g., player
        cancels a plan or starts a new action while presenting).
        """
        self._pending_presentation_ms = 0

    def on_player_action(self) -> None:
        """Handle per-turn updates when player acts.

        This method is called once per player action and handles:
        1. Energy accumulation for all actors
        2. Terrain hazard damage for all NPCs

        Action processing is handled separately by process_all_npc_reactions()
        which runs every tick to check if NPCs can afford actions.
        """
        # Update cache if needed for performance
        self._update_energy_actors_cache()

        # Update energy for all actors with energy components.
        # The cache only contains actors with non-None energy.
        for actor in self._energy_actors_cache:
            assert actor.energy is not None  # Guaranteed by cache filter
            actor.energy.accumulate_energy(actor.energy.get_speed_based_energy_amount())
            # Cap NPC energy to prevent double-actions from accumulation.
            # Without this, NPCs could store 200 energy (2 actions) while
            # the player walks toward them, causing double-moves on contact.
            if actor is not self.player:
                actor.energy.accumulated_energy = min(
                    actor.energy.accumulated_energy,
                    config.ACTION_COST,
                )

        # Apply terrain hazard damage to all NPCs once per player action.
        # This ensures NPCs on hot coals take damage each turn, not each tick.
        for actor in self._energy_actors_cache:
            if actor is self.player:
                continue
            self._apply_terrain_hazard(actor)

    def process_all_npc_reactions(self) -> None:
        """Process ONE NPC who can currently afford an action.

        Called every game tick. Processes only one NPC per call so that
        presentation timing gates each NPC action individually, creating
        sequential cause-and-effect between NPCs.

        Note: Hazard damage is NOT applied here - it's applied once per player
        action in on_player_action() to avoid damage being applied every tick.
        """
        from catley.game.action_plan import ApproachStep
        from catley.game.actors.core import Character

        for actor in self._energy_actors_cache:
            if actor is self.player:
                continue

            # Cache only contains actors with non-None energy
            assert actor.energy is not None

            if actor.energy.can_afford(config.ACTION_COST):
                action = actor.get_next_action(self.controller)

                # If AI returned None but NPC has an active plan, get intent
                # from plan (mirrors player autopilot in process_player_input).
                if (
                    action is None
                    and isinstance(actor, Character)
                    and actor.active_plan is not None
                ):
                    action = self._get_intent_from_plan(actor)

                if action is not None:
                    # Check if actor is prevented from acting BEFORE update_turn
                    is_prevented = actor.status_effects.is_action_prevented()

                    if is_prevented:
                        # Actor is prevented - this counts as their turn.
                        # Spend ALL energy so they can't attempt again this cycle.
                        # Call update_turn to decrement effect duration.
                        actor.update_turn(self.controller)
                        actor.energy.accumulated_energy = 0
                        continue

                    # Actor can act - update their turn effects first
                    actor.update_turn(self.controller)

                    # Execute the action
                    result = self.execute_intent(action)
                    actor.energy.spend(config.ACTION_COST)

                    # Handle plan advancement for NPCs with active plans
                    if isinstance(actor, Character) and actor.active_plan is not None:
                        plan = actor.active_plan
                        step = plan.get_current_step()
                        if isinstance(step, ApproachStep):
                            self._on_approach_result(actor, result)

                    if hasattr(self.controller, "invalidate_combat_tooltip"):
                        self.controller.invalidate_combat_tooltip()

                    # Process only ONE NPC per call - presentation timing
                    # will gate the next call, sequencing NPC actions
                    return

    def _apply_terrain_hazard(self, actor: Actor) -> None:
        """Check if actor is on hazardous terrain and apply damage if so.

        Called after an actor completes their turn. If they're standing on
        hazardous terrain (acid pools, hot coals, etc.), they take damage.

        Args:
            actor: The actor to check for terrain hazard damage.
        """
        from catley.environment.tile_types import (
            get_tile_hazard_info,
            get_tile_type_name_by_id,
        )
        from catley.game.actions.environmental import EnvironmentalDamageIntent
        from catley.util.dice import Dice

        # Skip actors without a game world reference
        if not hasattr(actor, "gw") or actor.gw is None:
            return

        # Get the tile the actor is standing on
        tile_id = int(actor.gw.game_map.tiles[actor.x, actor.y])
        damage_dice, damage_type = get_tile_hazard_info(tile_id)

        # Apply damage if this tile is hazardous (non-empty dice string)
        if damage_dice:
            damage = Dice(damage_dice).roll()
            tile_name = get_tile_type_name_by_id(tile_id)
            intent = EnvironmentalDamageIntent(
                controller=self.controller,
                source_actor=None,  # Terrain hazard has no actor source
                damage_amount=damage,
                damage_type=damage_type,
                affected_coords=[(actor.x, actor.y)],
                source_description=tile_name,
            )
            self.execute_intent(intent)

    def execute_intent(self, intent: GameIntent) -> GameActionResult:
        """Execute a single GameIntent by routing it to the ActionRouter.

        Returns the GameActionResult so callers can inspect whether the action
        succeeded and decide how to proceed (e.g., for plan advancement).

        Also records the action's presentation timing to delay subsequent actions.

        Args:
            intent: The GameIntent to execute

        Returns:
            GameActionResult from the executor, or a failed result if no executor.
        """
        result = self.action_router.execute_intent(intent)
        # Record timing for presentation delay before next action
        self._record_action_timing(result)
        return result

    # === ActionPlan System Methods ===
    # These methods handle the ActionPlan system for multi-step player actions.

    def get_next_player_intent(self) -> GameIntent | None:
        """Get the next player intent, checking active_plan first.

        This is the primary entry point for getting player actions when using
        the ActionPlan system. It checks in order:
        1. Player's active_plan (if any)
        2. Queued manual actions

        Returns:
            The next GameIntent to execute, or None if no action available.
        """
        # Check active_plan first
        if self.player.active_plan is not None:
            intent = self._get_intent_from_plan(self.player)
            if intent is not None:
                return intent
            # Plan completed or couldn't generate intent - fall through

        # Fall back to queued actions
        return self.dequeue_player_action()

    def execute_player_intent(self, intent: GameIntent) -> GameActionResult:
        """Execute a player intent and handle plan advancement if applicable.

        This method wraps execute_intent() and handles the plan advancement
        logic internally. The Controller doesn't need to know about plan
        internals - it just calls this method.

        Args:
            intent: The GameIntent to execute.

        Returns:
            The GameActionResult from execution.
        """
        result = self.execute_intent(intent)

        # Handle plan advancement for ApproachStep moves
        if self.player.active_plan is not None:
            self._on_approach_result(self.player, result)

        return result

    def _get_intent_from_plan(self, actor: Character) -> GameIntent | None:
        """Generate the next intent from an actor's active plan.

        Handles step skipping, IntentStep execution, and ApproachStep
        pathfinding. Returns None if the plan is complete or can't proceed.

        Args:
            actor: The character with an active plan.

        Returns:
            A GameIntent to execute, or None if plan is done/blocked.
        """
        from catley.game.action_plan import ApproachStep, IntentStep

        plan = actor.active_plan
        if plan is None:
            return None

        step = plan.get_current_step()

        # Plan complete?
        if step is None:
            actor.active_plan = None
            return None

        # Skip steps whose conditions are met
        while (
            step is not None and step.skip_if is not None and step.skip_if(plan.context)
        ):
            plan.advance()
            step = plan.get_current_step()

        if step is None:
            actor.active_plan = None
            return None

        # Generate intent based on step type
        if isinstance(step, IntentStep):
            params = step.params(plan.context)
            intent = step.intent_class(plan.context.controller, **params)
            plan.advance()
            return intent

        if isinstance(step, ApproachStep):
            return self._handle_approach_step(actor, plan, step)

        # Unknown step type
        return None

    def _handle_approach_step(
        self,
        actor: Character,
        plan: ActivePlan,
        step: ApproachStep,
    ) -> GameIntent | None:
        """Handle an ApproachStep by generating the next MoveIntent.

        This method:
        1. Checks if we've arrived (based on stop_distance)
        2. Calculates path if needed
        3. PEEKs at the next position (does NOT pop)
        4. Returns a MoveIntent

        The path is only popped after the move succeeds, in _on_approach_result().

        Args:
            actor: The character executing the plan.
            plan: The active plan containing the step.
            step: The ApproachStep to handle.

        Returns:
            A MoveIntent for the next step, or None if arrived/blocked.
        """
        from catley.game.actions.movement import MoveIntent
        from catley.util.pathfinding import find_local_path

        # Determine target position
        if plan.context.target_actor is not None:
            target_pos = (plan.context.target_actor.x, plan.context.target_actor.y)
        elif plan.context.target_position is not None:
            target_pos = plan.context.target_position
        else:
            # No target - can't approach
            actor.active_plan = None
            return None

        # Check if we've arrived (based on stop_distance)
        dx_to_target = target_pos[0] - actor.x
        dy_to_target = target_pos[1] - actor.y
        current_distance = max(abs(dx_to_target), abs(dy_to_target))  # Chebyshev

        if current_distance <= step.stop_distance:
            # Arrived - advance to next step
            plan.advance()
            # Recurse to get intent from next step (if any)
            return self._get_intent_from_plan(actor)

        # Generate path if needed
        if plan.cached_path is None or not plan.cached_path:
            gm = self.controller.gw.game_map
            asi = self.controller.gw.actor_spatial_index
            start_pos: tuple[int, int] = (actor.x, actor.y)

            # First try direct path to target
            plan.cached_path = find_local_path(gm, asi, actor, start_pos, target_pos)

            # If direct path fails, try hierarchical (cross-region) pathfinding
            if not plan.cached_path:
                hierarchical = self._try_hierarchical_path(actor, start_pos, target_pos)
                if hierarchical:
                    plan.cached_path, plan.cached_hierarchical_path = hierarchical

            # If still no path and we have a stop_distance, try adjacent tiles.
            # The target tile is likely occupied (e.g., by an enemy we're approaching).
            if not plan.cached_path and step.stop_distance > 0:
                best_path: list[tuple[int, int]] | None = None
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
                        blocker = self.controller.gw.get_actor_at_location(tx, ty)
                        if blocker and blocker.blocks_movement and blocker is not actor:
                            continue
                        candidate = find_local_path(gm, asi, actor, start_pos, (tx, ty))
                        if candidate and (
                            best_path is None or len(candidate) < len(best_path)
                        ):
                            best_path = candidate
                if best_path is not None:
                    plan.cached_path = best_path

        if not plan.cached_path:
            # Can't reach target - cancel plan
            actor.active_plan = None
            return None

        # PEEK at next position - don't pop yet!
        # We only pop after confirming the move succeeded.
        next_pos = plan.cached_path[0]
        dx = next_pos[0] - actor.x
        dy = next_pos[1] - actor.y

        return MoveIntent(plan.context.controller, actor, dx, dy)

    def _on_approach_result(self, actor: Character, result: GameActionResult) -> None:
        """Handle the result of an approach move.

        Called after executing a MoveIntent from an ApproachStep. Updates the
        plan state based on whether the move succeeded:
        - Success: Pop the path entry we just moved to
        - Failure: Invalidate path for recalculation on next turn

        Args:
            actor: The character who attempted the move.
            result: The result from executing the MoveIntent.
        """
        from catley.game.action_plan import ApproachStep

        plan = actor.active_plan
        if plan is None:
            return

        step = plan.get_current_step()
        if not isinstance(step, ApproachStep):
            return

        if result.succeeded:
            # Move succeeded - now safe to pop the path
            if plan.cached_path:
                plan.cached_path.pop(0)

            # After popping, check if we need to compute the next hierarchical segment
            if not plan.cached_path and plan.cached_hierarchical_path:
                self._advance_hierarchical_path(actor, plan)

            # Check if we've now arrived at destination
            if plan.context.target_actor is not None:
                target_pos = (plan.context.target_actor.x, plan.context.target_actor.y)
            elif plan.context.target_position is not None:
                target_pos = plan.context.target_position
            else:
                return

            dx = target_pos[0] - actor.x
            dy = target_pos[1] - actor.y
            distance = max(abs(dx), abs(dy))

            if distance <= step.stop_distance:
                # Arrived - advance to next step (or complete plan)
                plan.advance()
                if plan.is_complete():
                    actor.active_plan = None
        else:
            # Move failed (blocked) - invalidate path for recalculation
            plan.cached_path = None
            plan.cached_hierarchical_path = None

    def _try_hierarchical_path(
        self,
        actor: Character,
        start_pos: WorldTilePos,
        target_pos: WorldTilePos,
    ) -> tuple[list[WorldTilePos], list[int] | None] | None:
        """Attempt hierarchical pathfinding across regions.

        Returns (local_path, high_level_path) if cross-region path is found,
        or None if regions aren't set up or positions are in the same region.

        Args:
            actor: The character pathfinding.
            start_pos: Starting position.
            target_pos: Target position.

        Returns:
            Tuple of (local_path, high_level_path) or None.
        """
        from catley.util.pathfinding import find_local_path, find_region_path

        gm = self.controller.gw.game_map
        asi = self.controller.gw.actor_spatial_index

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

        return (local_path, high_level_path)

    def _advance_hierarchical_path(self, actor: Character, plan: ActivePlan) -> None:
        """Compute next path segment when current local path is exhausted.

        For hierarchical (cross-region) paths, when cached_path becomes empty
        but cached_hierarchical_path still has regions to traverse, this computes
        the local path to the next connection point (or to target_pos if we're
        in the final region).

        Args:
            actor: The character executing the plan.
            plan: The active plan with hierarchical path data.
        """
        from catley.util.pathfinding import find_local_path

        if plan.cached_hierarchical_path is None:
            return

        # Only act if local path is exhausted and we have more regions to go
        if plan.cached_path:
            return

        gm = self.controller.gw.game_map
        asi = self.controller.gw.actor_spatial_index

        # Pop the region we just entered
        current_region_id = plan.cached_hierarchical_path.pop(0)

        # Determine target position
        if plan.context.target_actor is not None:
            target_pos: WorldTilePos = (
                plan.context.target_actor.x,
                plan.context.target_actor.y,
            )
        elif plan.context.target_position is not None:
            target_pos = plan.context.target_position
        else:
            actor.active_plan = None
            return

        if plan.cached_hierarchical_path:
            # More regions to traverse - path to next connection point
            next_region_id = plan.cached_hierarchical_path[0]
            current_region = gm.regions.get(current_region_id)
            if current_region is None:
                actor.active_plan = None
                return

            connection_point = current_region.connections.get(next_region_id)
            if connection_point is None:
                actor.active_plan = None
                return

            # Compute local path to connection point
            new_path = find_local_path(
                gm, asi, actor, (actor.x, actor.y), connection_point
            )
            if new_path:
                plan.cached_path = new_path
            else:
                # Can't reach connection - clear plan
                actor.active_plan = None
        else:
            # We're in the final region - path to target
            new_path = find_local_path(gm, asi, actor, (actor.x, actor.y), target_pos)
            if new_path:
                plan.cached_path = new_path
            # If no path, leave cached_path empty - plan will handle it next turn

    # === Preserved Methods for Compatibility ===
    # These methods are kept to maintain compatibility with existing code
    # that might still reference them during the transition.

    def queue_action(self, action: GameIntent) -> None:
        """Queue a player action (preserved for compatibility)."""
        self._player_action_queue.append(action)

    def has_pending_actions(self) -> bool:
        """Check if there are pending player actions (preserved for compatibility)."""
        return len(self._player_action_queue) > 0

    def dequeue_player_action(self) -> GameIntent | None:
        """Dequeue and return a pending player action (preserved for compatibility)."""
        return (
            self._player_action_queue.popleft() if self._player_action_queue else None
        )

    def is_player_turn_available(self) -> bool:
        """Return True if player has pending actions or active plans."""
        has_manual_action = self.has_pending_actions()
        has_active_plan = getattr(self.player, "active_plan", None) is not None
        return has_manual_action or has_active_plan

    # Backwards compatibility for old name
    def is_turn_available(self) -> bool:  # pragma: no cover - legacy
        return self.is_player_turn_available()

    # === Debug and Diagnostic Methods ===

    def get_npc_queue_length(self) -> int:
        """Return the current number of queued NPC actions (for debugging).

        Note: In RAF V2, NPCs act immediately so this always returns 0.
        Kept for compatibility with debugging code.
        """
        return 0

    def clear_npc_queue(self) -> None:
        """Clear all pending NPC actions (for testing/debugging).

        Note: In RAF V2, NPCs act immediately so there's no queue to clear.
        Kept for compatibility with testing code.
        """
        pass

    def debug_energy_state(self) -> None:
        """Print current energy state for all actors (debugging only)."""
        print("=== RAF Energy State Debug ===")
        for actor in self.controller.gw.actors:
            if actor.energy is not None:
                energy_per_action = actor.energy.get_speed_based_energy_amount()
                energy_info = f"{actor.energy.energy:.1f}/{actor.energy.max_energy}"
                speed_info = f"speed: {actor.energy.speed}, +{energy_per_action:.1f}"
                print(f"{actor.name}: {energy_info} ({speed_info} per player action)")
        print("NPC Queue Length: 0 (RAF V2 - immediate processing)")
        print("==============================")

    def get_speed_ratios(self) -> dict[str, float]:
        """Get speed ratios for balancing analysis.

        Returns:
            Dictionary mapping actor names to their relative speed ratios
        """
        if not self.controller.gw.actors:
            return {}

        # Find base speed (usually player speed)
        base_speed = self.player.energy.speed if self.player.energy is not None else 100

        ratios = {}
        for actor in self.controller.gw.actors:
            if actor.energy is not None:
                ratio = actor.energy.speed / base_speed
                ratios[actor.name] = ratio

        return ratios

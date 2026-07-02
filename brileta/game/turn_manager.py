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
- Combat mode: energy accumulates when the player acts (turn-based, reactive)
- Explore mode: NPC energy accrues over real time so the world keeps moving
  while the player is idle; the player still only gains energy by acting
- Faster actors get proportionally more energy over time
- Action execution is distributed smoothly over frames for visual appeal

Presentation Timing:
- Actions return `duration_ms` in their GameActionResult
- The TurnManager delays the next action until presentation completes
- This creates readable sequencing where players can follow cause and effect
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from typing import TYPE_CHECKING

from brileta import config
from brileta.constants.movement import MovementConstants as Movement
from brileta.game.action_router import ActionRouter
from brileta.game.actions.base import GameActionResult, GameIntent
from brileta.game.enums import ActionBlockReason
from brileta.game.ranges import calculate_distance
from brileta.types import DIRECTIONS, ActorId, FixedTimestep
from brileta.util import rng
from brileta.util.coordinates import WorldTilePos

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.action_plan import ActivePlan, ApproachStep
    from brileta.game.actors import PC, Actor
    from brileta.game.actors.core import Character

# The ephemeral identity of one live ActivePlan instance.
# AdjacencyRetryKey is the key for adjacency-rewind retry counts.
type PlanInstanceId = int
type AdjacencyRetryKey = tuple[ActorId, PlanInstanceId]

# Deterministic RNG stream for each NPC's one-time ambient-energy phase offset,
# so same-speed NPCs don't cross their action threshold on the same tick.
_ambient_phase_rng = rng.get("npc.energy_phase")


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

    _MAX_ADJACENCY_REWINDS = 3
    """Maximum chase rewinds for adjacency-required plans."""

    def __init__(self, controller: Controller) -> None:
        """Initialize the action scheduler with performance optimizations.

        Args:
            controller: The main game controller for accessing actors and world state
        """
        self.controller = controller

        # Queue for non-plan player actions (e.g. manual key-presses)
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
        self._pending_duration_ms: int = 0

        # Per-(actor, plan) retry counters for adjacency-chase rewinds.
        # This keeps retry policy in TurnManager instead of ActivePlan.
        self._adjacency_rewind_counts: dict[AdjacencyRetryKey, int] = {}

    @property
    def player(self) -> PC:
        """The player character, accessed via controller to avoid stale references."""
        return self.controller.gw.player

    def reset(self) -> None:
        """Reset the turn manager for a new world.

        Clears all queues and caches. Called when the game world is regenerated.
        """
        self._player_action_queue.clear()
        self._energy_actors_cache.clear()
        self._cache_dirty = True
        self._last_action_completed_time = 0.0
        self._pending_duration_ms = 0
        self._adjacency_rewind_counts.clear()

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

    def on_actor_removed(self, actor_id: ActorId) -> None:
        """Purge retry state for a removed actor to prevent stale entries.

        Without this, killed NPCs with adjacency-retried plans would leave
        dangling entries in _adjacency_rewind_counts whose id-based keys
        could collide with future objects.
        """
        for key in [k for k in self._adjacency_rewind_counts if k[0] == actor_id]:
            del self._adjacency_rewind_counts[key]

    def is_presentation_complete(self) -> bool:
        """Check if the current action's presentation time has elapsed.

        Returns True if either:
        - No presentation is pending (duration_ms was 0)
        - Enough time has passed since the last action completed

        This method is non-blocking and should be called before processing
        the next action to create readable action sequencing.
        """
        if self._pending_duration_ms == 0:
            return True

        elapsed_ms = (time.perf_counter() - self._last_action_completed_time) * 1000
        return elapsed_ms >= self._pending_duration_ms

    def _record_action_timing(self, result: GameActionResult) -> None:
        """Record timing after an action completes for presentation delay.

        Args:
            result: The GameActionResult containing duration_ms.
        """
        self._last_action_completed_time = time.perf_counter()
        self._pending_duration_ms = result.duration_ms

    def clear_presentation_timing(self) -> None:
        """Clear any pending presentation timing.

        Call this when manually interrupting action flow (e.g., player
        cancels a plan or starts a new action while presenting).
        """
        self._pending_duration_ms = 0

    def _npc_energy_actors(self) -> Iterator[Actor]:
        """Yield each non-player actor with an energy component.

        Refreshes the energy-actor cache first so callers get a current view,
        and centralizes the shared "skip the player" rule of the energy passes.
        """
        self._update_energy_actors_cache()
        for actor in self._energy_actors_cache:
            assert actor.energy is not None  # Guaranteed by cache filter
            if actor is not self.player:
                yield actor

    @staticmethod
    def _cap_npc_energy(actor: Actor) -> None:
        """Cap an NPC's energy at one action's worth so it can't bank double-actions.

        Without this, an NPC could store 200 energy (two actions) while the
        player walks toward them, causing double-moves on contact.
        """
        assert actor.energy is not None
        actor.energy.accumulated_energy = min(
            actor.energy.accumulated_energy,
            config.ACTION_COST,
        )

    def on_player_action(self) -> None:
        """Handle per-turn updates when the player acts.

        The player always gains their own speed-based energy here (autopilot
        checks can_afford, so the player needs an income). NPC handling depends
        on the mode:

        - Combat mode: NPCs also accumulate speed-based energy and take terrain
          hazard damage once per player action - the reactive, turn-based RAF.
        - Explore mode: NPCs are skipped entirely. Their energy is time-driven
          via accumulate_ambient_energy() and their hazard damage is applied at
          their own action moment, so the two energy sources never stack.

        Action processing is handled separately by process_all_npc_reactions()
        (combat) or process_all_ready_npcs_immediately() (explore).
        """
        # The player always earns their own energy from acting.
        if self.player.energy is not None:
            self.player.energy.accumulate_energy(
                self.player.energy.get_speed_based_energy_amount()
            )

        # Explore mode: NPC energy is purely time-driven, so player actions grant
        # NPCs nothing - keeping the two sources from stacking. Combat keeps the
        # reactive per-player-action RAF grant.
        if not self.controller.is_combat_mode():
            return

        for actor in self._npc_energy_actors():
            assert actor.energy is not None  # Narrow for the type checker
            actor.energy.accumulate_energy(actor.energy.get_speed_based_energy_amount())
            self._cap_npc_energy(actor)
            # Apply terrain hazard damage (e.g. hot coals) once per player action
            # so NPCs don't take damage every tick.
            self._apply_terrain_hazard(actor)

    def accumulate_ambient_energy(self, timestep: FixedTimestep) -> None:
        """Accrue time-driven energy for NPCs each fixed logic step (explore mode).

        Each NPC gains a slice of energy scaled by its own speed, sized so a
        standard-speed NPC affords one action per
        config.AMBIENT_ACTION_INTERVAL_SECONDS of real time. Faster NPCs cross
        the action threshold sooner and slower ones later, so each acts on its
        own cadence instead of a synchronized global pulse.

        The player is skipped - the player only gains energy from their own
        actions. NPC energy is capped at one action's worth so NPCs can't bank
        double-actions. Terrain hazards are applied at the NPC's own action
        moment (in _try_process_npc), not here.

        Args:
            timestep: The fixed logic step duration in seconds.
        """
        interval = config.AMBIENT_ACTION_INTERVAL_SECONDS
        for actor in self._npc_energy_actors():
            assert actor.energy is not None  # Narrow for the type checker

            # On an NPC's first ambient accrual, give it a one-time random phase
            # offset (so same-speed NPCs don't march in lockstep) and a small
            # speed multiplier (so they stroll at slightly different paces). The
            # wander AI's own random pauses spread them further apart.
            if not actor.energy.ambient_phased:
                actor.energy.ambient_phased = True
                actor.energy.ambient_speed_multiplier = _ambient_phase_rng.uniform(
                    0.8, 1.2
                )
                actor.energy.accumulate_energy(
                    _ambient_phase_rng.uniform(0.0, config.ACTION_COST)
                )

            # get_speed_based_energy_amount() is ACTION_COST worth for a
            # standard NPC, so spreading it over `interval` seconds of steps
            # yields one action per interval at standard speed. The per-NPC
            # multiplier varies that pace slightly.
            per_step = (
                actor.energy.get_speed_based_energy_amount()
                * actor.energy.ambient_speed_multiplier
                * timestep
                / interval
            )
            actor.energy.accumulate_energy(per_step)
            self._cap_npc_energy(actor)

    def clamp_npc_energy_for_combat(self) -> None:
        """Zero NPC accumulated energy on combat entry.

        In explore mode NPCs accrue energy over real time and may wander up with
        a full bar. Combat is player-action-driven, so this resets NPC energy on
        entry to prevent an NPC getting a free instant attack before the player
        has acted. The player's energy is left untouched.
        """
        for actor in self._npc_energy_actors():
            assert actor.energy is not None  # Narrow for the type checker
            actor.energy.accumulated_energy = 0.0

    def process_all_npc_reactions(self) -> None:
        """Process ONE NPC who can currently afford an action.

        Called every game tick. Processes only one NPC per call so that
        presentation timing gates each NPC action individually, creating
        sequential cause-and-effect between NPCs.

        Note: Hazard damage is NOT applied here - it's applied once per player
        action in on_player_action() to avoid damage being applied every tick.
        """
        for actor in self._energy_actors_cache:
            if actor is self.player:
                continue
            if self._try_process_npc(actor, record_timing=True):
                self.controller.invalidate_combat_tooltip()
                # Process only ONE NPC per call - presentation timing
                # will gate the next call, sequencing NPC actions.
                return

    def process_all_ready_npcs_immediately(self) -> None:
        """Process all NPCs who can afford an action in one call.

        Unlike process_all_npc_reactions() which processes one NPC per call
        for visual pacing, this method processes ALL ready NPCs immediately.
        Used for held-key movement where the player is moving fast and
        presentation timing would starve NPCs of their turns.
        """
        for actor in self._energy_actors_cache:
            if actor is self.player:
                continue
            # Skip presentation timing so we don't overwrite the player's
            # timing state during held-key movement.
            self._try_process_npc(actor, record_timing=False)

    def _try_process_npc(self, actor: Actor, *, record_timing: bool) -> bool:
        """Attempt to process one NPC action if the actor can afford it.

        Handles energy checks, status effect prevention, action execution,
        and plan advancement. Returns True if the actor acted (or was
        prevented from acting), False if the actor couldn't afford an action.

        Args:
            actor: The NPC to process.
            record_timing: If True, record presentation timing via
                execute_intent(). If False, use action_router directly.
        """
        from brileta.game.action_plan import ApproachStep
        from brileta.game.actors.core import Character

        assert actor.energy is not None  # Guaranteed by _energy_actors_cache

        if not actor.energy.can_afford(config.ACTION_COST):
            return False

        action = actor.get_next_action(self.controller)

        # If AI returned None but NPC has an active plan, get intent from plan
        # (mirrors player autopilot in process_player_input).
        if (
            action is None
            and isinstance(actor, Character)
            and actor.active_plan is not None
        ):
            action = self._get_intent_from_plan(actor)

        if action is None:
            # Explore mode: an affordable NPC that chooses not to act (e.g. a
            # wander pause) consumes its turn - tick its effects and spend a
            # turn's energy - so it actually loiters for an action interval
            # instead of being re-polled every logic step (60Hz), which would
            # burn the wander AI's linger counters in milliseconds. Combat keeps
            # its reactive per-tick polling unchanged.
            if not self.controller.is_combat_mode():
                actor.update_turn(self.controller)
                actor.energy.spend(config.ACTION_COST)
                self._apply_explore_turn_hazard(actor)
            return False

        # Check if actor is prevented from acting BEFORE update_turn
        if actor.status_effects.is_action_prevented():
            # Prevented - this counts as their turn. Spend ALL energy so they
            # can't attempt again this cycle. update_turn decrements duration.
            actor.update_turn(self.controller)
            actor.energy.accumulated_energy = 0
            self._apply_explore_turn_hazard(actor)
            return True

        # Actor can act - update their turn effects first
        actor.update_turn(self.controller)

        # Execute the action (with or without presentation timing)
        if record_timing:
            result = self.execute_intent(action)
        else:
            result = self.action_router.execute_intent(action)
        actor.energy.spend(config.ACTION_COST)

        self._apply_explore_turn_hazard(actor)

        # Handle plan advancement for NPCs with active plans
        if isinstance(actor, Character) and actor.active_plan is not None:
            step = actor.active_plan.get_current_step()
            if isinstance(step, ApproachStep):
                self._on_approach_result(actor, result)
            if (
                not result.succeeded
                and result.block_reason == ActionBlockReason.NOT_ADJACENT
            ):
                self._handle_intent_step_result(actor, result)

        return True

    def _apply_explore_turn_hazard(self, actor: Actor) -> None:
        """Apply terrain hazard once per NPC turn, in explore mode only.

        In explore, an NPC takes hazard damage whenever it consumes a turn -
        acting, waiting (wander pause), or being prevented - so a loitering NPC
        standing in acid or on hot coals still gets hurt. In combat, hazard is
        applied per player action in on_player_action() instead, so we skip it
        here to avoid doubling it.
        """
        if not self.controller.is_combat_mode():
            self._apply_terrain_hazard(actor)

    def _apply_terrain_hazard(self, actor: Actor) -> None:
        """Check if actor is on hazardous terrain and apply damage if so.

        Called after an actor completes their turn. If they're standing on
        hazardous terrain (acid pools, hot coals, etc.), they take damage.

        Args:
            actor: The actor to check for terrain hazard damage.
        """
        from brileta.environment.tile_types import (
            get_tile_hazard_info,
            get_tile_type_name_by_id,
        )
        from brileta.game.actions.environmental import EnvironmentalDamageIntent
        from brileta.util.dice import Dice

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
            # Route through the router directly, not execute_intent(), so this
            # passive side effect never records presentation timing - hazards
            # shouldn't overwrite the pacing state of the turn that triggered them.
            self.action_router.execute_intent(intent)

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
            if (
                not result.succeeded
                and result.block_reason == ActionBlockReason.NOT_ADJACENT
            ):
                self._handle_intent_step_result(self.player, result)

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
        from brileta.game.action_plan import ApproachStep, IntentStep

        plan = actor.active_plan
        if plan is None:
            return None
        self._prune_actor_retry_state(actor, plan)

        step = plan.get_current_step()

        # Plan complete?
        if step is None:
            self._cancel_plan(actor)
            return None

        # Skip steps whose conditions are met
        while (
            step is not None and step.skip_if is not None and step.skip_if(plan.context)
        ):
            plan.advance()
            step = plan.get_current_step()

        if step is None:
            self._cancel_plan(actor)
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
        """Handle an ApproachStep by generating the next MoveIntent or OpenDoorIntent.

        This method:
        1. Checks if we've arrived (based on stop_distance)
        2. Calculates path if needed (door-capable NPCs can route through doors)
        3. PEEKs at the next position (does NOT pop)
        4. If the next tile is a closed door, returns an OpenDoorIntent
        5. Otherwise returns a MoveIntent

        The path is only popped after a successful move, in _on_approach_result().
        Door opens don't pop the path - the NPC opens the door this turn and
        moves through it on the next turn.

        Args:
            actor: The character executing the plan.
            plan: The active plan containing the step.
            step: The ApproachStep to handle.

        Returns:
            A MoveIntent or OpenDoorIntent for the next step, or None if
            arrived/blocked.
        """
        from brileta.game.actions.environment import OpenDoorIntent
        from brileta.game.actions.movement import MoveIntent
        from brileta.util.pathfinding import find_local_path

        actor_can_open_doors = actor.can_open_doors

        # Determine target position
        target_pos = plan.context.resolve_target_pos()
        if target_pos is None:
            self._cancel_plan(actor)
            return None

        # If a target actor moved since path computation, invalidate stale path.
        # This forces re-pathing toward the target's current location.
        if (
            plan.cached_path
            and plan.context.target_actor is not None
            and plan.cached_hierarchical_path is None
        ):
            endpoint = plan.cached_path[-1]
            if calculate_distance(*endpoint, *target_pos) > step.stop_distance:
                plan.cached_path = None
                plan.cached_hierarchical_path = None

        # Check if we've arrived (based on stop_distance)
        current_distance = calculate_distance(actor.x, actor.y, *target_pos)

        if current_distance <= step.stop_distance:
            # Arrived - advance to next step
            plan.advance()
            # Recurse to get intent from next step (if any)
            return self._get_intent_from_plan(actor)

        # Generate path if needed
        if plan.cached_path is None or not plan.cached_path:
            gm = self.controller.gw.game_map
            asi = self.controller.gw.actor_spatial_index
            start_pos: WorldTilePos = (actor.x, actor.y)

            # First try direct path to target. Door-capable NPCs treat
            # closed doors as high-cost passable tiles so they can plan
            # routes through indoor/outdoor boundaries.
            direct = find_local_path(
                gm,
                asi,
                actor,
                start_pos,
                target_pos,
                can_open_doors=actor_can_open_doors,
            )
            if direct:
                plan.cached_path = deque(direct)

            # If direct path fails, try hierarchical (cross-region) pathfinding
            if not plan.cached_path:
                hierarchical = self._try_hierarchical_path(
                    actor,
                    start_pos,
                    target_pos,
                    can_open_doors=actor_can_open_doors,
                )
                if hierarchical:
                    local_path, plan.cached_hierarchical_path = hierarchical
                    plan.cached_path = deque(local_path)

            # If still no path and we have a stop_distance, try adjacent tiles.
            # The target tile is likely occupied (e.g., by an enemy we're approaching).
            if not plan.cached_path and step.stop_distance > 0:
                from brileta.util.pathfinding import probe_step

                best_path: list[WorldTilePos] | None = None
                for dx, dy in DIRECTIONS:
                    tx = target_pos[0] + dx
                    ty = target_pos[1] + dy
                    if (
                        probe_step(gm, self.controller.gw, tx, ty, exclude_actor=actor)
                        is not None
                    ):
                        continue
                    candidate = find_local_path(
                        gm,
                        asi,
                        actor,
                        start_pos,
                        (tx, ty),
                        can_open_doors=actor_can_open_doors,
                    )
                    if candidate and (
                        best_path is None or len(candidate) < len(best_path)
                    ):
                        best_path = candidate
                if best_path is not None:
                    plan.cached_path = deque(best_path)

        if not plan.cached_path:
            # Can't reach target - cancel plan
            self._cancel_plan(actor)
            return None

        # PEEK at next position - don't pop yet!
        # We only pop after confirming the move succeeded.
        next_pos = plan.cached_path[0]

        # If the next tile is a closed door and the actor can open it,
        # emit an OpenDoorIntent instead of a MoveIntent. The NPC spends
        # this turn opening the door, then moves through on the next turn.
        if actor_can_open_doors:
            from brileta.environment.tile_types import TileTypeID

            gm = self.controller.gw.game_map
            if gm.tiles[next_pos[0], next_pos[1]] == TileTypeID.DOOR_CLOSED:
                return OpenDoorIntent(
                    plan.context.controller, actor, next_pos[0], next_pos[1]
                )

        dx = next_pos[0] - actor.x
        dy = next_pos[1] - actor.y

        duration_ms = self._approach_move_duration_ms(current_distance)
        return MoveIntent(plan.context.controller, actor, dx, dy, duration_ms)

    def _approach_move_duration_ms(self, distance: int) -> int:
        """Return a movement duration based on remaining approach distance.

        The curve slows movement as the actor gets closer, creating a more
        deliberate approach near the target while keeping long-distance travel
        snappy. The duration is clamped to [70, 140] ms.
        """
        min_ms = Movement.APPROACH_MIN_DURATION_MS
        max_ms = Movement.APPROACH_MAX_DURATION_MS
        far = Movement.APPROACH_FAR_DISTANCE
        span = Movement.APPROACH_DISTANCE_SPAN

        progress = (far - distance) / span
        duration_ms = min_ms + (max_ms - min_ms) * progress

        clamped = max(min_ms, min(max_ms, duration_ms))
        return round(clamped)

    def _handle_intent_step_result(
        self, actor: Character, result: GameActionResult
    ) -> None:
        """Handle IntentStep failures that should rewind plans to approach.

        For adjacency-required plans, a ``not_adjacent`` failure means the
        target moved after approach but before intent execution. Rewinding to
        the nearest ApproachStep resumes chase. If retry budget is exhausted,
        the plan is cancelled.
        """
        if result.succeeded or result.block_reason != ActionBlockReason.NOT_ADJACENT:
            return

        plan = actor.active_plan
        if plan is None or not plan.plan.requires_adjacency:
            return

        key = self._plan_retry_key(actor, plan)
        retries = self._adjacency_rewind_counts.get(key, 0)
        if retries >= self._MAX_ADJACENCY_REWINDS:
            self._cancel_plan(actor)
            return

        if not plan.rewind_to_previous_approach_step():
            self._cancel_plan(actor)
            return

        self._adjacency_rewind_counts[key] = retries + 1

    def _on_approach_result(self, actor: Character, result: GameActionResult) -> None:
        """Handle the result of a MoveIntent or OpenDoorIntent from an ApproachStep.

        Updates the plan state based on whether the action succeeded:
        - Move success: Pop the path entry the actor moved to.
        - Door open success: Keep the path entry (actor didn't move yet).
        - Failure: Invalidate path for recalculation on next turn.

        Args:
            actor: The character who attempted the action.
            result: The result from executing the intent.
        """
        from brileta.game.action_plan import ApproachStep

        plan = actor.active_plan
        if plan is None:
            return

        step = plan.get_current_step()
        if not isinstance(step, ApproachStep):
            return

        if result.succeeded:
            # Only pop the path entry if the actor actually moved to the
            # expected position. Door opens succeed without moving the actor,
            # so the path entry stays for the follow-up move next turn.
            if plan.cached_path and (actor.x, actor.y) == plan.cached_path[0]:
                plan.cached_path.popleft()

            # After popping, check if we need to compute the next hierarchical segment
            if not plan.cached_path and plan.cached_hierarchical_path:
                self._advance_hierarchical_path(actor, plan)

            # Check if we've now arrived at destination
            target_pos = plan.context.resolve_target_pos()
            if target_pos is None:
                return

            if calculate_distance(actor.x, actor.y, *target_pos) <= step.stop_distance:
                # Arrived - advance to next step (or complete plan)
                plan.advance()
                if plan.is_complete():
                    self._cancel_plan(actor)
        else:
            # Move failed (blocked) - invalidate path for recalculation
            plan.cached_path = None
            plan.cached_hierarchical_path = None

    def _try_hierarchical_path(
        self,
        actor: Character,
        start_pos: WorldTilePos,
        target_pos: WorldTilePos,
        *,
        can_open_doors: bool = False,
    ) -> tuple[list[WorldTilePos], list[int] | None] | None:
        """Attempt hierarchical pathfinding across regions.

        Returns (local_path, high_level_path) if cross-region path is found,
        or None if regions aren't set up or positions are in the same region.

        Args:
            actor: The character pathfinding.
            start_pos: Starting position.
            target_pos: Target position.
            can_open_doors: Whether the actor can open closed doors.

        Returns:
            Tuple of (local_path, high_level_path) or None.
        """
        from brileta.util.pathfinding import find_local_path, find_region_path

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

        # Compute local path to the connection point. Door-capable actors can
        # route through closed doors (e.g., building entrances between regions).
        local_path = find_local_path(
            gm,
            asi,
            actor,
            start_pos,
            connection_point,
            can_open_doors=can_open_doors,
        )
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
        from brileta.util.pathfinding import find_local_path

        if plan.cached_hierarchical_path is None:
            return

        # Only act if local path is exhausted and we have more regions to go
        if plan.cached_path:
            return

        gm = self.controller.gw.game_map
        asi = self.controller.gw.actor_spatial_index
        actor_can_open_doors = actor.can_open_doors

        # Pop the region we just entered
        current_region_id = plan.cached_hierarchical_path.pop(0)

        # Determine target position
        target_pos = plan.context.resolve_target_pos()
        if target_pos is None:
            self._cancel_plan(actor)
            return

        if plan.cached_hierarchical_path:
            # More regions to traverse - path to next connection point
            next_region_id = plan.cached_hierarchical_path[0]
            current_region = gm.regions.get(current_region_id)
            if current_region is None:
                self._cancel_plan(actor)
                return

            connection_point = current_region.connections.get(next_region_id)
            if connection_point is None:
                self._cancel_plan(actor)
                return

            # Compute local path to connection point
            new_path = find_local_path(
                gm,
                asi,
                actor,
                (actor.x, actor.y),
                connection_point,
                can_open_doors=actor_can_open_doors,
            )
            if new_path:
                plan.cached_path = deque(new_path)
            else:
                # Can't reach connection - clear plan
                self._cancel_plan(actor)
        else:
            # We're in the final region - path to target
            new_path = find_local_path(
                gm,
                asi,
                actor,
                (actor.x, actor.y),
                target_pos,
                can_open_doors=actor_can_open_doors,
            )
            if new_path:
                plan.cached_path = deque(new_path)
            # If no path, leave cached_path empty - plan will handle it next turn

    @staticmethod
    def _plan_retry_key(actor: Character, plan: ActivePlan) -> AdjacencyRetryKey:
        """Build a stable key for retry bookkeeping of a specific active plan."""
        return (actor.actor_id, id(plan))

    def _cancel_plan(self, actor: Character) -> None:
        """Cancel an actor's active plan and clean up retry state.

        This is the single point for plan cancellation, ensuring retry
        bookkeeping is always cleaned up alongside the plan reference.
        """
        plan = actor.active_plan
        if plan is not None:
            self._adjacency_rewind_counts.pop(self._plan_retry_key(actor, plan), None)
            actor.active_plan = None

    def _prune_actor_retry_state(
        self, actor: Character, active_plan: ActivePlan
    ) -> None:
        """Drop stale retry entries from prior plans for this actor."""
        current_plan_id: PlanInstanceId = id(active_plan)
        for key in list(self._adjacency_rewind_counts):
            actor_id, plan_id = key
            if actor_id == actor.actor_id and plan_id != current_plan_id:
                self._adjacency_rewind_counts.pop(key, None)

    # === Player Action Queue ===

    def queue_action(self, action: GameIntent) -> None:
        """Queue a non-plan player action (e.g. manual key-press)."""
        self._player_action_queue.append(action)

    def has_pending_actions(self) -> bool:
        """Check if there are pending player actions in the queue."""
        return len(self._player_action_queue) > 0

    def dequeue_player_action(self) -> GameIntent | None:
        """Dequeue and return the next pending player action, or None."""
        return (
            self._player_action_queue.popleft() if self._player_action_queue else None
        )

    def is_player_turn_available(self) -> bool:
        """Return True if player has pending actions or active plans."""
        return self.has_pending_actions() or self.player.active_plan is not None

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
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from catley import config
from catley.game.action_router import ActionRouter
from catley.game.actions.base import GameIntent

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Actor


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

    def _update_energy_actors_cache(self) -> None:
        """Update cached list of actors with energy components for performance."""
        if self._cache_dirty:
            self._energy_actors_cache = [
                actor for actor in self.controller.gw.actors if hasattr(actor, "energy")
            ]
            self._cache_dirty = False

    def invalidate_cache(self) -> None:
        """Mark actor cache as dirty (call when actors are added/removed)."""
        self._cache_dirty = True

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

        # Update energy for all actors with energy components
        for actor in self._energy_actors_cache:
            energy_amount = actor.energy.get_speed_based_energy_amount()
            actor.energy.accumulate_energy(energy_amount)

        # Apply terrain hazard damage to all NPCs once per player action.
        # This ensures NPCs on hot coals take damage each turn, not each tick.
        for actor in self._energy_actors_cache:
            if actor is self.player:
                continue
            self._apply_terrain_hazard(actor)

    def process_all_npc_reactions(self) -> None:
        """Process all NPCs who can currently afford actions immediately.

        Note: This method is called every game tick to check if NPCs can act.
        Hazard damage is NOT applied here - it's applied once per player action
        in on_player_action() to avoid damage being applied every tick.
        """
        for actor in self._energy_actors_cache:
            if actor is self.player:
                continue

            if actor.energy.can_afford(config.ACTION_COST):
                action = actor.get_next_action(self.controller)
                if action is not None:
                    # Update turn effects for this specific actor right before it acts.
                    actor.update_turn(self.controller)
                    # Execute immediately - let the action system handle everything
                    self.execute_intent(action)
                    if hasattr(actor, "energy"):
                        actor.energy.spend(config.ACTION_COST)

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

    def execute_intent(self, intent: GameIntent) -> None:
        """Execute a single GameIntent by routing it to the ActionRouter.

        This method is unchanged from the original - it handles the actual
        execution of actions regardless of whether they're player or NPC actions.

        Args:
            intent: The GameIntent to execute
        """
        self.action_router.execute_intent(intent)

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
        """Return True if player has pending actions or autopilot goals."""
        has_manual_action = self.has_pending_actions()
        has_autopilot_goal = getattr(self.player, "pathfinding_goal", None) is not None
        return has_manual_action or has_autopilot_goal

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
            if hasattr(actor, "energy"):
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
        base_speed = self.player.energy.speed if hasattr(self.player, "energy") else 100

        ratios = {}
        for actor in self.controller.gw.actors:
            if hasattr(actor, "energy"):
                ratio = actor.energy.speed / base_speed
                ratios[actor.name] = ratio

        return ratios

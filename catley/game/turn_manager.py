"""
Manages the game's action scheduling using the Reactive Actor Framework (RAF).

The RAF TurnManager eliminates the concept of synchronized "rounds" in favor
of event-driven scheduling. When the player acts, the world reacts immediately:

1. All actors get their energy updated based on their individual speeds
2. Any NPCs who can now afford actions get those actions queued
3. The Controller smoothly executes these queued actions one per frame

This creates responsive player input (zero delay) while maintaining fair
energy economy (faster actors naturally get more actions over time) and
smooth world presentation (simultaneous NPC reactions).

Key Differences from Round-Based Systems:
- No waiting for "rounds" to complete before accepting player input
- NPCs react immediately to world state changes rather than on fixed schedules
- Energy accumulates independently per actor rather than in synchronized batches
- Action execution is distributed smoothly over frames rather than in bursts
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from catley.game.action_router import ActionRouter
from catley.game.actions.base import GameIntent

if TYPE_CHECKING:
    from catley.controller import Controller


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
        """Initialize the action scheduler.

        Args:
            controller: The main game controller for accessing actors and world state
        """
        self.controller = controller
        self.player = self.controller.gw.player

        # Queue for pending NPC actions (FIFO)
        self._npc_action_queue: deque[GameIntent] = deque()

        # Player action queue (preserved for compatibility)
        self._player_action_queue: deque[GameIntent] = deque()

        # Action execution router (unchanged)
        self.action_router = ActionRouter(self.controller)

    def on_player_action(self) -> None:
        """Called when the player has just acted.

        This is the core of RAF scheduling - immediately update energy for all
        actors and queue any NPC actions that become available. This creates
        the "reactive" behavior where NPCs respond instantly to player actions.
        """
        # Regenerate energy for all actors based on their speed
        for actor in self.controller.gw.actors:
            if hasattr(actor, "energy"):
                actor.energy.regenerate()

        # Queue actions for any NPCs who can now afford them
        for actor in self.controller.gw.actors:
            if actor is self.player:
                continue  # Skip the player

            if (
                hasattr(actor, "energy")
                and hasattr(actor.energy, "can_afford")
                and actor.energy.can_afford(self.controller.action_cost)
            ):
                action = actor.get_next_action(self.controller)
                if action is not None:
                    self._npc_action_queue.append(action)

    def get_next_npc_action(self) -> GameIntent | None:
        """Return the next queued NPC action, or None if no actions pending.

        This method is called by the Controller each frame to smoothly execute
        NPC actions one at a time. Returns None when no NPCs have pending actions.

        Returns:
            The next NPC action to execute, or None if queue is empty
        """
        if self._npc_action_queue:
            return self._npc_action_queue.popleft()
        return None

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
        """Return the current number of queued NPC actions (for debugging)."""
        return len(self._npc_action_queue)

    def clear_npc_queue(self) -> None:
        """Clear all pending NPC actions (for testing/debugging)."""
        self._npc_action_queue.clear()

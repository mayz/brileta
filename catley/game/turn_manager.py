"""
Manages the game's turn-based clock and action queue.
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
    Manages the game's turn-based clock, actor energy, and action queuing.

    The TurnManager is responsible for the "when" of actions, but not the "how."
    Its core responsibility is to manage the `process_unified_round` loop,
    giving energy to actors and allowing them to act when they have enough.

    When an actor decides to perform an action, the TurnManager takes the
    generated `GameIntent` and passes it to its internal `ActionRouter` instance,
    which handles the actual execution and arbitration. The TurnManager does not
    contain any specific game rules about what actions mean or how they are
    resolved.
    """

    def __init__(self, controller: Controller):
        self.controller = controller
        self.player = self.controller.gw.player
        self._action_queue: deque[GameIntent] = deque()
        self.action_router = ActionRouter(self.controller)

    def queue_action(self, action: GameIntent) -> None:
        """Queue a game action to be processed on the next turn."""
        self._action_queue.append(action)

    def has_pending_actions(self) -> bool:
        """
        Check if there is a pending action.
        """
        return len(self._action_queue) > 0

    def is_player_turn_available(self) -> bool:
        """Return True if player input or autopilot requires a turn."""

        has_manual_action = self.has_pending_actions()
        has_autopilot_goal = getattr(self.player, "pathfinding_goal", None) is not None
        return has_manual_action or has_autopilot_goal

    # Backwards compatibility for old name
    def is_turn_available(self) -> bool:  # pragma: no cover - legacy
        return self.is_player_turn_available()

    def dequeue_player_action(self) -> GameIntent | None:
        """Dequeue and return the pending player action."""
        return self._action_queue.popleft() if len(self._action_queue) > 0 else None

    def execute_intent(self, intent: GameIntent) -> None:
        """Execute a single GameIntent by routing it to the ActionRouter."""
        self.action_router.execute_intent(intent)

    def process_all_npc_turns(self) -> None:
        """Resolve all NPC actions for the current turn."""
        pass

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

    def process_unified_round(self) -> None:
        """Process a single round where all actors can act."""
        if not self.has_pending_actions():
            return

        for actor in self.controller.gw.actors:
            actor.update_turn(self.controller)

        for actor in self.controller.gw.actors:
            actor.energy.regenerate()

        max_iterations = 50
        iteration = 0
        while iteration < max_iterations:
            someone_acted = False
            iteration += 1

            for actor in self.controller.gw.actors.copy():
                if actor.energy.can_afford(self.controller.action_cost):
                    action = actor.get_next_action(self.controller)
                    actor.energy.spend(self.controller.action_cost)

                    if action:
                        self.action_router.execute_intent(action)
                        someone_acted = True

                    if not self.player.health.is_alive():
                        # The router handles events, but we stop the loop here.
                        return

            if not someone_acted:
                break

        if hasattr(self.controller, "active_mode") and self.controller.active_mode:
            self.controller.active_mode.update()

    def dequeue_player_action(self) -> GameIntent | None:
        """Dequeue and return the pending player action."""
        return self._action_queue.popleft() if len(self._action_queue) > 0 else None

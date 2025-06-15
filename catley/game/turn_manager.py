from __future__ import annotations

from typing import TYPE_CHECKING

from catley.game.action_router import ActionRouter
from catley.game.actions.base import GameIntent

if TYPE_CHECKING:
    from catley.controller import Controller


class TurnManager:
    """Manages turn-based gameplay, actor energy, and action queuing."""

    def __init__(self, controller: Controller):
        self.controller = controller
        self.player = self.controller.gw.player
        self._pending_action: GameIntent | None = None
        self.action_router = ActionRouter(self.controller)

    def queue_action(self, action: GameIntent) -> None:
        """Queue a game action to be processed on the next turn."""
        self._pending_action = action

    def process_unified_round(self) -> None:
        """Process a single round where all actors can act."""
        if not self._pending_action:
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
        action = self._pending_action
        self._pending_action = None
        return action

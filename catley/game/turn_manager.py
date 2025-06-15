from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.events import MessageEvent, publish_event
from catley.game.actions.area_effects import AreaEffectIntent
from catley.game.actions.base import GameAction, GameActionResult, GameIntent
from catley.game.actions.combat import AttackIntent, ReloadIntent
from catley.game.actions.environment import CloseDoorIntent, OpenDoorIntent
from catley.game.actions.executors.area_effects import AreaEffectExecutor
from catley.game.actions.executors.base import ActionExecutor
from catley.game.actions.executors.combat import AttackExecutor, ReloadExecutor
from catley.game.actions.executors.environment import (
    CloseDoorExecutor,
    OpenDoorExecutor,
)
from catley.game.actions.executors.misc import PickupExecutor, SwitchWeaponExecutor
from catley.game.actions.executors.movement import MoveExecutor
from catley.game.actions.executors.recovery import (
    ComfortableSleepExecutor,
    RestExecutor,
    SleepExecutor,
    UseConsumableExecutor,
)
from catley.game.actions.misc import PickupIntent, SwitchWeaponIntent
from catley.game.actions.movement import MoveIntent
from catley.game.actions.recovery import (
    ComfortableSleepIntent,
    RestIntent,
    SleepIntent,
    UseConsumableIntent,
)

if TYPE_CHECKING:
    from catley.controller import Controller


class TurnManager:
    """Manages turn-based gameplay, actor energy, and action execution."""

    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.player = self.controller.gw.player
        self._pending_action: GameAction | GameIntent | None = None
        # Executor registry for future Intent/Executor dispatch
        # TODO: Populate in Task 2 - not used yet
        self._executor_registry: dict[type, ActionExecutor] = {}

    def queue_action(self, action: GameAction | GameIntent) -> None:
        """Queue a game action to be processed on the next turn."""
        self._pending_action = action

    def process_unified_round(self) -> None:
        """Process a single round where all actors can act."""
        # Only process if there's actually a pending action to start the round
        if not self._pending_action:
            return

        # Let actors spend their accumulated points to take actions
        max_iterations = 50  # Prevent infinite loops
        iteration = 0

        # Process ongoing status effects at the start of the round so new
        # effects applied during a round last until the next round begins.
        for actor in self.controller.gw.actors:
            actor.update_turn(self.controller)

        # Give all actors energy once per round
        for actor in self.controller.gw.actors:
            actor.energy.regenerate()

        while iteration < max_iterations:
            someone_acted = False
            iteration += 1

            # Check each actor to see if they can act
            for actor in self.controller.gw.actors.copy():
                if actor.energy.can_afford(self.controller.action_cost):
                    action = actor.get_next_action(self.controller)
                    # Always spend energy if you can afford to act. No hoarding!
                    actor.energy.spend(self.controller.action_cost)

                    if action:
                        self._execute_action(action)
                        someone_acted = True

                    if not self.player.health.is_alive():
                        # Player died, handle game over and stop processing
                        message = (
                            f"{self.controller.gw.player.name} has been defeated!"
                            " Game Over."
                        )
                        publish_event(MessageEvent(message, colors.RED))
                        return

            # If nobody acted this round, we're done
            if not someone_acted:
                break

        # After the round, update any active modes
        if hasattr(self.controller, "active_mode") and self.controller.active_mode:
            self.controller.active_mode.update()

    def dequeue_player_action(self) -> GameAction | GameIntent | None:
        """Dequeue and return the pending player action."""
        action = self._pending_action
        self._pending_action = None
        return action

    def _execute_action(self, action: GameAction | GameIntent | None) -> None:
        """Execute an action immediately without turn processing."""
        if not action:
            return

        try:
            if isinstance(action, MoveIntent):
                executor = MoveExecutor()
                result = executor.execute(action)
            elif isinstance(action, AttackIntent):
                executor = AttackExecutor()
                result = executor.execute(action)
            elif isinstance(action, ReloadIntent):
                executor = ReloadExecutor()
                result = executor.execute(action)
            elif isinstance(action, PickupIntent):
                executor = PickupExecutor()
                result = executor.execute(action)
            elif isinstance(action, OpenDoorIntent):
                executor = OpenDoorExecutor()
                result = executor.execute(action)
            elif isinstance(action, CloseDoorIntent):
                executor = CloseDoorExecutor()
                result = executor.execute(action)
            elif isinstance(action, SwitchWeaponIntent):
                executor = SwitchWeaponExecutor()
                result = executor.execute(action)
            elif isinstance(action, UseConsumableIntent):
                executor = UseConsumableExecutor()
                result = executor.execute(action)
            elif isinstance(action, RestIntent):
                executor = RestExecutor()
                result = executor.execute(action)
            elif isinstance(action, SleepIntent):
                executor = SleepExecutor()
                result = executor.execute(action)
            elif isinstance(action, ComfortableSleepIntent):
                executor = ComfortableSleepExecutor()
                result = executor.execute(action)
            elif isinstance(action, AreaEffectIntent):
                executor = AreaEffectExecutor()
                result = executor.execute(action)
            else:
                assert isinstance(action, GameAction)
                result = action.execute()
        except SystemExit:
            raise
        except Exception as e:
            error_message = f"Error executing action '{type(action).__name__}': {e!s}"
            publish_event(MessageEvent(error_message, colors.RED))
            return

        if result is None:
            result = GameActionResult()

        # Update FOV if the player's action indicated a change in visibility
        if action.actor == self.controller.gw.player and result.should_update_fov:
            self.controller.update_fov()

    def _get_executor_for_intent(
        self, intent: GameAction | GameIntent
    ) -> ActionExecutor | None:
        """Get the appropriate executor for an intent type.

        TODO: Implement routing logic in Task 2
        """
        return self._executor_registry.get(type(intent))

"""
The central dispatcher and arbiter for the game's action system.

This module contains the ActionRouter, which is the heart of the Intent/Executor
pattern. It is the single point of entry for all game actions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from catley.game.actions.area_effects import AreaEffectIntent
from catley.game.actions.base import GameActionResult, GameIntent
from catley.game.actions.combat import AttackIntent, ReloadIntent
from catley.game.actions.environment import (
    CloseDoorIntent,
    OpenDoorIntent,
    SearchContainerIntent,
)
from catley.game.actions.environmental import EnvironmentalDamageIntent
from catley.game.actions.executors.area_effects import WeaponAreaEffectExecutor
from catley.game.actions.executors.base import ActionExecutor
from catley.game.actions.executors.combat import AttackExecutor, ReloadExecutor
from catley.game.actions.executors.containers import SearchContainerExecutor
from catley.game.actions.executors.doors import (
    CloseDoorExecutor,
    OpenDoorExecutor,
)
from catley.game.actions.executors.environmental import EnvironmentalDamageExecutor
from catley.game.actions.executors.misc import (
    DropItemExecutor,
    PickupExecutor,
    PickupItemsAtLocationExecutor,
    SwitchWeaponExecutor,
)
from catley.game.actions.executors.movement import MoveExecutor
from catley.game.actions.executors.recovery import (
    ComfortableSleepExecutor,
    RestExecutor,
    SleepExecutor,
    UseConsumableExecutor,
    UseConsumableOnTargetExecutor,
)
from catley.game.actions.executors.social import TalkExecutor
from catley.game.actions.executors.stunts import PushExecutor
from catley.game.actions.misc import (
    DropItemIntent,
    PickupIntent,
    PickupItemsAtLocationIntent,
    SwitchWeaponIntent,
)
from catley.game.actions.movement import MoveIntent
from catley.game.actions.recovery import (
    ComfortableSleepIntent,
    RestIntent,
    SleepIntent,
    UseConsumableIntent,
    UseConsumableOnTargetIntent,
)
from catley.game.actions.social import TalkIntent
from catley.game.actions.stunts import PushIntent
from catley.game.actors import Character
from catley.game.actors.container import Container

if TYPE_CHECKING:
    from catley.controller import Controller


class ActionRouter:
    """
    Dispatches intents to executors and arbitrates their results.

    The ActionRouter has two primary responsibilities:
    1.  Dispatching: It maintains a registry mapping `GameIntent` types to their
        corresponding `ActionExecutor` classes. When it receives an intent, it
        looks up and calls the correct executor.
    2.  Arbitration: It inspects the `GameActionResult` returned by an executor.
        It is the single, centralized location for all "World Interaction Rules"
        that define what happens in special cases. For example, it contains the
        rule that a failed `MoveIntent` due to collision with an actor should be
        translated into a new `AttackIntent`.

    This class is the "private implementation" of the action system and should
    only be instantiated and used by the `TurnManager`.
    """

    def __init__(self, controller: Controller):
        self.controller = controller
        # This registry is the heart of the dispatcher.
        self._executor_registry: dict[type, ActionExecutor] = {
            MoveIntent: MoveExecutor(),
            AttackIntent: AttackExecutor(),
            ReloadIntent: ReloadExecutor(),
            PickupIntent: PickupExecutor(),
            PickupItemsAtLocationIntent: PickupItemsAtLocationExecutor(),
            SwitchWeaponIntent: SwitchWeaponExecutor(),
            DropItemIntent: DropItemExecutor(),
            OpenDoorIntent: OpenDoorExecutor(),
            CloseDoorIntent: CloseDoorExecutor(),
            SearchContainerIntent: SearchContainerExecutor(),
            UseConsumableIntent: UseConsumableExecutor(),
            UseConsumableOnTargetIntent: UseConsumableOnTargetExecutor(),
            RestIntent: RestExecutor(),
            SleepIntent: SleepExecutor(),
            ComfortableSleepIntent: ComfortableSleepExecutor(),
            AreaEffectIntent: WeaponAreaEffectExecutor(),
            EnvironmentalDamageIntent: EnvironmentalDamageExecutor(),
            PushIntent: PushExecutor(),
            TalkIntent: TalkExecutor(),
        }

    def execute_intent(self, intent: GameIntent) -> None:
        """The single public entry point for executing any action.
        It dispatches the intent and arbitrates its result."""
        executor = self._executor_registry.get(type(intent))
        if not executor:
            # Optionally, log an error for an unhandled intent type.
            return

        result = executor.execute(intent)
        if not result:
            result = GameActionResult()

        # Arbitrate the result to handle special cases and chained actions.
        self._arbitrate_result(intent, result)

    def _arbitrate_result(
        self, original_intent: GameIntent, result: GameActionResult
    ) -> None:
        """Contains all the world's interaction rules for action outcomes."""
        if result.succeeded:
            # Handle successful action finalization (e.g., FOV updates).
            if (
                original_intent.actor == self.controller.gw.player
                and result.should_update_fov
            ):
                self.controller.update_fov()
            return

        # Handle failed actions that might trigger a new intent.
        if isinstance(original_intent, MoveIntent):
            self._handle_failed_move(original_intent, result)

    def _handle_failed_move(self, intent: MoveIntent, result: GameActionResult) -> None:
        """Rulebook for what happens after a failed move."""
        if result.block_reason == "actor" and result.blocked_by:
            # The actor who tried to move.
            bumper = intent.actor
            # The actor who was in the way.
            bumpee = cast(Character, result.blocked_by)

            # Ensure the bumpee is a valid, living target.
            if not bumpee.health or not bumpee.health.is_alive():
                return

            # --- WORLD INTERACTION RULE ---
            # Player bumping an NPC does NOT trigger attack. The player stays
            # adjacent and can choose an action from the ActionPanel. This shifts
            # the game away from combat-by-default toward deliberate interaction.
            if bumper is self.controller.gw.player:
                return

            # An NPC ramming another actor should ONLY trigger an attack
            # if the target is the player. This prevents friendly fire from
            # pathfinding traffic jams.
            if bumpee is self.controller.gw.player:
                new_intent = AttackIntent(
                    self.controller,
                    bumper,
                    bumpee,
                    weapon=None,
                    attack_mode="melee",
                )
                self.execute_intent(new_intent)
                return

            # In all other cases (NPC bumping into another NPC), do nothing.
            # The NPC will just wait its turn and the AI will reroute on its
            # next action.

        elif result.block_reason == "door" and result.blocked_by:
            # Rule: Bumping into a closed door means you try to open it.
            door_pos = cast(tuple[int, int], result.blocked_by)
            new_intent = OpenDoorIntent(
                self.controller, intent.actor, door_pos[0], door_pos[1]
            )
            self.execute_intent(new_intent)  # Recursive call

        elif result.block_reason == "container" and result.blocked_by:
            # Rule: Bumping into a container means you try to search it.
            container = cast(Container, result.blocked_by)
            new_intent = SearchContainerIntent(self.controller, intent.actor, container)
            self.execute_intent(new_intent)  # Recursive call

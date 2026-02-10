"""
The central dispatcher and arbiter for the game's action system.

This module contains the ActionRouter, which is the heart of the Intent/Executor
pattern. It is the single point of entry for all game actions.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

from brileta import colors
from brileta.events import (
    FloatingTextEvent,
    FloatingTextSize,
    FloatingTextValence,
    publish_event,
)
from brileta.game.actions.area_effects import AreaEffectIntent
from brileta.game.actions.base import GameActionResult, GameIntent
from brileta.game.actions.combat import AttackIntent, ReloadIntent
from brileta.game.actions.environment import (
    CloseDoorIntent,
    OpenDoorIntent,
    SearchContainerIntent,
)
from brileta.game.actions.environmental import EnvironmentalDamageIntent
from brileta.game.actions.executors.area_effects import WeaponAreaEffectExecutor
from brileta.game.actions.executors.base import ActionExecutor
from brileta.game.actions.executors.combat import AttackExecutor, ReloadExecutor
from brileta.game.actions.executors.containers import SearchContainerExecutor
from brileta.game.actions.executors.doors import (
    CloseDoorExecutor,
    OpenDoorExecutor,
)
from brileta.game.actions.executors.environmental import EnvironmentalDamageExecutor
from brileta.game.actions.executors.misc import (
    DropCountableExecutor,
    DropItemExecutor,
    PickupCountableExecutor,
    PickupExecutor,
    PickupItemsAtLocationExecutor,
    SwitchWeaponExecutor,
)
from brileta.game.actions.executors.movement import MoveExecutor
from brileta.game.actions.executors.recovery import (
    ComfortableSleepExecutor,
    RestExecutor,
    SleepExecutor,
    UseConsumableExecutor,
    UseConsumableOnTargetExecutor,
)
from brileta.game.actions.executors.social import TalkExecutor
from brileta.game.actions.executors.stunts import (
    HolsterWeaponExecutor,
    KickExecutor,
    PunchExecutor,
    PushExecutor,
    TripExecutor,
)
from brileta.game.actions.misc import (
    DropCountableIntent,
    DropItemIntent,
    PickupCountableIntent,
    PickupIntent,
    PickupItemsAtLocationIntent,
    SwitchWeaponIntent,
)
from brileta.game.actions.movement import MoveIntent
from brileta.game.actions.recovery import (
    ComfortableSleepIntent,
    RestIntent,
    SleepIntent,
    UseConsumableIntent,
    UseConsumableOnTargetIntent,
)
from brileta.game.actions.social import TalkIntent
from brileta.game.actions.stunts import (
    HolsterWeaponIntent,
    KickIntent,
    PunchIntent,
    PushIntent,
    TripIntent,
)
from brileta.game.actors import NPC, Character
from brileta.game.actors.barks import pick_bump_bark
from brileta.game.actors.container import Container

if TYPE_CHECKING:
    from brileta.controller import Controller


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
        self._bark_block_until: dict[int, float] = {}
        self._bark_cooldown_seconds = 0.25
        # This registry is the heart of the dispatcher.
        self._executor_registry: dict[type, ActionExecutor] = {
            MoveIntent: MoveExecutor(),
            AttackIntent: AttackExecutor(),
            ReloadIntent: ReloadExecutor(),
            PickupIntent: PickupExecutor(),
            PickupItemsAtLocationIntent: PickupItemsAtLocationExecutor(),
            SwitchWeaponIntent: SwitchWeaponExecutor(),
            DropItemIntent: DropItemExecutor(),
            PickupCountableIntent: PickupCountableExecutor(),
            DropCountableIntent: DropCountableExecutor(),
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
            TripIntent: TripExecutor(),
            KickIntent: KickExecutor(),
            PunchIntent: PunchExecutor(),
            HolsterWeaponIntent: HolsterWeaponExecutor(),
            TalkIntent: TalkExecutor(),
        }

    def execute_intent(self, intent: GameIntent) -> GameActionResult:
        """The single public entry point for executing any action.

        Dispatches the intent to its executor and arbitrates the result.
        Returns the GameActionResult so callers (like the plan system) can
        inspect whether the action succeeded and decide how to proceed.

        Returns:
            GameActionResult with succeeded=False if no executor found,
            otherwise the result from the executor after arbitration.
        """
        executor = self._executor_registry.get(type(intent))
        if not executor:
            # No executor registered for this intent type.
            return GameActionResult(succeeded=False)

        result = executor.execute(intent)
        if not result:
            result = GameActionResult()

        # Arbitrate the result to handle special cases and chained actions.
        self._arbitrate_result(intent, result)

        return result

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
                self._emit_bump_bark(bumpee)
                return

            # NPC bumping into any actor does nothing. Attack intent must come
            # from utility scoring rather than collision side effects. This also
            # prevents wandering/pathing traffic from causing combat. The NPC will
            # just wait its turn and the AI will reroute on its next action.

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

    def _emit_bump_bark(self, target: Character) -> None:
        """Show a short speech bubble when the player bumps an NPC."""
        if not isinstance(target, NPC):
            return

        now = time.perf_counter()
        block_until = self._bark_block_until.get(id(target))
        if block_until is not None and now < block_until:
            return

        player = self.controller.gw.player
        if player is None:
            return

        bark = pick_bump_bark(target, player)
        if not bark:
            return

        duration = 1.1
        self._bark_block_until[id(target)] = (
            now + duration + self._bark_cooldown_seconds
        )

        publish_event(
            FloatingTextEvent(
                text=bark,
                target_actor_id=target.actor_id,
                valence=FloatingTextValence.NEUTRAL,
                size=FloatingTextSize.NORMAL,
                duration=duration,
                color=colors.LIGHT_GREY,
                world_x=target.x,
                world_y=target.y,
                bubble=True,
            )
        )

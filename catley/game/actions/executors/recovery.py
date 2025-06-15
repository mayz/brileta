from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.events import MessageEvent, publish_event
from catley.game.actions.base import GameActionResult
from catley.game.actions.executors.base import ActionExecutor
from catley.game.actors import Character, conditions

if TYPE_CHECKING:
    from catley.game.actions.recovery import (
        ComfortableSleepIntent,
        RestIntent,
        SleepIntent,
        UseConsumableIntent,
    )


class UseConsumableExecutor(ActionExecutor):
    """Executes consumable item usage intents."""

    def __init__(self) -> None:
        """Create a UseConsumableExecutor without requiring a controller."""
        pass

    def execute(self, intent: UseConsumableIntent) -> GameActionResult | None:  # type: ignore[override]
        if not intent.item.consumable_effect:
            publish_event(
                MessageEvent(f"{intent.item.name} cannot be used", colors.RED)
            )
            return GameActionResult(succeeded=False)

        inventory = intent.actor.inventory
        if not inventory:
            return GameActionResult(succeeded=False)

        if not inventory.remove_from_inventory(intent.item):
            publish_event(MessageEvent("Item not found in inventory", colors.RED))
            return GameActionResult(succeeded=False)

        success = intent.item.consumable_effect.consume(intent.actor, intent.controller)

        if not success:
            inventory.add_to_inventory(intent.item)
            publish_event(MessageEvent(f"Cannot use {intent.item.name}", colors.RED))

        return GameActionResult()


class RestExecutor(ActionExecutor):
    """Executes rest intents for armor point recovery."""

    def __init__(self) -> None:
        """Create a RestExecutor without requiring a controller."""
        pass

    def execute(self, intent: RestIntent) -> GameActionResult | None:  # type: ignore[override]
        from catley.game.actions.recovery import is_safe_location

        assert isinstance(intent.actor, Character)
        safe, reason = is_safe_location(intent.actor)
        if not safe:
            publish_event(MessageEvent(f"Cannot rest: {reason}", colors.RED))
            return GameActionResult(succeeded=False)

        if not intent.actor.health:
            return GameActionResult(succeeded=False)

        if intent.actor.health.ap == 0:
            publish_event(
                MessageEvent(
                    f"{intent.actor.name}'s armor is broken and cannot be restored.",
                    colors.YELLOW,
                )
            )
            return GameActionResult(succeeded=False)

        if intent.actor.health.ap < intent.actor.health.max_ap:
            intent.actor.health.ap = intent.actor.health.max_ap
            publish_event(MessageEvent("Recovered all AP", colors.GREEN))

        return GameActionResult()


class SleepExecutor(ActionExecutor):
    """Executes sleep intents for HP restoration and exhaustion removal."""

    def __init__(self) -> None:
        """Create a SleepExecutor without requiring a controller."""
        pass

    def execute(self, intent: SleepIntent) -> GameActionResult | None:  # type: ignore[override]
        from catley.game.actions.recovery import is_safe_location

        assert isinstance(intent.actor, Character)
        safe, reason = is_safe_location(intent.actor)
        if not safe:
            publish_event(MessageEvent(f"Cannot sleep: {reason}", colors.RED))
            return GameActionResult(succeeded=False)

        if intent.actor.health:
            intent.actor.health.hp = intent.actor.health.max_hp
            publish_event(MessageEvent("Restored all HP", colors.GREEN))

        if intent.actor.conditions:
            exhaustion = intent.actor.conditions.get_conditions_by_type(
                conditions.Exhaustion
            )
            if exhaustion:
                intent.actor.conditions.remove_condition(exhaustion[0])
                publish_event(MessageEvent("Removed 1 Exhaustion", colors.GREEN))

        return GameActionResult()


class ComfortableSleepExecutor(ActionExecutor):
    """Executes comfortable sleep intents for full exhaustion removal."""

    def __init__(self) -> None:
        """Create a ComfortableSleepExecutor without requiring a controller."""
        pass

    def execute(self, intent: ComfortableSleepIntent) -> GameActionResult | None:  # type: ignore[override]
        from catley.game.actions.recovery import is_safe_location

        assert isinstance(intent.actor, Character)
        safe, reason = is_safe_location(intent.actor)
        if not safe:
            publish_event(MessageEvent(f"Cannot sleep: {reason}", colors.RED))
            return GameActionResult(succeeded=False)

        if intent.actor.health:
            intent.actor.health.hp = intent.actor.health.max_hp
            publish_event(MessageEvent("Restored all HP", colors.GREEN))

        if intent.actor.conditions:
            removed = intent.actor.conditions.remove_conditions_by_type(
                {conditions.Exhaustion}
            )
            if removed:
                publish_event(
                    MessageEvent(
                        f"Removed all {len(removed)} Exhaustion conditions",
                        colors.GREEN,
                    )
                )
            else:
                publish_event(MessageEvent("No Exhaustion to remove", colors.YELLOW))

        return GameActionResult()

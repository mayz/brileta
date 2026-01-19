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
        UseConsumableOnTargetIntent,
    )


class UseConsumableExecutor(ActionExecutor):
    """Executes consumable item usage intents."""

    def execute(self, intent: UseConsumableIntent) -> GameActionResult | None:  # type: ignore[override]
        if not intent.item.consumable_effect:
            publish_event(
                MessageEvent(f"{intent.item.name} cannot be used", colors.RED)
            )
            return GameActionResult(succeeded=False)

        inventory = intent.actor.inventory
        if not inventory:
            return GameActionResult(succeeded=False)

        # Use try_remove_item which checks attack_slots first, then outfit, then stored
        if not inventory.try_remove_item(intent.item):
            publish_event(MessageEvent("Item not found in inventory", colors.RED))
            return GameActionResult(succeeded=False)

        success = intent.item.consumable_effect.consume(intent.actor, intent.controller)

        if not success:
            inventory.add_to_inventory(intent.item)
            publish_event(MessageEvent(f"Cannot use {intent.item.name}", colors.RED))

        return GameActionResult()


class UseConsumableOnTargetExecutor(ActionExecutor):
    """Executes consumable item usage on another character.

    Similar to UseConsumableExecutor but applies the effect to a target character
    rather than the actor themselves. Validates adjacency before use.
    """

    def execute(self, intent: UseConsumableOnTargetIntent) -> GameActionResult | None:  # type: ignore[override]
        from catley.game import ranges

        if not intent.item.consumable_effect:
            publish_event(
                MessageEvent(f"{intent.item.name} cannot be used", colors.RED)
            )
            return GameActionResult(succeeded=False)

        # Validate adjacency
        distance = ranges.calculate_distance(
            intent.actor.x, intent.actor.y, intent.target.x, intent.target.y
        )
        if distance > 1:
            publish_event(
                MessageEvent(
                    f"Too far away to use {intent.item.name} on target", colors.RED
                )
            )
            return GameActionResult(succeeded=False)

        # Validate target is alive
        if not intent.target.health.is_alive():
            publish_event(
                MessageEvent(
                    f"Cannot use {intent.item.name} on a dead character", colors.RED
                )
            )
            return GameActionResult(succeeded=False)

        inventory = intent.actor.inventory
        if not inventory:
            return GameActionResult(succeeded=False)

        # Use try_remove_item which checks attack_slots first, then outfit, then stored
        if not inventory.try_remove_item(intent.item):
            publish_event(MessageEvent("Item not found in inventory", colors.RED))
            return GameActionResult(succeeded=False)

        # Apply consumable effect to target (not the user/actor who is doing the action)
        success = intent.item.consumable_effect.consume(
            intent.target, intent.controller
        )

        if not success:
            inventory.add_to_inventory(intent.item)
            publish_event(
                MessageEvent(
                    f"Cannot use {intent.item.name} on {intent.target.name}", colors.RED
                )
            )

        return GameActionResult()


class RestExecutor(ActionExecutor):
    """Executes rest intents for outfit AP recovery.

    Rest recovers 1 AP if:
    - Character has an outfit with AP
    - The outfit was damaged since last rest
    - The outfit is not broken (AP > 0)

    Broken armor requires repair, not rest.
    """

    def execute(self, intent: RestIntent) -> GameActionResult | None:  # type: ignore[override]
        from catley.game import outfit
        from catley.game.actions.recovery import is_safe_location

        assert isinstance(intent.actor, Character)
        safe, reason = is_safe_location(intent.actor)
        if not safe:
            publish_event(MessageEvent(f"Cannot rest: {reason}", colors.RED))
            return GameActionResult(succeeded=False)

        # Get equipped outfit capability
        outfit_cap = intent.actor.inventory.outfit_capability

        if outfit_cap is None:
            publish_event(MessageEvent("Nothing to recover.", colors.YELLOW))
            return GameActionResult(succeeded=False)

        # Check if outfit is broken (requires repair, not rest)
        if outfit_cap.is_broken:
            equipped_outfit = intent.actor.inventory.equipped_outfit
            outfit_name = equipped_outfit[0].name if equipped_outfit else "Armor"
            publish_event(
                MessageEvent(
                    f"{outfit_name} is broken and requires repair.",
                    colors.YELLOW,
                )
            )
            return GameActionResult(succeeded=False)

        # Try to recover AP
        if outfit.rest_recovery(outfit_cap):
            equipped_outfit = intent.actor.inventory.equipped_outfit
            outfit_name = equipped_outfit[0].name if equipped_outfit else "Armor"
            publish_event(
                MessageEvent(
                    f"Recovered 1 AP on {outfit_name}.",
                    colors.GREEN,
                )
            )
        else:
            publish_event(MessageEvent("Nothing to recover.", colors.YELLOW))

        return GameActionResult()


class SleepExecutor(ActionExecutor):
    """Executes sleep intents for HP restoration and exhaustion removal."""

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

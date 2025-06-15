"""
Combat actions for direct actor-vs-actor confrontations.

Handles attack resolution, damage calculation, weapon reloading, and all
combat-related mechanics including hit/miss determination and effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.events import MessageEvent, publish_event
from catley.game.actions.base import GameAction, GameActionResult
from catley.game.actors import Character
from catley.game.items.item_core import Item

if TYPE_CHECKING:
    from catley.controller import Controller


class AttackIntent(GameAction):
    """Intent for one `Actor` to attack another."""

    def __init__(
        self,
        controller: Controller,
        attacker: Character,
        defender: Character,
        weapon: Item | None = None,
        attack_mode: str | None = None,
    ) -> None:
        """Create a combat intent.

        Args:
            controller: Game controller providing context.
            attacker: The character performing the attack.
            defender: The target of the attack.
            weapon: Optional weapon to use. Defaults to the attacker's active
                weapon or bare hands.
            attack_mode: Explicit attack mode ("melee" or "ranged"). ``None``
                preserves the old behaviour of automatically choosing based on
                distance.
        """

        super().__init__(controller, attacker)
        self.attacker = attacker
        self.defender = defender
        self.weapon = weapon
        self.attack_mode = attack_mode

    def execute(self) -> GameActionResult | None:
        """Temporary method for compatibility."""
        raise NotImplementedError(
            "AttackIntent.execute() removed. Use AttackExecutor in Task 2B."
        )


class ReloadAction(GameAction):
    """Action for reloading a ranged weapon."""

    def __init__(self, controller: Controller, actor: Character, weapon: Item) -> None:
        super().__init__(controller, actor)
        self.weapon = weapon

        # Type narrowing.
        self.actor: Character

    def execute(self) -> GameActionResult | None:
        ranged_attack = self.weapon.ranged_attack
        if not ranged_attack:
            return None

        # Find compatible ammo in inventory
        ammo_item = None
        for item in self.actor.inventory:
            if (
                isinstance(item, Item)
                and item.ammo
                and item.ammo.ammo_type == ranged_attack.ammo_type
            ):
                ammo_item = item
                break

        if ammo_item:
            # Remove ammo from inventory and reload weapon
            self.actor.inventory.remove_from_inventory(ammo_item)
            ranged_attack.current_ammo = ranged_attack.max_ammo
            publish_event(
                MessageEvent(
                    f"{self.actor.name} reloaded {self.weapon.name}.", colors.GREEN
                )
            )
        else:
            publish_event(
                MessageEvent(
                    f"No {ranged_attack.ammo_type} ammo available!", colors.RED
                )
            )
        return None

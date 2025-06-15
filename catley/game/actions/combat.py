"""
Combat actions for direct actor-vs-actor confrontations.

Handles attack resolution, damage calculation, weapon reloading, and all
combat-related mechanics including hit/miss determination and effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley.game.actions.base import GameIntent
from catley.game.actors import Character
from catley.game.items.item_core import Item

if TYPE_CHECKING:
    from catley.controller import Controller


class AttackIntent(GameIntent):
    """Intent for one :class:`Actor` to attack another.

    This object only stores attack parameters. Execution occurs via
    :class:`~catley.game.actions.executors.combat.AttackExecutor` through the
    :class:`~catley.game.turn_manager.TurnManager`.
    """

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


class ReloadIntent(GameIntent):
    """Intent for reloading a ranged weapon.

    Execution is performed by
    :class:`~catley.game.actions.executors.combat.ReloadExecutor`.
    """

    def __init__(self, controller: Controller, actor: Character, weapon: Item) -> None:
        super().__init__(controller, actor)
        self.weapon = weapon

        # Type narrowing.
        self.actor: Character

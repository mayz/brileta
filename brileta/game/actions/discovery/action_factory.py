from __future__ import annotations

from typing import TYPE_CHECKING

from brileta.game.actions.combat import AttackIntent, ReloadIntent
from brileta.game.actions.recovery import UseConsumableIntent

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actors import Character
    from brileta.game.items.item_core import Item


class ActionFactory:
    """Factory for creating game action instances."""

    def create_melee_attack(
        self,
        controller: Controller,
        attacker: Character,
        target: Character,
        weapon: Item,
    ) -> AttackIntent:
        """Helper to build a melee AttackIntent."""
        return AttackIntent(controller, attacker, target, weapon, attack_mode="melee")

    def create_ranged_attack(
        self,
        controller: Controller,
        attacker: Character,
        target: Character,
        weapon: Item,
    ) -> AttackIntent:
        """Helper to build a ranged AttackIntent."""
        return AttackIntent(controller, attacker, target, weapon, attack_mode="ranged")

    def create_reload_action(
        self, controller: Controller, actor: Character, weapon: Item
    ) -> ReloadIntent:
        return ReloadIntent(controller, actor, weapon)

    def create_use_consumable_action(
        self, controller: Controller, actor: Character, item: Item
    ) -> UseConsumableIntent:
        return UseConsumableIntent(controller, actor, item)

from __future__ import annotations

from typing import TYPE_CHECKING

from catley.game.actions.combat import AttackAction, ReloadAction
from catley.game.actions.recovery import UseConsumableAction

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Character
    from catley.game.items.item_core import Item


class ActionFactory:
    """Factory for creating game action instances."""

    def create_melee_attack(
        self,
        controller: Controller,
        attacker: Character,
        target: Character,
        weapon: Item,
    ) -> AttackAction:
        """Helper to build a melee AttackAction."""
        return AttackAction(controller, attacker, target, weapon, attack_mode="melee")

    def create_ranged_attack(
        self,
        controller: Controller,
        attacker: Character,
        target: Character,
        weapon: Item,
    ) -> AttackAction:
        """Helper to build a ranged AttackAction."""
        return AttackAction(controller, attacker, target, weapon, attack_mode="ranged")

    def create_reload_action(
        self, controller: Controller, actor: Character, weapon: Item
    ) -> ReloadAction:
        return ReloadAction(controller, actor, weapon)

    def create_use_consumable_action(
        self, controller: Controller, actor: Character, item: Item
    ) -> UseConsumableAction:
        return UseConsumableAction(controller, actor, item)

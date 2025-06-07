from __future__ import annotations

from catley.game import range_system
from catley.game.actors import Actor, Character
from catley.game.components import HealthComponent
from catley.game.conditions import Injury
from catley.game.items.capabilities import Attack
from catley.game.items.item_core import Item

from .base import ResolutionResult
from .outcomes import CombatOutcome


def determine_outcome(
    resolution_result: ResolutionResult,
    attacker: Character,
    defender: Actor,
    weapon: Item,
) -> CombatOutcome:
    """Translate a resolution result into combat-specific consequences."""
    outcome = CombatOutcome()

    attack_handler = _select_attack(attacker, defender, weapon)
    if attack_handler is None:
        return outcome

    if not resolution_result.success:
        return outcome

    damage = attack_handler.damage_dice.roll()
    if resolution_result.is_critical_success:
        damage += attack_handler.damage_dice.roll()
        health = defender.health
        if isinstance(health, HealthComponent) and health.ap > 0:
            outcome.armor_damage = health.ap
        else:
            outcome.injury_inflicted = Injury()

    outcome.damage_dealt = damage
    return outcome


def _select_attack(attacker: Character, defender: Actor, weapon: Item) -> Attack | None:
    """Choose which attack mode the weapon should use."""
    distance = range_system.calculate_distance(
        attacker.x, attacker.y, defender.x, defender.y
    )

    if distance == 1 and weapon.melee_attack:
        return weapon.melee_attack

    if distance > 1 and weapon.ranged_attack:
        return weapon.ranged_attack

    return weapon.melee_attack or weapon.ranged_attack

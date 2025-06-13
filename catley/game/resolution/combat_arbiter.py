from __future__ import annotations

import random

from catley.game import ranges
from catley.game.actors import Actor, Character, components, conditions
from catley.game.enums import InjuryLocation, OutcomeTier
from catley.game.items.capabilities import Attack
from catley.game.items.item_core import Item

from .base import ResolutionResult
from .outcomes import CombatOutcome

INJURY_LOCATION_WEIGHTS: dict[InjuryLocation, float] = {
    # Body location hit probabilities based on combat targeting
    InjuryLocation.HEAD: 0.15,
    InjuryLocation.LEFT_ARM: 0.15,
    InjuryLocation.RIGHT_ARM: 0.15,
    InjuryLocation.LEFT_LEG: 0.15,
    InjuryLocation.RIGHT_LEG: 0.15,
    InjuryLocation.TORSO: 0.25,
}


def _determine_injury_location() -> InjuryLocation:
    """Randomly select an injury location using ``INJURY_LOCATION_WEIGHTS``."""
    roll = random.random()
    cumulative = 0.0
    for location, weight in INJURY_LOCATION_WEIGHTS.items():
        cumulative += weight
        if roll < cumulative:
            return location
    return InjuryLocation.TORSO


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

    if resolution_result.outcome_tier in (
        OutcomeTier.FAILURE,
        OutcomeTier.CRITICAL_FAILURE,
    ):
        return outcome

    damage = attack_handler.damage_dice.roll()
    if resolution_result.outcome_tier == OutcomeTier.CRITICAL_SUCCESS:
        damage += attack_handler.damage_dice.roll()
        health = defender.health
        if isinstance(health, components.HealthComponent) and health.ap > 0:
            outcome.armor_damage = health.ap
        else:
            outcome.injury_inflicted = conditions.Injury(
                _determine_injury_location(),
                "Combat Wound",
            )

    outcome.damage_dealt = damage
    return outcome


def _select_attack(attacker: Character, defender: Actor, weapon: Item) -> Attack | None:
    """Choose which attack mode the weapon should use."""
    distance = ranges.calculate_distance(attacker.x, attacker.y, defender.x, defender.y)

    if distance == 1 and weapon.melee_attack:
        return weapon.melee_attack

    if distance > 1 and weapon.ranged_attack:
        return weapon.ranged_attack

    return weapon.melee_attack or weapon.ranged_attack

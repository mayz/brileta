from __future__ import annotations

import random

from catley.game import outfit, ranges
from catley.game.actors import Actor, Character, conditions
from catley.game.enums import InjuryLocation, OutcomeTier
from catley.game.items.capabilities import Attack
from catley.game.items.item_core import Item
from catley.game.items.properties import TacticalProperty, WeaponProperty

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
    """Translate a resolution result into combat-specific consequences.

    Uses the outfit system to calculate damage reduction:
    1. Roll base damage
    2. Apply outfit protection (if defender has one equipped)
    3. Track armor penetration for AP loss
    4. Critical hits bypass armor and may inflict injuries
    5. Armor-piercing and radiation weapons bypass armor entirely
    """
    outcome = CombatOutcome()

    attack_handler = _select_attack(attacker, defender, weapon)
    if attack_handler is None:
        return outcome

    if resolution_result.outcome_tier in (
        OutcomeTier.FAILURE,
        OutcomeTier.CRITICAL_FAILURE,
    ):
        return outcome

    # Roll base damage
    raw_damage = attack_handler.damage_dice.roll()
    if resolution_result.outcome_tier == OutcomeTier.CRITICAL_SUCCESS:
        # Critical hit - double damage dice
        raw_damage += attack_handler.damage_dice.roll()

    # Get defender's outfit capability (if any)
    defender_outfit: outfit.OutfitCapability | None = None
    if (
        isinstance(defender, Character)
        and defender.inventory is not None
        and hasattr(defender.inventory, "outfit_capability")
    ):
        defender_outfit = defender.inventory.outfit_capability

    # Check if weapon bypasses armor (armor-piercing or radiation)
    bypasses_armor = False
    if weapon:
        props = weapon.get_weapon_properties()
        if (
            TacticalProperty.RADIATION in props
            or WeaponProperty.ARMOR_PIERCING in props
        ):
            bypasses_armor = True

    # Calculate damage after armor protection
    if resolution_result.outcome_tier == OutcomeTier.CRITICAL_SUCCESS or bypasses_armor:
        # Critical hits and armor-piercing attacks bypass armor entirely
        outcome.damage_dealt = raw_damage
        outcome.armor_penetrated = True

        # Critical hits on armored targets damage the armor's AP
        # Critical hits on unarmored targets inflict injuries
        # Armor-piercing weapons don't damage armor (they just ignore it)
        if resolution_result.outcome_tier == OutcomeTier.CRITICAL_SUCCESS:
            if defender_outfit is not None and defender_outfit.has_protection:
                outfit.apply_hit(defender_outfit, penetrated=True)
            else:
                outcome.injury_inflicted = conditions.Injury(
                    _determine_injury_location(),
                    "Combat Wound",
                )
    else:
        # Normal hit - apply armor protection
        damage_result = outfit.calculate_damage(raw_damage, defender_outfit)
        outcome.damage_dealt = damage_result.damage_to_hp
        outcome.armor_penetrated = damage_result.penetrated

        # Apply AP loss if armor was penetrated
        if defender_outfit is not None and damage_result.penetrated:
            outfit.apply_hit(defender_outfit, penetrated=True)

    return outcome


def _select_attack(attacker: Character, defender: Actor, weapon: Item) -> Attack | None:
    """Choose which attack mode the weapon should use."""
    distance = ranges.calculate_distance(attacker.x, attacker.y, defender.x, defender.y)

    if distance == 1 and weapon.melee_attack:
        return weapon.melee_attack

    if distance > 1 and weapon.ranged_attack:
        return weapon.ranged_attack

    return weapon.melee_attack or weapon.ranged_attack

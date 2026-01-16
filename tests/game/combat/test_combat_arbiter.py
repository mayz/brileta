"""Tests for combat arbiter damage calculation.

These tests verify the combat outcome determination, including:
- Basic hit damage calculation
- Critical hit behavior with and without armor
- Armor protection and AP
- Miss handling
"""

from typing import cast
from unittest.mock import patch

from catley.game.actors import Character
from catley.game.enums import OutcomeTier
from catley.game.game_world import GameWorld
from catley.game.items.item_types import FISTS_TYPE
from catley.game.resolution.combat_arbiter import determine_outcome
from catley.game.resolution.d20_system import D20ResolutionResult
from tests.helpers import DummyGameWorld


def make_characters() -> tuple[Character, Character, GameWorld]:
    gw = cast(GameWorld, DummyGameWorld())
    attacker = Character(1, 1, "A", (255, 255, 255), "Attacker", game_world=gw)
    defender = Character(2, 1, "D", (255, 255, 255), "Defender", game_world=gw)
    gw.add_actor(attacker)
    gw.add_actor(defender)
    return attacker, defender, gw


def test_determine_consequences_basic_hit() -> None:
    """Basic hit deals damage (no armor - full damage to HP)."""
    attacker, defender, _ = make_characters()
    weapon = FISTS_TYPE.create()
    attack = weapon.melee_attack
    assert attack is not None
    with patch.object(attack.damage_dice, "roll", return_value=3):
        result = D20ResolutionResult(outcome_tier=OutcomeTier.SUCCESS)
        outcome = determine_outcome(result, attacker, defender, weapon)
    # No armor - full damage dealt
    assert outcome.damage_dealt == 3
    assert outcome.armor_penetrated is True


def test_determine_consequences_critical_hit_damages_armor() -> None:
    """Critical hit bypasses armor and marks armor as penetrated."""
    from catley.game.outfit import LEATHER_ARMOR_TYPE

    attacker, defender, _ = make_characters()
    weapon = FISTS_TYPE.create()
    attack = weapon.melee_attack
    assert attack is not None

    # Give defender leather armor
    armor_item = LEATHER_ARMOR_TYPE.create()
    defender.inventory.set_starting_outfit(armor_item)

    with patch.object(attack.damage_dice, "roll", side_effect=[2, 2]):
        result = D20ResolutionResult(outcome_tier=OutcomeTier.CRITICAL_SUCCESS)
        outcome = determine_outcome(result, attacker, defender, weapon)

    # Critical bypasses armor - full damage dealt
    assert outcome.damage_dealt == 4
    assert outcome.armor_penetrated is True
    # Critical on armored target does not inflict injury
    assert outcome.injury_inflicted is None
    # Armor AP should be reduced (capability is on the item)
    assert armor_item.outfit_capability is not None
    assert armor_item.outfit_capability.ap == 3  # Started at 4, lost 1


def test_determine_consequences_critical_hit_injures_unarmored() -> None:
    """Critical hit on unarmored target inflicts injury."""
    attacker, defender, _ = make_characters()
    weapon = FISTS_TYPE.create()
    attack = weapon.melee_attack
    assert attack is not None
    # No armor equipped
    with patch.object(attack.damage_dice, "roll", side_effect=[2, 2]):
        result = D20ResolutionResult(outcome_tier=OutcomeTier.CRITICAL_SUCCESS)
        outcome = determine_outcome(result, attacker, defender, weapon)

    assert outcome.damage_dealt == 4
    assert outcome.armor_penetrated is True
    assert outcome.injury_inflicted is not None


def test_determine_consequences_miss_deals_no_damage() -> None:
    """Miss deals no damage."""
    attacker, defender, _ = make_characters()
    weapon = FISTS_TYPE.create()
    attack = weapon.melee_attack
    assert attack is not None
    with patch.object(attack.damage_dice, "roll", return_value=5):
        result = D20ResolutionResult(outcome_tier=OutcomeTier.FAILURE)
        outcome = determine_outcome(result, attacker, defender, weapon)

    assert outcome.damage_dealt == 0
    assert outcome.armor_penetrated is False

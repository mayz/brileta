from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from catley.game.actors import Character
from catley.game.items.item_types import FISTS_TYPE
from catley.game.resolution.combat_arbiter import determine_outcome
from catley.game.resolution.d20_system import D20ResolutionResult
from catley.world.game_state import GameWorld
from catley.world.map import GameMap


@dataclass
class DummyGameWorld:
    def __init__(self) -> None:
        self.game_map = GameMap(5, 5)
        self.actors: list[Character] = []
        self.player: Character | None = None


def make_characters() -> tuple[Character, Character, GameWorld]:
    gw = cast(GameWorld, DummyGameWorld())
    attacker = Character(1, 1, "A", (255, 255, 255), "Attacker", game_world=gw)
    defender = Character(2, 1, "D", (255, 255, 255), "Defender", game_world=gw)
    gw.actors.extend([attacker, defender])
    return attacker, defender, gw


def test_determine_consequences_basic_hit() -> None:
    attacker, defender, _ = make_characters()
    weapon = FISTS_TYPE.create()
    attack = weapon.melee_attack
    assert attack is not None
    with patch.object(attack.damage_dice, "roll", return_value=3):
        result = D20ResolutionResult(success=True)
        outcome = determine_outcome(result, attacker, defender, weapon)
    assert outcome.damage_dealt == 3
    assert outcome.armor_damage == 0


def test_determine_consequences_critical_hit_breaks_armor() -> None:
    attacker, defender, _ = make_characters()
    weapon = FISTS_TYPE.create()
    attack = weapon.melee_attack
    assert attack is not None
    defender.health.ap = 2
    with patch.object(attack.damage_dice, "roll", side_effect=[2, 2]):
        result = D20ResolutionResult(success=True, is_critical_success=True)
        outcome = determine_outcome(result, attacker, defender, weapon)

    assert outcome.damage_dealt == 4
    assert outcome.armor_damage == 2
    assert outcome.injury_inflicted is None


def test_determine_consequences_critical_hit_injures_unarmored() -> None:
    attacker, defender, _ = make_characters()
    weapon = FISTS_TYPE.create()
    attack = weapon.melee_attack
    assert attack is not None
    defender.health.ap = 0
    with patch.object(attack.damage_dice, "roll", side_effect=[2, 2]):
        result = D20ResolutionResult(success=True, is_critical_success=True)
        outcome = determine_outcome(result, attacker, defender, weapon)

    assert outcome.damage_dealt == 4
    assert outcome.armor_damage == 0
    assert outcome.injury_inflicted is not None


def test_determine_consequences_miss_deals_no_damage() -> None:
    attacker, defender, _ = make_characters()
    weapon = FISTS_TYPE.create()
    attack = weapon.melee_attack
    assert attack is not None
    with patch.object(attack.damage_dice, "roll", return_value=5):
        result = D20ResolutionResult(success=False)
        outcome = determine_outcome(result, attacker, defender, weapon)

    assert outcome.damage_dealt == 0
    assert outcome.armor_damage == 0

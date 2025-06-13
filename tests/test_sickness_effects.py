from typing import cast

from catley import colors
from catley.game.actors import Character, conditions
from catley.game.game_world import GameWorld
from tests.helpers import DummyGameWorld


def make_actor() -> Character:
    gw = DummyGameWorld()
    actor = Character(0, 0, "A", colors.WHITE, "Act", game_world=cast(GameWorld, gw))
    gw.add_actor(actor)
    return actor


def test_poisoned_turn_damage_and_disadvantage() -> None:
    actor = make_actor()
    poison = conditions.Sickness(sickness_type="Poisoned")
    actor.inventory.add_to_inventory(poison)
    actor.health.ap = 0
    starting_hp = actor.health.hp
    poison.apply_turn_effect(actor)
    assert actor.health.hp == starting_hp - 1

    args = {"has_advantage": False, "has_disadvantage": False, "stat_name": "toughness"}
    result = poison.apply_to_resolution(args)
    assert result["has_disadvantage"]


def test_venom_damage_and_agility_penalty() -> None:
    actor = make_actor()
    venom = conditions.Sickness(sickness_type="Venom")
    actor.inventory.add_to_inventory(venom)
    actor.health.ap = 0
    starting_hp = actor.health.hp
    venom.apply_turn_effect(actor)
    assert actor.health.hp == starting_hp - 2

    args = {"has_advantage": False, "has_disadvantage": False, "stat_name": "agility"}
    result = venom.apply_to_resolution(args)
    assert result["has_disadvantage"]


def test_disease_physical_disadvantage() -> None:
    actor = make_actor()
    disease = conditions.Sickness(sickness_type="Disease")
    actor.inventory.add_to_inventory(disease)
    for stat in ("strength", "toughness", "agility"):
        args = {"has_advantage": False, "has_disadvantage": False, "stat_name": stat}
        result = disease.apply_to_resolution(args)
        assert result["has_disadvantage"]


def test_radiation_sickness_bypasses_armor() -> None:
    actor = make_actor()
    rad = conditions.Sickness(sickness_type="Radiation Sickness")
    actor.inventory.add_to_inventory(rad)
    actor.health.ap = 5
    starting_hp = actor.health.hp
    starting_ap = actor.health.ap
    rad.apply_turn_effect(actor)
    assert actor.health.hp == starting_hp - 1
    assert actor.health.ap == starting_ap

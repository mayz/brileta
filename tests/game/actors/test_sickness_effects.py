from typing import cast

from brileta import colors
from brileta.game.actors import Character, conditions
from brileta.game.game_world import GameWorld
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
    # No default armor - damage goes directly to HP
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
    # No default armor - damage goes directly to HP
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


def test_radiation_sickness_damages_hp() -> None:
    """Radiation sickness deals damage to HP (armor is now handled by outfit system)."""
    actor = make_actor()
    rad = conditions.Sickness(sickness_type="Radiation Sickness")
    actor.inventory.add_to_inventory(rad)
    starting_hp = actor.health.hp
    rad.apply_turn_effect(actor)
    assert actor.health.hp == starting_hp - 1


def test_radiation_sickness_death_handling() -> None:
    """Radiation sickness properly handles death visuals when it kills an actor.

    Regression test: When radiation sickness damage kills an actor, the death
    handling must run (glyph -> 'x', color -> DEAD). Previously, direct HP
    manipulation bypassed Actor.take_damage() and left corpses with their
    original glyph.
    """
    actor = make_actor()
    rad = conditions.Sickness(sickness_type="Radiation Sickness")
    actor.inventory.add_to_inventory(rad)

    # Set HP to 1 so radiation damage will kill them
    actor.health._hp = 1

    rad.apply_turn_effect(actor)

    # Actor should be dead
    assert not actor.health.is_alive()

    # Death visuals should have been applied (this was the bug)
    assert actor.ch == "x", "Death handling should change glyph to 'x'"
    assert actor.color == colors.DEAD, "Death handling should change color to DEAD"
    assert not actor.blocks_movement, "Dead actors should not block movement"


def test_sickness_expires_after_remaining_turns() -> None:
    """A sickness with remaining_turns removes itself when the timer hits zero."""
    actor = make_actor()
    venom = conditions.Sickness(sickness_type="Venom", remaining_turns=2)
    actor.inventory.add_to_inventory(venom)

    # Turn 1: damage dealt, 1 turn left.
    hp_before = actor.health.hp
    venom.apply_turn_effect(actor)
    assert actor.health.hp == hp_before - 2
    assert venom.remaining_turns == 1
    assert venom in list(actor.conditions)

    # Turn 2: damage dealt, timer expires, condition removed.
    hp_before = actor.health.hp
    venom.apply_turn_effect(actor)
    assert actor.health.hp == hp_before - 2
    assert venom not in list(actor.conditions)


def test_sickness_without_duration_persists_indefinitely() -> None:
    """A sickness with remaining_turns=None never expires on its own."""
    actor = make_actor()
    disease = conditions.Sickness(sickness_type="Disease", remaining_turns=None)
    actor.inventory.add_to_inventory(disease)

    # Tick several turns - should still be present.
    for _ in range(10):
        disease.apply_turn_effect(actor)
    assert disease in list(actor.conditions)

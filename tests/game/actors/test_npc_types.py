"""Tests for NPCType templates and tag-driven action composition."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from brileta import colors
from brileta.game.actions.combat import AttackIntent
from brileta.game.actions.movement import MoveIntent
from brileta.game.actors import NPC, Character
from brileta.game.actors.ai.actions import AttackAction
from brileta.game.actors.ai.behaviors.flee import FleeAction, FleeGoal
from brileta.game.actors.ai.utility import (
    has_escape_route,
    is_any_threat_perceived,
    is_target_nearby,
)
from brileta.game.actors.npc_core import NPCType, StatDistribution, compose_actions
from brileta.game.actors.npc_types import (
    BRIGAND_TYPE,
    DOG_TYPE,
    GIANT_SCORPION_TYPE,
    RESIDENT_TYPE,
    TROG_TYPE,
)
from brileta.game.enums import CreatureSize
from brileta.game.game_world import GameWorld
from brileta.types import DIRECTIONS
from tests.helpers import DummyController, DummyGameWorld


def _actions_by_id(npc: NPC) -> dict[str, object]:
    return {action.action_id: action for action in npc.ai.brain.actions}


@pytest.mark.parametrize(
    ("npc_type", "glyph", "color", "size", "can_open_doors", "default_disposition"),
    [
        (TROG_TYPE, "T", colors.DARK_GREY, CreatureSize.LARGE, True, 0),
        (BRIGAND_TYPE, "B", colors.DARK_GREY, CreatureSize.MEDIUM, True, -75),
        (GIANT_SCORPION_TYPE, "s", colors.TAN, CreatureSize.LARGE, False, -40),
        (DOG_TYPE, "d", colors.BROWN, CreatureSize.SMALL, False, 20),
        (RESIDENT_TYPE, "R", colors.LIGHT_GREY, CreatureSize.MEDIUM, True, 0),
    ],
)
def test_npc_type_create_applies_core_attributes(
    npc_type: NPCType,
    glyph: str,
    color: tuple[int, int, int],
    size: CreatureSize,
    can_open_doors: bool,
    default_disposition: int,
) -> None:
    gw = DummyGameWorld(width=20, height=20)
    npc = npc_type.create(5, 5, "Test NPC", game_world=cast(GameWorld, gw))
    other = Character(0, 0, "@", colors.WHITE, "Other")

    assert npc.ch == glyph
    assert npc.color == color
    assert npc.can_open_doors is can_open_doors
    assert npc.visual_effects is not None
    assert npc.shadow_height == size.shadow_height
    assert npc.ai.disposition_toward(other) == default_disposition


def test_brigand_action_pool_contains_combat_and_sapient_actions() -> None:
    npc = BRIGAND_TYPE.create(0, 0, "Brigand")
    action_ids = set(_actions_by_id(npc))

    assert action_ids == {
        "attack",
        "flee",
        "watch",
        "avoid",
        "idle",
        "wander",
    }


def test_trog_action_pool_is_base_plus_combatant_only() -> None:
    npc = TROG_TYPE.create(0, 0, "Trog")
    action_ids = set(_actions_by_id(npc))

    assert action_ids == {"attack", "flee", "idle", "wander"}
    assert "watch" not in action_ids
    assert "avoid" not in action_ids


def test_scorpion_predator_tag_overrides_attack_with_proximity_logic() -> None:
    npc = GIANT_SCORPION_TYPE.create(0, 0, "Scorpion")
    actions = _actions_by_id(npc)

    assert set(actions) == {"attack", "flee", "idle", "wander"}

    attack = actions["attack"]
    assert isinstance(attack, AttackAction)
    assert attack.preconditions == [is_target_nearby]
    assert [c.input_key for c in attack.considerations] == [
        "target_proximity",
        "health_percent",
    ]


def test_dog_is_skittish_and_never_attacks() -> None:
    npc = DOG_TYPE.create(0, 0, "Dog")
    actions = _actions_by_id(npc)

    assert set(actions) == {"flee", "idle", "wander"}
    assert "attack" not in actions

    flee = actions["flee"]
    assert isinstance(flee, FleeAction)
    assert flee.preconditions == [is_target_nearby, has_escape_route]
    assert [c.input_key for c in flee.considerations] == ["target_proximity"]


def test_dog_flees_from_nearby_actor_even_when_friendly() -> None:
    """Skittish dogs should flee on proximity, not hostility."""
    gw = DummyGameWorld(width=40, height=40)
    player = Character(
        5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    dog = DOG_TYPE.create(6, 5, "Dog", game_world=cast(GameWorld, gw))
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(dog)
    controller = DummyController(gw)

    # Dog defaults to friendly (+20), so this regression catches missing
    # proximity fallback when no hostile relationship exists.
    assert dog.ai.disposition_toward(player) > 0
    action = dog.ai.get_action(controller, dog)

    assert isinstance(action, MoveIntent)
    assert isinstance(dog.current_goal, FleeGoal)


def test_neutral_predator_attacks_on_proximity() -> None:
    """Predator tag should work even when default disposition is non-hostile."""
    neutral_predator = NPCType(
        id="neutral_predator",
        tags=("base", "combatant", "predator"),
        glyph="n",
        color=colors.TAN,
        creature_size=CreatureSize.MEDIUM,
        default_disposition=20,
        can_open_doors=False,
    )
    gw = DummyGameWorld(width=20, height=20)
    player = Character(
        5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    predator = neutral_predator.create(6, 5, "Predator", game_world=cast(GameWorld, gw))
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(predator)
    controller = DummyController(gw)

    action = predator.ai.get_action(controller, predator)

    assert isinstance(action, AttackIntent)


def test_resident_action_pool_is_combatant_sapient_profile() -> None:
    """Residents can fight, flee, and use social awareness actions."""
    npc = RESIDENT_TYPE.create(0, 0, "Resident")
    action_ids = set(_actions_by_id(npc))

    assert action_ids == {
        "attack",
        "flee",
        "watch",
        "avoid",
        "idle",
        "wander",
    }

    flee = _actions_by_id(npc)["flee"]
    assert isinstance(flee, FleeAction)
    assert flee.preconditions == [is_any_threat_perceived, has_escape_route]
    assert [c.input_key for c in flee.considerations] == ["sapient_flee_urgency"]


def test_brigand_adjacent_at_full_health_attacks_instead_of_fleeing() -> None:
    """Hostile sapient combatants should not flee when uninjured."""
    gw = DummyGameWorld(width=40, height=40)
    player = Character(
        5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    brigand = BRIGAND_TYPE.create(6, 5, "Brigand", game_world=cast(GameWorld, gw))
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(brigand)
    controller = DummyController(gw)

    action = brigand.ai.get_action(controller, brigand)
    assert isinstance(action, AttackIntent)


def test_resident_flees_when_hostile_and_hurt() -> None:
    """A hurt resident who's been made hostile flees rather than fighting."""
    gw = DummyGameWorld(width=40, height=40)
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    resident = RESIDENT_TYPE.create(3, 0, "Resident", game_world=cast(GameWorld, gw))
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(resident)

    resident.ai.set_hostile(player)
    # Damage the resident so flee outscores attack (health_percent consideration).
    resident.health._apply_damage(resident.health.max_hp - 1)
    controller = DummyController(gw)

    action = resident.ai.get_action(controller, resident)

    assert isinstance(action, MoveIntent)
    assert isinstance(resident.current_goal, FleeGoal)


def test_resident_sapient_flee_triggers_from_incoming_threat_only() -> None:
    """Sapient flee should trigger with incoming_threat even at threat_level 0."""
    gw = DummyGameWorld(width=40, height=40)
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    resident = RESIDENT_TYPE.create(5, 5, "Resident", game_world=cast(GameWorld, gw))
    attacker = NPC(
        6,
        5,
        "s",
        colors.TAN,
        "Attacker",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(resident)
    gw.add_actor(attacker)
    attacker.ai.set_hostile(resident)
    controller = DummyController(gw)

    context = resident.ai._build_context(controller, resident)
    assert context.threat_level == 0.0
    assert context.incoming_threat > 0.0

    action = resident.ai.get_action(controller, resident)
    assert isinstance(action, MoveIntent)
    assert isinstance(resident.current_goal, FleeGoal)


def test_resident_sapient_flee_goal_persists_on_second_tick() -> None:
    """Sapient flee continuation should use sapient_flee_urgency and persist."""
    gw = DummyGameWorld(width=40, height=40)
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    resident = RESIDENT_TYPE.create(5, 5, "Resident", game_world=cast(GameWorld, gw))
    attacker = NPC(
        6,
        5,
        "s",
        colors.TAN,
        "Attacker",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(resident)
    gw.add_actor(attacker)
    attacker.ai.set_hostile(resident)
    controller = DummyController(gw)

    action1 = resident.ai.get_action(controller, resident)
    assert isinstance(action1, MoveIntent)
    assert isinstance(resident.current_goal, FleeGoal)
    flee_goal = resident.current_goal

    resident.move(action1.dx, action1.dy)

    action2 = resident.ai.get_action(controller, resident)
    assert isinstance(action2, MoveIntent)
    assert resident.current_goal is flee_goal


def test_resident_attacks_when_cornered() -> None:
    """A cornered resident fights back instead of cowering."""
    gw = DummyGameWorld(width=20, height=20)
    player = Character(
        5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    resident = RESIDENT_TYPE.create(6, 5, "Resident", game_world=cast(GameWorld, gw))
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(resident)
    resident.ai.set_hostile(player)

    # Trap the resident so flee is impossible.
    for dx, dy in DIRECTIONS:
        if dx == 0 and dy == 0:
            continue
        tx, ty = resident.x + dx, resident.y + dy
        if (tx, ty) != (player.x, player.y):
            gw.game_map.walkable[tx, ty] = False

    controller = DummyController(gw)
    resident.ai.get_action(controller, resident)

    assert resident.ai.last_chosen_action == "Attack"


def test_npc_type_stat_sampling_produces_variation_across_spawns() -> None:
    varied_type = NPCType(
        id="varied",
        tags=("base",),
        glyph="v",
        color=colors.WHITE,
        creature_size=CreatureSize.MEDIUM,
        strength_dist=StatDistribution(mean=0, std_dev=2.5),
    )

    sampled_strengths = {
        varied_type.create(0, 0, f"Varied {i}").stats.strength for i in range(30)
    }
    assert len(sampled_strengths) > 1


def test_compose_actions_allows_later_tags_to_override_action_id() -> None:
    actions = compose_actions(("combatant", "predator"))
    action_by_id = {action.action_id: action for action in actions}

    attack = action_by_id["attack"]
    assert isinstance(attack, AttackAction)
    assert attack.preconditions == [is_target_nearby]


def test_compose_actions_raises_for_unknown_tag() -> None:
    with pytest.raises(ValueError, match="Unknown NPC tag"):
        compose_actions(("nonexistent",))


@pytest.mark.parametrize(
    ("gauss_value", "expected"),
    [
        (-999.0, -2),
        (999.0, 2),
    ],
)
def test_stat_distribution_sample_clamps_to_bounds(
    gauss_value: float, expected: int
) -> None:
    dist = StatDistribution(mean=0, std_dev=1.0, min_val=-2, max_val=2)
    random_stream = MagicMock()
    random_stream.gauss.return_value = gauss_value

    assert dist.sample(random_stream) == expected


def test_default_disposition_flows_to_unknown_relationships() -> None:
    npc = BRIGAND_TYPE.create(1, 1, "Brigand")
    stranger = Character(2, 2, "@", colors.WHITE, "Stranger")

    assert npc.ai.disposition_toward(stranger) == -75


# --- Natural weapon tests ---


def test_scorpion_has_venomous_stinger() -> None:
    """Giant Scorpion spawns with a stinger that uses 'sting' verb and is venomous."""
    from brileta.game.items.properties import StatusProperty, WeaponProperty

    npc = GIANT_SCORPION_TYPE.create(0, 0, "Scorpion")
    weapon = npc.inventory.get_active_item()
    assert weapon is not None, "Scorpion should spawn with an equipped weapon"

    melee = weapon.melee_attack
    assert melee is not None
    assert melee._spec.verb == "sting"
    assert StatusProperty.POISONING in melee.properties
    assert WeaponProperty.UNARMED in melee.properties
    assert weapon.can_materialize is False


def test_dog_has_bite_attack() -> None:
    """Dog spawns with jaws that use 'bite' verb."""
    from brileta.game.items.properties import WeaponProperty

    npc = DOG_TYPE.create(0, 0, "Dog")
    weapon = npc.inventory.get_active_item()
    assert weapon is not None, "Dog should spawn with an equipped weapon"

    melee = weapon.melee_attack
    assert melee is not None
    assert melee._spec.verb == "bite"
    assert WeaponProperty.UNARMED in melee.properties
    assert weapon.can_materialize is False

from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from catley.controller import Controller
from catley.environment import tile_types
from catley.game.actions.combat import AttackAction
from catley.game.actors import Character
from catley.game.enums import OutcomeTier
from catley.game.game_world import GameWorld
from catley.game.items.item_types import FISTS_TYPE
from catley.game.resolution.d20_system import D20ResolutionResult
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    gw: DummyGameWorld
    frame_manager: object | None = None
    message_log: object | None = None

    def create_resolver(self, **kwargs: object) -> object:
        from catley.game.resolution.d20_system import D20Resolver

        return D20Resolver(**kwargs)  # type: ignore[call-arg]


def test_cover_bonus_reduces_hit_chance() -> None:
    raw_gw = DummyGameWorld()
    gw = cast(GameWorld, raw_gw)
    gm = raw_gw.game_map
    gm.tiles[:] = tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
    gm.tiles[1, 2] = tile_types.TILE_TYPE_ID_BOULDER  # type: ignore[attr-defined]

    attacker = Character(1, 0, "A", (255, 255, 255), "Att", game_world=gw, strength=5)
    defender = Character(1, 1, "D", (255, 255, 255), "Def", game_world=gw)
    gw.add_actor(attacker)
    gw.add_actor(defender)

    weapon = FISTS_TYPE.create()
    attack = weapon.melee_attack
    assert attack is not None

    controller = DummyController(gw=raw_gw)
    action = AttackAction(cast(Controller, controller), attacker, defender, weapon)

    with patch("random.randint", return_value=7):
        result = cast(
            D20ResolutionResult,
            action._execute_attack_roll(attack, weapon, {}),
        )
    assert result.outcome_tier in (
        OutcomeTier.FAILURE,
        OutcomeTier.PARTIAL_SUCCESS,
    )

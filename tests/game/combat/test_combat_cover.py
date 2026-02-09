from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from brileta.controller import Controller
from brileta.environment.tile_types import TileTypeID
from brileta.game.actions.combat import AttackIntent
from brileta.game.actions.executors.combat import AttackExecutor
from brileta.game.actors import Character
from brileta.game.enums import OutcomeTier
from brileta.game.game_world import GameWorld
from brileta.game.items.item_types import FISTS_TYPE
from brileta.game.resolution import d20_system
from brileta.game.resolution.d20_system import D20ResolutionResult
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    gw: DummyGameWorld
    frame_manager: object | None = None
    message_log: object | None = None

    def create_resolver(self, **kwargs: object) -> object:
        from brileta.game.resolution.d20_system import D20System

        return D20System(**kwargs)  # type: ignore[call-arg]


def test_cover_bonus_reduces_hit_chance() -> None:
    raw_gw = DummyGameWorld()
    gw = cast(GameWorld, raw_gw)
    gm = raw_gw.game_map
    gm.tiles[:] = TileTypeID.FLOOR
    gm.tiles[1, 2] = TileTypeID.BOULDER

    attacker = Character(1, 0, "A", (255, 255, 255), "Att", game_world=gw, strength=5)
    defender = Character(1, 1, "D", (255, 255, 255), "Def", game_world=gw)
    gw.add_actor(attacker)
    gw.add_actor(defender)

    weapon = FISTS_TYPE.create()
    attack = weapon.melee_attack
    assert attack is not None

    controller = DummyController(gw=raw_gw)
    intent = AttackIntent(cast(Controller, controller), attacker, defender, weapon)
    executor = AttackExecutor()

    with patch.object(d20_system._rng, "randint", return_value=7):
        result = cast(
            D20ResolutionResult,
            executor._execute_attack_roll(intent, attack, weapon, {}),
        )
    assert result.outcome_tier in (
        OutcomeTier.FAILURE,
        OutcomeTier.PARTIAL_SUCCESS,
    )

from typing import cast

from catley import colors
from catley.game.actors import NPC
from catley.game.actors.barks import (
    BARKS_BY_DISPOSITION,
    BARKS_BY_STATE,
    pick_bump_bark,
)
from catley.game.game_world import GameWorld
from tests.helpers import DummyGameWorld


def test_pick_bump_bark_uses_disposition() -> None:
    gw = DummyGameWorld()
    npc = NPC(
        0,
        0,
        "N",
        colors.WHITE,
        "NPC",
        game_world=cast(GameWorld, gw),
    )
    gw.add_actor(npc)

    bark = pick_bump_bark(npc)
    assert bark in BARKS_BY_DISPOSITION[npc.ai.disposition]


def test_pick_bump_bark_prefers_wounded_state() -> None:
    gw = DummyGameWorld()
    npc = NPC(
        0,
        0,
        "N",
        colors.WHITE,
        "NPC",
        game_world=cast(GameWorld, gw),
    )
    gw.add_actor(npc)
    npc.health._hp = 1

    bark = pick_bump_bark(npc)
    assert bark in BARKS_BY_STATE["wounded"]

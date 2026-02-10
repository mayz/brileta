from typing import cast

from brileta import colors
from brileta.game.actors import NPC, Character
from brileta.game.actors.ai import AIComponent, disposition_label
from brileta.game.actors.barks import (
    BARKS_BY_DISPOSITION,
    BARKS_BY_STATE,
    pick_bump_bark,
)
from brileta.game.game_world import GameWorld
from tests.helpers import DummyGameWorld


def test_pick_bump_bark_uses_disposition() -> None:
    gw = DummyGameWorld()
    player = Character(
        1,
        0,
        "@",
        colors.WHITE,
        "Player",
        game_world=cast(GameWorld, gw),
    )
    npc = NPC(
        0,
        0,
        "N",
        colors.WHITE,
        "NPC",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)

    bark = pick_bump_bark(npc, player)
    assert isinstance(npc.ai, AIComponent)
    assert (
        bark
        in BARKS_BY_DISPOSITION[disposition_label(npc.ai.disposition_toward(player))]
    )


def test_pick_bump_bark_prefers_wounded_state() -> None:
    gw = DummyGameWorld()
    player = Character(
        1,
        0,
        "@",
        colors.WHITE,
        "Player",
        game_world=cast(GameWorld, gw),
    )
    npc = NPC(
        0,
        0,
        "N",
        colors.WHITE,
        "NPC",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    npc.health._hp = 1

    bark = pick_bump_bark(npc, player)
    assert bark in BARKS_BY_STATE["wounded"]

from typing import cast
from unittest.mock import patch

from brileta import colors
from brileta.game.actors import NPC, Character
from brileta.game.actors.ai import AIComponent, disposition_label
from brileta.game.actors.barks import (
    BARKS_BY_DISPOSITION,
    BARKS_BY_STATE,
    SHOVE_BARKS,
    emit_bark,
    pick_bump_bark,
    pick_shove_bark,
)
from brileta.game.game_world import GameWorld
from tests.helpers import DummyGameWorld


def _make_npc(gw: DummyGameWorld) -> NPC:
    npc = NPC(0, 0, "N", colors.WHITE, "NPC", game_world=cast(GameWorld, gw))
    gw.add_actor(npc)
    return npc


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


def test_pick_shove_bark_uses_shove_pool() -> None:
    npc = _make_npc(DummyGameWorld())
    bark = pick_shove_bark(npc)
    assert bark in SHOVE_BARKS


def test_pick_shove_bark_prefers_wounded_state() -> None:
    npc = _make_npc(DummyGameWorld())
    npc.health._hp = 1
    bark = pick_shove_bark(npc)
    assert bark in BARKS_BY_STATE["wounded"]


def test_critters_have_no_bump_or_shove_bark() -> None:
    # Dogs and other critters are NPCs but have no human dialogue; bumping or
    # shoving them must stay silent.
    from brileta.sprites.quadrupeds import DOG_PRESET

    gw = DummyGameWorld()
    player = Character(
        1, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    npc = _make_npc(gw)
    npc.critter_preset = DOG_PRESET
    gw.player = player
    gw.add_actor(player)

    assert pick_bump_bark(npc, player) is None
    assert pick_shove_bark(npc) is None


def test_emit_bark_throttles_repeat_calls() -> None:
    npc = _make_npc(DummyGameWorld())

    with (
        patch("brileta.game.actors.barks.publish_event") as mock_publish,
        patch("brileta.game.actors.barks.time.perf_counter", side_effect=[1.0, 1.2]),
    ):
        assert emit_bark(npc, "Hey!") is True
        # Second call within the block window is silently dropped.
        assert emit_bark(npc, "Hey!") is False

    assert mock_publish.call_count == 1
    event = mock_publish.call_args[0][0]
    assert event.text == "Hey!"
    assert event.bubble is True

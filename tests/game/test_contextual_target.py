"""Tests for controller contextual target tracking."""

from __future__ import annotations

from typing import cast

from catley import colors
from catley.controller import Controller
from catley.game.actors import Character
from catley.game.game_world import GameWorld
from tests.helpers import get_controller_with_player_and_map


def _reset_world(controller: Controller) -> None:
    gw = controller.gw
    player = gw.player
    for actor in list(gw.actors):
        if actor is not player:
            gw.remove_actor(actor)

    gw.actor_spatial_index.remove(player)
    player.x = 5
    player.y = 5
    gw.actor_spatial_index.add(player)
    gw.game_map.visible[:] = True


def test_contextual_target_follows_movement_direction() -> None:
    controller = get_controller_with_player_and_map()
    _reset_world(controller)
    gw = controller.gw

    npc = Character(6, 5, "N", colors.WHITE, "Neighbor", game_world=cast(GameWorld, gw))
    gw.add_actor(npc)

    controller.update_contextual_target_from_movement(1, 0)

    assert controller.contextual_target is npc


def test_contextual_target_uses_deterministic_fallback() -> None:
    controller = get_controller_with_player_and_map()
    _reset_world(controller)
    gw = controller.gw

    north = Character(5, 4, "N", colors.WHITE, "North", game_world=cast(GameWorld, gw))
    west = Character(4, 5, "W", colors.WHITE, "West", game_world=cast(GameWorld, gw))
    gw.add_actor(north)
    gw.add_actor(west)

    controller.update_contextual_target_from_movement(1, 0)

    assert controller.contextual_target is north


def test_contextual_target_hover_override_reverts() -> None:
    controller = get_controller_with_player_and_map()
    _reset_world(controller)
    gw = controller.gw

    adjacent = Character(
        6, 5, "A", colors.WHITE, "Adjacent", game_world=cast(GameWorld, gw)
    )
    hovered = Character(
        8, 5, "H", colors.WHITE, "Hovered", game_world=cast(GameWorld, gw)
    )
    gw.add_actor(adjacent)
    gw.add_actor(hovered)

    controller.update_contextual_target_from_movement(1, 0)
    assert controller.contextual_target is adjacent

    controller.update_contextual_target_from_hover((hovered.x, hovered.y))
    assert controller.contextual_target is hovered

    controller.update_contextual_target_from_hover(None)
    assert controller.contextual_target is adjacent

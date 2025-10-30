from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.game import ranges
from catley.game.actions.combat import AttackIntent
from catley.game.actors import NPC, Character
from catley.game.enums import Disposition
from catley.game.game_world import GameWorld
from catley.game.turn_manager import TurnManager
from tests.helpers import DummyGameWorld


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld

    def __post_init__(self) -> None:
        self.turn_manager = TurnManager(self)
        self.frame_manager = None
        self.message_log = None
        self.action_cost = 100


def make_world() -> tuple[DummyController, Character, NPC]:
    gw = DummyGameWorld()
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    npc = NPC(
        3,
        0,
        "g",
        colors.RED,
        "Enemy",
        game_world=cast(GameWorld, gw),
        disposition=Disposition.HOSTILE,
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    controller = DummyController(gw)
    return controller, player, npc


def test_hostile_ai_sets_pathfinding_goal() -> None:
    controller, player, npc = make_world()
    action = npc.ai.get_action(controller, npc)
    assert action is None
    goal = npc.pathfinding_goal
    assert goal is not None
    tx, ty = goal.target_pos
    assert ranges.calculate_distance(player.x, player.y, tx, ty) == 1


def test_hostile_ai_attacks_when_adjacent() -> None:
    controller, _player, npc = make_world()
    npc.x = 1
    npc.y = 0
    action = npc.ai.get_action(controller, npc)
    assert isinstance(action, AttackIntent)
    assert npc.pathfinding_goal is None

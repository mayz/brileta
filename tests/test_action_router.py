from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from catley import colors
from catley.controller import Controller
from catley.environment import tile_types
from catley.game.action_router import ActionRouter
from catley.game.actions.base import GameActionResult
from catley.game.actions.combat import AttackIntent
from catley.game.actions.movement import MoveIntent
from catley.game.actors import Character
from catley.game.game_world import GameWorld
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    gw: DummyGameWorld
    frame_manager: object | None = None

    def __post_init__(self) -> None:
        self.update_fov_called = False
        self.message_log = type("Log", (), {"add_message": lambda *a, **kw: None})()

    def update_fov(self) -> None:
        self.update_fov_called = True


def _make_world() -> tuple[DummyController, Character]:
    gw = DummyGameWorld()
    player = Character(
        0,
        0,
        "@",
        colors.WHITE,
        "Player",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)
    controller = DummyController(gw=gw)
    return controller, player


def test_router_opens_door_on_bump() -> None:
    controller, player = _make_world()
    gm = controller.gw.game_map
    gm.tiles[1, 0] = tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]

    router = ActionRouter(cast(Controller, controller))
    intent = MoveIntent(cast(Controller, controller), player, 1, 0)
    router.execute_intent(intent)

    assert gm.tiles[1, 0] == tile_types.TILE_TYPE_ID_DOOR_OPEN  # type: ignore[attr-defined]
    assert controller.update_fov_called


def test_router_attacks_actor_on_bump() -> None:
    controller, player = _make_world()
    enemy = Character(
        1,
        0,
        "E",
        colors.WHITE,
        "Enemy",
        game_world=cast(GameWorld, controller.gw),
    )
    controller.gw.add_actor(enemy)

    with patch(
        "catley.game.actions.executors.combat.AttackExecutor.execute",
        return_value=GameActionResult(),
    ) as mock_execute:
        router = ActionRouter(cast(Controller, controller))
        intent = MoveIntent(cast(Controller, controller), player, 1, 0)
        router.execute_intent(intent)
        assert mock_execute.called
        assert (player.x, player.y) == (0, 0)
        assert not controller.update_fov_called


def test_player_bumping_npc_triggers_attack() -> None:
    controller, player = _make_world()
    npc = Character(
        1,
        0,
        "N",
        colors.WHITE,
        "NPC",
        game_world=cast(GameWorld, controller.gw),
    )
    controller.gw.add_actor(npc)

    with patch(
        "catley.game.actions.executors.combat.AttackExecutor.execute",
        return_value=GameActionResult(),
    ) as mock_execute:
        router = ActionRouter(cast(Controller, controller))
        intent = MoveIntent(cast(Controller, controller), player, 1, 0)
        router.execute_intent(intent)

        assert mock_execute.called
        call_args = mock_execute.call_args[0][0]
        assert isinstance(call_args, AttackIntent)
        assert call_args.attacker is player
        assert call_args.defender is npc


def test_npc_bumping_player_triggers_attack() -> None:
    controller, player = _make_world()
    npc = Character(
        0,
        1,
        "N",
        colors.WHITE,
        "NPC",
        game_world=cast(GameWorld, controller.gw),
    )
    controller.gw.add_actor(npc)
    # Player stays at (0, 0), NPC starts at (0, 1) and tries to move to (0, 0)

    with patch(
        "catley.game.actions.executors.combat.AttackExecutor.execute",
        return_value=GameActionResult(),
    ) as mock_execute:
        router = ActionRouter(cast(Controller, controller))
        # NPC tries to move to (0, 0) where player is located
        # dx=0, dy=-1 to move from (0, 1) to (0, 0)
        intent = MoveIntent(cast(Controller, controller), npc, 0, -1)
        router.execute_intent(intent)

        assert mock_execute.called
        call_args = mock_execute.call_args[0][0]
        assert isinstance(call_args, AttackIntent)
        assert call_args.attacker is npc
        assert call_args.defender is player


def test_npc_bumping_npc_does_nothing() -> None:
    controller, player = _make_world()
    npc1 = Character(
        1,
        0,
        "N1",
        colors.WHITE,
        "NPC1",
        game_world=cast(GameWorld, controller.gw),
    )
    npc2 = Character(
        2,
        0,
        "N2",
        colors.WHITE,
        "NPC2",
        game_world=cast(GameWorld, controller.gw),
    )
    controller.gw.add_actor(npc1)
    controller.gw.add_actor(npc2)

    with patch(
        "catley.game.actions.executors.combat.AttackExecutor.execute",
        return_value=GameActionResult(),
    ) as mock_execute:
        router = ActionRouter(cast(Controller, controller))
        intent = MoveIntent(cast(Controller, controller), npc1, 1, 0)
        router.execute_intent(intent)

        # Now this assert correctly tests that a failed move into an allied
        # NPC does NOT result in an attack.
        assert not mock_execute.called


def test_actor_bumping_dead_actor_does_nothing() -> None:
    controller, player = _make_world()
    dead_npc = Character(
        1,
        0,
        "D",
        colors.WHITE,
        "Dead NPC",
        game_world=cast(GameWorld, controller.gw),
    )
    dead_npc.health.hp = 0
    controller.gw.add_actor(dead_npc)

    with patch(
        "catley.game.actions.executors.combat.AttackExecutor.execute",
        return_value=GameActionResult(),
    ) as mock_execute:
        router = ActionRouter(cast(Controller, controller))
        intent = MoveIntent(cast(Controller, controller), player, 1, 0)
        router.execute_intent(intent)

        assert not mock_execute.called

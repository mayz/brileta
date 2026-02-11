from dataclasses import dataclass
from typing import cast
from unittest.mock import MagicMock, patch

from brileta import colors
from brileta.controller import Controller
from brileta.environment.tile_types import TileTypeID
from brileta.events import FloatingTextEvent
from brileta.game.action_router import ActionRouter
from brileta.game.actions.base import GameActionResult
from brileta.game.actions.environment import SearchContainerIntent
from brileta.game.actions.movement import MoveIntent
from brileta.game.actors import NPC, Character
from brileta.game.actors.container import create_bookcase
from brileta.game.game_world import GameWorld
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
    gm.tiles[1, 0] = TileTypeID.DOOR_CLOSED

    router = ActionRouter(cast(Controller, controller))
    intent = MoveIntent(cast(Controller, controller), player, 1, 0)
    router.execute_intent(intent)

    assert gm.tiles[1, 0] == TileTypeID.DOOR_OPEN
    assert controller.update_fov_called


def test_player_bumping_npc_does_not_attack() -> None:
    """Player bumping into an NPC stays adjacent without attacking.

    This shifts the game away from combat-by-default. The player sees
    options in the ActionPanel and chooses deliberately.
    """
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
        "brileta.game.actions.executors.combat.AttackExecutor.execute",
        return_value=GameActionResult(),
    ) as mock_execute:
        router = ActionRouter(cast(Controller, controller))
        intent = MoveIntent(cast(Controller, controller), player, 1, 0)
        router.execute_intent(intent)

        # Player should NOT attack - they stay adjacent and can choose an action
        assert not mock_execute.called
        # Player remains at original position (blocked by NPC)
        assert (player.x, player.y) == (0, 0)
        assert not controller.update_fov_called


def test_player_bump_npc_emits_bark() -> None:
    controller, player = _make_world()
    npc = NPC(
        1,
        0,
        "N",
        colors.WHITE,
        "NPC",
        game_world=cast(GameWorld, controller.gw),
    )
    controller.gw.add_actor(npc)

    with (
        patch("brileta.game.action_router.pick_bump_bark", return_value="Hey"),
        patch("brileta.game.action_router.publish_event") as mock_publish,
        patch("brileta.game.action_router.time.perf_counter", return_value=1.0),
    ):
        router = ActionRouter(cast(Controller, controller))
        intent = MoveIntent(cast(Controller, controller), player, 1, 0)
        router.execute_intent(intent)

        assert mock_publish.called
        event = mock_publish.call_args[0][0]
        assert isinstance(event, FloatingTextEvent)
        assert event.text == "Hey"
        assert event.bubble is True


def test_bark_respects_cooldown() -> None:
    controller, player = _make_world()
    npc = NPC(
        1,
        0,
        "N",
        colors.WHITE,
        "NPC",
        game_world=cast(GameWorld, controller.gw),
    )
    controller.gw.add_actor(npc)

    with (
        patch("brileta.game.action_router.pick_bump_bark", return_value="Hey"),
        patch("brileta.game.action_router.publish_event") as mock_publish,
        patch("brileta.game.action_router.time.perf_counter", side_effect=[1.0, 1.2]),
    ):
        router = ActionRouter(cast(Controller, controller))
        intent = MoveIntent(cast(Controller, controller), player, 1, 0)
        router.execute_intent(intent)
        router.execute_intent(intent)

        assert mock_publish.call_count == 1


def test_npc_bumping_player_does_nothing() -> None:
    controller, _player = _make_world()
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
        "brileta.game.actions.executors.combat.AttackExecutor.execute",
        return_value=GameActionResult(),
    ) as mock_execute:
        router = ActionRouter(cast(Controller, controller))
        # NPC tries to move to (0, 0) where player is located
        # dx=0, dy=-1 to move from (0, 1) to (0, 0)
        intent = MoveIntent(cast(Controller, controller), npc, 0, -1)
        router.execute_intent(intent)

        assert not mock_execute.called


def test_npc_bumping_npc_does_nothing() -> None:
    controller, _player = _make_world()
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
        "brileta.game.actions.executors.combat.AttackExecutor.execute",
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
    dead_npc.health._hp = 0
    controller.gw.add_actor(dead_npc)

    with patch(
        "brileta.game.actions.executors.combat.AttackExecutor.execute",
        return_value=GameActionResult(),
    ) as mock_execute:
        router = ActionRouter(cast(Controller, controller))
        intent = MoveIntent(cast(Controller, controller), player, 1, 0)
        router.execute_intent(intent)

        assert not mock_execute.called


def test_router_searches_container_on_bump() -> None:
    """Test that bumping into a blocking container triggers search."""
    controller, player = _make_world()

    # Create a blocking container (crate blocks_movement=True by default)
    crate = create_bookcase(x=1, y=0, game_world=cast(GameWorld, controller.gw))
    controller.gw.add_actor(crate)

    with patch(
        "brileta.game.actions.executors.containers.SearchContainerExecutor.execute",
        return_value=GameActionResult(),
    ) as mock_execute:
        router = ActionRouter(cast(Controller, controller))
        intent = MoveIntent(cast(Controller, controller), player, 1, 0)
        router.execute_intent(intent)

        assert mock_execute.called
        call_args = mock_execute.call_args[0][0]
        assert isinstance(call_args, SearchContainerIntent)
        assert call_args.actor is player
        assert call_args.target is crate
        # Player should not have moved
        assert (player.x, player.y) == (0, 0)


def test_bump_into_container_opens_search_menu() -> None:
    """Integration test: bumping container calls show_overlay (not show)."""
    controller, player = _make_world()

    # Create a blocking container with items
    crate = create_bookcase(x=1, y=0, game_world=cast(GameWorld, controller.gw))
    # Add an item so the search doesn't fail with "Nothing to loot"
    from brileta.game.items.item_types import ROCK_TYPE

    crate.inventory.add_item(ROCK_TYPE.create())
    controller.gw.add_actor(crate)

    # Mock the overlay_system.show_overlay to verify it's called
    mock_overlay = MagicMock()
    controller.overlay_system = mock_overlay  # type: ignore[unresolved-attribute]

    # Patch DualPaneMenu so it doesn't need full controller setup
    with patch("brileta.view.ui.inventory.DualPaneMenu") as mock_menu_class:
        mock_menu_instance = MagicMock()
        mock_menu_class.return_value = mock_menu_instance

        router = ActionRouter(cast(Controller, controller))
        intent = MoveIntent(cast(Controller, controller), player, 1, 0)
        router.execute_intent(intent)

        # Verify DualPaneMenu was created
        assert mock_menu_class.called
        # Verify show_overlay was called with the menu (not show)
        mock_overlay.show_overlay.assert_called_once_with(mock_menu_instance)


def test_execute_intent_returns_game_action_result() -> None:
    """ActionRouter.execute_intent returns a GameActionResult.

    This is required for the ActionPlan system to detect step success/failure
    and decide whether to advance, retry, or cancel.
    """
    controller, player = _make_world()
    router = ActionRouter(cast(Controller, controller))

    # Successful move should return result with succeeded=True
    intent = MoveIntent(cast(Controller, controller), player, 1, 0)
    result = router.execute_intent(intent)

    assert isinstance(result, GameActionResult)
    assert result.succeeded is True


def test_execute_intent_returns_failure_for_blocked_move() -> None:
    """ActionRouter.execute_intent returns succeeded=False for blocked moves."""
    controller, player = _make_world()
    gm = controller.gw.game_map
    gm.tiles[1, 0] = TileTypeID.WALL

    router = ActionRouter(cast(Controller, controller))
    intent = MoveIntent(cast(Controller, controller), player, 1, 0)
    result = router.execute_intent(intent)

    assert isinstance(result, GameActionResult)
    assert result.succeeded is False
    assert result.block_reason == "wall"


def test_execute_intent_returns_failure_for_unregistered_intent() -> None:
    """ActionRouter.execute_intent returns succeeded=False for unknown intents."""
    from brileta.game.actions.base import GameIntent

    class UnknownIntent(GameIntent):
        pass

    controller, player = _make_world()
    router = ActionRouter(cast(Controller, controller))

    intent = UnknownIntent(cast(Controller, controller), player)
    result = router.execute_intent(intent)

    assert isinstance(result, GameActionResult)
    assert result.succeeded is False

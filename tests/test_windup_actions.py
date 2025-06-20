from __future__ import annotations

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from catley.controller import Controller, GameState
from catley.game.actions.base import GameIntent
from catley.game.actions.types import AnimationType
from catley.util.coordinates import TileDimensions
from catley.view.animation import Animation


class DummyAnimation(Animation):
    def update(self, delta_time: float) -> bool:
        return True


class DummyClock:
    def sync(self, fps: float | None = None) -> float:
        return 0.0


class DummyPlayer:
    def __init__(self) -> None:
        self.x = 0
        self.y = 0
        self.health = SimpleNamespace(is_alive=lambda: True)

    def get_next_action(self, controller: Controller) -> None:
        return None


class DummyLighting:
    def update(self, dt: float) -> None:
        pass


class DummyGameWorld:
    def __init__(self, w: int, h: int) -> None:
        self.player = DummyPlayer()
        self.lighting = DummyLighting()
        self.actors: list = []
        self.game_map = SimpleNamespace(transparent=[], visible=[], explored=[])


class DummyRenderer:
    def __init__(
        self, context: object, root_console: object, tile_dimensions: TileDimensions
    ) -> None:
        self.coordinate_converter = None
        self.root_console = SimpleNamespace(width=80, height=50)
        self.tile_dimensions = tile_dimensions


class DummyFrameManager:
    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.counter = 0

    def render_frame(self, delta: float) -> None:
        self.counter += 1
        if self.counter >= getattr(self.controller, "stop_after", 1):
            raise StopIteration()


class DummyInputHandler:
    def __init__(self, controller: Controller) -> None:
        self.movement_keys: set = set()

    def dispatch(self, event: object) -> None:
        pass


class DummyMovementHandler:
    def __init__(self, controller: Controller) -> None:
        pass

    def generate_intent(self, keys: set) -> None:
        return None


class DummyTurnManager:
    def __init__(self, controller: Controller) -> None:
        self._queue: list[GameIntent] = []
        self.processed: list[GameIntent] = []

    def queue_action(self, action: GameIntent) -> None:
        self._queue.append(action)

    def dequeue_player_action(self) -> GameIntent | None:
        return self._queue.pop(0) if self._queue else None

    def has_pending_actions(self) -> bool:
        return bool(self._queue)

    def is_player_turn_available(self) -> bool:
        return bool(self._queue)

    def process_player_action(self, action: GameIntent) -> None:
        self.processed.append(action)

    def process_all_npc_turns(self) -> None:
        pass


class DummyOverlaySystem:
    def __init__(self, controller: Controller) -> None:
        pass

    def handle_input(self, event: object) -> bool:
        return False

    def has_active_menus(self) -> bool:
        return False


class DummyTargetingMode:
    def __init__(self, controller: Controller) -> None:
        pass

    def enter(self) -> None:
        pass

    def _exit(self) -> None:
        pass

    def update(self) -> None:
        pass


@contextmanager
def patched_controller(stop_after: int):
    with ExitStack() as stack:
        stack.enter_context(patch("catley.controller.GameWorld", DummyGameWorld))
        stack.enter_context(patch("catley.controller.InputHandler", DummyInputHandler))
        stack.enter_context(
            patch("catley.controller.MovementInputHandler", DummyMovementHandler)
        )
        stack.enter_context(patch("catley.controller.FrameManager", DummyFrameManager))
        stack.enter_context(patch("catley.controller.Renderer", DummyRenderer))
        stack.enter_context(
            patch("catley.controller.OverlaySystem", DummyOverlaySystem)
        )
        stack.enter_context(
            patch("catley.controller.TargetingMode", DummyTargetingMode)
        )
        stack.enter_context(patch("catley.controller.Clock", DummyClock))
        stack.enter_context(patch("catley.controller.TurnManager", DummyTurnManager))
        stack.enter_context(patch.object(Controller, "update_fov", lambda self: None))
        stack.enter_context(patch("tcod.event.get", return_value=[]))
        stack.enter_context(patch("tcod.sdl.mouse.show", lambda val: None))

        controller = Controller(MagicMock(), MagicMock(), (1, 1))
        controller.stop_after = stop_after  # type: ignore[attr-defined]
        yield controller


def make_windup_action(controller: Controller) -> GameIntent:
    action = GameIntent(controller, controller.gw.player)
    action.animation_type = AnimationType.WIND_UP
    action.windup_animation = DummyAnimation()
    return action


def test_windup_enters_windup_state() -> None:
    with patched_controller(stop_after=1) as controller:
        intent = make_windup_action(controller)
        controller.queue_action(intent)

        with pytest.raises(StopIteration):
            controller.run_game_loop()

        assert controller.game_state == GameState.AWAITING_WINDUP
        assert controller._pending_player_action is intent
        assert not controller.animation_manager.is_queue_empty()


def test_windup_completes_to_processing() -> None:
    with patched_controller(stop_after=2) as controller:
        intent = make_windup_action(controller)
        controller.queue_action(intent)

        with pytest.raises(StopIteration):
            controller.run_game_loop()

        assert controller.game_state == GameState.PROCESSING_TURN
        assert controller._pending_player_action is intent
        assert controller.animation_manager.is_queue_empty()


def test_windup_action_processed() -> None:
    with patched_controller(stop_after=3) as controller:
        intent = make_windup_action(controller)
        controller.queue_action(intent)

        with pytest.raises(StopIteration):
            controller.run_game_loop()

        assert controller.game_state == GameState.PLAYING_ANIMATIONS
        assert controller._pending_player_action is None
        assert intent in controller.turn_manager.processed  # type: ignore[attr-defined]

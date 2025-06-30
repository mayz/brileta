from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import tcod.event

from catley import colors
from catley.game.actors import Character
from catley.input_handler import InputHandler
from catley.view.ui.commands import OpenExistingMenuUICommand
from tests.helpers import DummyGameWorld


@dataclass
class DummyFrameManager:
    graphics: Any = None
    cursor_manager: Any = field(
        default_factory=lambda: SimpleNamespace(update_mouse_position=lambda *a: None)
    )

    def get_world_coords_from_root_tile_coords(
        self, pos: tuple[int, int]
    ) -> tuple[int, int] | None:
        return pos


@dataclass
class DummyController:
    gw: DummyGameWorld
    graphics: Any
    coordinate_converter: Any
    frame_manager: DummyFrameManager
    start_actor_pathfinding: Any
    active_mode: Any = None
    overlay_system: Any = None


def make_input_handler() -> tuple[InputHandler, list[tuple[Any, tuple[int, int], Any]]]:
    gw = DummyGameWorld(width=10, height=10)
    player = Character(0, 0, "@", colors.WHITE, "Player", game_world=cast(Any, gw))
    gw.player = player
    gw.add_actor(player)

    calls: list[tuple[Any, tuple[int, int], Any]] = []

    def start_path(
        actor: Any, pos: tuple[int, int], final_intent: Any | None = None
    ) -> bool:
        calls.append((actor, pos, final_intent))
        return True

    from types import SimpleNamespace

    renderer = SimpleNamespace(
        tile_dimensions=(1, 1),
        root_console=SimpleNamespace(width=80, height=50),
        pixel_to_tile=lambda x, y: (x, y),
        get_display_scale_factor=lambda: (1.0, 1.0),
    )
    coordinate_converter = SimpleNamespace(pixel_to_tile=lambda x, y: (x, y))
    frame_manager = DummyFrameManager(graphics=renderer)
    overlay_system = SimpleNamespace()
    controller = DummyController(
        gw=gw,
        graphics=renderer,
        coordinate_converter=coordinate_converter,
        frame_manager=frame_manager,
        start_actor_pathfinding=start_path,
        overlay_system=overlay_system,
    )

    from catley.app import App

    class DummyApp(App):
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self) -> None:
            pass

        def prepare_for_new_frame(self) -> None:
            pass

        def present_frame(self) -> None:
            pass

        def toggle_fullscreen(self) -> None:
            pass

        def _exit_backend(self) -> None:
            pass

    dummy_app = DummyApp()
    ih = InputHandler(dummy_app, cast(Any, controller))
    return ih, calls


def test_shift_click_starts_pathfinding(monkeypatch: Any) -> None:
    ih, calls = make_input_handler()
    monkeypatch.setattr(
        tcod.event,
        "get_modifier_state",
        lambda: tcod.event.Modifier.SHIFT,
    )
    event = tcod.event.MouseButtonDown((5, 5), (5, 5), tcod.event.MouseButton.LEFT)
    result = ih._handle_mouse_button_down_event(event)
    assert result is None
    assert calls == [(ih.p, (5, 5), None)]


def test_right_click_distant_tile_opens_menu() -> None:
    ih, _ = make_input_handler()
    event = tcod.event.MouseButtonDown((5, 5), (5, 5), tcod.event.MouseButton.RIGHT)
    result = ih._handle_mouse_button_down_event(event)
    assert isinstance(result, OpenExistingMenuUICommand)


def test_escape_does_not_quit() -> None:
    ih, _ = make_input_handler()
    event = tcod.event.KeyDown(0, tcod.event.KeySym.ESCAPE, 0)
    assert ih._check_for_ui_command(event) is None

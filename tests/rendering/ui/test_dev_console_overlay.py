from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
import tcod.event

from catley.app import App, AppConfig
from catley.controller import Controller
from catley.util.live_vars import live_variable_registry
from catley.view.ui.dev_console_overlay import DevConsoleOverlay


class DummyApp(App):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__(AppConfig(width=80, height=50, title="test", vsync=False))

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

    def quit(self) -> None:
        """Simulate exit for testing."""
        raise SystemExit()


class DummyController:
    def __init__(self) -> None:
        self.graphics = SimpleNamespace(
            root_console=SimpleNamespace(width=80, height=50),
            tile_dimensions=(8, 16),
            sdl_renderer=None,
        )
        self.app = DummyApp()


def make_overlay() -> DevConsoleOverlay:
    return DevConsoleOverlay(cast("Controller", DummyController()))


def test_execute_command_set_get_toggle_watch() -> None:
    store: dict[str, object] = {"i": 0, "f": 0.0, "b": False}
    live_variable_registry.register(
        "test.int",
        getter=lambda: store["i"],
        setter=lambda v: store.__setitem__("i", v),
    )
    live_variable_registry.register(
        "test.float",
        getter=lambda: store["f"],
        setter=lambda v: store.__setitem__("f", v),
    )
    live_variable_registry.register(
        "test.bool",
        getter=lambda: store["b"],
        setter=lambda v: store.__setitem__("b", v),
    )

    ov = make_overlay()

    ov.input_buffer = "set test.int 5"
    ov._execute_command()
    assert store["i"] == 5

    ov.input_buffer = "set test.float 3.5"
    ov._execute_command()
    assert store["f"] == 3.5

    ov.input_buffer = "set test.bool true"
    ov._execute_command()
    assert store["b"] is True

    ov.input_buffer = "get test.int"
    ov._execute_command()
    assert "test.int =" in ov.history[-1]

    ov.input_buffer = "toggle test.bool"
    ov._execute_command()
    assert store["b"] is False

    ov.input_buffer = "show test.int"
    ov._execute_command()
    assert live_variable_registry.is_watched("test.int") is True

    prev_len = len(ov.history)
    ov.input_buffer = "help"
    ov._execute_command()
    added = list(ov.history)[prev_len + 1 :]
    assert added[0] == "Available Commands:"
    assert any(line.startswith("  help") for line in added[1:])


def test_execute_command_errors() -> None:
    ov = make_overlay()

    ov.input_buffer = "set missing 1"
    ov._execute_command()
    assert "not found" in ov.history[-1]

    ov.input_buffer = "unknowncmd"
    ov._execute_command()
    assert any("Unknown command" in line for line in ov.history)
    assert any("Available Commands:" in line for line in ov.history)


def test_show_command_invalid_var() -> None:
    ov = make_overlay()
    ov.input_buffer = "show missing"
    ov._execute_command()
    assert ov.history[-1] == "Error: Variable 'missing' not found."


@pytest.mark.parametrize("cmd", ["quit", "exit"])
def test_quit_and_exit_commands(cmd: str) -> None:
    ov = make_overlay()
    ov.input_buffer = cmd
    with pytest.raises(SystemExit):
        ov._execute_command()


def test_execute_command_list_outputs_and_filter() -> None:
    live_variable_registry.register(
        "cache.size",
        getter=lambda: 1,
        description="Size of cache",
    )
    live_variable_registry.register(
        "cache.enabled",
        getter=lambda: True,
        description="Cache enabled",
    )
    live_variable_registry.register(
        "misc.value",
        getter=lambda: 0,
        description="Misc value",
    )

    ov = make_overlay()

    prev_len = len(ov.history)
    ov.input_buffer = "list"
    ov._execute_command()
    all_lines = list(ov.history)[prev_len + 1 :]
    assert any("cache.size" in line for line in all_lines)
    assert any("cache.enabled" in line for line in all_lines)
    assert any("misc.value" in line for line in all_lines)

    prev_len = len(ov.history)
    ov.input_buffer = "list cache"
    ov._execute_command()
    filtered = list(ov.history)[prev_len + 1 :]
    assert len(filtered) == 2
    assert all(line.startswith("  cache") for line in filtered)


def test_handle_input_typing_and_submit(monkeypatch) -> None:
    ov = make_overlay()
    ov.show()
    exec_mock = MagicMock()
    ov._execute_command = exec_mock

    for ch in "set":
        e = tcod.event.TextInput(ch)
        ov.handle_input(e)

    assert ov.input_buffer == "set"

    e = tcod.event.TextInput("`")
    ov.handle_input(e)
    assert ov.input_buffer == "set"  # backtick ignored

    e = tcod.event.KeyDown(0, tcod.event.KeySym.BACKSPACE, 0)
    ov.handle_input(e)
    assert ov.input_buffer == "se"

    e = tcod.event.KeyDown(0, tcod.event.KeySym.RETURN, 0)
    ov.handle_input(e)
    assert exec_mock.called
    assert ov.input_buffer == ""


@pytest.mark.parametrize("key_sym", [tcod.event.KeySym.ESCAPE, tcod.event.KeySym.GRAVE])
def test_handle_input_escape_and_backquote_hide_console(
    key_sym: tcod.event.KeySym,
) -> None:
    ov = make_overlay()
    ov.show()

    assert ov.is_active

    e = tcod.event.KeyDown(0, key_sym, 0)
    ov.handle_input(e)

    assert not ov.is_active


def test_draw_content_uses_tile_height() -> None:
    ov = make_overlay()
    ov.canvas = MagicMock()
    ov.tile_dimensions = (8, 16)
    ov.height = 3
    ov.pixel_width = 10
    ov.pixel_height = 48
    ov.history.append("line")
    ov._cursor_visible = False
    ov.canvas.get_effective_line_height.return_value = 12
    ov.canvas.get_text_metrics.return_value = (4, 0, 12)
    ov.canvas.get_font_metrics.return_value = (8, 4)

    ov.draw_content()

    calls = ov.canvas.draw_text.call_args_list
    assert calls[0].args[1] == 16  # history line y position
    assert calls[1].args[1] == 32  # prompt y position


def test_tab_completion_cycles_candidates() -> None:
    live_variable_registry.register("tab.alpha", getter=lambda: 0)
    live_variable_registry.register("tab.beta", getter=lambda: 0)

    ov = make_overlay()
    ov.show()
    ov.input_buffer = "get tab."

    e = tcod.event.KeyDown(0, tcod.event.KeySym.TAB, 0)
    ov.handle_input(e)
    first = ov.input_buffer
    assert first in {"get tab.alpha", "get tab.beta"}

    ov.handle_input(tcod.event.KeyDown(0, tcod.event.KeySym.TAB, 0))
    assert ov.input_buffer != first


def test_history_navigation_up_down() -> None:
    ov = make_overlay()
    ov.input_buffer = "cmd1"
    ov._execute_command()
    ov.input_buffer = "cmd2"
    ov._execute_command()
    ov.show()
    ov.input_buffer = ""

    ov.handle_input(tcod.event.KeyDown(0, tcod.event.KeySym.UP, 0))
    assert ov.input_buffer == "cmd2"
    ov.handle_input(tcod.event.KeyDown(0, tcod.event.KeySym.UP, 0))
    assert ov.input_buffer == "cmd1"
    ov.handle_input(tcod.event.KeyDown(0, tcod.event.KeySym.DOWN, 0))
    assert ov.input_buffer == "cmd2"
    ov.handle_input(tcod.event.KeyDown(0, tcod.event.KeySym.DOWN, 0))
    assert ov.input_buffer == ""

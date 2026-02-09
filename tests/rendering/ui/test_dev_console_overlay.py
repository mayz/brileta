from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from brileta import input_events
from brileta.app import App, AppConfig
from brileta.controller import Controller
from brileta.util.live_vars import live_variable_registry
from brileta.view.ui.dev_console_overlay import DevConsoleOverlay


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
    # With pattern matching, unknown vars are treated as prefix patterns
    assert ov.history[-1] == "No variables matching 'missing'."


@pytest.mark.parametrize("cmd", ["quit", "exit"])
def test_quit_and_exit_commands(cmd: str) -> None:
    ov = make_overlay()
    ov.input_buffer = cmd
    with pytest.raises(SystemExit):
        ov._execute_command()


def test_list_no_args_enters_filter_mode() -> None:
    """'list' with no arguments enters interactive filter mode."""
    live_variable_registry.register("filt.a", getter=lambda: 1)

    ov = make_overlay()
    ov.input_buffer = "list"
    ov._execute_command()

    assert ov._filter_mode is True
    assert len(ov._filter_results) > 0


def test_list_with_pattern_shows_tagged_output() -> None:
    """'list <pattern>' uses substring matching and includes [var]/[metric] tags."""
    live_variable_registry.register(
        "cache.size", getter=lambda: 1, description="Size of cache"
    )
    live_variable_registry.register(
        "cache.enabled", getter=lambda: True, description="Cache enabled"
    )
    live_variable_registry.register(
        "misc.value", getter=lambda: 0, description="Misc value"
    )

    ov = make_overlay()

    # Substring filter
    prev_len = len(ov.history)
    ov.input_buffer = "list cache"
    ov._execute_command()
    filtered = list(ov.history)[prev_len + 1 :]
    assert len(filtered) == 2
    assert all("[var]" in line for line in filtered)
    assert all("cache." in line for line in filtered)


def test_list_vars_and_metrics_categories() -> None:
    """'list vars' and 'list metrics' filter by kind."""
    live_variable_registry.register("cat.var", getter=lambda: 1)
    live_variable_registry.register_metric("cat.metric", description="A metric")

    ov = make_overlay()

    prev_len = len(ov.history)
    ov.input_buffer = "list vars"
    ov._execute_command()
    var_lines = list(ov.history)[prev_len + 1 :]
    assert any("cat.var" in line for line in var_lines)
    assert not any("cat.metric" in line for line in var_lines)

    prev_len = len(ov.history)
    ov.input_buffer = "list metrics"
    ov._execute_command()
    metric_lines = list(ov.history)[prev_len + 1 :]
    assert any("cat.metric" in line for line in metric_lines)
    assert not any("cat.var" in line for line in metric_lines)


def test_natural_syntax_get() -> None:
    """Typing a bare variable name prints its value."""
    store = {"v": 42}
    live_variable_registry.register(
        "nat.get",
        getter=lambda: store["v"],
    )

    ov = make_overlay()
    ov.input_buffer = "nat.get"
    ov._execute_command()
    assert "nat.get = 42" in ov.history[-1]


def test_natural_syntax_set() -> None:
    """Typing 'varname = value' sets the variable."""
    store = {"v": 0}
    live_variable_registry.register(
        "nat.set",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", v),
    )

    ov = make_overlay()

    # With spaces around =
    ov.input_buffer = "nat.set = 7"
    ov._execute_command()
    assert store["v"] == 7
    assert "nat.set = 7" in ov.history[-1]

    # Without spaces
    ov.input_buffer = "nat.set=99"
    ov._execute_command()
    assert store["v"] == 99


def test_natural_syntax_set_missing_value() -> None:
    """'varname =' without a value shows usage."""
    live_variable_registry.register("nat.empty", getter=lambda: 0)

    ov = make_overlay()
    ov.input_buffer = "nat.empty ="
    ov._execute_command()
    assert "Usage:" in ov.history[-1]


def test_handle_input_typing_and_submit(monkeypatch) -> None:
    ov = make_overlay()
    ov.show()
    exec_mock = MagicMock()
    ov._execute_command = exec_mock

    for ch in "set":
        e = input_events.TextInput(ch)
        ov.handle_input(e)

    assert ov.input_buffer == "set"

    e = input_events.TextInput("`")
    ov.handle_input(e)
    assert ov.input_buffer == "set"  # backtick ignored

    e = input_events.KeyDown(sym=input_events.KeySym.BACKSPACE)
    ov.handle_input(e)
    assert ov.input_buffer == "se"

    e = input_events.KeyDown(sym=input_events.KeySym.RETURN)
    ov.handle_input(e)
    assert exec_mock.called
    assert ov.input_buffer == ""


@pytest.mark.parametrize(
    "key_sym", [input_events.KeySym.ESCAPE, input_events.KeySym.GRAVE]
)
def test_handle_input_escape_and_backquote_hide_console(
    key_sym: input_events.KeySym,
) -> None:
    ov = make_overlay()
    ov.show()

    assert ov.is_active

    e = input_events.KeyDown(sym=key_sym)
    ov.handle_input(e)

    assert not ov.is_active


def test_draw_content_uses_tile_height() -> None:
    ov = make_overlay()
    ov.canvas = MagicMock()
    ov.tile_dimensions = (8, 16)
    ov.height = 3
    ov.pixel_width = 10
    ov.pixel_height = 52  # 3 * 16 + 4 descender padding
    ov.history.clear()
    ov.history.append("line")
    ov._cursor_visible = False
    ov.canvas.get_effective_line_height.return_value = 12
    ov.canvas.get_text_metrics.return_value = (4, 0, 12)
    ov.canvas.get_font_metrics.return_value = (8, 4)

    ov.draw_content()

    calls = ov.canvas.draw_text.call_args_list
    assert calls[0].args[1] == 16  # history line y position
    assert calls[1].args[1] == 32  # prompt y position


def test_tab_completion_shows_candidates_and_common_prefix() -> None:
    """First Tab completes to common prefix and lists candidates in history."""
    live_variable_registry.register("tab.alpha", getter=lambda: 0)
    live_variable_registry.register("tab.beta", getter=lambda: 0)

    ov = make_overlay()
    ov.show()
    ov.input_buffer = "get tab."

    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.TAB))

    # Common prefix of "tab.alpha" and "tab.beta" is "tab.".
    # Input should stay at common prefix (no further common chars).
    assert ov.input_buffer == "get tab."

    # Both candidates should be listed in history.
    assert any("tab.alpha" in line for line in ov.history)
    assert any("tab.beta" in line for line in ov.history)


def test_tab_completion_unique_match() -> None:
    """Single candidate completes immediately without listing."""
    live_variable_registry.register("uniq.only", getter=lambda: 0)

    ov = make_overlay()
    ov.show()
    prev_history_len = len(ov.history)
    ov.input_buffer = "get uniq."

    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.TAB))
    assert ov.input_buffer == "get uniq.only"
    # No candidates should be printed for a unique match.
    assert len(ov.history) == prev_history_len


def test_tab_completion_cycles_after_listing() -> None:
    """Subsequent Tabs cycle through candidates after the initial listing."""
    live_variable_registry.register("cyc.a", getter=lambda: 0)
    live_variable_registry.register("cyc.b", getter=lambda: 0)

    ov = make_overlay()
    ov.show()
    ov.input_buffer = "get cyc."

    # First Tab: show candidates, complete to common prefix.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.TAB))
    assert ov.input_buffer == "get cyc."

    # Second Tab: cycle to first candidate.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.TAB))
    first = ov.input_buffer
    assert first in {"get cyc.a", "get cyc.b"}

    # Third Tab: cycle to second candidate.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.TAB))
    assert ov.input_buffer != first


def test_tab_completion_reverse_with_shift_tab() -> None:
    """Shift+Tab cycles candidates in reverse order."""
    live_variable_registry.register("stab.a", getter=lambda: 0)
    live_variable_registry.register("stab.b", getter=lambda: 0)
    live_variable_registry.register("stab.c", getter=lambda: 0)

    ov = make_overlay()
    ov.show()
    ov.input_buffer = "get stab."

    # First Tab: show candidates + common prefix.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.TAB))

    # Forward to first candidate.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.TAB))
    first = ov.input_buffer

    # Forward to second.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.TAB))
    second = ov.input_buffer
    assert second != first

    # Shift+Tab back to first.
    ov.handle_input(
        input_events.KeyDown(
            sym=input_events.KeySym.TAB, mod=input_events.Modifier.SHIFT
        )
    )
    assert ov.input_buffer == first


def test_history_navigation_up_down() -> None:
    ov = make_overlay()
    ov.input_buffer = "cmd1"
    ov._execute_command()
    ov.input_buffer = "cmd2"
    ov._execute_command()
    ov.show()
    ov.input_buffer = ""

    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.UP))
    assert ov.input_buffer == "cmd2"
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.UP))
    assert ov.input_buffer == "cmd1"
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.DOWN))
    assert ov.input_buffer == "cmd2"
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.DOWN))
    assert ov.input_buffer == ""


def test_console_consumes_all_keydown_events() -> None:
    """Console should consume all KeyDown events to prevent mode interference.

    This tests the fix for the bug where pressing I while console is open
    would open the inventory menu instead of just typing in the console.
    """
    ov = make_overlay()
    ov.show()

    # KeyDown for 'I' should be consumed (returns True) even though
    # the console doesn't explicitly handle this key
    key_i = input_events.KeySym(ord("i"))
    event = input_events.KeyDown(sym=key_i)
    result = ov.handle_input(event)

    assert result is True  # Event was consumed, won't leak to mode

    # Same for other arbitrary keys like 'T' (targeting mode trigger)
    key_t = input_events.KeySym(ord("t"))
    event = input_events.KeyDown(sym=key_t)
    result = ov.handle_input(event)

    assert result is True


def test_scrollback_shift_up_down() -> None:
    """Shift+Up/Down scrolls through command history."""
    ov = make_overlay()
    ov.show()
    shift = input_events.Modifier.SHIFT

    # Fill history with some lines.
    for i in range(20):
        ov.history.append(f"line {i}")

    # Shift+Up increases scroll offset.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.UP, mod=shift))
    assert ov._scroll_offset == 3

    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.UP, mod=shift))
    assert ov._scroll_offset == 6

    # Shift+Down decreases scroll offset.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.DOWN, mod=shift))
    assert ov._scroll_offset == 3

    # Can't go below zero.
    for _ in range(5):
        ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.DOWN, mod=shift))
    assert ov._scroll_offset == 0


def test_filter_mode_navigation() -> None:
    """Filter mode supports typing, navigation, and selection."""
    live_variable_registry.register("flt.alpha", getter=lambda: 1)
    live_variable_registry.register("flt.beta", getter=lambda: 2)
    live_variable_registry.register("flt.gamma", getter=lambda: 3)

    ov = make_overlay()
    ov.show()

    # Enter filter mode.
    ov._enter_filter_mode()
    assert ov._filter_mode is True
    assert len(ov._filter_results) >= 3  # at least our three vars

    # Type to filter.
    ov.handle_input(input_events.TextInput("f"))
    ov.handle_input(input_events.TextInput("l"))
    ov.handle_input(input_events.TextInput("t"))
    assert all("flt." in v.name for v in ov._filter_results)

    # Navigate with Down.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.DOWN))
    assert ov._filter_selected == 1

    # Navigate with Up.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.UP))
    assert ov._filter_selected == 0

    # Select with Enter.
    selected_name = ov._filter_results[0].name
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.RETURN))
    assert ov._filter_mode is False
    assert ov.input_buffer == selected_name


def test_filter_mode_escape_cancels() -> None:
    """Escape in filter mode exits without selecting."""
    live_variable_registry.register("fesc.a", getter=lambda: 0)

    ov = make_overlay()
    ov.show()
    ov.input_buffer = "original"

    ov._enter_filter_mode()
    assert ov._filter_mode is True

    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.ESCAPE))
    assert ov._filter_mode is False
    # Input buffer should NOT have been changed (no selection made).
    assert ov.input_buffer == "original"


def test_filter_mode_backspace() -> None:
    """Backspace in filter mode removes characters from the filter text."""
    live_variable_registry.register("fbsp.x", getter=lambda: 0)

    ov = make_overlay()
    ov.show()
    ov._enter_filter_mode("fbsp")
    assert ov._filter_text == "fbsp"

    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.BACKSPACE))
    assert ov._filter_text == "fbs"


def test_filter_mode_consumes_keydown() -> None:
    """Filter mode consumes all KeyDown events."""
    ov = make_overlay()
    ov.show()
    ov._enter_filter_mode()

    key_i = input_events.KeySym(ord("i"))
    event = input_events.KeyDown(sym=key_i)
    result = ov.handle_input(event)
    assert result is True

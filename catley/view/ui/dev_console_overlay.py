from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

import tcod.event

from catley import colors
from catley.util.live_vars import live_variable_registry
from catley.util.misc import string_to_type
from catley.view.render.backends.pillow.canvas import PillowImageCanvas
from catley.view.render.canvas import Canvas
from catley.view.ui.overlays import TextOverlay

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from catley.controller import Controller


class DevConsoleOverlay(TextOverlay):
    """Interactive developer console for inspecting live variables."""

    def __init__(self, controller: Controller) -> None:
        super().__init__(controller)
        self.input_buffer: str = ""
        self.history: deque[str] = deque(maxlen=100)
        self.command_cursor_pos: int = 0
        self._cursor_visible: bool = True
        self._last_blink_time: float = 0.0
        self.tab_completion_candidates: list[str] = []
        self.tab_completion_index: int = -1
        self.tab_completion_prefix: str = ""
        self.history_index: int = -1
        self._saved_input: str = ""
        self.is_interactive = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reset_input_state(self) -> None:
        """Reset tab completion and history navigation state."""
        self.tab_completion_candidates = []
        self.tab_completion_index = -1
        self.tab_completion_prefix = ""
        self.history_index = -1
        self._saved_input = ""

    def _handle_tab_completion(self) -> None:
        """Perform or cycle tab completion for variable names."""
        if not self.input_buffer:
            return

        if not self.tab_completion_candidates:
            prefix = self.input_buffer.split()[-1]
            self.tab_completion_prefix = prefix
            candidates = [
                v.name
                for v in live_variable_registry.get_all_variables()
                if v.name.startswith(prefix)
            ]
            if not candidates:
                return
            self.tab_completion_candidates = candidates
            self.tab_completion_index = 0
        else:
            self.tab_completion_index = (self.tab_completion_index + 1) % len(
                self.tab_completion_candidates
            )

        completion = self.tab_completion_candidates[self.tab_completion_index]
        start = self.input_buffer.rfind(self.tab_completion_prefix)
        self.input_buffer = self.input_buffer[:start] + completion

    def _handle_history_navigation(self, direction: int) -> None:
        """Navigate command history with ``direction`` of ``-1`` or ``1``."""
        commands = [line[2:] for line in self.history if line.startswith("> ")]
        if not commands:
            return

        if self.history_index == -1:
            self._saved_input = self.input_buffer
            self.history_index = len(commands)

        self.history_index += direction

        if self.history_index < 0:
            self.history_index = 0

        if self.history_index >= len(commands):
            self.history_index = -1
            self.input_buffer = self._saved_input
            return

        self.input_buffer = commands[self.history_index]

    def _show_help(self) -> None:
        """Append the list of available commands to the history."""
        commands = {
            "help": "Shows this list of commands.",
            "list [p]": "Lists variables, optionally filtering by pattern [p].",
            "get <var>": "Gets the current value of a variable.",
            "set <var> <val>": "Sets the value of a variable.",
            "toggle <var>": "Toggles a boolean variable.",
            "show <var>": "Toggles watching a variable in the stats overlay.",
            "quit": "Quit the game.",
            "exit": "Quit the game.",
        }
        self.history.append("Available Commands:")
        for cmd, desc in commands.items():
            self.history.append(f"  {cmd:<20} - {desc}")

    # ------------------------------------------------------------------
    # Overlay interface
    # ------------------------------------------------------------------
    def _get_backend(self) -> Canvas:
        return PillowImageCanvas(self.renderer, transparent=True)

    def _calculate_dimensions(self) -> None:
        self.tile_dimensions = self.renderer.tile_dimensions
        self.width = self.renderer.console_width_tiles
        self.height = max(1, self.renderer.console_height_tiles // 3)
        self.pixel_width = self.width * self.tile_dimensions[0]
        self.pixel_height = self.height * self.tile_dimensions[1]
        self.x_tiles = 0
        self.y_tiles = 0

    def draw_content(self) -> None:
        assert self.canvas is not None
        now = time.perf_counter()
        if now - self._last_blink_time >= 0.5:
            self._cursor_visible = not self._cursor_visible
            self._last_blink_time = now

        # Background
        self.canvas.draw_rect(
            0,
            0,
            self.pixel_width,
            self.pixel_height,
            (0, 0, 0, 200),  # type: ignore[arg-type]
            True,
        )

        # Border outlines
        self.canvas.draw_rect(0, 0, 1, self.pixel_height, colors.WHITE, True)
        self.canvas.draw_rect(
            self.pixel_width - 1,
            0,
            1,
            self.pixel_height,
            colors.WHITE,
            True,
        )
        self.canvas.draw_rect(
            0,
            self.pixel_height - 1,
            self.pixel_width,
            1,
            colors.WHITE,
            True,
        )

        tile_w, tile_h = self.tile_dimensions
        prompt_y = (self.height - 1) * tile_h

        # History lines (most recent at bottom)
        y = prompt_y - tile_h
        for line in reversed(self.history):
            if y < 0:
                break
            self.canvas.draw_text(0, y, line, colors.WHITE)
            y -= tile_h

        prompt = f"> {self.input_buffer}"
        self.canvas.draw_text(0, prompt_y, prompt, colors.YELLOW)
        if self._cursor_visible:
            cursor_x, _, _ = self.canvas.get_text_metrics(prompt)
            ascent, _ = self.canvas.get_font_metrics()
            self.canvas.draw_text(cursor_x, prompt_y, "|", colors.WHITE)

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle keyboard input for the developer console."""
        match event:
            case tcod.event.KeyDown(sym=sym) if sym in (
                tcod.event.KeySym.ESCAPE,
                tcod.event.KeySym.BACKQUOTE,
            ):
                self.hide()
                self._reset_input_state()
                return True
            case tcod.event.KeyDown(sym=tcod.event.KeySym.RETURN):
                self._execute_command()
                self.input_buffer = ""
                self._reset_input_state()
                return True
            case tcod.event.KeyDown(sym=tcod.event.KeySym.BACKSPACE):
                if self.input_buffer:
                    self.input_buffer = self.input_buffer[:-1]
                self._reset_input_state()
                return True
            case tcod.event.KeyDown(sym=tcod.event.KeySym.TAB, mod=mod):
                if mod == 0:  # Only respond to Tab without modifiers
                    self._handle_tab_completion()
                return True
            case tcod.event.KeyDown(sym=tcod.event.KeySym.UP):
                self._handle_history_navigation(-1)
                return True
            case tcod.event.KeyDown(sym=tcod.event.KeySym.DOWN):
                self._handle_history_navigation(1)
                return True
            case tcod.event.TextInput(text=text):
                if text.isprintable() and text != "`":
                    self.input_buffer += text
                self._reset_input_state()
                return True
        return False

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------
    def _execute_command(self) -> None:
        self.history.append(f"> {self.input_buffer}")
        command = self.input_buffer.strip()
        if not command:
            return

        parts = command.split()
        cmd = parts[0].lower()

        match cmd:
            case "?" | "help":
                self._show_help()
            case "list":
                pattern = parts[1] if len(parts) > 1 else ""
                all_vars = live_variable_registry.get_all_variables()
                matched_vars = [v for v in all_vars if v.name.startswith(pattern)]
                if not matched_vars:
                    self.history.append(f"No variables found matching '{pattern}'.")
                else:
                    for var in matched_vars:
                        self.history.append(f"  {var.name:<40} - {var.description}")
            case "set":
                if len(parts) < 3:
                    self.history.append("Usage: set <var> <value>")
                    return
                name = parts[1]
                value_str = " ".join(parts[2:])
                var = live_variable_registry.get_variable(name)
                if var is None:
                    self.history.append(f"Variable '{name}' not found.")
                    return
                current = var.get_value()
                value = string_to_type(value_str, type(current))
                if var.set_value(value):
                    self.history.append(f"Set {name} = {value}")
                else:
                    self.history.append(f"Variable '{name}' is read-only.")
            case "get":
                if len(parts) < 2:
                    self.history.append("Usage: get <var>")
                    return
                name = parts[1]
                var = live_variable_registry.get_variable(name)
                if var is None:
                    self.history.append(f"Variable '{name}' not found.")
                else:
                    self.history.append(f"{var.name} = {var.get_value()}")
            case "toggle":
                if len(parts) < 2:
                    self.history.append("Usage: toggle <var>")
                    return
                name = parts[1]
                var = live_variable_registry.get_variable(name)
                if var is None:
                    self.history.append(f"Variable '{name}' not found.")
                    return
                current = var.get_value()
                if var.set_value(not bool(current)):
                    self.history.append(f"{name} toggled to {not bool(current)}")
                else:
                    self.history.append(f"Variable '{name}' is read-only.")
            case "show" | "watch":
                if len(parts) < 2:
                    self.history.append("Usage: show <var>")
                    return
                name = parts[1]
                if live_variable_registry.get_variable(name) is None:
                    self.history.append(f"Error: Variable '{name}' not found.")
                    return
                live_variable_registry.toggle_watch(name)
                if live_variable_registry.is_watched(name):
                    self.history.append(f"Now watching: {name}")
                else:
                    self.history.append(f"No longer watching: {name}")
            case "quit" | "exit":
                raise SystemExit()
            case _:
                self.history.append(f"Unknown command: '{cmd}'")
                self._show_help()

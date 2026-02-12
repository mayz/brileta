from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

from brileta import colors, config, input_events
from brileta.backends.pillow.canvas import PillowImageCanvas
from brileta.types import PixelRect
from brileta.util.live_vars import LiveVariable, live_variable_registry
from brileta.util.misc import string_to_type
from brileta.view.render.canvas import Canvas
from brileta.view.ui.overlays import TextOverlay

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from brileta.controller import Controller

# Color used for the highlighted row in filter mode.
_HIGHLIGHT_BG = (40, 60, 90, 255)
# Muted color for the [var]/[metric] tag in list output.
_TAG_COLOR = (140, 140, 140)
# Subtle border color for the console frame.
_BORDER_COLOR = (80, 80, 80)
_SLIDER_TRACK_COLOR = (64, 64, 64)
_SLIDER_FILL_COLOR = (70, 130, 210)
_SLIDER_THUMB_COLOR = (190, 190, 190)
_SLIDER_TRACK_PADDING_PX = 8
_SLIDER_TRACK_MIN_WIDTH_PX = 140
_SLIDER_TRACK_MAX_WIDTH_PX = 520
_SLIDER_TRACK_WIDTH_RATIO = 0.6


def _common_prefix(strings: list[str]) -> str:
    """Return the longest common prefix of a list of strings."""
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def _var_tag(var: LiveVariable) -> str:
    """Return a short tag indicating the variable's kind."""
    return "[metric]" if var.metric else "[var]   "


class DevConsoleOverlay(TextOverlay):
    """Interactive developer console for inspecting and modifying live variables.

    Supports two modes:
    - **Command mode** (default): Type commands like ``set``, ``get``, ``toggle``,
      ``show``, ``world``, or use natural syntax (``varname`` to read,
      ``varname = value`` to write).
    - **Filter mode**: An interactive, searchable list of all registered variables.
      Enter it with ``list`` (no arguments) or ``Ctrl+L``. Type to filter,
      Up/Down to navigate, Enter to select, Escape to cancel.
    """

    def __init__(self, controller: Controller) -> None:
        super().__init__(controller)
        self.input_buffer: str = ""
        self.history: deque[str] = deque(maxlen=100)
        self.history.append("Type 'help' for available commands.")
        self.history.append("")
        self.command_cursor_pos: int = 0
        self._cursor_visible: bool = True
        self._last_blink_time: float = 0.0

        # Tab completion state
        self.tab_completion_candidates: list[str] = []
        self.tab_completion_index: int = -1
        self.tab_completion_prefix: str = ""

        # Command history navigation state
        self.history_index: int = -1
        self._saved_input: str = ""

        # Scroll offset for browsing command history (0 = bottom / most recent)
        self._scroll_offset: int = 0

        # Filter mode state (interactive variable browser)
        self._filter_mode: bool = False
        self._filter_text: str = ""
        self._filter_results: list[LiveVariable] = []
        self._filter_selected: int = 0
        self._filter_scroll: int = 0

        # Slider state for ranged writable live variables.
        self._slider_var: LiveVariable | None = None
        self._slider_dragging: bool = False
        self._slider_track_rect: PixelRect | None = None
        self._slider_track_y_center: int | None = None
        self._slider_tile_h: int = 0
        self._slider_last_value: float | None = None

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
        self._scroll_offset = 0

    def _format_live_value(self, var: LiveVariable, value: object) -> str:
        """Format a variable value for display across console and slider UI."""
        if var.formatter is not None:
            return var.formatter(value)
        # Default display style keeps numeric readouts compact for interactive
        # tweaking while preserving precise custom formats when provided.
        if isinstance(value, bool):
            return repr(value)
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            decimals = var.display_decimals if var.display_decimals is not None else 3
            decimals = max(0, decimals)
            formatted = f"{value:.{decimals}f}"
            # Explicit display_decimals means fixed precision formatting.
            if var.display_decimals is not None:
                return formatted
            if "." in formatted:
                formatted = formatted.rstrip("0").rstrip(".")
            return formatted
        return repr(value)

    def _handle_tab_completion(self, *, reverse: bool = False) -> None:
        """Perform or cycle tab completion for variable names.

        First Tab press: complete to the longest common prefix among candidates.
        If ambiguous, print all candidates. Subsequent presses cycle through them.

        Args:
            reverse: If True, cycle backwards through candidates (Shift+Tab).
        """
        if not self.input_buffer:
            return

        if not self.tab_completion_candidates:
            # First press: build the candidate list from the last token.
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
            self.tab_completion_index = -1

            # Complete to the longest common prefix.
            common = _common_prefix(candidates)
            start = self.input_buffer.rfind(prefix)
            self.input_buffer = self.input_buffer[:start] + common

            if len(candidates) == 1:
                # Unique match - nothing more to show.
                return

            # Ambiguous: show candidates so the user can see what's available.
            for name in candidates:
                self.history.append(f"  {name}")
            return

        # Subsequent press: cycle forward or backward.
        step = -1 if reverse else 1
        self.tab_completion_index = (self.tab_completion_index + step) % len(
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
            "list": "Interactive variable browser (also Ctrl+L).",
            "list <p>": "List variables matching pattern <p> (substring).",
            "list vars": "List only settable variables.",
            "list metrics": "List only metrics.",
            "<var>": "Print the current value of a variable (opens slider if ranged).",
            "<var> = <val>": "Set the value of a variable.",
            "set <var> <val>": "Set the value of a variable.",
            "get <var>": "Get the current value of a variable.",
            "toggle <var>": "Toggle a boolean variable.",
            "show <pattern>": "Toggle watching variable(s). Supports prefix* patterns.",
            "world [seed]": "Show current seed, or regenerate world with new seed.",
            "clear": "Clear console history.",
            "quit": "Quit the game.",
        }
        self.history.append("Available Commands:")
        for cmd, desc in commands.items():
            self.history.append(f"  {cmd:<20} - {desc}")
        self.history.append("")
        self.history.append("Vars are game state you can read and set.")
        self.history.append("Metrics are diagnostic data (use 'show' to watch).")

    # ------------------------------------------------------------------
    # Filter mode helpers
    # ------------------------------------------------------------------
    def _enter_filter_mode(self, initial_text: str = "") -> None:
        """Enter interactive filter mode for browsing variables."""
        self._filter_mode = True
        self._filter_text = initial_text
        self._filter_selected = 0
        self._filter_scroll = 0
        self._update_filter_results()
        self.invalidate()

    def _exit_filter_mode(self, *, selected: bool = False) -> None:
        """Leave filter mode, optionally pasting the selected variable name."""
        if selected and self._filter_results:
            idx = self._filter_selected
            if 0 <= idx < len(self._filter_results):
                self.input_buffer = self._filter_results[idx].name
        self._filter_mode = False
        self._filter_text = ""
        self._filter_results = []
        self._filter_selected = 0
        self._filter_scroll = 0
        self.invalidate()

    def _update_filter_results(self) -> None:
        """Rebuild the filtered variable list from the current filter text.

        Searches against both the variable name and its type tag (``var`` or
        ``metric``), so typing "var" or "metric" filters by kind.
        """
        query = self._filter_text.lower()
        self._filter_results = [
            v
            for v in live_variable_registry.get_all_variables()
            if query in f"{_var_tag(v)} {v.name}".lower()
        ]
        # Clamp the selection to the new list length.
        if self._filter_results:
            self._filter_selected = min(
                self._filter_selected, len(self._filter_results) - 1
            )
        else:
            self._filter_selected = 0

    def _activate_slider(self, var: LiveVariable) -> None:
        """Activate the slider for ``var`` if it supports slider control."""
        if not var.supports_slider():
            return

        self._slider_var = var
        self._slider_dragging = False
        self._slider_last_value = None
        self.invalidate()

    def _print_or_activate_slider(self, var: LiveVariable) -> None:
        """Print the value for normal vars or activate slider for ranged vars."""
        if var.supports_slider():
            self._activate_slider(var)
            return
        self.history.append(
            f"{var.name} = {self._format_live_value(var, var.get_value())}"
        )

    def _clear_slider(self) -> None:
        """Clear active slider state and drag interaction state."""
        had_slider = self._slider_var is not None
        self._slider_var = None
        self._slider_dragging = False
        self._slider_track_rect = None
        self._slider_track_y_center = None
        self._slider_tile_h = 0
        self._slider_last_value = None
        if had_slider:
            self.invalidate()

    def _get_slider_numeric_value(self) -> float | None:
        """Return the active slider variable's value as ``float`` if possible."""
        if self._slider_var is None:
            return None
        value = self._slider_var.get_value()
        return self._coerce_slider_numeric_value(value)

    def _coerce_slider_numeric_value(self, value: object) -> float | None:
        """Convert a candidate slider value to float when possible."""
        if isinstance(value, bool):
            return None
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def _hit_test_slider(self, px_x: float, px_y: float) -> bool:
        """Return ``True`` when a pixel coordinate is inside the slider row."""
        if self._slider_var is None or self._slider_track_y_center is None:
            return False
        if self._slider_tile_h <= 0:
            return False
        row_y = self._slider_track_y_center - (self._slider_tile_h / 2)
        return (
            0 <= px_x < self.pixel_width and row_y <= px_y < row_y + self._slider_tile_h
        )

    def _apply_slider_from_pixel(self, px_x: float) -> None:
        """Map ``px_x`` to slider value and apply it to the active variable."""
        var = self._slider_var
        rect = self._slider_track_rect
        if var is None or rect is None or var.value_range is None:
            return

        track_x, _track_y, track_w, _track_h = rect
        if track_w <= 0:
            return

        lo, hi = var.value_range
        if hi <= lo:
            value = lo
        else:
            normalized = (px_x - track_x) / track_w
            normalized = max(0.0, min(1.0, normalized))
            value = lo + normalized * (hi - lo)

        var.set_value(value)
        self._slider_last_value = float(value)

    # ------------------------------------------------------------------
    # Overlay interface
    # ------------------------------------------------------------------
    def _get_backend(self) -> Canvas:
        return PillowImageCanvas(
            self.controller.graphics, font_path=config.UI_FONT_PATH, transparent=True
        )

    def _calculate_dimensions(self) -> None:
        g = self.controller.graphics

        self.tile_dimensions = g.tile_dimensions
        self.width = g.console_width_tiles
        self.height = max(1, g.console_height_tiles // 2)
        self.pixel_width = self.width * self.tile_dimensions[0]
        # Add a few pixels below the last tile row so font descenders
        # (g, y, p, q) aren't clipped by the bottom border.
        _descender_pad = 4
        self.pixel_height = self.height * self.tile_dimensions[1] + _descender_pad
        self.x_tiles = 0
        self.y_tiles = 0

    def hide(self) -> None:
        """Hide the console and stop any in-progress slider drag."""
        super().hide()
        self._slider_dragging = False

    def draw_content(self) -> None:
        assert self.canvas is not None

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
        self.canvas.draw_rect(0, 0, 1, self.pixel_height, _BORDER_COLOR, True)
        self.canvas.draw_rect(
            self.pixel_width - 1,
            0,
            1,
            self.pixel_height,
            _BORDER_COLOR,
            True,
        )
        self.canvas.draw_rect(
            0,
            self.pixel_height - 1,
            self.pixel_width,
            1,
            _BORDER_COLOR,
            True,
        )

        _tile_w, tile_h = self.tile_dimensions
        prompt_y = (self.height - 1) * tile_h

        if self._filter_mode:
            self._draw_filter_mode(tile_h, prompt_y)
        else:
            self._draw_command_mode(tile_h, prompt_y)

    def _draw_command_mode(self, tile_h: int, prompt_y: int) -> None:
        """Draw the normal command-mode view with history and prompt."""
        assert self.canvas is not None

        bottom_y = prompt_y - tile_h
        if self._slider_var is not None:
            # Slider takes two rows: label then track.
            slider_label_y = prompt_y - (2 * tile_h)
            self._draw_slider(slider_label_y, tile_h)
            bottom_y = slider_label_y - tile_h
        else:
            self._slider_track_rect = None
            self._slider_track_y_center = None
            self._slider_tile_h = 0

        history_list = list(self.history)
        visible_end = len(history_list) - self._scroll_offset

        # Reserve rows for scroll indicators so they don't overlap content.
        # Top row (y=0) is reserved when there are clipped lines above.
        # Bottom history row (prompt_y - tile_h) is reserved when scrolled up.
        top_limit = tile_h  # always reserve top row for potential indicator
        if self._scroll_offset > 0:
            bottom_y -= tile_h  # reserve a row for the "below" indicator

        clipped_above = False
        y = bottom_y
        for i in range(visible_end - 1, -1, -1):
            if y < top_limit:
                clipped_above = True
                break
            self.canvas.draw_text(0, y, history_list[i], colors.WHITE)
            y -= tile_h

        if clipped_above:
            self.canvas.draw_text(0, 0, "  [Shift+Up to scroll]", _TAG_COLOR)

        if self._scroll_offset > 0:
            indicator_y = prompt_y - tile_h
            if self._slider_var is not None:
                indicator_y -= 2 * tile_h
            self.canvas.draw_text(
                0, indicator_y, "  [Shift+Down to scroll]", _TAG_COLOR
            )

        prompt = f"> {self.input_buffer}"
        self.canvas.draw_text(0, prompt_y, prompt, colors.YELLOW)
        if self._cursor_visible:
            cursor_x, _, _ = self.canvas.get_text_metrics(prompt)
            _ascent, _ = self.canvas.get_font_metrics()
            self.canvas.draw_text(cursor_x, prompt_y, "|", colors.WHITE)

    def _draw_slider(self, row_y: int, tile_h: int) -> None:
        """Draw the current slider row and update hit-test geometry."""
        assert self.canvas is not None

        var = self._slider_var
        if var is None or var.value_range is None:
            self._slider_track_rect = None
            self._slider_track_y_center = None
            self._slider_tile_h = 0
            return

        lo, hi = var.value_range
        # Read once so expensive getters do not run twice per draw.
        raw_value = var.get_value()
        current_value = self._coerce_slider_numeric_value(raw_value)
        if current_value is None:
            self._slider_track_rect = None
            self._slider_track_y_center = None
            self._slider_tile_h = 0
            return

        ratio = (current_value - lo) / (hi - lo) if hi > lo else 0.0
        ratio = max(0.0, min(1.0, ratio))

        label = f"{var.name} = {self._format_live_value(var, raw_value)}"
        track_row_y = row_y + tile_h
        track_x = _SLIDER_TRACK_PADDING_PX
        desired_track_w = int(self.pixel_width * _SLIDER_TRACK_WIDTH_RATIO)
        base_track_w = max(
            _SLIDER_TRACK_MIN_WIDTH_PX,
            min(_SLIDER_TRACK_MAX_WIDTH_PX, desired_track_w),
        )
        available_w = self.pixel_width - (_SLIDER_TRACK_PADDING_PX * 2)
        track_w = max(8, min(base_track_w, available_w))
        track_h = max(4, tile_h // 4)
        track_y = track_row_y + (tile_h - track_h) // 2

        fill_w = int(track_w * ratio)
        thumb_size = max(8, min(tile_h - 2, 12))
        thumb_center_x = track_x + int((track_w - 1) * ratio)
        thumb_x = round(thumb_center_x - (thumb_size / 2))
        thumb_x = max(track_x, min(track_x + track_w - thumb_size, thumb_x))
        thumb_y = track_row_y + (tile_h - thumb_size) // 2

        self.canvas.draw_text(0, row_y, label, colors.WHITE)
        self.canvas.draw_rect(
            track_x, track_y, track_w, track_h, _SLIDER_TRACK_COLOR, True
        )
        self.canvas.draw_rect(
            track_x, track_y, fill_w, track_h, _SLIDER_FILL_COLOR, True
        )
        self.canvas.draw_rect(
            thumb_x, thumb_y, thumb_size, thumb_size, _SLIDER_THUMB_COLOR, True
        )

        self._slider_track_rect = (track_x, track_y, track_w, track_h)
        self._slider_track_y_center = track_row_y + (tile_h // 2)
        self._slider_tile_h = tile_h

    def _draw_filter_mode(self, tile_h: int, prompt_y: int) -> None:
        """Draw the interactive filter-mode view with a scrollable variable list."""
        assert self.canvas is not None
        self._slider_track_rect = None
        self._slider_track_y_center = None
        self._slider_tile_h = 0

        # Available rows for displaying results (above the prompt line).
        visible_rows = max(1, (prompt_y // tile_h) - 1)

        # Ensure the selected item is visible by adjusting scroll.
        if self._filter_selected < self._filter_scroll:
            self._filter_scroll = self._filter_selected
        elif self._filter_selected >= self._filter_scroll + visible_rows:
            self._filter_scroll = self._filter_selected - visible_rows + 1

        # Header line showing result count.
        total = len(live_variable_registry.get_all_variables())
        showing = len(self._filter_results)
        header = f"  {showing}/{total} variables"
        if self._filter_text:
            header += f" matching '{self._filter_text}'"
        self.canvas.draw_text(0, 0, header, _TAG_COLOR)

        # Draw filtered results.
        y = tile_h  # Start below header.
        for i in range(self._filter_scroll, len(self._filter_results)):
            if y >= prompt_y:
                break
            var = self._filter_results[i]
            is_selected = i == self._filter_selected

            # Highlight background for the selected row.
            if is_selected:
                self.canvas.draw_rect(
                    0,
                    y,
                    self.pixel_width,
                    tile_h,
                    _HIGHLIGHT_BG,  # type: ignore[arg-type]
                    True,
                )

            tag = _var_tag(var)
            line = f"  {tag} {var.name:<35} {var.description}"
            text_color = colors.WHITE if is_selected else (180, 180, 180)
            self.canvas.draw_text(0, y, line, text_color)
            y += tile_h

        # Prompt line.
        prompt = f"filter> {self._filter_text}"
        self.canvas.draw_text(0, prompt_y, prompt, colors.YELLOW)
        if self._cursor_visible:
            cursor_x, _, _ = self.canvas.get_text_metrics(prompt)
            _ascent, _ = self.canvas.get_font_metrics()
            self.canvas.draw_text(cursor_x, prompt_y, "|", colors.WHITE)

    def draw(self) -> None:
        """Update cursor blink state and draw when needed."""
        if not self.is_active:
            return

        if self._slider_var is not None and not self._filter_mode:
            slider_value = self._get_slider_numeric_value()
            if self._slider_dragging:
                self.invalidate()
            elif slider_value is not None and slider_value != self._slider_last_value:
                self._slider_last_value = slider_value
                self.invalidate()

        now = time.perf_counter()
        if now - self._last_blink_time >= 0.5:
            self._cursor_visible = not self._cursor_visible
            self._last_blink_time = now
            self.invalidate()

        # Filter mode needs to re-render every frame (selection highlight, etc.).
        if self._filter_mode:
            self.invalidate()

        super().draw()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def handle_input(self, event: input_events.InputEvent) -> bool:
        """Handle keyboard input for the developer console."""
        if self._filter_mode:
            return self._handle_filter_input(event)
        return self._handle_command_input(event)

    def _handle_command_input(self, event: input_events.InputEvent) -> bool:
        """Handle input in normal command mode."""
        match event:
            case input_events.KeyDown(sym=sym) if sym in (
                input_events.KeySym.ESCAPE,
                input_events.KeySym.GRAVE,
            ):
                self.hide()
                self._reset_input_state()
                return True
            case input_events.KeyDown(sym=input_events.KeySym.RETURN):
                self._execute_command()
                self.input_buffer = ""
                self._reset_input_state()
                return True
            case input_events.KeyDown(sym=input_events.KeySym.BACKSPACE):
                if self.input_buffer:
                    self.input_buffer = self.input_buffer[:-1]
                self._reset_input_state()
                return True
            case input_events.KeyDown(sym=input_events.KeySym.TAB, mod=mod):
                if mod & input_events.Modifier.SHIFT:
                    self._handle_tab_completion(reverse=True)
                elif mod == 0:
                    self._handle_tab_completion()
                return True
            case input_events.KeyDown(sym=input_events.KeySym.UP, mod=mod):
                if mod & input_events.Modifier.SHIFT:
                    max_scroll = max(0, len(self.history) - 1)
                    self._scroll_offset = min(self._scroll_offset + 3, max_scroll)
                    self.invalidate()
                else:
                    self._handle_history_navigation(-1)
                return True
            case input_events.KeyDown(sym=input_events.KeySym.DOWN, mod=mod):
                if mod & input_events.Modifier.SHIFT:
                    self._scroll_offset = max(self._scroll_offset - 3, 0)
                    self.invalidate()
                else:
                    self._handle_history_navigation(1)
                return True
            case input_events.TextInput(text=text):
                if text.isprintable() and text != "`":
                    self.input_buffer += text
                self._reset_input_state()
                return True
            case input_events.MouseButtonDown(
                button=input_events.MouseButton.LEFT, position=position
            ):
                if self._hit_test_slider(position.x, position.y):
                    rect = self._slider_track_rect
                    if rect is not None:
                        track_x, _track_y, track_w, _track_h = rect
                        if track_x <= position.x <= track_x + track_w:
                            self._slider_dragging = True
                            self._apply_slider_from_pixel(position.x)
                            self.invalidate()
                # Consume mouse while console is active to avoid leaking to gameplay.
                return True
            case input_events.MouseMotion(position=position):
                if self._slider_dragging:
                    self._apply_slider_from_pixel(position.x)
                    self.invalidate()
                return True
            case input_events.MouseButtonUp(button=input_events.MouseButton.LEFT):
                self._slider_dragging = False
                return True
            case (
                input_events.MouseButtonDown()
                | input_events.MouseButtonUp()
                | input_events.MouseWheel()
            ):
                # Consume all mouse input while console is active.
                return True
            case input_events.KeyDown(sym=sym, mod=mod):
                # Ctrl+L enters filter mode.
                if (mod & input_events.Modifier.CTRL) and sym.value == ord("l"):
                    self._enter_filter_mode()
                # Consume all other keyboard input while console is active.
                return True
        return False

    def _handle_filter_input(self, event: input_events.InputEvent) -> bool:
        """Handle input in interactive filter mode."""
        match event:
            case input_events.KeyDown(sym=input_events.KeySym.ESCAPE):
                self._exit_filter_mode()
                return True
            case input_events.KeyDown(sym=input_events.KeySym.GRAVE):
                self._exit_filter_mode()
                self.hide()
                return True
            case input_events.KeyDown(sym=input_events.KeySym.RETURN):
                self._exit_filter_mode(selected=True)
                return True
            case input_events.KeyDown(sym=input_events.KeySym.BACKSPACE):
                if self._filter_text:
                    self._filter_text = self._filter_text[:-1]
                    self._update_filter_results()
                self.invalidate()
                return True
            case input_events.KeyDown(sym=input_events.KeySym.UP):
                if self._filter_results:
                    self._filter_selected = max(0, self._filter_selected - 1)
                self.invalidate()
                return True
            case input_events.KeyDown(sym=input_events.KeySym.DOWN):
                if self._filter_results:
                    self._filter_selected = min(
                        len(self._filter_results) - 1,
                        self._filter_selected + 1,
                    )
                self.invalidate()
                return True
            case input_events.KeyDown(sym=input_events.KeySym.TAB, mod=mod):
                # Tab/Shift+Tab also cycle through results in filter mode.
                if self._filter_results:
                    step = -1 if (mod & input_events.Modifier.SHIFT) else 1
                    self._filter_selected = (self._filter_selected + step) % len(
                        self._filter_results
                    )
                self.invalidate()
                return True
            case input_events.TextInput(text=text):
                if text.isprintable() and text != "`":
                    self._filter_text += text
                    self._update_filter_results()
                self.invalidate()
                return True
            case input_events.KeyDown():
                # Consume all keyboard input while filter mode is active.
                return True
            case (
                input_events.MouseButtonDown()
                | input_events.MouseButtonUp()
                | input_events.MouseMotion()
                | input_events.MouseWheel()
            ):
                # Consume all mouse input while console is active.
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
        # A newly submitted command dismisses any active slider, then command
        # handlers may reactivate it for a ranged variable.
        self._clear_slider()

        parts = command.split()
        cmd = parts[0].lower()

        match cmd:
            case "?" | "help":
                self._show_help()
            case "list":
                self._handle_list_command(parts)
            case "set":
                self._handle_set_command(parts)
            case "get":
                if len(parts) < 2:
                    self.history.append("Usage: get <var>")
                    return
                name = parts[1]
                var = live_variable_registry.get_variable(name)
                if var is None:
                    self.history.append(f"Variable '{name}' not found.")
                else:
                    self._print_or_activate_slider(var)
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
                if not isinstance(current, bool):
                    self.history.append(
                        f"Variable '{name}' is not a boolean (use 'show' or 'watch')."
                    )
                    return
                if var.set_value(not current):
                    self.history.append(f"{name} toggled to {not current}")
                else:
                    self.history.append(f"Variable '{name}' is read-only.")
            case "show" | "watch":
                if len(parts) < 2:
                    self.history.append("Usage: show/watch <var> or <prefix*>")
                    return
                pattern = parts[1]
                # Support pattern matching with * or prefix matching
                if (
                    "*" in pattern
                    or live_variable_registry.get_variable(pattern) is None
                ):
                    # Treat as prefix pattern (strip trailing * if present)
                    prefix = pattern.rstrip("*")
                    all_vars = live_variable_registry.get_all_variables()
                    matched = [v for v in all_vars if v.name.startswith(prefix)]
                    if not matched:
                        self.history.append(f"No variables matching '{pattern}'.")
                        return
                    for var in matched:
                        live_variable_registry.toggle_watch(var.name)
                        status = (
                            "+" if live_variable_registry.is_watched(var.name) else "-"
                        )
                        self.history.append(f"  {status} {var.name}")
                    self.history.append(f"Toggled {len(matched)} variable(s).")
                else:
                    # Exact match
                    live_variable_registry.toggle_watch(pattern)
                    if live_variable_registry.is_watched(pattern):
                        self.history.append(f"Now watching: {pattern}")
                    else:
                        self.history.append(f"No longer watching: {pattern}")
            case "world":
                if len(parts) < 2:
                    # No seed provided - show current seed
                    current_seed = self.controller._current_seed
                    self.history.append(f"Current seed: {current_seed}")
                else:
                    # Seed provided - regenerate world
                    seed_str = parts[1]
                    # Try to parse as int, otherwise use as string seed
                    try:
                        seed = int(seed_str)
                    except ValueError:
                        seed = seed_str
                    self.history.append(f"Generating new world with seed: {seed}")
                    self.controller.new_world(seed)
                    self.history.append("World regenerated.")
            case "clear":
                self.history.clear()
                self._scroll_offset = 0
            case "quit" | "exit":
                self.controller.app.quit()
            case _:
                self._handle_natural_syntax(command, cmd)

    def _handle_list_command(self, parts: list[str]) -> None:
        """Handle the 'list' command with its various forms.

        - ``list`` with no args enters interactive filter mode.
        - ``list vars`` shows only settable variables.
        - ``list metrics`` shows only metrics.
        - ``list <pattern>`` shows variables matching the substring pattern.
        """
        if len(parts) < 2:
            # No argument: enter interactive filter mode.
            self._enter_filter_mode()
            return

        arg = parts[1].lower()
        all_vars = live_variable_registry.get_all_variables()

        if arg == "vars":
            matched = [v for v in all_vars if not v.metric]
        elif arg == "metrics":
            matched = [v for v in all_vars if v.metric]
        else:
            # Substring match (case-insensitive).
            pattern = parts[1].lower()
            matched = [v for v in all_vars if pattern in v.name.lower()]

        if not matched:
            self.history.append(f"No variables found matching '{parts[1]}'.")
        else:
            for var in matched:
                tag = _var_tag(var)
                self.history.append(f"  {tag} {var.name:<35} {var.description}")

    def _handle_set_command(self, parts: list[str]) -> None:
        """Handle the explicit 'set <var> <value>' command."""
        if len(parts) < 3:
            self.history.append("Usage: set <var> <value>")
            return
        name = parts[1]
        value_str = " ".join(parts[2:])
        self._set_variable(name, value_str)

    def _handle_natural_syntax(self, command: str, cmd: str) -> None:
        """Handle natural syntax: bare variable name to get, ``var = val`` to set.

        Falls back to "Unknown command" if nothing matches.
        """
        # Assignment syntax: "varname = value" or "varname=value"
        if "=" in command:
            name, _, value_str = command.partition("=")
            name = name.strip()
            value_str = value_str.strip()
            if not value_str:
                self.history.append("Usage: <var> = <value>")
                return
            self._set_variable(name, value_str)
            return

        # Bare variable name: implicit get.
        stripped = command.strip()
        var = live_variable_registry.get_variable(stripped)
        if var is not None:
            self._print_or_activate_slider(var)
            return

        # Prefix pattern: "sun.*" or "sun." prints all matching values.
        if "*" in stripped or stripped.endswith("."):
            prefix = stripped.rstrip("*")
            matched = [
                v
                for v in live_variable_registry.get_all_variables()
                if v.name.startswith(prefix)
            ]
            if matched:
                if len(matched) == 1:
                    self._print_or_activate_slider(matched[0])
                else:
                    for v in matched:
                        self.history.append(
                            f"{v.name} = {self._format_live_value(v, v.get_value())}"
                        )
                return

        self.history.append(f"Unknown command: '{cmd}'")
        self._show_help()

    def _set_variable(self, name: str, value_str: str) -> None:
        """Shared logic for setting a variable by name from a string value."""
        var = live_variable_registry.get_variable(name)
        if var is None:
            self.history.append(f"Variable '{name}' not found.")
            return
        # Metric variables are for recording measurements, not direct assignment.
        if var.metric:
            self.history.append(
                f"Variable '{name}' is a metric (use 'show' to monitor it)."
            )
            return
        current = var.get_value()
        value = string_to_type(value_str, type(current))
        if var.set_value(value):
            self._print_or_activate_slider(var)
        else:
            self.history.append(f"Variable '{name}' is read-only.")

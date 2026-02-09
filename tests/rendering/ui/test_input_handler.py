"""Tests for InputHandler and ExploreMode input handling.

The InputHandler is a thin dispatcher that routes events to modes.
ExploreMode handles the actual input processing for gameplay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

from brileta import colors, input_events
from brileta.game.actors import Character
from brileta.input_events import Keys
from brileta.input_handler import InputHandler
from brileta.modes.explore import ExploreMode
from tests.helpers import DummyGameWorld


@dataclass
class DummyFrameManager:
    graphics: Any = None
    cursor_manager: Any = field(
        default_factory=lambda: SimpleNamespace(
            update_mouse_position=lambda *a: None,
            set_active_cursor_type=lambda *a: None,
        )
    )
    combat_tooltip_overlay: Any = None
    action_panel_view: Any = field(
        default_factory=lambda: SimpleNamespace(
            x=0,
            y=0,
            width=0,
            height=0,
            get_hotkeys=lambda: {},
            update_hover_from_pixel=lambda *_a, **_kw: False,
            execute_at_pixel=lambda *_a, **_kw: False,
            get_action_at_pixel=lambda *_a, **_kw: None,
            invalidate_cache=lambda: None,
        )
    )
    equipment_view: Any = field(
        default_factory=lambda: SimpleNamespace(
            x=0,
            y=0,
            width=0,
            height=0,
            set_hover_row=lambda *_a, **_kw: None,
            is_row_in_active_slot=lambda *_a, **_kw: False,
            handle_click=lambda *_a, **_kw: False,
        )
    )
    dev_console_overlay: Any = field(default_factory=SimpleNamespace)
    world_view: Any = field(
        default_factory=lambda: SimpleNamespace(
            _render_selection_and_hover_outlines=lambda: None
        )
    )

    def get_world_coords_from_root_tile_coords(
        self, pos: tuple[int, int]
    ) -> tuple[int, int] | None:
        return pos

    def get_visible_bounds(self) -> None:
        return None


@dataclass
class DummyController:
    gw: DummyGameWorld
    graphics: Any
    coordinate_converter: Any
    frame_manager: DummyFrameManager
    start_actor_pathfinding: Any
    active_mode: Any = None
    explore_mode: Any = None
    overlay_system: Any = None
    app: Any = None
    mode_stack: list[Any] = field(default_factory=list)

    def update_hovered_actor(self, _mouse_pos: Any) -> None:
        """No-op placeholder to satisfy InputHandler interactions."""
        return

    def is_combat_mode(self) -> bool:
        return False


def make_input_handler() -> tuple[InputHandler, list[tuple[Any, tuple[int, int], Any]]]:
    """Create an InputHandler for testing."""
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

    renderer = SimpleNamespace(
        tile_dimensions=(1, 1),
        root_console=SimpleNamespace(width=80, height=50),
        pixel_to_tile=lambda x, y: (x, y),
        get_display_scale_factor=lambda: (1.0, 1.0),
        create_canvas=lambda transparent=True: MagicMock(),
    )
    coordinate_converter = SimpleNamespace(pixel_to_tile=lambda x, y: (x, y))
    frame_manager = DummyFrameManager(graphics=renderer)
    overlay_system = SimpleNamespace(
        handle_input=lambda e: False,
        has_interactive_overlays=lambda: False,
    )

    from brileta.app import App

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

    controller = DummyController(
        gw=gw,
        graphics=renderer,
        coordinate_converter=coordinate_converter,
        frame_manager=frame_manager,
        start_actor_pathfinding=start_path,
        overlay_system=overlay_system,
        app=dummy_app,
    )

    # Create a mock active_mode that doesn't consume input
    # (so InputHandler falls through to overlay handling)
    controller.active_mode = SimpleNamespace(handle_input=lambda e: False)
    controller.mode_stack = [controller.active_mode]

    ih = InputHandler(dummy_app, cast(Any, controller))
    return ih, calls


def make_explore_mode() -> tuple[ExploreMode, Any, list[tuple[Any, tuple[int, int]]]]:
    """Create an ExploreMode for testing mouse click handling."""
    gw = DummyGameWorld(width=10, height=10)
    player = Character(0, 0, "@", colors.WHITE, "Player", game_world=cast(Any, gw))
    gw.player = player
    gw.add_actor(player)

    calls: list[tuple[Any, tuple[int, int]]] = []

    def start_plan(
        actor: Any, plan: Any, target_position: tuple[int, int] | None = None, **kwargs
    ) -> bool:
        if target_position:
            calls.append((actor, target_position))
        return True

    def start_path(
        actor: Any, pos: tuple[int, int], final_intent: Any | None = None
    ) -> bool:
        # Legacy pathfinding - not used for simple walk-to-tile anymore
        return True

    renderer = SimpleNamespace(
        tile_dimensions=(1, 1),
        root_console=SimpleNamespace(width=80, height=50),
        pixel_to_tile=lambda x, y: (x, y),
        get_display_scale_factor=lambda: (1.0, 1.0),
        create_canvas=lambda transparent=True: MagicMock(),
    )
    frame_manager = DummyFrameManager(graphics=renderer)
    overlay_system = SimpleNamespace(
        handle_input=lambda e: False,
        toggle_overlay=lambda o: None,
        show_menu=lambda m: None,
    )

    from brileta.app import App

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

    controller = SimpleNamespace(
        gw=gw,
        graphics=renderer,
        frame_manager=frame_manager,
        start_actor_pathfinding=start_path,
        start_plan=start_plan,
        overlay_system=overlay_system,
        app=dummy_app,
        enter_combat_mode=lambda: None,
        queue_action=lambda a: None,
        deselect_target=lambda: None,
    )

    mode = ExploreMode(cast(Any, controller))
    mode.enter()
    return mode, controller, calls


def test_shift_click_starts_pathfinding() -> None:
    """Shift+click should start pathfinding to the clicked tile."""
    mode, _controller, calls = make_explore_mode()
    event = input_events.MouseButtonDown(
        position=input_events.Point(5, 5),
        button=input_events.MouseButton.LEFT,
        mod=input_events.Modifier.SHIFT,
    )
    result = mode._handle_mouse_click(event)
    assert result is True
    assert calls == [(mode.player, (5, 5))]


def test_right_click_distant_tile_opens_menu() -> None:
    """Right-clicking a visible tile should return True (consumed)."""
    mode, _controller, _ = make_explore_mode()
    event = input_events.MouseButtonDown(
        position=input_events.Point(5, 5), button=input_events.MouseButton.RIGHT
    )
    result = mode._handle_mouse_click(event)
    # Right click is always consumed
    assert result is True


def test_escape_does_not_quit() -> None:
    """Pressing Escape in ExploreMode should not be handled (not a quit)."""
    mode, _, _ = make_explore_mode()
    event = input_events.KeyDown(sym=input_events.KeySym.ESCAPE)
    # ExploreMode doesn't handle Escape (it's not in the match cases)
    result = mode.handle_input(event)
    assert result is False


def test_quit_key_is_handled_by_input_handler() -> None:
    """Q key should be handled by InputHandler as quit."""
    ih, _ = make_input_handler()
    event = input_events.KeyDown(sym=Keys.KEY_Q)
    # _is_quit_key should return True for Q key
    assert ih._is_quit_key(event) is True


def test_t_key_not_handled_by_explore_mode() -> None:
    """T key is not handled by ExploreMode - it was removed as combat toggle.

    Note: T key was removed in the equipment slot interaction rework.
    Combat is now entered by clicking the active equipment slot.
    """
    mode, _, _ = make_explore_mode()
    event = input_events.KeyDown(sym=Keys.KEY_T)
    result = mode.handle_input(event)
    assert result is False  # T key is not consumed


def test_mouse_motion_invalidates_combat_tooltip() -> None:
    """Mouse motion should invalidate the combat tooltip when in combat."""
    ih, _ = make_input_handler()

    class DummyTooltip:
        def __init__(self) -> None:
            self.invalidated = 0
            self.is_active = True

        def invalidate(self) -> None:
            self.invalidated += 1

    tooltip = DummyTooltip()
    ih.controller.frame_manager.combat_tooltip_overlay = tooltip
    ih.controller.is_combat_mode = lambda: True

    event = input_events.MouseMotion(
        position=input_events.Point(5, 5), motion=input_events.Point(1, 1)
    )
    ih.dispatch(event)

    assert tooltip.invalidated == 1

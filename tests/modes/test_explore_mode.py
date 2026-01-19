"""Tests for ExploreMode.

Tests cover movement key tracking, timing state management,
and mode lifecycle behavior.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import tcod.event

from catley import colors
from catley.game.actors import Character
from catley.modes.explore import ExploreMode
from tests.helpers import DummyGameWorld


def make_explore_mode() -> tuple[ExploreMode, Any, DummyGameWorld]:
    """Create an ExploreMode for testing.

    Returns:
        Tuple of (ExploreMode, controller, game_world)
    """
    gw = DummyGameWorld(width=10, height=10)
    player = Character(0, 0, "@", colors.WHITE, "Player", game_world=cast(Any, gw))
    gw.player = player
    gw.add_actor(player)

    renderer = SimpleNamespace(
        tile_dimensions=(1, 1),
        root_console=SimpleNamespace(width=80, height=50),
        pixel_to_tile=lambda x, y: (x, y),
        get_display_scale_factor=lambda: (1.0, 1.0),
        create_canvas=lambda transparent=True: MagicMock(),
    )
    frame_manager = SimpleNamespace(
        graphics=renderer,
        cursor_manager=SimpleNamespace(
            update_mouse_position=lambda *a: None,
            set_active_cursor_type=lambda *a: None,
        ),
        get_world_coords_from_root_tile_coords=lambda pos: pos,
    )
    overlay_system = SimpleNamespace(
        handle_input=lambda e: False,
        toggle_overlay=lambda o: None,
        show_menu=lambda m: None,
        has_interactive_overlays=lambda: False,
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

    controller = SimpleNamespace(
        gw=gw,
        graphics=renderer,
        frame_manager=frame_manager,
        overlay_system=overlay_system,
        app=dummy_app,
        enter_combat_mode=lambda: None,
        queue_action=lambda a: None,
        start_actor_pathfinding=lambda *a: True,
        last_input_time=0.0,
        action_count_for_latency_metric=0,
        _execute_player_action_immediately=lambda intent: None,
        turn_manager=SimpleNamespace(on_player_action=lambda: None),
    )

    mode = ExploreMode(cast(Any, controller))
    mode.enter()
    return mode, controller, gw


# -----------------------------------------------------------------------------
# Movement Key Tracking Tests
# -----------------------------------------------------------------------------


def test_keydown_adds_to_movement_keys() -> None:
    """KeyDown for movement key adds it to the set."""
    mode, _, _ = make_explore_mode()
    event = tcod.event.KeyDown(0, tcod.event.KeySym.UP, 0)
    mode.handle_input(event)
    assert tcod.event.KeySym.UP in mode.movement_keys


def test_keyup_removes_from_movement_keys() -> None:
    """KeyUp for movement key removes it from the set."""
    mode, _, _ = make_explore_mode()
    mode.movement_keys.add(tcod.event.KeySym.UP)
    event = tcod.event.KeyUp(0, tcod.event.KeySym.UP, 0)
    mode.handle_input(event)
    assert tcod.event.KeySym.UP not in mode.movement_keys


def test_keyup_handles_missing_key_gracefully() -> None:
    """KeyUp for key not in set doesn't raise."""
    mode, _, _ = make_explore_mode()
    # movement_keys is empty, but KeyUp should not raise
    event = tcod.event.KeyUp(0, tcod.event.KeySym.UP, 0)
    result = mode.handle_input(event)
    assert result is True  # Event was consumed


def test_movement_keys_cleared_on_exit() -> None:
    """Movement keys are cleared when exiting the mode."""
    mode, _, _ = make_explore_mode()
    mode.movement_keys.add(tcod.event.KeySym.UP)
    mode.movement_keys.add(tcod.event.KeySym.LEFT)
    mode._exit()
    assert len(mode.movement_keys) == 0


def test_multiple_movement_keys_tracked() -> None:
    """Multiple movement keys can be held simultaneously."""
    mode, _, _ = make_explore_mode()
    mode.handle_input(tcod.event.KeyDown(0, tcod.event.KeySym.UP, 0))
    mode.handle_input(tcod.event.KeyDown(0, tcod.event.KeySym.RIGHT, 0))
    assert tcod.event.KeySym.UP in mode.movement_keys
    assert tcod.event.KeySym.RIGHT in mode.movement_keys
    assert len(mode.movement_keys) == 2


# -----------------------------------------------------------------------------
# Movement Timing Reset Test
# -----------------------------------------------------------------------------


def test_update_resets_timing_when_no_keys() -> None:
    """Calling update() with no keys resets is_first_move_of_burst.

    This is the regression test for the movement timing bug. When the user
    releases all movement keys, update() must still call generate_intent()
    so that is_first_move_of_burst gets reset. Otherwise, the next key press
    would use the fast repeat interval instead of the initial delay.
    """
    mode, _, _ = make_explore_mode()

    # Simulate: user pressed key, moved, released
    mode.move_generator.is_first_move_of_burst = False
    mode.movement_keys = set()  # No keys held

    # Call update - this should reset timing via generate_intent()
    mode.update()

    # After update with empty keys, timing should be reset
    assert mode.move_generator.is_first_move_of_burst is True


def test_update_preserves_timing_when_keys_held() -> None:
    """Calling update() with keys held preserves is_first_move_of_burst state.

    When keys are held, is_first_move_of_burst should be set to False after
    the first move is generated.
    """
    mode, _, _ = make_explore_mode()

    # Simulate: user holding a key
    mode.movement_keys.add(tcod.event.KeySym.UP)

    # First update - should generate a move and set is_first_move_of_burst=False
    mode.update()

    assert mode.move_generator.is_first_move_of_burst is False


# -----------------------------------------------------------------------------
# Mode Active State Tests
# -----------------------------------------------------------------------------


def test_enter_sets_active_true() -> None:
    """Entering the mode sets active to True."""
    gw = DummyGameWorld(width=10, height=10)
    player = Character(0, 0, "@", colors.WHITE, "Player", game_world=cast(Any, gw))
    gw.player = player
    gw.add_actor(player)

    controller = SimpleNamespace(
        gw=gw,
        graphics=SimpleNamespace(get_display_scale_factor=lambda: (1.0, 1.0)),
        frame_manager=None,
        overlay_system=None,
        app=None,
    )

    mode = ExploreMode(cast(Any, controller))
    assert mode.active is False

    mode.enter()
    assert mode.active is True


def test_exit_sets_active_false() -> None:
    """Exiting the mode sets active to False."""
    mode, _, _ = make_explore_mode()
    assert mode.active is True

    mode._exit()
    assert mode.active is False


# -----------------------------------------------------------------------------
# _has_available_actions Tests
# -----------------------------------------------------------------------------


def test_has_available_actions_returns_false_for_non_character_actor() -> None:
    """Non-Character Actors (like items) return False for available actions."""
    from catley.game.actors import Actor

    mode, _, gw = make_explore_mode()

    # Create a basic Actor (not a Character)
    actor = Actor(3, 3, "!", colors.WHITE, "Test Actor", game_world=cast(Any, gw))
    gw.add_actor(actor)

    result = mode._has_available_actions(actor)
    assert result is False


def test_has_available_actions_returns_false_for_out_of_bounds_tile() -> None:
    """Tile positions outside map bounds return False."""
    mode, _, _ = make_explore_mode()

    # Test position outside map bounds (map is 10x10)
    out_of_bounds_pos = (100, 100)
    result = mode._has_available_actions(out_of_bounds_pos)
    assert result is False


def test_has_available_actions_returns_false_for_invisible_tile() -> None:
    """Tile positions that aren't visible return False."""
    mode, _, gw = make_explore_mode()

    # Ensure tile is not visible
    gw.game_map.visible[5, 5] = False

    in_bounds_pos = (5, 5)
    result = mode._has_available_actions(in_bounds_pos)
    assert result is False


def test_has_available_actions_returns_true_for_visible_tile() -> None:
    """Visible tile positions return True (context menu can show tile info)."""
    mode, _, gw = make_explore_mode()

    # Make tile visible
    gw.game_map.visible[5, 5] = True

    in_bounds_pos = (5, 5)
    result = mode._has_available_actions(in_bounds_pos)
    assert result is True

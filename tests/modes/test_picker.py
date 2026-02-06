"""Tests for PickerMode - modal target/tile selection.

PickerMode is a standalone selection interface that any mode can use
to get a target selection from the user.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from catley import input_events
from catley.controller import Controller
from catley.modes.picker import PickerResult
from tests.helpers import reset_dummy_controller


@pytest.fixture
def picker_controller(dummy_controller: Controller) -> Controller:
    reset_dummy_controller(dummy_controller)
    return dummy_controller


def test_start_pushes_mode_onto_stack(picker_controller: Controller) -> None:
    """start() pushes PickerMode onto the mode stack."""
    controller = picker_controller

    assert controller.picker_mode not in controller.mode_stack

    controller.picker_mode.start(on_select=lambda r: None)

    assert controller.picker_mode in controller.mode_stack
    assert controller.active_mode is controller.picker_mode
    assert controller.picker_mode.active is True


def test_escape_calls_on_cancel_and_pops(picker_controller: Controller) -> None:
    """Escape key calls on_cancel callback and pops picker from stack."""
    controller = picker_controller
    on_cancel = MagicMock()
    on_select = MagicMock()

    controller.picker_mode.start(on_select=on_select, on_cancel=on_cancel)
    assert controller.active_mode is controller.picker_mode

    # Press Escape
    event = input_events.KeyDown(sym=input_events.KeySym.ESCAPE)
    result = controller.picker_mode.handle_input(event)

    assert result is True
    on_cancel.assert_called_once()
    on_select.assert_not_called()
    assert controller.picker_mode not in controller.mode_stack
    assert controller.picker_mode.active is False


def test_t_key_not_handled_by_picker(picker_controller: Controller) -> None:
    """T key is not handled by picker - it falls through to modes below.

    Note: T key was removed as a combat toggle in the equipment slot interaction
    rework. Combat is now entered by clicking the active equipment slot.
    """
    from catley.input_events import Keys

    controller = picker_controller
    on_cancel = MagicMock()
    on_select = MagicMock()

    controller.picker_mode.start(on_select=on_select, on_cancel=on_cancel)
    assert controller.active_mode is controller.picker_mode

    # Press T - should NOT be consumed by picker
    event = input_events.KeyDown(sym=Keys.KEY_T)
    result = controller.picker_mode.handle_input(event)

    # T key should fall through (not handled)
    assert result is False
    on_cancel.assert_not_called()
    on_select.assert_not_called()
    # Picker should still be active
    assert controller.picker_mode in controller.mode_stack
    assert controller.picker_mode.active is True


def test_escape_works_without_on_cancel(picker_controller: Controller) -> None:
    """Escape works even when on_cancel is not provided."""
    controller = picker_controller
    on_select = MagicMock()

    controller.picker_mode.start(on_select=on_select)  # No on_cancel
    assert controller.active_mode is controller.picker_mode

    # Press Escape - should not raise
    event = input_events.KeyDown(sym=input_events.KeySym.ESCAPE)
    result = controller.picker_mode.handle_input(event)

    assert result is True
    assert controller.picker_mode not in controller.mode_stack


def test_click_on_invalid_target_consumed_but_no_callback(
    picker_controller: Controller,
) -> None:
    """Click on invalid tile is consumed but doesn't call on_select."""
    controller = picker_controller
    on_select = MagicMock()
    on_cancel = MagicMock()

    # Filter that rejects all tiles
    controller.picker_mode.start(
        on_select=on_select,
        on_cancel=on_cancel,
        valid_filter=lambda x, y: False,
    )

    # Mock _get_tile_at_mouse to return a valid tile (simulating a click)
    controller.picker_mode._get_tile_at_mouse = MagicMock(return_value=(5, 5))

    # Create a mock click event (pixel_pos, tile_pos, button)
    event = input_events.MouseButtonDown(
        position=input_events.Point(100, 100), button=input_events.MouseButton.LEFT
    )

    result = controller.picker_mode.handle_input(event)

    # Event is consumed
    assert result is True
    # But no callbacks are invoked (filter rejected the tile)
    on_select.assert_not_called()
    on_cancel.assert_not_called()
    # Picker is still active (didn't pop)
    assert controller.picker_mode in controller.mode_stack


def test_click_on_valid_target_calls_on_select_and_pops(
    picker_controller: Controller,
) -> None:
    """Click on valid tile calls on_select with result and pops picker."""
    controller = picker_controller
    on_select = MagicMock()

    # Filter that accepts all tiles
    controller.picker_mode.start(
        on_select=on_select,
        valid_filter=lambda x, y: True,
    )

    # Mock _get_tile_at_mouse to return a tile
    controller.picker_mode._get_tile_at_mouse = MagicMock(return_value=(5, 10))
    # Mock _get_actor_at_tile to return None (no actor at tile)
    controller.picker_mode._get_actor_at_tile = MagicMock(return_value=None)

    # Create a click event
    event = input_events.MouseButtonDown(
        position=input_events.Point(100, 100), button=input_events.MouseButton.LEFT
    )

    result = controller.picker_mode.handle_input(event)

    assert result is True
    on_select.assert_called_once()

    # Check the PickerResult passed to on_select
    call_args = on_select.call_args[0][0]
    assert call_args.tile == (5, 10)
    assert call_args.actor is None

    # Picker should be popped
    assert controller.picker_mode not in controller.mode_stack
    assert controller.picker_mode.active is False


def test_non_picker_input_falls_through(picker_controller: Controller) -> None:
    """Non-picker input falls through to modes below."""
    controller = picker_controller
    on_select = MagicMock()

    controller.picker_mode.start(on_select=on_select)

    # Keys not handled by picker should fall through (return False)
    for sym in [
        input_events.KeySym.UP,
        input_events.KeySym.SPACE,
        input_events.KeySym.TAB,
    ]:
        event = input_events.KeyDown(sym=sym)
        result = controller.picker_mode.handle_input(event)
        assert result is False, f"Key {sym} should fall through"

    # Picker should still be active
    assert controller.picker_mode in controller.mode_stack


def test_cursor_changes_to_crosshair_on_enter(picker_controller: Controller) -> None:
    """Cursor changes to crosshair when picker enters."""
    controller = picker_controller

    # Mock the cursor manager
    mock_cursor_manager = MagicMock()
    controller.picker_mode._cursor_manager = mock_cursor_manager

    controller.picker_mode.start(on_select=lambda r: None)

    mock_cursor_manager.set_active_cursor_type.assert_called_with("crosshair")


def test_cursor_changes_to_arrow_on_exit(picker_controller: Controller) -> None:
    """Cursor changes back to arrow when picker exits."""
    controller = picker_controller

    # Mock the cursor manager
    mock_cursor_manager = MagicMock()
    controller.picker_mode._cursor_manager = mock_cursor_manager

    controller.picker_mode.start(on_select=lambda r: None)
    mock_cursor_manager.reset_mock()

    # Exit via Escape
    event = input_events.KeyDown(sym=input_events.KeySym.ESCAPE)
    controller.picker_mode.handle_input(event)

    mock_cursor_manager.set_active_cursor_type.assert_called_with("arrow")


def test_render_underneath_callback_invoked(picker_controller: Controller) -> None:
    """render_underneath callback is called during render_world."""
    controller = picker_controller
    render_underneath = MagicMock()

    controller.picker_mode.start(
        on_select=lambda r: None, render_underneath=render_underneath
    )

    controller.picker_mode.render_world()

    render_underneath.assert_called_once()


def test_render_world_without_callback(picker_controller: Controller) -> None:
    """render_world works without render_underneath callback."""
    controller = picker_controller

    controller.picker_mode.start(on_select=lambda r: None)

    # Should not raise
    controller.picker_mode.render_world()


def test_callbacks_cleared_on_exit(picker_controller: Controller) -> None:
    """Callbacks are cleared when picker exits to avoid holding references."""
    controller = picker_controller
    on_select = MagicMock()
    on_cancel = MagicMock()
    valid_filter = MagicMock()
    render_underneath = MagicMock()

    controller.picker_mode.start(
        on_select=on_select,
        on_cancel=on_cancel,
        valid_filter=valid_filter,
        render_underneath=render_underneath,
    )

    # Exit via Escape
    event = input_events.KeyDown(sym=input_events.KeySym.ESCAPE)
    controller.picker_mode.handle_input(event)

    # All callbacks should be None
    assert controller.picker_mode._on_select is None
    assert controller.picker_mode._on_cancel is None
    assert controller.picker_mode._valid_filter is None
    assert controller.picker_mode._render_underneath is None


def test_picker_auto_pushed_by_combat_mode(picker_controller: Controller) -> None:
    """CombatMode automatically pushes PickerMode for target selection.

    This is the new behavior where CombatMode delegates click-to-select
    to PickerMode.
    """
    controller = picker_controller

    # Enter combat mode - picker is automatically pushed
    controller.enter_combat_mode()

    # Stack should have explore -> combat -> picker
    assert len(controller.mode_stack) == 3
    assert controller.mode_stack[0] is controller.explore_mode
    assert controller.mode_stack[1] is controller.combat_mode
    assert controller.mode_stack[2] is controller.picker_mode
    assert controller.active_mode is controller.picker_mode

    # Escape in picker mode should exit combat entirely
    # (picker's on_cancel calls combat's pop_mode)
    event = input_events.KeyDown(sym=input_events.KeySym.ESCAPE)
    controller.picker_mode.handle_input(event)

    # Should be back to explore mode (combat was also popped)
    assert controller.active_mode is controller.explore_mode
    assert len(controller.mode_stack) == 1


def test_combat_mode_repushes_picker_after_selection(
    picker_controller: Controller,
) -> None:
    """After selecting a target, CombatMode re-pushes PickerMode.

    This keeps the player in combat mode for subsequent attacks.
    """
    controller = picker_controller
    controller.enter_combat_mode()

    # Verify initial state
    assert controller.active_mode is controller.picker_mode
    initial_on_select = controller.picker_mode._on_select

    # Manually call the on_select callback to simulate a selection
    # (In real usage, PickerMode would pop itself first, then call the callback)
    controller.pop_mode()  # Simulate picker popping itself
    assert controller.active_mode is controller.combat_mode

    # The callback was stored before pop cleared it, so we use the initial one
    # This simulates what PickerMode does: save callback, pop, call callback
    from catley.modes.picker import PickerResult

    result = PickerResult(actor=None, tile=(5, 5))
    initial_on_select(result)  # type: ignore[misc]

    # Combat mode should have re-pushed picker
    assert controller.active_mode is controller.picker_mode
    assert len(controller.mode_stack) == 3


def test_picker_result_dataclass() -> None:
    """PickerResult dataclass holds actor and tile."""
    result = PickerResult(actor=None, tile=(5, 10))

    assert result.actor is None
    assert result.tile == (5, 10)


def test_inactive_picker_returns_false(picker_controller: Controller) -> None:
    """handle_input returns False when picker is not active."""
    controller = picker_controller

    # Picker is not started, so not active
    assert controller.picker_mode.active is False

    event = input_events.KeyDown(sym=input_events.KeySym.ESCAPE)
    result = controller.picker_mode.handle_input(event)

    assert result is False


def test_render_world_skipped_when_inactive(picker_controller: Controller) -> None:
    """render_world does nothing when picker is not active."""
    controller = picker_controller
    render_underneath = MagicMock()

    # Set callback but don't start picker
    controller.picker_mode._render_underneath = render_underneath
    controller.picker_mode.render_world()

    # Callback should not have been called
    render_underneath.assert_not_called()


def test_combat_mode_with_no_valid_targets(picker_controller: Controller) -> None:
    """Combat mode works when there are no enemies in range.

    Picker should still be pushed when entering combat mode, even if there
    are no valid targets. Cancelling should exit cleanly back to explore mode.
    """
    controller = picker_controller

    # Remove all actors except the player to simulate "no enemies in range"
    gw = controller.gw
    player = gw.player
    actors_to_remove = [actor for actor in list(gw.actors) if actor is not player]
    for actor in actors_to_remove:
        gw.remove_actor(actor)

    # Only the player remains
    assert len(gw.actors) == 1
    assert gw.actors[0] is player

    # Enter combat mode
    controller.enter_combat_mode()

    # Picker is still pushed on top of combat mode
    assert len(controller.mode_stack) == 3
    assert controller.mode_stack[0] is controller.explore_mode
    assert controller.mode_stack[1] is controller.combat_mode
    assert controller.mode_stack[2] is controller.picker_mode
    assert controller.active_mode is controller.picker_mode

    # No candidates in combat mode
    assert controller.combat_mode.candidates == []

    # Cancel via Escape - should exit cleanly back to explore
    event = input_events.KeyDown(sym=input_events.KeySym.ESCAPE)
    controller.picker_mode.handle_input(event)

    # Back to explore mode
    assert controller.active_mode is controller.explore_mode
    assert len(controller.mode_stack) == 1


def test_click_outside_map_bounds_falls_through(picker_controller: Controller) -> None:
    """Click outside the game map bounds falls through to modes below.

    When a click is outside the map area (e.g., on UI elements like the
    equipment view), the event should NOT be consumed. This allows UI
    handlers in modes below to process the click - for example, clicking
    the active equipment slot to toggle combat mode.
    """
    controller = picker_controller
    on_select = MagicMock()
    on_cancel = MagicMock()

    # Filter that accepts all valid tiles
    controller.picker_mode.start(
        on_select=on_select,
        on_cancel=on_cancel,
        valid_filter=lambda x, y: True,
    )

    # Mock _get_tile_at_mouse to return None (simulating click outside map)
    controller.picker_mode._get_tile_at_mouse = MagicMock(return_value=None)

    # Create a click event
    event = input_events.MouseButtonDown(
        position=input_events.Point(100, 100), button=input_events.MouseButton.LEFT
    )

    result = controller.picker_mode.handle_input(event)

    # Event falls through (not consumed) so UI handlers can process it
    assert result is False
    # No callbacks are invoked (click was outside map)
    on_select.assert_not_called()
    on_cancel.assert_not_called()
    # Picker is still active (didn't pop)
    assert controller.picker_mode in controller.mode_stack

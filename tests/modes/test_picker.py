"""Tests for PickerMode - modal target/tile selection.

PickerMode is a standalone selection interface that any mode can use
to get a target selection from the user.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import tcod.event

from catley.modes.picker import PickerResult
from tests.helpers import get_controller_with_player_and_map


def test_start_pushes_mode_onto_stack() -> None:
    """start() pushes PickerMode onto the mode stack."""
    controller = get_controller_with_player_and_map()

    assert controller.picker_mode not in controller.mode_stack

    controller.picker_mode.start(on_select=lambda r: None)

    assert controller.picker_mode in controller.mode_stack
    assert controller.active_mode is controller.picker_mode
    assert controller.picker_mode.active is True


def test_escape_calls_on_cancel_and_pops() -> None:
    """Escape key calls on_cancel callback and pops picker from stack."""
    controller = get_controller_with_player_and_map()
    on_cancel = MagicMock()
    on_select = MagicMock()

    controller.picker_mode.start(on_select=on_select, on_cancel=on_cancel)
    assert controller.active_mode is controller.picker_mode

    # Press Escape
    event = tcod.event.KeyDown(0, tcod.event.KeySym.ESCAPE, 0)
    result = controller.picker_mode.handle_input(event)

    assert result is True
    on_cancel.assert_called_once()
    on_select.assert_not_called()
    assert controller.picker_mode not in controller.mode_stack
    assert controller.picker_mode.active is False


def test_t_key_not_handled_by_picker() -> None:
    """T key is not handled by picker - it falls through to modes below.

    Note: T key was removed as a combat toggle in the equipment slot interaction
    rework. Combat is now entered by clicking the active equipment slot.
    """
    from catley.input_handler import Keys

    controller = get_controller_with_player_and_map()
    on_cancel = MagicMock()
    on_select = MagicMock()

    controller.picker_mode.start(on_select=on_select, on_cancel=on_cancel)
    assert controller.active_mode is controller.picker_mode

    # Press T - should NOT be consumed by picker
    event = tcod.event.KeyDown(0, Keys.KEY_T, 0)
    result = controller.picker_mode.handle_input(event)

    # T key should fall through (not handled)
    assert result is False
    on_cancel.assert_not_called()
    on_select.assert_not_called()
    # Picker should still be active
    assert controller.picker_mode in controller.mode_stack
    assert controller.picker_mode.active is True


def test_escape_works_without_on_cancel() -> None:
    """Escape works even when on_cancel is not provided."""
    controller = get_controller_with_player_and_map()
    on_select = MagicMock()

    controller.picker_mode.start(on_select=on_select)  # No on_cancel
    assert controller.active_mode is controller.picker_mode

    # Press Escape - should not raise
    event = tcod.event.KeyDown(0, tcod.event.KeySym.ESCAPE, 0)
    result = controller.picker_mode.handle_input(event)

    assert result is True
    assert controller.picker_mode not in controller.mode_stack


def test_click_on_invalid_target_consumed_but_no_callback() -> None:
    """Click on invalid tile is consumed but doesn't call on_select."""
    controller = get_controller_with_player_and_map()
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
    event = tcod.event.MouseButtonDown(
        (100, 100), (10, 10), tcod.event.MouseButton.LEFT
    )

    result = controller.picker_mode.handle_input(event)

    # Event is consumed
    assert result is True
    # But no callbacks are invoked (filter rejected the tile)
    on_select.assert_not_called()
    on_cancel.assert_not_called()
    # Picker is still active (didn't pop)
    assert controller.picker_mode in controller.mode_stack


def test_click_on_valid_target_calls_on_select_and_pops() -> None:
    """Click on valid tile calls on_select with result and pops picker."""
    controller = get_controller_with_player_and_map()
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
    event = tcod.event.MouseButtonDown(
        (100, 100), (10, 10), tcod.event.MouseButton.LEFT
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


def test_non_picker_input_falls_through() -> None:
    """Non-picker input falls through to modes below."""
    controller = get_controller_with_player_and_map()
    on_select = MagicMock()

    controller.picker_mode.start(on_select=on_select)

    # Keys not handled by picker should fall through (return False)
    for sym in [tcod.event.KeySym.UP, tcod.event.KeySym.SPACE, tcod.event.KeySym.TAB]:
        event = tcod.event.KeyDown(0, sym, 0)
        result = controller.picker_mode.handle_input(event)
        assert result is False, f"Key {sym} should fall through"

    # Picker should still be active
    assert controller.picker_mode in controller.mode_stack


def test_cursor_changes_to_crosshair_on_enter() -> None:
    """Cursor changes to crosshair when picker enters."""
    controller = get_controller_with_player_and_map()

    # Mock the cursor manager
    mock_cursor_manager = MagicMock()
    controller.picker_mode._cursor_manager = mock_cursor_manager

    controller.picker_mode.start(on_select=lambda r: None)

    mock_cursor_manager.set_active_cursor_type.assert_called_with("crosshair")


def test_cursor_changes_to_arrow_on_exit() -> None:
    """Cursor changes back to arrow when picker exits."""
    controller = get_controller_with_player_and_map()

    # Mock the cursor manager
    mock_cursor_manager = MagicMock()
    controller.picker_mode._cursor_manager = mock_cursor_manager

    controller.picker_mode.start(on_select=lambda r: None)
    mock_cursor_manager.reset_mock()

    # Exit via Escape
    event = tcod.event.KeyDown(0, tcod.event.KeySym.ESCAPE, 0)
    controller.picker_mode.handle_input(event)

    mock_cursor_manager.set_active_cursor_type.assert_called_with("arrow")


def test_render_underneath_callback_invoked() -> None:
    """render_underneath callback is called during render_world."""
    controller = get_controller_with_player_and_map()
    render_underneath = MagicMock()

    controller.picker_mode.start(
        on_select=lambda r: None, render_underneath=render_underneath
    )

    controller.picker_mode.render_world()

    render_underneath.assert_called_once()


def test_render_world_without_callback() -> None:
    """render_world works without render_underneath callback."""
    controller = get_controller_with_player_and_map()

    controller.picker_mode.start(on_select=lambda r: None)

    # Should not raise
    controller.picker_mode.render_world()


def test_callbacks_cleared_on_exit() -> None:
    """Callbacks are cleared when picker exits to avoid holding references."""
    controller = get_controller_with_player_and_map()
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
    event = tcod.event.KeyDown(0, tcod.event.KeySym.ESCAPE, 0)
    controller.picker_mode.handle_input(event)

    # All callbacks should be None
    assert controller.picker_mode._on_select is None
    assert controller.picker_mode._on_cancel is None
    assert controller.picker_mode._valid_filter is None
    assert controller.picker_mode._render_underneath is None


def test_picker_auto_pushed_by_combat_mode() -> None:
    """CombatMode automatically pushes PickerMode for target selection.

    This is the new behavior where CombatMode delegates click-to-select
    to PickerMode.
    """
    controller = get_controller_with_player_and_map()

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
    event = tcod.event.KeyDown(0, tcod.event.KeySym.ESCAPE, 0)
    controller.picker_mode.handle_input(event)

    # Should be back to explore mode (combat was also popped)
    assert controller.active_mode is controller.explore_mode
    assert len(controller.mode_stack) == 1


def test_combat_mode_repushes_picker_after_selection() -> None:
    """After selecting a target, CombatMode re-pushes PickerMode.

    This keeps the player in combat mode for subsequent attacks.
    """
    controller = get_controller_with_player_and_map()
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


def test_inactive_picker_returns_false() -> None:
    """handle_input returns False when picker is not active."""
    controller = get_controller_with_player_and_map()

    # Picker is not started, so not active
    assert controller.picker_mode.active is False

    event = tcod.event.KeyDown(0, tcod.event.KeySym.ESCAPE, 0)
    result = controller.picker_mode.handle_input(event)

    assert result is False


def test_render_world_skipped_when_inactive() -> None:
    """render_world does nothing when picker is not active."""
    controller = get_controller_with_player_and_map()
    render_underneath = MagicMock()

    # Set callback but don't start picker
    controller.picker_mode._render_underneath = render_underneath
    controller.picker_mode.render_world()

    # Callback should not have been called
    render_underneath.assert_not_called()


def test_combat_mode_with_no_valid_targets() -> None:
    """Combat mode works when there are no enemies in range.

    Picker should still be pushed when entering combat mode, even if there
    are no valid targets. Cancelling should exit cleanly back to explore mode.
    """
    controller = get_controller_with_player_and_map()

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
    event = tcod.event.KeyDown(0, tcod.event.KeySym.ESCAPE, 0)
    controller.picker_mode.handle_input(event)

    # Back to explore mode
    assert controller.active_mode is controller.explore_mode
    assert len(controller.mode_stack) == 1


def test_click_outside_map_bounds_falls_through() -> None:
    """Click outside the game map bounds falls through to modes below.

    When a click is outside the map area (e.g., on UI elements like the
    equipment view), the event should NOT be consumed. This allows UI
    handlers in modes below to process the click - for example, clicking
    the active equipment slot to toggle combat mode.
    """
    controller = get_controller_with_player_and_map()
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
    event = tcod.event.MouseButtonDown(
        (100, 100), (10, 10), tcod.event.MouseButton.LEFT
    )

    result = controller.picker_mode.handle_input(event)

    # Event falls through (not consumed) so UI handlers can process it
    assert result is False
    # No callbacks are invoked (click was outside map)
    on_select.assert_not_called()
    on_cancel.assert_not_called()
    # Picker is still active (didn't pop)
    assert controller.picker_mode in controller.mode_stack

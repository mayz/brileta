"""Tests for mode layering behavior.

CombatMode layers on top of ExploreMode, falling back to ExploreMode
for common functionality like inventory access and movement.
"""

from __future__ import annotations

import tcod.event

from catley.input_handler import Keys
from tests.helpers import get_controller_with_player_and_map


def test_combat_mode_falls_back_to_explore_for_inventory() -> None:
    """I key opens inventory while in combat mode via stack fallback.

    CombatMode doesn't handle I key directly, so it returns False
    and the input falls through the stack to ExploreMode.

    Note: PickerMode blocks most input, so we pop it to test CombatMode directly.
    """
    controller = get_controller_with_player_and_map()
    controller.enter_combat_mode()

    # Pop picker so we can test CombatMode directly
    controller.pop_mode()  # Pop picker
    assert controller.active_mode is controller.combat_mode

    event = tcod.event.KeyDown(0, Keys.KEY_I, 0)

    # CombatMode returns False for unhandled input
    combat_result = controller.combat_mode.handle_input(event)
    assert combat_result is False

    # ExploreMode handles the inventory key
    explore_result = controller.explore_mode.handle_input(event)
    assert explore_result is True


def test_combat_mode_handles_escape_directly() -> None:
    """Escape key exits combat mode (handled by CombatMode, not fallback).

    Note: PickerMode blocks most input, so we pop it to test CombatMode directly.
    """
    controller = get_controller_with_player_and_map()
    controller.enter_combat_mode()

    # Pop picker so we can test CombatMode directly
    controller.pop_mode()  # Pop picker
    assert controller.active_mode is controller.combat_mode

    event = tcod.event.KeyDown(0, tcod.event.KeySym.ESCAPE, 0)
    result = controller.combat_mode.handle_input(event)

    # Event handled by CombatMode directly
    assert result is True
    # Should have exited combat mode
    assert controller.active_mode is controller.explore_mode


def test_combat_mode_handles_tab_directly() -> None:
    """Tab key cycles targets (handled by CombatMode, not fallback).

    Note: PickerMode blocks most input, so we pop it to test CombatMode directly.
    """
    controller = get_controller_with_player_and_map()
    controller.enter_combat_mode()

    # Pop picker so we can test CombatMode directly
    controller.pop_mode()  # Pop picker
    assert controller.active_mode is controller.combat_mode

    event = tcod.event.KeyDown(0, tcod.event.KeySym.TAB, 0)
    result = controller.combat_mode.handle_input(event)

    # Event handled by CombatMode directly (even without valid targets)
    assert result is True


def test_controller_updates_all_modes_in_stack() -> None:
    """Controller.process_player_input() calls update() on all modes.

    This allows ExploreMode to generate movement intents even when
    CombatMode or PickerMode is on top of the stack.
    """
    from unittest.mock import MagicMock, patch

    controller = get_controller_with_player_and_map()
    controller.enter_combat_mode()

    # Stack is now [Explore, Combat, Picker]
    assert len(controller.mode_stack) == 3

    # Add a movement key to explore mode's tracking
    controller.explore_mode.movement_keys.add(tcod.event.KeySym.UP)

    # Ensure input_handler is set (test helper may not set it)
    if controller.input_handler is None:
        controller.input_handler = MagicMock()

    # Mock has_interactive_overlays to return False (no menus open)
    with patch.object(
        controller.overlay_system, "has_interactive_overlays", return_value=False
    ):
        controller.process_player_input()

    # Verify explore mode received the update (timing state changed)
    # even though PickerMode is on top of the stack
    assert controller.explore_mode.move_generator.is_first_move_of_burst is False


def test_combat_mode_tracks_movement_via_fallback() -> None:
    """Movement key events fall through to ExploreMode via stack.

    Note: PickerMode blocks most input, so we pop it to test CombatMode directly.
    """
    controller = get_controller_with_player_and_map()
    controller.enter_combat_mode()

    # Pop picker so we can test CombatMode directly
    controller.pop_mode()  # Pop picker
    assert controller.active_mode is controller.combat_mode

    # Press movement key while in combat mode
    event = tcod.event.KeyDown(0, tcod.event.KeySym.UP, 0)

    # CombatMode doesn't handle movement keys
    combat_result = controller.combat_mode.handle_input(event)
    assert combat_result is False

    # ExploreMode handles it (via stack fallback in production)
    explore_result = controller.explore_mode.handle_input(event)
    assert explore_result is True
    assert tcod.event.KeySym.UP in controller.explore_mode.movement_keys


def test_combat_mode_releases_movement_via_fallback() -> None:
    """Movement key releases fall through to ExploreMode via stack.

    Note: PickerMode blocks most input, so we pop it to test CombatMode directly.
    """
    controller = get_controller_with_player_and_map()
    controller.enter_combat_mode()

    # Pop picker so we can test CombatMode directly
    controller.pop_mode()  # Pop picker
    assert controller.active_mode is controller.combat_mode

    # First press the key
    controller.explore_mode.movement_keys.add(tcod.event.KeySym.UP)

    # Release it
    event = tcod.event.KeyUp(0, tcod.event.KeySym.UP, 0)

    # CombatMode doesn't handle movement key releases
    combat_result = controller.combat_mode.handle_input(event)
    assert combat_result is False

    # ExploreMode handles it (via stack fallback in production)
    explore_result = controller.explore_mode.handle_input(event)
    assert explore_result is True
    assert tcod.event.KeySym.UP not in controller.explore_mode.movement_keys


def test_overlay_consumes_escape_before_mode() -> None:
    """When overlay is active, it handles Escape before mode sees it.

    This tests the fix for the bug where pressing Escape with a menu open
    would exit CombatMode instead of closing the menu.
    """
    from unittest.mock import MagicMock

    from catley.input_handler import InputHandler

    controller = get_controller_with_player_and_map()

    # The test helper patches InputHandler to None, so create it manually.
    # InputHandler needs app, controller, and frame_manager with graphics/cursor.
    mock_app = MagicMock()
    input_handler = InputHandler(mock_app, controller)
    controller.input_handler = input_handler

    controller.enter_combat_mode()

    # Track what each handler received
    overlay_received_escape: list[bool] = []
    mode_received_escape: list[bool] = []

    original_overlay_handle = controller.overlay_system.handle_input
    original_mode_handle = controller.combat_mode.handle_input

    def mock_overlay_handle(event: tcod.event.Event) -> bool:
        if (
            isinstance(event, tcod.event.KeyDown)
            and event.sym == tcod.event.KeySym.ESCAPE
        ):
            overlay_received_escape.append(True)
            return True  # Overlay consumes Escape
        return original_overlay_handle(event)

    def mock_mode_handle(event: tcod.event.Event) -> bool:
        if (
            isinstance(event, tcod.event.KeyDown)
            and event.sym == tcod.event.KeySym.ESCAPE
        ):
            mode_received_escape.append(True)
        return original_mode_handle(event)

    controller.overlay_system.handle_input = mock_overlay_handle  # type: ignore[method-assign]
    controller.combat_mode.handle_input = mock_mode_handle  # type: ignore[method-assign]

    # Dispatch Escape through InputHandler
    escape_event = tcod.event.KeyDown(0, tcod.event.KeySym.ESCAPE, 0)
    input_handler.dispatch(escape_event)

    # Overlay should have received and consumed Escape
    assert len(overlay_received_escape) == 1
    # Mode should NOT have received Escape (overlay consumed it)
    assert len(mode_received_escape) == 0
    # CombatMode should still be active
    assert controller.combat_mode.active is True


def test_overlay_consumes_q_before_quit() -> None:
    """When overlay is active, it handles Q before quit command runs.

    This tests the fix for the bug where pressing Q with a menu open
    would quit the game instead of closing the menu.
    """
    from unittest.mock import MagicMock

    from catley.input_handler import InputHandler

    controller = get_controller_with_player_and_map()

    # Create InputHandler with a mock app so we can track quit calls
    mock_app = MagicMock()
    input_handler = InputHandler(mock_app, controller)
    controller.input_handler = input_handler

    # Track what the overlay received
    overlay_received_q: list[bool] = []

    original_overlay_handle = controller.overlay_system.handle_input

    def mock_overlay_handle(event: tcod.event.Event) -> bool:
        if isinstance(event, tcod.event.KeyDown) and event.sym == Keys.KEY_Q:
            overlay_received_q.append(True)
            return True  # Overlay consumes Q (simulating menu close)
        return original_overlay_handle(event)

    controller.overlay_system.handle_input = mock_overlay_handle  # type: ignore[method-assign]

    # Dispatch Q through InputHandler
    q_event = tcod.event.KeyDown(0, Keys.KEY_Q, 0)
    input_handler.dispatch(q_event)

    # Overlay should have received and consumed Q
    assert len(overlay_received_q) == 1
    # App's quit method should NOT have been called (overlay consumed the event)
    # QuitUICommand calls app._exit_backend(), so check that wasn't called
    mock_app._exit_backend.assert_not_called()

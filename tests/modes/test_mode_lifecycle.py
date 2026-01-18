"""Tests for mode lifecycle management.

Tests verify that the Controller always has an active mode,
starts in ExploreMode, and properly calls _exit()/enter() on transitions.
"""

from __future__ import annotations

import tcod.event

from catley.modes.explore import ExploreMode
from tests.helpers import get_controller_with_player_and_map


def test_active_mode_never_none() -> None:
    """Controller always has an active mode after initialization."""
    controller = get_controller_with_player_and_map()
    assert controller.active_mode is not None


def test_active_mode_starts_as_explore_mode() -> None:
    """Controller starts in ExploreMode."""
    controller = get_controller_with_player_and_map()
    assert controller.active_mode is controller.explore_mode
    assert isinstance(controller.active_mode, ExploreMode)


def test_explore_mode_is_active_on_init() -> None:
    """ExploreMode's active flag is True after controller initialization."""
    controller = get_controller_with_player_and_map()
    assert controller.explore_mode.active is True


def test_targeting_mode_is_inactive_on_init() -> None:
    """TargetingMode's active flag is False after controller initialization."""
    controller = get_controller_with_player_and_map()
    assert controller.targeting_mode.active is False


def test_mode_transition_calls_exit_and_enter() -> None:
    """Mode transitions properly call _exit() and enter().

    When transitioning from ExploreMode to TargetingMode:
    - ExploreMode's active should become False
    - TargetingMode's active should become True
    - active_mode should point to TargetingMode
    """
    controller = get_controller_with_player_and_map()

    # Verify initial state
    assert controller.explore_mode.active is True
    assert controller.targeting_mode.active is False
    assert controller.active_mode is controller.explore_mode

    # Transition to targeting mode
    controller.enter_targeting_mode()

    # Verify new state
    assert controller.explore_mode.active is False
    assert controller.targeting_mode.active is True
    assert controller.active_mode is controller.targeting_mode


def test_exit_targeting_mode_returns_to_explore() -> None:
    """Exiting targeting mode returns to explore mode."""
    controller = get_controller_with_player_and_map()

    # Enter and exit targeting mode
    controller.enter_targeting_mode()
    controller.exit_targeting_mode()

    # Should be back in explore mode
    assert controller.explore_mode.active is True
    assert controller.targeting_mode.active is False
    assert controller.active_mode is controller.explore_mode


def test_reenter_targeting_mode() -> None:
    """Can enter targeting mode again after exiting."""
    controller = get_controller_with_player_and_map()

    # First round
    controller.enter_targeting_mode()
    controller.exit_targeting_mode()

    # Second round
    controller.enter_targeting_mode()
    assert controller.targeting_mode.active is True
    assert controller.active_mode is controller.targeting_mode


def test_transition_to_same_mode_is_noop() -> None:
    """Transitioning to the already-active mode is a no-op.

    This tests the guard in transition_to_mode that prevents redundant
    _exit()/enter() calls when already in the target mode.
    """
    controller = get_controller_with_player_and_map()

    # Start in explore mode
    assert controller.active_mode is controller.explore_mode
    assert controller.explore_mode.active is True

    # Track if _exit was called by checking movement_keys behavior
    # If _exit were called, movement_keys would be cleared
    test_key = tcod.event.KeySym.UP
    controller.explore_mode.movement_keys.add(test_key)

    # Transition to the same mode (explore -> explore)
    controller.transition_to_mode(controller.explore_mode)

    # Should still be in explore mode with state preserved
    assert controller.active_mode is controller.explore_mode
    assert controller.explore_mode.active is True
    # Movement keys should NOT have been cleared (no _exit call)
    assert test_key in controller.explore_mode.movement_keys


def test_transition_to_same_targeting_mode_is_noop() -> None:
    """Transitioning to targeting mode when already there is a no-op."""
    controller = get_controller_with_player_and_map()

    # Enter targeting mode
    controller.enter_targeting_mode()
    assert controller.active_mode is controller.targeting_mode

    # Try to enter again via transition_to_mode
    controller.transition_to_mode(controller.targeting_mode)

    # Should still be in targeting mode, explore should still be inactive
    assert controller.active_mode is controller.targeting_mode
    assert controller.targeting_mode.active is True
    assert controller.explore_mode.active is False

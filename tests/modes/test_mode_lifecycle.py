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


def test_combat_mode_is_inactive_on_init() -> None:
    """CombatMode's active flag is False after controller initialization."""
    controller = get_controller_with_player_and_map()
    assert controller.combat_mode.active is False


def test_mode_transition_calls_exit_and_enter() -> None:
    """Mode transitions properly call _exit() and enter().

    When transitioning from ExploreMode to CombatMode:
    - ExploreMode's active should become False
    - CombatMode's active should become True
    - active_mode should point to CombatMode
    """
    controller = get_controller_with_player_and_map()

    # Verify initial state
    assert controller.explore_mode.active is True
    assert controller.combat_mode.active is False
    assert controller.active_mode is controller.explore_mode

    # Transition to combat mode
    controller.enter_combat_mode()

    # Verify new state
    assert controller.explore_mode.active is False
    assert controller.combat_mode.active is True
    assert controller.active_mode is controller.combat_mode


def test_exit_combat_mode_returns_to_explore() -> None:
    """Exiting combat mode returns to explore mode."""
    controller = get_controller_with_player_and_map()

    # Enter and exit combat mode
    controller.enter_combat_mode()
    controller.exit_combat_mode()

    # Should be back in explore mode
    assert controller.explore_mode.active is True
    assert controller.combat_mode.active is False
    assert controller.active_mode is controller.explore_mode


def test_reenter_combat_mode() -> None:
    """Can enter combat mode again after exiting."""
    controller = get_controller_with_player_and_map()

    # First round
    controller.enter_combat_mode()
    controller.exit_combat_mode()

    # Second round
    controller.enter_combat_mode()
    assert controller.combat_mode.active is True
    assert controller.active_mode is controller.combat_mode


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


def test_transition_to_same_combat_mode_is_noop() -> None:
    """Transitioning to combat mode when already there is a no-op."""
    controller = get_controller_with_player_and_map()

    # Enter combat mode
    controller.enter_combat_mode()
    assert controller.active_mode is controller.combat_mode

    # Try to enter again via transition_to_mode
    controller.transition_to_mode(controller.combat_mode)

    # Should still be in combat mode, explore should still be inactive
    assert controller.active_mode is controller.combat_mode
    assert controller.combat_mode.active is True
    assert controller.explore_mode.active is False

"""Tests for mode lifecycle management.

Tests verify that the Controller always has an active mode,
starts in ExploreMode, and properly calls _exit()/enter() on transitions.
"""

from __future__ import annotations

from catley import input_events
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
    """Mode transitions properly call enter() on the new mode.

    When transitioning from ExploreMode to CombatMode:
    - ExploreMode remains active (it's the base of the stack)
    - CombatMode's active should become True
    - CombatMode pushes PickerMode, so active_mode is PickerMode
    """
    controller = get_controller_with_player_and_map()

    # Verify initial state
    assert controller.explore_mode.active is True
    assert controller.combat_mode.active is False
    assert controller.active_mode is controller.explore_mode

    # Transition to combat mode
    controller.enter_combat_mode()

    # Verify new state - CombatMode pushes PickerMode for target selection
    assert controller.explore_mode.active is True  # Still active as base of stack
    assert controller.combat_mode.active is True
    assert controller.picker_mode.active is True  # Picker is pushed on top
    # Active mode is picker (top of stack), not combat
    assert controller.active_mode is controller.picker_mode
    assert controller.combat_mode in controller.mode_stack


def test_exit_combat_mode_returns_to_explore() -> None:
    """Exiting combat mode pops both PickerMode and CombatMode.

    CombatMode pushes PickerMode on entry. When exiting combat mode,
    we need to pop both (done via transition_to_mode's pop then no-push
    when ExploreMode is already at base).
    """
    controller = get_controller_with_player_and_map()

    # Enter and exit combat mode
    controller.enter_combat_mode()
    # Stack is now [Explore, Combat, Picker]
    assert len(controller.mode_stack) == 3

    controller.exit_combat_mode()

    # Both PickerMode and CombatMode should be popped
    assert controller.explore_mode.active is True
    assert controller.combat_mode.active is False  # _exit() was called
    assert controller.picker_mode.active is False  # _exit() was called
    assert controller.active_mode is controller.explore_mode
    assert len(controller.mode_stack) == 1
    assert controller.mode_stack[0] is controller.explore_mode


def test_reenter_combat_mode() -> None:
    """Can enter combat mode again after exiting."""
    controller = get_controller_with_player_and_map()

    # First round
    controller.enter_combat_mode()
    controller.exit_combat_mode()

    # Second round
    controller.enter_combat_mode()
    assert controller.combat_mode.active is True
    assert controller.picker_mode.active is True  # Picker pushed on top
    assert controller.combat_mode in controller.mode_stack


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
    test_key = input_events.KeySym.UP
    controller.explore_mode.movement_keys.add(test_key)

    # Transition to the same mode (explore -> explore)
    controller.transition_to_mode(controller.explore_mode)

    # Should still be in explore mode with state preserved
    assert controller.active_mode is controller.explore_mode
    assert controller.explore_mode.active is True
    # Movement keys should NOT have been cleared (no _exit call)
    assert test_key in controller.explore_mode.movement_keys


def test_transition_to_same_combat_mode_is_noop() -> None:
    """Transitioning to combat mode when already there pops to combat.

    With the new stack architecture, entering combat pushes [Combat, Picker].
    Transitioning to combat again should pop picker and leave combat on top.
    """
    controller = get_controller_with_player_and_map()

    # Enter combat mode - stack is [Explore, Combat, Picker]
    controller.enter_combat_mode()
    assert controller.combat_mode.active is True
    assert controller.picker_mode.active is True
    assert len(controller.mode_stack) == 3

    # Try to enter again via transition_to_mode
    # This should pop picker, then see combat is already active
    controller.transition_to_mode(controller.combat_mode)

    # Should have popped picker, combat is now on top
    assert controller.active_mode is controller.combat_mode
    assert controller.combat_mode.active is True
    assert controller.explore_mode.active is True
    assert controller.picker_mode.active is False  # Was popped
    assert len(controller.mode_stack) == 2

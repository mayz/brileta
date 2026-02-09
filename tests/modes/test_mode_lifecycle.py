"""Tests for mode lifecycle management.

Tests verify that the Controller always has an active mode,
starts in ExploreMode, and properly calls _exit()/enter() on transitions.
"""

from __future__ import annotations

from brileta import input_events
from brileta.controller import Controller
from brileta.modes.explore import ExploreMode


def test_active_mode_never_none(controller: Controller) -> None:
    """Controller always has an active mode after initialization."""
    assert controller.active_mode is not None


def test_active_mode_starts_as_explore_mode(controller: Controller) -> None:
    """Controller starts in ExploreMode."""
    assert controller.active_mode is controller.explore_mode
    assert isinstance(controller.active_mode, ExploreMode)


def test_explore_mode_is_active_on_init(controller: Controller) -> None:
    """ExploreMode's active flag is True after controller initialization."""
    assert controller.explore_mode.active is True


def test_combat_mode_is_inactive_on_init(controller: Controller) -> None:
    """CombatMode's active flag is False after controller initialization."""
    assert controller.combat_mode.active is False


def test_mode_transition_calls_exit_and_enter(controller: Controller) -> None:
    """Mode transitions properly call enter() on the new mode.

    When transitioning from ExploreMode to CombatMode:
    - ExploreMode remains active (it's the base of the stack)
    - CombatMode's active should become True
    - CombatMode pushes PickerMode, so active_mode is PickerMode
    """
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


def test_exit_combat_mode_returns_to_explore(controller: Controller) -> None:
    """Exiting combat mode pops both PickerMode and CombatMode.

    CombatMode pushes PickerMode on entry. When exiting combat mode,
    we need to pop both (done via transition_to_mode's pop then no-push
    when ExploreMode is already at base).
    """
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


def test_reenter_combat_mode(controller: Controller) -> None:
    """Can enter combat mode again after exiting."""
    # First round
    controller.enter_combat_mode()
    controller.exit_combat_mode()

    # Second round
    controller.enter_combat_mode()
    assert controller.combat_mode.active is True
    assert controller.picker_mode.active is True  # Picker pushed on top
    assert controller.combat_mode in controller.mode_stack


def test_transition_to_same_mode_is_noop(controller: Controller) -> None:
    """Transitioning to the already-active mode is a no-op.

    This tests the guard in transition_to_mode that prevents redundant
    _exit()/enter() calls when already in the target mode.
    """
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


def test_transition_to_same_combat_mode_is_noop(controller: Controller) -> None:
    """Transitioning to combat mode when already there pops to combat.

    With the new stack architecture, entering combat pushes [Combat, Picker].
    Transitioning to combat again should pop picker and leave combat on top.
    """
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

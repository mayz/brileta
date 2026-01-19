"""Tests for mode stack behavior.

The mode stack allows modes to layer on top of each other. ExploreMode
is always at the base. Other modes push onto the stack and pop when done.
Input flows top-to-bottom until handled.
"""

from __future__ import annotations

import pytest
import tcod.event

from catley.input_handler import Keys
from catley.modes.base import Mode
from tests.helpers import get_controller_with_player_and_map


def test_mode_stack_initialized_with_explore_mode() -> None:
    """Mode stack starts with ExploreMode as the only element."""
    controller = get_controller_with_player_and_map()

    assert len(controller.mode_stack) == 1
    assert controller.mode_stack[0] is controller.explore_mode
    assert controller.active_mode is controller.explore_mode


def test_push_mode_adds_to_stack() -> None:
    """push_mode() adds the mode to the stack and calls enter().

    Note: CombatMode.enter() now pushes PickerMode, so the stack ends up
    being [Explore, Combat, Picker] and active_mode is PickerMode.
    """
    controller = get_controller_with_player_and_map()

    assert len(controller.mode_stack) == 1
    assert controller.combat_mode.active is False

    controller.push_mode(controller.combat_mode)

    # CombatMode.enter() pushes PickerMode on top
    assert len(controller.mode_stack) == 3
    assert controller.mode_stack[1] is controller.combat_mode
    assert controller.mode_stack[2] is controller.picker_mode
    assert controller.combat_mode.active is True
    assert controller.picker_mode.active is True
    assert controller.active_mode is controller.picker_mode


def test_pop_mode_removes_from_stack() -> None:
    """pop_mode() removes top mode and calls _exit()."""
    controller = get_controller_with_player_and_map()
    controller.push_mode(controller.combat_mode)

    # After pushing combat mode, picker is on top: [Explore, Combat, Picker]
    assert controller.combat_mode.active is True
    assert controller.picker_mode.active is True
    assert len(controller.mode_stack) == 3

    # Pop picker first
    controller.pop_mode()
    assert len(controller.mode_stack) == 2
    assert controller.picker_mode.active is False
    assert controller.active_mode is controller.combat_mode

    # Pop combat
    controller.pop_mode()
    assert len(controller.mode_stack) == 1
    assert controller.combat_mode.active is False
    assert controller.active_mode is controller.explore_mode


def test_cannot_pop_last_mode() -> None:
    """Cannot pop ExploreMode - it's always present as the base."""
    controller = get_controller_with_player_and_map()

    assert len(controller.mode_stack) == 1

    # Try to pop the last mode - should be a no-op
    controller.pop_mode()

    assert len(controller.mode_stack) == 1
    assert controller.active_mode is controller.explore_mode
    assert controller.explore_mode.active is True


def test_active_mode_property_returns_top_of_stack() -> None:
    """active_mode property returns the mode at the top of the stack."""
    controller = get_controller_with_player_and_map()

    assert controller.active_mode is controller.explore_mode

    # Pushing combat mode also pushes picker on top
    controller.push_mode(controller.combat_mode)
    assert controller.active_mode is controller.picker_mode

    # Pop picker
    controller.pop_mode()
    assert controller.active_mode is controller.combat_mode

    # Pop combat
    controller.pop_mode()
    assert controller.active_mode is controller.explore_mode


def test_input_falls_through_stack() -> None:
    """Input not handled by top mode falls through to modes below.

    PickerMode blocks most input, but CombatMode doesn't handle 'I' for
    inventory - it falls through to ExploreMode.
    """
    controller = get_controller_with_player_and_map()
    controller.enter_combat_mode()

    # Pop picker so we can test CombatMode directly
    controller.pop_mode()  # Pop picker
    assert controller.active_mode is controller.combat_mode

    # I key is not handled by CombatMode, should fall through to ExploreMode
    event = tcod.event.KeyDown(0, Keys.KEY_I, 0)
    result = controller.combat_mode.handle_input(event)

    # CombatMode should return False for unhandled input
    assert result is False

    # ExploreMode handles it
    result = controller.explore_mode.handle_input(event)
    assert result is True


def test_movement_keys_handled_through_stack() -> None:
    """Movement keys in CombatMode fall through to ExploreMode."""
    controller = get_controller_with_player_and_map()
    controller.enter_combat_mode()

    # Pop picker so we can test CombatMode directly
    controller.pop_mode()  # Pop picker
    assert controller.active_mode is controller.combat_mode

    # Movement key should fall through to ExploreMode
    key_down = tcod.event.KeyDown(0, tcod.event.KeySym.UP, 0)
    combat_result = controller.combat_mode.handle_input(key_down)
    assert combat_result is False  # CombatMode doesn't handle movement

    explore_result = controller.explore_mode.handle_input(key_down)
    assert explore_result is True  # ExploreMode tracks it
    assert tcod.event.KeySym.UP in controller.explore_mode.movement_keys


def test_combat_mode_consumes_escape() -> None:
    """CombatMode consumes Escape and pops itself from the stack.

    When testing CombatMode directly (with picker popped), Escape pops combat.
    """
    controller = get_controller_with_player_and_map()
    controller.enter_combat_mode()

    # Pop picker so we can test CombatMode directly
    controller.pop_mode()  # Pop picker
    assert controller.active_mode is controller.combat_mode

    event = tcod.event.KeyDown(0, tcod.event.KeySym.ESCAPE, 0)
    result = controller.combat_mode.handle_input(event)

    assert result is True  # Escape was consumed
    assert controller.active_mode is controller.explore_mode
    assert len(controller.mode_stack) == 1


def test_combat_mode_consumes_t_key() -> None:
    """CombatMode consumes T key and pops itself from the stack."""
    controller = get_controller_with_player_and_map()
    controller.enter_combat_mode()

    # Pop picker so we can test CombatMode directly
    controller.pop_mode()  # Pop picker
    assert controller.active_mode is controller.combat_mode

    event = tcod.event.KeyDown(0, Keys.KEY_T, 0)
    result = controller.combat_mode.handle_input(event)

    assert result is True
    assert controller.active_mode is controller.explore_mode


def test_mode_stack_preserves_explore_mode_state() -> None:
    """ExploreMode state persists when CombatMode is pushed on top.

    Movement keys are tracked by ExploreMode. When CombatMode is pushed,
    those keys should still be tracked (ExploreMode is still in the stack).
    """
    controller = get_controller_with_player_and_map()

    # Start tracking a movement key in ExploreMode
    controller.explore_mode.movement_keys.add(tcod.event.KeySym.UP)
    assert tcod.event.KeySym.UP in controller.explore_mode.movement_keys

    # Enter combat mode
    controller.enter_combat_mode()

    # Movement key should still be tracked
    assert tcod.event.KeySym.UP in controller.explore_mode.movement_keys

    # Exit combat mode
    controller.exit_combat_mode()

    # Still tracked
    assert tcod.event.KeySym.UP in controller.explore_mode.movement_keys


class MockPickerMode(Mode):
    """A mock picker mode for testing stack behavior."""

    def __init__(self, controller):
        super().__init__(controller)
        self.enter_called = False
        self.exit_called = False

    def enter(self) -> None:
        super().enter()
        self.enter_called = True

    def _exit(self) -> None:
        self.exit_called = True
        super()._exit()

    def handle_input(self, event: tcod.event.Event) -> bool:
        # Only handle Escape
        if (
            isinstance(event, tcod.event.KeyDown)
            and event.sym == tcod.event.KeySym.ESCAPE
        ):
            self.controller.pop_mode()
            return True
        return False


def test_four_level_stack_with_mock_mode() -> None:
    """Can have four modes in the stack: Explore -> Combat -> Picker -> Mock.

    CombatMode now auto-pushes PickerMode, so pushing a mock on top creates
    a 4-level stack.
    """
    controller = get_controller_with_player_and_map()
    mock_mode = MockPickerMode(controller)

    controller.enter_combat_mode()
    # Stack is now [Explore, Combat, Picker]
    assert len(controller.mode_stack) == 3

    controller.push_mode(mock_mode)

    assert len(controller.mode_stack) == 4
    assert controller.mode_stack[0] is controller.explore_mode
    assert controller.mode_stack[1] is controller.combat_mode
    assert controller.mode_stack[2] is controller.picker_mode
    assert controller.mode_stack[3] is mock_mode
    assert controller.active_mode is mock_mode

    # Pop mock
    controller.pop_mode()
    assert controller.active_mode is controller.picker_mode
    assert len(controller.mode_stack) == 3

    # Pop picker
    controller.pop_mode()
    assert controller.active_mode is controller.combat_mode
    assert len(controller.mode_stack) == 2

    # Pop combat
    controller.pop_mode()
    assert controller.active_mode is controller.explore_mode
    assert len(controller.mode_stack) == 1


def test_input_flows_through_four_level_stack() -> None:
    """Input flows through all four levels until handled.

    CombatMode auto-pushes PickerMode, so with MockPickerMode on top
    we have a 4-level stack.
    """
    controller = get_controller_with_player_and_map()
    mock_mode = MockPickerMode(controller)

    controller.enter_combat_mode()
    # Stack: [Explore, Combat, Picker]
    controller.push_mode(mock_mode)
    # Stack: [Explore, Combat, Picker, Mock]

    # 'I' key is not handled by mock, picker (blocks), combat, should reach explore
    event = tcod.event.KeyDown(0, Keys.KEY_I, 0)

    mock_result = mock_mode.handle_input(event)
    assert mock_result is False  # Mock doesn't handle I

    # Note: Real PickerMode blocks all input (returns True), but for this test
    # we skip it and test CombatMode directly
    combat_result = controller.combat_mode.handle_input(event)
    assert combat_result is False

    explore_result = controller.explore_mode.handle_input(event)
    assert explore_result is True


def test_escape_pops_correct_mode() -> None:
    """Escape in mock mode pops mock, not the modes below."""
    controller = get_controller_with_player_and_map()
    mock_mode = MockPickerMode(controller)

    controller.enter_combat_mode()
    # Stack: [Explore, Combat, Picker]
    controller.push_mode(mock_mode)
    # Stack: [Explore, Combat, Picker, Mock]

    assert controller.active_mode is mock_mode

    # Press Escape - should pop mock, leaving picker on top
    event = tcod.event.KeyDown(0, tcod.event.KeySym.ESCAPE, 0)
    mock_mode.handle_input(event)

    # Mock popped, picker is now on top
    assert controller.active_mode is controller.picker_mode
    assert len(controller.mode_stack) == 3
    assert controller.combat_mode.active is True


def test_push_mode_raises_if_already_in_stack() -> None:
    """push_mode() raises error if mode is already in the stack."""
    controller = get_controller_with_player_and_map()
    controller.push_mode(controller.combat_mode)

    with pytest.raises(RuntimeError, match="already in the stack"):
        controller.push_mode(controller.combat_mode)


def test_push_mode_raises_if_same_type_in_stack() -> None:
    """push_mode() raises error if a mode of the same type is in the stack.

    This prevents accidentally creating a new instance of a mode and pushing
    it when the original instance is already active.
    """
    from catley.modes.combat import CombatMode

    controller = get_controller_with_player_and_map()
    controller.push_mode(controller.combat_mode)

    # Create a new instance of CombatMode
    new_combat_mode = CombatMode(controller)

    with pytest.raises(RuntimeError, match="already in the stack"):
        controller.push_mode(new_combat_mode)


# --- InputHandler Integration Tests ---


class InputTrackingModeA(Mode):
    """A mode that tracks input calls and handles 'a' key."""

    def __init__(self, controller):
        super().__init__(controller)
        self.received_events: list[tcod.event.Event] = []

    def enter(self) -> None:
        super().enter()

    def _exit(self) -> None:
        super()._exit()

    def handle_input(self, event: tcod.event.Event) -> bool:
        self.received_events.append(event)
        return (
            isinstance(event, tcod.event.KeyDown) and event.sym == tcod.event.KeySym.a
        )


class InputTrackingModeB(Mode):
    """A mode that tracks input calls and handles 'b' key."""

    def __init__(self, controller):
        super().__init__(controller)
        self.received_events: list[tcod.event.Event] = []

    def enter(self) -> None:
        super().enter()

    def _exit(self) -> None:
        super()._exit()

    def handle_input(self, event: tcod.event.Event) -> bool:
        self.received_events.append(event)
        return (
            isinstance(event, tcod.event.KeyDown) and event.sym == tcod.event.KeySym.b
        )


class InputTrackingModePassthrough(Mode):
    """A mode that tracks input calls but handles nothing."""

    def __init__(self, controller):
        super().__init__(controller)
        self.received_events: list[tcod.event.Event] = []

    def enter(self) -> None:
        super().enter()

    def _exit(self) -> None:
        super()._exit()

    def handle_input(self, event: tcod.event.Event) -> bool:
        self.received_events.append(event)
        return False


def test_input_handler_dispatches_to_mode_stack_in_reverse_order() -> None:
    """InputHandler.dispatch() iterates mode stack top-to-bottom until handled.

    This is an integration test that verifies the InputHandler correctly
    passes input through the mode stack, starting from the top.
    """
    from unittest.mock import MagicMock

    from catley.input_handler import InputHandler

    controller = get_controller_with_player_and_map()

    # Create mock modes that track which keys they receive
    mode_a = InputTrackingModeA(controller)  # Only handles 'a'
    mode_b = InputTrackingModeB(controller)  # Only handles 'b'

    # Push modes onto stack: [Explore, A, B]
    controller.push_mode(mode_a)
    controller.push_mode(mode_b)

    assert controller.mode_stack == [controller.explore_mode, mode_a, mode_b]

    # Create InputHandler with minimal mocking
    mock_app = MagicMock()
    input_handler = InputHandler(mock_app, controller)

    # Test 1: 'b' key should be handled by mode_b (top of stack)
    event_b = tcod.event.KeyDown(0, tcod.event.KeySym.b, 0)
    input_handler.dispatch(event_b)

    # mode_b received the event and handled it
    assert len(mode_b.received_events) == 1
    assert mode_b.received_events[0] == event_b
    # mode_a should NOT have received it (stopped at mode_b)
    assert len(mode_a.received_events) == 0

    # Test 2: 'a' key falls through mode_b to mode_a
    event_a = tcod.event.KeyDown(0, tcod.event.KeySym.a, 0)
    input_handler.dispatch(event_a)

    # mode_b received it first but didn't handle it
    assert len(mode_b.received_events) == 2
    assert mode_b.received_events[1] == event_a
    # mode_a received it and handled it
    assert len(mode_a.received_events) == 1
    assert mode_a.received_events[0] == event_a


def test_input_handler_input_falls_through_entire_stack() -> None:
    """Input not handled by any mode falls through the entire stack."""
    from unittest.mock import MagicMock

    from catley.input_handler import InputHandler

    controller = get_controller_with_player_and_map()

    # Mode that handles nothing
    mode_passthrough = InputTrackingModePassthrough(controller)
    controller.push_mode(mode_passthrough)

    mock_app = MagicMock()
    input_handler = InputHandler(mock_app, controller)

    # An unhandled key should pass through all modes
    event = tcod.event.KeyDown(0, tcod.event.KeySym.F12, 0)
    input_handler.dispatch(event)

    # passthrough received it but didn't handle
    assert len(mode_passthrough.received_events) == 1

    # ExploreMode should have also received it via handle_input
    # (We can't easily track ExploreMode's received events, but we verify
    # the passthrough mode was called, confirming dispatch order)

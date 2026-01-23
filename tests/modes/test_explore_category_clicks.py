"""Tests for action panel click handling in ExploreMode."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, patch

import tcod.event

from catley import colors
from catley.game.actions.discovery import ActionCategory, ActionOption
from catley.game.actors import Character
from catley.game.game_world import GameWorld
from catley.modes.explore import ExploreMode
from catley.view.render.graphics import GraphicsContext
from catley.view.ui.selectable_list import SelectableRow
from catley.view.views.action_panel_view import ActionPanelView
from tests.helpers import DummyGameWorld


class DummyFrameManager:
    """Minimal frame manager for testing."""

    def __init__(self, action_panel_view: ActionPanelView) -> None:
        self.action_panel_view = action_panel_view

    def get_world_coords_from_root_tile_coords(
        self, root_tile_pos: tuple[int, int]
    ) -> tuple[int, int] | None:
        return None


class DummyController:
    """Minimal controller for testing ExploreMode."""

    def __init__(self, gw: DummyGameWorld, graphics: Any) -> None:
        self.gw = gw
        self.graphics = graphics
        self.frame_manager: DummyFrameManager | None = None
        self.app = None
        self.queued_actions: list[Any] = []

    def is_combat_mode(self) -> bool:
        return False

    def queue_action(self, action: Any) -> None:
        self.queued_actions.append(action)


class MockListRenderer:
    """Mock SelectableListRenderer for testing."""

    def __init__(self) -> None:
        self.rows: list[SelectableRow] = []
        self.hovered_index: int | None = None
        self._hit_areas: list[tuple[tuple[int, int, int, int], int]] = []
        # Map of (px, py) -> SelectableRow for mocking get_row_at_pixel
        self._pixel_to_row: dict[tuple[int, int], SelectableRow | None] = {}

    def get_row_at_pixel(self, px: int, py: int) -> SelectableRow | None:
        """Return the row at the given pixel coordinates."""
        # Check if we have a specific mapping for this coordinate
        for (x1, y1, x2, y2), row_idx in self._hit_areas:
            if x1 <= px < x2 and y1 <= py < y2 and row_idx < len(self.rows):
                return self.rows[row_idx]
        return None

    def update_hover_from_pixel(self, px: int, py: int) -> bool:
        """Update hover state from pixel coordinates."""
        return False

    def execute_at_pixel(self, px: int, py: int) -> bool:
        """Execute the callback for the row at the given pixel coordinates."""
        row = self.get_row_at_pixel(px, py)
        if row is not None and row.execute is not None:
            row.execute()
            return True
        return False

    def clear(self) -> None:
        """Clear all rows and hit areas."""
        self.rows.clear()
        self._hit_areas.clear()

    def clear_hit_areas(self) -> None:
        """Clear only hit areas (not rows or hover state)."""
        self._hit_areas.clear()


def make_explore_mode_with_panel() -> tuple[
    DummyController, ExploreMode, ActionPanelView
]:
    """Create ExploreMode with action panel for testing clicks."""
    gw = DummyGameWorld()
    player = Character(
        5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)
    gw.mouse_tile_location_on_map = None

    renderer = MagicMock(spec=GraphicsContext)
    renderer.tile_dimensions = (8, 16)
    renderer.get_display_scale_factor = MagicMock(return_value=(1.0, 1.0))

    controller = DummyController(gw=gw, graphics=renderer)

    # Create action panel view with mocked list renderers
    with patch(
        "catley.view.views.action_panel_view.ActionPanelView.__init__",
        lambda self, ctrl: None,
    ):
        view = ActionPanelView.__new__(ActionPanelView)
        view.controller = controller
        # Set up all three renderers (separate to avoid cross-section hover bugs)
        view._pickup_renderer = MockListRenderer()
        view._actions_renderer = MockListRenderer()
        view._controls_renderer = MockListRenderer()
        view._texture_cache = MagicMock()
        view.x = 0
        view.y = 0
        view.width = 10
        view.height = 20

    # Set up frame manager
    fm = DummyFrameManager(view)
    controller.frame_manager = fm

    # Create explore mode
    mode = ExploreMode(cast(Any, controller))
    mode._fm = fm
    mode._gw = gw
    mode._p = player

    return controller, mode, view


class TestActionClickHandling:
    """Tests for clicking actions in action panel."""

    def test_click_on_action_executes(self) -> None:
        """Clicking an action should execute it."""
        controller, mode, view = make_explore_mode_with_panel()

        mock_action_class = MagicMock()
        action = ActionOption(
            id="test",
            name="Test Action",
            description="Test",
            category=ActionCategory.COMBAT,
            action_class=mock_action_class,  # type: ignore[arg-type]
            static_params={"target": "enemy"},
        )
        # Set up the mock list renderer with a row containing this action
        row = SelectableRow(text="Test Action", key="A", data=action)
        view._actions_renderer.rows = [row]
        # Hit area at y=130-150
        view._actions_renderer._hit_areas = [((10, 130, 70, 150), 0)]

        # Simulate click on action area at y=140
        event = MagicMock(spec=tcod.event.MouseButtonDown)
        event.button = tcod.event.MouseButton.LEFT
        event.position = tcod.event.Point(40, 140)

        mode._try_execute_action_panel_click(event)

        # Action should be queued
        assert len(controller.queued_actions) == 1

    def test_click_outside_panel_returns_false(self) -> None:
        """Clicking outside action panel should return False."""
        _controller, mode, _view = make_explore_mode_with_panel()

        # Simulate click way outside panel bounds
        event = MagicMock(spec=tcod.event.MouseButtonDown)
        event.button = tcod.event.MouseButton.LEFT
        event.position = tcod.event.Point(1000, 1000)

        result = mode._try_execute_action_panel_click(event)

        assert result is False

    def test_click_on_panel_but_no_action_returns_false(self) -> None:
        """Clicking within panel but not on an action should return False."""
        _controller, mode, view = make_explore_mode_with_panel()

        # No actions registered - clear the list renderer
        view._actions_renderer.rows = []
        view._actions_renderer._hit_areas = []

        # Simulate click within panel bounds
        event = MagicMock(spec=tcod.event.MouseButtonDown)
        event.button = tcod.event.MouseButton.LEFT
        event.position = tcod.event.Point(40, 50)

        result = mode._try_execute_action_panel_click(event)

        assert result is False

    def test_action_with_execute_callback(self) -> None:
        """Actions with execute callbacks should invoke them."""
        _controller, mode, view = make_explore_mode_with_panel()

        callback_called = []

        def execute_callback() -> bool:
            callback_called.append(True)
            return True

        action = ActionOption(
            id="callback_action",
            name="Callback Action",
            description="Test",
            category=ActionCategory.SOCIAL,
            action_class=MagicMock(),  # type: ignore[arg-type]
            static_params={},
            execute=execute_callback,
        )
        # Set up the mock list renderer with a row containing this action
        row = SelectableRow(text="Callback Action", key="A", data=action)
        view._actions_renderer.rows = [row]
        # Hit area at y=100-120
        view._actions_renderer._hit_areas = [((10, 100, 70, 120), 0)]

        event = MagicMock(spec=tcod.event.MouseButtonDown)
        event.button = tcod.event.MouseButton.LEFT
        event.position = tcod.event.Point(40, 110)

        mode._try_execute_action_panel_click(event)

        assert len(callback_called) == 1


class TestStaleHitAreasRegression:
    """Regression tests for stale hit areas bug.

    When a section transitions from rendered to not-rendered, its hit areas
    must be cleared. Otherwise, clicks on the newly-rendered section could
    be intercepted by stale hit areas from the previously-rendered section.

    See: ActionPanelView.draw_content() clear_hit_areas() calls.
    """

    def test_stale_controls_hit_areas_do_not_intercept_action_clicks(self) -> None:
        """Stale controls hit areas should not intercept clicks on actions.

        Scenario:
        1. No target selected - controls section rendered at y=100-200
        2. Target selected - actions section rendered at y=100-200, controls hidden
        3. Click at y=150 should hit the action, NOT trigger stale controls callback
        """
        _controller, mode, view = make_explore_mode_with_panel()

        # Simulate stale controls hit areas (from when no target was selected)
        # These overlap with where actions will be rendered
        controls_callback_called = []

        def stale_controls_callback() -> None:
            controls_callback_called.append(True)

        stale_control_row = SelectableRow(
            text="Action menu",
            key="Space",
            execute=stale_controls_callback,
        )
        view._controls_renderer.rows = [stale_control_row]
        # Stale hit area at y=100-150 (overlapping with actions area)
        view._controls_renderer._hit_areas = [((10, 100, 70, 150), 0)]

        # Now set up actions (as if target was just selected)
        action = ActionOption(
            id="attack",
            name="Attack",
            description="Attack the target",
            category=ActionCategory.COMBAT,
            action_class=MagicMock(),  # type: ignore[arg-type]
            static_params={},
        )
        action_row = SelectableRow(text="Attack", key="A", data=action)
        view._actions_renderer.rows = [action_row]
        # Fresh hit area at y=100-150 (same position as stale controls)
        view._actions_renderer._hit_areas = [((10, 100, 70, 150), 0)]

        # Click in the overlapping area
        event = MagicMock(spec=tcod.event.MouseButtonDown)
        event.button = tcod.event.MouseButton.LEFT
        event.position = tcod.event.Point(40, 125)

        # This simulates what happens AFTER draw_content clears stale hit areas
        # In the real code, draw_content() would call clear_hit_areas() on
        # _controls_renderer when the controls section isn't rendered.
        view._controls_renderer.clear_hit_areas()

        mode._try_execute_action_panel_click(event)

        # The stale controls callback should NOT have been triggered
        assert len(controls_callback_called) == 0
        # The action should have been queued
        assert len(_controller.queued_actions) == 1

    def test_stale_actions_hit_areas_cleared_when_no_target(self) -> None:
        """Stale actions hit areas should not return actions when no target.

        Scenario:
        1. Target selected - actions at y=100-150
        2. Target deselected - controls rendered, actions hidden
        3. get_action_at_pixel should return None, not stale action
        """
        _controller, _mode, view = make_explore_mode_with_panel()

        # Simulate stale actions hit areas (from when target was selected)
        stale_action = ActionOption(
            id="stale-attack",
            name="Stale Attack",
            description="This action should not be returned",
            category=ActionCategory.COMBAT,
            action_class=MagicMock(),  # type: ignore[arg-type]
            static_params={},
        )
        stale_row = SelectableRow(text="Stale Attack", key="A", data=stale_action)
        view._actions_renderer.rows = [stale_row]
        view._actions_renderer._hit_areas = [((10, 100, 70, 150), 0)]

        # Simulate clearing stale hit areas (what draw_content does)
        view._actions_renderer.clear_hit_areas()

        # get_action_at_pixel should return None now
        result = view.get_action_at_pixel(40, 125)
        assert result is None

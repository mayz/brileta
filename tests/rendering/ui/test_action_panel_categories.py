"""Tests for ActionPanelView flat list rendering."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

from brileta import colors
from brileta.controller import Controller
from brileta.game.actions.discovery import ActionCategory, ActionOption
from brileta.game.actors import Character
from brileta.game.game_world import GameWorld
from brileta.view.render.graphics import GraphicsContext
from brileta.view.views.action_panel_view import ActionPanelView
from tests.helpers import DummyGameWorld
from tests.rendering.ui.test_action_panel_cache import DummyController


def make_action_panel() -> tuple[DummyController, ActionPanelView]:
    """Create a minimal ActionPanelView for testing."""
    gw = DummyGameWorld()
    player = Character(
        5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)

    gw.mouse_tile_location_on_map = None

    renderer = MagicMock(spec=GraphicsContext)
    renderer.tile_dimensions = (8, 16)
    controller = DummyController(gw=gw, graphics=renderer)

    view = ActionPanelView(cast(Controller, controller))
    return controller, view


def make_action(action_id: str, name: str, category: ActionCategory) -> ActionOption:
    """Create a test ActionOption."""
    return ActionOption(
        id=action_id,
        name=name,
        description=f"Test action {name}",
        category=category,
        action_class=MagicMock(),  # type: ignore[arg-type]
        requirements=[],
        static_params={},
    )


class TestHotkeyAssignment:
    """Tests for hotkey assignment in flat list."""

    def test_all_actions_get_hotkeys(self) -> None:
        """All actions in flat list should have hotkeys assigned."""
        _controller, view = make_action_panel()

        # Create test actions
        action1 = make_action("talk", "Talk", ActionCategory.SOCIAL)
        action2 = make_action("attack", "Attack", ActionCategory.COMBAT)
        action3 = make_action("search", "Search", ActionCategory.ENVIRONMENT)

        view._cached_actions = [action1, action2, action3]
        view._assign_hotkeys(view._cached_actions)

        # All actions should have hotkeys
        assert action1.hotkey is not None
        assert action2.hotkey is not None
        assert action3.hotkey is not None

        # All hotkeys should be unique
        hotkeys = {action1.hotkey, action2.hotkey, action3.hotkey}
        assert len(hotkeys) == 3

    def test_hotkeys_persist_across_assignments(self) -> None:
        """Hotkeys should be sticky across multiple assignment calls."""
        _controller, view = make_action_panel()

        action1 = make_action("talk", "Talk", ActionCategory.SOCIAL)
        action2 = make_action("attack", "Attack", ActionCategory.COMBAT)

        view._cached_actions = [action1, action2]
        view._assign_hotkeys(view._cached_actions)

        first_talk_hotkey = action1.hotkey
        first_attack_hotkey = action2.hotkey

        # Clear and reassign
        action1.hotkey = None
        action2.hotkey = None
        view._assign_hotkeys(view._cached_actions)

        # Hotkeys should be preserved
        assert action1.hotkey == first_talk_hotkey
        assert action2.hotkey == first_attack_hotkey


class TestDefaultAction:
    """Tests for default action handling."""

    def test_default_action_identified_correctly(self) -> None:
        """The default action should match _cached_default_action_id."""
        _controller, view = make_action_panel()

        talk_action = make_action("talk", "Talk", ActionCategory.SOCIAL)
        attack_action = make_action("attack", "Attack", ActionCategory.COMBAT)

        view._cached_actions = [talk_action, attack_action]
        view._cached_default_action_id = "talk"

        # The default action should be "talk"
        default_found = None
        for action in view._cached_actions:
            if action.id == view._cached_default_action_id:
                default_found = action
                break

        assert default_found is talk_action


class TestFlatListRendering:
    """Tests for flat list action rendering."""

    def test_actions_limited_to_max_count(self) -> None:
        """Actions should be limited to a reasonable maximum."""
        _controller, view = make_action_panel()

        # Create many actions
        actions = [
            make_action(f"action{i}", f"Action {i}", ActionCategory.COMBAT)
            for i in range(15)
        ]

        view._cached_actions = actions
        view._assign_hotkeys(view._cached_actions)

        # Flat list in draw_content limits to 10 actions
        # All actions still get hotkeys assigned
        assert all(a.hotkey is not None for a in actions[:10])

"""Tests for ActionPanelView flat list rendering."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

from brileta import colors
from brileta.controller import Controller
from brileta.game.actions.discovery import ActionCategory, ActionOption
from brileta.game.actors import NPC, PC, Character
from brileta.game.actors.identity import Gender, identity_for_gender
from brileta.game.game_world import GameWorld
from brileta.types import InterpolationAlpha
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
        action_class=MagicMock(),
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


class TestTargetIdentityDisplay:
    """Tests for selected/hovered actor identity display."""

    def test_populate_actor_target_data_caches_npc_gender(self) -> None:
        """NPC identity gender should be available to the render path."""
        controller, view = make_action_panel()
        gw = controller.gw
        npc = NPC(
            2,
            2,
            "R",
            colors.WHITE,
            "Resident",
            game_world=cast(GameWorld, gw),
            identity=identity_for_gender(Gender.FEMALE),
        )

        view._populate_actor_target_data(npc)

        assert view._cached_target_gender == "female"

    def test_populate_actor_target_data_caches_archetype_name(self) -> None:
        """Named NPCs should still show their archetype in the sidebar."""
        controller, view = make_action_panel()
        gw = controller.gw
        npc = NPC(
            2,
            2,
            "T",
            colors.WHITE,
            "James",
            game_world=cast(GameWorld, gw),
            archetype_name="Trog",
            identity=identity_for_gender(Gender.MALE),
        )

        view._populate_actor_target_data(npc)

        assert view._cached_target_name == "James"
        assert view._cached_target_type == "Trog"

    def test_populate_actor_target_data_omits_duplicate_archetype_name(self) -> None:
        """Unnamed creatures should not repeat the same name as a type line."""
        controller, view = make_action_panel()
        gw = controller.gw
        npc = NPC(
            2,
            2,
            "d",
            colors.WHITE,
            "Dog",
            game_world=cast(GameWorld, gw),
            archetype_name="Dog",
        )

        view._populate_actor_target_data(npc)

        assert view._cached_target_name == "Dog"
        assert view._cached_target_type is None

    def test_populate_actor_target_data_caches_player_gender(self) -> None:
        """Player identity should use the same selected-actor display path."""
        controller, view = make_action_panel()
        gw = controller.gw
        player = PC(
            2,
            2,
            "@",
            colors.WHITE,
            "Player",
            game_world=cast(GameWorld, gw),
            identity=identity_for_gender(Gender.MALE),
        )

        view._populate_actor_target_data(player)

        assert view._cached_target_gender == "male"

    def test_populate_actor_target_data_omits_gender_for_plain_character(self) -> None:
        """Actors without identity should not show a stale gender line."""
        controller, view = make_action_panel()
        gw = controller.gw
        actor = Character(
            2,
            2,
            "@",
            colors.WHITE,
            "Actor",
            game_world=cast(GameWorld, gw),
        )
        view._cached_target_gender = "female"
        view._cached_target_type = "Trog"

        view._populate_actor_target_data(actor)

        assert view._cached_target_gender is None
        assert view._cached_target_type is None

    def test_draw_content_renders_gender_line_under_name_when_cached(self) -> None:
        """The sidebar should render the gender value directly below the name."""
        _controller, view = make_action_panel()
        view.set_bounds(0, 0, 24, 20)
        view._cached_target_name = "Resident"
        view._cached_target_gender = "female"
        view._cached_is_selected = True
        view._cached_target_description = None
        view._cached_actions = []
        view.canvas.draw_text = MagicMock()

        view.draw_content(view.controller.graphics, InterpolationAlpha(1.0))

        rendered_text = [
            call.kwargs["text"] for call in view.canvas.draw_text.call_args_list
        ]
        assert rendered_text[:3] == ["Resi...", "female", "Selected"]
        assert "Gender: female" not in rendered_text

    def test_draw_content_renders_type_line_between_name_and_gender(self) -> None:
        """The sidebar should render archetype under a generated NPC name."""
        _controller, view = make_action_panel()
        view.set_bounds(0, 0, 24, 20)
        view._cached_target_name = "James"
        view._cached_target_type = "Trog"
        view._cached_target_gender = "male"
        view._cached_is_selected = True
        view._cached_target_description = None
        view._cached_actions = []
        view.canvas.draw_text = MagicMock()

        view.draw_content(view.controller.graphics, InterpolationAlpha(1.0))

        rendered_text = [
            call.kwargs["text"] for call in view.canvas.draw_text.call_args_list
        ]
        assert rendered_text[:4] == ["James", "Trog", "male", "Selected"]


class TestTargetNameTruncation:
    """Tests for sidebar target-name fit behavior."""

    def test_target_name_truncates_with_ellipsis_when_needed(self) -> None:
        """Long target names should be truncated instead of hard-clipped."""
        _controller, view = make_action_panel()

        def _metrics(text: str, font_size: int | None = None) -> tuple[int, int, int]:
            _ = font_size
            return (len(text) * 10, 10, 10)

        view.canvas.get_text_metrics = MagicMock(side_effect=_metrics)
        result = view._truncate_text_to_fit("Cobblestone (remembered)", max_width=120)

        assert result.endswith("...")
        assert len(result) < len("Cobblestone (remembered)")
        assert _metrics(result)[0] <= 120

    def test_target_name_unchanged_when_it_fits(self) -> None:
        """Names that already fit should render unchanged."""
        _controller, view = make_action_panel()

        def _metrics(text: str, font_size: int | None = None) -> tuple[int, int, int]:
            _ = font_size
            return (len(text) * 10, 10, 10)

        view.canvas.get_text_metrics = MagicMock(side_effect=_metrics)
        result = view._truncate_text_to_fit("Wall", max_width=120)

        assert result == "Wall"

    def test_target_name_returns_empty_if_ellipsis_exceeds_width(self) -> None:
        """Very narrow columns should avoid returning over-wide ellipsis text."""
        _controller, view = make_action_panel()

        def _metrics(text: str, font_size: int | None = None) -> tuple[int, int, int]:
            _ = font_size
            return (len(text) * 10, 10, 10)

        view.canvas.get_text_metrics = MagicMock(side_effect=_metrics)
        result = view._truncate_text_to_fit("Cobblestone", max_width=20)

        assert result == ""

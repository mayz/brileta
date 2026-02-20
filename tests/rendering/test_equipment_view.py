from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

from brileta import colors
from brileta.controller import Controller
from brileta.game.actors import Character
from brileta.game.game_world import GameWorld
from brileta.types import InterpolationAlpha
from brileta.view.render.graphics import GraphicsContext
from brileta.view.views.equipment_view import EquipmentView
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    gw: DummyGameWorld
    graphics: Any = None
    queued_actions: list[Any] = None

    def __post_init__(self) -> None:
        if self.queued_actions is None:
            self.queued_actions = []

    def queue_action(self, action: Any) -> None:
        """Queue an action for later execution."""
        self.queued_actions.append(action)

    def is_combat_mode(self) -> bool:
        """Default to explore mode for equipment view tests."""
        return False


def make_equipment_view() -> tuple[DummyController, Character, EquipmentView]:
    """Create a test equipment view with a player character."""
    gw = DummyGameWorld()
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)

    renderer = MagicMock(spec=GraphicsContext)
    renderer.create_canvas = MagicMock(return_value=MagicMock())
    # PillowImageCanvas requires tile_dimensions to return actual integers
    renderer.tile_dimensions = (16, 16)
    controller = DummyController(gw=gw, graphics=renderer)
    view = EquipmentView(cast(Controller, controller), renderer)

    return controller, player, view


def test_handle_click_on_inactive_slot_switches() -> None:
    """Clicking on an inactive weapon slot should queue a SwitchWeaponIntent."""
    from brileta.game.actions.misc import SwitchWeaponIntent

    controller, player, view = make_equipment_view()

    assert player.inventory.active_slot == 0

    # Simulate the slot pixel bounds that would be set during rendering.
    # With tile_dimensions (16, 16), row 0 center is at pixel 8, row 1 center at 24.
    # Each slot spans one line_height (approx tile_height=16 pixels).
    view._slot_pixel_bounds = {0: (0, 16), 1: (16, 32)}

    # Click row 1 which is slot 1 (inactive)
    result = view.handle_click(1)

    assert result is True
    # Verify correct intent was queued
    assert len(controller.queued_actions) == 1
    action = controller.queued_actions[0]
    assert isinstance(action, SwitchWeaponIntent)
    assert action.slot == 1


def test_handle_click_outside_slots_does_nothing() -> None:
    """Clicking outside the slot bounds should not switch weapons."""
    _controller, player, view = make_equipment_view()

    assert player.inventory.active_slot == 0

    # Simulate the slot pixel bounds (each slot spans 16 pixels)
    view._slot_pixel_bounds = {0: (0, 16), 1: (16, 32)}

    # Click row 2 which is outside the slot bounds (center at pixel 40)
    result = view.handle_click(2)

    assert result is False
    assert player.inventory.active_slot == 0


def test_calculate_min_width_shrinks_when_content_shrinks() -> None:
    """Equipment width should reflect current content, not a grow-only high-water mark."""
    _controller, player, view = make_equipment_view()
    view.tile_dimensions = (16, 16)

    # Start with very short labels ("Fists" in both slots) near the floor width.
    player.inventory.ready_slots[0] = None
    player.inventory.ready_slots[1] = None
    width_short = view.calculate_min_width()

    # Add a long weapon name and verify required width grows.
    player.inventory.ready_slots[0] = SimpleNamespace(
        name="ExtremelyLongExperimentalWeaponName",
        ranged_attack=None,
    )
    width_long = view.calculate_min_width()
    assert width_long > width_short

    # Replace with short content again; width should shrink back.
    player.inventory.ready_slots[0] = None
    width_short_again = view.calculate_min_width()
    assert width_short_again == width_short


def test_slot_text_uses_bracket_indices_prefix() -> None:
    """Equipment rows should use the classic [1]/[2] prefix format."""
    _controller, _player, view = make_equipment_view()

    active_text = view._build_slot_text(
        slot_index=0,
        item=SimpleNamespace(name="Pistol", ranged_attack=None),
        is_active=True,
    )
    inactive_text = view._build_slot_text(slot_index=1, item=None, is_active=False)

    assert active_text.startswith("▶ [1] ")
    assert inactive_text.startswith("  [2] ")


def test_draw_content_adds_top_divider_and_padding() -> None:
    """Equipment rows should render below a divider with a small top inset."""
    _controller, player, view = make_equipment_view()
    view.tile_dimensions = (16, 16)
    view.width = 10
    view.height = 4
    player.inventory.ready_slots[0] = None
    player.inventory.ready_slots[1] = None

    mock_canvas = MagicMock()
    mock_canvas.get_font_metrics.return_value = (8, 4)  # line height = 12
    view.canvas = mock_canvas

    renderer = MagicMock(spec=GraphicsContext)
    renderer.tile_dimensions = (16, 16)
    view.draw_content(renderer, InterpolationAlpha(0.0))

    assert mock_canvas.draw_rect.call_count >= 2
    divider_call = mock_canvas.draw_rect.call_args_list[1]
    assert divider_call.args[:4] == (0, 0, 160, 1)
    assert divider_call.args[4] == colors.DARK_GREY

    first_text_call = mock_canvas.draw_text.call_args_list[0]
    assert first_text_call.args[1] == 25
    assert view._slot_pixel_bounds[0] == (25, 37)


def test_cache_key_changes_when_view_dimensions_change() -> None:
    """Resize should invalidate cached equipment texture to avoid stretch."""
    _controller, _player, view = make_equipment_view()
    view.tile_dimensions = (16, 16)

    view.set_bounds(0, 0, 20, 4)
    key_before = view.get_cache_key()

    view.set_bounds(0, 0, 14, 4)
    key_after = view.get_cache_key()

    assert key_before != key_after

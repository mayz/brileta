from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock

from brileta import colors
from brileta.controller import Controller
from brileta.game.actors import Character
from brileta.game.game_world import GameWorld
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

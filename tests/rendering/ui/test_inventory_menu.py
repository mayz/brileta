from types import SimpleNamespace
from typing import cast

from brileta.controller import Controller
from brileta.game.actors import components
from brileta.game.items.item_types import COMBAT_KNIFE_TYPE, PISTOL_TYPE
from brileta.util.message_log import MessageLog
from brileta.view.ui.dual_pane_menu import DualPaneMenu
from tests.helpers import _make_renderer


def test_dual_pane_menu_equips_to_active_slot() -> None:
    """Test that DualPaneMenu._equip_item equips to the active weapon slot."""
    stats = components.StatsComponent(strength=0)
    inv = components.CharacterInventory(stats)
    pistol = PISTOL_TYPE.create()
    knife = COMBAT_KNIFE_TYPE.create()

    inv.equip_to_slot(pistol, 0)
    inv.add_to_inventory(knife)
    inv.switch_to_slot(1)

    player = SimpleNamespace(inventory=inv)
    controller = cast(
        Controller,
        SimpleNamespace(
            gw=SimpleNamespace(player=player),
            message_log=MessageLog(),
            graphics=_make_renderer(),
        ),
    )

    menu = DualPaneMenu(controller)
    menu._equip_item(knife)

    assert inv.ready_slots[1] == knife
    assert knife not in inv

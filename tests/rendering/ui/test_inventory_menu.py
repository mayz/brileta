from types import SimpleNamespace
from typing import cast

from catley.controller import Controller
from catley.game.actors import components
from catley.game.items.item_types import COMBAT_KNIFE_TYPE, PISTOL_TYPE
from catley.util.message_log import MessageLog
from catley.view.ui.dual_pane_menu import DualPaneMenu
from tests.rendering.backends.test_canvases import _make_renderer


def test_dual_pane_menu_equips_to_active_slot() -> None:
    """Test that DualPaneMenu._use_item equips to the active weapon slot."""
    stats = components.StatsComponent(strength=0)
    inv = components.CharacterInventory(stats)
    pistol = PISTOL_TYPE.create()
    knife = COMBAT_KNIFE_TYPE.create()

    inv.equip_to_slot(pistol, 0)
    inv.add_to_inventory(knife)
    inv.switch_to_weapon_slot(1)

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
    menu._use_item(knife)

    assert inv.attack_slots[1] == knife
    assert knife not in inv

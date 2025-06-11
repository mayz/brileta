from catley.game.components import InventoryComponent, StatsComponent
from catley.game.conditions import Injury
from catley.game.enums import InjuryLocation, ItemSize
from catley.game.items.item_core import Item, ItemType


def make_item(name: str, size: ItemSize) -> Item:
    return Item(ItemType(name=name, description="", size=size))


def test_get_used_inventory_slots_and_tiny_sharing():
    stats = StatsComponent(strength=0)
    inv = InventoryComponent(stats)
    tiny1 = make_item("t1", ItemSize.TINY)
    tiny2 = make_item("t2", ItemSize.TINY)
    normal = make_item("n1", ItemSize.NORMAL)
    big = make_item("b1", ItemSize.BIG)
    injury = Injury(InjuryLocation.LEFT_ARM, "Wound")

    inv.add_to_inventory(tiny1)
    assert inv.get_used_inventory_slots() == 1
    inv.add_to_inventory(tiny2)
    assert inv.get_used_inventory_slots() == 1
    inv.add_to_inventory(normal)
    assert inv.get_used_inventory_slots() == 2
    inv.attack_slots[0] = big
    assert inv.get_used_inventory_slots() == 4
    inv.add_to_inventory(injury)
    assert inv.get_used_inventory_slots() == 5


def test_can_add_to_inventory_capacity_and_tiny():
    stats = StatsComponent(strength=0)
    inv = InventoryComponent(stats)
    normal_items = [make_item(f"n{i}", ItemSize.NORMAL) for i in range(5)]
    for item in normal_items[:4]:
        assert inv.add_to_inventory(item)
    assert inv.get_used_inventory_slots() == 4

    tiny = make_item("t", ItemSize.TINY)
    assert inv.can_add_to_inventory(tiny)
    assert inv.add_to_inventory(tiny)
    assert inv.get_used_inventory_slots() == 5
    assert not inv.can_add_to_inventory(normal_items[4])

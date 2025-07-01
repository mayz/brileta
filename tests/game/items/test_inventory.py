from catley.game.actors import components, conditions
from catley.game.enums import InjuryLocation, ItemSize
from catley.game.items.item_core import Item, ItemType


def make_item(name: str, size: ItemSize) -> Item:
    return Item(ItemType(name=name, description="", size=size))


def test_get_used_inventory_slots_and_tiny_sharing():
    stats = components.StatsComponent(strength=0)
    inv = components.InventoryComponent(stats)
    tiny1 = make_item("t1", ItemSize.TINY)
    tiny2 = make_item("t2", ItemSize.TINY)
    normal = make_item("n1", ItemSize.NORMAL)
    big = make_item("b1", ItemSize.BIG)
    injury = conditions.Injury(InjuryLocation.LEFT_ARM, "Wound")

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
    stats = components.StatsComponent(strength=0)
    inv = components.InventoryComponent(stats)
    normal_items = [make_item(f"n{i}", ItemSize.NORMAL) for i in range(5)]
    for item in normal_items[:4]:
        success, _, _ = inv.add_to_inventory(item)
        assert success
    assert inv.get_used_inventory_slots() == 4

    tiny = make_item("t", ItemSize.TINY)
    assert inv.can_add_voluntary_item(tiny)
    success, _, _ = inv.add_to_inventory(tiny)
    assert success
    assert inv.get_used_inventory_slots() == 5
    assert not inv.can_add_voluntary_item(normal_items[4])


def test_equip_from_inventory_swaps_and_removes_item() -> None:
    stats = components.StatsComponent(strength=0)
    inv = components.InventoryComponent(stats)
    stored = make_item("stored", ItemSize.NORMAL)
    equipped = make_item("equipped", ItemSize.NORMAL)

    inv.add_to_inventory(stored)
    inv.equip_to_slot(equipped, 0)

    success, _ = inv.equip_from_inventory(stored, 0)

    assert success
    assert inv.attack_slots[0] is stored
    assert equipped in inv
    assert stored not in inv


def test_equip_from_inventory_fails_when_full() -> None:
    stats = components.StatsComponent(strength=0)
    inv = components.InventoryComponent(stats)
    items = [make_item(str(i), ItemSize.NORMAL) for i in range(5)]
    for item in items:
        inv.add_to_inventory(item)
    old_item = make_item("old", ItemSize.NORMAL)
    inv.equip_to_slot(old_item, 0)

    success, _ = inv.equip_from_inventory(items[0], 0)

    assert not success
    assert inv.attack_slots[0] is old_item
    assert items[0] in inv


def test_unequip_to_inventory_moves_item() -> None:
    stats = components.StatsComponent(strength=0)
    inv = components.InventoryComponent(stats)
    item = make_item("tool", ItemSize.NORMAL)
    inv.equip_to_slot(item, 0)

    success, _ = inv.unequip_to_inventory(0)

    assert success
    assert inv.attack_slots[0] is None
    assert item in inv

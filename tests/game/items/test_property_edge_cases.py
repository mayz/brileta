from catley.game.enums import ItemSize
from catley.game.items.capabilities import MeleeAttackSpec, RangedAttackSpec
from catley.game.items.item_core import Item, ItemType
from catley.game.items.item_types import PISTOL_MAGAZINE_TYPE
from catley.game.items.properties import (
    StatusProperty,
    TacticalProperty,
    WeaponProperty,
)


def test_item_with_no_attack_components() -> None:
    ammo = PISTOL_MAGAZINE_TYPE.create()
    assert ammo.get_preferred_attack_mode(1) is None
    assert not ammo.is_improvised_weapon()
    assert not ammo.is_designed_weapon()
    assert not ammo.get_weapon_properties()


def test_item_with_multiple_property_types() -> None:
    mixed_type = ItemType(
        name="Mixed",
        description="",
        size=ItemSize.NORMAL,
        melee_attack=MeleeAttackSpec(
            "d4",
            {
                WeaponProperty.AWKWARD,
                StatusProperty.STUNNING,
                TacticalProperty.CHEMICAL,
            },
        ),
    )
    item = Item(mixed_type)
    assert item.get_weapon_properties() == {
        WeaponProperty.AWKWARD,
        TacticalProperty.CHEMICAL,
    }
    assert item.get_status_effects() == {StatusProperty.STUNNING}


def test_empty_property_sets() -> None:
    item_type = ItemType(
        name="Empty",
        description="",
        size=ItemSize.NORMAL,
        melee_attack=MeleeAttackSpec("d4", set()),
        ranged_attack=RangedAttackSpec("d4", "a", 1, 2, 4, properties=set()),
    )
    item = Item(item_type)
    melee = item.melee_attack
    ranged = item.ranged_attack
    assert melee is not None and not melee.properties
    assert ranged is not None and not ranged.properties
    assert not item.get_weapon_properties()
    assert not item.get_status_effects()


def test_property_inheritance() -> None:
    props: set[WeaponProperty | StatusProperty | TacticalProperty] = {
        WeaponProperty.UNARMED,
        StatusProperty.BLEEDING,
        TacticalProperty.EXPLOSIVE,
    }
    assert WeaponProperty.UNARMED in props
    assert StatusProperty.BLEEDING in props
    assert TacticalProperty.EXPLOSIVE in props

from brileta.game.enums import AreaType, ItemSize
from brileta.game.items.capabilities import (
    AreaEffectSpec,
    MeleeAttackSpec,
    RangedAttackSpec,
)
from brileta.game.items.item_core import Item, ItemType
from brileta.game.items.properties import (
    StatusProperty,
    TacticalProperty,
    WeaponProperty,
)


def _make_basic_weapon(preferred_ranged: bool = False) -> Item:
    return Item(
        ItemType(
            name="Basic",
            description="",
            size=ItemSize.NORMAL,
            melee_attack=MeleeAttackSpec("d4"),
            ranged_attack=RangedAttackSpec(
                damage_die="d4",
                ammo_type="a",
                max_ammo=1,
                optimal_range=3,
                max_range=6,
                properties={WeaponProperty.PREFERRED} if preferred_ranged else None,
            ),
        )
    )


def test_get_preferred_attack_mode_melee_distance() -> None:
    item = _make_basic_weapon()
    assert item.get_preferred_attack_mode(1) == item.melee_attack


def test_get_preferred_attack_mode_ranged_distance() -> None:
    item = _make_basic_weapon()
    assert item.get_preferred_attack_mode(3) == item.ranged_attack


def test_get_preferred_attack_mode_preferred_property() -> None:
    item = _make_basic_weapon(preferred_ranged=True)
    assert item.get_preferred_attack_mode(2) == item.ranged_attack


def test_get_preferred_attack_mode_designed_over_improvised() -> None:
    item = Item(
        ItemType(
            name="Club",
            description="",
            size=ItemSize.NORMAL,
            melee_attack=MeleeAttackSpec("d4", {WeaponProperty.IMPROVISED}),
            area_effect=AreaEffectSpec("d4", AreaType.CIRCLE, 1),
        )
    )
    assert item.get_preferred_attack_mode(1) == item.area_effect


def test_get_preferred_attack_mode_no_attacks() -> None:
    empty = Item(ItemType(name="Nothing", description="", size=ItemSize.TINY))
    assert empty.get_preferred_attack_mode(1) is None


def test_get_preferred_attack_mode_wrong_distance() -> None:
    melee_only = Item(
        ItemType(
            name="M",
            description="",
            size=ItemSize.NORMAL,
            melee_attack=MeleeAttackSpec("d4"),
        )
    )
    ranged_only = Item(
        ItemType(
            name="R",
            description="",
            size=ItemSize.NORMAL,
            ranged_attack=RangedAttackSpec("d4", "a", 1, 2, 4),
        )
    )
    assert melee_only.get_preferred_attack_mode(3) is None
    assert ranged_only.get_preferred_attack_mode(1) == ranged_only.ranged_attack


def test_is_improvised_weapon_true() -> None:
    bottle = Item(
        ItemType(
            name="Bottle",
            description="",
            size=ItemSize.NORMAL,
            melee_attack=MeleeAttackSpec("d4", {WeaponProperty.IMPROVISED}),
        )
    )
    assert bottle.is_improvised_weapon()


def test_is_improvised_weapon_false() -> None:
    knife = _make_basic_weapon()
    assert not knife.is_improvised_weapon()


def test_is_designed_weapon_true() -> None:
    knife = _make_basic_weapon()
    assert knife.is_designed_weapon()


def test_is_designed_weapon_false() -> None:
    bottle = Item(
        ItemType(
            name="Bottle",
            description="",
            size=ItemSize.NORMAL,
            melee_attack=MeleeAttackSpec("d4", {WeaponProperty.IMPROVISED}),
        )
    )
    assert not bottle.is_designed_weapon()


def test_get_weapon_properties() -> None:
    item = Item(
        ItemType(
            name="Combo",
            description="",
            size=ItemSize.NORMAL,
            melee_attack=MeleeAttackSpec("d4", {WeaponProperty.AWKWARD}),
            ranged_attack=RangedAttackSpec(
                "d4",
                "a",
                1,
                2,
                4,
                properties={WeaponProperty.THROWN, WeaponProperty.IMPROVISED},
            ),
            area_effect=AreaEffectSpec(
                "d4",
                AreaType.CIRCLE,
                1,
                properties={WeaponProperty.CONTINUOUS, TacticalProperty.SPRAY},
            ),
        )
    )
    props = item.get_weapon_properties()
    assert {
        WeaponProperty.AWKWARD,
        WeaponProperty.THROWN,
        WeaponProperty.IMPROVISED,
        WeaponProperty.CONTINUOUS,
        TacticalProperty.SPRAY,
    } <= props


def test_get_status_effects() -> None:
    item = Item(
        ItemType(
            name="Dart",
            description="",
            size=ItemSize.NORMAL,
            ranged_attack=RangedAttackSpec(
                "d4",
                "dart",
                1,
                2,
                4,
                properties={StatusProperty.POISONING, WeaponProperty.SILENT},
            ),
        )
    )
    assert item.get_status_effects() == {StatusProperty.POISONING}

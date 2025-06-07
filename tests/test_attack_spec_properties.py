from catley.game.items.capabilities import MeleeAttackSpec
from catley.game.items.properties import WeaponProperty


def test_has_property_true() -> None:
    spec = MeleeAttackSpec("d6", {WeaponProperty.PREFERRED})
    assert spec.has_property(WeaponProperty.PREFERRED)


def test_has_property_false() -> None:
    spec = MeleeAttackSpec("d6")
    assert not spec.has_property(WeaponProperty.PREFERRED)


def test_has_any_property_true() -> None:
    spec = MeleeAttackSpec("d6", {WeaponProperty.AWKWARD})
    assert spec.has_any_property(WeaponProperty.THROWN, WeaponProperty.AWKWARD)


def test_has_any_property_false() -> None:
    spec = MeleeAttackSpec("d6", {WeaponProperty.AWKWARD})
    assert not spec.has_any_property(WeaponProperty.THROWN, WeaponProperty.UNARMED)


def test_has_all_properties_true() -> None:
    spec = MeleeAttackSpec("d6", {WeaponProperty.AWKWARD, WeaponProperty.TWO_HANDED})
    assert spec.has_all_properties(WeaponProperty.AWKWARD, WeaponProperty.TWO_HANDED)


def test_has_all_properties_false() -> None:
    spec = MeleeAttackSpec("d6", {WeaponProperty.AWKWARD})
    assert not spec.has_all_properties(
        WeaponProperty.AWKWARD, WeaponProperty.TWO_HANDED
    )


def test_empty_properties_set() -> None:
    spec = MeleeAttackSpec("d6")
    assert not spec.properties
    assert not spec.has_any_property(WeaponProperty.AWKWARD)
    assert not spec.has_all_properties(WeaponProperty.AWKWARD)

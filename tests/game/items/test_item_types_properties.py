from brileta.game.items.item_types import (
    COMBAT_KNIFE_TYPE,
    FISTS_TYPE,
    GRENADE_TYPE,
    PISTOL_TYPE,
    SNIPER_RIFLE_TYPE,
)
from brileta.game.items.properties import TacticalProperty, WeaponProperty


def test_pistol_properties() -> None:
    pistol = PISTOL_TYPE.create()
    attack_r = pistol.ranged_attack
    assert attack_r is not None
    assert WeaponProperty.PREFERRED in attack_r.properties
    attack_m = pistol.melee_attack
    assert attack_m is not None
    assert WeaponProperty.IMPROVISED in attack_m.properties


def test_combat_knife_properties() -> None:
    knife = COMBAT_KNIFE_TYPE.create()
    m_attack = knife.melee_attack
    assert m_attack is not None
    assert WeaponProperty.PREFERRED in m_attack.properties
    r_attack = knife.ranged_attack
    assert r_attack is not None
    assert WeaponProperty.THROWN in r_attack.properties


def test_sniper_rifle_properties() -> None:
    rifle = SNIPER_RIFLE_TYPE.create()
    r_attack = rifle.ranged_attack
    assert r_attack is not None
    assert {WeaponProperty.SCOPED, WeaponProperty.ARMOR_PIERCING} <= r_attack.properties


def test_fists_properties() -> None:
    fists = FISTS_TYPE.create()
    m_attack = fists.melee_attack
    assert m_attack is not None
    assert WeaponProperty.UNARMED in m_attack.properties


def test_grenade_properties() -> None:
    grenade = GRENADE_TYPE.create()
    area = grenade.area_effect
    assert area is not None
    assert {TacticalProperty.EXPLOSIVE, TacticalProperty.FIRE} <= area.properties

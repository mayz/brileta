# catley/game/item_types.py
from catley.game.enums import ItemSize
from catley.game.items.capabilities import (
    AmmoSpec,
    AreaEffectSpec,
    MeleeAttackSpec,
    RangedAttackSpec,
)

from .item_core import ItemType

# === ItemType Definitions ===


FISTS_TYPE = ItemType(
    name="Fists",
    description="Your bare hands. Better than nothing, but not by much.",
    size=ItemSize.TINY,
    melee_attack=MeleeAttackSpec("d4", {"unarmed"}),
)

SLEDGEHAMMER_TYPE = ItemType(
    name="Sledgehammer",
    description="Heavy two-handed weapon",
    size=ItemSize.BIG,
    melee_attack=MeleeAttackSpec("d8", {"two_handed"}),
)

COMBAT_KNIFE_TYPE = ItemType(
    name="Combat Knife",
    description="Sharp and deadly, can be thrown",
    size=ItemSize.NORMAL,
    melee_attack=MeleeAttackSpec("d6"),
    ranged_attack=RangedAttackSpec(
        damage_die="d4",
        ammo_type="thrown",
        max_ammo=1,
        optimal_range=4,
        max_range=8,
        properties={"thrown"},
    ),
)

PISTOL_TYPE = ItemType(
    name="Pistol",
    description="Reliable sidearm",
    size=ItemSize.NORMAL,
    ranged_attack=RangedAttackSpec(
        damage_die="d6",
        ammo_type="pistol",
        max_ammo=6,
        optimal_range=6,
        max_range=12,
    ),
    melee_attack=MeleeAttackSpec("d4", {"improvised"}),  # Pistol-whipping
)

HUNTING_RIFLE_TYPE = ItemType(
    name="Hunting Rifle",
    description="Accurate long-range weapon",
    size=ItemSize.BIG,
    ranged_attack=RangedAttackSpec(
        damage_die="d8",
        ammo_type="rifle",
        max_ammo=5,
        optimal_range=12,
        max_range=15,
    ),
    melee_attack=MeleeAttackSpec("d6", {"two_handed", "awkward"}),
)

SNIPER_RIFLE_TYPE = ItemType(
    name="Sniper Rifle",
    description="Lethal from afar",
    size=ItemSize.BIG,
    ranged_attack=RangedAttackSpec(
        damage_die="d12",
        ammo_type="rifle",
        max_ammo=5,
        optimal_range=20,
        max_range=30,
        properties={"scoped", "armor_piercing"},
    ),
)

SUBMACHINE_GUN_TYPE = ItemType(
    name="Submachine Gun",
    description="Full-auto weapon effective at close range",
    size=ItemSize.NORMAL,
    ranged_attack=RangedAttackSpec(
        damage_die="d6",
        ammo_type="pistol",
        max_ammo=20,
        optimal_range=8,
        max_range=15,
        properties={"automatic"},
    ),
    area_effect=AreaEffectSpec(
        damage_die="d6",
        area_type="cone",  # Spray pattern
        size=4,
        properties={"automatic", "spray"},
        damage_falloff=True,
        requires_line_of_sight=True,
        penetrates_walls=False,
        friendly_fire=True,
    ),
)

POISON_DART_GUN_TYPE = ItemType(
    name="Poison Dart Gun",
    description="Silent weapon that applies poison",
    size=ItemSize.NORMAL,
    ranged_attack=RangedAttackSpec(
        damage_die="d4",
        ammo_type="dart",
        max_ammo=3,
        optimal_range=6,
        max_range=12,
        properties={"poison", "silent"},
    ),
)

GRENADE_TYPE = ItemType(
    name="Grenade",
    description="Explosive device with wide blast radius",
    size=ItemSize.TINY,
    area_effect=AreaEffectSpec(
        damage_die="d6",
        area_type="circle",
        size=3,
        properties={"explosive", "fire"},  # Multiple effects as properties
        damage_falloff=True,
        requires_line_of_sight=False,
        penetrates_walls=False,
        friendly_fire=True,
    ),
)

FLAMETHROWER_TYPE = ItemType(
    name="Flamethrower",
    description="Sprays burning fuel in a line",
    size=ItemSize.BIG,
    area_effect=AreaEffectSpec(
        damage_die="d8",
        area_type="line",
        size=5,
        properties={"fire", "continuous"},  # Fire handled as property
        damage_falloff=True,
        requires_line_of_sight=True,
        penetrates_walls=False,
        friendly_fire=True,
    ),
)

PISTOL_MAGAZINE_TYPE = ItemType(
    name="Pistol Ammo",
    description="Pistol ammo magazine",
    size=ItemSize.TINY,
    ammo=AmmoSpec("pistol", 6),
)

RIFLE_MAGAZINE_TYPE = ItemType(
    name="Rifle Ammo",
    description="Rifle ammo magazine",
    size=ItemSize.TINY,
    ammo=AmmoSpec("rifle", 5),
)

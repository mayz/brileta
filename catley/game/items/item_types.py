import copy

from catley.game.actors import conditions
from catley.game.enums import AreaType, ConsumableEffectType, ItemCategory, ItemSize
from catley.game.items.capabilities import (
    AmmoSpec,
    AreaEffectSpec,
    ConsumableEffectSpec,
    MeleeAttackSpec,
    RangedAttackSpec,
)
from catley.game.items.properties import (
    StatusProperty,
    TacticalProperty,
    WeaponProperty,
)

from .item_core import ItemType

# === ItemType Definitions ===


FISTS_TYPE = ItemType(
    name="Fists",
    description="Your bare hands. Better than nothing, but not by much.",
    size=ItemSize.TINY,
    category=ItemCategory.WEAPON,
    melee_attack=MeleeAttackSpec("d3", {WeaponProperty.UNARMED}, verb="punch"),
    can_materialize=False,
)

SLEDGEHAMMER_TYPE = ItemType(
    name="Sledgehammer",
    description="Heavy two-handed weapon",
    size=ItemSize.BIG,
    category=ItemCategory.WEAPON,
    melee_attack=MeleeAttackSpec("d8", {WeaponProperty.TWO_HANDED}, verb="bludgeon"),
)

COMBAT_KNIFE_TYPE = ItemType(
    name="Combat Knife",
    description="Sharp and deadly, can be thrown",
    size=ItemSize.NORMAL,
    category=ItemCategory.WEAPON,
    melee_attack=MeleeAttackSpec(
        "d6", properties={WeaponProperty.PREFERRED}, verb="stab"
    ),
    ranged_attack=RangedAttackSpec(
        damage_die="d4",
        ammo_type="thrown",
        max_ammo=1,
        optimal_range=4,
        max_range=8,
        verb="throw",
        properties={WeaponProperty.THROWN},
    ),
)

PISTOL_TYPE = ItemType(
    name="Pistol",
    description="Reliable sidearm",
    size=ItemSize.NORMAL,
    category=ItemCategory.WEAPON,
    ranged_attack=RangedAttackSpec(
        damage_die="d6",
        ammo_type="pistol",
        max_ammo=6,
        optimal_range=6,
        max_range=12,
        properties={WeaponProperty.PREFERRED},
        verb="shoot",
    ),
    # Pistol-whipping.
    melee_attack=MeleeAttackSpec(
        "d4", verb="pistol-whip", properties={WeaponProperty.IMPROVISED}
    ),
)

REVOLVER_TYPE = copy.copy(PISTOL_TYPE)
REVOLVER_TYPE.name = "Revolver"

HUNTING_RIFLE_TYPE = ItemType(
    name="Hunting Rifle",
    description="Accurate long-range weapon",
    size=ItemSize.BIG,
    category=ItemCategory.WEAPON,
    ranged_attack=RangedAttackSpec(
        damage_die="d8",
        ammo_type="rifle",
        max_ammo=5,
        optimal_range=12,
        max_range=15,
        properties={WeaponProperty.PREFERRED},
    ),
    melee_attack=MeleeAttackSpec(
        "d6", {WeaponProperty.TWO_HANDED, WeaponProperty.AWKWARD}
    ),
)

SNIPER_RIFLE_TYPE = ItemType(
    name="Sniper Rifle",
    description="Lethal from afar",
    size=ItemSize.BIG,
    category=ItemCategory.WEAPON,
    ranged_attack=RangedAttackSpec(
        damage_die="d12",
        ammo_type="rifle",
        max_ammo=5,
        optimal_range=20,
        max_range=30,
        properties={
            WeaponProperty.PREFERRED,
            WeaponProperty.SCOPED,
            WeaponProperty.ARMOR_PIERCING,
        },
    ),
    melee_attack=MeleeAttackSpec(
        "d6", {WeaponProperty.TWO_HANDED, WeaponProperty.AWKWARD}
    ),
)

HUNTING_SHOTGUN_TYPE = ItemType(
    name="Hunting Shotgun",
    description="Powerful at close range",
    size=ItemSize.BIG,
    category=ItemCategory.WEAPON,
    ranged_attack=RangedAttackSpec(
        damage_die="d8",
        ammo_type="shotgun",
        max_ammo=5,
        optimal_range=5,
        max_range=10,
        properties={WeaponProperty.PREFERRED},
    ),
    melee_attack=MeleeAttackSpec(
        "d6", {WeaponProperty.TWO_HANDED, WeaponProperty.AWKWARD}
    ),
)

SUBMACHINE_GUN_TYPE = ItemType(
    name="Submachine Gun",
    description="Full-auto weapon effective at close range",
    size=ItemSize.NORMAL,
    category=ItemCategory.WEAPON,
    ranged_attack=RangedAttackSpec(
        damage_die="d6",
        ammo_type="pistol",
        max_ammo=20,
        optimal_range=8,
        max_range=15,
        properties={WeaponProperty.PREFERRED, WeaponProperty.AUTOMATIC},
    ),
    area_effect=AreaEffectSpec(
        damage_die="d6",
        area_type=AreaType.CONE,  # Spray pattern
        size=4,
        properties={WeaponProperty.AUTOMATIC, TacticalProperty.SPRAY},
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
    category=ItemCategory.WEAPON,
    ranged_attack=RangedAttackSpec(
        damage_die="d4",
        ammo_type="dart",
        max_ammo=3,
        optimal_range=6,
        max_range=12,
        properties={StatusProperty.POISONING, WeaponProperty.SILENT},
    ),
)

GRENADE_TYPE = ItemType(
    name="Grenade",
    description="Explosive device with wide blast radius",
    size=ItemSize.TINY,
    category=ItemCategory.WEAPON,
    area_effect=AreaEffectSpec(
        damage_die="d6",
        area_type=AreaType.CIRCLE,
        size=3,
        properties={TacticalProperty.EXPLOSIVE, TacticalProperty.FIRE},
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
    category=ItemCategory.WEAPON,
    area_effect=AreaEffectSpec(
        damage_die="d8",
        area_type=AreaType.LINE,
        size=5,
        properties={TacticalProperty.FIRE, WeaponProperty.CONTINUOUS},
        damage_falloff=True,
        requires_line_of_sight=True,
        penetrates_walls=False,
        friendly_fire=True,
    ),
)

RADIUM_GUN_TYPE = ItemType(
    name="Radium Gun",
    description="Weapon that fires radioactive projectiles",
    size=ItemSize.BIG,
    category=ItemCategory.WEAPON,
    ranged_attack=RangedAttackSpec(
        damage_die="d8",
        ammo_type="radium",
        max_ammo=3,
        optimal_range=10,
        max_range=20,
        properties={WeaponProperty.PREFERRED, TacticalProperty.RADIATION},
    ),
)

DIRTY_BOMB_TYPE = ItemType(
    name="Dirty Bomb",
    description="Explosive device that spreads radiation",
    size=ItemSize.TINY,
    category=ItemCategory.WEAPON,
    area_effect=AreaEffectSpec(
        damage_die="d6",
        area_type=AreaType.CIRCLE,
        size=4,
        properties={TacticalProperty.EXPLOSIVE, TacticalProperty.RADIATION},
        damage_falloff=True,
        requires_line_of_sight=False,
        penetrates_walls=False,
        friendly_fire=True,
    ),
)

PISTOL_MAGAZINE_TYPE = ItemType(
    name="Pistol Ammo",
    description="Pistol ammo magazine",
    size=ItemSize.TINY,
    category=ItemCategory.MUNITIONS,
    ammo=AmmoSpec("pistol", 6),
)

RIFLE_MAGAZINE_TYPE = ItemType(
    name="Rifle Ammo",
    description="Rifle ammo magazine",
    size=ItemSize.TINY,
    category=ItemCategory.MUNITIONS,
    ammo=AmmoSpec("rifle", 5),
)

SHOTGUN_SHELLS_TYPE = ItemType(
    name="Shotgun Shells",
    description="12 gauge shotgun shells",
    size=ItemSize.TINY,
    category=ItemCategory.MUNITIONS,
    ammo=AmmoSpec("shotgun", 5),
)

# Various items to test.

ROCK_TYPE = ItemType(
    name="Rock",
    description="A rough chunk of stone. Good for throwing or bashing.",
    size=ItemSize.TINY,
    # MISC (default) - improvised item, not a purpose-built weapon or junk
    ranged_attack=RangedAttackSpec(
        damage_die="d4",
        ammo_type="thrown",
        max_ammo=1,
        optimal_range=4,
        max_range=8,
        properties={WeaponProperty.THROWN, WeaponProperty.IMPROVISED},
    ),
    melee_attack=MeleeAttackSpec("d3", properties={WeaponProperty.IMPROVISED}),
)

ALARM_CLOCK_TYPE = ItemType(
    name="Alarm Clock",
    description=(
        "Heavy metal timepiece. Makes a satisfying thunk when it hits something."
    ),
    size=ItemSize.NORMAL,
    # MISC (default) - improvised item, not a purpose-built weapon or junk
    ranged_attack=RangedAttackSpec(
        damage_die="d6",
        ammo_type="thrown",
        max_ammo=1,
        optimal_range=3,
        max_range=6,
        properties={WeaponProperty.THROWN, WeaponProperty.IMPROVISED},
    ),
    melee_attack=MeleeAttackSpec("d6", properties={WeaponProperty.IMPROVISED}),
)

BLEACH_TYPE = ItemType(
    name="Bleach",
    description="Household disinfectant. Caustic and dangerous if misused.",
    size=ItemSize.NORMAL,
    # MISC (default) - improvised item, not a purpose-built weapon or junk
    consumable_effect=ConsumableEffectSpec(
        effect_type=ConsumableEffectType.POISON,
        effect_value=-15,  # Toxic if consumed
        max_uses=1,
    ),
    ranged_attack=RangedAttackSpec(
        damage_die="d8",
        ammo_type="splash",
        max_ammo=1,
        optimal_range=2,
        max_range=4,
        properties={
            WeaponProperty.IMPROVISED,
            TacticalProperty.CHEMICAL,
            StatusProperty.BLINDING,
        },
    ),
)

RUBBER_CHICKEN_TYPE = ItemType(
    name="Rubber Chicken",
    description="A squeaky novelty toy. Utterly ridiculous but surprisingly durable.",
    size=ItemSize.NORMAL,
    # MISC (default) - improvised item, not a purpose-built weapon or junk
    ranged_attack=RangedAttackSpec(
        damage_die="d2",
        ammo_type="thrown",
        max_ammo=1,
        optimal_range=3,
        max_range=6,
        properties={WeaponProperty.THROWN, WeaponProperty.IMPROVISED},
    ),
    melee_attack=MeleeAttackSpec("d2", properties={WeaponProperty.IMPROVISED}),
)

# === Medical and Recovery Items ===
CURE_ALL_TYPE = ItemType(
    name="Cure-All",
    description="Powerful remedy that cures sickness and radiation",
    size=ItemSize.NORMAL,
    category=ItemCategory.CONSUMABLE,
    consumable_effect=ConsumableEffectSpec(
        effect_type=ConsumableEffectType.CURE_ALL_CONDITIONS,
        effect_value=0,
        target_condition_types={conditions.Rads, conditions.Sickness},
    ),
)

MEGATONIC_TYPE = ItemType(
    name="Megatonic",
    description="Revitalizing tonic that heals fully and removes exhaustion",
    size=ItemSize.NORMAL,
    category=ItemCategory.CONSUMABLE,
    consumable_effect=ConsumableEffectSpec(
        effect_type=ConsumableEffectType.HEAL,
        target_condition_types={conditions.Exhaustion},
    ),
)

STIM_TYPE = ItemType(
    name="Stimpack",
    description="Instantly restores all health",
    size=ItemSize.TINY,
    category=ItemCategory.CONSUMABLE,
    consumable_effect=ConsumableEffectSpec(
        effect_type=ConsumableEffectType.HEAL,
    ),
)

BANDAGES_TYPE = ItemType(
    name="Bandages",
    description="Clean dressings for treating wounds",
    size=ItemSize.TINY,
    category=ItemCategory.CONSUMABLE,
    consumable_effect=ConsumableEffectSpec(
        effect_type=ConsumableEffectType.CURE_INJURIES,
        effect_value=0,
        target_condition_types={conditions.Injury},
    ),
)

FOOD_TYPE = ItemType(
    name="Rations",
    description="Basic food that restores all health",
    size=ItemSize.NORMAL,
    category=ItemCategory.CONSUMABLE,
    consumable_effect=ConsumableEffectSpec(
        effect_type=ConsumableEffectType.HEAL,
    ),
)

DRINK_TYPE = ItemType(
    name="Water Flask",
    description="Clean water that restores all health",
    size=ItemSize.NORMAL,
    category=ItemCategory.CONSUMABLE,
    consumable_effect=ConsumableEffectSpec(
        effect_type=ConsumableEffectType.HEAL,
    ),
)

"""Concrete NPC archetypes used by world generation and spawning.

Each type defines a creature kind - its physical properties, capabilities,
and stat distributions. Behavioral variation within a type will come from
personality traits (Phase 5). For now, tags approximate behavioral profiles
that personality will eventually replace.
"""

from __future__ import annotations

from brileta import colors
from brileta.game.enums import CreatureSize
from brileta.game.items.item_types import (
    DOG_BITE_TYPE,
    REVOLVER_TYPE,
    SCORPION_STING_TYPE,
    SLEDGEHAMMER_TYPE,
)

from .npc_core import NPCType, StatDistribution

# Large mutant humanoid. Charges into melee, hits hard, slow and stupid.
# No sapient tag - attacks or flees without the Watch/Avoid social reasoning.
TROG_TYPE = NPCType(
    id="trog",
    display_name="Trog",
    tags=("base", "combatant"),
    glyph="T",
    color=colors.DARK_GREY,
    creature_size=CreatureSize.LARGE,
    default_disposition=0,
    can_open_doors=True,
    starting_weapon=SLEDGEHAMMER_TYPE,
    speed=80,
    awareness_radius=10,  # Dumb brute, limited awareness
    strength_dist=StatDistribution(mean=3, std_dev=1.0),
    toughness_dist=StatDistribution(mean=3, std_dev=1.0),
    agility_dist=StatDistribution(mean=-1, std_dev=1.0),
    observation_dist=StatDistribution(mean=0, std_dev=1.0),
    intelligence_dist=StatDistribution(mean=-3, std_dev=0.7),
    demeanor_dist=StatDistribution(mean=-1, std_dev=1.0),
    weirdness_dist=StatDistribution(mean=3, std_dev=0.8),
)

# Armed human raider. Hostile on sight (disposition -75). Slightly faster
# than the player (105) - can't be trivially kited. Watches and avoids
# before engaging, flees when hurt. The sapient tag gives social threat
# reasoning that trogs lack.
BRIGAND_TYPE = NPCType(
    id="brigand",
    display_name="Brigand",
    tags=("base", "combatant", "sapient"),
    glyph="B",
    color=colors.DARK_GREY,
    creature_size=CreatureSize.MEDIUM,
    default_disposition=-75,
    can_open_doors=True,
    starting_weapon=REVOLVER_TYPE,
    speed=105,
    awareness_radius=14,  # Alert human, above-average perception
    strength_dist=StatDistribution(mean=1, std_dev=1.2),
    toughness_dist=StatDistribution(mean=1, std_dev=1.1),
    agility_dist=StatDistribution(mean=1, std_dev=1.0),
    observation_dist=StatDistribution(mean=1, std_dev=1.0),
    intelligence_dist=StatDistribution(mean=2, std_dev=1.0),
    demeanor_dist=StatDistribution(mean=-1, std_dev=1.0),
    weirdness_dist=StatDistribution(mean=1, std_dev=1.0),
)

# Hostile wildlife. Attacks based on proximity rather than disposition - the
# predator tag overrides the standard Attack with proximity-triggered
# considerations. Very tough, can't open doors. Slower than the player (85)
# so you can outrun it in open ground, but dangerous in tight spaces.
GIANT_SCORPION_TYPE = NPCType(
    id="giant_scorpion",
    display_name="Giant Scorpion",
    tags=("base", "combatant", "predator"),
    glyph="s",
    color=colors.TAN,
    creature_size=CreatureSize.LARGE,
    default_disposition=-40,
    can_open_doors=False,
    starting_weapon=SCORPION_STING_TYPE,
    speed=85,
    awareness_radius=6,  # Arachnid, relies on vibration at short range
    strength_dist=StatDistribution(mean=2, std_dev=1.0),
    toughness_dist=StatDistribution(mean=4, std_dev=0.7),
    agility_dist=StatDistribution(mean=1, std_dev=1.0),
    observation_dist=StatDistribution(mean=0, std_dev=1.0),
    intelligence_dist=StatDistribution(mean=-4, std_dev=0.5),
    demeanor_dist=StatDistribution(mean=-3, std_dev=0.8),
    weirdness_dist=StatDistribution(mean=2, std_dev=1.0),
)

# Domestic dog. Fast, small, can't open doors. Currently uses the skittish
# tag as a stand-in: flees when anything gets close. Once personality traits
# land (Phase 5), the skittish tag should be replaced by personality-driven
# scoring - high-Neuroticism dogs flee, high-Extraversion dogs approach.
DOG_TYPE = NPCType(
    id="dog",
    display_name="Dog",
    tags=("base", "skittish"),
    glyph="d",
    color=colors.BROWN,
    creature_size=CreatureSize.SMALL,
    default_disposition=20,
    can_open_doors=False,
    starting_weapon=DOG_BITE_TYPE,
    speed=110,
    awareness_radius=18,  # Keen hearing and smell, large detection range
    strength_dist=StatDistribution(mean=-2, std_dev=1.0),
    toughness_dist=StatDistribution(mean=0, std_dev=1.0),
    agility_dist=StatDistribution(mean=3, std_dev=1.0),
    observation_dist=StatDistribution(mean=2, std_dev=1.0),
    intelligence_dist=StatDistribution(mean=-2, std_dev=1.0),
    demeanor_dist=StatDistribution(mean=2, std_dev=1.0),
    weirdness_dist=StatDistribution(mean=0, std_dev=1.0),
)

# Unarmed settlement human. Neutral toward strangers. Watches and avoids
# threats but will fight back if provoked (disposition drops) and cornered.
# Flees when hurt. Average stats across the board.
RESIDENT_TYPE = NPCType(
    id="resident",
    display_name="Resident",
    tags=("base", "combatant", "sapient"),
    glyph="R",
    color=colors.LIGHT_GREY,
    creature_size=CreatureSize.MEDIUM,
    default_disposition=0,
    can_open_doors=True,
    starting_weapon=None,
    strength_dist=StatDistribution(mean=0, std_dev=1.0),
    toughness_dist=StatDistribution(mean=0, std_dev=1.0),
    agility_dist=StatDistribution(mean=0, std_dev=1.0),
    observation_dist=StatDistribution(mean=1, std_dev=1.0),
    intelligence_dist=StatDistribution(mean=0, std_dev=1.0),
    demeanor_dist=StatDistribution(mean=1, std_dev=1.0),
    weirdness_dist=StatDistribution(mean=0, std_dev=1.0),
)

# Lookup table for the dev console ``spawn`` command and other systems
# that need to resolve an NPC type from a string identifier.
NPC_TYPE_REGISTRY: dict[str, NPCType] = {
    t.id: t
    for t in (TROG_TYPE, BRIGAND_TYPE, GIANT_SCORPION_TYPE, DOG_TYPE, RESIDENT_TYPE)
}

__all__ = [
    "BRIGAND_TYPE",
    "DOG_TYPE",
    "GIANT_SCORPION_TYPE",
    "NPC_TYPE_REGISTRY",
    "RESIDENT_TYPE",
    "TROG_TYPE",
]

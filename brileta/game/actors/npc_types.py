"""Concrete NPC archetypes used by world generation and spawning.

Each type defines a creature kind - its physical properties, capabilities, and
stat distributions. Behavioral variation within a type comes from OCEAN
personality traits (Phase 5): every type samples five traits at spawn from its
own per-trait distributions, so two NPCs of the same type at the same
disposition still behave differently.
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
from brileta.sprites.quadrupeds import DOG_PRESET

from .identity import Gender
from .npc_core import NPCType, StatDistribution, personality_trait

_BALANCED_HUMANOID_IDENTITY = ((Gender.MALE, 1.0), (Gender.FEMALE, 1.0))

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
    identity_weights=_BALANCED_HUMANOID_IDENTITY,
    speed=80,
    awareness_radius=10,  # Dumb brute, limited awareness
    strength_dist=StatDistribution(mean=3, std_dev=1.0),
    toughness_dist=StatDistribution(mean=3, std_dev=1.0),
    agility_dist=StatDistribution(mean=-1, std_dev=1.0),
    observation_dist=StatDistribution(mean=0, std_dev=1.0),
    intelligence_dist=StatDistribution(mean=-3, std_dev=0.7),
    demeanor_dist=StatDistribution(mean=-1, std_dev=1.0),
    weirdness_dist=StatDistribution(mean=3, std_dev=0.8),
    # Fearless, aggressive brute: low neuroticism (rarely flees), low
    # agreeableness (attacks readily), incurious and undisciplined.
    openness_dist=personality_trait(mean=3),
    conscientiousness_dist=personality_trait(mean=3),
    agreeableness_dist=personality_trait(mean=2),
    neuroticism_dist=personality_trait(mean=2),
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
    identity_weights=_BALANCED_HUMANOID_IDENTITY,
    speed=105,
    awareness_radius=14,  # Alert human, above-average perception
    strength_dist=StatDistribution(mean=1, std_dev=1.2),
    toughness_dist=StatDistribution(mean=1, std_dev=1.1),
    agility_dist=StatDistribution(mean=1, std_dev=1.0),
    observation_dist=StatDistribution(mean=1, std_dev=1.0),
    intelligence_dist=StatDistribution(mean=2, std_dev=1.0),
    demeanor_dist=StatDistribution(mean=-1, std_dev=1.0),
    weirdness_dist=StatDistribution(mean=1, std_dev=1.0),
    # Hostile and confrontational, but individuals vary in nerve: a wide
    # neuroticism spread means some brigands break and run early while others
    # hold the line, which is the core "same type, different behavior" showcase.
    conscientiousness_dist=personality_trait(mean=4),
    agreeableness_dist=personality_trait(mean=3),
    neuroticism_dist=personality_trait(mean=5, spread=2.5),
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
    # Aggressive ambush predator: very low neuroticism (does not flee) and low
    # agreeableness (strikes on proximity). Narrow spreads - little individual
    # variation in an instinct-driven creature.
    openness_dist=personality_trait(mean=3, spread=1.0),
    agreeableness_dist=personality_trait(mean=1, spread=1.0),
    neuroticism_dist=personality_trait(mean=1, spread=1.0),
)

# Domestic dog. Fast, small, can't open doors. Uses the skittish tag for its
# flee-from-proximity mechanism, but the flight threshold is now personality-
# driven (Phase 5): high-Neuroticism dogs bolt, low-Neuroticism dogs stay near.
# The wide neuroticism spread below makes both kinds show up in one litter.
DOG_TYPE = NPCType(
    id="dog",
    display_name="Dog",
    tags=("base", "skittish"),
    glyph="d",
    color=colors.BROWN,
    creature_size=CreatureSize.SMALL,
    critter_preset=DOG_PRESET,
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
    # Wide neuroticism spread is the whole point of retiring the stand-in tag:
    # a skittish dog (high neuroticism) bolts from anyone close, a bold dog (low
    # neuroticism) lets its wander/idle win and stays near people. High
    # extraversion and openness keep dogs generally roaming and inquisitive.
    openness_dist=personality_trait(mean=6),
    extraversion_dist=personality_trait(mean=7),
    neuroticism_dist=personality_trait(mean=5, spread=3.0),
)

# Unarmed settlement human. Neutral toward strangers. Watches and avoids
# threats but will fight back if provoked (disposition drops) and cornered.
# Flees when hurt. Average stats across the board.
RESIDENT_TYPE = NPCType(
    id="resident",
    display_name="Resident",
    tags=("base", "combatant", "sapient", "routine", "social"),
    glyph="R",
    color=colors.LIGHT_GREY,
    creature_size=CreatureSize.MEDIUM,
    default_disposition=0,
    can_open_doors=True,
    starting_weapon=None,
    identity_weights=_BALANCED_HUMANOID_IDENTITY,
    strength_dist=StatDistribution(mean=0, std_dev=1.0),
    toughness_dist=StatDistribution(mean=0, std_dev=1.0),
    agility_dist=StatDistribution(mean=0, std_dev=1.0),
    observation_dist=StatDistribution(mean=1, std_dev=1.0),
    intelligence_dist=StatDistribution(mean=0, std_dev=1.0),
    demeanor_dist=StatDistribution(mean=1, std_dev=1.0),
    weirdness_dist=StatDistribution(mean=0, std_dev=1.0),
    # Ordinary townsfolk: average in every trait but with wide spreads, so a
    # crowd shows the full behavioral range - some residents roam and run
    # errands (high openness/extraversion), others stand about (low), some
    # startle easily (high neuroticism), some are unflappable (low).
    openness_dist=personality_trait(spread=2.5),
    conscientiousness_dist=personality_trait(spread=2.5),
    extraversion_dist=personality_trait(spread=2.5),
    agreeableness_dist=personality_trait(spread=2.0),
    neuroticism_dist=personality_trait(spread=2.5),
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

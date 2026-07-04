"""OCEAN personality traits for NPCs.

PersonalityComponent holds five Big Five (OCEAN) traits on a 0-10 integer
scale where 5 is the human average. Traits are sampled once at spawn from an
NPCType's per-trait distribution and are then essentially static - they
differentiate individuals *within* a type and disposition range, unlike
disposition (which changes with interactions) or ability scores (which
determine capability rather than tendency).

Traits feed the utility system the same way disposition and distance do: each
is exposed as a normalized 0-1 input on UtilityContext and shaped by a response
curve on whichever actions it biases. See the trait-to-consideration wiring in
actions.py, behaviors/, and goals.py.

For sapient NPCs these are genuine personality traits; for creatures they are
analogous behavioral parameters read through the same curves (a high-Neuroticism
scorpion simply has a lower flight threshold). The scoring system does not
distinguish the two interpretations.
"""

from __future__ import annotations

from dataclasses import dataclass

# The trait scale is a 0-10 integer with 5 as the human average.
TRAIT_MIN = 0
TRAIT_AVERAGE = 5
TRAIT_MAX = 10


def normalize_trait(value: int) -> float:
    """Map a 0-10 trait onto the 0-1 range utility considerations expect."""
    return value / TRAIT_MAX


@dataclass(slots=True)
class PersonalityComponent:
    """Five OCEAN traits, each a 0-10 integer (5 = human average).

    Openness: curiosity and exploration vs. routine-bound caution.
    Conscientiousness: discipline and goal persistence vs. erratic impulsivity.
    Extraversion: gregarious, roaming assertiveness vs. reserved solitude.
    Agreeableness: cooperation, slow to attack vs. quick confrontation.
    Neuroticism: reactive, low flight threshold vs. calm, steady nerve.
    """

    openness: int = TRAIT_AVERAGE
    conscientiousness: int = TRAIT_AVERAGE
    extraversion: int = TRAIT_AVERAGE
    agreeableness: int = TRAIT_AVERAGE
    neuroticism: int = TRAIT_AVERAGE

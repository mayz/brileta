from __future__ import annotations

from dataclasses import dataclass, field

from brileta.game.actors import conditions


@dataclass
class CombatOutcome:
    """Represents the specific results of a combat action.

    Attributes:
        damage_dealt: Final damage to the defender's HP (after armor reduction).
        armor_penetrated: True if the attack penetrated armor (damage exceeded PR).
        attacker_recoil_damage: Damage dealt back to the attacker.
        injury_inflicted: Any injury condition inflicted on the defender.
        status_effects_applied: List of status effects applied.
    """

    damage_dealt: int = 0
    armor_penetrated: bool = False
    attacker_recoil_damage: int = 0
    injury_inflicted: conditions.Injury | None = None
    status_effects_applied: list = field(default_factory=list)

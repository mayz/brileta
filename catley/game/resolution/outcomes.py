from __future__ import annotations

from dataclasses import dataclass, field

from catley.game.conditions import Injury


@dataclass
class CombatOutcome:
    """Represents the specific results of a combat action."""

    damage_dealt: int = 0
    armor_damage: int = 0
    attacker_recoil_damage: int = 0
    injury_inflicted: Injury | None = None
    status_effects_applied: list = field(default_factory=list)

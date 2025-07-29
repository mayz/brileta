"""
Environmental damage actions for hazards like fire, radiation, and falling damage.

Handles passive environmental hazards that don't have an attacker or weapon,
such as fires, radiation zones, acid pools, and falling damage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley.types import WorldTilePos

from .base import GameIntent

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors.core import Actor


class EnvironmentalDamageIntent(GameIntent):
    """Intent for environmental damage from hazards like fire or radiation.

    Used for passive environmental damage sources that don't have an attacker
    or weapon, such as:
    - Fire damage from campfires, burning areas
    - Radiation damage from contaminated zones
    - Falling damage from cliffs
    - Acid damage from pools
    - Electrical damage from exposed wires
    """

    def __init__(
        self,
        controller: Controller,
        source_actor: Actor,
        damage_amount: int,
        damage_type: str = "normal",
        affected_coords: list[WorldTilePos] | None = None,
        source_description: str = "environmental hazard",
    ) -> None:
        super().__init__(controller, source_actor)
        self.source_actor = source_actor
        self.damage_amount = damage_amount
        self.damage_type = damage_type
        self.affected_coords = affected_coords or [(source_actor.x, source_actor.y)]
        self.source_description = source_description

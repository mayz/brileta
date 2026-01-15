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

    For terrain-based hazards (like acid pools), source_actor can be None since
    the hazard is part of the map rather than an actor. In this case,
    affected_coords must be explicitly provided.
    """

    def __init__(
        self,
        controller: Controller,
        source_actor: Actor | None,
        damage_amount: int,
        damage_type: str = "normal",
        affected_coords: list[WorldTilePos] | None = None,
        source_description: str = "environmental hazard",
    ) -> None:
        # For terrain hazards without a source actor, use player as the actor
        # for GameIntent base class (this is just for compatibility).
        effective_actor = source_actor if source_actor else controller.gw.player
        super().__init__(controller, effective_actor)
        self.source_actor = source_actor  # May be None for terrain hazards
        self.damage_amount = damage_amount
        self.damage_type = damage_type
        # For terrain hazards, affected_coords must be explicitly provided
        if affected_coords is None and source_actor is not None:
            self.affected_coords = [(source_actor.x, source_actor.y)]
        else:
            self.affected_coords = affected_coords or []
        self.source_description = source_description

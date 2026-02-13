"""Perception system for NPC awareness.

PerceptionComponent gates NPC awareness behind range and line-of-sight
checks. It answers "which actors can this NPC currently detect?" without
interpreting threat or making behavioral decisions - that stays in the
utility scoring layer.

Detection is omnidirectional (no facing or vision cones). An actor is
perceived when it is within awareness_radius AND visible via Bresenham
line-of-sight against the tile map's transparency grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from brileta.game import ranges

if TYPE_CHECKING:
    from brileta.environment.map import GameMap
    from brileta.game.actors.core import Actor, Character


@dataclass(frozen=True, slots=True)
class PerceivedActor:
    """A single actor detected by PerceptionComponent this tick.

    Attributes:
        actor: The detected actor reference.
        distance: Chebyshev distance from the perceiver to this actor.
        perception_strength: Signal strength in [0, 1]. 1.0 at point
            blank, falls off linearly with distance. Actors at the exact
            awareness radius are excluded from perception entirely.
    """

    actor: Character
    distance: int
    perception_strength: float


class PerceptionComponent:
    """Omnidirectional perception model for NPCs.

    Determines which actors an NPC can currently detect based on range
    and line-of-sight. No vision cones, no facing direction - this is
    a simple "can I sense them?" check combining hearing/peripheral
    vision into a single radius.

    Attributes:
        awareness_radius: Maximum detection range in tiles (Chebyshev
            distance). Actors beyond this range are never detected.
    """

    def __init__(self, awareness_radius: int = 12) -> None:
        self.awareness_radius = awareness_radius

    def get_perceived_actors(
        self,
        actor: Actor,
        game_map: GameMap,
        candidates: list[Actor],
    ) -> list[PerceivedActor]:
        """Return all actors this NPC can currently detect.

        Detection requires two conditions:
        1. Target is strictly within awareness_radius (Chebyshev distance).
        2. Target is visible via line-of-sight (Bresenham ray against
           the tile map's transparency grid).

        Args:
            actor: The perceiving NPC.
            game_map: The game map for LOS checks.
            candidates: Actors to check for detection. Typically from
                a spatial index query.

        Returns:
            List of PerceivedActor entries sorted by distance (closest
            first).
        """
        from brileta.game.actors.core import Character

        perceived: list[PerceivedActor] = []

        for other in candidates:
            if other is actor:
                continue
            if not isinstance(other, Character):
                continue
            if not other.health.is_alive():
                continue

            distance = ranges.calculate_distance(actor.x, actor.y, other.x, other.y)
            if distance >= self.awareness_radius:
                continue

            # Line-of-sight check: Bresenham ray must pass through only
            # transparent tiles between perceiver and target.
            if not ranges.has_line_of_sight(
                game_map, actor.x, actor.y, other.x, other.y
            ):
                continue

            # Perception strength: 1.0 at distance 0, linear falloff with
            # distance. Exact-radius actors are already excluded above.
            strength = 1.0 - (distance / self.awareness_radius)

            perceived.append(
                PerceivedActor(
                    actor=other,
                    distance=distance,
                    perception_strength=strength,
                )
            )

        # Sort by distance so closest actors are first.
        perceived.sort(key=lambda p: p.distance)
        return perceived

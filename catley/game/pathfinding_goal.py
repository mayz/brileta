from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from catley.game.actions.base import GameIntent
    from catley.util.coordinates import WorldTilePos


@dataclass
class PathfindingGoal:
    """
    Represents an actor's autonomous movement goal.

    This is a stateful object that holds the final destination and manages
    hierarchical pathfinding state. For cross-region paths, it maintains both
    a high-level sequence of regions to traverse and a local path within the
    current region.

    Attributes:
        target_pos: The final destination tile for the movement.
        final_intent: An optional GameIntent to be queued upon arrival.
        high_level_path: For HPA*, a sequence of region IDs to traverse.
            When crossing regions, this holds the remaining regions after
            the current one. None for same-region paths.
        _cached_path: A list of tiles representing the current local path.
            For cross-region paths, this goes only to the next region's
            connection point. Can be invalidated and recalculated.
    """

    target_pos: WorldTilePos
    final_intent: GameIntent | None = None
    high_level_path: list[int] | None = None
    _cached_path: list[WorldTilePos] | None = None

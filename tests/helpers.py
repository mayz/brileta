from __future__ import annotations

import contextlib
from collections.abc import Sequence

from catley.environment import tile_types
from catley.environment.map import GameMap
from catley.game.actors import Actor, Character
from catley.game.game_world import GameWorld
from catley.util.spatial import SpatialHashGrid


class DummyGameWorld(GameWorld):
    """Lightweight GameWorld for tests."""

    def __init__(
        self,
        width: int = 30,
        height: int = 30,
        *,
        game_map: GameMap | None = None,
        actors: Sequence[Actor] | None = None,
    ) -> None:
        # Avoid heavy GameWorld initialization.
        if game_map is None:
            game_map = GameMap(width, height)
            game_map.tiles[:] = tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
            game_map.transparent[:] = True
            game_map.visible[:] = True
        self.game_map = game_map
        self.actors = list(actors) if actors is not None else []
        self.player: Character | None = None
        self.selected_actor: Character | None = None
        self.items: dict[tuple[int, int], list] = {}
        self.actor_spatial_index = SpatialHashGrid(cell_size=16)

    def add_actor(self, actor: Actor) -> None:
        self.actors.append(actor)
        self.actor_spatial_index.add(actor)

    def remove_actor(self, actor: Actor) -> None:
        try:
            self.actors.remove(actor)
            self.actor_spatial_index.remove(actor)
        except ValueError:
            pass

    def get_pickable_items_at_location(self, x: int, y: int) -> list:
        """Return items stored at ``(x, y)`` or carried by ground actors."""
        items = list(self.items.get((x, y), []))
        with contextlib.suppress(AttributeError):
            items.extend(super().get_pickable_items_at_location(x, y))
        return items

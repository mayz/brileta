from __future__ import annotations

from collections.abc import Sequence

from catley.controller import GameWorld
from catley.environment import tile_types
from catley.environment.map import GameMap
from catley.game.actors import Actor, Character
from catley.util.spatial import SpatialHashGrid


class DummyGameWorld(GameWorld):
    """A lightweight, standalone dummy GameWorld for testing."""

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
        self.game_map.gw = self

        self.actor_spatial_index = SpatialHashGrid(cell_size=16)
        self.actors: list[Actor] = []

        # Add initial actors through the proper lifecycle method.
        if actors:
            for actor in actors:
                self.add_actor(actor)

        self.player: Character | None = None
        self.selected_actor: Character | None = None
        self.items: dict[tuple[int, int], list] = {}

    def add_actor(self, actor: Actor) -> None:
        """Adds an actor to the list and the spatial index."""
        self.actors.append(actor)
        self.actor_spatial_index.add(actor)

    def remove_actor(self, actor: Actor) -> None:
        """Removes an actor from the list and the spatial index."""
        try:
            self.actors.remove(actor)
            self.actor_spatial_index.remove(actor)
        except ValueError:
            pass

    def get_pickable_items_at_location(self, x: int, y: int) -> list:
        """Return items stored at ``(x, y)``."""
        return self.items.get((x, y), [])

    def get_actor_at_location(self, x: int, y: int) -> Actor | None:
        """Return an actor at a location, prioritizing blockers."""
        actors_at_point = self.actor_spatial_index.get_at_point(x, y)
        if not actors_at_point:
            return None
        for actor in actors_at_point:
            if getattr(actor, "blocks_movement", False):
                return actor
        return actors_at_point[0]

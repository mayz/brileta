"""Item spawning system for materializing items as world actors."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard

from brileta import colors
from brileta.game.actors import Actor, ItemPile
from brileta.game.countables import CountableType
from brileta.game.items.item_core import Item
from brileta.types import WorldTileCoord, WorldTilePos

if TYPE_CHECKING:
    from brileta.game.game_world import GameWorld


class ItemSpawner:
    """Handles spawning items as actors in the game world with smart placement."""

    def __init__(self, game_world: GameWorld) -> None:
        self.game_world = game_world

    def spawn_item(
        self,
        item: Item,
        x: WorldTileCoord,
        y: WorldTileCoord,
        *,
        consolidate: bool = True,
    ) -> Actor:
        """Spawn an item on the ground, optionally consolidating with existing piles."""
        # Handle consolidation if requested
        if consolidate:
            existing = self.game_world.get_actor_at_location(x, y)
            if self._can_consolidate_with(existing, item):
                success, _ = existing.inventory.add_item(item)
                if success:
                    self._update_pile_appearance(existing)
                    return existing

        final_x, final_y = self._find_valid_spawn_location(x, y)
        return self._create_ground_actor([item], final_x, final_y)

    def spawn_multiple(self, items: list[Item], x: int, y: int) -> Actor:
        """Efficiently spawn multiple items as a single pile."""
        if not items:
            raise ValueError("Cannot spawn empty item list")

        final_x, final_y = self._find_valid_spawn_location(x, y)
        return self._create_ground_actor(items, final_x, final_y)

    def spawn_ground_countable(
        self,
        position: tuple[int, int],
        countable_type: CountableType,
        amount: int,
    ) -> ItemPile:
        """Spawn countables on the ground, consolidating with existing piles.

        If an item pile already exists at the position, the countables are added
        to it. Otherwise, a new empty pile is created to hold the countables.

        Args:
            position: The (x, y) world tile position.
            countable_type: The type of countable to spawn.
            amount: The quantity to spawn.

        Returns:
            The ItemPile containing the countables (existing or newly created).
        """
        x, y = position

        # Check all actors at position for an existing ItemPile
        # (get_actor_at_location prioritizes blocking actors like the player)
        for actor in self.game_world.actor_spatial_index.get_at_point(x, y):
            if isinstance(actor, ItemPile):
                actor.inventory.add_countable(countable_type, amount)
                return actor

        # Create a new pile with no items, just countables
        pile = ItemPile(
            x=x,
            y=y,
            ch="%",  # same as regular item piles
            color=colors.WHITE,
            name="Countables",
            items=None,
            game_world=self.game_world,
        )
        pile.inventory.add_countable(countable_type, amount)
        self.game_world.add_actor(pile)
        return pile

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _can_consolidate_with(
        self, actor: Actor | None, item: Item
    ) -> TypeGuard[ItemPile]:
        """Check if an item can be added to an existing item pile."""
        return isinstance(actor, ItemPile)

    def _create_ground_actor(self, items: list[Item], x: int, y: int) -> ItemPile:
        """Create a new item pile containing the specified items."""
        if not items:
            raise ValueError("Cannot create item pile with no items")

        ch, color = self._get_pile_appearance(items)

        if len(items) == 1:
            name = f"Dropped {items[0].name}"
        else:
            name = f"Item pile ({len(items)} items)"

        pile = ItemPile(
            x=x,
            y=y,
            ch=ch,
            color=color,
            name=name,
            items=items,
            game_world=self.game_world,
        )

        self.game_world.add_actor(pile)
        return pile

    def _find_valid_spawn_location(
        self, x: WorldTileCoord, y: WorldTileCoord
    ) -> WorldTilePos:
        """Find a valid location to spawn items, preferring the target location."""
        game_map = self.game_world.game_map

        if (
            0 <= x < game_map.width
            and 0 <= y < game_map.height
            and game_map.walkable[x, y]
        ):
            return x, y

        for radius in range(1, 4):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    new_x, new_y = x + dx, y + dy
                    if (
                        0 <= new_x < game_map.width
                        and 0 <= new_y < game_map.height
                        and game_map.walkable[new_x, new_y]
                    ):
                        return new_x, new_y

        return x, y

    def _get_pile_appearance(self, items: list[Item]) -> tuple[str, colors.Color]:
        """Determine visual appearance for an item pile."""
        if not items:
            return "%", colors.WHITE
        if len(items) == 1:
            return "%", colors.WHITE
        if len(items) <= 3:
            return "%", colors.LIGHT_GREY
        return "#", colors.LIGHT_GREY

    def _update_pile_appearance(self, pile: ItemPile) -> None:
        """Update the appearance of an existing item pile based on current contents."""
        items = pile.inventory.get_items()
        pile.ch, pile.color = self._get_pile_appearance(items)
        if len(items) == 1:
            pile.name = f"Dropped {items[0].name}"
        else:
            pile.name = f"Item pile ({len(items)} items)"

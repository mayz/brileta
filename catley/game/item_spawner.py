"""Item spawning system for materializing items as world actors."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from catley import colors
from catley.game.actors import Actor, Character, components
from catley.game.items.item_core import Item

if TYPE_CHECKING:
    from catley.game.game_world import GameWorld


class ItemSpawner:
    """Handles spawning items as actors in the game world with smart placement."""

    def __init__(self, game_world: GameWorld) -> None:
        self.game_world = game_world

    def spawn_item(
        self, item: Item, x: int, y: int, *, consolidate: bool = True
    ) -> Actor:
        """Spawn an item on the ground, optionally consolidating with existing piles."""
        # Handle consolidation if requested
        if consolidate:
            existing = self.game_world.get_actor_at_location(x, y)
            if self._can_consolidate_with(existing, item):
                success, _, _ = existing.inventory.add_to_inventory(item)  # type: ignore[union-attr]
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _can_consolidate_with(self, actor: Actor | None, item: Item) -> bool:
        """Check if an item can be added to an existing actor."""
        return (
            actor is not None
            and getattr(actor, "inventory", None) is not None
            and not actor.blocks_movement
            and not isinstance(actor, Character)
        )

    def _create_ground_actor(self, items: list[Item], x: int, y: int) -> Actor:
        """Create a new ground actor containing the specified items."""
        if not items:
            raise ValueError("Cannot create ground actor with no items")

        ch, color = self._get_pile_appearance(items)

        if len(items) == 1:
            name = f"Dropped {items[0].name}"
        else:
            name = f"Item pile ({len(items)} items)"

        ground_actor = Actor(
            x=x,
            y=y,
            ch=ch,
            color=color,
            name=name,
            game_world=self.game_world,
            blocks_movement=False,
            inventory=components.InventoryComponent(components.StatsComponent()),
        )
        inv = cast(components.InventoryComponent, ground_actor.inventory)
        for item in items:
            inv.add_to_inventory(item)

        self.game_world.add_actor(ground_actor)
        return ground_actor

    def _find_valid_spawn_location(self, x: int, y: int) -> tuple[int, int]:
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

    def _update_pile_appearance(self, pile_actor: Actor) -> None:
        """Update the appearance of an existing item pile based on current contents."""
        if not getattr(pile_actor, "inventory", None):
            return

        items = [item for item in pile_actor.inventory if isinstance(item, Item)]  # type: ignore[union-attr]
        pile_actor.ch, pile_actor.color = self._get_pile_appearance(items)
        if len(items) == 1:
            pile_actor.name = f"Dropped {items[0].name}"
        else:
            pile_actor.name = f"Item pile ({len(items)} items)"

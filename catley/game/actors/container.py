"""Container actors for item storage in the game world.

Containers are static actors that can hold items and be searched/looted.
Examples include crates, chests, lockers, barrels, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.types import WorldTileCoord

from .components import ContainerStorage
from .core import Actor

if TYPE_CHECKING:
    from catley.game.game_world import GameWorld
    from catley.game.items.item_core import Item


class Container(Actor):
    """A static container that holds items and can be searched.

    Containers are non-character actors that exist in the world primarily
    to store items. They can be searched by the player to transfer items
    to/from their inventory.

    Unlike characters, containers:
    - Have fixed-capacity storage (not stats-based)
    - Cannot move or take actions
    - Have no health, equipment slots, or conditions
    - Use ContainerStorage instead of CharacterInventory
    """

    def __init__(
        self,
        x: WorldTileCoord,
        y: WorldTileCoord,
        ch: str = "~",
        color: colors.Color = colors.LIGHT_GREY,
        name: str = "Container",
        capacity: int = 10,
        items: list[Item] | None = None,
        game_world: GameWorld | None = None,
        blocks_movement: bool = True,
    ) -> None:
        """Create a container actor.

        Args:
            x: X coordinate in world tiles
            y: Y coordinate in world tiles
            ch: Display character for the container
            color: Display color
            name: Name shown when examining/searching
            capacity: Maximum number of items this container can hold
            items: Initial items to place in the container
            game_world: Reference to the game world
            blocks_movement: Whether this container blocks movement
        """
        # Create storage first (without actor reference)
        storage = ContainerStorage(capacity=capacity, actor=None)

        super().__init__(
            x=x,
            y=y,
            ch=ch,
            color=color,
            name=name,
            inventory=storage,
            game_world=game_world,
            blocks_movement=blocks_movement,
        )

        # Set back-reference now that self exists
        storage.actor = self

        # Add initial items
        if items:
            for item in items:
                storage.add_item(item)

    # Type narrowing - inventory is always ContainerStorage for containers
    inventory: ContainerStorage


# === Factory Functions ===


def create_crate(
    x: WorldTileCoord,
    y: WorldTileCoord,
    items: list[Item] | None = None,
    game_world: GameWorld | None = None,
    capacity: int = 8,
) -> Container:
    """Create a wooden crate container.

    Wooden crates are common storage containers found throughout the wasteland.
    They typically contain junk, supplies, or occasional useful items.
    """
    return Container(
        x=x,
        y=y,
        ch="=",  # Stacked planks appearance
        color=(139, 90, 43),  # Wood brown
        name="Wooden Crate",
        capacity=capacity,
        items=items,
        game_world=game_world,
        blocks_movement=True,
    )


def create_locker(
    x: WorldTileCoord,
    y: WorldTileCoord,
    items: list[Item] | None = None,
    game_world: GameWorld | None = None,
    capacity: int = 12,
) -> Container:
    """Create a metal locker container.

    Metal lockers are sturdy storage containers often found in facilities,
    bunkers, and offices. They have higher capacity than crates.
    """
    return Container(
        x=x,
        y=y,
        ch="L",  # L for Locker
        color=(128, 128, 140),  # Metal grey-blue
        name="Metal Locker",
        capacity=capacity,
        items=items,
        game_world=game_world,
        blocks_movement=True,
    )


def create_barrel(
    x: WorldTileCoord,
    y: WorldTileCoord,
    items: list[Item] | None = None,
    game_world: GameWorld | None = None,
    capacity: int = 6,
) -> Container:
    """Create a barrel container.

    Barrels are cylindrical containers that can hold a modest number of items.
    Common in industrial areas and storage facilities.
    """
    return Container(
        x=x,
        y=y,
        ch="O",
        color=(100, 70, 40),  # Dark wood/metal
        name="Barrel",
        capacity=capacity,
        items=items,
        game_world=game_world,
        blocks_movement=True,
    )


def create_footlocker(
    x: WorldTileCoord,
    y: WorldTileCoord,
    items: list[Item] | None = None,
    game_world: GameWorld | None = None,
    capacity: int = 10,
) -> Container:
    """Create a footlocker container.

    Footlockers are personal storage chests often found in barracks,
    bedrooms, and living quarters.
    """
    return Container(
        x=x,
        y=y,
        ch="+",  # Chest-like appearance
        color=(80, 60, 40),  # Dark leather/wood
        name="Footlocker",
        capacity=capacity,
        items=items,
        game_world=game_world,
        blocks_movement=False,  # Can step over footlockers
    )

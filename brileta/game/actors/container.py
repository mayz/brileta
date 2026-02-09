"""Container and item pile actors for item storage in the game world.

Containers are static actors that can hold items and be searched/looted.
Examples include crates, chests, lockers, barrels, etc.

ItemPiles are ephemeral actors representing dropped loot on the ground.
They disappear when emptied and don't block movement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from brileta import colors
from brileta.game.countables import get_countable_display_name
from brileta.types import WorldTileCoord

from .components import ContainerStorage
from .core import Actor, CharacterLayer

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actions.discovery import ActionOption
    from brileta.game.actors.core import Character
    from brileta.game.game_world import GameWorld
    from brileta.game.items.item_core import Item


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
        character_layers: list[CharacterLayer] | None = None,
        visual_scale: float = 1.0,
    ) -> None:
        """Create a container actor.

        Args:
            x: X coordinate in world tiles
            y: Y coordinate in world tiles
            ch: Display character for the container (fallback if no layers)
            color: Display color (fallback if no layers)
            name: Name shown when examining/searching
            capacity: Maximum number of items this container can hold
            items: Initial items to place in the container
            game_world: Reference to the game world
            blocks_movement: Whether this container blocks movement
            character_layers: Optional multi-character visual composition
            visual_scale: Scale factor for rendering (1.0 = normal size)
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
            character_layers=character_layers,
            visual_scale=visual_scale,
        )

        # Set back-reference now that self exists
        storage.actor = self

        # Add initial items
        if items:
            for item in items:
                storage.add_item(item)

    # Type narrowing - inventory is always ContainerStorage for containers
    inventory: ContainerStorage

    def get_target_description(self) -> str | None:
        """Return a short description for the action panel."""
        return "A container"

    def get_contextual_actions(
        self, controller: Controller, player: Character
    ) -> list[ActionOption]:
        """Return context-sensitive actions for this container."""
        from brileta.game.actions.discovery import ActionDiscovery

        discovery = (
            controller.action_discovery
            if hasattr(controller, "action_discovery")
            else ActionDiscovery()
        )
        context = discovery.context_builder.build_context(controller, player)
        return discovery.environment_discovery.discover_environment_actions_for_tile(
            controller,
            player,
            context,
            self.x,
            self.y,
        )


class ItemPile(Actor):
    """An ephemeral pile of items dropped on the ground.

    Item piles represent loot that can be picked up by the player. Unlike
    containers, item piles:
    - Never block movement (you walk over them)
    - Disappear when emptied
    - Use "Pick up" rather than "Search" as the interaction verb

    Item piles are created by the ItemSpawner when items are dropped or
    spawned in the world.
    """

    def __init__(
        self,
        x: WorldTileCoord,
        y: WorldTileCoord,
        ch: str = "%",
        color: colors.Color = colors.WHITE,
        name: str = "Item pile",
        items: list[Item] | None = None,
        game_world: GameWorld | None = None,
    ) -> None:
        """Create an item pile actor.

        Args:
            x: X coordinate in world tiles
            y: Y coordinate in world tiles
            ch: Display character (defaults to %)
            color: Display color
            name: Name shown when examining
            items: Initial items in the pile
            game_world: Reference to the game world
        """
        # Item piles use ContainerStorage with high capacity
        storage = ContainerStorage(capacity=100, actor=None)

        super().__init__(
            x=x,
            y=y,
            ch=ch,
            color=color,
            name=name,
            inventory=storage,
            game_world=game_world,
            blocks_movement=False,  # Always walkable
        )

        # Set back-reference now that self exists
        storage.actor = self

        # Add initial items
        if items:
            for item in items:
                storage.add_item(item)

    # Type narrowing - inventory is always ContainerStorage for item piles
    inventory: ContainerStorage

    def get_target_description(self) -> str | None:
        """Return a short description for the action panel."""
        items = list(self.inventory)
        has_countables = bool(self.inventory.countables)
        if len(items) == 1 and not has_countables:
            return "An item on the ground"
        return "On the ground"

    def get_contextual_actions(
        self, controller: Controller, player: Character
    ) -> list[ActionOption]:
        """Return context-sensitive actions for this item pile."""
        from brileta.game.actions.discovery import ActionCategory, ActionOption
        from brileta.game.actions.misc import PickupItemsPlan

        distance = max(abs(self.x - player.x), abs(self.y - player.y))
        if distance == 0:
            return []

        item_x, item_y = self.x, self.y

        def create_pathfind_and_pickup(x: int, y: int):
            def pathfind_and_pickup():
                return controller.start_plan(
                    player,
                    PickupItemsPlan,
                    target_position=(x, y),
                )

            return pathfind_and_pickup

        pickup_action = ActionOption(
            id="pickup-walk",
            name="Walk to and pick up",
            description="Move to the items and pick them up",
            category=ActionCategory.ITEMS,
            action_class=None,
            requirements=[],
            static_params={},
            execute=create_pathfind_and_pickup(item_x, item_y),
        )
        return [pickup_action]

    def is_empty(self) -> bool:
        """Check if this pile has no items and no countables."""
        return len(self.inventory.get_items()) == 0 and not self.inventory.countables

    @property
    def display_name(self) -> str:
        """Get a descriptive name for this item pile.

        Includes both items and countables in the description.
        """
        item_count = len(self.inventory.get_items())
        countable_parts = [
            get_countable_display_name(ct, qty)
            for ct, qty in self.inventory.countables.items()
        ]

        if item_count and countable_parts:
            return f"Items and {', '.join(countable_parts)}"
        if countable_parts:
            return ", ".join(countable_parts)
        if item_count == 1:
            return self.inventory.get_items()[0].name
        return f"Item pile ({item_count} items)"


# === Factory Functions ===


def create_bookcase(
    x: WorldTileCoord,
    y: WorldTileCoord,
    items: list[Item] | None = None,
    game_world: GameWorld | None = None,
    capacity: int = 12,
) -> Container:
    """Create a bookcase container with a clean visual composition.

    Uses a full-size shelf frame with small scaled book glyphs for a
    recognizable silhouette. Bookcases are commonly found in libraries,
    offices, and living quarters.
    """
    # Wood frame color (dark brown)
    frame_color: colors.Color = (70, 45, 25)

    # Book spine colors - high contrast for visibility at small scale
    book_colors: list[colors.Color] = [
        (180, 50, 50),  # Bright red
        (50, 70, 160),  # Deep blue
        (220, 200, 150),  # Cream/parchment
    ]

    # Build the character layers for the bookcase composition.
    # Uses left and right brackets to create a full frame, with books inside.
    layers: list[CharacterLayer] = [
        # Left and right brackets form the bookcase frame
        CharacterLayer("[", frame_color, offset_x=-0.2, offset_y=0.0),
        CharacterLayer("]", frame_color, offset_x=0.2, offset_y=0.0),
        # Book spines - uniform height, varied widths
        CharacterLayer(
            "|", book_colors[0], offset_x=-0.1, offset_y=0.0, scale_x=0.8, scale_y=0.85
        ),
        CharacterLayer(
            "|", book_colors[1], offset_x=0.0, offset_y=0.0, scale_x=0.6, scale_y=0.85
        ),
        CharacterLayer(
            "|", book_colors[2], offset_x=0.1, offset_y=0.0, scale_x=0.75, scale_y=0.85
        ),
    ]

    return Container(
        x=x,
        y=y,
        ch="[",  # Fallback glyph if layers not rendered
        color=frame_color,
        name="Bookcase",
        capacity=capacity,
        items=items,
        game_world=game_world,
        blocks_movement=True,
        character_layers=layers,
        visual_scale=1.2,  # Slight scale up for presence
    )

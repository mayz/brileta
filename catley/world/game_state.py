from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.config import PLAYER_BASE_TOUGHNESS
from catley.game.actors import Actor, make_pc
from catley.render.lighting import LightingSystem, LightSource

from .map import GameMap

if TYPE_CHECKING:
    from catley.game.items.item_core import Item


class GameWorld:
    """
    Represents the complete state of the game world.

    Includes the game map, all actors (player, NPCs, items), their properties, and the
    core game rules that govern how these elements interact. Does not handle input,
    rendering, or high-level application flow. Its primary responsibility is to be
    the single source of truth for the game's state.
    """

    def __init__(self, map_width: int, map_height: int) -> None:
        self.mouse_tile_location_on_map: tuple[int, int] | None = None
        self.lighting = LightingSystem()
        self.selected_actor: Actor | None = None

        from catley.game.items.item_types import PISTOL_MAGAZINE_TYPE, PISTOL_TYPE

        # Create player with a torch light source
        player_light = LightSource.create_torch()
        self.player = make_pc(
            x=0,
            y=0,
            ch="@",
            name="Player",
            color=colors.PLAYER_COLOR,
            game_world=self,
            light_source=player_light,
            toughness=PLAYER_BASE_TOUGHNESS,
            starting_weapon=PISTOL_TYPE.create(),
            # Other abilities will default to 0
        )

        # Give the player some ammo
        self.player.inventory.add_to_inventory(PISTOL_MAGAZINE_TYPE.create())

        self.actors = [self.player]
        self.game_map = GameMap(map_width, map_height)

    def update_player_light(self) -> None:
        """Update player light source position"""
        if self.player.light_source:
            self.player.light_source.position = (self.player.x, self.player.y)

    def get_pickable_items_at_location(self, x: int, y: int) -> list[Item]:
        """Get all pickable items at the specified location.

        Currently, this includes items from dead actors' inventories and their
        equipped weapons.
        """
        items_found: list[Item] = []
        # Check items from dead actors at this location
        for actor in self.actors:
            if (
                actor.x == x
                and actor.y == y
                and actor.health
                and not actor.health.is_alive()  # Only from dead actors
            ):
                items_found.extend(actor.inventory)
                if actor.inventory.equipped_weapon:
                    items_found.append(
                        actor.inventory.equipped_weapon
                    )  # Add equipped weapon
        # Future: Add items directly on the ground if we implement that
        # e.g., items_found.extend(self.game_map.get_items_on_ground(x,y))
        return items_found

    def get_actor_at_location(self, x: int, y: int) -> Actor | None:
        """Return the first actor found at the given location, or None."""
        for actor in self.actors:
            if actor.x == x and actor.y == y:
                return actor
        return None

    def has_pickable_items_at_location(self, x: int, y: int) -> bool:
        """Check if there are any pickable items at the specified location."""
        return bool(self.get_pickable_items_at_location(x, y))

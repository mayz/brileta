import random

from catley import colors
from catley.config import PLAYER_BASE_STRENGTH, PLAYER_BASE_TOUGHNESS
from catley.environment.map import GameMap
from catley.game import conditions
from catley.game.actors import Actor, Character
from catley.game.items.item_core import Item
from catley.game.items.item_types import (
    COMBAT_KNIFE_TYPE,
    PISTOL_MAGAZINE_TYPE,
    PISTOL_TYPE,
    REVOLVER_TYPE,
    RIFLE_MAGAZINE_TYPE,
    SNIPER_RIFLE_TYPE,
)
from catley.util.spatial import SpatialHashGrid, SpatialIndex
from catley.view.effects.lighting import LightingSystem, LightSource


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

        # Create player with a torch light source
        from catley.game.actors import PC

        player_light = LightSource.create_torch()
        self.player = PC(
            x=0,
            y=0,
            ch="@",
            name="Player",
            color=colors.PLAYER_COLOR,
            game_world=self,
            light_source=player_light,
            strength=PLAYER_BASE_STRENGTH,
            toughness=PLAYER_BASE_TOUGHNESS,
            starting_weapon=PISTOL_TYPE.create(),
            # Other abilities will default to 0
        )

        self.player.inventory.equip_to_slot(SNIPER_RIFLE_TYPE.create(), 1)
        self.player.inventory.add_to_inventory(COMBAT_KNIFE_TYPE.create())

        # Give the player some ammo
        self.player.inventory.add_to_inventory(PISTOL_MAGAZINE_TYPE.create())
        self.player.inventory.add_to_inventory(PISTOL_MAGAZINE_TYPE.create())
        self.player.inventory.add_to_inventory(RIFLE_MAGAZINE_TYPE.create())
        self.player.inventory.add_to_inventory(RIFLE_MAGAZINE_TYPE.create())

        self.player.inventory.add_to_inventory(
            conditions.Injury(injury_type="Sprained Ankle")
        )

        self.actors: list[Actor] = []
        self.actor_spatial_index: SpatialIndex[Actor] = SpatialHashGrid(cell_size=16)

        # The player must now be added via the new lifecycle method.
        self.add_actor(self.player)
        self.game_map = GameMap(map_width, map_height)
        self.lighting.set_game_map(self.game_map)

    def add_actor(self, actor: Actor) -> None:
        """Adds an actor to the world and registers it with the spatial index."""
        self.actors.append(actor)
        self.actor_spatial_index.add(actor)

    def remove_actor(self, actor: Actor) -> None:
        """Removes an actor from the world and spatial index."""
        try:
            self.actors.remove(actor)
            self.actor_spatial_index.remove(actor)
        except ValueError:
            # Actor was not in the list; ignore.
            pass

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
                and isinstance(actor, Character)
                and not actor.health.is_alive()  # Only from dead actors
            ):
                # Add all equipped items from all slots
                attack_slots = actor.inventory.attack_slots
                items_found.extend(item for item in attack_slots if item)

            elif (
                actor.x == x
                and actor.y == y
                and hasattr(actor, "inventory")
                and actor.inventory is not None
                and not isinstance(actor, Character)
            ):
                items_found.extend(
                    item for item in list(actor.inventory) if isinstance(item, Item)
                )
                items_found.extend(
                    item for item in actor.inventory.attack_slots if item is not None
                )

        # Future: Add items directly on the ground if we implement that
        # e.g., items_found.extend(self.game_map.get_items_on_ground(x,y))
        return items_found

    def get_actor_at_location(self, x: int, y: int) -> Actor | None:
        """Return an actor at the given location using the spatial index.

        Prioritizes returning a blocking actor if multiple actors are present.
        """
        actors_at_point = self.actor_spatial_index.get_at_point(x, y)
        if not actors_at_point:
            return None

        # Prefer blocking actors (e.g., NPCs) so that selecting or targeting
        # chooses solid objects over items or corpses that may occupy the same
        # tile.
        for actor in actors_at_point:
            if actor.blocks_movement:
                return actor

        # If no blocking actor exists, return whichever actor is found first.
        return actors_at_point[0]

    def has_pickable_items_at_location(self, x: int, y: int) -> bool:
        """Check if there are any pickable items at the specified location."""
        return bool(self.get_pickable_items_at_location(x, y))

    def populate_npcs(
        self, rooms: list, num_npcs: int = 10, max_attempts_per_npc: int = 10
    ) -> None:
        """Add NPCs to random locations in rooms."""
        from catley.game.actors import NPC
        from catley.game.items.item_types import SLEDGEHAMMER_TYPE

        for npc_index in range(num_npcs):
            placed = False

            for _ in range(max_attempts_per_npc):
                # Pick a random room
                room = random.choice(rooms)

                # Pick a random tile within the room (avoiding walls)
                npc_x = random.randint(room.x1 + 1, room.x2 - 2)
                npc_y = random.randint(room.y1 + 1, room.y2 - 2)

                # Check if tile is walkable and free
                if (
                    self.game_map.walkable[npc_x, npc_y]
                    and self.get_actor_at_location(npc_x, npc_y) is None
                ):
                    # Place the NPC
                    if False:
                        npc = NPC(
                            x=npc_x,
                            y=npc_y,
                            ch="T",
                            name=f"Trog {npc_index + 1}" if npc_index > 0 else "Trog",
                            color=colors.DARK_GREY,
                            game_world=self,
                            blocks_movement=True,
                            weirdness=3,
                            strength=3,
                            toughness=3,
                            intelligence=-3,
                            speed=80,
                            starting_weapon=SLEDGEHAMMER_TYPE.create(),
                        )
                        self.add_actor(npc)
                    npc = NPC(
                        x=npc_x,
                        y=npc_y,
                        ch="H",
                        name=f"Hackadoo {npc_index + 1}"
                        if npc_index > 0
                        else "Hackadoo",
                        color=colors.DARK_GREY,
                        game_world=self,
                        blocks_movement=True,
                        weirdness=1,
                        intelligence=2,
                        starting_weapon=REVOLVER_TYPE.create(),
                    )
                    self.add_actor(npc)
                    placed = True
                    break

            if not placed:
                print(
                    f"Could not place Trog {npc_index + 1} after "
                    f"{max_attempts_per_npc} attempts"
                )

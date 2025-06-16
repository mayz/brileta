import random

from catley import colors, config
from catley.config import PLAYER_BASE_STRENGTH, PLAYER_BASE_TOUGHNESS
from catley.environment.map import GameMap
from catley.game.actors import (
    Actor,
    Character,
    conditions,
)
from catley.game.item_spawner import ItemSpawner
from catley.game.items.item_core import Item
from catley.game.items.item_types import (
    COMBAT_KNIFE_TYPE,
    FOOD_TYPE,
    PISTOL_MAGAZINE_TYPE,
    PISTOL_TYPE,
    REVOLVER_TYPE,
    RIFLE_MAGAZINE_TYPE,
    SNIPER_RIFLE_TYPE,
)
from catley.input_handler import WorldTileCoord
from catley.util.coordinates import Rect
from catley.util.spatial import SpatialHashGrid, SpatialIndex
from catley.view.render.effects.lighting import LightingSystem, LightSource


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
        self.item_spawner = ItemSpawner(self)
        self.selected_actor: Actor | None = None

        self._init_actor_storage()
        self.game_map, rooms = self._generate_map(map_width, map_height)
        self.lighting.set_game_map(self.game_map)

        self.player = self._create_player()
        self.add_actor(self.player)
        self._position_player(rooms[0])
        self._add_starting_injury()

        self._populate_npcs(rooms)

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

    # ---------------------------------------------------------------------
    # Initialization helpers
    # ---------------------------------------------------------------------

    def _init_actor_storage(self) -> None:
        """Initialize the collections used to track actors."""
        self.actors: list[Actor] = []
        self.actor_spatial_index: SpatialIndex[Actor] = SpatialHashGrid(cell_size=16)

    def _create_player(self) -> Character:
        """Instantiate the player character and starting inventory."""
        from catley.game.actors import PC

        light = LightSource.create_torch()
        player = PC(
            x=0,
            y=0,
            ch="@",
            name="Player",
            color=colors.PLAYER_COLOR,
            game_world=self,
            light_source=light,
            strength=PLAYER_BASE_STRENGTH,
            toughness=PLAYER_BASE_TOUGHNESS,
            starting_weapon=PISTOL_TYPE.create(),
        )
        self._setup_player_inventory(player)
        return player

    def _setup_player_inventory(self, player: Character) -> None:
        """Equip the player's initial gear and ammunition."""
        player.inventory.equip_to_slot(SNIPER_RIFLE_TYPE.create(), 1)
        player.inventory.add_to_inventory(COMBAT_KNIFE_TYPE.create())

        player.inventory.add_to_inventory(PISTOL_MAGAZINE_TYPE.create())
        player.inventory.add_to_inventory(PISTOL_MAGAZINE_TYPE.create())
        player.inventory.add_to_inventory(RIFLE_MAGAZINE_TYPE.create())
        player.inventory.add_to_inventory(RIFLE_MAGAZINE_TYPE.create())

        player.inventory.add_to_inventory(FOOD_TYPE.create())

    def _generate_map(
        self, map_width: int, map_height: int
    ) -> tuple[GameMap, list[Rect]]:
        """Create the game map and return it along with generated rooms."""
        game_map = GameMap(map_width, map_height)
        game_map.gw = self
        rooms = game_map.make_map(
            config.MAX_NUM_ROOMS,
            config.MIN_ROOM_SIZE,
            config.MAX_ROOM_SIZE,
        )
        return game_map, rooms

    def _position_player(self, room: Rect) -> None:
        """Place the player in the center of ``room``."""
        # Use teleport() to sync logical and visual positions instantly
        self.player.teleport(*room.center())

    def _add_starting_injury(self) -> None:
        """Give the player their initial injury and drop any overflow items."""
        sprained_ankle = conditions.Injury(
            conditions.InjuryLocation.LEFT_LEG,
            "Sprained Ankle",
        )
        _, _, dropped_items = self.player.inventory.add_to_inventory(sprained_ankle)
        if dropped_items:
            self.spawn_ground_items(dropped_items, self.player.x, self.player.y)

    def spawn_ground_item(self, item: Item, x: int, y: int, **kwargs) -> Actor:
        """Spawn an item on the ground with smart placement and consolidation."""
        return self.item_spawner.spawn_item(item, x, y, **kwargs)

    def spawn_ground_items(self, items: list[Item], x: int, y: int) -> Actor:
        """Spawn multiple items efficiently as a single pile."""
        return self.item_spawner.spawn_multiple(items, x, y)

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

    def get_actor_at_location(
        self, x: WorldTileCoord, y: WorldTileCoord
    ) -> Actor | None:
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

    def _populate_npcs(
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
                # NPC could not be placed after several attempts; skip it
                pass

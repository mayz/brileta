import math
import random

from catley import colors, config
from catley.config import PLAYER_BASE_STRENGTH, PLAYER_BASE_TOUGHNESS
from catley.environment.generators import RoomsAndCorridorsGenerator
from catley.environment.map import GameMap, MapRegion
from catley.environment.tile_types import TileTypeID
from catley.game.actors import NPC, PC, Actor, Character, create_bookcase
from catley.game.actors.environmental import ContainedFire
from catley.game.countables import CountableType
from catley.game.enums import CreatureSize
from catley.game.item_spawner import ItemSpawner
from catley.game.items.item_core import Item
from catley.game.items.item_types import (
    ALARM_CLOCK_TYPE,
    COMBAT_KNIFE_TYPE,
    FOOD_TYPE,
    HUNTING_SHOTGUN_TYPE,
    PISTOL_MAGAZINE_TYPE,
    PISTOL_TYPE,
    REVOLVER_TYPE,
    SHOTGUN_SHELLS_TYPE,
    SLEDGEHAMMER_TYPE,
)
from catley.game.lights import DirectionalLight, DynamicLight, GlobalLight, LightSource
from catley.game.outfit import LEATHER_ARMOR_TYPE
from catley.types import TileCoord, WorldTileCoord, WorldTilePos
from catley.util.coordinates import Rect
from catley.util.spatial import SpatialHashGrid, SpatialIndex
from catley.view.render.lighting.base import LightingConfig, LightingSystem

# Cache the default sun azimuth to avoid repeated instantiation
_DEFAULT_SUN_AZIMUTH = LightingConfig().sun_azimuth_degrees


class GameWorld:
    """
    Represents the complete state of the game world.

    Includes the game map, all actors (player, NPCs, items), their properties, and the
    core game rules that govern how these elements interact. Does not handle input,
    rendering, or high-level application flow. Its primary responsibility is to be
    the single source of truth for the game's state.
    """

    def __init__(self, map_width: TileCoord, map_height: TileCoord) -> None:
        self.mouse_tile_location_on_map: WorldTilePos | None = None
        self.item_spawner = ItemSpawner(self)
        self.selected_actor: Actor | None = None

        self.lights: list[LightSource] = []  # All light sources in the world
        self.lighting_system: LightingSystem | None = None

        self._init_actor_storage()

        self.game_map, rooms = self._generate_map(map_width, map_height)

        if not rooms:
            raise RuntimeError(
                "Map generation produced no rooms - check generator parameters"
            )

        self.player = self._create_player()
        self.add_actor(self.player)
        self._position_player(rooms[0])

        # Note: Rooms now have random 20% chance of being outdoor (set in generator)

        self._populate_npcs(rooms)

        self._place_containers(rooms)

        if not config.IS_TEST_ENVIRONMENT:
            self._add_test_fire(rooms)
            self._add_starting_room_items(rooms[0])

    def add_actor(self, actor: Actor) -> None:
        """Adds an actor to the world and registers it with the spatial index."""
        self.actors.append(actor)
        self.actor_spatial_index.add(actor)
        # Register actor by its Python object id for O(1) lookup.
        self._actor_id_registry[id(actor)] = actor

    def remove_actor(self, actor: Actor) -> None:
        """Removes an actor from the world and spatial index."""
        try:
            self.actors.remove(actor)
            self.actor_spatial_index.remove(actor)
        except ValueError:
            # Actor was not in the list; ignore.
            pass
        # Always attempt to unregister from the id registry.
        self._actor_id_registry.pop(id(actor), None)

    def add_light(self, light: LightSource) -> None:
        """Add a light source to the world and notify the lighting system.

        Args:
            light: The light source to add
        """
        self.lights.append(light)
        if self.lighting_system is not None:
            self.lighting_system.on_light_added(light)

    def remove_light(self, light: LightSource) -> None:
        """Remove a light source from the world and notify the lighting system.

        Args:
            light: The light source to remove
        """
        try:
            self.lights.remove(light)
            if self.lighting_system is not None:
                self.lighting_system.on_light_removed(light)
        except ValueError:
            # Light was not in the list; ignore.
            pass

    def get_global_lights(self) -> list[GlobalLight]:
        """Get all global lights (sun, moon, etc.) in the world.

        Returns:
            A list of all GlobalLight instances
        """
        return [light for light in self.lights if isinstance(light, GlobalLight)]

    def get_static_lights(self) -> list[LightSource]:
        """Get all static point lights in the world.

        Returns:
            A list of all static LightSource instances (excluding global lights)
        """
        return [
            light
            for light in self.lights
            if light.is_static() and not isinstance(light, GlobalLight)
        ]

    def set_time_of_day(self, time_hours: float) -> None:
        """Update sun position based on time of day and invalidate lighting cache.

        Args:
            time_hours: Time in 24-hour format (0.0 = midnight, 12.0 = noon)
        """
        # Update sun direction based on time for all directional lights
        for light in self.get_global_lights():
            if isinstance(light, DirectionalLight):
                # Calculate sun elevation (peaks at noon)
                # Simple sinusoidal model: 0° at sunrise/sunset, 90° at noon
                time_normalized = (
                    time_hours - 6.0
                ) / 12.0  # -0.5 to 0.5 (sunrise to sunset)
                if -0.5 <= time_normalized <= 0.5:
                    # Sun is above horizon
                    elevation_rad = math.pi * (0.5 - abs(time_normalized))  # 0 to π/2
                    elevation_degrees = math.degrees(elevation_rad)

                    # Update the light with new sun position using shared defaults.
                    azimuth = _DEFAULT_SUN_AZIMUTH
                    light.direction = DirectionalLight.create_sun(
                        elevation_degrees=elevation_degrees,
                        azimuth_degrees=azimuth,
                        intensity=light.intensity,
                        color=light.color,
                    ).direction
                else:
                    # Sun is below horizon (night)
                    light.intensity = 0.0

        # Invalidate lighting cache since global lighting changed
        if self.lighting_system:
            self.lighting_system.on_global_light_changed()

    def set_region_sky_exposure(self, world_pos: WorldTilePos, exposure: float) -> bool:
        """Debug function to set sky exposure for a region at given position.

        Args:
            world_pos: World tile position to find the region
            exposure: Sky exposure value (0.0 = no sky, 1.0 = full sky)

        Returns:
            True if region was found and modified, False otherwise
        """
        if not self.game_map:
            return False

        region = self.game_map.get_region_at(world_pos)
        if region:
            region.sky_exposure = exposure
            # Invalidate lighting cache since global lighting conditions changed
            if self.lighting_system:
                self.lighting_system.on_global_light_changed()
            # Invalidate appearance caches since region properties changed
            if self.game_map:
                self.game_map.invalidate_appearance_caches()
            return True
        return False

    # ---------------------------------------------------------------------
    # Initialization helpers
    # ---------------------------------------------------------------------

    def _init_actor_storage(self) -> None:
        """Initialize the collections used to track actors."""
        self.actors: list[Actor] = []
        self.actor_spatial_index: SpatialIndex[Actor] = SpatialHashGrid(cell_size=16)
        # Registry for O(1) actor lookup by Python object id.
        # Used by floating text system to track actor positions.
        self._actor_id_registry: dict[int, Actor] = {}

    def _create_player(self) -> Character:
        """Instantiate the player character and starting inventory."""
        player = PC(
            x=0,
            y=0,
            ch="@",
            name="Player",
            color=colors.PLAYER_COLOR,
            game_world=self,
            strength=PLAYER_BASE_STRENGTH,
            toughness=PLAYER_BASE_TOUGHNESS,
            starting_weapon=PISTOL_TYPE.create(),
        )
        self._setup_player_inventory(player)

        return player

    def _setup_player_inventory(self, player: Character) -> None:
        """Equip the player's initial gear and ammunition."""
        player.inventory.equip_to_slot(HUNTING_SHOTGUN_TYPE.create(), 1)
        player.inventory.add_to_inventory(COMBAT_KNIFE_TYPE.create())

        player.inventory.add_to_inventory(PISTOL_MAGAZINE_TYPE.create())
        player.inventory.add_to_inventory(PISTOL_MAGAZINE_TYPE.create())
        player.inventory.add_to_inventory(SHOTGUN_SHELLS_TYPE.create())
        player.inventory.add_to_inventory(SHOTGUN_SHELLS_TYPE.create())

        player.inventory.add_to_inventory(FOOD_TYPE.create())

        # Equip starting armor
        armor_item = LEATHER_ARMOR_TYPE.create()
        player.inventory.set_starting_outfit(armor_item)

        # Starting money
        player.inventory.add_countable(CountableType.COIN, 50)

    def _generate_map(
        self, map_width: int, map_height: int
    ) -> tuple[GameMap, list[Rect]]:
        """Create the game map and return it along with generated rooms."""
        generator = RoomsAndCorridorsGenerator(
            map_width,
            map_height,
            max_rooms=config.MAX_NUM_ROOMS,
            min_room_size=config.MIN_ROOM_SIZE,
            max_room_size=config.MAX_ROOM_SIZE,
        )
        map_data = generator.generate()

        game_map = GameMap(map_width, map_height, map_data)
        game_map.gw = self

        room_regions = [r for r in map_data.regions.values() if r.region_type == "room"]
        room_rects = [r.bounds[0] for r in room_regions if r.bounds]

        return game_map, room_rects

    def _setup_test_outdoor_room(self) -> None:
        """TEMPORARY: Convert starting room to outdoor for sunlight testing.

        This modifies the region containing the player's starting position
        to have sky exposure, allowing sunlight testing in an otherwise
        indoor-only scenario.
        """
        if not self.game_map:
            return

        # Get the region at player's starting position
        player_pos = (self.player.x, self.player.y)
        region = self.game_map.get_region_at(player_pos)

        if region:
            # TEMPORARY: Simulate full outdoor area for testing
            region.sky_exposure = 1.0  # Full outdoor exposure
            region.region_type = "test_outdoor"  # Mark as test

            # Convert floor tiles in this region to outdoor tiles
            self._convert_region_to_outdoor_tiles(region)

            # Invalidate lighting cache since global lighting conditions changed
            if self.lighting_system:
                self.lighting_system.on_global_light_changed()
            # Invalidate appearance caches since region properties changed
            self.game_map.invalidate_appearance_caches()

    def _convert_region_to_outdoor_tiles(self, region: MapRegion) -> None:
        """TEMPORARY: Convert indoor floor tiles to outdoor tiles in a region."""
        if not self.game_map:
            return

        # Find all tiles that belong to this region
        for x in range(self.game_map.width):
            for y in range(self.game_map.height):
                if self.game_map.tile_to_region_id[x, y] == region.id:
                    current_tile_id = self.game_map.tiles[x, y]
                    # Convert indoor tiles to outdoor variants
                    if current_tile_id == TileTypeID.FLOOR:
                        self.game_map.tiles[x, y] = TileTypeID.OUTDOOR_FLOOR
                    elif current_tile_id == TileTypeID.WALL:
                        self.game_map.tiles[x, y] = TileTypeID.OUTDOOR_WALL

        # Invalidate cached property maps since tile types changed
        self.game_map.invalidate_property_caches()

    def _position_player(self, room: Rect) -> None:
        """Place the player in the center of ``room``."""
        # Use teleport() to sync logical and visual positions instantly
        self.player.teleport(*room.center())

    def spawn_ground_item(
        self, item: Item, x: WorldTileCoord, y: WorldTileCoord, **kwargs
    ) -> Actor:
        """Spawn an item on the ground with smart placement and consolidation."""
        return self.item_spawner.spawn_item(item, x, y, **kwargs)

    def spawn_ground_items(self, items: list[Item], x: int, y: int) -> Actor:
        """Spawn multiple items efficiently as a single pile."""
        return self.item_spawner.spawn_multiple(items, x, y)

    def on_actor_moved(self, actor: Actor) -> None:
        """Notify the lighting system when an actor moves.

        This method should be called whenever an actor moves. It will:
        1. Update any dynamic lights owned by this actor
        2. Invalidate the shadow caster cache (actors cast shadows)

        Args:
            actor: The actor that moved
        """
        if self.lighting_system is not None:
            # Find and update any lights owned by this actor
            for light in self.lights:
                if isinstance(light, DynamicLight) and light.owner is actor:
                    light.position = (actor.x, actor.y)
                    self.lighting_system.on_light_moved(light)

            # Invalidate shadow caster cache since actor positions affect shadows
            self.lighting_system.on_actor_moved(actor)

    def get_pickable_items_at_location(
        self, x: WorldTileCoord, y: WorldTileCoord
    ) -> list[Item]:
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
                ready_slots = actor.inventory.ready_slots
                items_found.extend(item for item in ready_slots if item)

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
                # Only check ready_slots for CharacterInventory, not ContainerStorage
                ready_slots = getattr(actor.inventory, "ready_slots", None)
                if ready_slots is not None:
                    items_found.extend(item for item in ready_slots if item is not None)

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

    def get_actor_by_id(self, actor_id: int) -> Actor | None:
        """Look up an actor by its Python object ID in O(1) time.

        Uses the internal actor registry for constant-time lookup.
        Used by floating text system to track actor positions.
        Returns None if actor no longer exists.
        """
        return self._actor_id_registry.get(actor_id)

    def has_pickable_items_at_location(
        self, x: WorldTileCoord, y: WorldTileCoord
    ) -> bool:
        """Check if there are any pickable items at the specified location."""
        return bool(self.get_pickable_items_at_location(x, y))

    def _populate_npcs(
        self, rooms: list, num_npcs: int = 10, max_attempts_per_npc: int = 10
    ) -> None:
        """Add NPCs to random locations in rooms."""
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
                    # Place the NPC - alternate between Trogs and Hackadoos
                    if npc_index % 2 == 0:
                        # Trog: hulking mutant with melee weapon
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
                            creature_size=CreatureSize.LARGE,
                            starting_weapon=SLEDGEHAMMER_TYPE.create(),
                        )
                    else:
                        # Hackadoo: standard ranged enemy
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

    def _add_test_fire(self, rooms: list) -> None:
        """Add a test fire to demonstrate the fire system."""
        # Pick the first room to add a campfire
        if rooms:
            room = rooms[0]
            # Find a good spot for the fire (offset from center to avoid player)
            fire_x = room.x1 + 2  # Near left wall but not on it
            fire_y = room.y1 + 2  # Near top wall but not on it

            # Make sure the spot is walkable and free
            if (
                self.game_map.walkable[fire_x, fire_y]
                and self.get_actor_at_location(fire_x, fire_y) is None
            ):
                campfire = ContainedFire.create_campfire(fire_x, fire_y, self)
                self.add_actor(campfire)

            # Add both hazards for side-by-side comparison
            self._add_test_acid_pool(room)
            self._add_test_hot_coals(room)

    def _add_test_acid_pool(self, room) -> None:
        """Add a small acid pool to demonstrate hazardous terrain.

        Creates a 2x2 acid pool in the bottom-right corner of the room.
        """
        # Place acid pool in bottom-right corner (away from campfire)
        pool_x = room.x2 - 3  # 3 tiles from right wall
        pool_y = room.y2 - 3  # 3 tiles from bottom wall

        # Create a small 2x2 acid pool
        for dx in range(2):
            for dy in range(2):
                x, y = pool_x + dx, pool_y + dy
                # Only place on walkable tiles that are within bounds
                if (
                    0 <= x < self.game_map.width
                    and 0 <= y < self.game_map.height
                    and self.game_map.walkable[x, y]
                ):
                    self.game_map.tiles[x, y] = TileTypeID.ACID_POOL

    def _add_test_hot_coals(self, room) -> None:
        """Add a small patch of hot coals to demonstrate animated terrain.

        Creates a 2x2 hot coals area in the bottom-right corner of the room.
        """
        from catley.environment.tile_types import TileTypeID

        # Place hot coals in bottom-left corner (opposite acid pool)
        coals_x = room.x1 + 2  # 2 tiles from left wall
        coals_y = room.y2 - 3  # 3 tiles from bottom wall

        # Create a small 2x2 hot coals area
        for dx in range(2):
            for dy in range(2):
                x, y = coals_x + dx, coals_y + dy
                # Only place on walkable tiles that are within bounds
                if (
                    0 <= x < self.game_map.width
                    and 0 <= y < self.game_map.height
                    and self.game_map.walkable[x, y]
                ):
                    self.game_map.tiles[x, y] = TileTypeID.HOT_COALS

    def _place_containers(
        self, rooms: list, num_containers: int = 5, max_attempts: int = 10
    ) -> None:
        """Place bookcases with random loot in rooms.

        Spawns bookcases throughout the dungeon, each containing random junk items.
        Bookcases use multi-character composition for a rich visual appearance.

        Args:
            rooms: List of Rect objects representing rooms
            num_containers: Number of bookcases to place
            max_attempts: Maximum attempts to find a valid spot for each bookcase
        """
        from catley.game.items.junk_item_types import get_random_junk_type

        for _ in range(num_containers):
            placed = False

            for _ in range(max_attempts):
                # Pick a random room (skip the first room where player spawns)
                room = rooms[0] if len(rooms) <= 1 else random.choice(rooms[1:])

                # Pick a random position within the room
                container_x = random.randint(room.x1 + 1, room.x2 - 2)
                container_y = random.randint(room.y1 + 1, room.y2 - 2)

                # Check if location is valid
                if (
                    self.game_map.walkable[container_x, container_y]
                    and self.get_actor_at_location(container_x, container_y) is None
                ):
                    # Generate random items for the container
                    num_items = random.randint(1, 4)
                    items = [get_random_junk_type().create() for _ in range(num_items)]

                    # Create a bookcase
                    container = create_bookcase(
                        x=container_x,
                        y=container_y,
                        items=items,
                        game_world=self,
                    )
                    self.add_actor(container)
                    placed = True
                    break

            if not placed:
                # Container could not be placed; skip it
                pass

    def _add_starting_room_items(self, room: Rect) -> None:
        """Place discoverable items in the starting room.

        Adds some junk items in a corner for the player to find and experiment with.
        """
        # Place in top-right corner (campfire is top-left)
        item_x = room.x2 - 2
        item_y = room.y1 + 2

        if (
            self.game_map.walkable[item_x, item_y]
            and self.get_actor_at_location(item_x, item_y) is None
        ):
            self.spawn_ground_item(ALARM_CLOCK_TYPE.create(), item_x, item_y)

        # Place a coin pile nearby for testing countables
        coin_x = item_x - 1
        coin_y = item_y
        if (
            self.game_map.walkable[coin_x, coin_y]
            and self.get_actor_at_location(coin_x, coin_y) is None
        ):
            self.item_spawner.spawn_ground_countable(
                (coin_x, coin_y), CountableType.COIN, 43
            )

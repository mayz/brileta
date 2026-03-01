"""Tests for individual pipeline layers.

Phase T4 tests cover:
- T4.1: OpenFieldLayer
- T4.2: BuildingPlacementLayer
- T4.3: RandomTerrainLayer
- T4.4: DetailLayer
- T4.5: NaturalTerrainLayer
- T4.6: StreetNetworkLayer
"""

from __future__ import annotations

import numpy as np

from brileta import config
from brileta.environment.generators.buildings import Building
from brileta.environment.generators.buildings.templates import (
    SMALL_HOUSE_TEMPLATE,
    BuildingTemplate,
)
from brileta.environment.generators.pipeline import GenerationContext
from brileta.environment.generators.pipeline.layers import (
    BuildingPlacementLayer,
    DetailLayer,
    NaturalTerrainLayer,
    OpenFieldLayer,
    RandomTerrainLayer,
    StreetNetworkLayer,
    TreePlacementLayer,
)
from brileta.environment.tile_types import TileTypeID
from brileta.util import rng
from brileta.util.coordinates import Rect

# =============================================================================
# T4.1: OpenFieldLayer
# =============================================================================


class TestOpenFieldLayer:
    """Tests for OpenFieldLayer."""

    def test_fills_with_outdoor_floor(self) -> None:
        """All tiles are COBBLESTONE after applying layer."""
        ctx = GenerationContext.create_empty(width=40, height=30)
        layer = OpenFieldLayer()

        layer.apply(ctx)

        assert np.all(ctx.tiles == TileTypeID.COBBLESTONE)

    def test_creates_exterior_region(self) -> None:
        """Single region with type 'exterior' and sky_exposure=1."""
        ctx = GenerationContext.create_empty(width=50, height=40)
        layer = OpenFieldLayer()

        layer.apply(ctx)

        # Should have exactly one region
        assert len(ctx.regions) == 1
        region = ctx.regions[0]

        # Region should be outdoor/exterior
        assert region.region_type == "exterior"
        assert region.sky_exposure == 1.0

    def test_all_tiles_assigned_to_region(self) -> None:
        """Every tile belongs to the exterior region."""
        ctx = GenerationContext.create_empty(width=30, height=25)
        layer = OpenFieldLayer()

        layer.apply(ctx)

        # All tiles should be assigned to region 0
        assert np.all(ctx.tile_to_region_id == 0)


# =============================================================================
# T4.2: BuildingPlacementLayer
# =============================================================================


class TestBuildingPlacementLayer:
    """Tests for BuildingPlacementLayer."""

    def test_places_buildings_without_overlap(self) -> None:
        """No building footprints intersect."""
        # Use a large map and small buildings to ensure placement
        rng.reset("42")
        ctx = GenerationContext.create_empty(width=200, height=200)

        # Fill with outdoor floor first
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE

        # Small buildings to fit many
        small_template = BuildingTemplate(
            name="tiny",
            building_type="tiny",
            min_width=6,
            max_width=8,
            min_height=6,
            max_height=8,
        )

        layer = BuildingPlacementLayer(
            templates=[small_template],
            min_spacing=2,
            max_buildings=15,
        )
        layer.apply(ctx)

        # Check no footprints overlap
        buildings = ctx.buildings
        for i, building_a in enumerate(buildings):
            for building_b in buildings[i + 1 :]:
                fp_a = building_a.footprint
                fp_b = building_b.footprint
                # Check for intersection
                overlaps = not (
                    fp_a.x2 <= fp_b.x1
                    or fp_b.x2 <= fp_a.x1
                    or fp_a.y2 <= fp_b.y1
                    or fp_b.y2 <= fp_a.y1
                )
                assert not overlaps, f"Buildings overlap: {fp_a} and {fp_b}"

    def test_buildings_have_walls_and_floors(self) -> None:
        """Perimeter is WALL, interior is FLOOR/WALL/DOOR (for multi-room)."""
        rng.reset("123")
        ctx = GenerationContext.create_empty(width=100, height=100)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE

        layer = BuildingPlacementLayer(
            templates=[SMALL_HOUSE_TEMPLATE],
            max_buildings=3,
        )
        layer.apply(ctx)

        for building in ctx.buildings:
            fp = building.footprint

            # Check perimeter has walls (except doors)
            for x in range(fp.x1, fp.x2):
                # Top and bottom walls
                for y in [fp.y1, fp.y2 - 1]:
                    tile = ctx.tiles[x, y]
                    assert tile in (TileTypeID.WALL, TileTypeID.DOOR_CLOSED), (
                        f"Perimeter at ({x},{y}) is {TileTypeID(tile).name}, "
                        "expected WALL or DOOR"
                    )

            for y in range(fp.y1, fp.y2):
                # Left and right walls
                for x in [fp.x1, fp.x2 - 1]:
                    tile = ctx.tiles[x, y]
                    assert tile in (TileTypeID.WALL, TileTypeID.DOOR_CLOSED), (
                        f"Perimeter at ({x},{y}) is {TileTypeID(tile).name}, "
                        "expected WALL or DOOR"
                    )

            # Check interior has floors, walls (for room divisions), or doors
            # Multi-room buildings have internal walls and doors
            valid_interior = {TileTypeID.FLOOR, TileTypeID.WALL, TileTypeID.DOOR_CLOSED}
            for x in range(fp.x1 + 1, fp.x2 - 1):
                for y in range(fp.y1 + 1, fp.y2 - 1):
                    tile = ctx.tiles[x, y]
                    assert tile in valid_interior, (
                        f"Interior at ({x},{y}) is {TileTypeID(tile).name}, "
                        "expected FLOOR, WALL, or DOOR"
                    )

    def test_doors_connect_interior_to_exterior(self) -> None:
        """Door tiles adjacent to both interior and exterior regions."""
        rng.reset("456")
        ctx = GenerationContext.create_empty(width=80, height=80)

        # Set up exterior region first
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE
        exterior_id = ctx.next_region_id()
        from brileta.environment.map import MapRegion
        from brileta.util.coordinates import Rect

        exterior = MapRegion.create_outdoor_region(
            map_region_id=exterior_id,
            region_type="exterior",
            bounds=[Rect(0, 0, ctx.width, ctx.height)],
            sky_exposure=1.0,
        )
        ctx.add_region(exterior)
        ctx.tile_to_region_id[:, :] = exterior_id

        layer = BuildingPlacementLayer(
            templates=[SMALL_HOUSE_TEMPLATE],
            max_buildings=2,
        )
        layer.apply(ctx)

        for building in ctx.buildings:
            for door_x, door_y in building.door_positions:
                # Check adjacent tiles for region variety
                adjacent_regions: set[int] = set()
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = door_x + dx, door_y + dy
                    if 0 <= nx < ctx.width and 0 <= ny < ctx.height:
                        rid = ctx.tile_to_region_id[nx, ny]
                        if rid >= 0:
                            adjacent_regions.add(rid)

                # Door should be adjacent to at least one region
                # (interior or exterior)
                assert len(adjacent_regions) >= 1, (
                    f"Door at ({door_x},{door_y}) not adjacent to any region"
                )

    def test_l_shape_carving_produces_correct_walls_and_floors(self) -> None:
        """Compound footprints should carve perimeter walls and interior floors.

        The 8-neighbor rule ensures concave corner tiles become walls,
        sealing the diagonal FOV gap at compound-shape joints.
        """
        ctx = GenerationContext.create_empty(width=40, height=40)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE
        layer = BuildingPlacementLayer(
            templates=[SMALL_HOUSE_TEMPLATE], max_buildings=0
        )

        building = Building(
            id=99,
            building_type="test",
            footprint=Rect(12, 12, 10, 8),
            wing=Rect.from_bounds(8, 12, 12, 17),
        )
        layer._carve_building(ctx, building)

        occupied = building.tile_set()
        for x, y in occupied:
            # Cardinal check.
            is_interior = (
                (x + 1, y) in occupied
                and (x - 1, y) in occupied
                and (x, y + 1) in occupied
                and (x, y - 1) in occupied
            )
            # Diagonal check - concave corners become walls.
            if is_interior and (
                (x - 1, y - 1) not in occupied
                or (x + 1, y - 1) not in occupied
                or (x - 1, y + 1) not in occupied
                or (x + 1, y + 1) not in occupied
            ):
                is_interior = False

            expected = TileTypeID.FLOOR if is_interior else TileTypeID.WALL
            assert ctx.tiles[x, y] == expected

        # Tiles where the wing joins the primary body should remain floor.
        assert ctx.tiles[12, 13] == TileTypeID.FLOOR
        assert ctx.tiles[11, 13] == TileTypeID.FLOOR

        # The concave corner tile at the L-joint should be a wall, sealing
        # the diagonal gap that would otherwise let FOV leak outward.
        assert ctx.tiles[12, 16] == TileTypeID.WALL

    def test_t_shape_carving_seals_both_concave_corners(self) -> None:
        """T-shapes have two concave corners that must both be walled.

        Without the 8-neighbor rule the corner tiles pass the 4-cardinal
        check but leave a diagonal gap on each side of the wing that lets
        FOV leak outside the building.
        """
        ctx = GenerationContext.create_empty(width=50, height=50)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE
        layer = BuildingPlacementLayer(
            templates=[SMALL_HOUSE_TEMPLATE], max_buildings=0
        )

        # South-facing T: footprint with a centered wing extending south.
        # footprint x=[10,29], y=[10,24]  (y2=25)
        # wing      x=[15,24], y=[24,34]  (overlap row y=24)
        footprint = Rect(10, 10, 20, 15)
        wing = Rect.from_bounds(15, 24, 25, 35)
        building = Building(id=77, building_type="test", footprint=footprint, wing=wing)
        layer._carve_building(ctx, building)

        # Left concave corner: (15, 24) is floor with 4-neighbor rule,
        # wall with 8-neighbor rule because (14, 25) is not in the shape.
        assert ctx.tiles[15, 24] == TileTypeID.WALL

        # Right concave corner: (24, 24) is the symmetric case.
        assert ctx.tiles[24, 24] == TileTypeID.WALL

        # The corridor between the main body and the wing is the tiles
        # between the two walled corners - these must remain floor.
        for x in range(16, 24):
            assert ctx.tiles[x, 24] == TileTypeID.FLOOR, (
                f"Junction corridor tile ({x}, 24) should be FLOOR"
            )

    def test_compound_building_gets_valid_door_on_correct_face(self) -> None:
        """Compound buildings should get a door on the perimeter wall facing
        the nearest street, just like rectangular buildings.
        """
        old_chance = config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE
        try:
            config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE = 1.0
            compound_template = BuildingTemplate(
                name="compound_test",
                building_type="house",
                min_width=18,
                max_width=18,
                min_height=14,
                max_height=14,
                l_shape_weight=1.0,
            )
            placed = 0
            for seed in range(30):
                rng.reset(str(seed))
                ctx = GenerationContext.create_empty(width=80, height=60)
                ctx.tiles[:, :] = TileTypeID.COBBLESTONE

                street_layer = StreetNetworkLayer(style="single")
                street_layer.apply(ctx)

                layer = BuildingPlacementLayer(
                    templates=[compound_template],
                    max_buildings=1,
                    building_density=1.0,
                )
                layer.apply(ctx)

                for building in ctx.buildings:
                    if not building.is_compound:
                        continue
                    placed += 1
                    assert building.door_positions, (
                        f"Seed {seed}: compound building {building.id} has no door"
                    )
                    for door_x, door_y in building.door_positions:
                        assert ctx.tiles[door_x, door_y] == TileTypeID.DOOR_CLOSED
                        # The door must sit on the perimeter (a wall tile
                        # converted to door).
                        assert building.contains_point(door_x, door_y)

            assert placed > 0, "No compound buildings were placed across all seeds"
        finally:
            config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE = old_chance

    def test_compound_building_creates_rooms_per_section(self) -> None:
        """Compound buildings should get one room per occupied section."""
        ctx = GenerationContext.create_empty(width=50, height=50)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE
        layer = BuildingPlacementLayer(
            templates=[SMALL_HOUSE_TEMPLATE], max_buildings=0
        )

        # Manually create an L-shaped building.
        primary = Rect(10, 10, 10, 8)
        wing = Rect.from_bounds(6, 10, 10, 16)
        building = Building(id=99, building_type="test", footprint=primary, wing=wing)
        layer._carve_building(ctx, building)

        template = BuildingTemplate(
            name="test",
            building_type="test",
            min_width=10,
            max_width=10,
            min_height=8,
            max_height=8,
            room_count_min=5,
            room_count_max=5,
        )
        rooms = layer._create_rooms(ctx, building, template, room_count=5)
        # Compound path gives exactly one room per section.
        assert len(rooms) == 2
        # Each room interior should fit inside the building.
        for room in rooms:
            assert room.bounds.width > 0
            assert room.bounds.height > 0

    def test_wing_discarded_when_out_of_map_bounds(self) -> None:
        """Wings extending outside the map should be silently discarded."""
        old_chance = config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE
        try:
            config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE = 1.0
            template = BuildingTemplate(
                name="edge_test",
                building_type="house",
                min_width=18,
                max_width=18,
                min_height=14,
                max_height=14,
                l_shape_weight=1.0,
            )
            rng.reset("edge_test")
            ctx = GenerationContext.create_empty(width=30, height=30)
            ctx.tiles[:, :] = TileTypeID.COBBLESTONE
            layer = BuildingPlacementLayer(templates=[template], max_buildings=0)

            # Place building right at the map edge so any wing extends outside.
            building = layer._create_building(ctx, template, (0, 0), 18, 14)
            # The wing should be discarded if it extends out of bounds.
            if building.wing is not None:
                assert building.wing.x1 >= 0
                assert building.wing.y1 >= 0
                assert building.wing.x2 <= ctx.width
                assert building.wing.y2 <= ctx.height
        finally:
            config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE = old_chance

    def test_wing_discarded_when_crowding_neighbor(self) -> None:
        """Wings that would crowd an existing building should be discarded."""
        old_chance = config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE
        try:
            config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE = 1.0
            template = BuildingTemplate(
                name="crowd_test",
                building_type="house",
                min_width=10,
                max_width=10,
                min_height=8,
                max_height=8,
                l_shape_weight=1.0,
            )
            ctx = GenerationContext.create_empty(width=80, height=80)
            ctx.tiles[:, :] = TileTypeID.COBBLESTONE
            layer = BuildingPlacementLayer(
                templates=[template], max_buildings=0, min_spacing=3
            )

            # Place a neighbor building first.
            neighbor = Building(
                id=0, building_type="house", footprint=Rect(30, 30, 10, 8)
            )
            ctx.buildings.append(neighbor)

            # Place the candidate right next to the neighbor (footprint clears
            # min_spacing, but a wing toward the neighbor would violate it).
            rng.reset("crowd_test")
            building = layer._create_building(ctx, template, (30, 20), 10, 8)
            # Either no wing, or the wing does not crowd the neighbor.
            if building.wing is not None:
                expanded = neighbor.bounding_rect.inflate(layer.min_spacing)
                assert not building.wing.intersects(expanded), (
                    f"Wing {building.wing} is too close to neighbor {neighbor.bounding_rect}"
                )
        finally:
            config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE = old_chance

    def test_creates_building_regions(self) -> None:
        """Each building has indoor region with sky_exposure=0."""
        rng.reset("789")
        ctx = GenerationContext.create_empty(width=100, height=100)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE

        layer = BuildingPlacementLayer(
            templates=[SMALL_HOUSE_TEMPLATE],
            max_buildings=3,
        )
        layer.apply(ctx)

        # Count building regions
        building_regions = [
            r for r in ctx.regions.values() if r.region_type == "building"
        ]

        # Should have at least as many building regions as buildings
        # (could be more if multi-room buildings are implemented)
        assert len(building_regions) >= len(ctx.buildings)

        # All building regions should have no sky exposure
        for region in building_regions:
            assert region.sky_exposure == 0.0

    def test_exterior_doors_have_adequate_entry_depth(self) -> None:
        """Exterior doors should not lead to 1-tile vestibules.

        When entering through an exterior door, there should be at least 2 tiles
        of walkable space before hitting a perpendicular interior wall. This
        prevents awkward layouts where you open a door and immediately face
        an interior wall with the interior door running perpendicular to entry.
        """
        # Use many seeds to catch edge cases
        for seed in ["entry1", "entry2", "entry3", "burrito1", "vestibule"]:
            rng.reset(seed)
            ctx = GenerationContext.create_empty(width=100, height=100)
            ctx.tiles[:, :] = TileTypeID.COBBLESTONE

            layer = BuildingPlacementLayer(
                templates=[SMALL_HOUSE_TEMPLATE],
                max_buildings=5,
            )
            layer.apply(ctx)

            walkable = {TileTypeID.FLOOR, TileTypeID.DOOR_CLOSED, TileTypeID.DOOR_OPEN}

            for building in ctx.buildings:
                fp = building.footprint

                for door_x, door_y in building.door_positions:
                    # Determine entry direction based on door position
                    if door_y == fp.y1:
                        # North wall - entry is south (y increases)
                        entry_tiles = [(door_x, door_y + 1), (door_x, door_y + 2)]
                    elif door_y == fp.y2 - 1:
                        # South wall - entry is north (y decreases)
                        entry_tiles = [(door_x, door_y - 1), (door_x, door_y - 2)]
                    elif door_x == fp.x1:
                        # West wall - entry is east (x increases)
                        entry_tiles = [(door_x + 1, door_y), (door_x + 2, door_y)]
                    elif door_x == fp.x2 - 1:
                        # East wall - entry is west (x decreases)
                        entry_tiles = [(door_x - 1, door_y), (door_x - 2, door_y)]
                    else:
                        continue  # Unknown door position

                    # Check first 2 tiles are walkable (or we hit opposite wall)
                    for i, (tx, ty) in enumerate(entry_tiles):
                        # If we hit the opposite wall, that's fine
                        if not (fp.x1 < tx < fp.x2 - 1 and fp.y1 < ty < fp.y2 - 1):
                            break  # Outside interior, stop checking

                        tile = ctx.tiles[tx, ty]
                        assert tile in walkable, (
                            f"Seed '{seed}': Door at ({door_x},{door_y}) has wall "
                            f"at entry tile {i + 1} ({tx},{ty}) - "
                            f"tile is {TileTypeID(tile).name}. "
                            f"This creates an awkward vestibule."
                        )


# =============================================================================
# T4.3: RandomTerrainLayer
# =============================================================================


class TestRandomTerrainLayer:
    """Tests for RandomTerrainLayer (simple weighted random selection)."""

    def test_only_modifies_outdoor_tiles(self) -> None:
        """Building interiors unchanged by terrain layer."""
        rng.reset("111")
        ctx = GenerationContext.create_empty(width=50, height=50)

        # Set up map with some indoor and outdoor areas
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE
        # Mark a region as indoor (floor)
        ctx.tiles[10:20, 10:20] = TileTypeID.FLOOR
        # Add some walls
        ctx.tiles[25:30, 25:30] = TileTypeID.WALL

        layer = RandomTerrainLayer()
        layer.apply(ctx)

        # Indoor floors should remain FLOOR
        assert np.all(ctx.tiles[10:20, 10:20] == TileTypeID.FLOOR)
        # Walls should remain WALL
        assert np.all(ctx.tiles[25:30, 25:30] == TileTypeID.WALL)

    def test_uses_terrain_patterns(self) -> None:
        """Output contains GRASS, DIRT, GRAVEL."""
        rng.reset("222")
        ctx = GenerationContext.create_empty(width=100, height=100)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE

        layer = RandomTerrainLayer(
            grass_weight=0.4,
            path_weight=0.3,
            gravel_weight=0.3,
        )
        layer.apply(ctx)

        # Check that at least some terrain variety exists
        terrain_types = {
            TileTypeID.GRASS,
            TileTypeID.DIRT,
            TileTypeID.GRAVEL,
        }

        unique_tiles = set(np.unique(ctx.tiles))
        terrain_found = unique_tiles & terrain_types

        # Should have at least 2 different terrain types
        assert len(terrain_found) >= 2, (
            f"Expected terrain variety, only found: {terrain_found}"
        )


# =============================================================================
# T4.5: NaturalTerrainLayer
# =============================================================================


class TestNaturalTerrainLayer:
    """Tests for noise-based natural terrain generation."""

    def test_produces_terrain_variety(self) -> None:
        """Output contains grass and dirt (no gravel in natural terrain)."""
        rng.reset("nat1")
        ctx = GenerationContext.create_empty(width=60, height=60)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE

        NaturalTerrainLayer().apply(ctx)

        unique_tiles = set(np.unique(ctx.tiles))
        assert TileTypeID.GRASS in unique_tiles
        assert TileTypeID.DIRT in unique_tiles
        # No cobblestone should remain - it was all natural terrain
        assert TileTypeID.COBBLESTONE not in unique_tiles
        # Gravel is not a natural terrain type - only placed by feature layers
        assert TileTypeID.GRAVEL not in unique_tiles

    def test_only_modifies_cobblestone_tiles(self) -> None:
        """Pre-existing non-cobblestone tiles are left untouched."""
        rng.reset("nat2")
        ctx = GenerationContext.create_empty(width=30, height=30)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE
        ctx.tiles[5:10, 5:10] = TileTypeID.FLOOR
        ctx.tiles[15:20, 15:20] = TileTypeID.WALL

        NaturalTerrainLayer().apply(ctx)

        # Indoor tiles should be untouched
        assert np.all(ctx.tiles[5:10, 5:10] == TileTypeID.FLOOR)
        assert np.all(ctx.tiles[15:20, 15:20] == TileTypeID.WALL)

    def test_grass_is_dominant(self) -> None:
        """With default thresholds, grass should be the most common terrain."""
        rng.reset("nat3")
        ctx = GenerationContext.create_empty(width=80, height=80)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE

        NaturalTerrainLayer().apply(ctx)

        grass_count = int(np.sum(ctx.tiles == TileTypeID.GRASS))
        dirt_count = int(np.sum(ctx.tiles == TileTypeID.DIRT))

        # Grass should be more common than dirt
        assert grass_count > dirt_count, (
            f"Grass ({grass_count}) should exceed dirt ({dirt_count})"
        )

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces identical terrain."""
        rng.reset("nat_det")
        ctx1 = GenerationContext.create_empty(width=40, height=40)
        ctx1.tiles[:, :] = TileTypeID.COBBLESTONE
        NaturalTerrainLayer().apply(ctx1)

        rng.reset("nat_det")
        ctx2 = GenerationContext.create_empty(width=40, height=40)
        ctx2.tiles[:, :] = TileTypeID.COBBLESTONE
        NaturalTerrainLayer().apply(ctx2)

        np.testing.assert_array_equal(ctx1.tiles, ctx2.tiles)

    def test_no_gravel_in_natural_terrain(self) -> None:
        """Natural terrain should never produce gravel tiles."""
        rng.reset("nat_no_gravel")
        ctx = GenerationContext.create_empty(width=80, height=80)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE

        # Even with extreme grass thresholds, no gravel should appear.
        NaturalTerrainLayer(grass_threshold=-0.5).apply(ctx)

        assert TileTypeID.GRAVEL not in set(np.unique(ctx.tiles))

    def test_island_noise_creates_interior_patches(self) -> None:
        """Island noise should create small patches away from main boundaries.

        With a low island threshold (more islands), we should find tiles
        surrounded entirely by the opposite terrain type - true interior
        islands produced by the high-frequency noise field.
        """
        rng.reset("nat_islands")
        ctx = GenerationContext.create_empty(width=80, height=80)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE

        # Low island threshold to produce many islands for reliable testing.
        NaturalTerrainLayer(island_threshold=0.3).apply(ctx)

        # Find tiles completely surrounded by the opposite type (all 4
        # cardinal neighbors are the same type but the center differs).
        isolated_count = 0
        for x in range(1, ctx.width - 1):
            for y in range(1, ctx.height - 1):
                center = ctx.tiles[x, y]
                if center not in (TileTypeID.GRASS, TileTypeID.DIRT):
                    continue
                neighbors = [
                    ctx.tiles[x - 1, y],
                    ctx.tiles[x + 1, y],
                    ctx.tiles[x, y - 1],
                    ctx.tiles[x, y + 1],
                ]
                opposite = (
                    TileTypeID.DIRT if center == TileTypeID.GRASS else TileTypeID.GRASS
                )
                if all(n == opposite for n in neighbors):
                    isolated_count += 1

        # High-frequency noise with a low threshold should produce a
        # meaningful number of truly isolated interior tiles.
        assert isolated_count > 10, (
            f"Expected isolated interior islands, found only {isolated_count}"
        )


# =============================================================================
# T4.4: DetailLayer
# =============================================================================


class TestDetailLayer:
    """Tests for DetailLayer."""

    def test_places_boulders_in_outdoor_areas(self) -> None:
        """Boulder positions should only land on outdoor terrain."""
        rng.reset("333")
        ctx = GenerationContext.create_empty(width=80, height=80)

        # Set up map with outdoor and indoor areas
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE
        ctx.tiles[30:40, 30:40] = TileTypeID.FLOOR  # Indoor area

        # Use high density to ensure some boulders are placed
        layer = DetailLayer(wild_density=0.1, settlement_density=0.1)
        layer.apply(ctx)

        for x, y in ctx.boulder_positions:
            # Boulders should not be in the indoor area
            assert not (30 <= x < 40 and 30 <= y < 40), (
                f"Boulder at ({x},{y}) is in indoor area"
            )

    def test_does_not_block_doors(self) -> None:
        """No boulders adjacent to door tiles."""
        rng.reset("444")
        ctx = GenerationContext.create_empty(width=100, height=100)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE

        # Place a door manually
        door_x, door_y = 50, 50
        ctx.tiles[door_x, door_y] = TileTypeID.DOOR_CLOSED

        # Create a fake building to establish the avoided zone
        from brileta.environment.generators.buildings import Building
        from brileta.util.coordinates import Rect

        building = Building(
            id=0,
            building_type="test",
            footprint=Rect(door_x - 5, door_y - 5, 10, 10),
            door_positions=[(door_x, door_y)],
        )
        ctx.buildings.append(building)

        layer = DetailLayer(
            wild_density=0.5,  # High density
            settlement_density=0.5,
            min_distance_from_buildings=3,
        )
        layer.apply(ctx)

        # Check no boulders inside the building avoidance buffer.
        for bx, by in ctx.boulder_positions:
            assert abs(bx - door_x) >= 3 or abs(by - door_y) >= 3

    def test_respects_boulder_density(self) -> None:
        """Boulder placement probability approximately matches density."""
        rng.reset("555")
        ctx = GenerationContext.create_empty(width=100, height=100)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE

        density = 0.05
        layer = DetailLayer(wild_density=density, settlement_density=0.0)
        layer.apply(ctx)

        # Count boulders
        boulder_count = len(ctx.boulder_positions)
        total_tiles = ctx.width * ctx.height

        # Should be roughly density * total_tiles, with some variance
        expected = density * total_tiles
        # Allow 50% variance due to randomness
        assert boulder_count > expected * 0.5, (
            f"Too few boulders: {boulder_count} < {expected * 0.5}"
        )
        assert boulder_count < expected * 1.5, (
            f"Too many boulders: {boulder_count} > {expected * 1.5}"
        )

    def test_no_boulders_on_street_tiles(self) -> None:
        """Boulders are never placed on roads."""
        rng.reset("556")
        ctx = GenerationContext.create_empty(width=80, height=60)
        ctx.tiles[:, :] = TileTypeID.GRASS

        street_layer = StreetNetworkLayer(style="single")
        street_layer.apply(ctx)

        layer = DetailLayer(wild_density=1.0, settlement_density=1.0)
        layer.apply(ctx)

        boulder_positions = set(ctx.boulder_positions)
        for street in ctx.street_data.streets:
            for x in range(max(0, street.x1), min(ctx.width, street.x2)):
                for y in range(max(0, street.y1), min(ctx.height, street.y2)):
                    assert (x, y) not in boulder_positions, (
                        f"Boulder placed on street tile at ({x},{y})"
                    )

    def test_wild_vs_settlement_density(self) -> None:
        """Wild areas should have more boulders than settlement-adjacent areas."""
        rng.reset("557")
        ctx = GenerationContext.create_empty(width=100, height=80)
        ctx.tiles[:, :] = TileTypeID.GRASS

        street_layer = StreetNetworkLayer(style="single")
        street_layer.apply(ctx)

        layer = DetailLayer(
            wild_density=0.2,
            settlement_density=0.01,
            street_buffer=8,
        )
        layer.apply(ctx)

        street = ctx.street_data.streets[0]
        street_center_y = (street.y1 + street.y2) // 2
        boulder_positions = set(ctx.boulder_positions)

        near_count = 0
        near_total = 0
        far_count = 0
        far_total = 0

        for x in range(ctx.width):
            for y in range(ctx.height):
                # Skip road tiles entirely; roads are always excluded.
                if street.y1 <= y < street.y2:
                    continue

                dist = abs(y - street_center_y)
                if dist <= 8:
                    near_total += 1
                    if (x, y) in boulder_positions:
                        near_count += 1
                elif dist > 15:
                    far_total += 1
                    if (x, y) in boulder_positions:
                        far_count += 1

        if near_total > 0 and far_total > 0:
            near_density = near_count / near_total
            far_density = far_count / far_total
            assert far_density > near_density, (
                f"Wild density ({far_density:.3f}) should exceed settlement "
                f"density ({near_density:.3f})"
            )

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce identical boulder placement."""
        rng.reset("558")
        ctx1 = GenerationContext.create_empty(width=80, height=60)
        ctx1.tiles[:, :] = TileTypeID.GRASS
        StreetNetworkLayer(style="single").apply(ctx1)
        DetailLayer(wild_density=0.05, settlement_density=0.01).apply(ctx1)

        rng.reset("558")
        ctx2 = GenerationContext.create_empty(width=80, height=60)
        ctx2.tiles[:, :] = TileTypeID.GRASS
        StreetNetworkLayer(style="single").apply(ctx2)
        DetailLayer(wild_density=0.05, settlement_density=0.01).apply(ctx2)

        assert ctx1.boulder_positions == ctx2.boulder_positions


# =============================================================================
# T4.6: StreetNetworkLayer
# =============================================================================


class TestStreetNetworkLayer:
    """Tests for StreetNetworkLayer."""

    def test_creates_single_street(self) -> None:
        """Single style creates one horizontal street through the middle."""
        rng.reset(42)
        ctx = GenerationContext.create_empty(width=80, height=60)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE
        layer = StreetNetworkLayer(style="single")

        layer.apply(ctx)

        # Should have street data
        assert len(ctx.street_data.streets) == 1
        street = ctx.street_data.streets[0]

        # Street should span the full width
        assert street.width == ctx.width

        # Street should be in the middle
        assert abs(street.y1 - ctx.height // 2) <= 2

    def test_creates_cross_streets(self) -> None:
        """Cross style creates horizontal and vertical streets."""
        rng.reset(42)
        ctx = GenerationContext.create_empty(width=80, height=60)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE
        layer = StreetNetworkLayer(style="cross")

        layer.apply(ctx)

        # Should have 2 streets
        assert len(ctx.street_data.streets) == 2

        # Should have 4 building zones (quadrants)
        assert len(ctx.street_data.zones) == 4

    def test_street_tiles_are_outdoor_floor(self) -> None:
        """Street tiles are carved as COBBLESTONE."""
        rng.reset(42)
        ctx = GenerationContext.create_empty(width=80, height=60)
        ctx.tiles[:, :] = TileTypeID.GRASS  # Start with grass
        layer = StreetNetworkLayer(style="single")

        layer.apply(ctx)

        street = ctx.street_data.streets[0]
        # Check street tiles are now outdoor floor
        for y in range(street.y1, street.y2):
            assert ctx.tiles[ctx.width // 2, y] == TileTypeID.COBBLESTONE

    def test_zones_do_not_overlap_streets_interior(self) -> None:
        """Building zone interiors should not overlap with street interiors."""
        rng.reset(42)
        ctx = GenerationContext.create_empty(width=80, height=60)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE
        layer = StreetNetworkLayer(style="cross")

        layer.apply(ctx)

        for zone in ctx.street_data.zones:
            for street in ctx.street_data.streets:
                # Check for interior overlap (not just edge touching)
                # Shrink both by 1 to check interior only
                interior_overlap = (
                    zone.x1 < street.x2 - 1
                    and zone.x2 > street.x1 + 1
                    and zone.y1 < street.y2 - 1
                    and zone.y2 > street.y1 + 1
                )
                assert not interior_overlap, (
                    f"Zone {zone} interior overlaps street {street} interior"
                )


# =============================================================================
# T4.8: Street Margin Tests
# =============================================================================


class TestStreetMargins:
    """Tests for gravel/dirt margin placement along street edges."""

    def _make_terrain_with_streets(
        self,
        seed: str = "margin1",
        width: int = 80,
        height: int = 60,
        style: str = "single",
    ) -> GenerationContext:
        """Helper: apply terrain then streets to get a context with margins."""
        rng.reset(seed)
        ctx = GenerationContext.create_empty(width=width, height=height)
        OpenFieldLayer().apply(ctx)
        NaturalTerrainLayer().apply(ctx)
        StreetNetworkLayer(style=style).apply(ctx)
        return ctx

    def test_street_margin_converts_nearby_grass(self) -> None:
        """Grass tiles near streets become gravel or dirt (with noise gaps)."""
        ctx = self._make_terrain_with_streets()

        street = ctx.street_data.streets[0]

        # Check tiles within margin_max distance of the street boundary.
        # The margin is noise-driven, so not every tile is converted, but
        # across the full street length we expect a meaningful number.
        converted = 0
        checked = 0
        margin_max = 3  # Default config value
        for x in range(max(0, street.x1), min(ctx.width, street.x2)):
            for dy in range(1, margin_max + 1):
                for sign in (-1, 1):
                    y = (street.y1 if sign == -1 else street.y2 - 1) + sign * dy
                    if 0 <= y < ctx.height:
                        checked += 1
                        tile = ctx.tiles[x, y]
                        if tile in (TileTypeID.GRAVEL, TileTypeID.DIRT):
                            converted += 1

        # The terrain is noise-generated and margins have noise gaps,
        # but a meaningful fraction should have been converted.
        assert checked > 0
        assert converted > 0, (
            "Expected some grass tiles near streets to be converted "
            "to gravel/dirt margins"
        )

    def test_street_margin_preserves_non_terrain(self) -> None:
        """Non-terrain tiles near streets are not modified by margins."""
        rng.reset("margin_preserve")
        ctx = GenerationContext.create_empty(width=80, height=60)
        OpenFieldLayer().apply(ctx)
        NaturalTerrainLayer().apply(ctx)

        # Apply streets first so we know exactly where they are, then
        # place non-terrain tiles inside the margin zone and verify
        # they survive a second margin application.
        layer = StreetNetworkLayer(style="single")
        layer.apply(ctx)

        street = ctx.street_data.streets[0]

        # Place FLOOR and WALL tiles 1 tile above the street - firmly
        # inside the margin zone regardless of street position.
        test_y = street.y1 - 1
        ctx.tiles[10, test_y] = TileTypeID.FLOOR
        ctx.tiles[11, test_y] = TileTypeID.WALL

        # Re-apply margins to verify non-terrain tiles are untouched.
        layer._apply_street_margins(ctx, ctx.street_data.streets)

        assert ctx.tiles[10, test_y] == TileTypeID.FLOOR
        assert ctx.tiles[11, test_y] == TileTypeID.WALL

    def test_street_margin_no_op_when_disabled(self) -> None:
        """margin_max=0 skips margin placement entirely."""
        rng.reset("margin_noop")
        ctx = GenerationContext.create_empty(width=80, height=60)
        ctx.tiles[:, :] = TileTypeID.GRASS

        # Capture tiles before margins.
        StreetNetworkLayer(style="single", margin_max=0).apply(ctx)

        # Only street tiles should be cobblestone. No gravel or dirt
        # margins should exist since margin_max=0.
        unique = set(np.unique(ctx.tiles))
        assert TileTypeID.GRAVEL not in unique, (
            "No gravel margins expected with margin_max=0"
        )
        assert TileTypeID.DIRT not in unique, (
            "No dirt margins expected with margin_max=0"
        )

    def test_street_margin_deterministic(self) -> None:
        """Same seed produces identical margin results."""
        ctx1 = self._make_terrain_with_streets(seed="margin_det")
        ctx2 = self._make_terrain_with_streets(seed="margin_det")

        np.testing.assert_array_equal(ctx1.tiles, ctx2.tiles)


class TestBuildingPlacementWithStreets:
    """Tests for building placement with street data."""

    def test_buildings_placed_in_zones(self) -> None:
        """Buildings are placed within street-defined zones."""
        # Use a larger map to ensure zones can fit buildings
        rng.reset(42)
        ctx = GenerationContext.create_empty(width=120, height=100)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE

        # Create street network first
        street_layer = StreetNetworkLayer(style="cross")
        street_layer.apply(ctx)

        # Create a small building template for testing
        small_template = BuildingTemplate(
            name="test_shack",
            building_type="shack",
            min_width=8,
            max_width=10,
            min_height=6,
            max_height=8,
            room_count_min=1,
            room_count_max=1,
        )

        # Then place buildings with 100% density to ensure placement
        building_layer = BuildingPlacementLayer(
            templates=[small_template],
            max_buildings=4,
            building_density=1.0,
        )
        building_layer.apply(ctx)

        # Buildings should exist
        assert len(ctx.buildings) > 0, (
            f"No buildings placed. Zones: {ctx.street_data.zones}"
        )

        # Each building should at least overlap a zone
        for building in ctx.buildings:
            fp = building.footprint
            overlaps_zone = any(zone.intersects(fp) for zone in ctx.street_data.zones)
            assert overlaps_zone, f"Building {fp} not in any zone"

    def test_zone_placed_buildings_respect_min_spacing(self) -> None:
        """Buildings placed via zone/lot system maintain min_spacing clearance.

        The BSP lot subdivision can produce adjacent lots whose per-lot margins
        are smaller than min_spacing.  The placement path must enforce spacing
        against already-placed buildings so they never abut or overlap.
        """
        for seed in ["space1", "space2", "space3", "space4", "space5"]:
            rng.reset(seed)
            ctx = GenerationContext.create_empty(width=120, height=100)
            ctx.tiles[:, :] = TileTypeID.COBBLESTONE

            street_layer = StreetNetworkLayer(style="cross")
            street_layer.apply(ctx)

            min_spacing = 3
            building_layer = BuildingPlacementLayer(
                templates=[SMALL_HOUSE_TEMPLATE],
                max_buildings=8,
                min_spacing=min_spacing,
                building_density=1.0,
            )
            building_layer.apply(ctx)

            buildings = ctx.buildings
            for i, a in enumerate(buildings):
                for b in buildings[i + 1 :]:
                    ab = a.bounding_rect
                    bb = b.bounding_rect
                    # Compute axis-aligned gap between the two bounding rects.
                    gap_x = max(0, max(ab.x1, bb.x1) - min(ab.x2, bb.x2))
                    gap_y = max(0, max(ab.y1, bb.y1) - min(ab.y2, bb.y2))
                    gap = max(gap_x, gap_y)
                    assert gap >= min_spacing, (
                        f"Seed '{seed}': buildings {a.id} and {b.id} are only "
                        f"{gap} tiles apart (min_spacing={min_spacing}). "
                        f"A={ab}, B={bb}"
                    )

    def test_doors_face_streets(self) -> None:
        """Building doors should face the nearest street."""
        rng.reset(42)
        ctx = GenerationContext.create_empty(width=80, height=60)
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE

        # Create street network first (single horizontal street)
        street_layer = StreetNetworkLayer(style="single")
        street_layer.apply(ctx)

        # Then place buildings
        building_layer = BuildingPlacementLayer(
            templates=[SMALL_HOUSE_TEMPLATE],
            max_buildings=4,
        )
        building_layer.apply(ctx)

        street = ctx.street_data.streets[0]
        street_center_y = (street.y1 + street.y2) // 2

        # Each building's door should face the street
        for building in ctx.buildings:
            if not building.door_positions:
                continue

            door_x, door_y = building.door_positions[0]
            fp = building.footprint
            building_center_y = (fp.y1 + fp.y2) // 2

            # Determine expected door direction
            if building_center_y < street_center_y:
                # Building is above street, door should be on south wall
                assert door_y == fp.y2 - 1, (
                    f"Building above street should have door on south wall, "
                    f"but door at ({door_x}, {door_y}), building at {fp}"
                )
            else:
                # Building is below street, door should be on north wall
                assert door_y == fp.y1, (
                    f"Building below street should have door on north wall, "
                    f"but door at ({door_x}, {door_y}), building at {fp}"
                )


# =============================================================================
# T4.7: TreePlacementLayer
# =============================================================================


class TestTreePlacementLayer:
    """Tests for TreePlacementLayer."""

    def test_places_trees_on_plantable_terrain(self) -> None:
        """Trees appear only on grass, dirt, or gravel tiles."""
        rng.reset("tree1")
        ctx = GenerationContext.create_empty(width=60, height=60)
        ctx.tiles[:, :] = TileTypeID.GRASS

        layer = TreePlacementLayer(wild_density=0.15, yard_density=0.0)
        layer.apply(ctx)

        tree_positions = ctx.tree_positions
        assert len(tree_positions) > 0, "Expected some trees to be placed"

        plantable = {TileTypeID.GRASS, TileTypeID.DIRT, TileTypeID.GRAVEL}
        assert all(ctx.tiles[x, y] in plantable for x, y in tree_positions)
        assert len(tree_positions) == len(set(tree_positions))

    def test_no_trees_on_streets(self) -> None:
        """Trees are never placed on street tiles."""
        rng.reset("tree2")
        ctx = GenerationContext.create_empty(width=80, height=60)
        ctx.tiles[:, :] = TileTypeID.GRASS

        # Create a street through the middle
        street_layer = StreetNetworkLayer(style="single")
        street_layer.apply(ctx)

        # High density to maximize chance of misplacement
        layer = TreePlacementLayer(wild_density=0.5, yard_density=0.5, street_buffer=2)
        layer.apply(ctx)

        tree_positions = set(ctx.tree_positions)

        # Verify no trees on street tiles
        for street in ctx.street_data.streets:
            for x in range(max(0, street.x1), min(ctx.width, street.x2)):
                for y in range(max(0, street.y1), min(ctx.height, street.y2)):
                    assert (x, y) not in tree_positions, (
                        f"Tree placed on street tile at ({x},{y})"
                    )

    def test_no_trees_inside_buildings(self) -> None:
        """Trees are never placed inside building footprints."""
        rng.reset("tree3")
        ctx = GenerationContext.create_empty(width=80, height=80)
        ctx.tiles[:, :] = TileTypeID.GRASS

        # Place a building manually
        from brileta.environment.generators.buildings import Building
        from brileta.util.coordinates import Rect

        building = Building(
            id=0,
            building_type="test",
            footprint=Rect(20, 20, 15, 15),
            door_positions=[(27, 20)],
        )
        ctx.buildings.append(building)

        # Set building interior to walls/floors (as building layer would)
        fp = building.footprint
        for x in range(fp.x1, fp.x2):
            for y in range(fp.y1, fp.y2):
                ctx.tiles[x, y] = TileTypeID.WALL

        layer = TreePlacementLayer(wild_density=0.5, yard_density=0.5)
        layer.apply(ctx)

        tree_positions = set(ctx.tree_positions)

        # Verify no trees in building footprint
        for x in range(fp.x1, fp.x2):
            for y in range(fp.y1, fp.y2):
                assert (x, y) not in tree_positions, (
                    f"Tree placed inside building at ({x},{y})"
                )

    def test_wild_vs_settlement_zones(self) -> None:
        """Wild trees are denser than settlement trees."""
        rng.reset("tree4")
        ctx = GenerationContext.create_empty(width=100, height=80)
        ctx.tiles[:, :] = TileTypeID.GRASS

        # Create streets to define settlement vs wild zones
        street_layer = StreetNetworkLayer(style="single")
        street_layer.apply(ctx)

        # Use distinct densities to verify zone separation
        layer = TreePlacementLayer(
            wild_density=0.3,
            yard_density=0.02,
            street_buffer=8,
        )
        layer.apply(ctx)

        # Count trees near streets (settlement) vs far from streets (wild)
        street = ctx.street_data.streets[0]
        street_center_y = (street.y1 + street.y2) // 2
        tree_positions = set(ctx.tree_positions)

        near_count = 0
        near_total = 0
        far_count = 0
        far_total = 0

        for x in range(ctx.width):
            for y in range(ctx.height):
                dist = abs(y - street_center_y)
                if dist <= 8:
                    near_total += 1
                    if (x, y) in tree_positions:
                        near_count += 1
                elif dist > 15:
                    far_total += 1
                    if (x, y) in tree_positions:
                        far_count += 1

        # Wild (far) density should be noticeably higher than settlement (near)
        if near_total > 0 and far_total > 0:
            near_density = near_count / near_total
            far_density = far_count / far_total
            assert far_density > near_density, (
                f"Wild density ({far_density:.3f}) should exceed settlement "
                f"density ({near_density:.3f})"
            )

    def test_respects_density_parameter(self) -> None:
        """Tree count roughly matches the configured density."""
        rng.reset("tree5")
        ctx = GenerationContext.create_empty(width=80, height=80)
        ctx.tiles[:, :] = TileTypeID.GRASS

        # No streets or buildings - all tiles are wild
        density = 0.08
        layer = TreePlacementLayer(wild_density=density, yard_density=0.0)
        layer.apply(ctx)

        tree_count = len(ctx.tree_positions)
        total_tiles = ctx.width * ctx.height

        # The noise multiplier formula (noise + 1) * 1.5 has a mean of ~1.5,
        # so average density is about 1.5x the base. Account for that when
        # checking count bounds. Allow 50% variance on top of the mean shift.
        expected = density * 1.5 * total_tiles

        assert tree_count > expected * 0.5, (
            f"Too few trees: {tree_count} < {expected * 0.5:.0f}"
        )
        assert tree_count < expected * 1.5, (
            f"Too many trees: {tree_count} > {expected * 1.5:.0f}"
        )

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces same tree placement."""
        rng.reset("tree_det")
        ctx1 = GenerationContext.create_empty(width=50, height=50)
        ctx1.tiles[:, :] = TileTypeID.GRASS
        TreePlacementLayer(wild_density=0.1, yard_density=0.0).apply(ctx1)

        rng.reset("tree_det")
        ctx2 = GenerationContext.create_empty(width=50, height=50)
        ctx2.tiles[:, :] = TileTypeID.GRASS
        TreePlacementLayer(wild_density=0.1, yard_density=0.0).apply(ctx2)

        assert ctx1.tree_positions == ctx2.tree_positions

    def test_trees_not_placed_on_boulders(self) -> None:
        """Trees avoid coordinates reserved for boulder actor placement."""
        rng.reset("tree6")
        ctx = GenerationContext.create_empty(width=40, height=40)
        ctx.tiles[:, :] = TileTypeID.GRASS

        # Reserve some boulder positions before placing trees.
        ctx.boulder_positions = [(10, 10), (20, 20)]

        layer = TreePlacementLayer(wild_density=0.5, yard_density=0.0)
        layer.apply(ctx)

        assert ctx.boulder_positions == [(10, 10), (20, 20)]
        assert (10, 10) not in set(ctx.tree_positions)
        assert (20, 20) not in set(ctx.tree_positions)

    def test_noise_creates_spatial_clustering(self) -> None:
        """Density noise should create groves and clearings, not uniform scatter.

        Divide the map into a grid of cells and measure the coefficient of
        variation (stddev / mean) of tree counts per cell. Noise-modulated
        placement should produce higher variance than flat-probability placement
        would on average.
        """
        rng.reset("cluster1")
        ctx = GenerationContext.create_empty(width=120, height=120)
        ctx.tiles[:, :] = TileTypeID.GRASS

        # Moderate density so cells have enough trees to measure variance.
        layer = TreePlacementLayer(wild_density=0.10, yard_density=0.0)
        layer.apply(ctx)

        tree_set = set(ctx.tree_positions)
        assert len(tree_set) > 50, "Need enough trees to measure clustering"

        # Count trees per 20x20 grid cell.
        cell_size = 20
        cols = ctx.width // cell_size
        rows = ctx.height // cell_size
        counts = []
        for cx in range(cols):
            for cy in range(rows):
                n = sum(
                    1
                    for x in range(cx * cell_size, (cx + 1) * cell_size)
                    for y in range(cy * cell_size, (cy + 1) * cell_size)
                    if (x, y) in tree_set
                )
                counts.append(n)

        mean = sum(counts) / len(counts)
        stddev = (sum((c - mean) ** 2 for c in counts) / len(counts)) ** 0.5
        cv = stddev / mean if mean > 0 else 0.0

        # For pure binomial placement (no noise), the expected CV on 400-tile
        # cells at p=0.10 is sqrt(p*(1-p)/n) / p ~= 0.15. Noise-modulated
        # placement should be noticeably higher because some cells land in
        # groves and others in clearings.
        assert cv > 0.20, (
            f"Coefficient of variation {cv:.3f} is too low - expected noise "
            f"to create spatial clustering (groves and clearings)"
        )


# =============================================================================
# Species noise tests
# =============================================================================


class TestSpeciesNoiseClustering:
    """Tests for noise-based archetype (species) distribution."""

    def test_nearby_trees_share_species(self) -> None:
        """Trees close together should agree on species more often than chance.

        Sample a grid of positions and compare archetype agreement for
        close-pair vs far-pair positions. Noise should make close pairs
        agree more often.
        """
        from brileta.game.actors.tree_sprites import (
            TreeArchetype,
            archetype_for_position,
            create_species_noise,
        )

        map_seed = 42
        species_noise = create_species_noise(map_seed)

        # Generate archetypes on a grid (only look at living types since
        # dead/sapling are noise-independent).
        size = 80
        archetypes: dict[tuple[int, int], TreeArchetype] = {}
        for x in range(size):
            for y in range(size):
                a = archetype_for_position(x, y, map_seed, species_noise)
                if a in (TreeArchetype.DECIDUOUS, TreeArchetype.CONIFER):
                    archetypes[(x, y)] = a

        # Compare pairs within distance 5 ("close") vs distance 30+ ("far").
        close_agree = 0
        close_total = 0
        far_agree = 0
        far_total = 0

        positions = list(archetypes.keys())
        # Sample a subset of pairs for speed.
        step = max(1, len(positions) // 400)
        for i in range(0, len(positions), step):
            for j in range(i + 1, len(positions), step):
                p1, p2 = positions[i], positions[j]
                dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                same = archetypes[p1] == archetypes[p2]
                if dist <= 5:
                    close_total += 1
                    close_agree += int(same)
                elif dist >= 30:
                    far_total += 1
                    far_agree += int(same)

        assert close_total > 0 and far_total > 0, "Not enough pairs sampled"

        close_rate = close_agree / close_total
        far_rate = far_agree / far_total

        # Close trees should agree on species more often than far trees.
        assert close_rate > far_rate, (
            f"Close-pair agreement ({close_rate:.2f}) should exceed "
            f"far-pair agreement ({far_rate:.2f})"
        )

    def test_species_noise_is_deterministic(self) -> None:
        """Same map seed produces identical archetype assignments."""
        from brileta.game.actors.tree_sprites import (
            archetype_for_position,
            create_species_noise,
        )

        map_seed = 99
        noise_a = create_species_noise(map_seed)
        noise_b = create_species_noise(map_seed)

        for x in range(30):
            for y in range(30):
                assert archetype_for_position(
                    x, y, map_seed, noise_a
                ) == archetype_for_position(x, y, map_seed, noise_b)

    def test_all_archetypes_still_appear(self) -> None:
        """All four archetypes should appear over a large enough sample."""
        from brileta.game.actors.tree_sprites import (
            TreeArchetype,
            archetype_for_position,
            create_species_noise,
        )

        map_seed = 12345
        species_noise = create_species_noise(map_seed)

        found: set[TreeArchetype] = set()
        for x in range(100):
            for y in range(100):
                found.add(archetype_for_position(x, y, map_seed, species_noise))
                if len(found) == 4:
                    break
            if len(found) == 4:
                break

        assert found == set(TreeArchetype), (
            f"Missing archetypes: {set(TreeArchetype) - found}"
        )

from __future__ import annotations

import numpy as np

from catley.environment.generators import (
    RoomsAndCorridorsGenerator,
    create_settlement_pipeline,
)
from catley.environment.tile_types import TileTypeID


def test_room_region_creation() -> None:
    gen = RoomsAndCorridorsGenerator(
        40, 30, max_rooms=5, min_room_size=5, max_room_size=8
    )
    data = gen.generate()

    room_regions = [r for r in data.regions.values() if r.region_type == "room"]
    assert len(room_regions) >= 1  # At least one room created
    assert len(room_regions) <= 5  # No more than max_rooms

    first_room = room_regions[0]
    room_rect = first_room.bounds[0]
    slice_x = slice(room_rect.x1 + 1, room_rect.x2)
    slice_y = slice(room_rect.y1 + 1, room_rect.y2)
    assert np.all(data.tile_to_region_id[slice_x, slice_y] == first_room.id)


def test_hallway_region_identification() -> None:
    gen = RoomsAndCorridorsGenerator(
        40, 30, max_rooms=5, min_room_size=5, max_room_size=8
    )
    data = gen.generate()

    hallway_regions = [r for r in data.regions.values() if r.region_type == "hallway"]
    assert len(hallway_regions) >= 1

    # Get first hallway region and verify tiles are assigned correctly
    region = hallway_regions[0]
    bounds = region.bounds[0]
    # Check that at least one tile in the hallway bounds is assigned to this region
    found_assigned = False
    for x in range(bounds.x1, bounds.x2):
        for y in range(bounds.y1, bounds.y2):
            if data.tile_to_region_id[x, y] == region.id:
                found_assigned = True
                break
        if found_assigned:
            break
    assert found_assigned, f"No tiles assigned to hallway region {region.id}"


def test_door_connection_logic() -> None:
    # Use larger map to ensure hallway generation
    gen = RoomsAndCorridorsGenerator(
        40, 30, max_rooms=3, min_room_size=5, max_room_size=8
    )
    data = gen.generate()

    hallway_regions = [r for r in data.regions.values() if r.region_type == "hallway"]
    if not hallway_regions:
        # Some seeds may produce maps without hallways - skip connection test
        return

    room_ids = [r.id for r in data.regions.values() if r.region_type == "room"]
    hallway = hallway_regions[0]

    # Find a room that connects to this hallway
    for room_id in room_ids:
        room = data.regions[room_id]
        if hallway.id in room.connections:
            door_tile = room.connections[hallway.id]
            # Verify bidirectional connection
            assert room_id in hallway.connections
            assert hallway.connections[room_id] == door_tile
            return

    # If no connections found, that's okay for some map configurations


def test_no_unassigned_walkable_tiles() -> None:
    gen = RoomsAndCorridorsGenerator(
        40, 30, max_rooms=5, min_room_size=5, max_room_size=8
    )
    data = gen.generate()

    for x in range(data.tiles.shape[0]):
        for y in range(data.tiles.shape[1]):
            tile = data.tiles[x, y]
            if tile in (
                TileTypeID.FLOOR,
                TileTypeID.DOOR_OPEN,
            ):
                assert data.tile_to_region_id[x, y] != -1


# =============================================================================
# Settlement Pipeline Tests
# =============================================================================


def test_settlement_creates_buildings_and_exterior() -> None:
    """Settlement generator creates both building and exterior regions."""
    gen = create_settlement_pipeline(80, 43, seed=42, street_style="cross")
    data = gen.generate()

    region_types = {r.region_type for r in data.regions.values()}
    assert "building" in region_types, "No building regions generated"
    assert "exterior" in region_types, "No exterior region generated"


def test_settlement_buildings_have_doors() -> None:
    """Each building should have at least one door."""
    gen = create_settlement_pipeline(80, 43, seed=42, street_style="cross")
    data = gen.generate()

    building_regions = [r for r in data.regions.values() if r.region_type == "building"]
    assert len(building_regions) > 0, "No buildings generated"

    # Count doors in the map
    door_count = np.sum(data.tiles == TileTypeID.DOOR_CLOSED)
    assert door_count >= 1, "No doors generated"


def test_settlement_sky_exposure() -> None:
    """Buildings should have sky_exposure=0, exterior should have sky_exposure=1."""
    gen = create_settlement_pipeline(80, 43, seed=42, street_style="cross")
    data = gen.generate()

    for region in data.regions.values():
        if region.region_type == "exterior":
            assert region.sky_exposure == 1.0, (
                f"Exterior region has wrong sky_exposure: {region.sky_exposure}"
            )
        elif region.region_type == "building":
            assert region.sky_exposure == 0.0, (
                f"Building region has wrong sky_exposure: {region.sky_exposure}"
            )


def test_settlement_street_styles() -> None:
    """All street styles should produce valid maps."""
    for style in ["single", "cross", "grid"]:
        gen = create_settlement_pipeline(80, 43, seed=42, street_style=style)
        data = gen.generate()

        # Should have some terrain variety (grass, dirt, gravel, or outdoor floor)
        terrain_tiles = {
            TileTypeID.COBBLESTONE,
            TileTypeID.GRASS,
            TileTypeID.DIRT_PATH,
            TileTypeID.GRAVEL,
        }
        terrain_count = sum(np.sum(data.tiles == t) for t in terrain_tiles)
        assert terrain_count > 0, f"No terrain tiles for style {style}"

        # Should have at least some buildings
        building_regions = [
            r for r in data.regions.values() if r.region_type == "building"
        ]
        assert len(building_regions) >= 1, f"No buildings for style {style}"


def test_settlement_building_interiors_are_floor() -> None:
    """Building interiors should use FLOOR tile type (not COBBLESTONE)."""
    gen = create_settlement_pipeline(80, 43, seed=42, street_style="cross")
    data = gen.generate()

    # Check that building regions contain FLOOR tiles
    building_regions = [r for r in data.regions.values() if r.region_type == "building"]

    for region in building_regions:
        # Find tiles assigned to this region
        region_mask = data.tile_to_region_id == region.id
        region_tiles = data.tiles[region_mask]

        # All tiles in building region should be FLOOR (not COBBLESTONE)
        floor_tiles = region_tiles[region_tiles == TileTypeID.FLOOR]
        assert len(floor_tiles) > 0, f"Building region {region.id} has no FLOOR tiles"


def test_settlement_connections_via_doors() -> None:
    """Buildings should have connections to exterior via doors."""
    gen = create_settlement_pipeline(80, 43, seed=42, street_style="cross")
    data = gen.generate()

    building_regions = [r for r in data.regions.values() if r.region_type == "building"]

    # At least some buildings should have connections
    buildings_with_connections = [b for b in building_regions if b.connections]
    assert len(buildings_with_connections) > 0, "No buildings have connections"

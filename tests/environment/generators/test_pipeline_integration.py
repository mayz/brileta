"""Integration tests for the pipeline map generator.

Phase T5 tests cover:
- T5.1: Full pipeline
- T5.2: GameWorld integration
"""

from __future__ import annotations

import numpy as np

from catley.environment.generators.pipeline import create_pipeline
from catley.environment.tile_types import TileTypeID

# =============================================================================
# T5.1: Full Pipeline
# =============================================================================


class TestFullPipeline:
    """Integration tests for the complete settlement pipeline."""

    def test_settlement_pipeline_produces_playable_map(self) -> None:
        """Has walkable spawn area, buildings accessible."""
        generator = create_pipeline("settlement", width=80, height=60, seed="12345")
        map_data = generator.generate()

        # Map should have correct dimensions
        assert map_data.tiles.shape == (80, 60)

        # Should have walkable tiles
        walkable_tiles = {
            TileTypeID.FLOOR,
            TileTypeID.COBBLESTONE,
            TileTypeID.GRASS,
            TileTypeID.DIRT_PATH,
            TileTypeID.GRAVEL,
        }

        walkable_mask = np.isin(map_data.tiles, list(walkable_tiles))
        walkable_count = np.sum(walkable_mask)

        # Significant portion should be walkable (at least 50%)
        total_tiles = 80 * 60
        assert walkable_count > total_tiles * 0.5, (
            f"Too few walkable tiles: {walkable_count} / {total_tiles}"
        )

        # Should have regions (exterior + buildings)
        assert len(map_data.regions) >= 1

        # Should have an exterior region
        exterior_regions = [
            r for r in map_data.regions.values() if r.region_type == "exterior"
        ]
        assert len(exterior_regions) >= 1

    def test_pipeline_deterministic_with_seed(self) -> None:
        """Same seed produces identical output."""
        gen1 = create_pipeline("settlement", width=60, height=50, seed="99999")
        map1 = gen1.generate()

        gen2 = create_pipeline("settlement", width=60, height=50, seed="99999")
        map2 = gen2.generate()

        # Tiles should be identical
        np.testing.assert_array_equal(map1.tiles, map2.tiles)

        # Region assignments should be identical
        np.testing.assert_array_equal(map1.tile_to_region_id, map2.tile_to_region_id)

        # Same number of regions
        assert len(map1.regions) == len(map2.regions)

    def test_settlement_has_buildings(self) -> None:
        """Settlement pipeline produces buildings at typical game size."""
        # Use actual game dimensions (80x43) to catch sizing issues
        generator = create_pipeline("settlement", width=80, height=43, seed="11111")
        map_data = generator.generate()

        # Should have building regions - require at least 2
        building_regions = [
            r for r in map_data.regions.values() if r.region_type == "building"
        ]
        assert len(building_regions) >= 2, (
            f"Too few buildings at 80x43: {len(building_regions)} (need at least 2)"
        )

    def test_settlement_has_buildings_larger_map(self) -> None:
        """Larger maps should have more buildings."""
        generator = create_pipeline("settlement", width=120, height=100, seed="11112")
        map_data = generator.generate()

        building_regions = [
            r for r in map_data.regions.values() if r.region_type == "building"
        ]
        # Larger map should have at least 4 buildings
        assert len(building_regions) >= 4, (
            f"Too few buildings at 120x100: {len(building_regions)} (need at least 4)"
        )

    def test_settlement_has_terrain_variety(self) -> None:
        """Settlement has varied outdoor terrain."""
        generator = create_pipeline("settlement", width=100, height=80, seed="22222")
        map_data = generator.generate()

        # Check for terrain variety in outdoor areas
        terrain_types = {
            TileTypeID.GRASS,
            TileTypeID.DIRT_PATH,
            TileTypeID.GRAVEL,
            TileTypeID.COBBLESTONE,
        }

        unique_tiles = set(np.unique(map_data.tiles))
        terrain_found = unique_tiles & terrain_types

        # Should have at least 2 different outdoor terrain types
        assert len(terrain_found) >= 2, f"Insufficient terrain variety: {terrain_found}"


# =============================================================================
# T5.2: GameWorld Integration
# =============================================================================


class TestGameWorldIntegration:
    """Tests for pipeline integration with GameWorld."""

    def test_game_world_loads_pipeline_map(self) -> None:
        """generator_type='pipeline_settlement' works."""
        # Note: This tests the generator creation, not full GameWorld
        # since GameWorld has many other dependencies
        from catley.environment.generators.pipeline import create_pipeline

        generator = create_pipeline("settlement", width=80, height=43)
        map_data = generator.generate()

        # Verify the map data can be used
        assert map_data.tiles is not None
        assert map_data.regions is not None
        assert map_data.tile_to_region_id is not None

    def test_player_can_spawn(self) -> None:
        """Spawn point selection succeeds - map has walkable tiles."""
        generator = create_pipeline("settlement", width=80, height=60, seed="33333")
        map_data = generator.generate()

        # Find walkable tiles for potential spawn
        walkable_tiles = {
            TileTypeID.FLOOR,
            TileTypeID.COBBLESTONE,
            TileTypeID.GRASS,
            TileTypeID.DIRT_PATH,
            TileTypeID.GRAVEL,
        }

        walkable_positions = [
            (x, y)
            for x in range(map_data.tiles.shape[0])
            for y in range(map_data.tiles.shape[1])
            if map_data.tiles[x, y] in walkable_tiles
        ]

        # Should have many walkable spawn positions
        assert len(walkable_positions) > 100, (
            f"Too few spawn positions: {len(walkable_positions)}"
        )

    def test_lighting_indoor_outdoor(self) -> None:
        """Building regions get no sun, exterior gets sun."""
        generator = create_pipeline("settlement", width=100, height=80, seed="44444")
        map_data = generator.generate()

        # Check exterior regions have sky exposure
        exterior_regions = [
            r for r in map_data.regions.values() if r.region_type == "exterior"
        ]
        for region in exterior_regions:
            assert region.sky_exposure > 0, (
                f"Exterior region {region.id} has no sky exposure"
            )

        # Check building regions have no sky exposure
        building_regions = [
            r for r in map_data.regions.values() if r.region_type == "building"
        ]
        for region in building_regions:
            assert region.sky_exposure == 0, (
                f"Building region {region.id} has sky exposure {region.sky_exposure}"
            )


class TestPipelineFactory:
    """Tests for the pipeline factory functions."""

    def test_create_pipeline_settlement(self) -> None:
        """Factory creates settlement pipeline."""
        generator = create_pipeline("settlement", width=50, height=40)
        assert generator is not None
        map_data = generator.generate()
        assert map_data.tiles.shape == (50, 40)

    def test_create_pipeline_unknown_raises(self) -> None:
        """Unknown pipeline name raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Unknown pipeline name"):
            create_pipeline("nonexistent", width=50, height=40)

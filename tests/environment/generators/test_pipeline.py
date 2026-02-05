"""Tests for the pipeline infrastructure.

Phase T3 tests cover:
- T3.1: GenerationContext
- T3.2: PipelineGenerator
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from catley.environment.generators.base import GeneratedMapData
from catley.environment.generators.pipeline import (
    GenerationContext,
    GenerationLayer,
    PipelineGenerator,
)
from catley.environment.map import MapRegion
from catley.environment.tile_types import TileTypeID
from catley.util import rng
from catley.util.coordinates import Rect

# =============================================================================
# T3.1: GenerationContext
# =============================================================================


class TestGenerationContext:
    """Tests for GenerationContext dataclass."""

    def test_create_empty_initializes_arrays(self) -> None:
        """tiles/tile_to_region_id have correct shape and dtype."""
        ctx = GenerationContext.create_empty(width=40, height=30)

        # Check tiles array
        assert ctx.tiles.shape == (40, 30)
        assert ctx.tiles.dtype == np.uint8

        # Check tile_to_region_id array
        assert ctx.tile_to_region_id.shape == (40, 30)
        assert ctx.tile_to_region_id.dtype == np.int16

        # Default fill is WALL
        assert np.all(ctx.tiles == TileTypeID.WALL)

        # Regions start at -1 (unassigned)
        assert np.all(ctx.tile_to_region_id == -1)

    def test_create_empty_custom_fill_tile(self) -> None:
        """create_empty respects fill_tile parameter."""
        ctx = GenerationContext.create_empty(
            width=20, height=20, fill_tile=TileTypeID.FLOOR
        )

        assert np.all(ctx.tiles == TileTypeID.FLOOR)

    def test_create_empty_basic(self) -> None:
        """create_empty works without seed parameter."""
        ctx = GenerationContext.create_empty(width=10, height=10)
        assert ctx.width == 10
        assert ctx.height == 10

    def test_next_region_id_increments(self) -> None:
        """Each call returns unique incrementing ID."""
        ctx = GenerationContext.create_empty(width=10, height=10)

        assert ctx.next_region_id() == 0
        assert ctx.next_region_id() == 1
        assert ctx.next_region_id() == 2
        assert ctx.next_region_id() == 3

    def test_add_region(self) -> None:
        """add_region stores region in dictionary."""
        ctx = GenerationContext.create_empty(width=10, height=10)

        region = MapRegion.create_outdoor_region(
            map_region_id=5,
            region_type="exterior",
            bounds=[Rect(0, 0, 10, 10)],
            sky_exposure=1.0,
        )
        ctx.add_region(region)

        assert 5 in ctx.regions
        assert ctx.regions[5] == region

    def test_to_generated_map_data_format(self) -> None:
        """Output is valid GeneratedMapData."""
        ctx = GenerationContext.create_empty(width=25, height=20)

        # Add a region
        region_id = ctx.next_region_id()
        region = MapRegion.create_outdoor_region(
            map_region_id=region_id,
            region_type="exterior",
            bounds=[Rect(0, 0, 25, 20)],
            sky_exposure=1.0,
        )
        ctx.add_region(region)
        ctx.tile_to_region_id[:, :] = region_id

        map_data = ctx.to_generated_map_data()

        assert isinstance(map_data, GeneratedMapData)
        assert map_data.tiles is ctx.tiles
        assert map_data.regions is ctx.regions
        assert map_data.tile_to_region_id is ctx.tile_to_region_id


# =============================================================================
# T3.2: PipelineGenerator
# =============================================================================


class RecordingLayer(GenerationLayer):
    """Test layer that records when it was applied."""

    call_order: ClassVar[list[str]] = []

    def __init__(self, name: str) -> None:
        self.name = name

    def apply(self, ctx: GenerationContext) -> None:
        RecordingLayer.call_order.append(self.name)
        # Mark the context to prove we modified it
        ctx.tiles[0, 0] = TileTypeID.FLOOR


class MutatingLayer(GenerationLayer):
    """Test layer that makes specific mutations."""

    def __init__(self, tile_value: TileTypeID) -> None:
        self.tile_value = tile_value

    def apply(self, ctx: GenerationContext) -> None:
        # Set all tiles to our value
        ctx.tiles[:, :] = self.tile_value


class TestPipelineGenerator:
    """Tests for PipelineGenerator."""

    def test_layers_applied_in_order(self) -> None:
        """Mock layers that record call order."""
        # Reset call order
        RecordingLayer.call_order = []

        layers = [
            RecordingLayer("first"),
            RecordingLayer("second"),
            RecordingLayer("third"),
        ]

        generator = PipelineGenerator(
            layers=layers,
            map_width=10,
            map_height=10,
        )
        generator.generate()

        assert RecordingLayer.call_order == ["first", "second", "third"]

    def test_context_passed_between_layers(self) -> None:
        """Mutations from layer N visible to layer N+1."""

        class CheckerLayer(GenerationLayer):
            """Layer that checks and records what previous layers did."""

            def __init__(self) -> None:
                self.saw_floor = False

            def apply(self, ctx: GenerationContext) -> None:
                # Check if previous layer set tile to FLOOR
                self.saw_floor = ctx.tiles[0, 0] == TileTypeID.FLOOR

        # First layer sets (0,0) to FLOOR
        first_layer = MutatingLayer(TileTypeID.FLOOR)
        checker = CheckerLayer()

        generator = PipelineGenerator(
            layers=[first_layer, checker],
            map_width=10,
            map_height=10,
        )
        generator.generate()

        # The checker layer should have seen the FLOOR tile
        assert checker.saw_floor

    def test_generate_returns_map_data(self) -> None:
        """Final output is GeneratedMapData."""

        class SimpleLayer(GenerationLayer):
            """Layer that creates a minimal valid map."""

            def apply(self, ctx: GenerationContext) -> None:
                ctx.tiles[:, :] = TileTypeID.OUTDOOR_FLOOR
                region_id = ctx.next_region_id()
                region = MapRegion.create_outdoor_region(
                    map_region_id=region_id,
                    region_type="exterior",
                    bounds=[Rect(0, 0, ctx.width, ctx.height)],
                    sky_exposure=1.0,
                )
                ctx.add_region(region)
                ctx.tile_to_region_id[:, :] = region_id

        generator = PipelineGenerator(
            layers=[SimpleLayer()],
            map_width=30,
            map_height=25,
        )
        map_data = generator.generate()

        assert isinstance(map_data, GeneratedMapData)
        assert map_data.tiles.shape == (30, 25)
        assert len(map_data.regions) == 1
        assert 0 in map_data.regions

    def test_generator_with_seed(self) -> None:
        """Seed produces deterministic generation."""
        _rng = rng.get("test.random_layer")

        class RandomLayer(GenerationLayer):
            """Layer that uses RNG to modify tiles."""

            def apply(self, ctx: GenerationContext) -> None:
                for x in range(ctx.width):
                    for y in range(ctx.height):
                        if _rng.random() < 0.5:
                            ctx.tiles[x, y] = TileTypeID.FLOOR
                        else:
                            ctx.tiles[x, y] = TileTypeID.OUTDOOR_FLOOR

        # Generate twice with same seed
        gen1 = PipelineGenerator(
            layers=[RandomLayer()],
            map_width=20,
            map_height=20,
            seed="12345",
        )
        map1 = gen1.generate()

        gen2 = PipelineGenerator(
            layers=[RandomLayer()],
            map_width=20,
            map_height=20,
            seed="12345",
        )
        map2 = gen2.generate()

        np.testing.assert_array_equal(map1.tiles, map2.tiles)

    def test_empty_pipeline(self) -> None:
        """Pipeline with no layers produces empty map."""
        generator = PipelineGenerator(
            layers=[],
            map_width=15,
            map_height=15,
        )
        map_data = generator.generate()

        # All tiles should be WALL (default fill)
        assert np.all(map_data.tiles == TileTypeID.WALL)
        # No regions
        assert len(map_data.regions) == 0

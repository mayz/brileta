"""Tests for audio material resolution system."""

import pytest

from brileta.environment.tile_types import TileTypeID, get_tile_material
from brileta.game.enums import ImpactMaterial
from brileta.sound.definitions import SOUND_DEFINITIONS, get_sound_definition
from brileta.sound.materials import (
    IMPACT_SOUND_MAP,
    AudioMaterialResolver,
    get_impact_sound_id,
)


class TestImpactMaterial:
    """Test ImpactMaterial enum."""

    def test_all_materials_defined(self) -> None:
        """Test that all expected materials exist."""
        assert ImpactMaterial.FLESH
        assert ImpactMaterial.METAL
        assert ImpactMaterial.STONE
        assert ImpactMaterial.WOOD

    def test_materials_are_distinct(self) -> None:
        """Test that all materials have distinct values."""
        materials = [m.value for m in ImpactMaterial]
        assert len(materials) == len(set(materials))


class TestGetTileMaterial:
    """Test get_tile_material function in tile_types module."""

    def test_wall_is_stone(self) -> None:
        """Test that walls return stone material."""
        assert get_tile_material(TileTypeID.WALL) == ImpactMaterial.STONE

    def test_floor_is_stone(self) -> None:
        """Test that floors return stone material."""
        assert get_tile_material(TileTypeID.FLOOR) == ImpactMaterial.STONE

    def test_cobblestone_is_stone(self) -> None:
        """Test that cobblestone returns stone material."""
        assert get_tile_material(TileTypeID.COBBLESTONE) == ImpactMaterial.STONE

    def test_door_closed_is_wood(self) -> None:
        """Test that closed doors return wood material."""
        assert get_tile_material(TileTypeID.DOOR_CLOSED) == ImpactMaterial.WOOD

    def test_door_open_is_wood(self) -> None:
        """Test that open doors return wood material."""
        assert get_tile_material(TileTypeID.DOOR_OPEN) == ImpactMaterial.WOOD

    def test_boulder_is_stone(self) -> None:
        """Test that boulders return stone material."""
        assert get_tile_material(TileTypeID.BOULDER) == ImpactMaterial.STONE

    def test_invalid_tile_id_returns_stone(self) -> None:
        """Test that invalid tile IDs default to stone."""
        assert get_tile_material(999) == ImpactMaterial.STONE
        assert get_tile_material(-1) == ImpactMaterial.STONE

    @pytest.mark.parametrize("tile_id", list(TileTypeID))
    def test_all_tile_types_return_valid_material(self, tile_id: TileTypeID) -> None:
        """Test that every tile type returns a valid ImpactMaterial."""
        material = get_tile_material(tile_id)
        assert isinstance(material, ImpactMaterial)


class TestAudioMaterialResolver:
    """Test AudioMaterialResolver class."""

    def test_resolve_actor_material_returns_flesh(self) -> None:
        """Test that resolve_actor_material returns FLESH for all actors.

        Currently all biological creatures return FLESH. When armor types
        are implemented, this behavior may change.
        """
        # Create a mock actor - we just need something that can be passed
        # The current implementation ignores the actor and returns FLESH
        from unittest.mock import MagicMock

        mock_actor = MagicMock()
        result = AudioMaterialResolver.resolve_actor_material(mock_actor)
        assert result == ImpactMaterial.FLESH

    def test_resolve_tile_material_delegates_to_get_tile_material(self) -> None:
        """Test that resolve_tile_material uses get_tile_material."""
        # Test a few known tile types
        assert (
            AudioMaterialResolver.resolve_tile_material(TileTypeID.WALL)
            == ImpactMaterial.STONE
        )
        assert (
            AudioMaterialResolver.resolve_tile_material(TileTypeID.DOOR_CLOSED)
            == ImpactMaterial.WOOD
        )


class TestGetImpactSoundId:
    """Test get_impact_sound_id function."""

    def test_flesh_returns_impact_flesh(self) -> None:
        """Test that FLESH material returns impact_flesh sound ID."""
        assert get_impact_sound_id(ImpactMaterial.FLESH) == "impact_flesh"

    def test_metal_returns_impact_metal(self) -> None:
        """Test that METAL material returns impact_metal sound ID."""
        assert get_impact_sound_id(ImpactMaterial.METAL) == "impact_metal"

    def test_stone_returns_impact_stone(self) -> None:
        """Test that STONE material returns impact_stone sound ID."""
        assert get_impact_sound_id(ImpactMaterial.STONE) == "impact_stone"

    def test_wood_returns_impact_wood(self) -> None:
        """Test that WOOD material returns impact_wood sound ID."""
        assert get_impact_sound_id(ImpactMaterial.WOOD) == "impact_wood"

    @pytest.mark.parametrize("material", list(ImpactMaterial))
    def test_all_materials_have_sound_mapping(self, material: ImpactMaterial) -> None:
        """Test that every material has a corresponding sound ID."""
        sound_id = get_impact_sound_id(material)
        assert isinstance(sound_id, str)
        assert len(sound_id) > 0


class TestImpactSoundDefinitions:
    """Test that impact sound definitions exist and are valid."""

    @pytest.mark.parametrize(
        "sound_id",
        ["impact_flesh", "impact_metal", "impact_stone", "impact_wood"],
    )
    def test_impact_sound_definition_exists(self, sound_id: str) -> None:
        """Test that each impact sound definition exists in SOUND_DEFINITIONS."""
        assert sound_id in SOUND_DEFINITIONS
        definition = get_sound_definition(sound_id)
        assert definition is not None
        assert definition.sound_id == sound_id

    @pytest.mark.parametrize(
        "sound_id",
        ["impact_flesh", "impact_metal", "impact_stone", "impact_wood"],
    )
    def test_impact_sound_has_valid_structure(self, sound_id: str) -> None:
        """Test that each impact sound has valid layers and settings."""
        definition = SOUND_DEFINITIONS[sound_id]

        # Must have at least one layer
        assert len(definition.layers) >= 1

        # Check primary layer
        primary_layer = definition.layers[0]
        assert primary_layer.file.endswith(".ogg")
        assert primary_layer.loop is False  # Impact sounds should not loop
        assert 0.0 < primary_layer.volume <= 1.0

        # Check pitch variation exists
        assert primary_layer.pitch_variation is not None
        assert len(primary_layer.pitch_variation) == 2

        # Check overall settings
        assert 0.0 < definition.base_volume <= 1.0
        assert definition.falloff_start > 0.0
        assert definition.max_distance > definition.falloff_start
        assert 1 <= definition.priority <= 10

    @pytest.mark.parametrize(
        "sound_id",
        ["impact_flesh", "impact_metal", "impact_stone", "impact_wood"],
    )
    def test_impact_sound_has_variants(self, sound_id: str) -> None:
        """Test that each impact sound has variant files for variety."""
        definition = SOUND_DEFINITIONS[sound_id]
        primary_layer = definition.layers[0]

        # Each impact sound should have variants for audio variety
        assert primary_layer.variants is not None
        assert len(primary_layer.variants) >= 2  # At least 2 variants


class TestMaterialSoundIntegration:
    """Test the integration between materials and sounds."""

    def test_all_materials_map_to_valid_sounds(self) -> None:
        """Test that every material maps to a sound that exists."""
        for material in ImpactMaterial:
            sound_id = IMPACT_SOUND_MAP[material]
            definition = get_sound_definition(sound_id)
            assert definition is not None, (
                f"Material {material.name} maps to non-existent sound {sound_id}"
            )

    def test_all_materials_in_sound_map(self) -> None:
        """Test that every ImpactMaterial has an entry in IMPACT_SOUND_MAP."""
        for material in ImpactMaterial:
            assert material in IMPACT_SOUND_MAP, (
                f"Material {material.name} missing from IMPACT_SOUND_MAP"
            )

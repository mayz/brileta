"""Tests for sound definitions and data structures."""

import pytest

from brileta.sound.definitions import (
    SOUND_DEFINITIONS,
    SoundDefinition,
    SoundLayer,
    get_sound_definition,
)
from brileta.types import SoundId


class TestSoundLayer:
    """Test SoundLayer dataclass."""

    def test_sound_layer_defaults(self) -> None:
        """Test SoundLayer with default values."""
        layer = SoundLayer(file="test.ogg")

        assert layer.file == "test.ogg"
        assert layer.volume == 1.0
        assert layer.loop is True
        assert layer.interval is None

    def test_sound_layer_custom_values(self) -> None:
        """Test SoundLayer with custom values."""
        layer = SoundLayer(
            file="custom.wav", volume=0.5, loop=False, interval=(2.0, 5.0)
        )

        assert layer.file == "custom.wav"
        assert layer.volume == 0.5
        assert layer.loop is False
        assert layer.interval == (2.0, 5.0)

    def test_sound_layer_interval_sound(self) -> None:
        """Test SoundLayer configured for interval-based playback."""
        layer = SoundLayer(file="pop.ogg", volume=0.3, loop=False, interval=(1.0, 3.0))

        assert layer.loop is False
        assert layer.interval == (1.0, 3.0)
        assert layer.volume == 0.3


class TestSoundDefinition:
    """Test SoundDefinition dataclass."""

    def test_sound_definition_defaults(self) -> None:
        """Test SoundDefinition with default values."""
        layers = [SoundLayer(file="test.ogg")]
        definition = SoundDefinition(sound_id="test_sound", layers=layers)

        assert definition.sound_id == "test_sound"
        assert definition.layers == layers
        assert definition.base_volume == 1.0
        assert definition.falloff_start == 2.0
        assert definition.max_distance == 10.0
        assert definition.priority == 5

    def test_sound_definition_custom_values(self) -> None:
        """Test SoundDefinition with custom values."""
        layers = [
            SoundLayer(file="loop.ogg", volume=0.8),
            SoundLayer(file="accent.ogg", volume=0.2, loop=False, interval=(2.0, 4.0)),
        ]
        definition = SoundDefinition(
            sound_id="custom_sound",
            layers=layers,
            base_volume=0.6,
            falloff_start=3.0,
            max_distance=15.0,
            priority=7,
        )

        assert definition.sound_id == "custom_sound"
        assert definition.layers == layers
        assert definition.base_volume == 0.6
        assert definition.falloff_start == 3.0
        assert definition.max_distance == 15.0
        assert definition.priority == 7

    def test_sound_definition_with_multiple_layers(self) -> None:
        """Test SoundDefinition with multiple sound layers."""
        layers = [
            SoundLayer(file="base.ogg", volume=1.0, loop=True),
            SoundLayer(file="crackle.ogg", volume=0.4, loop=False, interval=(1.0, 4.0)),
            SoundLayer(file="pop.ogg", volume=0.2, loop=False, interval=(3.0, 8.0)),
        ]
        definition = SoundDefinition(sound_id="complex_fire", layers=layers)

        assert len(definition.layers) == 3
        assert definition.layers[0].loop is True
        assert definition.layers[1].loop is False
        assert definition.layers[2].loop is False
        assert definition.layers[1].interval == (1.0, 4.0)
        assert definition.layers[2].interval == (3.0, 8.0)


class TestSoundDefinitionLookup:
    """Test sound definition lookup functionality."""

    def test_get_sound_definition_valid_id(self) -> None:
        """Test getting a valid sound definition."""
        sound_def = get_sound_definition("fire_ambient")

        assert sound_def is not None
        assert sound_def.sound_id == "fire_ambient"
        assert isinstance(sound_def.layers, list)
        assert len(sound_def.layers) > 0

    def test_get_sound_definition_invalid_id(self) -> None:
        """Test getting an invalid sound definition."""
        sound_def = get_sound_definition("nonexistent_sound")

        assert sound_def is None

    def test_get_sound_definition_empty_string(self) -> None:
        """Test getting sound definition with empty string."""
        sound_def = get_sound_definition("")

        assert sound_def is None

    def test_get_sound_definition_type_consistency(self) -> None:
        """Test that function accepts SoundId type."""
        sound_id: SoundId = "fire_ambient"
        sound_def = get_sound_definition(sound_id)

        assert sound_def is not None
        assert sound_def.sound_id == sound_id


class TestBuiltInSoundDefinitions:
    """Test the built-in sound definitions."""

    def test_sound_definitions_not_empty(self) -> None:
        """Test that SOUND_DEFINITIONS contains entries."""
        assert len(SOUND_DEFINITIONS) > 0

    def test_fire_ambient_definition(self) -> None:
        """Test the fire_ambient sound definition."""
        fire_def = SOUND_DEFINITIONS["fire_ambient"]

        assert fire_def.sound_id == "fire_ambient"
        assert len(fire_def.layers) == 4  # Loop + 3 interval layers for variation
        assert fire_def.base_volume == 0.7
        assert fire_def.falloff_start == 2.8
        assert fire_def.max_distance == 17.0
        assert fire_def.priority == 6

        # Check layers
        loop_layer = fire_def.layers[0]
        assert loop_layer.file == "fire_crackle_loop.ogg"
        assert loop_layer.loop is True

        # Test the primary pop layer
        interval_layer = fire_def.layers[1]
        assert interval_layer.file == "fire_pops.ogg"
        assert interval_layer.loop is False
        assert interval_layer.interval == (1.5, 6.0)

        # Test that we have multiple pop layers with different intervals
        pop_layers = [layer for layer in fire_def.layers if not layer.loop]
        assert len(pop_layers) == 3  # Three different pop layers

        # Verify they all use the same file but have different properties
        for layer in pop_layers:
            assert layer.file == "fire_pops.ogg"
            assert layer.loop is False
            assert layer.interval is not None

    def test_all_definitions_have_consistent_sound_ids(self) -> None:
        """Test that all definitions have sound_id matching their dictionary key."""
        for key, definition in SOUND_DEFINITIONS.items():
            assert definition.sound_id == key, (
                f"Definition {key} has mismatched sound_id: {definition.sound_id}"
            )

    def test_all_definitions_have_valid_structure(self) -> None:
        """Test that all built-in definitions have valid structure."""
        for sound_id, definition in SOUND_DEFINITIONS.items():
            # Basic structure checks
            assert isinstance(definition.sound_id, str), (
                f"{sound_id}: sound_id must be string"
            )
            assert isinstance(definition.layers, list), (
                f"{sound_id}: layers must be list"
            )
            assert len(definition.layers) > 0, (
                f"{sound_id}: must have at least one layer"
            )

            # Numeric value checks
            assert 0.0 <= definition.base_volume <= 1.0, (
                f"{sound_id}: base_volume must be 0.0-1.0"
            )
            assert definition.falloff_start >= 0.0, (
                f"{sound_id}: falloff_start must be >= 0"
            )
            assert definition.max_distance > definition.falloff_start, (
                f"{sound_id}: max_distance must be > falloff_start"
            )
            assert 1 <= definition.priority <= 10, f"{sound_id}: priority must be 1-10"

            # Layer checks
            for i, layer in enumerate(definition.layers):
                assert isinstance(layer.file, str), (
                    f"{sound_id} layer {i}: file must be string"
                )
                assert len(layer.file) > 0, (
                    f"{sound_id} layer {i}: file must not be empty"
                )
                assert 0.0 <= layer.volume <= 1.0, (
                    f"{sound_id} layer {i}: volume must be 0.0-1.0"
                )

                if not layer.loop and layer.interval:
                    assert len(layer.interval) == 2, (
                        f"{sound_id} layer {i}: interval must be tuple of 2 values"
                    )
                    assert layer.interval[0] > 0, (
                        f"{sound_id} layer {i}: interval min must be > 0"
                    )
                    assert layer.interval[1] >= layer.interval[0], (
                        f"{sound_id} layer {i}: interval max must be >= min"
                    )

    @pytest.mark.parametrize(
        "sound_id", ["fire_ambient", "waterfall_ambient", "engine_loop", "steam_hiss"]
    )
    def test_individual_sound_definitions(self, sound_id: str) -> None:
        """Test each built-in sound definition individually."""
        definition = SOUND_DEFINITIONS[sound_id]

        assert definition.sound_id == sound_id
        assert len(definition.layers) > 0
        assert definition.base_volume > 0.0
        assert definition.max_distance > 0.0
        assert definition.falloff_start >= 0.0

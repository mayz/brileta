"""Tests for SoundEmitter class."""

import pytest

from catley.sound.emitter import SoundEmitter
from catley.types import SoundId


class TestSoundEmitterInitialization:
    """Test SoundEmitter initialization."""

    def test_emitter_with_defaults(self) -> None:
        """Test SoundEmitter with default parameters."""
        emitter = SoundEmitter("fire_ambient")

        assert emitter.sound_id == "fire_ambient"
        assert emitter.volume_multiplier == 1.0
        assert emitter.active is True
        assert emitter.playing_instances == []

    def test_emitter_with_custom_values(self) -> None:
        """Test SoundEmitter with custom parameters."""
        emitter = SoundEmitter(
            sound_id="waterfall_ambient", volume_multiplier=0.7, active=False
        )

        assert emitter.sound_id == "waterfall_ambient"
        assert emitter.volume_multiplier == 0.7
        assert emitter.active is False
        assert emitter.playing_instances == []

    def test_emitter_with_sound_id_type(self) -> None:
        """Test SoundEmitter accepts SoundId type."""
        sound_id: SoundId = "engine_loop"
        emitter = SoundEmitter(sound_id)

        assert emitter.sound_id == sound_id
        assert emitter.sound_id == "engine_loop"

    def test_emitter_playing_instances_is_new_list(self) -> None:
        """Test that each emitter gets its own playing_instances list."""
        emitter1 = SoundEmitter("fire_ambient")
        emitter2 = SoundEmitter("waterfall_ambient")

        assert emitter1.playing_instances is not emitter2.playing_instances
        assert emitter1.playing_instances == []
        assert emitter2.playing_instances == []


class TestSoundEmitterActivation:
    """Test SoundEmitter activation methods."""

    def test_set_active_true(self) -> None:
        """Test setting emitter to active."""
        emitter = SoundEmitter("fire_ambient", active=False)
        assert emitter.active is False

        emitter.set_active(True)
        assert emitter.active is True

    def test_set_active_false(self) -> None:
        """Test setting emitter to inactive."""
        emitter = SoundEmitter("fire_ambient", active=True)
        assert emitter.active is True

        emitter.set_active(False)
        assert emitter.active is False

    def test_set_active_idempotent(self) -> None:
        """Test that setting the same active state multiple times works."""
        emitter = SoundEmitter("fire_ambient", active=True)

        emitter.set_active(True)
        emitter.set_active(True)
        assert emitter.active is True

        emitter.set_active(False)
        emitter.set_active(False)
        assert emitter.active is False


class TestSoundEmitterVolumeMultiplier:
    """Test SoundEmitter volume multiplier methods."""

    def test_set_volume_multiplier_normal(self) -> None:
        """Test setting volume multiplier to normal values."""
        emitter = SoundEmitter("fire_ambient")

        emitter.set_volume_multiplier(0.5)
        assert emitter.volume_multiplier == 0.5

        emitter.set_volume_multiplier(0.8)
        assert emitter.volume_multiplier == 0.8

    def test_set_volume_multiplier_clamping_low(self) -> None:
        """Test that volume multiplier is clamped to minimum 0.0."""
        emitter = SoundEmitter("fire_ambient")

        emitter.set_volume_multiplier(-0.5)
        assert emitter.volume_multiplier == 0.0

        emitter.set_volume_multiplier(-10.0)
        assert emitter.volume_multiplier == 0.0

    def test_set_volume_multiplier_clamping_high(self) -> None:
        """Test that volume multiplier is clamped to maximum 1.0."""
        emitter = SoundEmitter("fire_ambient")

        emitter.set_volume_multiplier(1.5)
        assert emitter.volume_multiplier == 1.0

        emitter.set_volume_multiplier(10.0)
        assert emitter.volume_multiplier == 1.0

    def test_set_volume_multiplier_edge_cases(self) -> None:
        """Test volume multiplier edge cases."""
        emitter = SoundEmitter("fire_ambient")

        # Test exact boundaries
        emitter.set_volume_multiplier(0.0)
        assert emitter.volume_multiplier == 0.0

        emitter.set_volume_multiplier(1.0)
        assert emitter.volume_multiplier == 1.0

        # Test very small positive value
        emitter.set_volume_multiplier(0.001)
        assert emitter.volume_multiplier == 0.001

    def test_volume_multiplier_precision(self) -> None:
        """Test that volume multiplier maintains floating point precision."""
        emitter = SoundEmitter("fire_ambient")

        test_values = [0.1, 0.33333, 0.66667, 0.9999]
        for value in test_values:
            emitter.set_volume_multiplier(value)
            assert abs(emitter.volume_multiplier - value) < 1e-10


class TestSoundEmitterState:
    """Test SoundEmitter state management."""

    def test_emitter_state_independence(self) -> None:
        """Test that emitter state changes don't affect other emitters."""
        emitter1 = SoundEmitter("fire_ambient", volume_multiplier=0.5, active=True)
        emitter2 = SoundEmitter(
            "waterfall_ambient", volume_multiplier=0.8, active=False
        )

        # Modify emitter1
        emitter1.set_active(False)
        emitter1.set_volume_multiplier(0.2)

        # Check that emitter2 is unchanged
        assert emitter2.active is False
        assert emitter2.volume_multiplier == 0.8
        assert emitter2.sound_id == "waterfall_ambient"

    def test_emitter_with_different_sound_ids(self) -> None:
        """Test emitters with different sound IDs."""
        sound_ids = ["fire_ambient", "waterfall_ambient", "engine_loop", "steam_hiss"]
        emitters = [SoundEmitter(sound_id) for sound_id in sound_ids]

        for i, emitter in enumerate(emitters):
            assert emitter.sound_id == sound_ids[i]
            assert emitter.volume_multiplier == 1.0
            assert emitter.active is True

    def test_emitter_playing_instances_persistence(self) -> None:
        """Test that playing_instances list persists through state changes."""
        emitter = SoundEmitter("fire_ambient")
        original_list = emitter.playing_instances

        # Modify emitter state
        emitter.set_active(False)
        emitter.set_volume_multiplier(0.5)

        # playing_instances should be the same list object
        assert emitter.playing_instances is original_list
        assert emitter.playing_instances == []


class TestSoundEmitterEdgeCases:
    """Test SoundEmitter edge cases and unusual inputs."""

    def test_emitter_with_empty_sound_id(self) -> None:
        """Test SoundEmitter with empty sound ID."""
        emitter = SoundEmitter("")

        assert emitter.sound_id == ""
        assert emitter.volume_multiplier == 1.0
        assert emitter.active is True

    def test_emitter_with_very_long_sound_id(self) -> None:
        """Test SoundEmitter with very long sound ID."""
        long_id = "very_long_sound_id_" * 100
        emitter = SoundEmitter(long_id)

        assert emitter.sound_id == long_id
        assert len(emitter.sound_id) == len(long_id)

    def test_emitter_with_special_characters_in_sound_id(self) -> None:
        """Test SoundEmitter with special characters in sound ID."""
        special_ids = [
            "sound-with-dashes",
            "sound_with_underscores",
            "sound.with.dots",
            "sound/with/slashes",
            "sound with spaces",
            "UPPERCASE_SOUND",
            "123_numeric_sound",
        ]

        for sound_id in special_ids:
            emitter = SoundEmitter(sound_id)
            assert emitter.sound_id == sound_id

    @pytest.mark.parametrize("volume", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_emitter_with_various_initial_volumes(self, volume: float) -> None:
        """Test SoundEmitter with various initial volume multipliers."""
        emitter = SoundEmitter("fire_ambient", volume_multiplier=volume)
        assert emitter.volume_multiplier == volume

    @pytest.mark.parametrize("active_state", [True, False])
    def test_emitter_with_various_active_states(self, active_state: bool) -> None:
        """Test SoundEmitter with various initial active states."""
        emitter = SoundEmitter("fire_ambient", active=active_state)
        assert emitter.active == active_state

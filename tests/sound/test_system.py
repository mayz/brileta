"""Tests for SoundSystem class."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from brileta.game.actors.core import Actor
from brileta.sound.audio_backend import AudioBackend, AudioChannel, LoadedSound
from brileta.sound.definitions import SoundDefinition, SoundLayer, get_sound_definition
from brileta.sound.emitter import SoundEmitter
from brileta.sound.system import SoundSystem
from brileta.util.spatial import SpatialHashGrid, SpatialIndex
from tests.helpers import dt


def create_spatial_index(actors: list[Actor]) -> SpatialIndex[Actor]:
    """Create a spatial index from a list of actors for testing."""
    spatial_index = SpatialHashGrid[Actor](cell_size=16)
    for actor in actors:
        spatial_index.add(actor)
    return spatial_index


class MockAudioChannel(AudioChannel):
    """Mock audio channel for testing that tracks its playing state."""

    def __init__(self) -> None:
        self._is_playing = False
        self._volume = 1.0
        self._current_sound: LoadedSound | None = None
        self._loop = False
        self._current_priority = 0

    @property
    def current_priority(self) -> int:
        """Get the priority of the currently playing sound."""
        return self._current_priority if self.is_playing() else 0

    def play(
        self,
        sound: LoadedSound,
        volume: float = 1.0,
        loop: bool = False,
        priority: int = 5,
    ) -> None:
        """Start playing a sound on this channel."""
        self._current_sound = sound
        self._volume = volume
        self._loop = loop
        self._current_priority = priority
        self._is_playing = True

    def stop(self) -> None:
        """Stop playback on this channel."""
        self._is_playing = False
        self._current_sound = None
        self._loop = False
        self._current_priority = 0

    def set_volume(self, volume: float) -> None:
        """Update the volume of this channel."""
        self._volume = volume
        if volume <= 0.0:
            # When volume is set to 0, stop the channel to free resources
            self.stop()

    def is_playing(self) -> bool:
        """Check if this channel is currently playing audio."""
        return self._is_playing

    def fadeout(self, duration_ms: int) -> None:
        """Fade out the channel over the specified duration."""
        # For testing, just stop immediately
        self.stop()


class MockAudioBackend(AudioBackend):
    """Mock audio backend for testing that tracks channel usage."""

    def __init__(self, max_channels: int = 16) -> None:
        self._max_channels = max_channels
        self._channels: list[MockAudioChannel] = []
        self._initialized = False
        self._master_volume = 1.0

    def initialize(self, sample_rate: int = 44100, channels: int = 2) -> None:
        """Initialize the audio backend."""
        self._channels = [MockAudioChannel() for _ in range(self._max_channels)]
        self._initialized = True

    def shutdown(self) -> None:
        """Shut down the audio backend and release resources."""
        self._channels.clear()
        self._initialized = False

    def get_channel(self, priority: int = 5) -> AudioChannel | None:
        """Get an available audio channel."""
        if not self._initialized:
            return None

        for channel in self._channels:
            if not channel.is_playing():
                return channel
        return None

    def play_sound(
        self,
        sound: LoadedSound,
        volume: float = 1.0,
        loop: bool = False,
        priority: int = 5,
    ) -> AudioChannel | None:
        """Play a sound on an available channel."""
        channel = self.get_channel(priority)
        if channel:
            channel.play(sound, volume, loop, priority)
        return channel

    def stop_all_sounds(self) -> None:
        """Stop all currently playing sounds."""
        for channel in self._channels:
            if channel.is_playing():
                channel.stop()

    def set_master_volume(self, volume: float) -> None:
        """Set the master volume for all audio."""
        self._master_volume = volume

    def update(self) -> None:
        """Update the audio backend."""
        pass

    def get_active_channel_count(self) -> int:
        """Get the number of channels currently playing audio."""
        return sum(1 for channel in self._channels if channel.is_playing())

    def load_sound(self, file_path: Path) -> LoadedSound:
        """Load a sound file into memory."""
        # Return mock sound data for testing
        return LoadedSound(
            data=np.zeros((1000, 1), dtype=np.float32),  # 1 second of silence at 1kHz
            sample_rate=44100,
            channels=1,
        )


class TestSoundSystemInitialization:
    """Test SoundSystem initialization."""

    def test_sound_system_defaults(self) -> None:
        """Test SoundSystem with default parameters."""
        system = SoundSystem()

        assert system.max_concurrent_sounds == 16
        assert system.playing_sounds == []
        assert system.audio_backend is None
        assert system.current_time == 0.0

    def test_sound_system_custom_max_sounds(self) -> None:
        """Test SoundSystem with custom max concurrent sounds."""
        system = SoundSystem(max_concurrent_sounds=32)

        assert system.max_concurrent_sounds == 32
        assert system.playing_sounds == []

    def test_sound_system_initial_state(self) -> None:
        """Test SoundSystem initial state."""
        system = SoundSystem()

        # Should start with empty state
        assert len(system.playing_sounds) == 0
        assert system.current_time == 0.0


class TestSoundSystemVolumeCalculation:
    """Test SoundSystem volume calculation logic."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.system = SoundSystem()

        # Initialize the audio listener position for tests
        # This simulates what happens in the first update() call
        self.system.audio_listener_x = 0.0
        self.system.audio_listener_y = 0.0
        self.system._listener_initialized = True

        # Create a test sound definition
        self.test_sound_def = SoundDefinition(
            sound_id="test_sound",
            layers=[SoundLayer(file="test.ogg")],
            base_volume=0.8,
            falloff_start=3.0,
            max_distance=10.0,
            rolloff_factor=1.0,  # Standard rolloff for testing
        )

    def test_calculate_volume_within_falloff_start(self) -> None:
        """Test volume calculation when within falloff start distance."""
        # At distance 0 (same position)
        self.system.audio_listener_x = 5.0
        self.system.audio_listener_y = 5.0
        volume = self.system._calculate_volume(5, 5, self.test_sound_def, 1.0)
        assert volume == 0.8  # base_volume

        # At distance 2 (within falloff_start of 3)
        self.system.audio_listener_x = 7.0
        self.system.audio_listener_y = 5.0
        volume = self.system._calculate_volume(5, 5, self.test_sound_def, 1.0)
        assert volume == 0.8  # base_volume

    def test_calculate_volume_at_falloff_start(self) -> None:
        """Test volume calculation exactly at falloff start distance."""
        # At distance 3 (exactly at falloff_start)
        self.system.audio_listener_x = 8.0
        self.system.audio_listener_y = 5.0
        volume = self.system._calculate_volume(5, 5, self.test_sound_def, 1.0)
        assert volume == 0.8  # base_volume

    def test_calculate_volume_beyond_falloff_start(self) -> None:
        """Test volume calculation beyond falloff start with rolloff model."""
        # At distance 4 (1 unit beyond falloff_start of 3)
        self.system.audio_listener_x = 9.0
        self.system.audio_listener_y = 5.0
        volume = self.system._calculate_volume(5, 5, self.test_sound_def, 1.0)
        # Industry rolloff: reference / (reference + rolloff * falloff_distance)
        falloff_distance = 1.0  # distance(4) - falloff_start(3)
        reference_distance = 3.0  # falloff_start
        rolloff_factor = 1.0
        expected = 0.8 * (
            reference_distance
            / (reference_distance + rolloff_factor * falloff_distance)
        )
        assert abs(volume - expected) < 1e-10

    def test_calculate_volume_beyond_max_distance(self) -> None:
        """Test volume calculation beyond max distance."""
        # At distance 15 (beyond max_distance of 10)
        self.system.audio_listener_x = 20.0
        self.system.audio_listener_y = 5.0
        volume = self.system._calculate_volume(5, 5, self.test_sound_def, 1.0)
        assert volume == 0.0

    def test_calculate_volume_at_max_distance(self) -> None:
        """Test volume calculation exactly at max distance."""
        # At distance 10 (exactly at max_distance)
        self.system.audio_listener_x = 15.0
        self.system.audio_listener_y = 5.0
        volume = self.system._calculate_volume(5, 5, self.test_sound_def, 1.0)
        falloff_distance = 10.0 - 3.0  # distance - falloff_start
        reference_distance = 3.0  # falloff_start
        rolloff_factor = 1.0
        expected = 0.8 * (
            reference_distance
            / (reference_distance + rolloff_factor * falloff_distance)
        )
        assert abs(volume - expected) < 1e-10

    def test_calculate_volume_with_volume_multiplier(self) -> None:
        """Test volume calculation with volume multiplier."""
        # Within falloff start with multiplier
        self.system.audio_listener_x = 5.0
        self.system.audio_listener_y = 5.0
        volume = self.system._calculate_volume(5, 5, self.test_sound_def, 0.5)
        assert volume == 0.4  # base_volume * multiplier

        # Beyond falloff start with multiplier
        self.system.audio_listener_x = 9.0
        self.system.audio_listener_y = 5.0
        volume = self.system._calculate_volume(5, 5, self.test_sound_def, 0.5)
        # Industry-standard rolloff with multiplier
        falloff_distance = 1.0  # distance(4) - falloff_start(3)
        reference_distance = 3.0  # falloff_start
        rolloff_factor = 1.0
        volume_factor = reference_distance / (
            reference_distance + rolloff_factor * falloff_distance
        )
        expected = 0.8 * 0.5 * volume_factor
        assert abs(volume - expected) < 1e-10

    def test_calculate_volume_diagonal_distances(self) -> None:
        """Test volume calculation with diagonal distances."""
        # Distance sqrt(3^2 + 4^2) = 5
        self.system.audio_listener_x = 3.0
        self.system.audio_listener_y = 4.0
        volume = self.system._calculate_volume(0, 0, self.test_sound_def, 1.0)
        falloff_distance = 5.0 - 3.0  # distance - falloff_start
        reference_distance = 3.0  # falloff_start
        rolloff_factor = 1.0
        expected = 0.8 * (
            reference_distance
            / (reference_distance + rolloff_factor * falloff_distance)
        )
        assert abs(volume - expected) < 1e-10

    def test_calculate_volume_edge_cases(self) -> None:
        """Test volume calculation edge cases."""
        # Zero volume multiplier
        self.system.audio_listener_x = 5.0
        self.system.audio_listener_y = 5.0
        volume = self.system._calculate_volume(5, 5, self.test_sound_def, 0.0)
        assert volume == 0.0

        # Very small distance beyond falloff_start
        self.system.audio_listener_x = 9.0
        self.system.audio_listener_y = 5.0
        volume = self.system._calculate_volume(5, 5, self.test_sound_def, 1.0)
        falloff_distance = 4.0 - 3.0  # distance(4) - falloff_start(3)
        reference_distance = 3.0  # falloff_start
        rolloff_factor = 1.0
        expected = 0.8 * (
            reference_distance
            / (reference_distance + rolloff_factor * falloff_distance)
        )
        assert volume > 0.0
        assert abs(volume - expected) < 1e-10


class TestSoundSystemActorProcessing:
    """Test SoundSystem actor processing."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.system = SoundSystem()

    def create_mock_actor(
        self, x: int, y: int, emitters: list[SoundEmitter] | None = None
    ) -> Actor:
        """Create a mock actor with sound emitters."""
        actor = MagicMock(spec=Actor)
        actor.x = x
        actor.y = y
        actor.sound_emitters = emitters
        return actor

    def test_update_with_no_actors(self) -> None:
        """Test update with no actors."""
        self.system.update(0, 0, create_spatial_index([]), dt(0.1))

        assert self.system.current_time == 0.1
        assert len(self.system.playing_sounds) == 0

    def test_update_with_actors_no_emitters(self) -> None:
        """Test update with actors that have no sound emitters."""
        actors = [
            self.create_mock_actor(5, 5, None),
            self.create_mock_actor(10, 10, []),
        ]

        self.system.update(0, 0, create_spatial_index(actors), dt(0.1))

        assert self.system.current_time == 0.1
        assert len(self.system.playing_sounds) == 0

    def test_update_with_actors_missing_sound_emitters_attribute(self) -> None:
        """Test update with actors that don't have sound_emitters attribute."""
        # Create actor without sound_emitters attribute
        actor = MagicMock()
        actor.x = 5
        actor.y = 5
        # Explicitly remove the sound_emitters attribute
        if hasattr(actor, "sound_emitters"):
            delattr(actor, "sound_emitters")

        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        assert self.system.current_time == 0.1
        assert len(self.system.playing_sounds) == 0

    def test_update_with_inactive_emitters(self) -> None:
        """Test update with inactive sound emitters."""
        inactive_emitter = SoundEmitter("fire_ambient", active=False)
        actors = [self.create_mock_actor(5, 5, [inactive_emitter])]

        self.system.update(0, 0, create_spatial_index(actors), dt(0.1))

        assert self.system.current_time == 0.1
        assert len(self.system.playing_sounds) == 0

    def test_update_with_active_emitters_in_range(self) -> None:
        """Test update with active emitters in range."""
        active_emitter = SoundEmitter("fire_ambient", active=True)
        actors = [
            self.create_mock_actor(2, 2, [active_emitter])
        ]  # Distance ~2.8 from origin

        self.system.update(0, 0, create_spatial_index(actors), dt(0.1))

        assert self.system.current_time == 0.1
        # Should have created a playing sound for the looping layer
        assert len(active_emitter.playing_instances) > 0

    def test_update_with_emitters_out_of_range(self) -> None:
        """Test update with emitters beyond max distance."""
        distant_emitter = SoundEmitter("fire_ambient", active=True)
        actors = [
            self.create_mock_actor(50, 50, [distant_emitter])
        ]  # Far beyond max_distance

        self.system.update(0, 0, create_spatial_index(actors), dt(0.1))

        assert self.system.current_time == 0.1
        assert len(distant_emitter.playing_instances) == 0

    def test_update_with_multiple_actors_and_emitters(self) -> None:
        """Test update with multiple actors having multiple emitters."""
        emitter1 = SoundEmitter("fire_ambient", active=True)
        emitter2 = SoundEmitter("waterfall_ambient", active=True)
        emitter3 = SoundEmitter("engine_loop", active=False)  # Inactive

        actors = [
            self.create_mock_actor(2, 2, [emitter1, emitter2]),
            self.create_mock_actor(3, 3, [emitter3]),
            self.create_mock_actor(
                100, 100, [SoundEmitter("steam_hiss", active=True)]
            ),  # Out of range
        ]

        self.system.update(0, 0, create_spatial_index(actors), dt(0.1))

        # Should process emitter1 and emitter2, but not emitter3 (inactive)
        # or distant emitter
        assert len(emitter1.playing_instances) > 0
        assert len(emitter2.playing_instances) > 0
        assert len(emitter3.playing_instances) == 0

    def test_update_time_tracking(self) -> None:
        """Test that update correctly tracks time."""
        self.system.update(0, 0, create_spatial_index([]), dt(0.1))
        assert self.system.current_time == 0.1

        self.system.update(0, 0, create_spatial_index([]), dt(0.05))
        assert abs(self.system.current_time - 0.15) < 1e-10

        self.system.update(0, 0, create_spatial_index([]), dt(1.0))
        assert abs(self.system.current_time - 1.15) < 1e-10


class TestSoundSystemSoundPlayback:
    """Test SoundSystem sound playback logic."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.system = SoundSystem()
        self.emitter = SoundEmitter("fire_ambient", active=True)

    def test_ensure_sound_playing_loop_sound(self) -> None:
        """Test ensuring a looping sound is playing."""
        fire_def = get_sound_definition("fire_ambient")
        assert fire_def is not None

        # Should create playing instances for looping layers
        self.system._ensure_sound_playing(self.emitter, fire_def, 0.5)

        # Should have created instances for the layers
        assert len(self.emitter.playing_instances) > 0

        # Check that we have the loop layer
        loop_instances = [p for p in self.emitter.playing_instances if p.layer.loop]
        assert len(loop_instances) > 0

    def test_ensure_sound_playing_idempotent(self) -> None:
        """Test that ensuring sound playing multiple times doesn't create duplicates."""
        fire_def = get_sound_definition("fire_ambient")
        assert fire_def is not None

        # Call multiple times
        self.system._ensure_sound_playing(self.emitter, fire_def, 0.5)
        initial_count = len(self.emitter.playing_instances)

        self.system._ensure_sound_playing(self.emitter, fire_def, 0.5)
        self.system._ensure_sound_playing(self.emitter, fire_def, 0.5)

        # Should not create additional instances
        assert len(self.emitter.playing_instances) == initial_count

    def test_stop_sound_if_playing(self) -> None:
        """Test stopping sounds for an emitter."""
        fire_def = get_sound_definition("fire_ambient")
        assert fire_def is not None

        # Start some sounds
        self.system._ensure_sound_playing(self.emitter, fire_def, 0.5)
        assert len(self.emitter.playing_instances) > 0

        # Stop the sounds
        self.system._stop_sound_if_playing(self.emitter)
        assert len(self.emitter.playing_instances) == 0

    def test_stop_sound_if_playing_no_sounds(self) -> None:
        """Test stopping sounds when no sounds are playing."""
        # Should not crash when no sounds are playing
        self.system._stop_sound_if_playing(self.emitter)
        assert len(self.emitter.playing_instances) == 0

    def test_interval_sound_scheduling(self) -> None:
        """Test that interval-based sounds are scheduled properly."""
        fire_def = get_sound_definition("fire_ambient")
        assert fire_def is not None

        self.system._ensure_sound_playing(self.emitter, fire_def, 0.5)

        # Check for interval-based instances
        interval_instances = [
            p
            for p in self.emitter.playing_instances
            if not p.layer.loop and p.layer.interval
        ]
        assert len(interval_instances) > 0

        # Should have scheduled next trigger time
        for instance in interval_instances:
            assert instance.next_trigger_time > self.system.current_time


class TestSoundSystemIntegration:
    """Test SoundSystem integration scenarios."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.system = SoundSystem()

    def create_mock_actor(
        self, x: int, y: int, emitters: list[SoundEmitter] | None = None
    ) -> Actor:
        """Create a mock actor with sound emitters."""
        actor = MagicMock(spec=Actor)
        actor.x = x
        actor.y = y
        actor.sound_emitters = emitters
        return actor

    def test_full_update_cycle(self) -> None:
        """Test a complete update cycle with various scenarios."""
        # Create actors with different scenarios
        close_emitter = SoundEmitter("fire_ambient", active=True)
        distant_emitter = SoundEmitter("waterfall_ambient", active=True)
        inactive_emitter = SoundEmitter("engine_loop", active=False)

        actors = [
            self.create_mock_actor(2, 2, [close_emitter]),  # In range
            self.create_mock_actor(50, 50, [distant_emitter]),  # Out of range
            self.create_mock_actor(1, 1, [inactive_emitter]),  # Inactive
        ]

        # First update - sounds should start
        self.system.update(0, 0, create_spatial_index(actors), dt(0.1))

        assert len(close_emitter.playing_instances) > 0
        assert len(distant_emitter.playing_instances) == 0  # Out of range
        assert len(inactive_emitter.playing_instances) == 0  # Inactive

        # Move distant actor closer
        actors[1].x, actors[1].y = 3, 3

        # Second update - distant actor should now have sounds
        self.system.update(0, 0, create_spatial_index(actors), dt(0.1))

        assert len(close_emitter.playing_instances) > 0
        assert len(distant_emitter.playing_instances) > 0  # Now in range
        assert len(inactive_emitter.playing_instances) == 0  # Still inactive

    def test_emitter_activation_deactivation(self) -> None:
        """Test emitter activation and deactivation during updates."""
        emitter = SoundEmitter("fire_ambient", active=True)
        actors = [self.create_mock_actor(2, 2, [emitter])]

        # Start with active emitter
        self.system.update(0, 0, create_spatial_index(actors), dt(0.1))
        assert len(emitter.playing_instances) > 0

        # Deactivate emitter
        emitter.set_active(False)
        self.system.update(0, 0, create_spatial_index(actors), dt(0.1))
        assert len(emitter.playing_instances) == 0

        # Reactivate emitter
        emitter.set_active(True)
        self.system.update(0, 0, create_spatial_index(actors), dt(0.1))
        assert len(emitter.playing_instances) > 0

    def test_volume_changes_with_distance(self) -> None:
        """Test that volume changes are properly calculated as actors move."""
        emitter = SoundEmitter("fire_ambient", active=True, volume_multiplier=1.0)
        actor = self.create_mock_actor(2, 2, [emitter])

        # Position actor at different distances and check volume calculations
        # fire_ambient has: base_volume=0.7, falloff_start=2.8, rolloff_factor=0.7
        positions_and_expected_volumes = [
            (0, 0, 0.7),  # At origin, within falloff_start, should get base_volume
            (1, 1, 0.7),  # Still within falloff_start (distance ~1.41 < 2.8)
            (
                3,
                4,
                # Distance = 5, falloff_distance = 5 - 2.8 = 2.2
                # volume_factor = 2.8 / (2.8 + 0.7 * 2.2) = 2.8 / 4.34
                0.7 * (2.8 / (2.8 + 0.7 * 2.2)),
            ),
        ]

        for x, y, expected_volume in positions_and_expected_volumes:
            actor.x, actor.y = x, y

            # Calculate what the volume should be
            fire_def = get_sound_definition("fire_ambient")
            assert fire_def is not None

            # Set listener at origin for test
            self.system.audio_listener_x = 0.0
            self.system.audio_listener_y = 0.0
            calculated_volume = self.system._calculate_volume(
                x, y, fire_def, emitter.volume_multiplier
            )

            assert abs(calculated_volume - expected_volume) < 1e-10

    def test_multiple_emitters_same_sound(self) -> None:
        """Test multiple emitters with the same sound definition."""
        emitter1 = SoundEmitter("fire_ambient", active=True)
        emitter2 = SoundEmitter("fire_ambient", active=True)

        actors = [
            self.create_mock_actor(2, 2, [emitter1]),
            self.create_mock_actor(3, 3, [emitter2]),
        ]

        self.system.update(0, 0, create_spatial_index(actors), dt(0.1))

        # Both emitters should have their own playing instances
        assert len(emitter1.playing_instances) > 0
        assert len(emitter2.playing_instances) > 0

        # Playing instances should be different objects
        assert emitter1.playing_instances != emitter2.playing_instances


class TestSoundSystemAudioStopping:
    """Test that audio channels are properly stopped when sounds should be silent."""

    def setup_method(self) -> None:
        """Set up test fixtures with mock audio backend."""
        self.system = SoundSystem()
        self.mock_backend = MockAudioBackend()
        self.mock_backend.initialize()
        self.system.set_audio_backend(self.mock_backend)

        # Don't pre-initialize audio listener - let the system do it properly
        # during the first update() call which sets it to the actual listener position

    def create_mock_actor(
        self, x: int, y: int, emitters: list[SoundEmitter] | None = None
    ) -> Actor:
        """Create a mock actor with sound emitters."""
        actor = MagicMock(spec=Actor)
        actor.x = x
        actor.y = y
        actor.sound_emitters = emitters or []
        return actor

    def test_distance_based_stopping(self) -> None:
        """Test that sounds stop when emitter moves beyond max distance."""
        emitter = SoundEmitter("fire_ambient", active=True)
        actor = self.create_mock_actor(5, 5, [emitter])  # Start in range

        # First update - sound should start playing
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # Verify sound is playing
        assert len(emitter.playing_instances) > 0
        assert self.mock_backend.get_active_channel_count() > 0

        # Move actor far beyond max_distance (fire_ambient max_distance = 17.0)
        actor.x, actor.y = 25, 25  # Distance ~35, well beyond max_distance

        # Update - sound should stop
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # Verify ALL channels are stopped
        assert self.mock_backend.get_active_channel_count() == 0, (
            "Expected no active channels when emitter is beyond max distance"
        )

        # Verify emitter has no playing instances
        assert len(emitter.playing_instances) == 0, (
            "Expected no playing instances when emitter is beyond max distance"
        )

    def test_distance_based_restart(self) -> None:
        """Test that sounds restart when emitter moves back into range."""
        emitter = SoundEmitter("fire_ambient", active=True)
        actor = self.create_mock_actor(25, 25, [emitter])  # Start out of range

        # First update - no sound should play
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))
        assert self.mock_backend.get_active_channel_count() == 0
        assert len(emitter.playing_instances) == 0

        # Move actor back into range
        actor.x, actor.y = 5, 5

        # Update - sound should start
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # Verify sound is now playing
        assert self.mock_backend.get_active_channel_count() > 0, (
            "Expected active channels when emitter moves back into range"
        )
        assert len(emitter.playing_instances) > 0, (
            "Expected playing instances when emitter moves back into range"
        )

    def test_max_distance_boundary(self) -> None:
        """Test behavior exactly at max distance boundary."""
        emitter = SoundEmitter("fire_ambient", active=True)

        # fire_ambient has max_distance = 17.0
        # Position exactly at max distance
        actor = self.create_mock_actor(17, 0, [emitter])  # Distance = 17.0

        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # At exactly max_distance, volume should be > 0 so sound should play
        assert self.mock_backend.get_active_channel_count() > 0, (
            "Expected sound to play at exactly max_distance"
        )

        # Move just beyond max distance
        actor.x = 17.1  # Distance = 17.1, just beyond max_distance

        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # Beyond max_distance, all channels should stop
        assert self.mock_backend.get_active_channel_count() == 0, (
            "Expected no active channels just beyond max_distance"
        )

    def test_volume_based_stopping(self) -> None:
        """Test that sounds stop when volume multiplier is set to 0."""
        emitter = SoundEmitter("fire_ambient", active=True, volume_multiplier=1.0)
        actor = self.create_mock_actor(5, 5, [emitter])

        # Start with normal volume
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))
        assert self.mock_backend.get_active_channel_count() > 0
        assert len(emitter.playing_instances) > 0

        # Set volume multiplier to 0
        emitter.volume_multiplier = 0.0

        # Update - should stop all channels
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # Verify ALL channels are stopped
        assert self.mock_backend.get_active_channel_count() == 0, (
            "Expected no active channels when volume multiplier is 0.0"
        )
        assert len(emitter.playing_instances) == 0, (
            "Expected no playing instances when volume multiplier is 0.0"
        )

    def test_volume_based_restart(self) -> None:
        """Test that sounds restart when volume multiplier is restored."""
        emitter = SoundEmitter("fire_ambient", active=True, volume_multiplier=0.0)
        actor = self.create_mock_actor(5, 5, [emitter])

        # Start with zero volume - no sound should play
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))
        assert self.mock_backend.get_active_channel_count() == 0
        assert len(emitter.playing_instances) == 0

        # Restore volume
        emitter.volume_multiplier = 1.0

        # Update - sound should start playing
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # Verify sound is now playing
        assert self.mock_backend.get_active_channel_count() > 0, (
            "Expected active channels when volume multiplier is restored"
        )
        assert len(emitter.playing_instances) > 0, (
            "Expected playing instances when volume multiplier is restored"
        )

    def test_emitter_deactivation_stopping(self) -> None:
        """Test that sounds stop when emitter is deactivated."""
        emitter = SoundEmitter("fire_ambient", active=True)
        actor = self.create_mock_actor(5, 5, [emitter])

        # Start with active emitter
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))
        assert self.mock_backend.get_active_channel_count() > 0
        assert len(emitter.playing_instances) > 0

        # Deactivate emitter
        emitter.set_active(False)

        # Update - should stop all channels
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # Verify ALL channels are stopped
        assert self.mock_backend.get_active_channel_count() == 0, (
            "Expected no active channels when emitter is deactivated"
        )
        assert len(emitter.playing_instances) == 0, (
            "Expected no playing instances when emitter is deactivated"
        )

    def test_emitter_reactivation_restart(self) -> None:
        """Test that sounds restart when emitter is reactivated."""
        emitter = SoundEmitter("fire_ambient", active=False)
        actor = self.create_mock_actor(5, 5, [emitter])

        # Start with inactive emitter - no sound should play
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))
        assert self.mock_backend.get_active_channel_count() == 0
        assert len(emitter.playing_instances) == 0

        # Reactivate emitter
        emitter.set_active(True)

        # Update - sound should start playing
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # Verify sound is now playing
        assert self.mock_backend.get_active_channel_count() > 0, (
            "Expected active channels when emitter is reactivated"
        )
        assert len(emitter.playing_instances) > 0, (
            "Expected playing instances when emitter is reactivated"
        )

    def test_multi_layer_sound_stopping(self) -> None:
        """Test that all layers of complex sounds (like fire_ambient) stop together."""
        emitter = SoundEmitter("fire_ambient", active=True)
        actor = self.create_mock_actor(5, 5, [emitter])

        # Start playing multi-layer fire sound
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # fire_ambient has multiple layers (1 loop + 3 interval layers)
        initial_channels = self.mock_backend.get_active_channel_count()
        initial_instances = len(emitter.playing_instances)

        assert initial_channels > 0, "Expected fire_ambient to use multiple channels"
        assert initial_instances > 1, "Expected fire_ambient to have multiple layers"

        # Move emitter far away to stop all layers
        actor.x, actor.y = 30, 30  # Way beyond max_distance

        # Update - ALL layers should stop
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # Verify ALL channels are stopped (no ghost layers)
        assert self.mock_backend.get_active_channel_count() == 0, (
            f"Expected all {initial_channels} channels stopped, but "
            f"{self.mock_backend.get_active_channel_count()} remain active"
        )

        # Verify ALL playing instances are removed
        assert len(emitter.playing_instances) == 0, (
            f"Expected all {initial_instances} instances removed, but "
            f"{len(emitter.playing_instances)} remain"
        )

    def test_multi_layer_partial_stopping_bug_prevention(self) -> None:
        """Test prevention of partial layer stopping (the original campfire bug)."""
        emitter = SoundEmitter("fire_ambient", active=True)
        actor = self.create_mock_actor(5, 5, [emitter])

        # Start playing
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))
        initial_channels = self.mock_backend.get_active_channel_count()
        initial_instances = len(emitter.playing_instances)

        assert initial_channels > 0
        assert initial_instances > 0

        # Simulate the exact campfire bug scenario:
        # Move from distance 10 to distance 28 (beyond max_distance of 17)
        actor.x, actor.y = 10, 10  # Distance ~14, should still play
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # Should still be playing at distance 14
        assert self.mock_backend.get_active_channel_count() > 0, (
            "Sound should still play at distance 14 (within max_distance 17)"
        )

        # Now move to the problematic distance from the original bug
        actor.x, actor.y = 28, 0  # Distance 28, well beyond max_distance 17
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # This is the critical test - NO channels should remain active
        active_channels = self.mock_backend.get_active_channel_count()
        assert active_channels == 0, (
            f"CAMPFIRE BUG DETECTED: {active_channels} channels still playing "
            f"at distance 28 (beyond max_distance 17). This indicates ghost audio!"
        )

        # Double-check at the emitter level
        playing_instances = len(emitter.playing_instances)
        assert playing_instances == 0, (
            f"CAMPFIRE BUG DETECTED: {playing_instances} instances still tracked "
            f"when emitter is beyond max_distance. This indicates incomplete cleanup!"
        )

    def test_individual_layer_channel_tracking(self) -> None:
        """Test that we can track individual layer channels for debugging."""
        emitter = SoundEmitter("fire_ambient", active=True)
        actor = self.create_mock_actor(5, 5, [emitter])

        # Start playing
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # Collect all channels that are playing
        playing_channels = [
            channel for channel in self.mock_backend._channels if channel.is_playing()
        ]

        assert len(playing_channels) > 0, "Expected some channels to be playing"

        # Move out of range
        actor.x, actor.y = 30, 30
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # Verify each individual channel stopped
        for i, channel in enumerate(playing_channels):
            assert not channel.is_playing(), (
                f"Channel {i} is still playing when it should have stopped"
            )

        # Final verification - no channels should be playing
        still_playing = [
            i for i, ch in enumerate(self.mock_backend._channels) if ch.is_playing()
        ]
        assert len(still_playing) == 0, (
            f"Channels {still_playing} are still playing when all should be stopped"
        )

    def test_system_level_channel_count_consistency(self) -> None:
        """Test that system-level channel counts remain consistent."""
        emitters = [
            SoundEmitter("fire_ambient", active=True),
            SoundEmitter("waterfall_ambient", active=True),
            SoundEmitter("engine_loop", active=True),
        ]
        actors = [
            self.create_mock_actor(5, 5, [emitters[0]]),
            self.create_mock_actor(6, 6, [emitters[1]]),
            self.create_mock_actor(7, 7, [emitters[2]]),
        ]

        # Start all sounds
        self.system.update(0, 0, create_spatial_index(actors), dt(0.1))

        initial_backend_count = self.mock_backend.get_active_channel_count()
        initial_system_count = len(self.system.playing_sounds)

        assert initial_backend_count > 0, "Expected active channels"
        assert initial_system_count > 0, "Expected playing sounds tracked by system"

        # Note: Backend tracks actively playing channels, system tracks all instances
        # System may track more instances than backend has active channels because:
        # 1. Interval-based sounds create instances but don't play until trigger time
        # 2. System tracks scheduled future sounds, backend only tracks playing
        # The important test is that when we stop everything, both should be zero

        # Move all actors out of range
        for actor in actors:
            actor.x, actor.y = 50, 50

        self.system.update(0, 0, create_spatial_index(actors), dt(0.1))

        final_backend_count = self.mock_backend.get_active_channel_count()
        final_system_count = len(self.system.playing_sounds)

        # Both counts should be zero - this is the critical test
        assert final_backend_count == 0, (
            f"Backend should have 0 active channels, got {final_backend_count}"
        )
        assert final_system_count == 0, (
            f"System should track 0 playing sounds, got {final_system_count}"
        )

    def test_channel_count_after_multiple_start_stop_cycles(self) -> None:
        """Test channel counts remain correct after multiple start/stop cycles."""
        emitter = SoundEmitter("fire_ambient", active=True)
        actor = self.create_mock_actor(5, 5, [emitter])

        for cycle in range(3):
            # Start playing
            self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

            active_count = self.mock_backend.get_active_channel_count()
            assert active_count > 0, (
                f"Cycle {cycle}: Expected active channels after start"
            )

            # Stop by moving out of range
            actor.x, actor.y = 50, 50
            self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

            active_count = self.mock_backend.get_active_channel_count()
            assert active_count == 0, (
                f"Cycle {cycle}: Expected no active channels after stop, "
                f"got {active_count}"
            )

            # Move back for next cycle
            actor.x, actor.y = 5, 5

    def test_channel_limit_enforcement(self) -> None:
        """Test that we don't exceed the maximum channel limit."""
        # Create many emitters to potentially exceed channel limit
        max_channels = self.mock_backend._max_channels
        emitters = []
        actors = []

        # Create more emitters than channels available
        for i in range(max_channels + 5):
            emitter = SoundEmitter("fire_ambient", active=True)
            actor = self.create_mock_actor(i, 0, [emitter])  # Space them out
            emitters.append(emitter)
            actors.append(actor)

        # Update - should not exceed max channels
        self.system.update(0, 0, create_spatial_index(actors), dt(0.1))

        active_count = self.mock_backend.get_active_channel_count()
        assert active_count <= max_channels, (
            f"Active channel count {active_count} exceeds maximum {max_channels}"
        )

        # Stop all sounds
        for actor in actors:
            actor.x, actor.y = 100, 100  # Move far away

        self.system.update(0, 0, create_spatial_index(actors), dt(0.1))

        # Should be back to zero
        final_count = self.mock_backend.get_active_channel_count()
        assert final_count == 0, (
            f"Expected 0 active channels after stopping all, got {final_count}"
        )


class TestStaleEmitterPosition:
    """Test handling of emitters that move out of range during playback."""

    def setup_method(self) -> None:
        """Set up test fixtures with mock audio backend."""
        self.system = SoundSystem()
        self.mock_backend = MockAudioBackend()
        self.mock_backend.initialize()
        self.system.set_audio_backend(self.mock_backend)

    def create_mock_actor(
        self, x: int, y: int, emitters: list[SoundEmitter] | None = None
    ) -> Actor:
        """Create a mock actor with sound emitters."""
        actor = MagicMock(spec=Actor)
        actor.x = x
        actor.y = y
        actor.sound_emitters = emitters or []
        return actor

    def test_emitter_moves_out_of_range_during_playback(self) -> None:
        """Test bug #3: verify volume calculation when emitter goes out of range.

        This tests the scenario where:
        1. An emitter starts playing while in range
        2. The emitter moves out of range (beyond nearby_actors radius)
        3. The volume update loop tries to find the emitter position but fails
        4. Current behavior: uses stale (0, 0) position for volume calculation
        5. Expected behavior: sound should be stopped or volume should be 0
        """
        emitter = SoundEmitter("fire_ambient", active=True)
        actor = self.create_mock_actor(5, 5, [emitter])

        # Start with emitter in range
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # Verify sound is playing
        assert len(emitter.playing_instances) > 0
        initial_channel_count = self.mock_backend.get_active_channel_count()
        assert initial_channel_count > 0

        # Now simulate the emitter going out of the nearby_actors range
        # but still having playing instances (the problematic scenario from bug #3)
        # We do this by moving the actor far away but keeping the playing_sounds
        # This simulates the window between when an emitter goes out of range
        # and when the cleanup happens

        # Move actor far away (beyond max_distance of 17 for fire_ambient)
        actor.x, actor.y = 50, 50

        # Update - this should stop the sound
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # The sound should be stopped when emitter is out of range
        # This test verifies that we don't use stale (0, 0) coordinates
        final_channel_count = self.mock_backend.get_active_channel_count()
        assert final_channel_count == 0, (
            f"Expected all channels stopped when emitter moves out of range, "
            f"but {final_channel_count} channels still playing. "
            f"This could indicate bug #3: volume calculated from stale position (0,0)"
        )

        # Also verify at the emitter level
        assert len(emitter.playing_instances) == 0, (
            "Expected no playing instances when emitter is out of range"
        )

    def test_volume_not_calculated_from_origin_for_distant_emitter(self) -> None:
        """Test that distant emitters don't get their volume calculated from (0,0).

        This is a more direct test of bug #3: if an emitter is not found in
        nearby_actors during the volume update loop, we should not calculate
        volume using the fallback (0, 0) position.
        """
        emitter = SoundEmitter("fire_ambient", active=True)
        # Place emitter at a position where (0,0) would give different volume
        # than actual position
        actor = self.create_mock_actor(10, 10, [emitter])

        # Start playing
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))
        assert self.mock_backend.get_active_channel_count() > 0

        # Now update with listener at a different position where:
        # - Distance from listener to (10, 10) is large (sound should be quiet/stopped)
        # - Distance from listener to (0, 0) would be small (sound would be loud)
        # If bug #3 exists, volume would be calculated from (0,0) incorrectly

        # Move listener to (30, 30) - far from both actor and origin
        self.system.update(30, 30, create_spatial_index([actor]), dt(0.1))

        # Sound should be stopped or very quiet (emitter is ~28 tiles away)
        # fire_ambient max_distance is 17, so it should be silent
        final_count = self.mock_backend.get_active_channel_count()
        assert final_count == 0, (
            f"Emitter at distance ~28 should be silent (max_distance=17), "
            f"but {final_count} channels playing. "
            f"Bug #3 might be causing volume calculation from wrong position."
        )

    def test_emitter_position_lookup_uses_dict_not_origin_fallback(self) -> None:
        """Test that emitter positions are looked up from dict, not defaulting to (0,0).

        Previously, the volume update loop would default to (0, 0) if an emitter's
        actor wasn't found in nearby_actors. The fix stores emitter positions in
        a dict during collection and looks them up from there.

        This test verifies that when an emitter goes out of range, we don't
        incorrectly calculate volume from (0, 0).
        """
        emitter = SoundEmitter("fire_ambient", active=True)
        # Place emitter at (15, 0) - within range of listener at origin
        actor = self.create_mock_actor(15, 0, [emitter])

        # Start with listener at origin - emitter is 15 tiles away (within max 17)
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))
        assert self.mock_backend.get_active_channel_count() > 0, (
            "Sound should be playing when emitter is within range"
        )

        # Move emitter to (50, 0) - way out of range
        actor.x = 50

        # Previously, the volume update would:
        # 1. Fail to find emitter in nearby_actors
        # 2. Default to emitter_x, emitter_y = 0, 0
        # 3. Calculate volume from origin (distance 0 = full volume)
        # 4. Sound would continue playing loudly despite emitter being far away

        # With the fix, the emitter position is looked up from a dict
        # If not found (emitter out of range), volume update is skipped
        self.system.update(0, 0, create_spatial_index([actor]), dt(0.1))

        # Sound should be stopped (emitter at distance 50, beyond max 17)
        final_count = self.mock_backend.get_active_channel_count()
        assert final_count == 0, (
            f"Sound position fallback regression: emitter at distance 50 should be "
            f"silent, but {final_count} channels still playing. Volume may have been "
            f"calculated from fallback position (0, 0) instead of actual position."
        )


class TestListenerInterpolation:
    """Test the improved listener interpolation system."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.system = SoundSystem()
        self.mock_backend = MockAudioBackend()
        self.mock_backend.initialize()
        self.system.set_audio_backend(self.mock_backend)

        # Create test emitter
        self.emitter = SoundEmitter("fire_ambient", active=True)
        self.actor = MagicMock(spec=Actor)
        self.actor.x = 5
        self.actor.y = 5
        self.actor.sound_emitters = [self.emitter]

    def test_teleportation_detection(self) -> None:
        """Test that teleportation is detected and causes instant listener update."""
        # Initialize listener at origin
        self.system.update(0, 0, create_spatial_index([self.actor]), dt(0.1))

        assert self.system.audio_listener_x == 0.0
        assert self.system.audio_listener_y == 0.0

        # Teleport beyond threshold (default 10.0)
        new_x, new_y = 20, 20  # Distance ~28, beyond threshold
        self.system.update(new_x, new_y, create_spatial_index([self.actor]), dt(0.1))

        # Should instantly update to new position
        assert self.system.audio_listener_x == new_x
        assert self.system.audio_listener_y == new_y

    def test_smooth_interpolation_normal_movement(self) -> None:
        """Test smooth interpolation for normal movement."""
        # Initialize listener at origin
        self.system.update(0, 0, create_spatial_index([self.actor]), dt(0.1))

        # Move within teleport threshold
        target_x, target_y = 3, 4  # Distance 5, within threshold of 10
        self.system.update(
            target_x, target_y, create_spatial_index([self.actor]), dt(0.1)
        )

        # Should not instantly reach target (interpolating)
        assert self.system.audio_listener_x != target_x
        assert self.system.audio_listener_y != target_y

        # But should have moved toward target
        assert self.system.audio_listener_x > 0.0
        assert self.system.audio_listener_y > 0.0

        # After multiple updates, should converge to target
        for _ in range(10):
            self.system.update(
                target_x, target_y, create_spatial_index([self.actor]), dt(0.1)
            )

        # Should be very close to target now
        assert abs(self.system.audio_listener_x - target_x) < 0.02
        assert abs(self.system.audio_listener_y - target_y) < 0.02

    def test_interpolation_speed_based_on_delta_time(self) -> None:
        """Test that interpolation speed adapts to frame rate."""
        # Initialize listener
        self.system.update(0, 0, create_spatial_index([self.actor]), dt(0.1))

        # Test with small delta_time (high frame rate)
        target_x, target_y = 5, 0
        self.system.update(
            target_x, target_y, create_spatial_index([self.actor]), dt(0.016)
        )  # ~60 FPS

        movement_small_dt = self.system.audio_listener_x

        # Reset
        self.system.audio_listener_x = 0.0
        self.system.audio_listener_y = 0.0
        self.system.previous_listener_x = 0.0
        self.system.previous_listener_y = 0.0

        # Test with larger delta_time (low frame rate)
        self.system.update(
            target_x, target_y, create_spatial_index([self.actor]), dt(0.1)
        )  # ~10 FPS

        movement_large_dt = self.system.audio_listener_x

        # Larger delta_time should result in more movement
        assert movement_large_dt > movement_small_dt

    def test_teleport_threshold_boundary(self) -> None:
        """Test behavior exactly at teleport threshold."""
        # Initialize listener
        self.system.update(0, 0, create_spatial_index([self.actor]), dt(0.1))

        # Move exactly at threshold distance (10.0)
        target_x = 10
        target_y = 0
        self.system.update(
            target_x, target_y, create_spatial_index([self.actor]), dt(0.1)
        )

        # At exactly threshold, should still interpolate (not teleport)
        assert self.system.audio_listener_x != target_x

        # Reset
        self.system.audio_listener_x = 0.0
        self.system.audio_listener_y = 0.0
        self.system.previous_listener_x = 0.0
        self.system.previous_listener_y = 0.0

        # Move just beyond threshold
        target_x = 11  # Changed from 10.1 to int
        self.system.update(
            target_x, target_y, create_spatial_index([self.actor]), dt(0.1)
        )

        # Should instantly teleport
        assert self.system.audio_listener_x == target_x

    def test_previous_position_tracking(self) -> None:
        """Test that previous position is correctly tracked."""
        # Initialize
        self.system.update(5, 5, create_spatial_index([self.actor]), dt(0.1))

        assert self.system.previous_listener_x == 5.0
        assert self.system.previous_listener_y == 5.0

        # Update to new position
        self.system.update(7, 8, create_spatial_index([self.actor]), dt(0.1))

        assert self.system.previous_listener_x == 7.0
        assert self.system.previous_listener_y == 8.0

    def test_interpolation_convergence(self) -> None:
        """Test that interpolation eventually converges to target."""
        # Initialize
        self.system.update(0, 0, create_spatial_index([self.actor]), dt(0.1))

        # Set target
        target_x, target_y = 2, 2

        # Update many times with same target
        for _ in range(50):
            self.system.update(
                target_x, target_y, create_spatial_index([self.actor]), dt(0.1)
            )

        # Should have converged very close to target
        assert abs(self.system.audio_listener_x - target_x) < 0.001
        assert abs(self.system.audio_listener_y - target_y) < 0.001

    def test_diagonal_teleportation(self) -> None:
        """Test teleportation detection with diagonal movement."""
        # Initialize
        self.system.update(0, 0, create_spatial_index([self.actor]), dt(0.1))

        # Diagonal teleport beyond threshold
        # Distance = sqrt(8^2 + 8^2) = ~11.3, beyond threshold of 10
        target_x, target_y = 8, 8
        self.system.update(
            target_x, target_y, create_spatial_index([self.actor]), dt(0.1)
        )

        # Should instantly teleport
        assert self.system.audio_listener_x == target_x
        assert self.system.audio_listener_y == target_y

    def test_rapid_direction_changes(self) -> None:
        """Test interpolation with rapid direction changes."""
        # Initialize
        self.system.update(0, 0, create_spatial_index([self.actor]), dt(0.1))

        # Move right
        self.system.update(2, 0, create_spatial_index([self.actor]), dt(0.1))
        x_after_right = self.system.audio_listener_x

        # Immediately move left
        self.system.update(-2, 0, create_spatial_index([self.actor]), dt(0.1))

        # Should start moving back toward left
        assert self.system.audio_listener_x < x_after_right

    def test_zero_delta_time(self) -> None:
        """Test that zero delta_time doesn't cause issues."""
        # Initialize
        self.system.update(0, 0, create_spatial_index([self.actor]), dt(0.1))

        # Update with zero delta_time
        self.system.update(5, 5, create_spatial_index([self.actor]), dt(0.0))

        # Should not move (or move very little)
        assert self.system.audio_listener_x < 0.1
        assert self.system.audio_listener_y < 0.1


class TestCampfireBugRegression:
    """Integration test that reproduces and prevents the campfire ghost audio bug."""

    def setup_method(self) -> None:
        """Set up test fixtures matching the original bug scenario."""
        self.system = SoundSystem()
        self.mock_backend = MockAudioBackend()
        self.mock_backend.initialize()
        self.system.set_audio_backend(self.mock_backend)

        # Disable audio listener interpolation for consistent testing
        # This ensures that when we move the player, the audio listener follows
        self.system.set_transition_duration(0.0)

        # Create campfire actor (mimics ContainedFire.create_campfire)
        self.campfire_emitter = SoundEmitter("fire_ambient", active=True)
        self.campfire_actor = MagicMock(spec=Actor)
        self.campfire_actor.sound_emitters = [self.campfire_emitter]

    def create_player_actor(self, x: int, y: int) -> Actor:
        """Create a mock player actor for position tracking."""
        player = MagicMock(spec=Actor)
        player.x = x
        player.y = y
        player.sound_emitters = []
        return player

    def test_campfire_bug_exact_reproduction(self) -> None:
        """Test the exact bug scenario: campfire at distance 28 should be silent."""
        # Place campfire at a fixed position
        self.campfire_actor.x = 10
        self.campfire_actor.y = 10

        # Start with player close to campfire (distance ~14, within max_distance 17)
        player = self.create_player_actor(0, 0)
        actors = [self.campfire_actor, player]

        # Update with player near campfire - sound should start
        self.system.update(0, 0, create_spatial_index(actors), dt(0.1))

        # Verify campfire is audible
        assert len(self.campfire_emitter.playing_instances) > 0, (
            "Campfire should be audible when player is close"
        )
        assert self.mock_backend.get_active_channel_count() > 0, (
            "Expected active audio channels when campfire is in range"
        )

        # Now move player to the problematic distance from the original bug report
        # "even when the player is way more than 17 tiles away from the campfire,
        # I still hear it at a fairly audible level"
        player.x = (
            38  # Distance from campfire (10,10) = sqrt((38-10)^2 + (0-10)^2) = ~30
        )
        player.y = 0

        # Update with player at distance ~30 (well beyond max_distance of 17)
        self.system.update(player.x, player.y, create_spatial_index(actors), dt(0.1))

        # THE CRITICAL TEST - this would fail with the original campfire bug
        active_channels = self.mock_backend.get_active_channel_count()
        assert active_channels == 0, (
            f"ORIGINAL CAMPFIRE BUG REPRODUCED: {active_channels} audio channels "
            f"still playing when campfire is at distance ~30 (beyond max_distance 17). "
            f"This indicates the Channel.stop() fix is not working!"
        )

        # Additional verification at emitter level
        playing_instances = len(self.campfire_emitter.playing_instances)
        assert playing_instances == 0, (
            f"ORIGINAL CAMPFIRE BUG REPRODUCED: {playing_instances} sound instances "
            f"still tracked when campfire should be silent. This indicates incomplete "
            f"sound cleanup in the SoundSystem!"
        )

    def test_campfire_bug_with_multiple_distances(self) -> None:
        """Test the campfire bug across multiple distance transitions."""
        # Place campfire at origin for easier distance calculations
        self.campfire_actor.x = 0
        self.campfire_actor.y = 0

        # Test distances: within range -> at boundary -> beyond range -> way beyond
        test_distances = [
            (5, True, "Close: should be audible"),
            (16, True, "Near max_distance: should still be audible"),
            (17, True, "At max_distance: should still be audible"),
            (18, False, "Just beyond max_distance: should be silent"),
            (28, False, "Far beyond (original bug distance): should be silent"),
            (50, False, "Very far: should be silent"),
        ]

        player = self.create_player_actor(0, 0)
        actors = [self.campfire_actor, player]

        for distance, should_be_audible, description in test_distances:
            # Position player at test distance
            player.x = distance
            player.y = 0

            # Update audio system
            self.system.update(
                player.x, player.y, create_spatial_index(actors), dt(0.1)
            )

            # Check results
            active_channels = self.mock_backend.get_active_channel_count()
            playing_instances = len(self.campfire_emitter.playing_instances)

            if should_be_audible:
                assert active_channels > 0, (
                    f"{description}: Expected audio channels but got {active_channels}"
                )
                assert playing_instances > 0, (
                    f"{description}: Expected playing instances but got "
                    f"{playing_instances}"
                )
            else:
                assert active_channels == 0, (
                    f"{description}: Expected silence but got {active_channels} "
                    f"active channels. CAMPFIRE BUG DETECTED at distance {distance}!"
                )
                assert playing_instances == 0, (
                    f"{description}: Expected no instances but got "
                    f"{playing_instances}. CAMPFIRE BUG DETECTED at {distance}!"
                )

    def test_campfire_bug_prevention_after_restart(self) -> None:
        """Test that restarting the game doesn't bring back ghost campfire audio."""
        # This tests the user's note: "this occurs when I restart the game"
        self.campfire_actor.x = 0
        self.campfire_actor.y = 0

        player = self.create_player_actor(30, 0)  # Start far from campfire
        actors = [self.campfire_actor, player]

        # Simulate first game session - player far from campfire
        self.system.update(player.x, player.y, create_spatial_index(actors), dt(0.1))

        # Should be silent
        assert self.mock_backend.get_active_channel_count() == 0, (
            "Campfire should be silent when player starts far away"
        )

        # Simulate "restart" by reinitializing the sound system
        self.system = SoundSystem()
        new_backend = MockAudioBackend()
        new_backend.initialize()
        self.system.set_audio_backend(new_backend)

        # Reset emitter state (simulates loading from save)
        self.campfire_emitter = SoundEmitter("fire_ambient", active=True)
        self.campfire_actor.sound_emitters = [self.campfire_emitter]

        # Update with player still far away
        self.system.update(player.x, player.y, create_spatial_index(actors), dt(0.1))

        # Should STILL be silent after restart
        assert new_backend.get_active_channel_count() == 0, (
            "CAMPFIRE BUG: After restart, campfire should remain silent when "
            "player is far away, but ghost audio was detected!"
        )

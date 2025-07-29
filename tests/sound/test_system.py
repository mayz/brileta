"""Tests for SoundSystem class."""

from unittest.mock import MagicMock

from catley.game.actors.core import Actor
from catley.sound.definitions import SoundDefinition, SoundLayer, get_sound_definition
from catley.sound.emitter import SoundEmitter
from catley.sound.system import SoundSystem


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

        # Create a test sound definition
        self.test_sound_def = SoundDefinition(
            sound_id="test_sound",
            layers=[SoundLayer(file="test.ogg")],
            base_volume=0.8,
            falloff_start=3.0,
            max_distance=10.0,
        )

    def test_calculate_volume_within_falloff_start(self) -> None:
        """Test volume calculation when within falloff start distance."""
        # At distance 0 (same position)
        volume = self.system._calculate_volume(5, 5, 5, 5, self.test_sound_def, 1.0)
        assert volume == 0.8  # base_volume

        # At distance 2 (within falloff_start of 3)
        volume = self.system._calculate_volume(5, 5, 7, 5, self.test_sound_def, 1.0)
        assert volume == 0.8  # base_volume

    def test_calculate_volume_at_falloff_start(self) -> None:
        """Test volume calculation exactly at falloff start distance."""
        # At distance 3 (exactly at falloff_start)
        volume = self.system._calculate_volume(5, 5, 8, 5, self.test_sound_def, 1.0)
        assert volume == 0.8  # base_volume

    def test_calculate_volume_beyond_falloff_start(self) -> None:
        """Test volume calculation beyond falloff start with inverse square law."""
        # At distance 4 (1 unit beyond falloff_start of 3)
        volume = self.system._calculate_volume(5, 5, 9, 5, self.test_sound_def, 1.0)
        expected = 0.8 * (
            1.0 / (1.0 + 1.0**2)
        )  # base_volume * 1/(1 + falloff_distance^2)
        assert abs(volume - expected) < 1e-10

    def test_calculate_volume_beyond_max_distance(self) -> None:
        """Test volume calculation beyond max distance."""
        # At distance 15 (beyond max_distance of 10)
        volume = self.system._calculate_volume(5, 5, 20, 5, self.test_sound_def, 1.0)
        assert volume == 0.0

    def test_calculate_volume_at_max_distance(self) -> None:
        """Test volume calculation exactly at max distance."""
        # At distance 10 (exactly at max_distance)
        volume = self.system._calculate_volume(5, 5, 15, 5, self.test_sound_def, 1.0)
        falloff_distance = 10.0 - 3.0  # distance - falloff_start
        expected = 0.8 * (1.0 / (1.0 + falloff_distance**2))
        assert abs(volume - expected) < 1e-10

    def test_calculate_volume_with_volume_multiplier(self) -> None:
        """Test volume calculation with volume multiplier."""
        # Within falloff start with multiplier
        volume = self.system._calculate_volume(5, 5, 5, 5, self.test_sound_def, 0.5)
        assert volume == 0.4  # base_volume * multiplier

        # Beyond falloff start with multiplier
        volume = self.system._calculate_volume(5, 5, 9, 5, self.test_sound_def, 0.5)
        falloff_factor = 1.0 / (1.0 + 1.0**2)
        expected = 0.8 * 0.5 * falloff_factor
        assert abs(volume - expected) < 1e-10

    def test_calculate_volume_diagonal_distances(self) -> None:
        """Test volume calculation with diagonal distances."""
        # Distance sqrt(3^2 + 4^2) = 5
        volume = self.system._calculate_volume(0, 0, 3, 4, self.test_sound_def, 1.0)
        falloff_distance = 5.0 - 3.0  # distance - falloff_start
        expected = 0.8 * (1.0 / (1.0 + falloff_distance**2))
        assert abs(volume - expected) < 1e-10

    def test_calculate_volume_edge_cases(self) -> None:
        """Test volume calculation edge cases."""
        # Zero volume multiplier
        volume = self.system._calculate_volume(5, 5, 5, 5, self.test_sound_def, 0.0)
        assert volume == 0.0

        # Very small distance beyond falloff_start
        volume = self.system._calculate_volume(5, 5, 9, 5, self.test_sound_def, 1.0)
        falloff_distance = 4.0 - 3.0
        expected = 0.8 * (1.0 / (1.0 + falloff_distance**2))
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
        self.system.update(0, 0, [], 0.1)

        assert self.system.current_time == 0.1
        assert len(self.system.playing_sounds) == 0

    def test_update_with_actors_no_emitters(self) -> None:
        """Test update with actors that have no sound emitters."""
        actors = [
            self.create_mock_actor(5, 5, None),
            self.create_mock_actor(10, 10, []),
        ]

        self.system.update(0, 0, actors, 0.1)

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

        self.system.update(0, 0, [actor], 0.1)

        assert self.system.current_time == 0.1
        assert len(self.system.playing_sounds) == 0

    def test_update_with_inactive_emitters(self) -> None:
        """Test update with inactive sound emitters."""
        inactive_emitter = SoundEmitter("fire_ambient", active=False)
        actors = [self.create_mock_actor(5, 5, [inactive_emitter])]

        self.system.update(0, 0, actors, 0.1)

        assert self.system.current_time == 0.1
        assert len(self.system.playing_sounds) == 0

    def test_update_with_active_emitters_in_range(self) -> None:
        """Test update with active emitters in range."""
        active_emitter = SoundEmitter("fire_ambient", active=True)
        actors = [
            self.create_mock_actor(2, 2, [active_emitter])
        ]  # Distance ~2.8 from origin

        self.system.update(0, 0, actors, 0.1)

        assert self.system.current_time == 0.1
        # Should have created a playing sound for the looping layer
        assert len(active_emitter.playing_instances) > 0

    def test_update_with_emitters_out_of_range(self) -> None:
        """Test update with emitters beyond max distance."""
        distant_emitter = SoundEmitter("fire_ambient", active=True)
        actors = [
            self.create_mock_actor(50, 50, [distant_emitter])
        ]  # Far beyond max_distance

        self.system.update(0, 0, actors, 0.1)

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

        self.system.update(0, 0, actors, 0.1)

        # Should process emitter1 and emitter2, but not emitter3 (inactive)
        # or distant emitter
        assert len(emitter1.playing_instances) > 0
        assert len(emitter2.playing_instances) > 0
        assert len(emitter3.playing_instances) == 0

    def test_update_time_tracking(self) -> None:
        """Test that update correctly tracks time."""
        self.system.update(0, 0, [], 0.1)
        assert self.system.current_time == 0.1

        self.system.update(0, 0, [], 0.05)
        assert abs(self.system.current_time - 0.15) < 1e-10

        self.system.update(0, 0, [], 1.0)
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
        self.system.update(0, 0, actors, 0.1)

        assert len(close_emitter.playing_instances) > 0
        assert len(distant_emitter.playing_instances) == 0  # Out of range
        assert len(inactive_emitter.playing_instances) == 0  # Inactive

        # Move distant actor closer
        actors[1].x, actors[1].y = 3, 3

        # Second update - distant actor should now have sounds
        self.system.update(0, 0, actors, 0.1)

        assert len(close_emitter.playing_instances) > 0
        assert len(distant_emitter.playing_instances) > 0  # Now in range
        assert len(inactive_emitter.playing_instances) == 0  # Still inactive

    def test_emitter_activation_deactivation(self) -> None:
        """Test emitter activation and deactivation during updates."""
        emitter = SoundEmitter("fire_ambient", active=True)
        actors = [self.create_mock_actor(2, 2, [emitter])]

        # Start with active emitter
        self.system.update(0, 0, actors, 0.1)
        assert len(emitter.playing_instances) > 0

        # Deactivate emitter
        emitter.set_active(False)
        self.system.update(0, 0, actors, 0.1)
        assert len(emitter.playing_instances) == 0

        # Reactivate emitter
        emitter.set_active(True)
        self.system.update(0, 0, actors, 0.1)
        assert len(emitter.playing_instances) > 0

    def test_volume_changes_with_distance(self) -> None:
        """Test that volume changes are properly calculated as actors move."""
        emitter = SoundEmitter("fire_ambient", active=True, volume_multiplier=1.0)
        actor = self.create_mock_actor(2, 2, [emitter])

        # Position actor at different distances and check volume calculations
        positions_and_expected_volumes = [
            (0, 0, 0.7),  # At origin, within falloff_start, should get base_volume
            (1, 1, 0.7),  # Still within falloff_start
            (3, 4, 0.7 * (1.0 / (1.0 + (5.0 - 2.0) ** 2))),  # Beyond falloff_start
        ]

        for x, y, expected_volume in positions_and_expected_volumes:
            actor.x, actor.y = x, y

            # Calculate what the volume should be
            fire_def = get_sound_definition("fire_ambient")
            assert fire_def is not None

            calculated_volume = self.system._calculate_volume(
                x, y, 0, 0, fire_def, emitter.volume_multiplier
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

        self.system.update(0, 0, actors, 0.1)

        # Both emitters should have their own playing instances
        assert len(emitter1.playing_instances) > 0
        assert len(emitter2.playing_instances) > 0

        # Playing instances should be different objects
        assert emitter1.playing_instances != emitter2.playing_instances

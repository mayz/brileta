import unittest
from unittest.mock import MagicMock, patch

from brileta.events import SoundEvent, publish_event, reset_event_bus_for_testing
from brileta.sound import system as sound_system
from brileta.sound.definitions import SOUND_DEFINITIONS, SoundDefinition, SoundLayer
from brileta.sound.system import SoundSystem


class TestOneShotSounds(unittest.TestCase):
    def setUp(self):
        reset_event_bus_for_testing()
        self.sound_system = SoundSystem()
        self.mock_backend = MagicMock()
        self.sound_system.set_audio_backend(self.mock_backend)

        # Mock sound definition
        self.test_sound_id = "test_shot"
        self.test_def = SoundDefinition(
            sound_id=self.test_sound_id,
            layers=[
                SoundLayer(file="bang.ogg", volume=1.0, loop=False),
                SoundLayer(file="echo.ogg", volume=0.5, loop=False),
            ],
            base_volume=1.0,
            max_distance=10.0,
        )
        # Patch the global definitions
        self.original_defs = SOUND_DEFINITIONS.copy()
        SOUND_DEFINITIONS[self.test_sound_id] = self.test_def

    def tearDown(self):
        SOUND_DEFINITIONS.clear()
        SOUND_DEFINITIONS.update(self.original_defs)

    def test_play_one_shot_direct(self):
        """Test calling play_one_shot directly."""
        # Set listener position
        self.sound_system.update(0, 0, MagicMock(), 0.1)

        # Play sound at listener position (full volume)
        self.sound_system.play_one_shot(self.test_sound_id, 0, 0)

        # Should play both layers
        self.assertEqual(self.mock_backend.play_sound.call_count, 2)

        # Check calls
        calls = self.mock_backend.play_sound.call_args_list
        # First layer (bang.ogg)
        self.assertEqual(calls[0].kwargs["volume"], 1.0)
        self.assertFalse(calls[0].kwargs["loop"])
        # Second layer (echo.ogg)
        self.assertEqual(calls[1].kwargs["volume"], 0.5)

    def test_play_one_shot_via_event(self):
        """Test triggering sound via SoundEvent."""
        # Set listener position
        self.sound_system.update(0, 0, MagicMock(), 0.1)

        # Publish event
        event = SoundEvent(sound_id=self.test_sound_id, x=0, y=0)
        publish_event(event)

        # Should play both layers
        self.assertEqual(self.mock_backend.play_sound.call_count, 2)

    def test_distance_attenuation(self):
        """Test that volume decreases with distance."""
        # Set listener at 0,0
        self.sound_system.update(0, 0, MagicMock(), 0.1)

        # Play sound at distance (5 units away)
        # Falloff start is 2.0 by default, max is 10.0
        self.sound_system.play_one_shot(self.test_sound_id, 5, 0)

        # Should play
        self.assertEqual(self.mock_backend.play_sound.call_count, 2)

        # Volume should be less than 1.0
        call_volume = self.mock_backend.play_sound.call_args_list[0].kwargs["volume"]
        self.assertLess(call_volume, 1.0)
        self.assertGreater(call_volume, 0.0)

    def test_out_of_range(self):
        """Test that sounds beyond max_distance are not played."""
        # Set listener at 0,0
        self.sound_system.update(0, 0, MagicMock(), 0.1)

        # Play sound at 15 units (max is 10)
        self.sound_system.play_one_shot(self.test_sound_id, 15, 0)

        # Should NOT play
        self.mock_backend.play_sound.assert_not_called()

    def test_variants(self):
        """Test that variants are chosen randomly."""
        # Update definition to use variants
        variant_def = SoundDefinition(
            sound_id="variant_test",
            layers=[
                SoundLayer(
                    file="default.ogg", variants=["var1.ogg", "var2.ogg"], loop=False
                )
            ],
        )
        SOUND_DEFINITIONS["variant_test"] = variant_def

        self.sound_system.update(0, 0, MagicMock(), 0.1)

        # Mock load_sound to return a dummy object so we can check what was passed
        self.mock_backend.load_sound.side_effect = lambda p: str(p)

        # Play multiple times
        played_files = set()
        for _ in range(10):
            self.sound_system.play_one_shot("variant_test", 0, 0)
            # Get the first arg of the last call to play_sound,
            # which is the loaded sound (path string)
            # Note: play_one_shot calls _load_sound which calls
            # backend.load_sound
            # Then it calls backend.play_sound(loaded_sound, ...)
            args, _ = self.mock_backend.play_sound.call_args
            played_files.add(str(args[0]))

        # We should see variants being played
        # Note: In a very unlucky run, we might only see one, but with 10 tries
        # and 2 variants, it's unlikely.
        # However, since we can't guarantee randomness in a test, we just check
        # that *something* was played.
        # A better test would mock random.choice.
        self.assertTrue(len(played_files) > 0)

        # Let's try mocking the RNG choice to be sure
        with patch.object(sound_system._rng, "choice", return_value="var1.ogg"):
            self.sound_system.play_one_shot("variant_test", 0, 0)
            args, _ = self.mock_backend.play_sound.call_args
            self.assertIn("var1.ogg", str(args[0]))


class TestParameterizedSounds(unittest.TestCase):
    """Tests for parameterized sound playback (repeat counts, after_layer, etc.)."""

    def setUp(self):
        reset_event_bus_for_testing()
        self.sound_system = SoundSystem()
        self.mock_backend = MagicMock()
        self.sound_system.set_audio_backend(self.mock_backend)

        # Set listener at origin
        self.sound_system.update(0, 0, MagicMock(), 0.1)

        # Create a parameterized sound definition (like shotgun reload)
        self.param_sound_id = "test_reload"
        self.param_sound_def = SoundDefinition(
            sound_id=self.param_sound_id,
            layers=[
                # Shell insert layer - repeat based on params
                SoundLayer(
                    file="shell_insert.ogg",
                    volume=0.9,
                    loop=False,
                    repeat_count_param="shell_count",
                    repeat_count=5,  # Fallback
                    repeat_delay=0.4,
                ),
                # Rack layer - plays after shells complete
                SoundLayer(
                    file="rack.ogg",
                    volume=0.9,
                    loop=False,
                    delay=0.1,
                    after_layer=0,
                ),
            ],
            base_volume=1.0,
            max_distance=10.0,
        )

        # Save and patch definitions
        self.original_defs = SOUND_DEFINITIONS.copy()
        SOUND_DEFINITIONS[self.param_sound_id] = self.param_sound_def

    def tearDown(self):
        SOUND_DEFINITIONS.clear()
        SOUND_DEFINITIONS.update(self.original_defs)

    def test_repeat_count_from_params(self):
        """Test that repeat count is read from params."""
        # Play with 3 shells
        self.sound_system.play_one_shot(
            self.param_sound_id, 0, 0, params={"shell_count": 3}
        )

        # First layer should be played immediately (delay=0)
        self.assertEqual(self.mock_backend.play_sound.call_count, 1)

        # Remaining 2 shell inserts + rack should be scheduled as delayed sounds
        # 3 shells total: 1 immediate + 2 delayed = 3 shell sounds
        # Plus 1 rack sound = 4 total scheduled events
        # Check delayed sounds queue
        delayed_count = len(self.sound_system._delayed_sounds)
        # 2 more shell inserts (at 0.4, 0.8) + 1 rack (at 0.8 + 0.1 = 0.9)
        self.assertEqual(delayed_count, 3)

    def test_repeat_count_fallback_when_param_missing(self):
        """Test fallback to repeat_count when param is missing."""
        # Play without params - should use repeat_count=5
        self.sound_system.play_one_shot(self.param_sound_id, 0, 0, params=None)

        # 1 immediate play + 4 delayed shell inserts + 1 delayed rack = 5 delayed
        delayed_count = len(self.sound_system._delayed_sounds)
        self.assertEqual(delayed_count, 5)  # 4 shells + 1 rack

    def test_after_layer_timing(self):
        """Test that after_layer calculates correct delay based on repeat count."""
        # With 3 shells:
        # - Shell 1 at 0.0, Shell 2 at 0.4, Shell 3 at 0.8
        # - Layer 0 completion time = (3-1) * 0.4 = 0.8
        # - Rack delay = layer_0_completion + 0.1 = 0.9
        self.sound_system.play_one_shot(
            self.param_sound_id, 0, 0, params={"shell_count": 3}
        )

        # Find the rack sound in delayed queue
        rack_events = [
            (t, e)
            for t, e in self.sound_system._delayed_sounds
            if e.layer == 1  # Rack is layer 1
        ]

        self.assertEqual(len(rack_events), 1)
        play_time, _ = rack_events[0]

        # Should be scheduled at current_time + 0.9
        expected_time = self.sound_system.current_time + 0.9
        self.assertAlmostEqual(play_time, expected_time, places=5)

    def test_single_shell_timing(self):
        """Test timing with a single shell (edge case)."""
        self.sound_system.play_one_shot(
            self.param_sound_id, 0, 0, params={"shell_count": 1}
        )

        # 1 shell plays immediately, rack is delayed
        self.assertEqual(self.mock_backend.play_sound.call_count, 1)

        # Only rack should be in delayed queue
        # Layer 0 completion = (1-1) * 0.4 = 0.0
        # Rack delay = 0.0 + 0.1 = 0.1
        delayed_count = len(self.sound_system._delayed_sounds)
        self.assertEqual(delayed_count, 1)

        rack_time, _ = self.sound_system._delayed_sounds[0]
        expected_time = self.sound_system.current_time + 0.1
        self.assertAlmostEqual(rack_time, expected_time, places=5)

    def test_params_passed_via_sound_event(self):
        """Test that params work when triggered via SoundEvent."""
        event = SoundEvent(
            sound_id=self.param_sound_id, x=0, y=0, params={"shell_count": 2}
        )
        publish_event(event)

        # 1 immediate + 1 delayed shell + 1 delayed rack = 2 delayed
        delayed_count = len(self.sound_system._delayed_sounds)
        self.assertEqual(delayed_count, 2)

    def test_zero_repeat_count_floors_to_one(self):
        """Test that repeat count of 0 is treated as 1."""
        self.sound_system.play_one_shot(
            self.param_sound_id, 0, 0, params={"shell_count": 0}
        )

        # Should play at least once (minimum 1)
        self.assertEqual(self.mock_backend.play_sound.call_count, 1)

    def test_delayed_sounds_process_correctly(self):
        """Test that delayed repeat sounds are processed correctly."""
        # current_time is 0.1 after setUp (from the update call)
        start_time = self.sound_system.current_time

        self.sound_system.play_one_shot(
            self.param_sound_id, 0, 0, params={"shell_count": 2}
        )

        # Initial: 1 immediate play (shell 1 at delay=0)
        initial_plays = self.mock_backend.play_sound.call_count
        self.assertEqual(initial_plays, 1)

        # With 2 shells:
        # - Shell 1: delay=0.0 (immediate)
        # - Shell 2: delay=0.4, scheduled at start_time + 0.4
        # - Rack: after_layer=0, delay=0.1, so delay = 0.4 + 0.1 = 0.5
        #         scheduled at start_time + 0.5

        # Advance time past shell 2 (at 0.4) but before rack (at 0.5)
        self.sound_system.current_time = start_time + 0.45
        self.sound_system._process_delayed_sounds()

        # Should have played only the second shell insert (not rack yet)
        self.assertEqual(self.mock_backend.play_sound.call_count, 2)

        # Advance time past rack sound (at 0.5 from start)
        self.sound_system.current_time = start_time + 0.6
        self.sound_system._process_delayed_sounds()

        # Should have played the rack
        self.assertEqual(self.mock_backend.play_sound.call_count, 3)


if __name__ == "__main__":
    unittest.main()

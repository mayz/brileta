import unittest
from unittest.mock import MagicMock, patch

from catley.events import SoundEvent, publish_event, reset_event_bus_for_testing
from catley.sound.definitions import SOUND_DEFINITIONS, SoundDefinition, SoundLayer
from catley.sound.system import SoundSystem


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

        # Let's try mocking random.choice to be sure
        with patch("random.choice", return_value="var1.ogg"):
            self.sound_system.play_one_shot("variant_test", 0, 0)
            args, _ = self.mock_backend.play_sound.call_args
            self.assertIn("var1.ogg", str(args[0]))


if __name__ == "__main__":
    unittest.main()

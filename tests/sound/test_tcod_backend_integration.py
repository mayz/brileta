"""Integration tests for TCOD audio backend that detect real bugs.

These tests use the actual TCODAudioBackend with TCOD's BasicMixer to verify
that our fixes work with the real TCOD library, not just our mock implementation.
"""

from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from catley.backends.tcod.audio import TCODAudioBackend
from catley.game.actors.core import Actor
from catley.sound.audio_backend import LoadedSound
from catley.sound.emitter import SoundEmitter
from catley.sound.system import SoundSystem
from catley.util.spatial import SpatialHashGrid, SpatialIndex


def create_spatial_index(actors: list[Actor]) -> SpatialIndex[Actor]:
    """Create a spatial index from a list of actors for testing."""
    spatial_index = SpatialHashGrid[Actor](cell_size=16)
    for actor in actors:
        spatial_index.add(actor)
    return spatial_index


class TestTCODAudioBackendBugDetection:
    """Test real TCOD backend to detect actual audio bugs like the campfire issue."""

    def setup_method(self) -> None:
        """Set up test fixtures with real TCOD audio backend."""
        self.backend = TCODAudioBackend(
            max_channels=4
        )  # Small number for easier testing
        try:
            self.backend.initialize()
        except Exception as e:
            pytest.skip(f"TCOD audio backend not available: {e}")

        # Create a simple test sound
        self.test_sound = LoadedSound(
            data=np.zeros((1000, 1), dtype=np.float32),  # 1000 samples of silence
            sample_rate=44100,
            channels=1,
        )

    def teardown_method(self) -> None:
        """Clean up after each test."""
        if hasattr(self, "backend") and self.backend:
            self.backend.shutdown()

    def test_tcod_channel_stop_actually_stops(self) -> None:
        """Test that TCOD channels actually stop when stop() is called.

        This test will FAIL if using the broken self._channel.stop() implementation
        and PASS with our sound_queue.clear() fix.
        """
        # Get a channel and start playing
        channel = self.backend.get_channel()
        assert channel is not None, "Should get a channel from backend"

        channel.play(self.test_sound, volume=1.0, loop=True)

        # Verify it's playing
        assert channel.is_playing(), "Channel should be playing after play()"
        assert self.backend.get_active_channel_count() == 1, (
            "Backend should report 1 active channel"
        )

        # Stop the channel
        channel.stop()

        # THE CRITICAL TEST - with broken TCOD implementation, this will fail
        assert not channel.is_playing(), (
            "TCOD CHANNEL BUG DETECTED: Channel reports is_playing()=True after "
            "stop(). This indicates TCOD's fadeout(0.0005) is keeping the channel "
            "busy instead of stopping it immediately!"
        )

        # Verify backend also reports no active channels
        assert self.backend.get_active_channel_count() == 0, (
            "TCOD BACKEND BUG DETECTED: Backend reports active channels after stop(). "
            "This indicates channels are not properly stopped."
        )

    def test_tcod_channel_busy_state_after_stop(self) -> None:
        """Test that TCOD channel.busy becomes False after stop().

        The original bug was that TCOD's stop() calls fadeout(0.0005) which
        keeps channel.busy=True even though the sound should be stopped.
        """
        channel = self.backend.get_channel()
        assert channel is not None

        channel.play(self.test_sound, volume=1.0, loop=True)

        # Access the underlying TCOD channel to check its busy state
        tcod_channel = channel._channel
        assert tcod_channel.busy, "TCOD channel should be busy when playing"

        # Stop the channel
        channel.stop()

        # THE CRITICAL TEST - the underlying TCOD channel should not be busy
        assert not tcod_channel.busy, (
            "TCOD CHANNEL BUSY BUG DETECTED: TCOD channel.busy=True after stop(). "
            "This indicates TCOD's fadeout(0.0005) implementation is keeping the "
            "channel busy. Our fix should directly clear the sound queue to make "
            "busy=False immediately."
        )

    def test_tcod_backend_campfire_bug_reproduction(self) -> None:
        """Reproduce the exact campfire bug using real TCOD backend.

        This integrates the real TCOD backend with the SoundSystem to test
        the complete audio pipeline that was causing the ghost campfire sounds.
        """
        # Set up sound system with real TCOD backend
        sound_system = SoundSystem()
        sound_system.set_audio_backend(self.backend)
        sound_system.set_transition_duration(0.0)  # Disable interpolation for testing

        # Create campfire emitter and actor
        campfire_emitter = SoundEmitter("fire_ambient", active=True)
        campfire_actor = MagicMock(spec=Actor)
        campfire_actor.x = 0
        campfire_actor.y = 0
        campfire_actor.sound_emitters = [campfire_emitter]

        # Start with player close to campfire (within max_distance=17)
        actors = cast(list[Actor], [campfire_actor])
        sound_system.update(
            5, 5, create_spatial_index(actors), 0.1
        )  # Player at distance ~7

        # Verify sound is playing
        initial_channels = self.backend.get_active_channel_count()
        assert initial_channels > 0, "Campfire should be audible when player is close"

        # Move player far beyond max_distance (the original bug scenario)
        sound_system.update(
            30, 30, create_spatial_index(actors), 0.1
        )  # Player at distance ~42, way beyond max_distance=17

        # THE CRITICAL TEST - no channels should be active
        final_channels = self.backend.get_active_channel_count()
        assert final_channels == 0, (
            f"CAMPFIRE BUG DETECTED WITH REAL TCOD BACKEND: {final_channels} "
            f"channels still active when player is at distance ~42 "
            f"(beyond max_distance=17). This indicates our TCOD Channel.stop() "
            f"fix is not working properly!"
        )

        # Additional verification - check all channels individually
        for i in range(self.backend._max_channels):
            channel = self.backend._channels[i]
            assert not channel.is_playing(), (
                f"CAMPFIRE BUG DETECTED: Channel {i} is still playing when all "
                f"should be stopped!"
            )

    def test_tcod_volume_zero_stops_channel(self) -> None:
        """Test that setting volume to 0 stops the TCOD channel.

        Our MockAudioChannel implements this behavior in set_volume(),
        but we need to verify the real TCOD implementation also stops.
        """
        channel = self.backend.get_channel()
        assert channel is not None

        channel.play(self.test_sound, volume=1.0, loop=True)
        assert channel.is_playing(), "Channel should be playing"
        assert self.backend.get_active_channel_count() == 1

        # Set volume to zero - this should stop the channel
        channel.set_volume(0.0)

        # Verify channel is stopped
        assert not channel.is_playing(), (
            "TCOD VOLUME BUG DETECTED: Channel still playing after set_volume(0.0). "
            "Our implementation should stop the channel when volume=0."
        )
        assert self.backend.get_active_channel_count() == 0, (
            "TCOD VOLUME BUG DETECTED: Backend reports active channels after volume=0."
        )

    def test_tcod_multiple_channels_stop_behavior(self) -> None:
        """Test that multiple TCOD channels all stop correctly.

        The campfire bug involved multiple layers (4 layers for fire_ambient),
        so we need to verify all channels stop when they should.
        """
        # Start multiple channels playing
        channels = []
        for i in range(3):  # Test with 3 channels
            channel = self.backend.get_channel()
            assert channel is not None, f"Should get channel {i}"
            channel.play(self.test_sound, volume=1.0, loop=True)
            channels.append(channel)

        # Verify all are playing
        assert self.backend.get_active_channel_count() == 3, (
            "All 3 channels should be active"
        )
        for i, channel in enumerate(channels):
            assert channel.is_playing(), f"Channel {i} should be playing"

        # Stop all channels
        for i, channel in enumerate(channels):
            channel.stop()

            # Verify this specific channel stopped
            assert not channel.is_playing(), (
                f"TCOD MULTI-CHANNEL BUG DETECTED: Channel {i} still playing "
                f"after stop()"
            )

        # Verify backend reports no active channels
        final_count = self.backend.get_active_channel_count()
        assert final_count == 0, (
            f"TCOD MULTI-CHANNEL BUG DETECTED: Backend reports {final_count} "
            f"active channels after stopping all channels. This indicates some "
            f"channels weren't properly stopped."
        )

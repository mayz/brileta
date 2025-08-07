"""TCOD audio backend implementation.

This module provides a concrete implementation of the AudioBackend interface
using TCOD's SDL2-based audio system.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tcod.sdl.audio

from catley.sound.audio_backend import AudioBackend, AudioChannel, LoadedSound
from catley.sound.loader import AudioLoader

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TCODChannel(AudioChannel):
    """TCOD implementation of an audio channel."""

    def __init__(
        self, channel: tcod.sdl.audio.Channel, backend: TCODAudioBackend
    ) -> None:
        """Initialize a TCOD audio channel wrapper.

        Args:
            channel: The underlying TCOD channel
            backend: Reference to the audio backend
        """
        self._channel = channel
        self._backend = backend
        self._current_sound: LoadedSound | None = None
        self._is_looping = False

    def play(self, sound: LoadedSound, volume: float = 1.0, loop: bool = False) -> None:
        """Start playing a sound on this channel."""
        # Convert sound data to TCOD format if needed
        audio_data = self._prepare_audio_data(sound)

        # Set volume
        self._channel.volume = volume * self._backend._master_volume

        # Play the sound
        if loop:
            self._channel.play(audio_data, loops=-1)  # -1 means infinite loops
        else:
            self._channel.play(audio_data, loops=0)  # 0 means play once

        self._current_sound = sound
        self._is_looping = loop

        volume_val = volume * self._backend._master_volume
        logger.debug(
            f"Playing sound on channel: volume={volume_val:.2f}, "
            f"loop={loop}, shape={audio_data.shape}"
        )

    def stop(self) -> None:
        """Stop playback on this channel."""
        # TCOD's stop() method is broken
        # It uses fadeout(0.0005) which keeps the channel busy
        # Directly clear the sound queue for immediate stop
        with self._channel._lock:
            self._channel.sound_queue.clear()
            self._channel.on_end_callback = None

        self._current_sound = None
        self._is_looping = False

    def set_volume(self, volume: float) -> None:
        """Update the volume of this channel."""
        if volume <= 0.0:
            # Stop the channel completely when volume is 0 to ensure silence
            self.stop()
        else:
            self._channel.volume = volume * self._backend._master_volume

    def is_playing(self) -> bool:
        """Check if this channel is currently playing audio."""
        return bool(self._channel.busy)

    def fadeout(self, duration_ms: int) -> None:
        """Fade out the channel over the specified duration."""
        # TCOD doesn't have built-in fadeout, so we'll stop immediately for now
        # TODO: Implement manual fadeout using volume ramping
        logger.debug(f"Fadeout requested for {duration_ms}ms (stopping immediately)")
        self.stop()

    def _prepare_audio_data(self, sound: LoadedSound) -> np.ndarray:
        """Prepare audio data for TCOD playback.

        Args:
            sound: The sound to prepare

        Returns:
            Audio data in TCOD-compatible format
        """
        # TCOD expects audio data as float32 numpy array
        # If stereo, it should be shape (frames, 2)
        # If mono, it should be shape (frames,) or (frames, 1)

        if sound.channels == 1 and sound.data.ndim == 1:
            # Mono audio with shape (frames,) - TCOD can handle this
            return sound.data
        if sound.channels == 2 and sound.data.ndim == 2:
            # Stereo audio with shape (frames, 2) - TCOD can handle this
            return sound.data
        # Need to reshape or convert
        if sound.channels == 1 and sound.data.ndim == 2:
            # Mono with shape (frames, 1) - flatten it
            return sound.data.flatten()
        logger.warning(
            f"Unexpected audio format: channels={sound.channels}, "
            f"shape={sound.data.shape}"
        )
        return sound.data


class TCODAudioBackend(AudioBackend):
    """TCOD implementation of the audio backend using SDL2 audio."""

    def __init__(self, max_channels: int = 16) -> None:
        """Initialize the TCOD audio backend.

        Args:
            max_channels: Maximum number of simultaneous audio channels
        """
        self._max_channels = max_channels
        self._mixer: tcod.sdl.audio.BasicMixer | None = None
        self._channels: list[TCODChannel] = []
        self._master_volume = 1.0
        self._audio_loader = AudioLoader()
        self._initialized = False

    def initialize(self, sample_rate: int = 44100, channels: int = 2) -> None:
        """Initialize the audio backend."""
        if self._initialized:
            logger.warning("Audio backend already initialized")
            return

        try:
            # Open SDL2 audio device with specified parameters
            device = tcod.sdl.audio.open(
                frequency=sample_rate,
                format=np.float32,  # 32-bit float as NumPy dtype
                channels=channels,
                samples=2048,  # Buffer size
                callback=None,  # Use push mode, not callback mode
            )

            # Create the mixer
            self._mixer = tcod.sdl.audio.BasicMixer(device)

            # Start the mixer thread if not already started
            if not self._mixer.is_alive():
                self._mixer.start()

            # Create channel wrappers
            self._channels = []
            for i in range(self._max_channels):
                # Get a channel with a unique integer key
                tcod_channel = self._mixer.get_channel(i)
                self._channels.append(TCODChannel(tcod_channel, self))

            self._initialized = True
            logger.info(
                f"TCOD audio initialized: {sample_rate}Hz, {channels} channels, "
                f"{self._max_channels} mixer channels"
            )

        except Exception as e:
            logger.error(f"Failed to initialize TCOD audio: {e}")
            raise

    def shutdown(self) -> None:
        """Shut down the audio backend and release resources."""
        if not self._initialized:
            return

        try:
            # Stop all sounds
            self.stop_all_sounds()

            # Clear channels
            self._channels.clear()

            # Close the mixer (this should close the device too)
            if self._mixer:
                # Stop the mixer thread and close
                self._mixer.close()
                self._mixer = None

            self._initialized = False
            logger.info("TCOD audio shut down")

        except Exception as e:
            logger.error(f"Error during audio shutdown: {e}")

    def get_channel(self) -> AudioChannel | None:
        """Get an available audio channel."""
        if not self._initialized:
            logger.warning("Audio backend not initialized")
            return None

        # Find an idle channel
        for channel in self._channels:
            if not channel.is_playing():
                return channel

        # No idle channels available
        logger.debug("No audio channels available")
        return None

    def play_sound(
        self, sound: LoadedSound, volume: float = 1.0, loop: bool = False
    ) -> AudioChannel | None:
        """Play a sound on an available channel."""
        channel = self.get_channel()
        if channel:
            channel.play(sound, volume, loop)
        return channel

    def stop_all_sounds(self) -> None:
        """Stop all currently playing sounds."""
        for channel in self._channels:
            if channel.is_playing():
                channel.stop()

    def set_master_volume(self, volume: float) -> None:
        """Set the master volume for all audio."""
        self._master_volume = max(0.0, min(1.0, volume))

        # Update volume on all active channels
        for channel in self._channels:
            if channel.is_playing():
                # Get current channel volume (might be float or tuple)
                current_vol = channel._channel.volume
                if isinstance(current_vol, tuple):
                    # Stereo volume - use first channel
                    current_vol = current_vol[0]
                # Re-apply volume to account for master volume change
                base_volume = (
                    current_vol / self._master_volume if self._master_volume > 0 else 0
                )
                channel.set_volume(base_volume)

    def update(self) -> None:
        """Update the audio backend."""
        # TCOD's audio system runs on a separate thread and doesn't need
        # explicit updates in the main loop
        pass

    def get_active_channel_count(self) -> int:
        """Get the number of channels currently playing audio."""
        return sum(1 for channel in self._channels if channel.is_playing())

    def load_sound(self, file_path: Path) -> LoadedSound:
        """Load a sound file into memory."""
        return self._audio_loader.load(file_path)

    def set_assets_path(self, assets_path: Path) -> None:
        """Set the base path for audio assets.

        Args:
            assets_path: Base path for audio assets
        """
        self._audio_loader.assets_path = assets_path

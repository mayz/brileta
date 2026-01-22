"""TCOD audio backend implementation using SDL3 AudioStream.

This module provides a concrete implementation of the AudioBackend interface
using TCOD's SDL3-based audio system with AudioStream for audio playback.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import numpy as np
import tcod.sdl.audio

from catley.sound.audio_backend import AudioBackend, AudioChannel, LoadedSound
from catley.sound.loader import AudioLoader

logger = logging.getLogger(__name__)


class SDL3AudioChannel(AudioChannel):
    """SDL3 AudioStream-based implementation of an audio channel.

    Each channel wraps an AudioStream and manages playback state.
    We track playing state internally to avoid querying the stream,
    which can crash if the stream is in an invalid state.
    """

    def __init__(
        self, stream: tcod.sdl.audio.AudioStream, backend: TCODAudioBackend
    ) -> None:
        """Initialize an SDL3 audio channel.

        Args:
            stream: The underlying SDL3 AudioStream
            backend: Reference to the audio backend for master volume
        """
        self._stream = stream
        self._backend = backend
        self._current_sound: LoadedSound | None = None
        self._is_looping = False
        self._current_priority = 0
        self._base_volume = 1.0
        self._playing = False  # Track playing state internally

    @property
    def current_priority(self) -> int:
        """Get the priority of the currently playing sound."""
        return self._current_priority if self._playing else 0

    def play(
        self,
        sound: LoadedSound,
        volume: float = 1.0,
        loop: bool = False,
        priority: int = 5,
    ) -> None:
        """Start playing a sound on this channel.

        Args:
            sound: The loaded sound data to play
            volume: Volume level (0.0 to 1.0)
            loop: Whether to loop the sound continuously
            priority: Sound priority (higher numbers = higher priority)
        """
        # Stop any current playback first
        self.stop()

        # Prepare audio data
        audio_data = self._prepare_audio_data(sound)

        try:
            # Set volume via gain
            self._base_volume = volume
            self._stream.gain = volume * self._backend._master_volume

            # Queue the audio data
            self._stream.queue_audio(audio_data)

            self._current_sound = sound
            self._is_looping = loop
            self._current_priority = priority
            self._playing = True

            effective_volume = volume * self._backend._master_volume
            logger.debug(
                f"Playing sound on channel: volume={effective_volume:.2f}, "
                f"loop={loop}, shape={audio_data.shape}"
            )
        except Exception as e:
            logger.warning(f"Failed to play sound on channel: {e}")
            self._playing = False

    def stop(self) -> None:
        """Stop playback on this channel."""
        # First set gain to 0 to silence immediately
        with contextlib.suppress(Exception):
            self._stream.gain = 0.0

        # Clear all queued audio data
        with contextlib.suppress(Exception):
            self._stream.flush()

        self._current_sound = None
        self._is_looping = False
        self._current_priority = 0
        self._playing = False

    def set_volume(self, volume: float) -> None:
        """Update the volume of this channel."""
        if volume <= 0.0:
            self.stop()
        else:
            self._base_volume = volume
            with contextlib.suppress(Exception):
                self._stream.gain = volume * self._backend._master_volume

    def is_playing(self) -> bool:
        """Check if this channel is currently playing audio."""
        return self._playing

    def fadeout(self, duration_ms: int) -> None:
        """Fade out the channel over the specified duration."""
        # AudioStream doesn't have built-in fadeout, stop immediately
        logger.debug(f"Fadeout requested for {duration_ms}ms (stopping immediately)")
        self.stop()

    def update(self) -> None:
        """Update the channel state, handling looping if needed."""
        if not self._playing or self._current_sound is None:
            return

        # Check if we need to re-queue audio for looping
        if self._is_looping:
            try:
                queued = self._stream.queued_samples
                if queued < self._current_sound.sample_rate // 4:
                    audio_data = self._prepare_audio_data(self._current_sound)
                    self._stream.queue_audio(audio_data)
            except Exception:
                # Stream became invalid, mark as not playing
                self._playing = False
        else:
            # For non-looping sounds, check if playback is done
            try:
                if self._stream.queued_samples == 0:
                    self._playing = False
            except Exception:
                self._playing = False

    def _prepare_audio_data(self, sound: LoadedSound) -> np.ndarray:
        """Prepare audio data for playback.

        Args:
            sound: The sound to prepare

        Returns:
            Audio data as float32 numpy array
        """
        # Ensure float32 format
        if sound.data.dtype != np.float32:
            data = sound.data.astype(np.float32)
        else:
            data = sound.data

        # Handle mono vs stereo
        if sound.channels == 1 and data.ndim == 2:
            # Mono with shape (frames, 1) - flatten it
            return data.flatten()

        return data


class TCODAudioBackend(AudioBackend):
    """TCOD audio backend using SDL3 AudioStream API."""

    def __init__(self, max_channels: int = 16) -> None:
        """Initialize the TCOD audio backend.

        Args:
            max_channels: Maximum number of simultaneous audio channels
        """
        self._max_channels = max_channels
        self._device: tcod.sdl.audio.AudioDevice | None = None
        self._channels: list[SDL3AudioChannel] = []
        self._master_volume = 1.0
        self._audio_loader = AudioLoader()
        self._initialized = False
        self._sample_rate = 44100
        self._num_channels = 2

    def initialize(self, sample_rate: int = 44100, channels: int = 2) -> None:
        """Initialize the audio backend.

        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
        """
        if self._initialized:
            logger.warning("Audio backend already initialized")
            return

        self._sample_rate = sample_rate
        self._num_channels = channels

        try:
            # SDL3 audio API: get default playback device and open it
            self._device = tcod.sdl.audio.get_default_playback().open(
                format=np.float32,
                channels=channels,
                frequency=sample_rate,
            )

            # Create audio streams as channels
            self._channels = []
            for _ in range(self._max_channels):
                # Create a new stream bound to the device
                stream = self._device.new_stream(
                    format=np.float32,
                    channels=channels,
                    frequency=sample_rate,
                )
                self._channels.append(SDL3AudioChannel(stream, self))

            self._initialized = True
            logger.info(
                f"SDL3 audio initialized: {sample_rate}Hz, {channels} channels, "
                f"{self._max_channels} streams"
            )

        except Exception as e:
            logger.error(f"Failed to initialize SDL3 audio: {e}")
            raise

    def shutdown(self) -> None:
        """Shut down the audio backend and release resources."""
        if not self._initialized:
            return

        try:
            # Stop all sounds
            self.stop_all_sounds()

            # Close all streams
            for channel in self._channels:
                with contextlib.suppress(Exception):
                    channel._stream.close()
            self._channels.clear()

            # Close the device
            if self._device:
                self._device.close()
                self._device = None

            self._initialized = False
            logger.info("SDL3 audio shut down")

        except Exception as e:
            logger.error(f"Error during audio shutdown: {e}")

    def get_channel(self, priority: int = 5) -> AudioChannel | None:
        """Get an available audio channel.

        Args:
            priority: Sound priority for voice stealing (higher = higher priority)

        Returns:
            An available AudioChannel, or None if all channels are in use and
            no lower-priority channels can be stolen
        """
        if not self._initialized:
            return None

        # First try to find an idle channel
        for channel in self._channels:
            if not channel.is_playing():
                return channel

        # Voice stealing: find the lowest priority channel we can steal
        lowest_priority = priority
        steal_candidate = None
        for channel in self._channels:
            if channel.current_priority < lowest_priority:
                lowest_priority = channel.current_priority
                steal_candidate = channel

        if steal_candidate:
            logger.debug(
                f"Voice stealing: priority {priority} taking channel "
                f"with priority {lowest_priority}"
            )
            steal_candidate.stop()
            return steal_candidate

        # No channels available and none can be stolen
        logger.debug(f"No audio channels available for priority {priority}")
        return None

    def play_sound(
        self,
        sound: LoadedSound,
        volume: float = 1.0,
        loop: bool = False,
        priority: int = 5,
    ) -> AudioChannel | None:
        """Play a sound on an available channel.

        Args:
            sound: The loaded sound data to play
            volume: Volume level (0.0 to 1.0)
            loop: Whether to loop the sound continuously
            priority: Sound priority (higher numbers = higher priority)

        Returns:
            The AudioChannel playing the sound, or None if no channels available
        """
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
        """Set the master volume for all audio.

        Args:
            volume: Master volume level (0.0 to 1.0)
        """
        self._master_volume = max(0.0, min(1.0, volume))

        # Update gain on all active channels
        for channel in self._channels:
            if channel.is_playing():
                channel._stream.gain = channel._base_volume * self._master_volume

    def update(self) -> None:
        """Update the audio backend.

        This should be called each frame to handle looping sounds.
        """
        for channel in self._channels:
            channel.update()

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

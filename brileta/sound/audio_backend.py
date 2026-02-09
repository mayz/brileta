"""Abstract base class for audio backends.

This module defines the abstract interface for audio playback backends in the game.
It provides a backend-agnostic API for playing sounds, managing channels, and
controlling audio playback.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class LoadedSound:
    """Represents a loaded sound with its audio data and metadata."""

    data: np.ndarray  # Audio samples as numpy array
    sample_rate: int  # Sample rate in Hz
    channels: int  # Number of audio channels (1=mono, 2=stereo)


class AudioChannel(abc.ABC):
    """Abstract base class for an audio channel.

    A channel represents a single sound playback instance with its own
    volume, playback position, and loop settings.
    """

    @property
    @abc.abstractmethod
    def current_priority(self) -> int:
        """Get the priority of the currently playing sound.

        Returns:
            Priority value (higher numbers = higher priority), or 0 if not playing
        """
        ...

    @abc.abstractmethod
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
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop playback on this channel."""
        ...

    @abc.abstractmethod
    def set_volume(self, volume: float) -> None:
        """Update the volume of this channel.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        ...

    @abc.abstractmethod
    def is_playing(self) -> bool:
        """Check if this channel is currently playing audio.

        Returns:
            True if audio is playing, False otherwise
        """
        ...

    @abc.abstractmethod
    def fadeout(self, duration_ms: int) -> None:
        """Fade out the channel over the specified duration.

        Args:
            duration_ms: Fadeout duration in milliseconds
        """
        ...


class AudioBackend(abc.ABC):
    """Abstract base class for audio playback backends.

    This class defines the interface that all audio backends must implement.
    It provides methods for playing sounds, managing channels, and controlling
    the overall audio system.
    """

    @abc.abstractmethod
    def initialize(self, sample_rate: int = 44100, channels: int = 2) -> None:
        """Initialize the audio backend.

        Args:
            sample_rate: Sample rate in Hz
            channels: Number of output channels (1=mono, 2=stereo)
        """
        ...

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Shut down the audio backend and release resources."""
        ...

    @abc.abstractmethod
    def get_channel(self, priority: int = 5) -> AudioChannel | None:
        """Get an available audio channel.

        Args:
            priority: Sound priority for voice stealing (higher = higher priority)

        Returns:
            An available AudioChannel, or None if all channels are in use and
            no lower-priority channels can be stolen
        """
        ...

    @abc.abstractmethod
    def play_sound(
        self,
        sound: LoadedSound,
        volume: float = 1.0,
        loop: bool = False,
        priority: int = 5,
    ) -> AudioChannel | None:
        """Play a sound on an available channel.

        This is a convenience method that gets an available channel and
        starts playing the sound on it.

        Args:
            sound: The loaded sound data to play
            volume: Volume level (0.0 to 1.0)
            loop: Whether to loop the sound continuously
            priority: Sound priority (higher numbers = higher priority)

        Returns:
            The AudioChannel playing the sound, or None if no channels available
        """
        ...

    @abc.abstractmethod
    def stop_all_sounds(self) -> None:
        """Stop all currently playing sounds."""
        ...

    @abc.abstractmethod
    def set_master_volume(self, volume: float) -> None:
        """Set the master volume for all audio.

        Args:
            volume: Master volume level (0.0 to 1.0)
        """
        ...

    @abc.abstractmethod
    def update(self) -> None:
        """Update the audio backend.

        This method should be called regularly (e.g., each frame) to
        process audio events and update the audio state.
        """
        ...

    @abc.abstractmethod
    def get_active_channel_count(self) -> int:
        """Get the number of channels currently playing audio.

        Returns:
            Number of active channels
        """
        ...

    @abc.abstractmethod
    def load_sound(self, file_path: Path) -> LoadedSound:
        """Load a sound file into memory.

        Args:
            file_path: Path to the sound file

        Returns:
            LoadedSound object containing the audio data

        Raises:
            IOError: If the file cannot be loaded
            ValueError: If the file format is not supported
        """
        ...

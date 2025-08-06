"""Audio file loading utilities.

This module provides utilities for loading audio files into memory using
the soundfile library. It handles format conversion and caching of loaded
sounds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from .audio_backend import LoadedSound

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


class AudioLoader:
    """Handles loading and caching of audio files.

    The AudioLoader is responsible for:
    - Loading audio files from disk using soundfile
    - Converting audio to the required format
    - Caching loaded sounds to avoid redundant disk I/O
    - Handling different audio file formats (.ogg, .wav, etc.)
    """

    def __init__(self, assets_path: Path | None = None) -> None:
        """Initialize the audio loader.

        Args:
            assets_path: Base path for audio assets. If None, paths must be absolute.
        """
        self.assets_path = assets_path
        self._cache: dict[Path, LoadedSound] = {}

    def load(self, file_path: Path | str) -> LoadedSound:
        """Load an audio file into memory.

        Args:
            file_path: Path to the audio file (relative to assets_path or absolute)

        Returns:
            LoadedSound object containing the audio data

        Raises:
            IOError: If the file cannot be loaded
            ValueError: If the file format is not supported
        """
        # Convert to Path object
        path = Path(file_path)

        # Make path absolute if needed
        if not path.is_absolute() and self.assets_path:
            path = self.assets_path / path

        # Check cache first
        if path in self._cache:
            logger.debug(f"Returning cached sound: {path}")
            return self._cache[path]

        # Load the audio file
        try:
            data, sample_rate = sf.read(path, dtype="float32")
            logger.info(f"Loaded sound: {path} ({sample_rate}Hz, shape={data.shape})")
        except Exception as e:
            raise OSError(f"Failed to load audio file {path}: {e}") from e

        # Ensure data is in the correct shape
        # soundfile returns (frames,) for mono or (frames, channels) for multi-channel
        channels = 1 if data.ndim == 1 else data.shape[1]

        # Create LoadedSound object
        sound = LoadedSound(
            data=data,
            sample_rate=sample_rate,
            channels=channels,
        )

        # Cache the loaded sound
        self._cache[path] = sound

        return sound

    def clear_cache(self) -> None:
        """Clear the sound cache to free memory."""
        logger.info(f"Clearing audio cache ({len(self._cache)} sounds)")
        self._cache.clear()

    def preload(self, file_paths: list[Path | str]) -> None:
        """Preload multiple audio files into the cache.

        Args:
            file_paths: List of paths to preload
        """
        logger.info(f"Preloading {len(file_paths)} audio files")
        for path in file_paths:
            try:
                self.load(path)
            except Exception as e:
                logger.error(f"Failed to preload {path}: {e}")

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about the current cache state.

        Returns:
            Dictionary with cache statistics
        """
        total_samples = 0
        total_memory = 0

        for sound in self._cache.values():
            total_samples += sound.data.size
            # float32 = 4 bytes per sample
            total_memory += sound.data.nbytes

        return {
            "cached_sounds": len(self._cache),
            "total_samples": total_samples,
            "memory_usage_mb": total_memory / (1024 * 1024),
        }

    @staticmethod
    def convert_to_mono(sound: LoadedSound) -> LoadedSound:
        """Convert a stereo sound to mono by averaging channels.

        Args:
            sound: The sound to convert

        Returns:
            New LoadedSound object with mono audio
        """
        if sound.channels == 1:
            return sound

        # Average the channels
        mono_data = np.mean(sound.data, axis=1)

        return LoadedSound(
            data=mono_data,
            sample_rate=sound.sample_rate,
            channels=1,
        )

    @staticmethod
    def resample(sound: LoadedSound, target_sample_rate: int) -> LoadedSound:
        """Resample audio to a different sample rate.

        This is a simple linear resampling implementation. For production use,
        consider using a proper resampling library like resampy.

        Args:
            sound: The sound to resample
            target_sample_rate: Target sample rate in Hz

        Returns:
            New LoadedSound object with resampled audio
        """
        if sound.sample_rate == target_sample_rate:
            return sound

        # Calculate resampling ratio
        ratio = target_sample_rate / sound.sample_rate

        # Calculate new length
        if sound.data.ndim == 1:
            new_length = int(len(sound.data) * ratio)
            # Simple linear interpolation for mono
            old_indices = np.linspace(0, len(sound.data) - 1, new_length)
            new_data = np.interp(old_indices, np.arange(len(sound.data)), sound.data)
        else:
            new_length = int(sound.data.shape[0] * ratio)
            # Resample each channel separately
            new_data = np.zeros((new_length, sound.channels), dtype=np.float32)
            for ch in range(sound.channels):
                old_indices = np.linspace(0, sound.data.shape[0] - 1, new_length)
                new_data[:, ch] = np.interp(
                    old_indices, np.arange(sound.data.shape[0]), sound.data[:, ch]
                )

        return LoadedSound(
            data=new_data,
            sample_rate=target_sample_rate,
            channels=sound.channels,
        )

    @staticmethod
    def pitch_shift(sound: LoadedSound, pitch_factor: float) -> LoadedSound:
        """Shift the pitch of audio by resampling.

        This changes the pitch by playing the audio faster or slower. A pitch_factor
        of 1.0 means no change, 0.5 means one octave down, 2.0 means one octave up.

        Args:
            sound: The sound to pitch shift
            pitch_factor: Pitch multiplier (e.g., 1.1 for 10% higher pitch)

        Returns:
            New LoadedSound object with pitch-shifted audio
        """
        if pitch_factor == 1.0:
            return sound

        # To shift pitch, we resample the audio to play it faster/slower
        # but keep the original sample rate in the metadata
        # This makes the audio play back at a different speed/pitch

        # Calculate new length (inverse of pitch factor)
        if sound.data.ndim == 1:
            old_length = len(sound.data)
            new_length = int(old_length / pitch_factor)
            # Stretch or compress the audio
            old_indices = np.linspace(0, old_length - 1, new_length)
            new_data = np.interp(old_indices, np.arange(old_length), sound.data).astype(
                np.float32
            )
        else:
            old_length = sound.data.shape[0]
            new_length = int(old_length / pitch_factor)
            # Resample each channel separately
            new_data = np.zeros((new_length, sound.channels), dtype=np.float32)
            for ch in range(sound.channels):
                old_indices = np.linspace(0, old_length - 1, new_length)
                new_data[:, ch] = np.interp(
                    old_indices, np.arange(old_length), sound.data[:, ch]
                ).astype(np.float32)

        return LoadedSound(
            data=new_data,
            sample_rate=sound.sample_rate,  # Keep original sample rate
            channels=sound.channels,
        )

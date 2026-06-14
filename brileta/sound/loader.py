"""Audio file loading utilities.

This module provides utilities for loading audio files into memory using
miniaudio's native decoders. Supports WAV, OGG/Vorbis, MP3, and FLAC
formats without any system-level audio library dependencies.
"""

from __future__ import annotations

import logging
from pathlib import Path

import miniaudio
import numpy as np

from .audio_backend import LoadedSound

logger = logging.getLogger(__name__)


class AudioLoader:
    """Handles loading and caching of audio files.

    The AudioLoader is responsible for:
    - Loading audio files from disk using miniaudio
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

        # Load the audio file using miniaudio's native decoders.
        # decode_file requires explicit channel/sample-rate values (0 is not
        # a valid sentinel), so we probe the file first to preserve its
        # native format.
        try:
            file_info = miniaudio.get_file_info(str(path))
            decoded = miniaudio.decode_file(
                str(path),
                output_format=miniaudio.SampleFormat.FLOAT32,
                nchannels=file_info.nchannels,
                sample_rate=file_info.sample_rate,
            )
            # Convert interleaved array.array to numpy
            samples = np.frombuffer(decoded.samples, dtype=np.float32)
            channels = decoded.nchannels
            sample_rate = decoded.sample_rate
            data = samples.reshape(-1, channels) if channels > 1 else samples.copy()
            logger.info(f"Loaded sound: {path} ({sample_rate}Hz, shape={data.shape})")
        except Exception as e:
            raise OSError(f"Failed to load audio file {path}: {e}") from e

        # Create LoadedSound object
        sound = LoadedSound(
            data=data,
            sample_rate=sample_rate,
            channels=channels,
        )

        # Cache the loaded sound
        self._cache[path] = sound

        return sound

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

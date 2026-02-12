"""Miniaudio audio backend implementation using a single mixed output stream.

This backend uses one miniaudio PlaybackDevice and mixes all active channels
inside a generator callback. It provides channel priority/voice stealing,
per-channel volume, looping, and master volume control.

Miniaudio requires no system-level audio libraries (no PortAudio, no libsndfile).
The C library is compiled into the Python extension at install time.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from pathlib import Path
from typing import Any

import miniaudio
import numpy as np

from brileta.sound.audio_backend import AudioBackend, AudioChannel, LoadedSound
from brileta.sound.loader import AudioLoader

logger = logging.getLogger(__name__)


class MiniaudioChannel(AudioChannel):
    """Per-sound playback channel used by the mixed miniaudio backend."""

    def __init__(self, backend: MiniaudioBackend) -> None:
        """Initialize a channel.

        Args:
            backend: Parent backend that owns channel state and mixer lock.
        """
        self._backend = backend
        self._audio_data: np.ndarray | None = None
        self._position = 0
        self._is_looping = False
        self._current_priority = 0
        self._base_volume = 1.0
        self._playing = False

    @property
    def current_priority(self) -> int:
        """Get the priority of the currently playing sound."""
        with self._backend._lock:
            return self._current_priority if self._playing else 0

    def play(
        self,
        sound: LoadedSound,
        volume: float = 1.0,
        loop: bool = False,
        priority: int = 5,
    ) -> None:
        """Start playing a sound on this channel."""
        try:
            prepared = self._backend._prepare_audio_data(sound)
        except Exception as exc:
            logger.warning("Failed to prepare sound for playback: %s", exc)
            self.stop()
            return

        with self._backend._lock:
            self._play_prepared_locked(prepared, volume, loop, priority)

    def stop(self) -> None:
        """Stop playback on this channel."""
        with self._backend._lock:
            self._stop_locked()

    def set_volume(self, volume: float) -> None:
        """Update the volume of this channel."""
        if volume <= 0.0:
            self.stop()
            return

        with self._backend._lock:
            self._base_volume = max(0.0, min(1.0, volume))

    def is_playing(self) -> bool:
        """Check if this channel is currently playing audio."""
        with self._backend._lock:
            return self._playing

    def fadeout(self, duration_ms: int) -> None:
        """Fade out the channel over the specified duration."""
        # TODO: Implement time-based fade ramps if gameplay needs non-instant fades.
        logger.debug("Fadeout requested for %sms (stopping immediately)", duration_ms)
        self.stop()

    def mix_into(self, outdata: np.ndarray, master_volume: float) -> None:
        """Mix this channel's audio into an output buffer.

        Called from the audio callback while the backend lock is held.

        Args:
            outdata: Output buffer with shape (frames, output_channels).
            master_volume: Backend master volume scalar.
        """
        if not self._playing or self._audio_data is None:
            return

        total_frames = self._audio_data.shape[0]
        if total_frames == 0:
            self._stop_locked()
            return

        gain = self._base_volume * master_volume
        if gain <= 0.0:
            self._stop_locked()
            return

        remaining = outdata.shape[0]
        out_pos = 0

        while remaining > 0 and self._playing:
            if self._position >= total_frames:
                if self._is_looping:
                    self._position = 0
                else:
                    self._stop_locked()
                    break

            available = total_frames - self._position
            chunk = min(remaining, available)
            if chunk <= 0:
                self._stop_locked()
                break

            segment = self._audio_data[self._position : self._position + chunk]
            if gain == 1.0:
                outdata[out_pos : out_pos + chunk] += segment
            else:
                outdata[out_pos : out_pos + chunk] += segment * gain

            self._position += chunk
            out_pos += chunk
            remaining -= chunk

    def _play_prepared_locked(
        self,
        prepared_data: np.ndarray,
        volume: float,
        loop: bool,
        priority: int,
    ) -> None:
        """Start playback from already prepared audio data.

        The backend lock must be held by the caller.
        """
        if prepared_data.shape[0] == 0:
            self._stop_locked()
            return

        self._audio_data = prepared_data
        self._position = 0
        self._is_looping = loop
        self._current_priority = priority
        self._base_volume = max(0.0, min(1.0, volume))
        self._playing = self._base_volume > 0.0

    def _stop_locked(self) -> None:
        """Reset playback state.

        The backend lock must be held by the caller.
        """
        self._audio_data = None
        self._position = 0
        self._is_looping = False
        self._current_priority = 0
        self._base_volume = 1.0
        self._playing = False


class MiniaudioBackend(AudioBackend):
    """Audio backend using miniaudio's PlaybackDevice.

    Uses a generator-based callback to mix all active channels into the
    output buffer. No system-level audio libraries are required beyond
    the miniaudio C header compiled into the Python extension.
    """

    def __init__(self, max_channels: int = 16) -> None:
        """Initialize the miniaudio backend.

        Args:
            max_channels: Maximum number of simultaneous audio channels.
        """
        self._max_channels = max_channels
        self._device: miniaudio.PlaybackDevice | None = None
        self._channels: list[MiniaudioChannel] = []
        self._master_volume = 1.0
        self._initialized = False
        self._sample_rate = 44100
        self._num_channels = 2
        self._lock = threading.RLock()
        self._loader = AudioLoader()

    def initialize(self, sample_rate: int = 44100, channels: int = 2) -> None:
        """Initialize the audio backend."""
        if self._initialized:
            logger.warning("Audio backend already initialized")
            return

        self._sample_rate = sample_rate
        self._num_channels = channels

        try:
            self._channels = [MiniaudioChannel(self) for _ in range(self._max_channels)]
            self._device = miniaudio.PlaybackDevice(
                output_format=miniaudio.SampleFormat.FLOAT32,
                nchannels=channels,
                sample_rate=sample_rate,
                buffersize_msec=0,  # 0 = let miniaudio pick platform default
            )
            # Create and start the callback generator
            generator = self._audio_generator()
            next(generator)  # Prime the generator
            self._device.start(generator)
            self._initialized = True
            logger.info(
                "Miniaudio backend initialized: %sHz, %s channels, %s streams",
                sample_rate,
                channels,
                self._max_channels,
            )
        except Exception:
            self._channels.clear()
            with contextlib.suppress(Exception):
                if self._device is not None:
                    self._device.close()
            self._device = None
            logger.exception("Failed to initialize miniaudio backend")
            raise

    def shutdown(self) -> None:
        """Shut down the audio backend and release resources."""
        if not self._initialized:
            return

        self.stop_all_sounds()

        with contextlib.suppress(Exception):
            if self._device is not None:
                self._device.close()

        with self._lock:
            self._device = None
            self._channels.clear()
            self._initialized = False

        logger.info("Miniaudio backend shut down")

    def get_channel(self, priority: int = 5) -> AudioChannel | None:
        """Get an available audio channel."""
        if not self._initialized:
            return None

        with self._lock:
            channel = self._get_channel_locked(priority)
            if channel is not None:
                return channel

        logger.debug("No audio channels available for priority %s", priority)
        return None

    def play_sound(
        self,
        sound: LoadedSound,
        volume: float = 1.0,
        loop: bool = False,
        priority: int = 5,
    ) -> AudioChannel | None:
        """Play a sound on an available channel."""
        if not self._initialized:
            return None

        try:
            prepared = self._prepare_audio_data(sound)
        except Exception as exc:
            logger.warning("Failed to prepare sound for playback: %s", exc)
            return None

        with self._lock:
            channel = self._get_channel_locked(priority)
            if channel is None:
                logger.debug("No audio channels available for priority %s", priority)
                return None
            channel._play_prepared_locked(prepared, volume, loop, priority)
            return channel

    def stop_all_sounds(self) -> None:
        """Stop all currently playing sounds."""
        with self._lock:
            for channel in self._channels:
                channel._stop_locked()

    def set_master_volume(self, volume: float) -> None:
        """Set the master volume for all audio."""
        with self._lock:
            self._master_volume = max(0.0, min(1.0, volume))

    def update(self) -> None:
        """Update the audio backend.

        The miniaudio backend is callback-driven. This method is a no-op
        since the generator callback handles all mixing.
        """

    def get_active_channel_count(self) -> int:
        """Get the number of channels currently playing audio."""
        with self._lock:
            return sum(1 for channel in self._channels if channel._playing)

    def load_sound(self, file_path: Path) -> LoadedSound:
        """Load a sound file into memory using miniaudio's native decoders.

        Delegates to AudioLoader, which handles decoding, numpy conversion,
        and caching. Supports WAV, OGG/Vorbis, MP3, and FLAC natively.

        Args:
            file_path: Path to the sound file.

        Returns:
            LoadedSound object containing the audio data.

        Raises:
            OSError: If the file cannot be loaded.
        """
        return self._loader.load(file_path)

    def _audio_generator(self) -> Any:
        """Generator that mixes all active channels for the PlaybackDevice.

        The device sends the required number of frames via .send(), and
        the generator yields raw bytes (float32 interleaved) for each callback.
        """
        # First yield is a no-op to prime the generator
        required_frames = yield b""

        while True:
            # Create a silence buffer for this callback
            outdata = np.zeros((required_frames, self._num_channels), dtype=np.float32)

            with self._lock:
                if self._master_volume > 0.0:
                    for channel in self._channels:
                        channel.mix_into(outdata, self._master_volume)

            np.clip(outdata, -1.0, 1.0, out=outdata)
            required_frames = yield outdata.tobytes()

    def _prepare_audio_data(self, sound: LoadedSound) -> np.ndarray:
        """Convert sound data to output sample rate/channels as float32."""
        data = sound.data.astype(np.float32, copy=False)

        if data.ndim == 1:
            data = data[:, np.newaxis]
        elif data.ndim != 2:
            raise ValueError(f"Unsupported audio data shape: {data.shape}")

        if sound.sample_rate != self._sample_rate:
            data = self._resample_audio(data, sound.sample_rate, self._sample_rate)

        source_channels = data.shape[1]
        if source_channels == self._num_channels:
            return data

        if self._num_channels == 1:
            return np.mean(data, axis=1, keepdims=True, dtype=np.float32)

        if source_channels == 1:
            return np.repeat(data, self._num_channels, axis=1)

        if source_channels > self._num_channels:
            return data[:, : self._num_channels]

        repeats = int(np.ceil(self._num_channels / source_channels))
        expanded = np.tile(data, (1, repeats))
        return expanded[:, : self._num_channels]

    def _get_channel_locked(self, priority: int) -> MiniaudioChannel | None:
        """Get or steal a channel while holding the backend lock."""
        for channel in self._channels:
            if not channel._playing:
                return channel

        lowest_priority = priority
        steal_candidate: MiniaudioChannel | None = None
        for channel in self._channels:
            if channel._playing and channel._current_priority < lowest_priority:
                lowest_priority = channel._current_priority
                steal_candidate = channel

        if steal_candidate is None:
            return None

        logger.debug(
            "Voice stealing: priority %s taking channel with priority %s",
            priority,
            lowest_priority,
        )
        steal_candidate._stop_locked()
        return steal_candidate

    @staticmethod
    def _resample_audio(
        data: np.ndarray,
        source_sample_rate: int,
        target_sample_rate: int,
    ) -> np.ndarray:
        """Resample audio with simple linear interpolation."""
        if source_sample_rate == target_sample_rate:
            return data

        source_frames = data.shape[0]
        if source_frames == 0:
            return data

        ratio = target_sample_rate / source_sample_rate
        target_frames = max(1, round(source_frames * ratio))

        source_indices = np.arange(source_frames, dtype=np.float64)
        target_indices = np.linspace(0, source_frames - 1, target_frames)

        resampled = np.empty((target_frames, data.shape[1]), dtype=np.float32)
        for channel_idx in range(data.shape[1]):
            resampled[:, channel_idx] = np.interp(
                target_indices,
                source_indices,
                data[:, channel_idx],
            ).astype(np.float32, copy=False)

        return resampled

"""Unit tests for the miniaudio audio backend internals."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import pytest

from brileta.backends.miniaudio_audio import (
    MiniaudioBackend,
    MiniaudioChannel,
)
from brileta.sound.audio_backend import LoadedSound
from brileta.sound.loader import AudioLoader


def _make_loaded_sound(data: np.ndarray, sample_rate: int = 44100) -> LoadedSound:
    """Build a LoadedSound for tests from a numpy array."""
    channels = 1 if data.ndim == 1 else data.shape[1]
    return LoadedSound(data=data, sample_rate=sample_rate, channels=channels)


def _make_backend_for_tests(num_channels: int = 2) -> MiniaudioBackend:
    """Create a backend instance without opening audio devices."""
    backend = MiniaudioBackend()
    backend._num_channels = num_channels
    backend._sample_rate = 44100
    return backend


class TestPrepareAudioData:
    """Tests for channel conversion logic in _prepare_audio_data."""

    def test_prepare_audio_data_upmixes_mono_to_stereo(self) -> None:
        backend = _make_backend_for_tests(num_channels=2)
        mono = np.array([0.1, -0.3, 0.8], dtype=np.float32)

        prepared = backend._prepare_audio_data(_make_loaded_sound(mono))

        assert prepared.shape == (3, 2)
        np.testing.assert_allclose(prepared[:, 0], mono)
        np.testing.assert_allclose(prepared[:, 1], mono)

    def test_prepare_audio_data_downmixes_stereo_to_mono(self) -> None:
        backend = _make_backend_for_tests(num_channels=1)
        stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)

        prepared = backend._prepare_audio_data(_make_loaded_sound(stereo))

        assert prepared.shape == (2, 1)
        np.testing.assert_allclose(prepared[:, 0], np.array([2.0, 3.0]))

    def test_prepare_audio_data_truncates_extra_channels(self) -> None:
        backend = _make_backend_for_tests(num_channels=2)
        surround = np.array(
            [[0.1, 0.2, 0.9], [0.3, 0.4, 0.8], [0.5, 0.6, 0.7]],
            dtype=np.float32,
        )

        prepared = backend._prepare_audio_data(_make_loaded_sound(surround))

        assert prepared.shape == (3, 2)
        np.testing.assert_allclose(prepared, surround[:, :2])


class TestResampleAudio:
    """Tests for linear resampling."""

    def test_resample_audio_scales_frame_count_with_sample_rate_ratio(self) -> None:
        source = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)

        resampled = MiniaudioBackend._resample_audio(
            source, source_sample_rate=4, target_sample_rate=8
        )

        assert resampled.shape == (8, 1)
        assert resampled.dtype == np.float32
        assert resampled[0, 0] == 0.0
        assert resampled[-1, 0] == 3.0

    def test_resample_audio_handles_zero_frame_input(self) -> None:
        empty = np.zeros((0, 2), dtype=np.float32)

        resampled = MiniaudioBackend._resample_audio(
            empty, source_sample_rate=22050, target_sample_rate=44100
        )

        assert resampled.shape == (0, 2)
        assert resampled.dtype == np.float32


class TestChannelMixing:
    """Tests for channel mix behavior without a live audio device."""

    def test_mix_into_applies_channel_and_master_gain(self) -> None:
        backend = _make_backend_for_tests(num_channels=2)
        channel = MiniaudioChannel(backend)
        sound = _make_loaded_sound(np.ones((2, 2), dtype=np.float32))
        channel.play(sound, volume=0.5, loop=False, priority=2)
        out = np.zeros((2, 2), dtype=np.float32)

        with backend._lock:
            channel.mix_into(out, master_volume=0.5)

        np.testing.assert_allclose(out, np.full((2, 2), 0.25, dtype=np.float32))

    def test_mix_into_wraps_when_looping(self) -> None:
        backend = _make_backend_for_tests(num_channels=2)
        channel = MiniaudioChannel(backend)
        loop_data = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
        channel.play(_make_loaded_sound(loop_data), volume=1.0, loop=True, priority=3)
        out = np.zeros((5, 2), dtype=np.float32)

        with backend._lock:
            channel.mix_into(out, master_volume=1.0)

        expected = np.array(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [1.0, 1.0], [2.0, 2.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(out, expected)

    def test_mix_into_stops_mid_buffer_for_non_looping_sound(self) -> None:
        backend = _make_backend_for_tests(num_channels=2)
        channel = MiniaudioChannel(backend)
        non_loop_data = np.array(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            dtype=np.float32,
        )
        channel.play(
            _make_loaded_sound(non_loop_data), volume=1.0, loop=False, priority=3
        )
        out = np.zeros((5, 2), dtype=np.float32)

        with backend._lock:
            channel.mix_into(out, master_volume=1.0)

        expected = np.array(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(out, expected)
        assert not channel.is_playing()

    def test_mix_into_is_noop_when_channel_is_not_playing(self) -> None:
        backend = _make_backend_for_tests(num_channels=2)
        channel = MiniaudioChannel(backend)
        out = np.full((4, 2), 0.75, dtype=np.float32)

        with backend._lock:
            channel.mix_into(out, master_volume=1.0)

        np.testing.assert_allclose(out, np.full((4, 2), 0.75, dtype=np.float32))


class TestAudioGenerator:
    """Tests for the _audio_generator mixing path."""

    def test_audio_generator_mixes_active_channels_into_bytes(self) -> None:
        """The generator should mix channels and yield raw float32 bytes."""
        backend = _make_backend_for_tests(num_channels=2)
        backend._channels = [MiniaudioChannel(backend) for _ in range(2)]

        # Start a sound on channel 0
        sound_data = np.full((4, 2), 0.5, dtype=np.float32)
        with backend._lock:
            backend._channels[0]._play_prepared_locked(sound_data, 1.0, False, 5)

        gen = backend._audio_generator()
        next(gen)  # Prime the generator

        # Request 4 frames from the generator
        result = gen.send(4)

        assert isinstance(result, bytes)
        # 4 frames * 2 channels * 4 bytes (float32) = 32 bytes
        assert len(result) == 32

        # Decode the bytes back to numpy and verify the mixed audio
        output = np.frombuffer(result, dtype=np.float32).reshape(4, 2)
        np.testing.assert_allclose(output, np.full((4, 2), 0.5))

    def test_audio_generator_yields_silence_when_no_channels_active(self) -> None:
        """The generator should yield silent bytes when nothing is playing."""
        backend = _make_backend_for_tests(num_channels=2)
        backend._channels = [MiniaudioChannel(backend)]

        gen = backend._audio_generator()
        next(gen)

        result = gen.send(3)

        assert isinstance(result, bytes)
        output = np.frombuffer(result, dtype=np.float32).reshape(3, 2)
        np.testing.assert_allclose(output, np.zeros((3, 2)))

    def test_audio_generator_respects_master_volume(self) -> None:
        """The generator should apply master volume to mixed output."""
        backend = _make_backend_for_tests(num_channels=2)
        backend._channels = [MiniaudioChannel(backend)]
        backend._master_volume = 0.5

        sound_data = np.ones((2, 2), dtype=np.float32)
        with backend._lock:
            backend._channels[0]._play_prepared_locked(sound_data, 1.0, False, 5)

        gen = backend._audio_generator()
        next(gen)
        result = gen.send(2)

        output = np.frombuffer(result, dtype=np.float32).reshape(2, 2)
        np.testing.assert_allclose(output, np.full((2, 2), 0.5))

    def test_audio_generator_clips_output(self) -> None:
        """The generator should clip mixed output to [-1.0, 1.0]."""
        backend = _make_backend_for_tests(num_channels=2)
        backend._channels = [MiniaudioChannel(backend) for _ in range(2)]

        # Two channels both at full volume with value 0.8 -> sum = 1.6, clipped to 1.0
        loud = np.full((2, 2), 0.8, dtype=np.float32)
        with backend._lock:
            backend._channels[0]._play_prepared_locked(loud.copy(), 1.0, False, 5)
            backend._channels[1]._play_prepared_locked(loud.copy(), 1.0, False, 5)

        gen = backend._audio_generator()
        next(gen)
        result = gen.send(2)

        output = np.frombuffer(result, dtype=np.float32).reshape(2, 2)
        np.testing.assert_allclose(output, np.full((2, 2), 1.0))


class TestPrepareAudioDataEndToEnd:
    """End-to-end tests for _prepare_audio_data with resample + channel conversion."""

    def test_resample_and_upmix_mono_22k_to_stereo_44k(self) -> None:
        """Mono 22050 Hz input should be resampled to 44100 Hz and upmixed to stereo."""
        backend = _make_backend_for_tests(num_channels=2)
        backend._sample_rate = 44100

        # 4 frames of mono audio at 22050 Hz
        mono_22k = np.array([0.0, 0.5, 1.0, 0.5], dtype=np.float32)
        sound = _make_loaded_sound(mono_22k, sample_rate=22050)

        prepared = backend._prepare_audio_data(sound)

        # Resampled from 4 frames at 22050 to ~8 frames at 44100, then upmixed to stereo
        assert prepared.shape[1] == 2
        assert prepared.shape[0] == 8  # 4 * (44100/22050) = 8
        # Both channels should be identical (upmixed from mono)
        np.testing.assert_allclose(prepared[:, 0], prepared[:, 1])

    def test_resample_and_downmix_stereo_48k_to_mono_44k(self) -> None:
        """Stereo 48000 Hz input should be resampled to 44100 Hz and downmixed to mono."""
        backend = _make_backend_for_tests(num_channels=1)
        backend._sample_rate = 44100

        # 48 frames of stereo at 48000 Hz (left=1.0, right=0.0)
        stereo_48k = np.zeros((48, 2), dtype=np.float32)
        stereo_48k[:, 0] = 1.0
        sound = _make_loaded_sound(stereo_48k, sample_rate=48000)

        prepared = backend._prepare_audio_data(sound)

        # Resampled: 48 * (44100/48000) = 44.1 -> 44 frames, downmixed to mono
        assert prepared.shape[1] == 1
        assert prepared.shape[0] == 44
        # Downmixed mono should be mean of left(1.0) and right(0.0) = 0.5
        np.testing.assert_allclose(prepared[:, 0], 0.5, atol=1e-5)

    def test_same_rate_same_channels_is_passthrough(self) -> None:
        """When sample rate and channels already match, data passes through unchanged."""
        backend = _make_backend_for_tests(num_channels=2)
        backend._sample_rate = 44100

        stereo_44k = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        sound = _make_loaded_sound(stereo_44k, sample_rate=44100)

        prepared = backend._prepare_audio_data(sound)

        np.testing.assert_allclose(prepared, stereo_44k)


class TestChannelSelectionAndLifecycle:
    """Tests for voice stealing and channel state transitions."""

    def test_get_channel_steals_lowest_priority_channel(self) -> None:
        backend = MiniaudioBackend(max_channels=3)
        backend._initialized = True
        backend._channels = [MiniaudioChannel(backend) for _ in range(3)]
        sample = np.ones((2, 2), dtype=np.float32)

        with backend._lock:
            backend._channels[0]._play_prepared_locked(sample, 1.0, False, priority=3)
            backend._channels[1]._play_prepared_locked(sample, 1.0, False, priority=1)
            backend._channels[2]._play_prepared_locked(sample, 1.0, False, priority=7)

        stolen = backend.get_channel(priority=5)

        assert stolen is backend._channels[1]
        assert not backend._channels[1].is_playing()

    def test_get_channel_returns_none_when_no_lower_priority_exists(self) -> None:
        backend = MiniaudioBackend(max_channels=2)
        backend._initialized = True
        backend._channels = [MiniaudioChannel(backend) for _ in range(2)]
        sample = np.ones((2, 2), dtype=np.float32)

        with backend._lock:
            backend._channels[0]._play_prepared_locked(sample, 1.0, False, priority=5)
            backend._channels[1]._play_prepared_locked(sample, 1.0, False, priority=8)

        assert backend.get_channel(priority=5) is None

    def test_play_sound_sets_playing_state_and_set_volume_zero_stops(self) -> None:
        backend = MiniaudioBackend(max_channels=1)
        backend._initialized = True
        backend._channels = [MiniaudioChannel(backend)]
        sound = _make_loaded_sound(np.array([[0.4, 0.4], [0.6, 0.6]], dtype=np.float32))

        channel = backend.play_sound(sound, volume=1.0, loop=False, priority=4)

        assert channel is backend._channels[0]
        assert channel is not None
        assert channel.is_playing()
        assert channel.current_priority == 4

        channel.stop()
        assert not channel.is_playing()

        channel.play(sound, volume=1.0, loop=False, priority=4)
        channel.set_volume(0.0)
        assert not channel.is_playing()

    def test_stop_all_sounds_stops_every_active_channel(self) -> None:
        backend = MiniaudioBackend(max_channels=3)
        backend._initialized = True
        backend._channels = [MiniaudioChannel(backend) for _ in range(3)]
        sample = np.ones((4, 2), dtype=np.float32)

        with backend._lock:
            for ch in backend._channels:
                ch._play_prepared_locked(sample, 1.0, True, 5)

        assert backend.get_active_channel_count() == 3

        backend.stop_all_sounds()

        assert backend.get_active_channel_count() == 0
        for ch in backend._channels:
            assert not ch.is_playing()

    def test_shutdown_on_uninitialized_backend_is_safe(self) -> None:
        """Calling shutdown() on a backend that was never initialized should be a no-op."""
        backend = MiniaudioBackend()
        backend.shutdown()  # Should not raise

    def test_shutdown_clears_channels(self) -> None:
        """After shutdown, the channel list should be empty and initialized should be False."""
        backend = MiniaudioBackend(max_channels=2)
        backend._initialized = True
        backend._channels = [MiniaudioChannel(backend) for _ in range(2)]

        backend.shutdown()

        assert not backend._initialized
        assert len(backend._channels) == 0


def _write_wav(
    path: Path,
    num_frames: int = 100,
    nchannels: int = 1,
    sample_rate: int = 44100,
) -> None:
    """Write a minimal PCM16 WAV file with a sine wave."""
    samples_per_channel = num_frames
    total_samples = samples_per_channel * nchannels
    data_size = total_samples * 2  # 16-bit = 2 bytes per sample

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", nchannels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * nchannels * 2))  # byte rate
    buf.write(struct.pack("<H", nchannels * 2))  # block align
    buf.write(struct.pack("<H", 16))  # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))

    import math

    for i in range(samples_per_channel):
        value = int(16000 * math.sin(2 * math.pi * 440 * i / sample_rate))
        for _ in range(nchannels):
            buf.write(struct.pack("<h", value))

    path.write_bytes(buf.getvalue())


class TestLoadSound:
    """Tests for loading audio files via miniaudio decoders.

    These tests write real WAV files to disk and load them through the same
    code path used at runtime, catching issues like invalid decode parameters.
    """

    def test_load_sound_mono_wav(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "mono.wav"
        _write_wav(wav_path, num_frames=200, nchannels=1, sample_rate=22050)

        backend = _make_backend_for_tests(num_channels=2)
        sound = backend.load_sound(wav_path)

        assert sound.channels == 1
        assert sound.sample_rate == 22050
        assert sound.data.shape == (200,)
        assert sound.data.dtype == np.float32

    def test_load_sound_stereo_wav(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "stereo.wav"
        _write_wav(wav_path, num_frames=150, nchannels=2, sample_rate=44100)

        backend = _make_backend_for_tests(num_channels=2)
        sound = backend.load_sound(wav_path)

        assert sound.channels == 2
        assert sound.sample_rate == 44100
        assert sound.data.shape == (150, 2)
        assert sound.data.dtype == np.float32

    def test_load_sound_nonexistent_file_raises(self, tmp_path: Path) -> None:
        backend = _make_backend_for_tests(num_channels=2)

        with pytest.raises(OSError, match="Failed to load"):
            backend.load_sound(tmp_path / "does_not_exist.wav")


class TestAudioLoaderLoad:
    """Tests for AudioLoader.load using miniaudio decoders."""

    def test_loader_loads_mono_wav(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "mono.wav"
        _write_wav(wav_path, num_frames=80, nchannels=1, sample_rate=48000)

        loader = AudioLoader()
        sound = loader.load(wav_path)

        assert sound.channels == 1
        assert sound.sample_rate == 48000
        assert sound.data.shape == (80,)
        assert sound.data.dtype == np.float32

    def test_loader_loads_stereo_wav(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "stereo.wav"
        _write_wav(wav_path, num_frames=60, nchannels=2, sample_rate=44100)

        loader = AudioLoader()
        sound = loader.load(wav_path)

        assert sound.channels == 2
        assert sound.sample_rate == 44100
        assert sound.data.shape == (60, 2)

    def test_loader_caches_loaded_sound(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "cached.wav"
        _write_wav(wav_path, num_frames=50, nchannels=1, sample_rate=44100)

        loader = AudioLoader()
        first = loader.load(wav_path)
        second = loader.load(wav_path)

        assert first is second

    def test_loader_nonexistent_file_raises(self, tmp_path: Path) -> None:
        loader = AudioLoader()

        with pytest.raises(OSError, match="Failed to load"):
            loader.load(tmp_path / "nope.wav")

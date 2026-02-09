"""Unit tests for the sounddevice audio backend internals."""

from __future__ import annotations

import numpy as np

from brileta.backends.sounddevice_audio import (
    SoundDeviceAudioBackend,
    SoundDeviceAudioChannel,
)
from brileta.sound.audio_backend import LoadedSound


def _make_loaded_sound(data: np.ndarray, sample_rate: int = 44100) -> LoadedSound:
    """Build a LoadedSound for tests from a numpy array."""
    channels = 1 if data.ndim == 1 else data.shape[1]
    return LoadedSound(data=data, sample_rate=sample_rate, channels=channels)


def _make_backend_for_tests(num_channels: int = 2) -> SoundDeviceAudioBackend:
    """Create a backend instance without opening PortAudio streams."""
    backend = SoundDeviceAudioBackend()
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

        resampled = SoundDeviceAudioBackend._resample_audio(
            source, source_sample_rate=4, target_sample_rate=8
        )

        assert resampled.shape == (8, 1)
        assert resampled.dtype == np.float32
        assert resampled[0, 0] == 0.0
        assert resampled[-1, 0] == 3.0

    def test_resample_audio_handles_zero_frame_input(self) -> None:
        empty = np.zeros((0, 2), dtype=np.float32)

        resampled = SoundDeviceAudioBackend._resample_audio(
            empty, source_sample_rate=22050, target_sample_rate=44100
        )

        assert resampled.shape == (0, 2)
        assert resampled.dtype == np.float32


class TestChannelMixing:
    """Tests for channel mix behavior without a live audio device."""

    def test_mix_into_applies_channel_and_master_gain(self) -> None:
        backend = _make_backend_for_tests(num_channels=2)
        channel = SoundDeviceAudioChannel(backend)
        sound = _make_loaded_sound(np.ones((2, 2), dtype=np.float32))
        channel.play(sound, volume=0.5, loop=False, priority=2)
        out = np.zeros((2, 2), dtype=np.float32)

        with backend._lock:
            channel.mix_into(out, master_volume=0.5)

        np.testing.assert_allclose(out, np.full((2, 2), 0.25, dtype=np.float32))

    def test_mix_into_wraps_when_looping(self) -> None:
        backend = _make_backend_for_tests(num_channels=2)
        channel = SoundDeviceAudioChannel(backend)
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
        channel = SoundDeviceAudioChannel(backend)
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
        channel = SoundDeviceAudioChannel(backend)
        out = np.full((4, 2), 0.75, dtype=np.float32)

        with backend._lock:
            channel.mix_into(out, master_volume=1.0)

        np.testing.assert_allclose(out, np.full((4, 2), 0.75, dtype=np.float32))


class TestChannelSelectionAndLifecycle:
    """Tests for voice stealing and channel state transitions."""

    def test_get_channel_steals_lowest_priority_channel(self) -> None:
        backend = SoundDeviceAudioBackend(max_channels=3)
        backend._initialized = True
        backend._channels = [SoundDeviceAudioChannel(backend) for _ in range(3)]
        sample = np.ones((2, 2), dtype=np.float32)

        with backend._lock:
            backend._channels[0]._play_prepared_locked(sample, 1.0, False, priority=3)
            backend._channels[1]._play_prepared_locked(sample, 1.0, False, priority=1)
            backend._channels[2]._play_prepared_locked(sample, 1.0, False, priority=7)

        stolen = backend.get_channel(priority=5)

        assert stolen is backend._channels[1]
        assert not backend._channels[1].is_playing()

    def test_get_channel_returns_none_when_no_lower_priority_exists(self) -> None:
        backend = SoundDeviceAudioBackend(max_channels=2)
        backend._initialized = True
        backend._channels = [SoundDeviceAudioChannel(backend) for _ in range(2)]
        sample = np.ones((2, 2), dtype=np.float32)

        with backend._lock:
            backend._channels[0]._play_prepared_locked(sample, 1.0, False, priority=5)
            backend._channels[1]._play_prepared_locked(sample, 1.0, False, priority=8)

        assert backend.get_channel(priority=5) is None

    def test_play_sound_sets_playing_state_and_set_volume_zero_stops(self) -> None:
        backend = SoundDeviceAudioBackend(max_channels=1)
        backend._initialized = True
        backend._channels = [SoundDeviceAudioChannel(backend)]
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

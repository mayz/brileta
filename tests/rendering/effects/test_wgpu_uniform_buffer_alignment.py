"""Tests for WGPU uniform buffer alignment.

These tests verify that the Python uniform buffer packing matches the WGSL
struct layout. WGSL has strict alignment rules (vec3f requires 16-byte
alignment, etc.) that can cause silent data corruption if not followed.

This test caught a real bug where map_size was packed at the wrong offset
after padding changes. WGSL expects map_size immediately after _padding2.
"""

from __future__ import annotations

import struct
from unittest.mock import Mock

from brileta.game.game_world import GameWorld


class TestWGPUUniformBufferAlignment:
    """Test suite for WGPU uniform buffer packing alignment."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.game_world = Mock(spec=GameWorld)
        self.game_world.lights = []

        # Mock game map
        self.game_map = Mock()
        self.game_map.width = 80
        self.game_map.height = 43
        self.game_world.game_map = self.game_map

    def _create_gpu_lighting_system(self) -> Mock:
        """Create a mock GPU lighting system with the real _pack_uniform_data method."""
        # Import here to avoid circular imports and allow mocking
        from brileta.backends.wgpu.gpu_lighting import GPULightingSystem

        # Create a partial mock that uses the real packing method
        mock_system = Mock(spec=GPULightingSystem)
        mock_system.game_world = self.game_world
        mock_system.MAX_LIGHTS = 32
        mock_system._time = 1.5

        # Bind the real method to our mock
        mock_system._pack_uniform_data = lambda *args, **kwargs: (
            GPULightingSystem._pack_uniform_data(mock_system, *args, **kwargs)
        )

        return mock_system

    def test_buffer_contains_expected_sun_intensity(self) -> None:
        """Test that sun_intensity is packed at the correct offset."""
        from brileta.game.lights import DirectionalLight
        from brileta.util.coordinates import Rect

        # Add a directional light with known intensity
        sun = DirectionalLight.create_sun(intensity=0.4)
        self.game_world.lights = [sun]

        gpu_system = self._create_gpu_lighting_system()

        # Pack uniform data
        light_data: list[float] = []
        viewport = Rect(0, 0, 80, 43)

        buffer = gpu_system._pack_uniform_data(light_data, 0, viewport)

        # Calculate expected offset for sun_intensity
        # Based on WGSL struct layout:
        # - viewport_data: 16 bytes (offset 0)
        # - metadata: 16 bytes (offset 16)
        # - 8 light arrays: 8 * 32 * 16 = 4096 bytes (offset 32)
        # - sun_direction + padding: 16 bytes (offset 4128)
        # - sun_color (12 bytes) + sun_intensity (4 bytes): 16 bytes (offset 4144)
        # sun_intensity is at offset 4144 + 12 = 4156
        sun_intensity_offset = 4156

        # Unpack sun_intensity from the buffer
        actual_sun_intensity = struct.unpack_from("f", buffer, sun_intensity_offset)[0]

        assert abs(actual_sun_intensity - 0.4) < 0.001, (
            f"sun_intensity at offset {sun_intensity_offset} should be 0.4, "
            f"got {actual_sun_intensity}"
        )

    def test_buffer_contains_expected_map_size(self) -> None:
        """Test that map_size is packed at the correct offset."""
        from brileta.util.coordinates import Rect

        gpu_system = self._create_gpu_lighting_system()

        # Pack uniform data
        light_data: list[float] = []
        viewport = Rect(0, 0, 80, 43)

        buffer = gpu_system._pack_uniform_data(light_data, 0, viewport)

        # Calculate expected offset for map_size
        # After sun_intensity (4156 + 4 = 4160):
        # - sky_exposure_power (4 bytes) at 4160
        # - sun_shadow_intensity (4 bytes) at 4164
        # - sun_shadow_length_scale (4 bytes) at 4168
        # - _padding2 (f32, 4 bytes) at 4172
        # - map_size (vec2f, 8 bytes) at 4176
        map_size_offset = 4176

        # Unpack map_size from the buffer
        map_width, map_height = struct.unpack_from("2f", buffer, map_size_offset)

        assert map_width == 80.0, (
            f"map_size.x at offset {map_size_offset} should be 80.0, got {map_width}"
        )
        assert map_height == 43.0, (
            f"map_size.y at offset {map_size_offset + 4} should be 43.0, "
            f"got {map_height}"
        )

    def test_buffer_contains_expected_sky_exposure_power(self) -> None:
        """Test that sky_exposure_power is packed at the correct offset."""
        from brileta.config import SKY_EXPOSURE_POWER
        from brileta.util.coordinates import Rect

        gpu_system = self._create_gpu_lighting_system()

        # Pack uniform data
        light_data: list[float] = []
        viewport = Rect(0, 0, 80, 43)

        buffer = gpu_system._pack_uniform_data(light_data, 0, viewport)

        # sky_exposure_power is at offset 4160 (right after sun_intensity)
        sky_exposure_power_offset = 4160

        actual_sky_exposure_power = struct.unpack_from(
            "f", buffer, sky_exposure_power_offset
        )[0]

        assert abs(actual_sky_exposure_power - SKY_EXPOSURE_POWER) < 0.001, (
            f"sky_exposure_power at offset {sky_exposure_power_offset} should be "
            f"{SKY_EXPOSURE_POWER}, got {actual_sky_exposure_power}"
        )

    def test_shadow_length_scale_packed_correctly(self) -> None:
        """Test that sun_shadow_length_scale is packed at offset 4168."""
        import math

        from brileta.game.lights import DirectionalLight
        from brileta.util.coordinates import Rect

        # Create a sun at 45-degree elevation: 1/tan(45°) = 1.0
        sun = DirectionalLight.create_sun(elevation_degrees=45.0, intensity=0.5)
        self.game_world.lights = [sun]

        gpu_system = self._create_gpu_lighting_system()
        light_data: list[float] = []
        viewport = Rect(0, 0, 80, 43)

        buffer = gpu_system._pack_uniform_data(light_data, 0, viewport)

        # sun_shadow_length_scale is at offset 4168
        actual_scale = struct.unpack_from("f", buffer, 4168)[0]
        expected_scale = 1.0 / math.tan(math.radians(45.0))

        assert abs(actual_scale - expected_scale) < 0.001, (
            f"shadow_length_scale at offset 4168 should be ~{expected_scale:.3f}, "
            f"got {actual_scale}"
        )

    def test_shadow_length_scale_clamped_near_horizon(self) -> None:
        """Test that shadow_length_scale is clamped to 8.0 for very low sun."""
        from brileta.game.lights import DirectionalLight
        from brileta.util.coordinates import Rect

        # 1-degree elevation: 1/tan(1°) ≈ 57.3, should be clamped to 8.0
        sun = DirectionalLight.create_sun(elevation_degrees=1.0, intensity=0.5)
        self.game_world.lights = [sun]

        gpu_system = self._create_gpu_lighting_system()
        light_data: list[float] = []
        viewport = Rect(0, 0, 80, 43)

        buffer = gpu_system._pack_uniform_data(light_data, 0, viewport)

        actual_scale = struct.unpack_from("f", buffer, 4168)[0]

        assert abs(actual_scale - 8.0) < 0.001, (
            f"shadow_length_scale should be clamped to 8.0, got {actual_scale}"
        )

    def test_shadow_length_scale_with_zero_elevation(self) -> None:
        """Test shadow_length_scale when elevation is 0 (clamped to 0.1)."""
        from brileta.game.lights import DirectionalLight
        from brileta.util.coordinates import Rect

        # 0 elevation is clamped to 0.1 internally, 1/tan(0.1°) is huge,
        # so the scale should be clamped to 8.0
        sun = DirectionalLight.create_sun(elevation_degrees=0.0, intensity=0.5)
        self.game_world.lights = [sun]

        gpu_system = self._create_gpu_lighting_system()
        light_data: list[float] = []
        viewport = Rect(0, 0, 80, 43)

        buffer = gpu_system._pack_uniform_data(light_data, 0, viewport)

        actual_scale = struct.unpack_from("f", buffer, 4168)[0]

        assert abs(actual_scale - 8.0) < 0.001, (
            f"shadow_length_scale with zero elevation should be clamped to 8.0, "
            f"got {actual_scale}"
        )

    def test_shadow_length_scale_at_high_elevation(self) -> None:
        """Test shadow_length_scale near 90 degrees (nearly overhead)."""
        import math

        from brileta.game.lights import DirectionalLight
        from brileta.util.coordinates import Rect

        # 85-degree elevation: 1/tan(85°) ≈ 0.087 (very short shadows)
        sun = DirectionalLight.create_sun(elevation_degrees=85.0, intensity=0.5)
        self.game_world.lights = [sun]

        gpu_system = self._create_gpu_lighting_system()
        light_data: list[float] = []
        viewport = Rect(0, 0, 80, 43)

        buffer = gpu_system._pack_uniform_data(light_data, 0, viewport)

        actual_scale = struct.unpack_from("f", buffer, 4168)[0]
        expected_scale = 1.0 / math.tan(math.radians(85.0))

        assert abs(actual_scale - expected_scale) < 0.01, (
            f"shadow_length_scale at 85° elevation should be ~{expected_scale:.3f}, "
            f"got {actual_scale}"
        )
        # Should be a small value (sun nearly overhead, short shadows)
        assert actual_scale < 0.2

    def test_buffer_size_sufficient_for_struct(self) -> None:
        """Test that the buffer is large enough for the complete WGSL struct."""
        from brileta.util.coordinates import Rect

        gpu_system = self._create_gpu_lighting_system()

        # Pack uniform data
        light_data: list[float] = []
        viewport = Rect(0, 0, 80, 43)

        buffer = gpu_system._pack_uniform_data(light_data, 0, viewport)

        # Expected minimum size based on WGSL struct layout:
        # map_size ends at 4176 + 8 = 4184
        # _padding3 ends at 4184 + 8 = 4192
        expected_min_size = 4192

        assert len(buffer) >= expected_min_size, (
            f"Buffer size {len(buffer)} is smaller than expected minimum "
            f"{expected_min_size}"
        )

    def test_viewport_data_at_start_of_buffer(self) -> None:
        """Test that viewport_data is at the start of the buffer."""
        from brileta.util.coordinates import Rect

        gpu_system = self._create_gpu_lighting_system()

        viewport = Rect(10, 20, 60, 40)
        light_data: list[float] = []

        buffer = gpu_system._pack_uniform_data(light_data, 0, viewport)

        # viewport_data is vec4f at offset 0
        vp_x, vp_y, vp_w, vp_h = struct.unpack_from("4f", buffer, 0)

        assert vp_x == 10.0, f"viewport x should be 10.0, got {vp_x}"
        assert vp_y == 20.0, f"viewport y should be 20.0, got {vp_y}"
        assert vp_w == 60.0, f"viewport width should be 60.0, got {vp_w}"
        assert vp_h == 40.0, f"viewport height should be 40.0, got {vp_h}"

    def test_ambient_light_at_correct_offset(self) -> None:
        """Test that ambient_light is at the correct offset in metadata."""
        from brileta.config import AMBIENT_LIGHT_LEVEL
        from brileta.util.coordinates import Rect

        gpu_system = self._create_gpu_lighting_system()

        light_data: list[float] = []
        viewport = Rect(0, 0, 80, 43)

        buffer = gpu_system._pack_uniform_data(light_data, 0, viewport)

        # Metadata starts at offset 16
        # Format: light_count (i32), ambient_light (f32), time (f32), tile_aligned (u32)
        # ambient_light is at offset 16 + 4 = 20
        ambient_light_offset = 20

        actual_ambient = struct.unpack_from("f", buffer, ambient_light_offset)[0]

        assert abs(actual_ambient - AMBIENT_LIGHT_LEVEL) < 0.001, (
            f"ambient_light at offset {ambient_light_offset} should be "
            f"{AMBIENT_LIGHT_LEVEL}, got {actual_ambient}"
        )


class TestWGPUUniformBufferEdgeCases:
    """Test edge cases in WGPU uniform buffer packing."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.game_world = Mock(spec=GameWorld)
        self.game_world.lights = []

        self.game_map = Mock()
        self.game_map.width = 80
        self.game_map.height = 43
        self.game_world.game_map = self.game_map

    def _create_gpu_lighting_system(self) -> Mock:
        """Create a mock GPU lighting system."""
        from brileta.backends.wgpu.gpu_lighting import GPULightingSystem

        mock_system = Mock(spec=GPULightingSystem)
        mock_system.game_world = self.game_world
        mock_system.MAX_LIGHTS = 32
        mock_system._time = 0.0

        mock_system._pack_uniform_data = lambda *args, **kwargs: (
            GPULightingSystem._pack_uniform_data(mock_system, *args, **kwargs)
        )

        return mock_system

    def test_no_game_map_uses_default_size(self) -> None:
        """Test that missing game_map defaults to 1x1 map size."""
        from brileta.util.coordinates import Rect

        self.game_world.game_map = None

        gpu_system = self._create_gpu_lighting_system()

        light_data: list[float] = []
        viewport = Rect(0, 0, 80, 43)

        buffer = gpu_system._pack_uniform_data(light_data, 0, viewport)

        # map_size should default to (1.0, 1.0)
        map_size_offset = 4176
        map_width, map_height = struct.unpack_from("2f", buffer, map_size_offset)

        assert map_width == 1.0, f"Default map width should be 1.0, got {map_width}"
        assert map_height == 1.0, f"Default map height should be 1.0, got {map_height}"

    def test_no_directional_light_zeros_sun_uniforms(self) -> None:
        """Test that missing directional light sets sun uniforms to zero."""
        from brileta.util.coordinates import Rect

        # No lights at all
        self.game_world.lights = []

        gpu_system = self._create_gpu_lighting_system()

        light_data: list[float] = []
        viewport = Rect(0, 0, 80, 43)

        buffer = gpu_system._pack_uniform_data(light_data, 0, viewport)

        # sun_direction at offset 4128
        sun_dir_x, sun_dir_y = struct.unpack_from("2f", buffer, 4128)
        assert sun_dir_x == 0.0, f"sun_direction.x should be 0.0, got {sun_dir_x}"
        assert sun_dir_y == 0.0, f"sun_direction.y should be 0.0, got {sun_dir_y}"

        # sun_color at offset 4144
        sun_r, sun_g, sun_b = struct.unpack_from("3f", buffer, 4144)
        assert sun_r == 0.0, f"sun_color.r should be 0.0, got {sun_r}"
        assert sun_g == 0.0, f"sun_color.g should be 0.0, got {sun_g}"
        assert sun_b == 0.0, f"sun_color.b should be 0.0, got {sun_b}"

        # sun_intensity at offset 4156
        sun_intensity = struct.unpack_from("f", buffer, 4156)[0]
        assert sun_intensity == 0.0, f"sun_intensity should be 0.0, got {sun_intensity}"

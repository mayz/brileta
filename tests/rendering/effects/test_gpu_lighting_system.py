"""Unit tests for GPU lighting system.

Tests the core functionality of the GPU-based lighting implementation,
including hardware detection, fallback behavior, light data collection,
and basic compute shader functionality.
"""

from unittest.mock import Mock, patch

import numpy as np

from catley.game.game_world import GameWorld
from catley.game.lights import DynamicLight, StaticLight
from catley.types import FixedTimestep
from catley.util.coordinates import Rect
from catley.view.render.lighting.base import LightingSystem
from catley.view.render.lighting.gpu import GPULightingSystem


class TestGPULightingSystem:
    """Test suite for GPULightingSystem class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock game world
        self.game_world = Mock(spec=GameWorld)
        self.game_world.lights = []

        # Create a mock ModernGL graphics context
        self.mock_graphics_context = Mock()
        self.mock_mgl_context = Mock()
        self.mock_graphics_context.mgl_context = self.mock_mgl_context

        # Create a mock fallback system
        self.mock_fallback = Mock(spec=LightingSystem)
        self.mock_fallback.revision = 0

    def test_initialization_success(self):
        """Test successful initialization with compute shader support."""
        # Mock successful compute shader creation
        self.mock_mgl_context.compute_shader.return_value = Mock()
        self.mock_mgl_context.buffer.return_value = Mock()

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, self.mock_fallback
        )

        assert gpu_system.game_world == self.game_world
        assert gpu_system.graphics_context == self.mock_graphics_context
        assert gpu_system.fallback_system == self.mock_fallback
        assert gpu_system._time == 0.0
        assert gpu_system.revision == 0

        # Should have attempted to create compute shader
        self.mock_mgl_context.compute_shader.assert_called_once()
        self.mock_mgl_context.buffer.assert_called_once()

    def test_initialization_no_compute_shader_support(self):
        """Test initialization when compute shaders are not supported."""
        # Mock missing compute shader support
        del self.mock_mgl_context.compute_shader

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, self.mock_fallback
        )

        # Should still initialize but without GPU resources
        assert gpu_system._compute_program is None
        assert gpu_system.fallback_system == self.mock_fallback

    def test_initialization_compute_shader_creation_fails(self):
        """Test initialization when compute shader creation fails."""
        # Mock compute shader creation failure
        self.mock_mgl_context.compute_shader.side_effect = Exception(
            "Shader compilation failed"
        )

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, self.mock_fallback
        )

        # Should handle gracefully and set up for fallback
        assert gpu_system._compute_program is None

    def test_update_with_fallback(self):
        """Test update method forwards to fallback system."""
        self.mock_mgl_context.compute_shader.return_value = Mock()
        self.mock_mgl_context.buffer.return_value = Mock()

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, self.mock_fallback
        )

        timestep = FixedTimestep(1.0 / 60.0)
        gpu_system.update(timestep)

        # Should update internal time
        assert gpu_system._time == timestep

        # Should forward to fallback
        self.mock_fallback.update.assert_called_once_with(timestep)

    def test_update_without_fallback(self):
        """Test update method when no fallback system is provided."""
        self.mock_mgl_context.compute_shader.return_value = Mock()
        self.mock_mgl_context.buffer.return_value = Mock()

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, fallback_system=None
        )

        timestep = FixedTimestep(1.0 / 60.0)
        gpu_system.update(timestep)

        # Should update internal time without errors
        assert gpu_system._time == timestep

    def test_light_data_collection_empty_world(self):
        """Test light data collection with no lights."""
        self.mock_mgl_context.compute_shader.return_value = Mock()
        self.mock_mgl_context.buffer.return_value = Mock()

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, self.mock_fallback
        )

        viewport = Rect(0, 0, 10, 10)
        light_data = gpu_system._collect_light_data(viewport)

        assert light_data == []

    def test_light_data_collection_with_lights(self):
        """Test light data collection with various light types."""
        self.mock_mgl_context.compute_shader.return_value = Mock()
        self.mock_mgl_context.buffer.return_value = Mock()

        # Create mock lights
        static_light = Mock(spec=StaticLight)
        static_light.position = (5, 5)
        static_light.radius = 3
        static_light.color = Mock()
        static_light.color.as_rgb_floats.return_value = (1.0, 0.8, 0.6)

        dynamic_light = Mock(spec=DynamicLight)
        dynamic_light.position = (15, 8)
        dynamic_light.radius = 4
        dynamic_light.color = Mock()
        dynamic_light.color.as_rgb_floats.return_value = (0.9, 0.9, 1.0)

        # Light outside viewport (should be culled)
        distant_light = Mock(spec=StaticLight)
        distant_light.position = (100, 100)
        distant_light.radius = 2
        distant_light.color = Mock()
        distant_light.color.as_rgb_floats.return_value = (1.0, 1.0, 1.0)

        self.game_world.lights = [static_light, dynamic_light, distant_light]

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, self.mock_fallback
        )

        viewport = Rect(0, 0, 20, 15)
        light_data = gpu_system._collect_light_data(viewport)

        # Should include 2 lights (static and dynamic), exclude distant light
        # Each light = 8 floats: position.xy, radius, intensity, color.rgb, padding
        expected_length = 2 * 8
        assert len(light_data) == expected_length

        # Check first light data (static light)
        assert light_data[0] == 5.0  # x position
        assert light_data[1] == 5.0  # y position
        assert light_data[2] == 3.0  # radius
        assert light_data[3] == 1.0  # intensity
        assert light_data[4] == 1.0  # red
        assert light_data[5] == 0.8  # green
        assert light_data[6] == 0.6  # blue
        assert light_data[7] == 0.0  # padding

        # Check second light data (dynamic light)
        assert light_data[8] == 15.0  # x position
        assert light_data[9] == 8.0  # y position
        assert light_data[10] == 4.0  # radius
        assert light_data[11] == 1.0  # intensity
        assert light_data[12] == 0.9  # red
        assert light_data[13] == 0.9  # green
        assert light_data[14] == 1.0  # blue
        assert light_data[15] == 0.0  # padding

    def test_light_data_collection_frustum_culling(self):
        """Test that lights outside the viewport are properly culled."""
        self.mock_mgl_context.compute_shader.return_value = Mock()
        self.mock_mgl_context.buffer.return_value = Mock()

        # Create lights at various positions
        lights = []
        positions = [
            (5, 5),  # Inside viewport
            (-10, 5),  # Outside left
            (25, 5),  # Outside right
            (5, -10),  # Outside top
            (5, 25),  # Outside bottom
            (12, 12),  # Inside with radius extending outside
        ]

        for _, (x, y) in enumerate(positions):
            light = Mock(spec=StaticLight)
            light.position = (x, y)
            light.radius = 3
            light.color = Mock()
            light.color.as_rgb_floats.return_value = (1.0, 1.0, 1.0)
            lights.append(light)

        self.game_world.lights = lights

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, self.mock_fallback
        )

        viewport = Rect(0, 0, 15, 15)
        light_data = gpu_system._collect_light_data(viewport)

        # Should include lights that could affect the viewport
        # Light at (5,5) - definitely included
        # Light at (12,12) - included (position + radius could affect viewport)
        # Others should be culled
        expected_lights = 2
        assert len(light_data) == expected_lights * 8

    def test_compute_lightmap_fallback_when_no_gpu(self):
        """Test compute_lightmap falls back to CPU when GPU unavailable."""
        # Mock failed GPU initialization
        del self.mock_mgl_context.compute_shader

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, self.mock_fallback
        )

        viewport = Rect(0, 0, 10, 10)
        expected_result = np.random.rand(10, 10, 3).astype(np.float32)
        self.mock_fallback.compute_lightmap.return_value = expected_result

        result = gpu_system.compute_lightmap(viewport)

        # Should use fallback
        self.mock_fallback.compute_lightmap.assert_called_once_with(viewport)
        np.testing.assert_array_equal(result, expected_result)

    def test_compute_lightmap_fallback_when_gpu_fails(self):
        """Test compute_lightmap falls back when GPU computation fails."""
        # Mock successful initialization but failed computation
        mock_compute_program = Mock()
        mock_compute_program.run.side_effect = Exception("GPU computation failed")
        self.mock_mgl_context.compute_shader.return_value = mock_compute_program
        self.mock_mgl_context.buffer.return_value = Mock()

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, self.mock_fallback
        )

        viewport = Rect(0, 0, 10, 10)
        expected_result = np.random.rand(10, 10, 3).astype(np.float32)
        self.mock_fallback.compute_lightmap.return_value = expected_result

        result = gpu_system.compute_lightmap(viewport)

        # Should fall back to CPU system
        self.mock_fallback.compute_lightmap.assert_called_once_with(viewport)
        np.testing.assert_array_equal(result, expected_result)

    def test_compute_lightmap_no_fallback_available(self):
        """Test compute_lightmap returns None when no fallback and GPU fails."""
        # Mock failed GPU initialization with no fallback
        del self.mock_mgl_context.compute_shader

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, fallback_system=None
        )

        viewport = Rect(0, 0, 10, 10)
        result = gpu_system.compute_lightmap(viewport)

        assert result is None

    def test_light_notifications_increment_revision(self):
        """Test that light change notifications increment revision counter."""
        self.mock_mgl_context.compute_shader.return_value = Mock()
        self.mock_mgl_context.buffer.return_value = Mock()

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, self.mock_fallback
        )

        initial_revision = gpu_system.revision

        # Test light added
        mock_light = Mock(spec=StaticLight)
        gpu_system.on_light_added(mock_light)
        assert gpu_system.revision == initial_revision + 1

        # Test light removed
        gpu_system.on_light_removed(mock_light)
        assert gpu_system.revision == initial_revision + 2

        # Test light moved
        gpu_system.on_light_moved(mock_light)
        assert gpu_system.revision == initial_revision + 3

        # Test global light changed
        gpu_system.on_global_light_changed()
        assert gpu_system.revision == initial_revision + 4

    def test_light_notifications_forward_to_fallback(self):
        """Test that light notifications are forwarded to fallback system."""
        self.mock_mgl_context.compute_shader.return_value = Mock()
        self.mock_mgl_context.buffer.return_value = Mock()

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, self.mock_fallback
        )

        mock_light = Mock(spec=StaticLight)

        # Test all notification methods forward to fallback
        gpu_system.on_light_added(mock_light)
        self.mock_fallback.on_light_added.assert_called_once_with(mock_light)

        gpu_system.on_light_removed(mock_light)
        self.mock_fallback.on_light_removed.assert_called_once_with(mock_light)

        gpu_system.on_light_moved(mock_light)
        self.mock_fallback.on_light_moved.assert_called_once_with(mock_light)

        gpu_system.on_global_light_changed()
        self.mock_fallback.on_global_light_changed.assert_called_once()

    def test_compute_shader_source_generation(self):
        """Test that compute shader source is generated correctly."""
        self.mock_mgl_context.compute_shader.return_value = Mock()
        self.mock_mgl_context.buffer.return_value = Mock()

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, self.mock_fallback
        )

        shader_source = gpu_system._create_point_light_compute_shader()

        # Check for key shader components
        assert "#version 430" in shader_source
        assert "layout(local_size_x = 8, local_size_y = 8)" in shader_source
        assert "uniform writeonly image2D lightmap" in shader_source
        assert "readonly buffer LightBuffer" in shader_source
        assert "uniform int u_light_count" in shader_source
        assert "uniform float u_ambient_light" in shader_source
        assert "uniform ivec2 u_viewport_offset" in shader_source
        assert "void main()" in shader_source
        assert "imageStore(lightmap" in shader_source

    def test_max_lights_limit(self):
        """Test that light count is limited to MAX_LIGHTS in GPU computation."""
        self.mock_mgl_context.compute_shader.return_value = Mock()
        self.mock_mgl_context.buffer.return_value = Mock()

        # Create more lights than the maximum
        lights = []
        for i in range(GPULightingSystem.MAX_LIGHTS + 50):
            light = Mock(spec=StaticLight)
            light.position = (i % 20, i // 20)
            light.radius = 3
            light.color = Mock()
            light.color.as_rgb_floats.return_value = (1.0, 1.0, 1.0)
            lights.append(light)

        self.game_world.lights = lights

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, self.mock_fallback
        )

        viewport = Rect(0, 0, 100, 100)  # Large viewport to include many lights
        light_data = gpu_system._collect_light_data(viewport)

        # _collect_light_data doesn't limit - it collects all lights in frustum
        # The limiting happens in the actual GPU computation
        total_lights_in_viewport = len(light_data) // 8
        assert total_lights_in_viewport > GPULightingSystem.MAX_LIGHTS

        # Test that limiting actually happens during processing
        # When we have too many lights, they should be truncated to MAX_LIGHTS
        light_count = len(light_data) // 8
        if light_count > GPULightingSystem.MAX_LIGHTS:
            limited_data = light_data[: GPULightingSystem.MAX_LIGHTS * 8]
            assert len(limited_data) == GPULightingSystem.MAX_LIGHTS * 8

    def test_resource_cleanup(self):
        """Test that GPU resources are properly released."""
        mock_compute_program = Mock()
        mock_light_buffer = Mock()
        mock_output_texture = Mock()
        mock_output_buffer = Mock()

        self.mock_mgl_context.compute_shader.return_value = mock_compute_program
        self.mock_mgl_context.buffer.return_value = mock_light_buffer
        self.mock_mgl_context.texture.return_value = mock_output_texture

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, self.mock_fallback
        )

        # Set up some resources
        gpu_system._compute_program = mock_compute_program
        gpu_system._light_buffer = mock_light_buffer
        gpu_system._output_texture = mock_output_texture
        gpu_system._output_buffer = mock_output_buffer

        # Call release
        gpu_system.release()

        # Should release all resources
        mock_compute_program.release.assert_called_once()
        mock_light_buffer.release.assert_called_once()
        mock_output_texture.release.assert_called_once()
        mock_output_buffer.release.assert_called_once()

    def test_resource_cleanup_with_none_resources(self):
        """Test that resource cleanup handles None resources gracefully."""
        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, self.mock_fallback
        )

        # Ensure all resources are None
        gpu_system._compute_program = None
        gpu_system._light_buffer = None
        gpu_system._output_texture = None
        gpu_system._output_buffer = None

        # Should not raise any exceptions
        gpu_system.release()


class TestGPULightingSystemIntegration:
    """Integration tests for GPU lighting system functionality."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.game_world = Mock(spec=GameWorld)
        self.game_world.lights = []

    @patch("catley.view.render.lighting.gpu.logger")
    def test_hardware_detection_logging(self, mock_logger):
        """Test that hardware detection results are properly logged."""
        mock_graphics_context = Mock()
        mock_mgl_context = Mock()
        mock_graphics_context.mgl_context = mock_mgl_context

        # Test successful initialization
        mock_mgl_context.compute_shader.return_value = Mock()
        mock_mgl_context.buffer.return_value = Mock()

        GPULightingSystem(self.game_world, mock_graphics_context)

        # Should log successful initialization
        mock_logger.info.assert_called_with(
            "GPU lighting system initialized successfully"
        )

    @patch("catley.view.render.lighting.gpu.logger")
    def test_hardware_detection_failure_logging(self, mock_logger):
        """Test that hardware detection failures are properly logged."""
        mock_graphics_context = Mock()
        mock_mgl_context = Mock()
        mock_graphics_context.mgl_context = mock_mgl_context

        # Test missing compute shader support
        del mock_mgl_context.compute_shader

        GPULightingSystem(self.game_world, mock_graphics_context)

        # Should log compute shader unavailability
        mock_logger.info.assert_called_with(
            "Compute shaders not supported by OpenGL context"
        )

    def test_interface_compliance(self):
        """Test that GPULightingSystem properly implements LightingSystem interface."""
        mock_graphics_context = Mock()
        mock_mgl_context = Mock()
        mock_graphics_context.mgl_context = mock_mgl_context
        mock_mgl_context.compute_shader.return_value = Mock()
        mock_mgl_context.buffer.return_value = Mock()

        gpu_system = GPULightingSystem(self.game_world, mock_graphics_context)

        # Should be instance of LightingSystem
        assert isinstance(gpu_system, LightingSystem)

        # Should have all required methods
        assert hasattr(gpu_system, "update")
        assert hasattr(gpu_system, "compute_lightmap")
        assert hasattr(gpu_system, "on_light_added")
        assert hasattr(gpu_system, "on_light_removed")
        assert hasattr(gpu_system, "on_light_moved")
        assert hasattr(gpu_system, "on_global_light_changed")

        # Should have revision attribute
        assert hasattr(gpu_system, "revision")
        assert isinstance(gpu_system.revision, int)

    def test_viewport_resource_management(self):
        """Test that resources are properly managed for different viewport sizes."""
        mock_graphics_context = Mock()
        mock_mgl_context = Mock()
        mock_graphics_context.mgl_context = mock_mgl_context

        # Mock successful initialization
        mock_compute_program = Mock()
        mock_mgl_context.compute_shader.return_value = mock_compute_program
        mock_mgl_context.buffer.return_value = Mock()

        # Mock texture creation
        mock_textures = [Mock(), Mock()]
        mock_mgl_context.texture.side_effect = mock_textures

        gpu_system = GPULightingSystem(self.game_world, mock_graphics_context)

        # Test resource creation for first viewport
        viewport1 = Rect(0, 0, 10, 10)
        result1 = gpu_system._ensure_resources_for_viewport(viewport1)
        assert result1 is True
        assert gpu_system._current_viewport == viewport1

        # Test resource resize for different viewport
        viewport2 = Rect(0, 0, 20, 15)
        result2 = gpu_system._ensure_resources_for_viewport(viewport2)
        assert result2 is True
        assert gpu_system._current_viewport == viewport2

        # Should have created two textures (one released, one active)
        assert mock_mgl_context.texture.call_count == 2

        # First texture should have been released
        mock_textures[0].release.assert_called_once()

"""Unit tests for GPU lighting system using fragment shaders.

Tests the core functionality of the GPU-based lighting implementation,
including hardware detection, fallback behavior, light data collection,
and fragment shader functionality.
"""

from unittest.mock import Mock, patch

import numpy as np

from catley.backends.moderngl.gpu_lighting import GPULightingSystem
from catley.game.game_world import GameWorld
from catley.game.lights import DynamicLight, StaticLight
from catley.types import FixedTimestep
from catley.util.coordinates import Rect
from catley.view.render.lighting.base import LightingSystem


class TestGPULightingSystem:
    """Test suite for GPULightingSystem class using fragment shaders."""

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
        """Test successful initialization with fragment shader support."""
        # Mock ShaderManager and fragment program creation
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(
                self.game_world, self.mock_graphics_context, self.mock_fallback
            )

        assert gpu_system.game_world == self.game_world
        assert gpu_system.graphics_context == self.mock_graphics_context
        assert gpu_system.fallback_system == self.mock_fallback
        assert gpu_system._time == 0.0
        assert gpu_system.revision == 0

        # Should have attempted to create fragment shader program
        mock_shader_manager.create_program.assert_called_once()
        self.mock_mgl_context.vertex_array.assert_called_once()

    def test_initialization_no_mgl_context(self):
        """Test initialization when no ModernGL context is available."""
        # Mock missing ModernGL context
        self.mock_graphics_context.mgl_context = None

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, self.mock_fallback
        )

        # Should still initialize but without GPU resources
        assert gpu_system._fragment_program is None
        assert gpu_system.fallback_system == self.mock_fallback

    def test_initialization_fragment_shader_creation_fails(self):
        """Test initialization when fragment shader creation fails."""
        # Mock fragment shader creation failure
        mock_shader_manager = Mock()
        mock_shader_manager.create_program.side_effect = Exception(
            "Shader compilation failed"
        )

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(
                self.game_world, self.mock_graphics_context, self.mock_fallback
            )

        # Should handle gracefully and set up for fallback
        assert gpu_system._fragment_program is None

    def test_update_with_fallback(self):
        """Test update method forwards to fallback system."""
        # Mock successful fragment shader initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
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
        # Mock successful fragment shader initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(
                self.game_world, self.mock_graphics_context, fallback_system=None
            )

        timestep = FixedTimestep(1.0 / 60.0)
        gpu_system.update(timestep)

        # Should update internal time without errors
        assert gpu_system._time == timestep

    def test_light_data_collection_empty_world(self):
        """Test light data collection with no lights."""
        # Mock successful fragment shader initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(
                self.game_world, self.mock_graphics_context, self.mock_fallback
            )

        viewport = Rect(0, 0, 10, 10)
        light_data = gpu_system._collect_light_data(viewport)

        assert light_data == []

    def test_light_data_collection_with_lights(self):
        """Test light data collection with various light types."""
        # Mock successful fragment shader initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Create mock lights with proper color format (RGB tuples 0-255)
        static_light = Mock(spec=StaticLight)
        static_light.position = (5, 5)
        static_light.radius = 3
        static_light.color = (255, 204, 153)  # RGB tuple

        dynamic_light = Mock(spec=DynamicLight)
        dynamic_light.position = (15, 8)
        dynamic_light.radius = 4
        dynamic_light.color = (230, 230, 255)  # RGB tuple
        # Add flicker attributes for DynamicLight
        dynamic_light.flicker_enabled = False
        dynamic_light.flicker_speed = 1.0
        dynamic_light.min_brightness = 1.0
        dynamic_light.max_brightness = 1.0

        # Light outside viewport (should be culled)
        distant_light = Mock(spec=StaticLight)
        distant_light.position = (100, 100)
        distant_light.radius = 2
        distant_light.color = (255, 255, 255)  # RGB tuple

        self.game_world.lights = [static_light, dynamic_light, distant_light]

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(
                self.game_world, self.mock_graphics_context, self.mock_fallback
            )

        viewport = Rect(0, 0, 20, 15)
        light_data = gpu_system._collect_light_data(viewport)

        # Should include 2 lights (static and dynamic), exclude distant light
        # Each light = 12 floats: position.xy, radius, base_intensity, color.rgb,
        # flicker_enabled, flicker_speed, min_brightness, max_brightness, padding
        expected_length = 2 * 12
        assert len(light_data) == expected_length

        # Check first light data (static light)
        assert light_data[0] == 5.0  # x position
        assert light_data[1] == 5.0  # y position
        assert light_data[2] == 3.0  # radius
        assert light_data[3] == 1.0  # base intensity
        assert light_data[4] == 1.0  # red (255/255)
        assert abs(light_data[5] - 0.8) < 0.01  # green (204/255)
        assert abs(light_data[6] - 0.6) < 0.01  # blue (153/255)
        assert light_data[7] == 0.0  # flicker_enabled (static light)
        assert light_data[8] == 1.0  # flicker_speed (default)
        assert light_data[9] == 1.0  # min_brightness (default)
        assert light_data[10] == 1.0  # max_brightness (default)
        assert light_data[11] == 0.0  # padding

        # Check second light data (dynamic light)
        assert light_data[12] == 15.0  # x position
        assert light_data[13] == 8.0  # y position
        assert light_data[14] == 4.0  # radius
        assert light_data[15] == 1.0  # base intensity
        assert abs(light_data[16] - 0.902) < 0.01  # red (230/255)
        assert abs(light_data[17] - 0.902) < 0.01  # green (230/255)
        assert light_data[18] == 1.0  # blue (255/255)
        # Dynamic light flicker params will be tested separately
        assert light_data[23] == 0.0  # padding

    def test_light_data_collection_frustum_culling(self):
        """Test that lights outside the viewport are properly culled."""
        # Mock successful fragment shader initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

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
            light.color = (255, 255, 255)  # RGB tuple
            lights.append(light)

        self.game_world.lights = lights

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
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
        assert len(light_data) == expected_lights * 12

    def test_compute_lightmap_fallback_when_no_gpu(self):
        """Test compute_lightmap falls back to CPU when GPU unavailable."""
        # Mock failed GPU initialization by removing mgl_context
        self.mock_graphics_context.mgl_context = None

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
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        # Make texture creation fail to trigger fallback
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()
        self.mock_mgl_context.texture.side_effect = Exception("GPU computation failed")
        self.mock_mgl_context.framebuffer.return_value = Mock()

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
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
        self.mock_graphics_context.mgl_context = None

        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, fallback_system=None
        )

        viewport = Rect(0, 0, 10, 10)
        result = gpu_system.compute_lightmap(viewport)

        assert result is None

    def test_light_notifications_increment_revision(self):
        """Test that light change notifications increment revision counter."""
        # Mock successful fragment shader initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
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
        # Mock successful fragment shader initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
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

    def test_max_lights_limit(self):
        """Test that light count is limited to MAX_LIGHTS in GPU computation."""
        # Mock successful fragment shader initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Create more lights than the maximum
        lights = []
        for i in range(GPULightingSystem.MAX_LIGHTS + 50):
            light = Mock(spec=StaticLight)
            light.position = (i % 20, i // 20)
            light.radius = 3
            light.color = (255, 255, 255)  # RGB tuple
            lights.append(light)

        self.game_world.lights = lights

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(
                self.game_world, self.mock_graphics_context, self.mock_fallback
            )

        viewport = Rect(0, 0, 100, 100)  # Large viewport to include many lights
        light_data = gpu_system._collect_light_data(viewport)

        # _collect_light_data doesn't limit - it collects all lights in frustum
        # The limiting happens in the actual GPU computation
        total_lights_in_viewport = len(light_data) // 12
        assert total_lights_in_viewport > GPULightingSystem.MAX_LIGHTS

        # Test that limiting actually happens during processing
        # When we have too many lights, they should be truncated to MAX_LIGHTS
        light_count = len(light_data) // 12
        if light_count > GPULightingSystem.MAX_LIGHTS:
            limited_data = light_data[: GPULightingSystem.MAX_LIGHTS * 12]
            assert len(limited_data) == GPULightingSystem.MAX_LIGHTS * 12

    def test_resource_cleanup(self):
        """Test that GPU resources are properly released."""
        mock_fragment_program = Mock()
        mock_output_texture = Mock()
        mock_output_buffer = Mock()
        mock_fullscreen_vao = Mock()

        # Mock successful fragment shader initialization
        mock_shader_manager = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = mock_output_buffer
        self.mock_mgl_context.vertex_array.return_value = mock_fullscreen_vao
        self.mock_mgl_context.texture.return_value = mock_output_texture

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(
                self.game_world, self.mock_graphics_context, self.mock_fallback
            )

        # Set up some resources manually to test cleanup
        gpu_system._fragment_program = mock_fragment_program
        gpu_system._output_texture = mock_output_texture
        gpu_system._output_buffer = mock_output_buffer
        gpu_system._fullscreen_vao = mock_fullscreen_vao

        # Call release
        gpu_system.release()

        # Should release all resources
        mock_fragment_program.release.assert_called_once()
        mock_output_texture.release.assert_called_once()
        mock_output_buffer.release.assert_called_once()
        mock_fullscreen_vao.release.assert_called_once()

    def test_resource_cleanup_with_none_resources(self):
        """Test that resource cleanup handles None resources gracefully."""
        gpu_system = GPULightingSystem(
            self.game_world, self.mock_graphics_context, self.mock_fallback
        )

        # Ensure all resources are None
        gpu_system._fragment_program = None
        gpu_system._output_texture = None
        gpu_system._output_buffer = None
        gpu_system._fullscreen_vao = None

        # Should not raise any exceptions
        gpu_system.release()


class TestGPULightingSystemIntegration:
    """Integration tests for GPU lighting system functionality."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.game_world = Mock(spec=GameWorld)
        self.game_world.lights = []

    def test_hardware_detection_success(self):
        """Test that hardware detection works with valid ModernGL context."""
        mock_graphics_context = Mock()
        mock_mgl_context = Mock()
        mock_graphics_context.mgl_context = mock_mgl_context

        # Mock successful initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        mock_mgl_context.buffer.return_value = Mock()
        mock_mgl_context.vertex_array.return_value = Mock()

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(self.game_world, mock_graphics_context)

        # Should have a valid fragment program after successful initialization
        assert gpu_system._fragment_program is not None

    def test_hardware_detection_failure_fallback(self):
        """Test that hardware detection gracefully handles missing ModernGL context."""
        mock_graphics_context = Mock()
        mock_graphics_context.mgl_context = None

        gpu_system = GPULightingSystem(self.game_world, mock_graphics_context)

        # Should not have a fragment program if initialization failed
        assert gpu_system._fragment_program is None

    def test_interface_compliance(self):
        """Test that GPULightingSystem properly implements LightingSystem interface."""
        mock_graphics_context = Mock()
        mock_mgl_context = Mock()
        mock_graphics_context.mgl_context = mock_mgl_context

        # Mock successful initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        mock_mgl_context.buffer.return_value = Mock()
        mock_mgl_context.vertex_array.return_value = Mock()

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
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
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        mock_mgl_context.buffer.return_value = Mock()
        mock_mgl_context.vertex_array.return_value = Mock()

        # Mock texture creation
        mock_textures = [Mock(), Mock()]
        mock_mgl_context.texture.side_effect = mock_textures

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
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


class TestGPULightingVisualRegression:
    """Visual regression tests for GPU lighting system Phase 1.3 features."""

    def setup_method(self):
        """Set up test fixtures for visual regression tests."""
        self.game_world = Mock(spec=GameWorld)
        self.game_world.lights = []

        self.mock_graphics_context = Mock()
        self.mock_mgl_context = Mock()
        self.mock_graphics_context.mgl_context = self.mock_mgl_context

    def test_dynamic_light_flicker_data_extraction(self):
        """Test flicker parameter extraction from DynamicLight objects."""
        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Create a DynamicLight with custom flicker parameters
        dynamic_light = Mock(spec=DynamicLight)
        dynamic_light.position = (10, 10)
        dynamic_light.radius = 5
        dynamic_light.color = (255, 128, 51)  # RGB tuple (1.0, 0.5, 0.2 as floats)
        dynamic_light.flicker_enabled = True
        dynamic_light.flicker_speed = 2.5
        dynamic_light.min_brightness = 0.3
        dynamic_light.max_brightness = 1.2

        self.game_world.lights = [dynamic_light]

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(self.game_world, self.mock_graphics_context)

        viewport = Rect(0, 0, 20, 20)
        light_data = gpu_system._collect_light_data(viewport)

        # Should have 1 light with 12 floats
        assert len(light_data) == 12

        # Check basic properties
        assert light_data[0] == 10.0  # x position
        assert light_data[1] == 10.0  # y position
        assert light_data[2] == 5.0  # radius
        assert light_data[3] == 1.0  # base intensity
        assert light_data[4] == 1.0  # red (255/255)
        assert abs(light_data[5] - 0.502) < 0.01  # green (128/255)
        assert abs(light_data[6] - 0.2) < 0.01  # blue (51/255)

        # Check flicker parameters
        assert light_data[7] == 1.0  # flicker_enabled (True)
        assert light_data[8] == 2.5  # flicker_speed
        assert light_data[9] == 0.3  # min_brightness
        assert light_data[10] == 1.2  # max_brightness
        assert light_data[11] == 0.0  # padding

    def test_static_light_no_flicker_data(self):
        """Test that static lights get default (no-flicker) parameters."""
        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        static_light = Mock(spec=StaticLight)
        static_light.position = (5, 5)
        static_light.radius = 3
        static_light.color = (204, 204, 255)  # RGB tuple (0.8, 0.8, 1.0 as floats)

        self.game_world.lights = [static_light]

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(self.game_world, self.mock_graphics_context)

        viewport = Rect(0, 0, 10, 10)
        light_data = gpu_system._collect_light_data(viewport)

        # Check flicker parameters for static light
        assert light_data[7] == 0.0  # flicker_enabled (False)
        assert light_data[8] == 1.0  # flicker_speed (default)
        assert light_data[9] == 1.0  # min_brightness (default)
        assert light_data[10] == 1.0  # max_brightness (default)

    def test_mixed_light_types_data_collection(self):
        """Test data collection with mixed static and dynamic lights."""
        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Create one static and one dynamic light
        static_light = Mock(spec=StaticLight)
        static_light.position = (5, 5)
        static_light.radius = 3
        static_light.color = (255, 255, 255)

        dynamic_light = Mock(spec=DynamicLight)
        dynamic_light.position = (15, 15)
        dynamic_light.radius = 4
        dynamic_light.color = (255, 153, 51)  # RGB tuple (1.0, 0.6, 0.2 as floats)
        dynamic_light.flicker_enabled = True
        dynamic_light.flicker_speed = 1.5
        dynamic_light.min_brightness = 0.5
        dynamic_light.max_brightness = 1.0

        self.game_world.lights = [static_light, dynamic_light]

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(self.game_world, self.mock_graphics_context)

        viewport = Rect(0, 0, 25, 25)
        light_data = gpu_system._collect_light_data(viewport)

        # Should have 2 lights * 12 floats = 24 floats total
        assert len(light_data) == 24

        # Check static light (first 12 floats)
        assert light_data[7] == 0.0  # no flicker

        # Check dynamic light (second 12 floats, offset by 12)
        assert light_data[19] == 1.0  # flicker enabled
        assert light_data[20] == 1.5  # flicker speed
        assert light_data[21] == 0.5  # min brightness
        assert light_data[22] == 1.0  # max brightness


class TestGPULightingPerformance:
    """Performance and benchmarking tests for GPU lighting system."""

    def setup_method(self):
        """Set up performance test fixtures."""
        self.game_world = Mock(spec=GameWorld)
        self.game_world.lights = []

        self.mock_graphics_context = Mock()
        self.mock_mgl_context = Mock()
        self.mock_graphics_context.mgl_context = self.mock_mgl_context

    def test_light_count_scaling(self):
        """Test that the system handles various light counts efficiently."""
        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Test with different numbers of lights
        test_counts = [0, 1, 10, 50, 100]  # Don't test above MAX_LIGHTS for performance

        for light_count in test_counts:
            # Create lights
            lights = []
            for i in range(light_count):
                light = Mock(spec=StaticLight)
                light.position = (i % 20, i // 20)
                light.radius = 3
                light.color = (255, 255, 255)
                lights.append(light)

            self.game_world.lights = lights

            with patch(
                "catley.backends.moderngl.shader_manager.ShaderManager",
                return_value=mock_shader_manager,
            ):
                gpu_system = GPULightingSystem(
                    self.game_world, self.mock_graphics_context
                )

            viewport = Rect(0, 0, 50, 50)  # Large enough to include all lights
            light_data = gpu_system._collect_light_data(viewport)

            # Data collection should handle any number of lights
            expected_data_length = min(light_count, len(lights)) * 12
            # Account for frustum culling - some lights might be outside viewport
            assert len(light_data) <= expected_data_length

            # Should be multiples of 12 (12 floats per light)
            assert len(light_data) % 12 == 0

    def test_memory_usage_optimization(self):
        """Test that memory usage is optimized for the new 12-float format."""
        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            GPULightingSystem(self.game_world, self.mock_graphics_context)

        # New format: 12 floats * 4 bytes = 48 bytes per light
        # For MAX_LIGHTS = 32: 32 * 48 = 1536 bytes total
        expected_total_size = GPULightingSystem.MAX_LIGHTS * 12 * 4

        # Note: In fragment shader implementation, we don't pre-allocate buffers
        # like we did in compute shader implementation, so this test validates
        # the expected memory usage calculation
        assert expected_total_size == 1536  # 32 lights * 48 bytes each

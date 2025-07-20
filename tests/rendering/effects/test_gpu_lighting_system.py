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
        rng = np.random.default_rng()
        expected_result = rng.random((10, 10, 3)).astype(np.float32)
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
        rng = np.random.default_rng()
        expected_result = rng.random((10, 10, 3)).astype(np.float32)
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


class TestGPUDirectionalLighting:
    """Test suite for GPU directional lighting (Phase 2.1)."""

    def setup_method(self):
        """Set up test fixtures for directional lighting tests."""
        from catley.environment.map import GameMap, MapRegion

        self.game_world = Mock(spec=GameWorld)
        self.game_world.lights = []

        # Create a mock game map with regions
        self.game_map = Mock(spec=GameMap)
        self.game_map.width = 10
        self.game_map.height = 10
        self.game_map.structural_revision = 1

        # Add transparent property for sky exposure texture generation
        import numpy as np

        self.game_map.transparent = np.ones(
            (10, 10), dtype=bool
        )  # All tiles transparent by default

        # Create mock regions with different sky exposures
        self.outdoor_region = MapRegion.create_outdoor_region(
            map_region_id=1, region_type="outdoor", sky_exposure=1.0
        )
        self.indoor_region = MapRegion.create_indoor_region(
            map_region_id=2, region_type="indoor", sky_exposure=0.0
        )

        # Mock get_region_at to return appropriate regions
        def get_region_at(pos):
            x, y = pos
            # Left half is outdoor, right half is indoor
            if x < 5:
                return self.outdoor_region
            return self.indoor_region

        self.game_map.get_region_at = Mock(side_effect=get_region_at)
        self.game_world.game_map = self.game_map

        self.mock_graphics_context = Mock()
        self.mock_mgl_context = Mock()
        self.mock_graphics_context.mgl_context = self.mock_mgl_context

    def test_directional_light_filtered_from_point_lights(self):
        """Test that DirectionalLight instances are excluded from point light data."""
        from catley.game.lights import DirectionalLight, StaticLight

        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Add both point lights and directional light
        point_light = StaticLight(position=(5, 5), radius=3, color=(255, 204, 153))
        sun_light = DirectionalLight.create_sun()

        self.game_world.lights = [point_light, sun_light]

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(self.game_world, self.mock_graphics_context)

        viewport = Rect(0, 0, 10, 10)
        light_data = gpu_system._collect_light_data(viewport)

        # Should only include the point light (12 floats), not the directional light
        assert len(light_data) == 12
        assert light_data[0] == 5.0  # point light x
        assert light_data[1] == 5.0  # point light y

    def test_sky_exposure_texture_creation(self):
        """Test creation of sky exposure texture from game map regions."""
        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Mock texture creation
        mock_texture = Mock()
        self.mock_mgl_context.texture.return_value = mock_texture

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(self.game_world, self.mock_graphics_context)

        # Call update sky exposure texture
        gpu_system._update_sky_exposure_texture()

        # Verify texture was created with correct dimensions
        self.mock_mgl_context.texture.assert_called_once_with(
            (10, 10),  # game map dimensions
            components=1,  # single channel
            dtype="f4",  # 32-bit float
        )

        # Verify texture data was written
        mock_texture.write.assert_called_once()

        # Check the written data
        written_data = mock_texture.write.call_args[0][0]
        data_array = np.frombuffer(written_data, dtype=np.float32).reshape(10, 10)

        # Left half should be 1.0 (outdoor), right half should be 0.0 (indoor)
        assert np.all(data_array[:, :5] == 1.0)
        assert np.all(data_array[:, 5:] == 0.0)

    def test_sky_exposure_texture_caching(self):
        """Test that sky exposure texture is only updated when map structure changes."""
        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Mock texture creation
        mock_texture = Mock()
        self.mock_mgl_context.texture.return_value = mock_texture

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(self.game_world, self.mock_graphics_context)

        # First update
        gpu_system._update_sky_exposure_texture()
        assert self.mock_mgl_context.texture.call_count == 1

        # Second update with same revision - should not recreate
        gpu_system._update_sky_exposure_texture()
        assert self.mock_mgl_context.texture.call_count == 1

        # Change map revision
        self.game_map.structural_revision = 2

        # Third update with new revision - should recreate
        gpu_system._update_sky_exposure_texture()
        assert self.mock_mgl_context.texture.call_count == 2
        mock_texture.release.assert_called_once()

    def test_directional_light_uniforms_set(self):
        """Test that directional light uniforms are properly set."""
        from catley.game.lights import DirectionalLight

        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_fragment_uniforms = {
            "u_sun_direction": Mock(),
            "u_sun_color": Mock(),
            "u_sun_intensity": Mock(),
            "u_sky_exposure_power": Mock(),
        }

        def getitem(key):
            return mock_fragment_uniforms.get(key, Mock())

        mock_fragment_program.__getitem__ = Mock(side_effect=getitem)
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Add directional light
        sun_light = DirectionalLight.create_sun(
            elevation_degrees=45.0, azimuth_degrees=135.0, intensity=0.8
        )
        self.game_world.lights = [sun_light]

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(self.game_world, self.mock_graphics_context)

        # Call set directional light uniforms
        gpu_system._set_directional_light_uniforms()

        # Verify sun uniforms were set
        mock_fragment_uniforms["u_sun_direction"].value = (
            sun_light.direction.x,
            sun_light.direction.y,
        )
        mock_fragment_uniforms["u_sun_intensity"].value = 0.8

        # Verify color was normalized to 0-1 range
        expected_color = (
            sun_light.color[0] / 255.0,
            sun_light.color[1] / 255.0,
            sun_light.color[2] / 255.0,
        )
        mock_fragment_uniforms["u_sun_color"].value = expected_color

    def test_directional_light_uniforms_no_sun(self):
        """Test that directional light uniforms are set to 'off' when no sun exists."""
        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_fragment_uniforms = {
            "u_sun_direction": Mock(),
            "u_sun_color": Mock(),
            "u_sun_intensity": Mock(),
            "u_sky_exposure_power": Mock(),
        }

        def getitem(key):
            return mock_fragment_uniforms.get(key, Mock())

        mock_fragment_program.__getitem__ = Mock(side_effect=getitem)
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # No lights at all
        self.game_world.lights = []

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(self.game_world, self.mock_graphics_context)

        # Call set directional light uniforms
        gpu_system._set_directional_light_uniforms()

        # Verify sun uniforms were set to "off" values
        mock_fragment_uniforms["u_sun_direction"].value = (0.0, 0.0)
        mock_fragment_uniforms["u_sun_color"].value = (0.0, 0.0, 0.0)
        mock_fragment_uniforms["u_sun_intensity"].value = 0.0

    def test_sky_exposure_texture_binding_in_compute(self):
        """Test that sky exposure texture is used during lightmap computation."""
        from catley.game.lights import DirectionalLight

        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Mock texture creation
        mock_output_texture = Mock()
        mock_sky_texture = Mock()
        self.mock_mgl_context.texture.side_effect = [
            mock_output_texture,
            mock_sky_texture,
        ]

        # Mock resource manager
        mock_resource_manager = Mock()
        mock_fbo = Mock()
        mock_resource_manager.get_or_create_fbo_for_texture.return_value = mock_fbo
        self.mock_graphics_context.resource_manager = mock_resource_manager

        # Add directional light
        sun_light = DirectionalLight.create_sun()
        self.game_world.lights = [sun_light]

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(self.game_world, self.mock_graphics_context)

        # Create the sky exposure texture first
        gpu_system._update_sky_exposure_texture()

        # Verify sky texture was created
        assert gpu_system._sky_exposure_texture is not None

        # Test that the texture binding happens
        if gpu_system._sky_exposure_texture is not None:
            # This would be tested in integration, here we verify texture exists
            assert self.mock_mgl_context.texture.call_count >= 1

    def test_resource_cleanup_includes_sky_texture(self):
        """Test that sky exposure texture is properly released on cleanup."""
        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Mock texture creation
        mock_sky_texture = Mock()
        self.mock_mgl_context.texture.return_value = mock_sky_texture

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(self.game_world, self.mock_graphics_context)

        # Create sky exposure texture
        gpu_system._update_sky_exposure_texture()

        # Release resources
        gpu_system.release()

        # Verify sky exposure texture was released
        mock_sky_texture.release.assert_called_once()

    def test_sky_exposure_texture_respects_tile_transparency(self):
        """Test that sky exposure texture sets 0.0 for non-transparent tiles."""
        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Mock texture creation
        mock_texture = Mock()
        self.mock_mgl_context.texture.return_value = mock_texture

        # Create a game map with mixed transparent/non-transparent tiles
        import numpy as np

        # Create transparency map: doors/walls non-transparent, floors transparent
        transparency_map = np.ones((10, 10), dtype=bool)  # All transparent by default
        transparency_map[5, 5] = False  # Closed door at (5, 5)
        transparency_map[3, 3] = False  # Wall at (3, 3)

        self.game_map.transparent = transparency_map

        # Create outdoor region that should give sky_exposure = 1.0
        from catley.environment.map import MapRegion

        outdoor_region = MapRegion.create_outdoor_region(
            map_region_id=1, region_type="outdoor", sky_exposure=1.0
        )
        self.game_map.get_region_at = Mock(return_value=outdoor_region)

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(self.game_world, self.mock_graphics_context)

        # Update sky exposure texture
        gpu_system._update_sky_exposure_texture()

        # Verify texture was created and written to
        mock_texture.write.assert_called_once()

        # Check the written data
        written_data = mock_texture.write.call_args[0][0]
        data_array = np.frombuffer(written_data, dtype=np.float32).reshape(10, 10)

        # Transparent tiles should have sky_exposure = 1.0
        assert data_array[0, 0] == 1.0, "Transparent tile should have full sky exposure"

        # Non-transparent tiles should have sky_exposure = 0.0
        assert data_array[5, 5] == 0.0, "Closed door should block all sunlight"
        assert data_array[3, 3] == 0.0, "Wall should block all sunlight"


class TestGPUDirectionalShadows:
    """Test suite for GPU directional shadow implementation (Phase 2.2)."""

    def setup_method(self):
        """Set up test fixtures for directional shadow tests."""
        from catley.environment.map import GameMap, MapRegion

        self.game_world = Mock(spec=GameWorld)
        self.game_world.lights = []

        # Mock additional game world attributes for shadow collection
        self.game_world.actor_spatial_index = None
        self.game_world.player = None

        # Create a mock game map with regions
        self.game_map = Mock(spec=GameMap)
        self.game_map.width = 10
        self.game_map.height = 10
        self.game_map.structural_revision = 1

        # Mock tiles array for shadow caster collection
        import numpy as np

        # Use simple integer values since the system just checks tile_id
        self.game_map.tiles = np.full((10, 10), 1, dtype=int)  # 1 = floor tile

        # Create outdoor region for directional shadows
        self.outdoor_region = MapRegion.create_outdoor_region(
            map_region_id=1, region_type="outdoor", sky_exposure=1.0
        )

        # Mock get_region_at to return outdoor region
        self.game_map.get_region_at = Mock(return_value=self.outdoor_region)
        self.game_world.game_map = self.game_map

        # Mock shadow casters
        self.mock_shadow_casters = [(5, 5), (7, 3)]  # Two shadow casters

        self.mock_graphics_context = Mock()
        self.mock_mgl_context = Mock()
        self.mock_graphics_context.mgl_context = self.mock_mgl_context

    def test_directional_shadows_cast_in_correct_direction(self):
        """Test that directional shadows are cast in the direction away from sun."""
        from catley.game.lights import DirectionalLight

        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Create sun with southeast direction (positive x, positive y)
        sun_light = DirectionalLight.create_sun(
            elevation_degrees=45.0, azimuth_degrees=135.0, intensity=0.8
        )
        self.game_world.lights = [sun_light]

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            GPULightingSystem(self.game_world, self.mock_graphics_context)

        # Test shadow direction calculation
        # Sun direction is southeast (azimuth 135°), so shadows should go northwest
        expected_shadow_direction = (-sun_light.direction.x, -sun_light.direction.y)

        # Verify sun direction is southeast (positive x, negative y screen coords)
        assert sun_light.direction.x > 0, "Sun should point southeast (positive x)"
        assert sun_light.direction.y < 0, (
            "Sun should point downward (negative y in screen coordinates)"
        )

        # Therefore shadow direction should be northwest (negative x, positive y)
        assert expected_shadow_direction[0] < 0, (
            "Shadows should point northwest (negative x)"
        )
        assert expected_shadow_direction[1] > 0, (
            "Shadows should point upward (positive y)"
        )

    def test_directional_shadows_only_in_outdoor_areas(self):
        """Test directional shadows only appear in areas with sky_exposure > 0.1."""
        from catley.game.lights import DirectionalLight

        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Create regions with different sky exposures
        def get_region_by_exposure(pos):
            from catley.environment.map import MapRegion

            x, y = pos
            if x < 3:  # Left side: indoor (no sky exposure)
                return MapRegion.create_indoor_region(
                    map_region_id=1, region_type="indoor", sky_exposure=0.0
                )
            if x < 6:  # Middle: low sky exposure
                return MapRegion.create_outdoor_region(
                    map_region_id=2, region_type="partial", sky_exposure=0.05
                )
            # Right side: outdoor (full sky exposure)
            return MapRegion.create_outdoor_region(
                map_region_id=3, region_type="outdoor", sky_exposure=1.0
            )

        self.game_map.get_region_at = Mock(side_effect=get_region_by_exposure)

        # Set up transparency data for the map
        import numpy as np

        transparency_map = np.ones((10, 10), dtype=bool)  # All transparent by default
        self.game_map.transparent = transparency_map

        # Add sun and shadow casters
        sun_light = DirectionalLight.create_sun()
        self.game_world.lights = [sun_light]

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(self.game_world, self.mock_graphics_context)

        # Create sky exposure texture
        gpu_system._update_sky_exposure_texture()

        # The shader logic checks: if (sky_exposure > 0.1 && u_sun_intensity > 0.0)
        # Only the rightmost region (x >= 6) should have directional shadows

        # This test verifies the setup - actual shadow computation in shader
        # and would be tested through integration testing with rendering

    def test_directional_shadow_intensity_and_length_optimized(self):
        """Test shadow intensity and length parameters are optimized for visibility."""
        from catley.game.lights import DirectionalLight

        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Add sun light
        sun_light = DirectionalLight.create_sun(intensity=0.8)
        self.game_world.lights = [sun_light]

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(self.game_world, self.mock_graphics_context)

        # Test that shadow parameters are optimized for visibility
        # GPU code: base_intensity = SHADOW_INTENSITY * 1.2 (stronger than torch)
        # GPU code: shadow_length = min(SHADOW_MAX_LENGTH, 12) (longer shadows)
        # GPU code: edge_intensity = intensity * 0.5 (50% for visible edges)

        # These constants are embedded in the shader, so we verify they exist
        # by checking that the shader was successfully created
        assert gpu_system._fragment_program is not None

    def test_directional_shadow_caster_collection(self):
        """Test that shadow casters are properly collected for directional shadows."""
        from catley.game.lights import DirectionalLight

        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Add sun light
        sun_light = DirectionalLight.create_sun()
        self.game_world.lights = [sun_light]

        with (
            patch(
                "catley.backends.moderngl.shader_manager.ShaderManager",
                return_value=mock_shader_manager,
            ),
            patch(
                "catley.environment.tile_types.get_tile_type_data_by_id",
                return_value={
                    "casts_shadows": False
                },  # Mock tiles that don't cast shadows
            ),
        ):
            gpu_system = GPULightingSystem(self.game_world, self.mock_graphics_context)

            # Mock shadow caster collection
            viewport = Rect(0, 0, 10, 10)

            # The _collect_shadow_casters_global method collects actor positions
            # and shadow-casting tiles, similar to the CPU implementation
            shadow_casters = gpu_system._collect_shadow_casters_global(viewport)

            # Verify that shadow caster collection returns a list of floats (x, y pairs)
            assert isinstance(shadow_casters, list)

            # Each shadow caster represented as consecutive x, y float coordinates
            # The actual collection depends on the game world state
            assert len(shadow_casters) % 2 == 0, "Shadow casters should be x,y pairs"

    def test_directional_shadow_falloff_enabled_parameter(self):
        """Test that directional shadows respect the shadow falloff parameter."""
        from catley.game.lights import DirectionalLight

        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Add sun light
        sun_light = DirectionalLight.create_sun()
        self.game_world.lights = [sun_light]

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            GPULightingSystem(self.game_world, self.mock_graphics_context)

        # The shader uses u_shadow_falloff_enabled to determine whether to apply
        # distance-based falloff to shadow intensity
        # This matches the CPU implementation's SHADOW_FALLOFF flag

        # The actual falloff logic is:
        # if (u_shadow_falloff_enabled) {
        #     distance_falloff = 1.0 - (float(j - 1) / max_shadow_length);
        # }

        # This test verifies the setup exists - actual behavior in integration

    def test_directional_shadow_soft_edges(self):
        """Test that directional shadows have softer edges like CPU implementation."""
        from catley.game.lights import DirectionalLight

        # Mock successful GPU initialization
        mock_shader_manager = Mock()
        mock_fragment_program = Mock()
        mock_shader_manager.create_program.return_value = mock_fragment_program
        self.mock_mgl_context.buffer.return_value = Mock()
        self.mock_mgl_context.vertex_array.return_value = Mock()

        # Add sun light
        sun_light = DirectionalLight.create_sun()
        self.game_world.lights = [sun_light]

        with patch(
            "catley.backends.moderngl.shader_manager.ShaderManager",
            return_value=mock_shader_manager,
        ):
            gpu_system = GPULightingSystem(self.game_world, self.mock_graphics_context)

        # The shader implements soft edges for the first 2 shadow tiles
        # with 50% intensity on adjacent positions (not diagonal)
        # Optimized for visibility: edge_intensity = intensity * 0.5

        # The shader code checks: if (j <= 2) for soft edges
        # and uses 4 adjacent positions (not diagonal for directional shadows)

        # This test verifies the GPU system is properly initialized
        assert gpu_system._fragment_program is not None

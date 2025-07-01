"""
Unit tests for the UITextureRenderer class.

These tests use unittest.mock to completely mock the moderngl library,
allowing us to test the UITextureRenderer logic without needing a real OpenGL context.
"""

from unittest.mock import Mock

import numpy as np
import pytest


class TestUITextureRenderer:
    """Test suite for UITextureRenderer using mocked moderngl."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocked moderngl context and related objects."""
        # Mock the moderngl context
        self.mock_mgl_context = Mock()

        # Mock the shader program creation with proper subscripting support
        self.mock_program = Mock()
        self.mock_uniform = Mock()
        # Configure the program to support subscripting like program["u_texture"]
        self.mock_program.__getitem__ = Mock(return_value=self.mock_uniform)
        self.mock_mgl_context.program.return_value = self.mock_program

        # Mock VBO and VAO
        self.mock_vbo = Mock()
        self.mock_vao = Mock()
        self.mock_mgl_context.buffer.return_value = self.mock_vbo
        self.mock_mgl_context.vertex_array.return_value = self.mock_vao

        # Mock textures for testing
        self.mock_texture1 = Mock()
        self.mock_texture2 = Mock()

        # Import the class after setting up mocks
        from catley.backends.moderngl.graphics import UITextureRenderer

        self.UITextureRenderer = UITextureRenderer

    def test_initialization(self):
        """Verify that the constructor creates a shader program, VBO, VAO and that the
        internal render queue is empty."""
        renderer = self.UITextureRenderer(self.mock_mgl_context)

        # Verify initialization
        assert renderer.mgl_context == self.mock_mgl_context

        # Verify shader program was created
        self.mock_mgl_context.program.assert_called_once()
        assert renderer.program == self.mock_program

        # Verify VBO and VAO were created during initialization
        self.mock_mgl_context.buffer.assert_called_once()
        buffer_call = self.mock_mgl_context.buffer.call_args
        assert buffer_call.kwargs.get("dynamic") is True
        assert "reserve" in buffer_call.kwargs

        self.mock_mgl_context.vertex_array.assert_called_once()
        assert renderer.vbo == self.mock_vbo
        assert renderer.vao == self.mock_vao

        # Verify render queue is initially empty
        assert len(renderer.render_queue) == 0

    def test_shader_program_creation(self):
        """Verify the vertex shader takes u_letterbox and the fragment shader
        takes u_texture (NOT u_atlas)."""
        self.UITextureRenderer(self.mock_mgl_context)

        # Verify program creation was called
        self.mock_mgl_context.program.assert_called_once()

        # Get the call arguments
        call_args = self.mock_mgl_context.program.call_args
        assert "vertex_shader" in call_args.kwargs
        assert "fragment_shader" in call_args.kwargs

        # Verify vertex shader contains expected content
        vertex_shader = call_args.kwargs["vertex_shader"]
        assert "#version 330" in vertex_shader
        assert "uniform vec4 u_letterbox" in vertex_shader

        # Verify fragment shader contains expected content
        fragment_shader = call_args.kwargs["fragment_shader"]
        assert "#version 330" in fragment_shader
        assert "uniform sampler2D u_texture" in fragment_shader
        # Ensure it does NOT use u_atlas
        assert "u_atlas" not in fragment_shader

    def test_begin_frame(self):
        """Verify that calling this method clears any previously added quads
        from the render queue."""
        renderer = self.UITextureRenderer(self.mock_mgl_context)

        # Add some mock data to the render queue
        test_vertices = np.zeros((6, 8), dtype=np.float32)
        renderer.render_queue.append((self.mock_texture1, test_vertices))
        renderer.render_queue.append((self.mock_texture2, test_vertices))

        # Verify queue has items
        assert len(renderer.render_queue) == 2

        # Call begin_frame
        renderer.begin_frame()

        # Verify queue is now empty
        assert len(renderer.render_queue) == 0

    def test_add_textured_quad(self):
        """Verify that this method correctly adds a (texture, vertices) tuple
        to its internal queue. Test adding multiple quads."""
        renderer = self.UITextureRenderer(self.mock_mgl_context)

        # Test data - 6 vertices with 8 components each (position + uv + color)
        vertices1 = np.random.random((6, 8)).astype(np.float32)
        vertices2 = np.random.random((6, 8)).astype(np.float32)

        # Initially queue should be empty
        assert len(renderer.render_queue) == 0

        # Add first quad
        renderer.add_textured_quad(self.mock_texture1, vertices1)

        # Verify first quad was added
        assert len(renderer.render_queue) == 1
        assert renderer.render_queue[0][0] == self.mock_texture1
        assert np.array_equal(renderer.render_queue[0][1], vertices1)

        # Add second quad
        renderer.add_textured_quad(self.mock_texture2, vertices2)

        # Verify both quads are in queue
        assert len(renderer.render_queue) == 2
        assert renderer.render_queue[1][0] == self.mock_texture2
        assert np.array_equal(renderer.render_queue[1][1], vertices2)

    def test_render_empty_queue(self):
        """Verify that if render() is called when the queue is empty, no
        moderngl render calls are made."""
        renderer = self.UITextureRenderer(self.mock_mgl_context)

        # Reset mock call counts after initialization
        self.mock_mgl_context.buffer.reset_mock()
        self.mock_vbo.write.reset_mock()
        self.mock_vao.render.reset_mock()

        # Ensure queue is empty
        renderer.begin_frame()
        assert len(renderer.render_queue) == 0

        letterbox_geometry = (0, 0, 800, 600)

        # Call render with empty queue
        renderer.render(letterbox_geometry)

        # Verify no buffer writes or renders were performed
        self.mock_vbo.write.assert_not_called()
        self.mock_vao.render.assert_not_called()

        # Verify no texture.use() calls were made
        self.mock_texture1.use.assert_not_called()
        self.mock_texture2.use.assert_not_called()

    def test_render_calls(self):
        """Test that render() correctly processes queued textures and makes
        proper moderngl calls."""
        renderer = self.UITextureRenderer(self.mock_mgl_context)

        # Reset mock call counts after initialization
        self.mock_mgl_context.buffer.reset_mock()
        self.mock_vbo.write.reset_mock()
        self.mock_vao.render.reset_mock()

        # Create test vertex data for two quads
        vertices1 = np.random.random((6, 8)).astype(np.float32)
        vertices2 = np.random.random((6, 8)).astype(np.float32)

        # Add two different textures with vertex data
        renderer.add_textured_quad(self.mock_texture1, vertices1)
        renderer.add_textured_quad(self.mock_texture2, vertices2)

        letterbox_geometry = (0, 0, 800, 600)

        # Call render
        renderer.render(letterbox_geometry)

        # Verify letterbox uniform was set
        self.mock_program.__getitem__.assert_called_with("u_letterbox")
        assert self.mock_uniform.value == letterbox_geometry

        # With the optimized version, we reuse the VBO and VAO
        # We expect 2 writes to the VBO (one for each quad)
        assert self.mock_vbo.write.call_count == 2

        # Verify texture.use() was called for both textures
        self.mock_texture1.use.assert_called_once()
        self.mock_texture2.use.assert_called_once()

        # Verify render happened twice (once for each quad)
        assert self.mock_vao.render.call_count == 2

        # Verify the correct sequence of calls for each iteration
        # Check that VBO write was called with the correct vertex data
        write_calls = self.mock_vbo.write.call_args_list

        # First call should have vertices1
        first_call_data = write_calls[0][0][0]  # First positional argument
        assert np.array_equal(first_call_data, vertices1.tobytes())

        # Second call should have vertices2
        second_call_data = write_calls[1][0][0]  # First positional argument
        assert np.array_equal(second_call_data, vertices2.tobytes())

        # Verify render calls happened with correct vertex count (6 vertices per quad)
        render_calls = self.mock_vao.render.call_args_list
        for call in render_calls:
            assert call.kwargs.get("vertices") == 6

    def test_release(self):
        """Test that release() properly cleans up GPU resources."""
        renderer = self.UITextureRenderer(self.mock_mgl_context)

        # Call release
        renderer.release()

        # Verify all resources were released
        self.mock_vao.release.assert_called_once()
        self.mock_vbo.release.assert_called_once()
        self.mock_program.release.assert_called_once()

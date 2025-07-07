"""Shader management utility for ModernGL backends.

This module provides utilities for loading and managing GLSL shaders from asset files,
replacing the previous embedded shader strings with a clean file-based approach.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import moderngl

logger = logging.getLogger(__name__)


class ShaderManager:
    """Manages loading and caching of GLSL shaders from asset files."""

    def __init__(self, mgl_context: moderngl.Context) -> None:
        """Initialize the shader manager.

        Args:
            mgl_context: The ModernGL context for creating shader programs
        """
        self.mgl_context = mgl_context
        self._shader_cache: dict[str, str] = {}
        self._program_cache: dict[str, moderngl.Program] = {}

        # Find the asset directory relative to this file
        # catley/backends/moderngl/shader_manager.py -> ../../assets/shaders/
        self.shader_dir = (
            Path(__file__).parent.parent.parent.parent / "assets" / "shaders"
        )

        if not self.shader_dir.exists():
            raise FileNotFoundError(f"Shader directory not found: {self.shader_dir}")

        logger.debug(
            f"ShaderManager initialized with shader directory: {self.shader_dir}"
        )

    def load_shader_source(self, shader_path: str) -> str:
        """Load shader source code from a file.

        Args:
            shader_path: Path to shader file relative to assets/shaders/
                        (e.g., "screen/main.vert", "lighting/point_light.frag")

        Returns:
            The shader source code as a string

        Raises:
            FileNotFoundError: If the shader file doesn't exist
        """
        if shader_path in self._shader_cache:
            return self._shader_cache[shader_path]

        full_path = self.shader_dir / shader_path

        if not full_path.exists():
            raise FileNotFoundError(f"Shader file not found: {full_path}")

        try:
            with full_path.open(encoding="utf-8") as f:
                source = f.read()

            self._shader_cache[shader_path] = source
            logger.debug(f"Loaded shader: {shader_path}")
            return source

        except Exception as e:
            raise RuntimeError(f"Failed to load shader {shader_path}: {e}") from e

    def create_program(
        self,
        vertex_shader_path: str,
        fragment_shader_path: str,
        cache_key: str | None = None,
    ) -> moderngl.Program:
        """Create a ModernGL program from vertex and fragment shader files.

        Args:
            vertex_shader_path: Path to vertex shader (e.g., "screen/main.vert")
            fragment_shader_path: Path to fragment shader (e.g., "screen/main.frag")
            cache_key: Optional cache key. If None, uses vertex_shader_path

        Returns:
            Compiled ModernGL program

        Raises:
            RuntimeError: If shader compilation fails
        """
        cache_key = cache_key or vertex_shader_path

        if cache_key in self._program_cache:
            return self._program_cache[cache_key]

        try:
            vertex_source = self.load_shader_source(vertex_shader_path)
            fragment_source = self.load_shader_source(fragment_shader_path)

            program = self.mgl_context.program(
                vertex_shader=vertex_source, fragment_shader=fragment_source
            )

            self._program_cache[cache_key] = program
            logger.debug(f"Created shader program: {cache_key}")
            return program

        except Exception as e:
            logger.error(f"Failed to create shader program {cache_key}: {e}")
            logger.debug(f"Vertex shader path: {vertex_shader_path}")
            logger.debug(f"Fragment shader path: {fragment_shader_path}")
            raise RuntimeError(f"Shader program compilation failed: {e}") from e

    def create_fragment_program(
        self, fragment_shader_path: str, cache_key: str | None = None
    ) -> moderngl.Program:
        """Create a program with full-screen quad vertex shader and
        custom fragment shader.

        This is useful for fragment shader-based compute operations like lighting.

        Args:
            fragment_shader_path: Path to fragment shader
                (e.g., "lighting/point_light.frag")
            cache_key: Optional cache key. If None, uses fragment_shader_path

        Returns:
            Compiled ModernGL program with full-screen vertex shader
        """
        cache_key = cache_key or f"fullscreen_{fragment_shader_path}"

        if cache_key in self._program_cache:
            return self._program_cache[cache_key]

        # Standard full-screen quad vertex shader
        fullscreen_vertex = """#version 330
in vec2 in_position;
out vec2 v_uv;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_uv = in_position * 0.5 + 0.5;  // Convert from [-1,1] to [0,1]
}"""

        try:
            fragment_source = self.load_shader_source(fragment_shader_path)

            program = self.mgl_context.program(
                vertex_shader=fullscreen_vertex, fragment_shader=fragment_source
            )

            self._program_cache[cache_key] = program
            logger.debug(f"Created full-screen fragment program: {cache_key}")
            return program

        except Exception as e:
            logger.error(f"Failed to create fragment program {cache_key}: {e}")
            raise RuntimeError(f"Fragment program compilation failed: {e}") from e

    def clear_cache(self) -> None:
        """Clear all cached shaders and programs."""
        self._shader_cache.clear()
        # Note: Don't release programs here as they may still be in use
        self._program_cache.clear()
        logger.debug("Shader cache cleared")

    def reload_shader(self, shader_path: str) -> None:
        """Reload a specific shader from disk, clearing its cache.

        Args:
            shader_path: Path to shader file to reload
        """
        if shader_path in self._shader_cache:
            del self._shader_cache[shader_path]
            logger.debug(f"Cleared cache for shader: {shader_path}")

        # Clear any programs that use this shader
        programs_to_clear = [key for key in self._program_cache if shader_path in key]

        for key in programs_to_clear:
            del self._program_cache[key]
            logger.debug(f"Cleared program cache for: {key}")

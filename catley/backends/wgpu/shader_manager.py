"""WGSL shader management and compilation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import wgpu

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class WGPUShaderManager:
    """Manages WGSL shader compilation and pipeline creation.

    This class provides utilities for loading and managing WGSL shaders from
    asset files, replacing the previous embedded shader strings with a clean
    file-based approach.
    """

    def __init__(self, device: wgpu.GPUDevice) -> None:
        """Initialize WGPU shader manager.

        Args:
            device: The WGPU device for creating shader modules and pipelines
        """
        self.device = device
        self._shader_cache: dict[str, str] = {}
        self._module_cache: dict[str, wgpu.GPUShaderModule] = {}
        self._pipeline_cache: dict[str, wgpu.GPURenderPipeline] = {}

        # Find the asset directory relative to this file
        # catley/backends/wgpu/shader_manager.py -> ../../assets/shaders/
        self.shader_dir = (
            Path(__file__).parent.parent.parent.parent / "assets" / "shaders"
        )

        if not self.shader_dir.exists():
            raise FileNotFoundError(f"Shader directory not found: {self.shader_dir}")

        logger.debug(
            f"WGPUShaderManager initialized with shader directory: {self.shader_dir}"
        )

    def load_shader_source(self, shader_path: str) -> str:
        """Load shader source code from a WGSL file.

        Args:
            shader_path: Path to shader file relative to assets/shaders/
                        (e.g., "wgsl/screen/main.wgsl",
                         "wgsl/lighting/point_light.wgsl")

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
            logger.debug(f"Loaded WGSL shader: {shader_path}")
            return source

        except Exception as e:
            raise RuntimeError(f"Failed to load shader {shader_path}: {e}") from e

    def create_shader_module(
        self,
        shader_path: str,
        cache_key: str | None = None,
    ) -> wgpu.GPUShaderModule:
        """Create a WGPU shader module from a WGSL file.

        Args:
            shader_path: Path to WGSL shader file (e.g., "wgsl/screen/main.wgsl")
            cache_key: Optional cache key. If None, uses shader_path

        Returns:
            Compiled WGPU shader module

        Raises:
            RuntimeError: If shader compilation fails
        """
        cache_key = cache_key or shader_path

        if cache_key in self._module_cache:
            return self._module_cache[cache_key]

        try:
            source = self.load_shader_source(shader_path)

            module = self.device.create_shader_module(
                code=source,
                label=f"shader_module_{cache_key}",
            )

            self._module_cache[cache_key] = module
            logger.debug(f"Created WGSL shader module: {cache_key}")
            return module

        except Exception as e:
            logger.error(f"Failed to create shader module {cache_key}: {e}")
            logger.debug(f"Shader path: {shader_path}")
            raise RuntimeError(f"WGSL shader compilation failed: {e}") from e

    def create_render_pipeline(
        self,
        vertex_shader_path: str,
        fragment_shader_path: str,
        vertex_layout: list[dict],
        bind_group_layouts: list[wgpu.GPUBindGroupLayout],
        primitive_topology: str = "triangle-list",
        targets: list[dict] | None = None,
        cache_key: str | None = None,
    ) -> wgpu.GPURenderPipeline:
        """Create a WGPU render pipeline from vertex and fragment shaders.

        Args:
            vertex_shader_path: Path to vertex shader (e.g., "wgsl/screen/main.wgsl")
            fragment_shader_path: Path to fragment shader
            vertex_layout: List of vertex buffer layout descriptors
            bind_group_layouts: List of bind group layouts for uniforms/textures
            primitive_topology: Primitive topology (default: "triangle-list")
            targets: Color target descriptors (default: RGBA8 unorm)
            cache_key: Optional cache key for the pipeline

        Returns:
            Compiled WGPU render pipeline

        Raises:
            RuntimeError: If pipeline creation fails
        """
        cache_key = cache_key or f"{vertex_shader_path}+{fragment_shader_path}"

        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]

        # Default color targets
        if targets is None:
            targets = [
                {
                    "format": "bgra8unorm",  # Common swap chain format
                    "blend": {
                        "color": {
                            "operation": "add",
                            "src_factor": "src-alpha",
                            "dst_factor": "one-minus-src-alpha",
                        },
                        "alpha": {
                            "operation": "add",
                            "src_factor": "one",
                            "dst_factor": "one-minus-src-alpha",
                        },
                    },
                }
            ]

        try:
            # Create shader modules
            vertex_module = self.create_shader_module(vertex_shader_path)
            fragment_module = self.create_shader_module(fragment_shader_path)

            # Create pipeline layout
            pipeline_layout = self.device.create_pipeline_layout(
                bind_group_layouts=bind_group_layouts,
                label=f"pipeline_layout_{cache_key}",
            )

            # Create render pipeline
            pipeline = self.device.create_render_pipeline(
                layout=pipeline_layout,
                vertex={
                    "module": vertex_module,
                    "entry_point": "vs_main",
                    "buffers": vertex_layout,
                },
                fragment={
                    "module": fragment_module,
                    "entry_point": "fs_main",
                    "targets": targets,
                },
                primitive={
                    "topology": primitive_topology,
                    "strip_index_format": None,
                    "front_face": "ccw",
                    "cull_mode": "none",
                },
                multisample={
                    "count": 1,
                    "mask": 0xFFFFFFFF,
                    "alpha_to_coverage_enabled": False,
                },
                label=f"render_pipeline_{cache_key}",
            )

            self._pipeline_cache[cache_key] = pipeline
            logger.debug(f"Created render pipeline: {cache_key}")
            return pipeline

        except Exception as e:
            logger.error(f"Failed to create render pipeline {cache_key}: {e}")
            logger.debug(f"Vertex shader: {vertex_shader_path}")
            logger.debug(f"Fragment shader: {fragment_shader_path}")
            raise RuntimeError(f"Render pipeline creation failed: {e}") from e

    def create_bind_group_layout(
        self,
        entries: list[dict],
        label: str | None = None,
    ) -> wgpu.GPUBindGroupLayout:
        """Create a bind group layout for uniforms and textures.

        Args:
            entries: List of bind group layout entries
            label: Optional debug label

        Returns:
            WGPU bind group layout
        """
        return self.device.create_bind_group_layout(
            entries=entries,
            label=label or "",
        )

    def create_bind_group(
        self,
        layout: wgpu.GPUBindGroupLayout,
        entries: list[dict],
        label: str | None = None,
    ) -> wgpu.GPUBindGroup:
        """Create a bind group for uniforms and textures.

        Args:
            layout: The bind group layout
            entries: List of resource bindings
            label: Optional debug label

        Returns:
            WGPU bind group
        """
        return self.device.create_bind_group(
            layout=layout,
            entries=entries,
            label=label or "",
        )

    def clear_cache(self) -> None:
        """Clear all cached shaders and pipelines."""
        self._shader_cache.clear()
        self._module_cache.clear()
        # Note: Pipelines don't need explicit cleanup in WGPU
        self._pipeline_cache.clear()
        logger.debug("WGPU shader cache cleared")

    def reload_shader(self, shader_path: str) -> None:
        """Reload a specific shader from disk, clearing its cache.

        Args:
            shader_path: Path to shader file to reload
        """
        if shader_path in self._shader_cache:
            del self._shader_cache[shader_path]
            logger.debug(f"Cleared cache for shader: {shader_path}")

        # Clear any modules that use this shader
        modules_to_clear = [key for key in self._module_cache if shader_path in key]
        for key in modules_to_clear:
            del self._module_cache[key]
            logger.debug(f"Cleared module cache for: {key}")

        # Clear any pipelines that use this shader
        pipelines_to_clear = [key for key in self._pipeline_cache if shader_path in key]
        for key in pipelines_to_clear:
            del self._pipeline_cache[key]
            logger.debug(f"Cleared pipeline cache for: {key}")

"""WGPU resource management - buffers, textures, pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import wgpu

if TYPE_CHECKING:
    from collections.abc import Hashable


class WGPUResourceManager:
    """Manages WGPU resources like buffers, textures, and pipelines.

    This class provides centralized caching and management of WGPU resources
    to avoid redundant allocations and improve performance. It handles:
    - Texture caching for commonly used sizes
    - Buffer caching for vertex/index data
    - Render pass attachment management
    - Resource cleanup and lifecycle management

    The resource manager is designed to be shared between different WGPU
    components like WGPUGraphicsContext and WGPUScreenRenderer.
    """

    def __init__(self, device: wgpu.GPUDevice, queue: wgpu.GPUQueue) -> None:
        """Initialize the resource manager.

        Args:
            device: The WGPU device to create resources with
            queue: The WGPU queue for resource updates
        """
        self.device = device
        self.queue = queue

        # Cache for render target textures keyed by (width, height, format)
        self._texture_cache: dict[tuple[int, int, str], wgpu.GPUTexture] = {}

        # Track custom cache keys for specialized textures
        self._custom_texture_cache: dict[Hashable, wgpu.GPUTexture] = {}

        # Cache for vertex buffers by usage and size
        self._buffer_cache: dict[tuple[int, int], wgpu.GPUBuffer] = {}

        # Track texture views for render targets
        self._texture_view_cache: dict[int, wgpu.GPUTextureView] = {}

    def get_or_create_render_texture(
        self,
        width: int,
        height: int,
        texture_format: str = "rgba8unorm",
        usage: int = wgpu.TextureUsage.RENDER_ATTACHMENT
        | wgpu.TextureUsage.TEXTURE_BINDING,
    ) -> wgpu.GPUTexture:
        """Get or create a cached render target texture for the given dimensions.

        This method prevents per-frame GPU resource creation/destruction by
        caching textures based on their dimensions and format.

        Args:
            width: Width of the texture in pixels
            height: Height of the texture in pixels
            format: Texture format (default: "rgba8unorm")
            usage: Texture usage flags

        Returns:
            WGPU texture for rendering
        """
        cache_key = (width, height, texture_format)

        # Check if we have a cached texture for these dimensions
        if cache_key in self._texture_cache:
            return self._texture_cache[cache_key]

        # Create new texture
        texture = self.device.create_texture(
            size=(width, height, 1),
            format=texture_format,  # type: ignore
            usage=usage,  # type: ignore
        )

        # Cache it for future use
        self._texture_cache[cache_key] = texture
        return texture

    def get_or_create_buffer(
        self, size: int, usage: int, label: str | None = None
    ) -> wgpu.GPUBuffer:
        """Get or create a cached buffer for the given size and usage.

        Args:
            size: Size of the buffer in bytes
            usage: Buffer usage flags
            label: Optional debug label

        Returns:
            WGPU buffer
        """
        cache_key = (size, usage)

        if cache_key in self._buffer_cache:
            return self._buffer_cache[cache_key]

        # Create new buffer
        buffer = self.device.create_buffer(
            size=size,
            usage=usage,  # type: ignore
            label=label or "",
        )

        # Cache it for future use
        self._buffer_cache[cache_key] = buffer
        return buffer

    def get_texture_view(
        self, texture: wgpu.GPUTexture, texture_format: str | None = None
    ) -> wgpu.GPUTextureView:
        """Get or create a cached texture view for a texture.

        Args:
            texture: The texture to create a view for
            format: Optional format override

        Returns:
            Texture view for the given texture
        """
        # Use texture object id as cache key
        texture_id = id(texture)

        if texture_id in self._texture_view_cache:
            return self._texture_view_cache[texture_id]

        # Create new texture view
        if texture_format:
            view = texture.create_view(format=texture_format)  # type: ignore
        else:
            view = texture.create_view()
        self._texture_view_cache[texture_id] = view
        return view

    def create_atlas_texture(
        self, width: int, height: int, data: bytes, texture_format: str = "rgba8unorm"
    ) -> wgpu.GPUTexture:
        """Create a texture from raw image data (typically for atlas textures).

        Args:
            width: Width of the texture
            height: Height of the texture
            data: Raw texture data bytes
            format: Texture format

        Returns:
            WGPU texture loaded with the provided data
        """
        texture = self.device.create_texture(
            size=(width, height, 1),
            format=texture_format,  # type: ignore
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,  # type: ignore
        )

        # Upload the texture data
        self.queue.write_texture(
            destination={
                "texture": texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            data=memoryview(data),  # type: ignore
            data_layout={
                "offset": 0,
                "bytes_per_row": width * 4,  # 4 bytes per pixel for RGBA
                "rows_per_image": height,
            },
            size=(width, height, 1),
        )

        return texture

    def release_all(self) -> None:
        """Release all cached GPU resources.

        This should be called when shutting down or when you need to free
        GPU memory. All cached textures and buffers will be destroyed.
        """
        # Release texture cache
        for texture in self._texture_cache.values():
            texture.destroy()
        self._texture_cache.clear()

        # Release custom texture cache
        for texture in self._custom_texture_cache.values():
            texture.destroy()
        self._custom_texture_cache.clear()

        # Release buffer cache
        for buffer in self._buffer_cache.values():
            buffer.destroy()
        self._buffer_cache.clear()

        # Clear texture view cache (views don't need explicit cleanup)
        self._texture_view_cache.clear()

    def get_cache_info(self) -> dict[str, int]:
        """Get information about cached resources.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "texture_count": len(self._texture_cache),
            "custom_texture_count": len(self._custom_texture_cache),
            "buffer_count": len(self._buffer_cache),
            "texture_view_count": len(self._texture_view_cache),
            "total_resource_count": len(self._texture_cache)
            + len(self._custom_texture_cache)
            + len(self._buffer_cache),
        }

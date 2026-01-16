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
    - Shared bind group layouts and samplers for renderer initialization

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

        # Cache for render target textures keyed by (width, height, format, suffix)
        self._texture_cache: dict[tuple[int, int, str, str], wgpu.GPUTexture] = {}

        # Track custom cache keys for specialized textures
        self._custom_texture_cache: dict[Hashable, wgpu.GPUTexture] = {}

        # Cache for vertex buffers by usage and size
        self._buffer_cache: dict[tuple[int, int], wgpu.GPUBuffer] = {}

        # Track texture views for render targets
        self._texture_view_cache: dict[int, wgpu.GPUTextureView] = {}

        # Shared bind group layouts and samplers (created lazily)
        self._standard_bind_group_layout: wgpu.GPUBindGroupLayout | None = None
        self._nearest_sampler: wgpu.GPUSampler | None = None
        self._linear_sampler: wgpu.GPUSampler | None = None

    @property
    def standard_bind_group_layout(self) -> wgpu.GPUBindGroupLayout:
        """Get the shared standard bind group layout for texture rendering.

        This layout is used by most renderers and includes:
        - binding 0: uniform buffer (vertex + fragment)
        - binding 1: 2d texture (fragment)
        - binding 2: filtering sampler (fragment)

        Creating this once and sharing it across renderers saves GPU calls
        during initialization.
        """
        if self._standard_bind_group_layout is None:
            self._standard_bind_group_layout = self.device.create_bind_group_layout(
                entries=[
                    {
                        "binding": 0,
                        "visibility": wgpu.ShaderStage.VERTEX
                        | wgpu.ShaderStage.FRAGMENT,
                        "buffer": {"type": wgpu.BufferBindingType.uniform},
                    },
                    {
                        "binding": 1,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "texture": {
                            "sample_type": wgpu.TextureSampleType.float,
                            "view_dimension": "2d",
                        },
                    },
                    {
                        "binding": 2,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "sampler": {"type": wgpu.SamplerBindingType.filtering},
                    },
                ],
                label="shared_standard_bind_group_layout",
            )
        return self._standard_bind_group_layout

    @property
    def nearest_sampler(self) -> wgpu.GPUSampler:
        """Get the shared nearest-neighbor sampler.

        Used by most renderers for pixel-perfect rendering of sprites and tiles.
        """
        if self._nearest_sampler is None:
            self._nearest_sampler = self.device.create_sampler(
                mag_filter=wgpu.FilterMode.nearest,  # type: ignore
                min_filter=wgpu.FilterMode.nearest,  # type: ignore
                mipmap_filter=wgpu.MipmapFilterMode.nearest,  # type: ignore
                address_mode_u=wgpu.AddressMode.clamp_to_edge,  # type: ignore
                address_mode_v=wgpu.AddressMode.clamp_to_edge,  # type: ignore
                label="shared_nearest_sampler",
            )
        return self._nearest_sampler

    @property
    def linear_sampler(self) -> wgpu.GPUSampler:
        """Get the shared linear filtering sampler.

        Used for smooth gradients and environmental effects.
        """
        if self._linear_sampler is None:
            self._linear_sampler = self.device.create_sampler(
                mag_filter=wgpu.FilterMode.linear,  # type: ignore
                min_filter=wgpu.FilterMode.linear,  # type: ignore
                mipmap_filter=wgpu.MipmapFilterMode.linear,  # type: ignore
                address_mode_u=wgpu.AddressMode.clamp_to_edge,  # type: ignore
                address_mode_v=wgpu.AddressMode.clamp_to_edge,  # type: ignore
                label="shared_linear_sampler",
            )
        return self._linear_sampler

    def get_or_create_render_texture(
        self,
        width: int,
        height: int,
        texture_format: str = "rgba8unorm",
        cache_key_suffix: str = "",
        usage: int = wgpu.TextureUsage.RENDER_ATTACHMENT
        | wgpu.TextureUsage.TEXTURE_BINDING,
    ) -> wgpu.GPUTexture:
        """Get or create a cached render target texture for the given dimensions.

        This method prevents per-frame GPU resource creation/destruction by
        caching textures based on their dimensions and format.

        Args:
            width: Width of the texture in pixels
            height: Height of the texture in pixels
            texture_format: Texture format (default: "rgba8unorm")
            cache_key_suffix: Additional suffix for unique caching (e.g., overlay ID)
            usage: Texture usage flags

        Returns:
            WGPU texture for rendering
        """
        cache_key = (width, height, texture_format, cache_key_suffix)

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

        # Clear shared resources (bind group layouts and samplers don't need
        # explicit cleanup in WGPU, just clear references)
        self._standard_bind_group_layout = None
        self._nearest_sampler = None
        self._linear_sampler = None

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

"""Shared resource manager for ModernGL backend.

This module provides centralized management of GPU resources like framebuffers,
textures, and other cached resources to prevent redundant allocations and improve
performance across different ModernGL components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import moderngl

if TYPE_CHECKING:
    from collections.abc import Hashable


class ModernGLResourceManager:
    """Manages shared GPU resources for the ModernGL backend.

    This class provides centralized caching and management of GPU resources
    to avoid redundant allocations and improve performance. It handles:
    - Framebuffer object (FBO) caching based on dimensions
    - Texture caching for commonly used sizes
    - Resource cleanup and lifecycle management

    The resource manager is designed to be shared between different ModernGL
    components like ModernGLGraphicsContext and GPULightingSystem.
    """

    def __init__(self, mgl_context: moderngl.Context) -> None:
        """Initialize the resource manager.

        Args:
            mgl_context: The ModernGL context to create resources with
        """
        self.mgl_context = mgl_context

        # Cache for framebuffer objects keyed by (width, height)
        self._fbo_cache: dict[
            tuple[int, int], tuple[moderngl.Framebuffer, moderngl.Texture]
        ] = {}

        # Track custom cache keys for specialized framebuffers
        self._custom_fbo_cache: dict[
            Hashable, tuple[moderngl.Framebuffer, moderngl.Texture]
        ] = {}

        # Cache for FBOs attached to existing textures (texture id -> FBO)
        self._texture_fbo_cache: dict[int, moderngl.Framebuffer] = {}

    def get_or_create_fbo(
        self, width: int, height: int, components: int = 4, dtype: str = "f1"
    ) -> tuple[moderngl.Framebuffer, moderngl.Texture]:
        """Get or create a cached framebuffer for the given dimensions.

        This method prevents per-frame GPU resource creation/destruction by
        caching framebuffers based on their dimensions and format.

        Args:
            width: Width of the framebuffer in pixels
            height: Height of the framebuffer in pixels
            components: Number of color components (default: 4 for RGBA)
            dtype: Data type for the texture (default: "f1" for unsigned byte)

        Returns:
            Tuple of (framebuffer, texture) for rendering
        """
        cache_key = (width, height, components, dtype)

        # Check if we have a cached FBO for these dimensions
        if cache_key in self._custom_fbo_cache:
            return self._custom_fbo_cache[cache_key]

        # Create new FBO and texture
        texture = self.mgl_context.texture((width, height), components, dtype=dtype)
        fbo = self.mgl_context.framebuffer(color_attachments=[texture])

        # Cache them for future use
        self._custom_fbo_cache[cache_key] = (fbo, texture)
        return fbo, texture

    def get_or_create_simple_fbo(
        self, width: int, height: int
    ) -> tuple[moderngl.Framebuffer, moderngl.Texture]:
        """Get or create a simple RGBA8 framebuffer (backward compatibility).

        This method maintains compatibility with the existing FBO caching
        interface used by ModernGLGraphicsContext.

        Args:
            width: Width of the framebuffer in pixels
            height: Height of the framebuffer in pixels

        Returns:
            Tuple of (framebuffer, texture) for rendering
        """
        cache_key = (width, height)

        if cache_key in self._fbo_cache:
            return self._fbo_cache[cache_key]

        # Create new FBO and texture with default RGBA8 format
        texture = self.mgl_context.texture((width, height), 4)
        fbo = self.mgl_context.framebuffer(color_attachments=[texture])

        # Cache them for future use
        self._fbo_cache[cache_key] = (fbo, texture)
        return fbo, texture

    def release_all(self) -> None:
        """Release all cached GPU resources.

        This should be called when shutting down or when you need to free
        GPU memory. All cached framebuffers and textures will be released.
        """
        # Release simple FBO cache
        for fbo, texture in self._fbo_cache.values():
            fbo.release()
            texture.release()
        self._fbo_cache.clear()

        # Release custom FBO cache
        for fbo, texture in self._custom_fbo_cache.values():
            fbo.release()
            texture.release()
        self._custom_fbo_cache.clear()

        # Release texture FBO cache
        for fbo in self._texture_fbo_cache.values():
            fbo.release()
        self._texture_fbo_cache.clear()

    def release_fbo(self, width: int, height: int) -> None:
        """Release a specific cached framebuffer.

        Args:
            width: Width of the framebuffer to release
            height: Height of the framebuffer to release
        """
        cache_key = (width, height)
        if cache_key in self._fbo_cache:
            fbo, texture = self._fbo_cache[cache_key]
            fbo.release()
            texture.release()
            del self._fbo_cache[cache_key]

    def get_or_create_fbo_for_texture(
        self,
        texture: moderngl.Texture,
        depth_attachment: moderngl.Renderbuffer | None = None,
    ) -> moderngl.Framebuffer:
        """Get or create a cached framebuffer for an existing texture.

        This method caches FBOs based on the texture's internal GL handle,
        preventing recreation of FBOs for the same texture.

        Args:
            texture: The texture to attach as color attachment
            depth_attachment: Optional depth buffer attachment (not cached if provided)

        Returns:
            A cached or new framebuffer with the given texture attached
        """
        # If depth attachment is provided, we can't cache (would need composite key)
        if depth_attachment is not None:
            return self.mgl_context.framebuffer(
                color_attachments=[texture], depth_attachment=depth_attachment
            )

        # Use texture's GL object ID as cache key
        texture_id = texture.glo

        # Check cache first
        if texture_id in self._texture_fbo_cache:
            return self._texture_fbo_cache[texture_id]

        # Create new FBO and cache it
        fbo = self.mgl_context.framebuffer(color_attachments=[texture])
        self._texture_fbo_cache[texture_id] = fbo
        return fbo

    def get_cache_info(self) -> dict[str, int]:
        """Get information about cached resources.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "simple_fbo_count": len(self._fbo_cache),
            "custom_fbo_count": len(self._custom_fbo_cache),
            "texture_fbo_count": len(self._texture_fbo_cache),
            "total_fbo_count": len(self._fbo_cache)
            + len(self._custom_fbo_cache)
            + len(self._texture_fbo_cache),
        }

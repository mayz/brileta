"""Floating text effect system for game feedback.

Displays rising, fading text above actors when significant outcomes occur.
Uses Pillow + Cozette font for rendering, enabling full Unicode support
including emoji-like glyphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from brileta import colors, config
from brileta.events import (
    FloatingTextEvent,
    FloatingTextSize,
    FloatingTextValence,
    subscribe_to_event,
)
from brileta.types import ActorId, DeltaTime, Opacity, ViewOffset

if TYPE_CHECKING:
    from brileta.game.game_world import GameWorld
    from brileta.view.render.graphics import GraphicsContext
    from brileta.view.render.viewport import ViewportSystem


# Color mapping for valence
VALENCE_COLORS: dict[FloatingTextValence, colors.Color] = {
    FloatingTextValence.NEGATIVE: (255, 130, 60),  # Orange-red (distinct from UI red)
    FloatingTextValence.POSITIVE: (80, 255, 80),  # Soft green
    FloatingTextValence.NEUTRAL: (255, 255, 100),  # Soft yellow
}

# Default duration for floating text (in seconds)
DEFAULT_DURATION: float = 0.7


@dataclass
class FloatingText:
    """A single floating text instance with cached texture and animation state.

    Renders text once using Pillow + Cozette, caches the texture, then
    draws it each frame with position offset and alpha modulation.
    """

    text: str
    target_actor_id: ActorId
    valence: FloatingTextValence
    size: FloatingTextSize
    world_x: float
    world_y: float
    color_override: colors.Color | None = None  # Overrides valence color
    bubble: bool = False

    # Cached texture (created on first render)
    _texture: Any = field(default=None, init=False, repr=False)
    _texture_width: int = field(default=0, init=False)
    _texture_height: int = field(default=0, init=False)

    # Animation state
    elapsed: float = 0.0
    duration: float = DEFAULT_DURATION
    rise_distance_tiles: float = 1.5

    # Computed per update
    _current_alpha: float = field(default=1.0, init=False)
    _current_offset_y: float = field(default=0.0, init=False)

    def update(self, delta_time: DeltaTime) -> bool:
        """Update animation state. Returns True if animation is complete."""
        self.elapsed += delta_time

        if self.elapsed >= self.duration:
            return True

        progress = self.elapsed / self.duration

        # Cubic ease-out for smooth deceleration
        ease_progress = 1 - (1 - progress) ** 3

        # Rise upward (negative Y offset in tiles)
        self._current_offset_y = -self.rise_distance_tiles * ease_progress

        # Fade out in the second half
        if progress > 0.5:
            fade_progress = (progress - 0.5) / 0.5
            self._current_alpha = 1.0 - fade_progress
        else:
            self._current_alpha = 1.0

        return False

    @property
    def alpha(self) -> float:
        """Current opacity (0.0 to 1.0)."""
        return self._current_alpha

    @property
    def offset_y(self) -> float:
        """Current Y offset in tiles (negative = upward)."""
        return self._current_offset_y

    def get_color(self) -> colors.Color:
        """Get display color. Uses color_override if set, otherwise valence color."""
        if self.color_override is not None:
            return self.color_override
        return VALENCE_COLORS.get(
            self.valence, VALENCE_COLORS[FloatingTextValence.NEUTRAL]
        )

    def ensure_texture(
        self, graphics: GraphicsContext, font: ImageFont.FreeTypeFont
    ) -> None:
        """Create the texture if not already cached."""
        if self._texture is not None:
            return

        color = self.get_color()

        # Measure text to create appropriately sized image
        bbox = font.getbbox(self.text)
        text_width = int(bbox[2] - bbox[0])
        text_height = int(bbox[3] - bbox[1])

        # Add padding
        padding_x = 6 if self.bubble else 4
        padding_y = 4
        tail_height = 6 if self.bubble else 0
        img_width = text_width + padding_x * 2
        img_height = text_height + padding_y * 2 + tail_height

        # Ensure minimum size
        img_width = max(1, img_width)
        img_height = max(1, img_height)

        # Create RGBA image with transparent background
        image = PILImage.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        if self.bubble:
            bubble_fill = (20, 20, 20, 210)
            bubble_outline = (235, 235, 235, 230)
            corner_radius = 4
            bubble_rect = (0, 0, img_width - 1, img_height - tail_height - 1)
            draw.rounded_rectangle(
                bubble_rect,
                radius=corner_radius,
                fill=bubble_fill,
                outline=bubble_outline,
                width=1,
            )
            tail_center_x = img_width // 2
            tail_base_y = img_height - tail_height - 1
            tail_tip_y = img_height - 1
            draw.polygon(
                [
                    (tail_center_x - 4, tail_base_y),
                    (tail_center_x + 4, tail_base_y),
                    (tail_center_x, tail_tip_y),
                ],
                fill=bubble_fill,
                outline=bubble_outline,
            )

        # Draw text centered in image
        text_x = padding_x - int(bbox[0])  # Adjust for glyph offset
        text_y = padding_y - int(bbox[1])
        draw.text((text_x, text_y), self.text, font=font, fill=(*color, 255))

        # Convert to numpy and create texture
        pixels = np.array(image, dtype=np.uint8)
        pixels = np.ascontiguousarray(pixels)

        self._texture = graphics.texture_from_numpy(pixels, transparent=True)
        self._texture_width = img_width
        self._texture_height = img_height

    def cleanup_texture(self, graphics: GraphicsContext) -> None:
        """Release texture resources via the graphics context.

        Args:
            graphics: The graphics context that created the texture.
        """
        if self._texture is not None:
            graphics.release_texture(self._texture)
            self._texture = None


class FloatingTextManager:
    """Manages floating text lifecycle: creation, animation, rendering, cleanup.

    Subscribes to FloatingTextEvent, maintains active texts, and coordinates
    with the graphics system for rendering.

    Texture cleanup is deferred to render() when graphics context is available,
    so _handle_event() and update() can safely queue texts for cleanup without
    needing the graphics context.
    """

    def __init__(self, max_texts: int = 20) -> None:
        """Initialize the floating text manager.

        Args:
            max_texts: Maximum concurrent floating texts. Oldest are
                removed when limit is reached.
        """
        self.max_texts = max_texts
        self._texts: list[FloatingText] = []
        self._pending_cleanup: list[FloatingText] = []
        self._fonts: dict[FloatingTextSize, ImageFont.FreeTypeFont] = {}

        subscribe_to_event(FloatingTextEvent, self._handle_event)

    def _ensure_font(self, size: FloatingTextSize) -> ImageFont.FreeTypeFont:
        """Lazy-load the Cozette font for the given size preset."""
        if size not in self._fonts:
            font_path = config.UI_FONT_PATH
            self._fonts[size] = ImageFont.truetype(str(font_path), size.value)
        return self._fonts[size]

    def _handle_event(self, event: FloatingTextEvent) -> None:
        """Create a new FloatingText from an event."""
        # Enforce limit by removing oldest (defer cleanup to render())
        while len(self._texts) >= self.max_texts:
            old = self._texts.pop(0)
            self._pending_cleanup.append(old)

        # Use event duration if provided, otherwise use default
        duration = event.duration if event.duration is not None else DEFAULT_DURATION

        text = FloatingText(
            text=event.text,
            target_actor_id=event.target_actor_id,
            valence=event.valence,
            size=event.size,
            world_x=float(event.world_x),
            world_y=float(event.world_y),
            color_override=event.color,
            duration=duration,
            bubble=event.bubble,
        )
        self._texts.append(text)

    def update(self, delta_time: DeltaTime) -> None:
        """Update all floating texts, removing completed ones.

        Completed texts are queued for cleanup, which happens in render()
        when the graphics context is available.
        """
        remaining: list[FloatingText] = []

        for text in self._texts:
            if not text.update(delta_time):
                remaining.append(text)
            else:
                self._pending_cleanup.append(text)

        self._texts = remaining

    def render(
        self,
        graphics: GraphicsContext,
        viewport_system: ViewportSystem,
        view_offset: ViewOffset,
        game_world: GameWorld,
    ) -> None:
        """Render all active floating texts.

        Also handles deferred texture cleanup for texts that were removed
        since the last render call.
        """
        # Clean up textures from removed texts (deferred from update/_handle_event)
        for text in self._pending_cleanup:
            text.cleanup_texture(graphics)
        self._pending_cleanup.clear()

        for text in self._texts:
            font = self._ensure_font(text.size)
            self._render_single(
                text, graphics, viewport_system, view_offset, game_world, font
            )

    def _render_single(
        self,
        text: FloatingText,
        graphics: GraphicsContext,
        viewport_system: ViewportSystem,
        view_offset: ViewOffset,
        game_world: GameWorld,
        font: ImageFont.FreeTypeFont,
    ) -> None:
        """Render a single floating text."""
        # Get actor position (or use stored position if actor is gone)
        world_x = text.world_x
        world_y = text.world_y

        actor = game_world.get_actor_by_id(text.target_actor_id)
        if actor is not None:
            world_x = float(actor.x)
            world_y = float(actor.y)

        # Apply rise offset
        render_y = world_y + text.offset_y

        # Check visibility
        if not viewport_system.is_visible(int(world_x), int(render_y)):
            return

        # Ensure texture is created
        text.ensure_texture(graphics, font)
        if text._texture is None:
            return

        # Convert world position to viewport coordinates
        vp_x, vp_y = viewport_system.world_to_screen_float(world_x, render_y)

        # Then to root console coordinates
        root_x = view_offset[0] + vp_x
        root_y = view_offset[1] + vp_y

        # Then to screen pixels
        screen_x, screen_y = graphics.console_to_screen_coords(root_x, root_y)

        # Center the texture horizontally on the tile
        tile_w, _ = graphics.tile_dimensions
        centered_x = screen_x + (tile_w - text._texture_width) / 2
        centered_y = screen_y - text._texture_height  # Above the tile

        # Draw with alpha
        graphics.draw_texture_alpha(
            text._texture,
            centered_x,
            centered_y,
            Opacity(text.alpha),
        )

    @property
    def active_count(self) -> int:
        """Number of currently active floating texts."""
        return len(self._texts)

    def clear(self, graphics: GraphicsContext | None = None) -> None:
        """Clear all floating texts. Use for game state resets.

        Args:
            graphics: If provided, textures are released immediately.
                If None, textures are queued for cleanup on next render().
        """
        if graphics is not None:
            # Clean up immediately when graphics context is available
            for text in self._texts:
                text.cleanup_texture(graphics)
            for text in self._pending_cleanup:
                text.cleanup_texture(graphics)
            self._pending_cleanup.clear()
        else:
            # Queue for cleanup on next render()
            self._pending_cleanup.extend(self._texts)
        self._texts.clear()

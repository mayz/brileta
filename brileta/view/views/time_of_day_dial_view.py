"""Compact HUD dial for the game clock's day/night cycle."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

from brileta import config
from brileta.game.clock import (
    MINUTES_PER_DAY,
    compute_low_sun_time_fraction,
    compute_sky_lighting,
)
from brileta.types import InterpolationAlpha, PixelPos, saturate
from brileta.util.caching import ResourceCache
from brileta.view.render.graphics import GraphicsContext

from .base import View

if TYPE_CHECKING:
    from brileta.controller import Controller

type _IconKind = Literal["sun", "moon"]


class TimeOfDayDialView(View):
    """Transparent HUD view showing the game clock as a sun/moon dial."""

    _DIAL_FILL: tuple[int, int, int, int] = (6, 8, 13, 154)
    _DIAL_OUTLINE: tuple[int, int, int, int] = (226, 214, 170, 70)
    _NIGHT_RING: tuple[int, int, int, int] = (72, 88, 142, 190)
    _DAY_RING: tuple[int, int, int, int] = (247, 201, 88, 230)
    _TWILIGHT_RING: tuple[int, int, int, int] = (255, 132, 82, 230)
    _ICON_SHADOW: tuple[int, int, int, int] = (3, 4, 8, 155)
    _BOTTOM_FADE_FRACTION = 0.32
    _ICON_TRANSITION_MAX_FRACTION = 0.018

    def __init__(self, controller: Controller, graphics: GraphicsContext) -> None:
        super().__init__()
        self.controller = controller
        self._graphics = graphics
        self.view_width_px = 0
        self.view_height_px = 0
        self._texture_cache = ResourceCache[tuple, Any](
            name=f"{self.__class__.__name__}Render",
            max_size=2,
            on_evict=lambda tex: self._graphics.release_texture(tex),
        )

    def set_bounds(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Update tile and pixel dimensions for the dial texture."""
        super().set_bounds(x1, y1, x2, y2)
        self.view_width_px = self.width * self.tile_dimensions[0]
        self.view_height_px = self.height * self.tile_dimensions[1]

    def _get_cache_key(self) -> tuple:
        """Return render state that affects the dial texture."""
        clock = self.controller.gw.clock
        minute = round(clock.time_of_day * MINUTES_PER_DAY) % MINUTES_PER_DAY
        return (
            minute,
            self.tile_dimensions,
            self.width,
            self.height,
            config.DAY_FRACTION,
        )

    def draw(self, graphics: GraphicsContext, alpha: InterpolationAlpha) -> None:
        """Render the dial texture when the clock minute or bounds change."""
        _ = alpha
        if not self.visible:
            return
        if self.width <= 0 or self.height <= 0:
            self._cached_texture = None
            return

        self.view_width_px = self.width * self.tile_dimensions[0]
        self.view_height_px = self.height * self.tile_dimensions[1]
        if self.view_width_px <= 0 or self.view_height_px <= 0:
            self._cached_texture = None
            return

        cache_key = self._get_cache_key()
        cached_texture = self._texture_cache.get(cache_key)
        if cached_texture is not None:
            self._cached_texture = cached_texture
            return

        pixels = self._render_pixels()
        texture = graphics.texture_from_numpy(pixels, transparent=True)
        if texture:
            self._texture_cache.store(cache_key, texture)
            self._cached_texture = texture

    def _render_pixels(self) -> np.ndarray:
        """Render the visible top half of the HUD dial into an RGBA pixel buffer."""
        width = max(1, int(self.view_width_px))
        height = max(1, int(self.view_height_px))
        image = PILImage.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image, "RGBA")

        pad = max(3, min(width, height) // 14)
        active_icon_extent_factor = 0.19 * 1.9
        radius = max(12.0, (width / 2.0 - pad) / (1.0 + active_icon_extent_factor))
        # Place the circle's center at the bottom of the shallow viewport.
        # The framebuffer clips the lower half, and a bottom alpha ramp makes
        # the visible arc sink away instead of ending on a hard horizontal cut.
        center: PixelPos = (width / 2.0, height - pad * 0.25)
        outline_width = max(1, round(radius * 0.045))
        draw.ellipse(
            (
                center[0] - radius - pad * 0.35,
                center[1] - radius - pad * 0.35,
                center[0] + radius + pad * 0.35,
                center[1] + radius + pad * 0.35,
            ),
            fill=self._DIAL_FILL,
            outline=self._DIAL_OUTLINE,
            width=outline_width,
        )
        ring_width = max(3, round(radius * 0.13))
        current_time = self.controller.gw.clock.time_of_day
        self._draw_dial_ring(draw, center, radius, ring_width, current_time)

        pixels = np.array(image, dtype=np.uint8)
        self._fade_bottom_edge(pixels)
        image = PILImage.fromarray(pixels)
        draw = ImageDraw.Draw(image, "RGBA")
        self._draw_current_icon(draw, width, height, radius, current_time)
        pixels = np.array(image, dtype=np.uint8)
        return np.ascontiguousarray(pixels)

    def _fade_bottom_edge(self, pixels: np.ndarray) -> None:
        """Fade the bottom of the half-dial so clipped elements disappear softly."""
        height = pixels.shape[0]
        fade_height = max(1, round(height * self._BOTTOM_FADE_FRACTION))
        fade_start = height - fade_height
        for y in range(fade_start, height):
            remaining = (height - 1 - y) / max(1, fade_height - 1)
            pixels[y, :, 3] = (pixels[y, :, 3].astype(np.float32) * remaining).astype(
                np.uint8
            )

    def _draw_dial_ring(
        self,
        draw: ImageDraw.ImageDraw,
        center: PixelPos,
        radius: float,
        ring_width: int,
        current_time: float,
    ) -> None:
        """Draw night/day/twilight arcs and small cardinal ticks."""
        self._draw_time_arc(
            draw,
            center,
            radius,
            0.0,
            1.0,
            self._NIGHT_RING,
            ring_width,
            current_time,
        )

        half_day = config.DAY_FRACTION / 2.0
        sunrise = 0.5 - half_day
        sunset = 0.5 + half_day
        low_sun = min(compute_low_sun_time_fraction(), half_day / 3.0)
        self._draw_time_arc(
            draw,
            center,
            radius,
            sunrise,
            sunrise + low_sun,
            self._TWILIGHT_RING,
            ring_width,
            current_time,
        )
        self._draw_time_arc(
            draw,
            center,
            radius,
            sunrise + low_sun,
            sunset - low_sun,
            self._DAY_RING,
            ring_width,
            current_time,
        )
        self._draw_time_arc(
            draw,
            center,
            radius,
            sunset - low_sun,
            sunset,
            self._TWILIGHT_RING,
            ring_width,
            current_time,
        )

        tick_radius = max(1.5, ring_width * 0.55)
        for tick_time, tick_color in (
            (0.0, (154, 174, 234, 220)),
            (sunrise, (255, 173, 92, 235)),
            (0.5, (255, 240, 158, 245)),
            (sunset, (255, 139, 92, 235)),
        ):
            x, y = self._rotated_dial_position(tick_time, current_time, center, radius)
            draw.ellipse(
                (
                    x - tick_radius,
                    y - tick_radius,
                    x + tick_radius,
                    y + tick_radius,
                ),
                fill=tick_color,
            )

    def _draw_time_arc(
        self,
        draw: ImageDraw.ImageDraw,
        center: PixelPos,
        radius: float,
        start_time: float,
        end_time: float,
        color: tuple[int, int, int, int],
        width: int,
        current_time: float,
    ) -> None:
        """Draw an arc by sampling clock positions along the dial."""
        if end_time < start_time:
            end_time += 1.0
        span = max(0.001, end_time - start_time)
        sample_count = max(8, round(span * 96))
        points = [
            self._rotated_dial_position(
                start_time + span * (i / sample_count),
                current_time,
                center,
                radius,
            )
            for i in range(sample_count + 1)
        ]
        draw.line(points, fill=color, width=width, joint="curve")

    def _draw_current_icon(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        radius: float,
        current_time: float,
    ) -> None:
        """Draw the fixed current-time symbol in the visible half-dial."""
        sky = compute_sky_lighting(current_time)
        icon_x = width / 2.0
        icon_y = height * 0.70
        transition = self._current_icon_transition(current_time)
        if transition is None:
            icon_kind: _IconKind = "sun" if sky.sun_intensity > 0.0 else "moon"
            self._draw_icon_kind(
                draw,
                icon_kind,
                icon_x,
                icon_y,
                height,
                radius,
                sky.sun_color,
                alpha=1.0,
            )
            return

        outgoing, incoming, progress = transition
        travel_y = max(height * 0.32, radius * 0.44)
        outgoing_progress = min(1.0, progress / 0.65)
        incoming_progress = max(0.0, (progress - 0.35) / 0.65)
        if outgoing_progress < 1.0:
            outgoing_eased = self._smoothstep(outgoing_progress)
            self._draw_icon_kind(
                draw,
                outgoing,
                icon_x,
                icon_y + outgoing_eased * travel_y,
                height,
                radius,
                sky.sun_color,
                alpha=1.0 - outgoing_eased,
            )
        if incoming_progress > 0.0:
            incoming_eased = self._smoothstep(incoming_progress)
            self._draw_icon_kind(
                draw,
                incoming,
                icon_x,
                icon_y + (1.0 - incoming_eased) * travel_y,
                height,
                radius,
                sky.sun_color,
                alpha=incoming_eased,
            )

    def _draw_icon_kind(
        self,
        draw: ImageDraw.ImageDraw,
        icon_kind: _IconKind,
        x: float,
        y: float,
        height: int,
        dial_radius: float,
        sun_color: tuple[int, int, int],
        *,
        alpha: float,
    ) -> None:
        """Draw a sun or moon icon in the fixed readout slot."""
        if alpha <= 0.0:
            return
        if icon_kind == "sun":
            icon_radius = max(7.0, min(height * 0.18, dial_radius * 0.29))
            self._draw_sun(draw, x, y, icon_radius, sun_color, alpha=alpha)
        else:
            icon_radius = max(9.0, min(height * 0.25, dial_radius * 0.36))
            self._draw_moon(draw, x, y, icon_radius, alpha=alpha)

    @classmethod
    def _current_icon_transition(
        cls, current_time: float
    ) -> tuple[_IconKind, _IconKind, float] | None:
        """Return outgoing/incoming icon transition state, if one is active."""
        span = cls._icon_transition_half_span()
        if span <= 0.0:
            return None

        half_day = config.DAY_FRACTION / 2.0
        sunrise = 0.5 - half_day
        sunset = 0.5 + half_day
        sunrise_progress = cls._transition_progress(current_time, sunrise, span)
        if sunrise_progress is not None:
            return ("moon", "sun", sunrise_progress)

        sunset_progress = cls._transition_progress(current_time, sunset, span)
        if sunset_progress is not None:
            return ("sun", "moon", sunset_progress)
        return None

    @classmethod
    def _icon_transition_half_span(cls) -> float:
        """Return half the sunrise/sunset icon transition duration in day units."""
        return min(
            compute_low_sun_time_fraction() * 0.45, cls._ICON_TRANSITION_MAX_FRACTION
        )

    @staticmethod
    def _transition_progress(
        current_time: float, event_time: float, half_span: float
    ) -> float | None:
        """Return 0..1 progress through a cyclic transition around ``event_time``."""
        delta = ((current_time - event_time + 0.5) % 1.0) - 0.5
        if abs(delta) > half_span:
            return None
        return (delta + half_span) / (half_span * 2.0)

    @staticmethod
    def _smoothstep(t: float) -> float:
        """Smoothly ease a 0..1 transition value."""
        clamped = saturate(t)
        return clamped * clamped * (3.0 - 2.0 * clamped)

    def _draw_sun(
        self,
        draw: ImageDraw.ImageDraw,
        x: float,
        y: float,
        radius: float,
        fill_color: tuple[int, int, int],
        *,
        alpha: float = 1.0,
    ) -> None:
        """Draw a compact sun glyph."""
        alpha = saturate(alpha)
        fill = (*fill_color, round(255 * alpha))
        ray = (255, 190, 82, round(245 * alpha))
        shadow = self._with_scaled_alpha(self._ICON_SHADOW, alpha)
        ray_inner = radius * 1.18
        ray_outer = radius * 1.55
        ray_width = max(1, round(radius * 0.23))
        for i in range(8):
            angle = (math.tau * i) / 8.0
            ray_line = (
                x + math.cos(angle) * ray_inner,
                y + math.sin(angle) * ray_inner,
                x + math.cos(angle) * ray_outer,
                y + math.sin(angle) * ray_outer,
            )
            draw.line(
                ray_line,
                fill=shadow,
                width=ray_width + 2,
            )
            draw.line(
                ray_line,
                fill=ray,
                width=ray_width,
            )
        draw.ellipse(
            (x - radius - 1, y - radius - 1, x + radius + 1, y + radius + 1),
            fill=shadow,
        )
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)

    def _draw_moon(
        self,
        draw: ImageDraw.ImageDraw,
        x: float,
        y: float,
        radius: float,
        *,
        alpha: float = 1.0,
    ) -> None:
        """Draw a crescent moon glyph."""
        alpha = saturate(alpha)
        fill = (210, 223, 255, round(255 * alpha))
        points = self._crescent_points(x, y, radius)
        shadow_points = [(px + 1.0, py + 1.0) for px, py in points]
        draw.polygon(
            shadow_points, fill=self._with_scaled_alpha(self._ICON_SHADOW, alpha)
        )
        draw.polygon(points, fill=fill)

    @staticmethod
    def _with_scaled_alpha(
        color: tuple[int, int, int, int], alpha: float
    ) -> tuple[int, int, int, int]:
        """Return ``color`` with its alpha channel multiplied by ``alpha``."""
        return (color[0], color[1], color[2], round(color[3] * alpha))

    @staticmethod
    def _crescent_points(x: float, y: float, radius: float) -> list[PixelPos]:
        """Return a crescent silhouette without painting a cutout oval."""
        inner_radius = radius * 0.88
        inner_x = x + radius * 0.48
        outer_angles = [math.radians(deg) for deg in range(92, 269, 8)]
        inner_angles = [math.radians(deg) for deg in range(270, 89, -8)]
        return [
            *[
                (x + math.cos(angle) * radius, y + math.sin(angle) * radius)
                for angle in outer_angles
            ],
            *[
                (
                    inner_x + math.cos(angle) * inner_radius,
                    y + math.sin(angle) * inner_radius,
                )
                for angle in inner_angles
            ],
        ]

    @staticmethod
    def _dial_position(time_of_day: float, center: PixelPos, radius: float) -> PixelPos:
        """Map normalized clock time to a point on the dial ring.

        Noon is at the top, midnight at the bottom, morning is on the left side,
        and evening is on the right side.
        """
        angle = math.radians(90.0 - ((time_of_day % 1.0) - 0.5) * 360.0)
        return (
            center[0] + math.cos(angle) * radius,
            center[1] - math.sin(angle) * radius,
        )

    @classmethod
    def _rotated_dial_position(
        cls,
        time_of_day: float,
        current_time: float,
        center: PixelPos,
        radius: float,
    ) -> PixelPos:
        """Map clock time to the moving dial with current time at the top."""
        display_time = (time_of_day - current_time + 0.5) % 1.0
        return cls._dial_position(display_time, center, radius)

"""Renders persistent presence indicators hovering above NPCs.

Unlike floating text (event-driven, rises and fades), an indicator is polled
each frame from NPC.indicator and drawn as a small speech bubble that stays put
while the NPC's state holds. Bubbles reuse floating text's builder so they look
identical, and are cached per IndicatorKind since the glyph set is tiny.

Indicators draw only for NPCs the player can actually see (FOV): an "!" over an
NPC hidden behind a wall would be a wallhack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PIL import ImageFont

from brileta import config
from brileta.events import FloatingTextSize
from brileta.game.actors.indicators import INDICATOR_STYLES, IndicatorKind
from brileta.types import Opacity, ViewOffset
from brileta.view.render.effects.floating_text import build_text_texture

if TYPE_CHECKING:
    from brileta.game.game_world import GameWorld
    from brileta.view.render.graphics import GraphicsContext
    from brileta.view.render.viewport import ViewportSystem


class IndicatorRenderer:
    """Draws NPC.indicator bubbles above visible NPCs each frame."""

    def __init__(self) -> None:
        self._font: ImageFont.FreeTypeFont | None = None
        # One cached (texture, width, height) per kind. The vocabulary is small
        # and world-independent, so these live for the renderer's lifetime.
        self._cache: dict[IndicatorKind, tuple[Any, int, int]] = {}

    def _ensure_font(self) -> ImageFont.FreeTypeFont:
        if self._font is None:
            self._font = ImageFont.truetype(
                str(config.UI_FONT_PATH), FloatingTextSize.NORMAL.value
            )
        return self._font

    def _ensure_texture(
        self, graphics: GraphicsContext, kind: IndicatorKind
    ) -> tuple[Any, int, int]:
        cached = self._cache.get(kind)
        if cached is None:
            glyph, color = INDICATOR_STYLES[kind]
            cached = build_text_texture(
                graphics, self._ensure_font(), glyph, color, bubble=True
            )
            self._cache[kind] = cached
        return cached

    def render(
        self,
        graphics: GraphicsContext,
        viewport_system: ViewportSystem,
        view_offset: ViewOffset,
        game_world: GameWorld,
    ) -> None:
        """Draw an indicator bubble above every visible NPC that has one."""
        game_map = game_world.game_map
        for actor in game_world.actors:
            kind: IndicatorKind | None = getattr(actor, "indicator", None)
            if kind is None:
                continue

            # FOV gate: never reveal an NPC the player can't see.
            if not game_map.visible[actor.x, actor.y]:
                continue

            # On-screen gate: skip actors outside the current camera bounds.
            if not viewport_system.is_visible(actor.x, actor.y):
                continue

            texture, tex_w, tex_h = self._ensure_texture(graphics, kind)

            # World -> viewport -> root console -> screen pixels (mirrors the
            # bubble path in FloatingTextManager). Bubbles draw at fixed pixel
            # size so they stay readable at any zoom.
            vp_x, vp_y = viewport_system.world_to_screen_float(
                float(actor.x), float(actor.y)
            )
            root_x = view_offset[0] + vp_x
            root_y = view_offset[1] + vp_y
            screen_x, screen_y = graphics.console_to_screen_coords(root_x, root_y)
            scale_x, _ = viewport_system.get_display_scale_factors()

            tile_w, _ = graphics.tile_dimensions
            scaled_tile_w = tile_w * scale_x
            centered_x = screen_x + (scaled_tile_w - tex_w) / 2
            centered_y = screen_y - tex_h  # Above the tile

            graphics.draw_texture_alpha(
                texture,
                centered_x,
                centered_y,
                Opacity(1.0),
                scale_x=1.0,
                scale_y=1.0,
            )

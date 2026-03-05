"""Tree actors and factory helpers for settlement generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from brileta import colors
from brileta.sprites.primitives import clamp
from brileta.sprites.trees import TreeArchetype
from brileta.types import WorldTileCoord
from brileta.util.rng import derive_spatial_seed

from .core import Actor, CharacterLayer

if TYPE_CHECKING:
    from brileta.game.game_world import GameWorld

# Trunk: warm brown, visible below the canopy.
_TRUNK_COLOR: colors.Color = (90, 58, 32)

# Deciduous canopy colors - wide gap so the two-tone shading is clearly visible.
# Outer is a vivid leafy green; inner is a deep shadow green.
_DECIDUOUS_CANOPY_COLOR: colors.Color = (65, 145, 50)
_DECIDUOUS_INNER_COLOR: colors.Color = (25, 70, 20)

# Conifer canopy colors - cooler, darker greens with the same contrast gap.
_CONIFER_CANOPY_COLOR: colors.Color = (40, 115, 40)
_CONIFER_INNER_COLOR: colors.Color = (18, 60, 18)

# Per-tree brightness variation so individuals look distinct.
_CANOPY_BRIGHTNESS_JITTER: int = 18


def _jitter_brightness(
    base: colors.Color,
    brightness_jitter: int,
    jitter_hash: int,
) -> colors.Color:
    """Apply deterministic brightness jitter while preserving hue."""
    if brightness_jitter <= 0:
        return base

    jitter_span = 2 * brightness_jitter + 1
    offset = int(jitter_hash % jitter_span) - brightness_jitter
    r, g, b = base
    return (
        clamp(r + offset),
        clamp(g + offset),
        clamp(b + offset),
    )


def _hash_offset(h: int, bits: int, amplitude: float) -> float:
    """Extract a small signed float from hash bits for position jitter.

    Pulls ``bits`` from the hash and maps them to [-amplitude, +amplitude].
    """
    span = 1 << bits
    raw = (h % span) / (span - 1)  # 0.0 .. 1.0
    return (raw * 2.0 - 1.0) * amplitude


class Tree(Actor):
    """A static tree actor with shared defaults for world behavior.

    The *tree_type* field identifies the archetype used for procedural sprite
    generation.  When the sprite atlas is populated (by the controller after
    world creation), each tree's *sprite_uv* is set and rendering switches
    from glyph compositing to the pre-baked sprite path. Trees also override
    the sprite ground anchor so scaled variants still root naturally in-tile.
    """

    tree_type: TreeArchetype
    _TREE_SHADOW_HEIGHTS: ClassVar[dict[TreeArchetype, int]] = {
        TreeArchetype.DECIDUOUS: 3,
        TreeArchetype.CONIFER: 3,
        TreeArchetype.DEAD: 3,
        TreeArchetype.SAPLING: 2,
    }
    _TREE_GROUND_ANCHOR_Y = 0.62

    def __init__(
        self,
        x: WorldTileCoord,
        y: WorldTileCoord,
        ch: str,
        color: colors.Color,
        game_world: GameWorld | None = None,
        character_layers: list[CharacterLayer] | None = None,
        tree_type: TreeArchetype = TreeArchetype.DECIDUOUS,
        visual_scale: float = 1.0,
    ) -> None:
        physical_shadow_height = self._TREE_SHADOW_HEIGHTS.get(tree_type, 3)
        super().__init__(
            x=x,
            y=y,
            ch=ch,
            color=color,
            name="Tree",
            game_world=game_world,
            blocks_movement=True,
            shadow_height=physical_shadow_height,
            character_layers=character_layers,
            visual_scale=visual_scale,
            # Trees look better when their trunk root is anchored above the
            # tile bottom in our angled top-down perspective.
            sprite_ground_anchor_y=self._TREE_GROUND_ANCHOR_Y,
        )
        self.tree_type = tree_type


def create_deciduous_tree(
    x: WorldTileCoord,
    y: WorldTileCoord,
    game_world: GameWorld | None = None,
) -> Tree:
    """Create a broadleaf tree with a dense, overlapping canopy.

    Layers bottom-to-top: narrow brown trunk, solid center fill, then
    foliage lobes and a crown peak. Per-tree hash jitters both colors
    and layer positions so every tree has a slightly different shape.
    """
    h = derive_spatial_seed(x, y, map_seed=0x0DEC1D00)
    canopy_color = _jitter_brightness(
        _DECIDUOUS_CANOPY_COLOR, _CANOPY_BRIGHTNESS_JITTER, h
    )
    inner_color = _jitter_brightness(
        _DECIDUOUS_INNER_COLOR,
        _CANOPY_BRIGHTNESS_JITTER,
        derive_spatial_seed(x, y, map_seed=0xA1),
    )

    # Per-tree shape jitter: small random offsets to lobe positions so
    # each tree's silhouette is unique. Uses different hash salts per axis.
    jx_l = _hash_offset(derive_spatial_seed(x, y, map_seed=0xF1), 6, 0.06)
    jy_l = _hash_offset(derive_spatial_seed(x, y, map_seed=0xF2), 6, 0.04)
    jx_r = _hash_offset(derive_spatial_seed(x, y, map_seed=0xF3), 6, 0.06)
    jy_r = _hash_offset(derive_spatial_seed(x, y, map_seed=0xF4), 6, 0.04)

    layers: list[CharacterLayer] = [
        # Trunk: narrow brown bar in the bottom third of the tile.
        CharacterLayer(
            "|", _TRUNK_COLOR, offset_x=0.0, offset_y=0.25, scale_x=0.3, scale_y=0.45
        ),
        # Center fill: solid circle that plugs the gap between lobes.
        CharacterLayer(
            "O", inner_color, offset_x=0.0, offset_y=-0.1, scale_x=0.9, scale_y=0.65
        ),
        # Inner canopy mass: large dark base filling the crown area.
        CharacterLayer(
            "#", inner_color, offset_x=0.0, offset_y=-0.08, scale_x=1.1, scale_y=0.7
        ),
        # Left foliage lobe: overhangs slightly left (jittered per-tree).
        CharacterLayer(
            "*",
            canopy_color,
            offset_x=-0.2 + jx_l,
            offset_y=-0.15 + jy_l,
            scale_x=0.85,
            scale_y=0.75,
        ),
        # Right foliage lobe: overhangs slightly right (jittered per-tree).
        CharacterLayer(
            "*",
            canopy_color,
            offset_x=0.2 + jx_r,
            offset_y=-0.12 + jy_r,
            scale_x=0.85,
            scale_y=0.75,
        ),
        # Top: rounded crown peak, extends slightly above the tile.
        CharacterLayer(
            "o", canopy_color, offset_x=0.0, offset_y=-0.3, scale_x=0.7, scale_y=0.6
        ),
    ]
    return Tree(
        x=x,
        y=y,
        ch="#",
        color=canopy_color,
        game_world=game_world,
        character_layers=layers,
    )


def create_conifer_tree(
    x: WorldTileCoord,
    y: WorldTileCoord,
    game_world: GameWorld | None = None,
) -> Tree:
    """Create a pointed conifer from stacked carets.

    Layers: narrow trunk, then 2-3 ``^`` glyphs at staggered heights and
    horizontal offsets to build a layered triangular silhouette. The number
    of branch tiers varies per tree (based on spatial hash) so conifers
    aren't all identical. Per-tree jitter shifts each tier slightly.
    """
    h = derive_spatial_seed(x, y, map_seed=0xC0F1F300)
    canopy_color = _jitter_brightness(
        _CONIFER_CANOPY_COLOR, _CANOPY_BRIGHTNESS_JITTER, h
    )
    inner_color = _jitter_brightness(
        _CONIFER_INNER_COLOR,
        _CANOPY_BRIGHTNESS_JITTER,
        derive_spatial_seed(x, y, map_seed=0xB2),
    )

    # Per-tree horizontal jitter for the lower branch tiers.
    jx_lo = _hash_offset(derive_spatial_seed(x, y, map_seed=0xE1), 5, 0.04)
    jx_hi = _hash_offset(derive_spatial_seed(x, y, map_seed=0xE2), 5, 0.03)

    layers: list[CharacterLayer] = [
        # Trunk: narrow, bottom quarter.
        CharacterLayer(
            "|", _TRUNK_COLOR, offset_x=0.0, offset_y=0.28, scale_x=0.25, scale_y=0.4
        ),
        # Lower branch tier: widest, darkest.
        CharacterLayer(
            "^",
            inner_color,
            offset_x=-0.06 + jx_lo,
            offset_y=0.02,
            scale_x=0.9,
            scale_y=0.5,
        ),
        CharacterLayer(
            "^",
            inner_color,
            offset_x=0.06 - jx_lo,
            offset_y=-0.02,
            scale_x=0.85,
            scale_y=0.48,
        ),
        # Upper tier: narrower, brighter.
        CharacterLayer(
            "^",
            canopy_color,
            offset_x=jx_hi,
            offset_y=-0.2,
            scale_x=0.65,
            scale_y=0.45,
        ),
    ]

    # Some conifers (roughly 40%) get a fifth layer - an extra peak that
    # makes them taller and more distinct.
    if h % 5 < 2:
        layers.append(
            CharacterLayer(
                "^",
                canopy_color,
                offset_x=0.0,
                offset_y=-0.38,
                scale_x=0.45,
                scale_y=0.35,
            )
        )

    return Tree(
        x=x,
        y=y,
        ch="^",
        color=canopy_color,
        game_world=game_world,
        character_layers=layers,
    )

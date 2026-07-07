"""Live sprite atlas for actor sprites, batch-built and incrementally grown.

One code path decides how an actor looks, whether it existed at world creation
or was spawned later. :meth:`ActorSpriteManager.build_for_world` does the fast
world-gen pass (census every actor, generate in parallel, size the atlas with
headroom, pack and upload in one flush) and keeps the atlas alive for the
world's lifetime. :meth:`ActorSpriteManager.ensure_actor_sprites` handles a
single late actor: it generates that actor's pose set, packs it into the live
atlas with a partial upload, and assigns its UVs - reusing the exact per-actor
generate/assign logic the batch pass uses, so a late dog and a world-gen dog
with the same actor id and map seed look identical.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from brileta.game.actors import Actor
from brileta.game.actors.boulder import Boulder
from brileta.game.actors.core import NPC, Character
from brileta.game.actors.trees import Tree
from brileta.types import MapDecorationSeed, SpriteUV
from brileta.util.live_vars import record_time_live_variable

if TYPE_CHECKING:
    import numpy as np

    from brileta.backends.wgpu.sprite_atlas import SpriteAtlas
    from brileta.game.game_world import GameWorld

    from .graphics import GraphicsContext

logger = logging.getLogger(__name__)

# Sprite kind for the two families that pack a 12-frame pose set. Trees and
# boulders are position-seeded, never spawn late, and stay in the batch path.
_ActorSpriteKind = Literal["humanoid", "quadruped"]

# Headroom reserved in the initial atlas: room for this many extra 12-frame
# pose sets beyond the world-gen census, so a few dozen late spawns pack into
# the live atlas without triggering a rebuild. Atlas UVs are normalized, so the
# larger texture changes nothing visually.
_HEADROOM_POSE_SETS = 32


class ActorSpriteManager:
    """Owns the live actor sprite atlas and assigns UVs to actors.

    Held by the Controller for the lifetime of a world. Map regeneration calls
    :meth:`build_for_world` again, which replaces the atlas.
    """

    def __init__(self, graphics: GraphicsContext) -> None:
        self._graphics = graphics
        # The live atlas, kept alive after the batch build so late actors can
        # pack into it. None when no environmental sprites justified an atlas
        # (see build_for_world) or the backend has no atlas support.
        self._atlas: SpriteAtlas | None = None
        # Seed and world captured at build time so late spawns generate with the
        # same seed and a growth rebuild can re-census the world.
        self._map_seed: MapDecorationSeed = 0
        self._gw: GameWorld | None = None

    # ------------------------------------------------------------------
    # Per-actor logic shared by the batch build and late-spawn path
    # ------------------------------------------------------------------

    def _classify(self, actor: Actor) -> _ActorSpriteKind | None:
        """Return the sprite family for an actor, or None if it has no pose set.

        Quadrupeds opt in via their NPCType's critter_preset (dogs today);
        humanoids are matched on glyph. Everything else (trees, boulders,
        items, the player's non-humanoid cases) has no 12-frame pose set here.
        """
        from brileta.sprites.characters import HUMANOID_GLYPHS

        if isinstance(actor, NPC) and actor.critter_preset is not None:
            return "quadruped"
        if isinstance(actor, Character) and actor.ch in HUMANOID_GLYPHS:
            return "humanoid"
        return None

    def _generate_pose_set(
        self, actor: Actor, kind: _ActorSpriteKind
    ) -> list[np.ndarray]:
        """Generate one actor's pose-set frames, seeded from its id + map seed."""
        from brileta.sprites.characters import (
            character_sprite_seed,
            generate_character_pose_set,
        )
        from brileta.sprites.quadrupeds import (
            generate_quadruped_pose_set,
            quadruped_sprite_seed,
        )

        if kind == "humanoid":
            return generate_character_pose_set(
                character_sprite_seed(actor.actor_id, self._map_seed),
                presentation_profile=getattr(actor, "character_presentation", None),
            )
        assert isinstance(actor, NPC) and actor.critter_preset is not None
        return generate_quadruped_pose_set(
            quadruped_sprite_seed(actor.actor_id, self._map_seed),
            actor.critter_preset,
        )

    def _assign_pose_uvs(
        self,
        actor: Actor,
        pose_sprites: list[np.ndarray],
        uvs: list[SpriteUV | None],
    ) -> None:
        """Assign packed UVs and derived sprite state onto one actor.

        Shared by the batch build and the late-spawn path. If any pose failed to
        pack (a None UV), the actor is left on glyph fallback.
        """
        from brileta.sprites.common import (
            sprite_content_bbox,
            sprite_visual_scale_for_shadow_height,
        )

        resolved: list[SpriteUV] = []
        for uv in uvs:
            if uv is None:
                return
            resolved.append(uv)

        actor.character_sprite_uvs = tuple(resolved)
        actor.sprite_uv = resolved[0]
        actor.visual_scale = sprite_visual_scale_for_shadow_height(
            pose_sprites[0], actor.shadow_height
        )
        actor.sprite_content_bbox = sprite_content_bbox(pose_sprites[0])

    # ------------------------------------------------------------------
    # Late-spawn path: one actor into the live atlas
    # ------------------------------------------------------------------

    def ensure_actor_sprites(self, actor: Actor) -> None:
        """Ensure a single actor has atlas sprites, generating them if needed.

        Idempotent: an actor that already has UVs is a no-op. If no live atlas
        exists (no environmental sprites, or a backend without atlas support),
        the actor keeps its glyph. If the atlas is full, falls back to a full
        rebuild that re-packs everyone including this actor.
        """
        if self._atlas is None:
            return  # No atlas - actor renders as a glyph, as before.

        kind = self._classify(actor)
        if kind is None:
            return  # Trees, boulders, items: no pose set on this path.

        if actor.character_sprite_uvs is not None:
            return  # Already has sprites - idempotent no-op, no re-pack.

        pose_sprites = self._generate_pose_set(actor, kind)
        uvs = self._atlas.pack_incremental(pose_sprites)
        if any(uv is None for uv in uvs):
            # Atlas full: rebuild the whole atlas with headroom re-applied. This
            # re-censuses the world (now including this actor), so it ends with
            # every actor holding valid UVs. Rare - log it.
            logger.info("Sprite atlas full on late spawn; rebuilding with headroom.")
            self._rebuild()
            return

        self._assign_pose_uvs(actor, pose_sprites, uvs)

    def _rebuild(self) -> None:
        """Grow the atlas by rebuilding the batch pass for the current world."""
        assert self._gw is not None  # only reached after a build
        self.build_for_world(self._gw)

    # ------------------------------------------------------------------
    # Batch build: the whole world at world-gen time
    # ------------------------------------------------------------------

    def build_for_world(self, gw: GameWorld) -> None:
        """Generate and upload sprites for every actor in the world at once.

        Censuses trees, boulders, humanoid characters, and quadruped critters;
        generates their sprites in parallel; sizes a power-of-two atlas that
        fits them all plus headroom for late spawns; packs and uploads in a
        single flush; and keeps the atlas alive for late-spawn packing.

        Preserves the original behavior that no trees/boulders means no atlas at
        all (characters piggyback on the environmental atlas), in which case
        everyone renders as glyphs.
        """

        from brileta.backends.wgpu.sprite_atlas import PADDING, compute_atlas_size
        from brileta.sprites.boulders import (
            archetype_for_position as boulder_archetype_for_position,
        )
        from brileta.sprites.boulders import (
            generate_boulder_sprite_for_position,
        )
        from brileta.sprites.boulders import (
            shadow_height_for_archetype as boulder_shadow_height,
        )
        from brileta.sprites.boulders import (
            visual_scale_with_height_jitter as boulder_visual_scale_with_height_jitter,
        )
        from brileta.sprites.characters import CHARACTER_POSE_COUNT
        from brileta.sprites.common import sprite_content_bbox
        from brileta.sprites.quadrupeds import QUADRUPED_POSE_COUNT
        from brileta.sprites.trees import (
            generate_tree_sprite_for_position,
        )
        from brileta.sprites.trees import (
            visual_scale_with_height_jitter as tree_visual_scale_with_height_jitter,
        )
        from brileta.util.parallel import parallel_map

        self._gw = gw
        self._atlas = None

        trees: list[Tree] = []
        boulders: list[Boulder] = []
        characters: list[Character] = []
        critters: list[NPC] = []
        for a in gw.actors:
            if isinstance(a, Tree):
                trees.append(a)
            elif isinstance(a, Boulder):
                boulders.append(a)
            elif (kind := self._classify(a)) == "quadruped":
                assert isinstance(a, NPC)  # _classify's quadruped check implies this
                critters.append(a)
            elif kind == "humanoid":
                assert isinstance(
                    a, Character
                )  # _classify's humanoid check implies this
                characters.append(a)
        # Need at least one environmental sprite to justify atlas creation.
        # Character sprites piggyback on the atlas - when no environmental
        # sprites exist, characters fall back to their text glyphs.
        if not trees and not boulders:
            return

        map_seed: MapDecorationSeed = int(gw.game_map.decoration_seed)
        self._map_seed = map_seed

        with record_time_live_variable("time.sprites.total_ms"):
            # Phase 1: Pre-generate all sprites into CPU memory so we can
            # measure the total area before creating the GPU texture.
            with record_time_live_variable("time.sprites.generate_ms"):
                tree_sprites = parallel_map(
                    generate_tree_sprite_for_position,
                    [t.x for t in trees],
                    [t.y for t in trees],
                    [map_seed] * len(trees),
                    [t.tree_type for t in trees],
                )

                boulder_archetypes = [
                    boulder_archetype_for_position(b.x, b.y, map_seed) for b in boulders
                ]
                for boulder, archetype in zip(
                    boulders, boulder_archetypes, strict=True
                ):
                    boulder.shadow_height = boulder_shadow_height(archetype)

                boulder_sprites = parallel_map(
                    generate_boulder_sprite_for_position,
                    [b.x for b in boulders],
                    [b.y for b in boulders],
                    [map_seed] * len(boulders),
                    boulder_archetypes,
                )

                # Character and critter pose sets go through the same per-actor
                # generate helper the late-spawn path uses, so both routes
                # produce identical pixels for the same actor.
                char_pose_sprites = [
                    self._generate_pose_set(c, "humanoid") for c in characters
                ]
                critter_pose_sprites = [
                    self._generate_pose_set(c, "quadruped") for c in critters
                ]

            # Phase 2: Compute the atlas size (with late-spawn headroom) and
            # create it.
            flat_char_sprites = [
                sprite for pose_set in char_pose_sprites for sprite in pose_set
            ]
            flat_critter_sprites = [
                sprite for pose_set in critter_pose_sprites for sprite in pose_set
            ]
            all_sprites = (
                tree_sprites
                + boulder_sprites
                + flat_char_sprites
                + flat_critter_sprites
            )
            extra_area = _headroom_area(
                flat_char_sprites + flat_critter_sprites, PADDING
            )
            gpu_max = self._graphics.gpu_max_texture_dimension_2d
            atlas_side = compute_atlas_size(all_sprites, gpu_max, extra_area)

            atlas = self._graphics.create_sprite_atlas(atlas_side, atlas_side)
            if atlas is None:
                logger.debug(
                    "Graphics backend has no sprite atlas support; "
                    "actors will use fallback glyph rendering."
                )
                return

            # Phase 3: Bulk-pack all sprites via shelf packing, then flush
            # to the GPU in a single write_texture call.
            with record_time_live_variable("time.sprites.atlas_pack_ms"):
                uvs = atlas.pack_all(all_sprites)

                # Assign UVs and visual scales back to the actors.
                # UV list order: [trees..., boulders..., characters...].
                n_trees = len(trees)
                for i, tree in enumerate(trees):
                    uv = uvs[i]
                    if uv is not None:
                        tree.sprite_uv = uv
                        tree.visual_scale = tree_visual_scale_with_height_jitter(
                            tree_sprites[i],
                            tree.shadow_height,
                            tree.x,
                            tree.y,
                            map_seed,
                        )
                        tree.sprite_content_bbox = sprite_content_bbox(tree_sprites[i])

                for j, boulder in enumerate(boulders):
                    uv = uvs[n_trees + j]
                    if uv is not None:
                        boulder.sprite_uv = uv
                        boulder.visual_scale = boulder_visual_scale_with_height_jitter(
                            boulder_sprites[j],
                            boulder.shadow_height,
                            boulder.x,
                            boulder.y,
                            map_seed,
                        )
                        boulder.sprite_content_bbox = sprite_content_bbox(
                            boulder_sprites[j]
                        )

                # Characters and critters both use the character_sprite_uvs
                # pipeline (same 12-frame layout), so the same per-actor assign
                # handles both - only the block offset and frame count differ.
                n_boulders = len(boulders)
                char_base = n_trees + n_boulders
                self._assign_block(
                    characters, char_pose_sprites, uvs, char_base, CHARACTER_POSE_COUNT
                )
                critter_base = char_base + len(characters) * CHARACTER_POSE_COUNT
                self._assign_block(
                    critters,
                    critter_pose_sprites,
                    uvs,
                    critter_base,
                    QUADRUPED_POSE_COUNT,
                )

                atlas.flush()

        # Keep the atlas alive for late-spawn packing (was dropped before).
        self._atlas = atlas
        if atlas.texture is not None:
            self._graphics.set_sprite_atlas_texture(atlas.texture)

        logger.info(
            "Sprite atlas: %dx%d (%d tree + %d boulder + %d character"
            " + %d critter pose sprites"
            " across %d characters and %d critters,"
            " %d allocations)",
            atlas_side,
            atlas_side,
            len(trees),
            len(boulders),
            len(characters) * CHARACTER_POSE_COUNT,
            len(critters) * QUADRUPED_POSE_COUNT,
            len(characters),
            len(critters),
            atlas.allocated_count,
        )

    def _assign_block(
        self,
        actors: list[Character] | list[NPC],
        pose_sprites: list[list[np.ndarray]],
        uvs: list[SpriteUV | None],
        base_offset: int,
        frame_count: int,
    ) -> None:
        """Slice each actor's block of UVs out of the bulk pack and assign it."""
        for k, actor in enumerate(actors):
            start = base_offset + k * frame_count
            self._assign_pose_uvs(
                actor, pose_sprites[k], uvs[start : start + frame_count]
            )


def _headroom_area(sample_sprites: list[np.ndarray], padding: int) -> int:
    """Padded px^2 to reserve for late spawns, based on a representative frame.

    Uses the largest character/critter frame so the estimate is generous, times
    a full pose set's frame count times the number of extra pose sets. Zero when
    there are no character/critter frames to size against (nothing spawns late).
    """
    from brileta.sprites.characters import CHARACTER_POSE_COUNT

    if not sample_sprites:
        return 0
    frame_area = max(
        (px.shape[0] + padding) * (px.shape[1] + padding) for px in sample_sprites
    )
    return frame_area * CHARACTER_POSE_COUNT * _HEADROOM_POSE_SETS

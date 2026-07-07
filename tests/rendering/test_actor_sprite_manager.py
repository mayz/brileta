"""Tests for ActorSpriteManager: one sprite path for batch and late actors.

Uses a fake graphics context that hands out a real SpriteAtlas backed by a
mock GPU resource manager, so packing/upload logic runs without a WGPU device.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from brileta.backends.wgpu.sprite_atlas import SpriteAtlas
from brileta.game.actors.npc_types import DOG_TYPE, RESIDENT_TYPE
from brileta.game.actors.trees import create_deciduous_tree
from brileta.sprites.characters import MASC_PRESENTATION
from brileta.sprites.quadrupeds import (
    generate_quadruped_pose_set,
    quadruped_sprite_seed,
)
from brileta.view.render.actor_sprite_manager import ActorSpriteManager


class _FakeGraphics:
    """Minimal GraphicsContext stand-in returning a mock-backed SpriteAtlas."""

    def __init__(self, gpu_max: int = 8192) -> None:
        self._rm = MagicMock()
        self._rm.device.create_texture.return_value = MagicMock(name="gpu_texture")
        self._gpu_max = gpu_max
        self.bound_texture: Any = None
        self.atlas: SpriteAtlas | None = None

    @property
    def gpu_max_texture_dimension_2d(self) -> int:
        return self._gpu_max

    def create_sprite_atlas(self, width: int, height: int) -> SpriteAtlas:
        self.atlas = SpriteAtlas(self._rm, width, height)
        return self.atlas

    def set_sprite_atlas_texture(self, texture: Any) -> None:
        self.bound_texture = texture


class _FakeMap:
    decoration_seed = 4242


class _FakeGameWorld:
    """Just the surface build_for_world touches: actors + map decoration seed."""

    def __init__(self, actors: list[Any]) -> None:
        self.actors = actors
        self.game_map = _FakeMap()


def _manager_with_atlas(
    extra_actors: list[Any] | None = None,
) -> tuple[ActorSpriteManager, _FakeGameWorld, _FakeGraphics]:
    """Build a manager with a live atlas (one tree justifies the atlas)."""
    graphics = _FakeGraphics()
    manager = ActorSpriteManager(graphics)  # ty: ignore[invalid-argument-type]
    tree = create_deciduous_tree(1, 1)
    actors: list[Any] = [tree, *(extra_actors or [])]
    gw = _FakeGameWorld(actors)
    manager.build_for_world(gw)  # ty: ignore[invalid-argument-type]
    assert manager._atlas is not None, "a tree should justify an atlas"
    return manager, gw, graphics


def _assert_has_sprites(actor: Any) -> None:
    assert actor.character_sprite_uvs is not None
    assert len(actor.character_sprite_uvs) == 12
    assert actor.sprite_uv is not None
    assert actor.visual_scale > 0
    assert actor.sprite_content_bbox is not None


def test_late_humanoid_gets_uvs() -> None:
    manager, _gw, _g = _manager_with_atlas()
    resident = RESIDENT_TYPE.create(5, 5, "Res")
    assert resident.character_sprite_uvs is None
    manager.ensure_actor_sprites(resident)
    _assert_has_sprites(resident)


def test_late_humanoid_passes_presentation_to_sprite_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _gw, _g = _manager_with_atlas()
    resident = RESIDENT_TYPE.create(5, 5, "Res")
    resident.character_presentation = MASC_PRESENTATION
    captured_profile: list[object] = []

    def fake_generate_character_pose_set(
        _seed: int,
        _size: int = 20,
        *,
        presentation_profile: object | None = None,
    ) -> list[np.ndarray]:
        captured_profile.append(presentation_profile)
        sprite = np.zeros((2, 2, 4), dtype=np.uint8)
        sprite[:, :] = np.array([255, 255, 255, 255], dtype=np.uint8)
        return [sprite.copy() for _ in range(12)]

    monkeypatch.setattr(
        "brileta.sprites.characters.generate_character_pose_set",
        fake_generate_character_pose_set,
    )

    manager.ensure_actor_sprites(resident)

    _assert_has_sprites(resident)
    assert captured_profile == [MASC_PRESENTATION]


def test_late_quadruped_gets_uvs() -> None:
    manager, _gw, _g = _manager_with_atlas()
    dog = DOG_TYPE.create(6, 6, "Rex")
    assert dog.character_sprite_uvs is None
    manager.ensure_actor_sprites(dog)
    _assert_has_sprites(dog)


def test_ensure_is_idempotent() -> None:
    manager, _gw, _g = _manager_with_atlas()
    dog = DOG_TYPE.create(6, 6, "Rex")
    manager.ensure_actor_sprites(dog)
    first = dog.character_sprite_uvs
    assert manager._atlas is not None
    count_after_first = manager._atlas.allocated_count

    manager.ensure_actor_sprites(dog)
    # Same UV tuple object, and nothing new packed.
    assert dog.character_sprite_uvs is first
    assert manager._atlas.allocated_count == count_after_first


def test_censused_actor_not_double_packed() -> None:
    """An actor present at build time gets batch UVs; a later ensure is a
    no-op (it does not re-pack), mirroring the world-gen no-double-pack guard."""
    resident = RESIDENT_TYPE.create(5, 5, "Res")
    manager, _gw, _g = _manager_with_atlas([resident])
    _assert_has_sprites(resident)  # covered by the batch build
    assert manager._atlas is not None
    count = manager._atlas.allocated_count

    manager.ensure_actor_sprites(resident)
    assert manager._atlas.allocated_count == count


def test_growth_fallback_rebuilds_and_all_actors_keep_uvs() -> None:
    manager, gw, graphics = _manager_with_atlas()
    old_atlas = manager._atlas
    assert old_atlas is not None

    # Exhaust the live atlas so the next pack_incremental fails and forces a
    # rebuild. The late dog must already be in the world so the rebuild census
    # covers it (add_actor appends before firing the hook in real code).
    old_atlas._shelf_y = old_atlas.height
    dog = DOG_TYPE.create(6, 6, "Rex")
    gw.actors.append(dog)

    manager.ensure_actor_sprites(dog)

    # A fresh atlas replaced the full one, and every actor has valid sprites.
    assert manager._atlas is not None
    assert manager._atlas is not old_atlas
    assert graphics.bound_texture is manager._atlas.texture
    _assert_has_sprites(dog)
    tree = gw.actors[0]
    assert tree.sprite_uv is not None


def test_no_atlas_means_no_sprites() -> None:
    """No trees/boulders -> no atlas -> late actors keep their glyph."""
    graphics = _FakeGraphics()
    manager = ActorSpriteManager(graphics)  # ty: ignore[invalid-argument-type]
    manager.build_for_world(_FakeGameWorld([]))  # ty: ignore[invalid-argument-type]
    assert manager._atlas is None

    dog = DOG_TYPE.create(6, 6, "Rex")
    manager.ensure_actor_sprites(dog)
    assert dog.character_sprite_uvs is None


def test_spawned_and_worldgen_dog_share_pixels() -> None:
    """Determinism: two dogs with the same actor id and map seed generate the
    identical pose set, whether censused at world gen or spawned later."""
    manager, _gw, _g = _manager_with_atlas()
    seed = manager._map_seed

    worldgen_dog = DOG_TYPE.create(2, 2, "Rex")
    worldgen_dog.actor_id = 777
    spawned_dog = DOG_TYPE.create(9, 9, "Rex")
    spawned_dog.actor_id = 777

    a = manager._generate_pose_set(worldgen_dog, "quadruped")
    b = manager._generate_pose_set(spawned_dog, "quadruped")
    for fa, fb in zip(a, b, strict=True):
        assert np.array_equal(fa, fb)

    # And it matches a direct generation from the seed functions.
    assert worldgen_dog.critter_preset is not None
    direct = generate_quadruped_pose_set(
        quadruped_sprite_seed(777, seed), worldgen_dog.critter_preset
    )
    for fa, fd in zip(a, direct, strict=True):
        assert np.array_equal(fa, fd)

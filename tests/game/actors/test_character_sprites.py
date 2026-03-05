"""Tests for directional character sprite generation."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from brileta.game.actors.character_sprites import (
    CHARACTER_DIRECTIONAL_POSE_COUNT,
    POSE_BACK_STAND,
    POSE_LEFT_STAND,
    POSE_RIGHT_STAND,
    POSE_STAND,
    POSES,
    CharacterAppearance,
    draw_character_pose,
    generate_character_pose_set,
    generate_character_sprite,
)


def _hair_like_count(
    patch: np.ndarray,
    hair_mid: np.ndarray,
    skin_mid: np.ndarray,
) -> int:
    """Count pixels closer to hair color than skin color in an RGBA patch."""
    alpha_mask = patch[:, :, 3] > 0
    if not bool(alpha_mask.any()):
        return 0
    rgb = patch[:, :, :3].astype(np.float64)
    hair_dist = np.linalg.norm(rgb - hair_mid, axis=2)
    skin_dist = np.linalg.norm(rgb - skin_mid, axis=2)
    return int(np.count_nonzero(alpha_mask & (hair_dist + 4.0 < skin_dist)))


def _pose_set_digest(pose_set: list[np.ndarray]) -> str:
    """Return a stable digest for a rendered pose set."""
    hasher = hashlib.sha256()
    for sprite in pose_set:
        hasher.update(sprite.tobytes())
    return hasher.hexdigest()


def test_generate_character_pose_set_returns_directional_rgba_sprites() -> None:
    """Pose generation should produce the 4 non-empty directional RGBA images."""
    pose_set = generate_character_pose_set(42)

    assert len(pose_set) == CHARACTER_DIRECTIONAL_POSE_COUNT
    for sprite in pose_set:
        assert sprite.ndim == 3
        assert sprite.shape[2] == 4
        assert sprite.shape[0] > 0
        assert sprite.shape[1] > 0
        assert int(np.count_nonzero(sprite[:, :, 3])) > 0


def test_generate_character_pose_set_is_deterministic() -> None:
    """Repeated generation with the same seed should be byte-identical."""
    pose_set_a = generate_character_pose_set(12345)
    pose_set_b = generate_character_pose_set(12345)

    assert len(pose_set_a) == len(pose_set_b)
    for sprite_a, sprite_b in zip(pose_set_a, pose_set_b, strict=True):
        assert np.array_equal(sprite_a, sprite_b)


def test_different_seeds_produce_different_pose_sets() -> None:
    """Different seeds should usually generate different pixel output."""
    pose_set_a = generate_character_pose_set(111)
    pose_set_b = generate_character_pose_set(222)

    assert _pose_set_digest(pose_set_a) != _pose_set_digest(pose_set_b)


def test_character_appearance_from_seed_is_consistent() -> None:
    """Appearance roll should be deterministic for a given seed and size."""
    appearance_a = CharacterAppearance.from_seed(9001, 20)
    appearance_b = CharacterAppearance.from_seed(9001, 20)

    assert appearance_a == appearance_b


def test_all_build_types_render_valid_sprites_in_all_poses() -> None:
    """Every build template should render all directional poses without empties."""
    for build_idx in range(5):
        appearance = CharacterAppearance.from_seed(
            seed=77,
            size=20,
            forced_build_idx=build_idx,
        )
        for pose in POSES:
            sprite = draw_character_pose(appearance, pose)
            assert sprite.ndim == 3
            assert sprite.shape[2] == 4
            assert int(np.count_nonzero(sprite[:, :, 3])) > 0


def test_front_stand_matches_generate_character_sprite_wrapper() -> None:
    """Backward-compatible wrapper should return pose-set front-standing sprite."""
    pose_set = generate_character_pose_set(31415)
    stand_sprite = generate_character_sprite(31415)

    assert np.array_equal(pose_set[0], stand_sprite)


def test_palette_guardrails_keep_skin_separate_from_hair_and_clothing() -> None:
    """Readability guardrails should avoid skin/cloth and skin/hair merging."""

    def _luma(rgb: tuple[int, int, int]) -> float:
        r, g, b = rgb
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    for seed in range(100):
        appearance = CharacterAppearance.from_seed(seed, 20)
        skin_luma = _luma(appearance.skin_pal[1])
        hair_luma = _luma(appearance.hair_pal[1])
        cloth_luma = _luma(appearance.cloth_pal[1])

        assert abs(hair_luma - skin_luma) >= 18.0
        assert abs(cloth_luma - skin_luma) >= 20.0


def test_front_medium_and_long_hair_keep_forehead_region_hair_colored() -> None:
    """Front-facing medium/long hair should not show a skin-colored bald patch."""

    for hair_style_idx in (2, 3):  # medium, long
        for seed in range(20):
            appearance = replace(
                CharacterAppearance.from_seed(seed, 20),
                hair_style_idx=hair_style_idx,
            )
            sprite = draw_character_pose(appearance, POSE_STAND)  # front stand
            params = appearance.body_params

            center_x = round(params.canvas_size / 2.0)
            head_center_y = params.head_radius + params.head_top_pad
            forehead_y = round(head_center_y - params.head_radius * 0.24)

            y1 = max(0, forehead_y - 1)
            y2 = min(params.canvas_size, forehead_y + 2)
            x1 = max(0, center_x - 1)
            x2 = min(params.canvas_size, center_x + 2)
            patch = sprite[y1:y2, x1:x2, :]
            alpha_mask = patch[:, :, 3] > 0
            assert bool(alpha_mask.any())

            patch_rgb = patch[:, :, :3][alpha_mask].astype(np.float64)
            mean_rgb = patch_rgb.mean(axis=0)

            skin_mid = np.array(appearance.skin_pal[1], dtype=np.float64)
            hair_mid = np.array(appearance.hair_pal[1], dtype=np.float64)

            hair_distance = float(np.linalg.norm(mean_rgb - hair_mid))
            skin_distance = float(np.linalg.norm(mean_rgb - skin_mid))

            assert hair_distance < skin_distance


def test_side_medium_hair_reads_as_profile_and_mirror_swaps_back_side() -> None:
    """Side medium-hair profile should carry more hair volume on the back side."""
    appearance = replace(
        CharacterAppearance.from_seed(1337, 20),
        hair_style_idx=2,  # medium
    )
    sprite = draw_character_pose(
        appearance, POSE_LEFT_STAND
    )  # side stand (authored side)
    mirrored = np.ascontiguousarray(sprite[:, ::-1, :])
    params = appearance.body_params

    center_x = round(params.canvas_size / 2.0)
    x_front_1 = max(0, round(center_x - params.head_radius * 1.1))
    x_front_2 = max(x_front_1 + 1, round(center_x - params.head_radius * 0.15))
    x_back_1 = min(params.canvas_size - 1, round(center_x + params.head_radius * 0.15))
    x_back_2 = min(params.canvas_size, round(center_x + params.head_radius * 1.2))
    y1 = max(0, round(params.head_top_pad))
    y2 = min(params.canvas_size, round(params.head_top_pad + params.head_radius * 1.9))

    hair_mid = np.array(appearance.hair_pal[1], dtype=np.float64)
    skin_mid = np.array(appearance.skin_pal[1], dtype=np.float64)

    front_count = _hair_like_count(
        sprite[y1:y2, x_front_1:x_front_2, :], hair_mid, skin_mid
    )
    back_count = _hair_like_count(
        sprite[y1:y2, x_back_1:x_back_2, :], hair_mid, skin_mid
    )
    mirrored_front_count = _hair_like_count(
        mirrored[y1:y2, x_front_1:x_front_2, :], hair_mid, skin_mid
    )
    mirrored_back_count = _hair_like_count(
        mirrored[y1:y2, x_back_1:x_back_2, :], hair_mid, skin_mid
    )

    assert back_count > front_count
    assert mirrored_front_count > mirrored_back_count

    # Guardrail for the common "bald spot" artifact in side tall-hair profiles:
    # the back-head patch should be hair-like, not skin-like.
    patch_x1 = min(params.canvas_size - 2, round(center_x + params.head_radius * 0.26))
    patch_x2 = min(params.canvas_size, patch_x1 + 2)
    patch_y1 = max(0, round(params.head_top_pad + params.head_radius * 0.78))
    patch_y2 = min(params.canvas_size, patch_y1 + 2)
    patch = sprite[patch_y1:patch_y2, patch_x1:patch_x2, :]
    alpha_mask = patch[:, :, 3] > 0
    assert bool(alpha_mask.any())
    patch_rgb = patch[:, :, :3][alpha_mask].astype(np.float64)
    mean_rgb = patch_rgb.mean(axis=0)
    hair_distance = float(np.linalg.norm(mean_rgb - hair_mid))
    skin_distance = float(np.linalg.norm(mean_rgb - skin_mid))
    assert hair_distance < skin_distance


def test_side_short_hair_reads_as_profile_and_mirror_swaps_back_side() -> None:
    """Side short-hair profile should keep more hair volume on back side."""
    appearance = replace(
        CharacterAppearance.from_seed(1338, 20),
        hair_style_idx=1,  # short
    )
    sprite = draw_character_pose(
        appearance, POSE_LEFT_STAND
    )  # side stand (authored side)
    mirrored = np.ascontiguousarray(sprite[:, ::-1, :])
    params = appearance.body_params

    center_x = round(params.canvas_size / 2.0)
    x_front_1 = max(0, round(center_x - params.head_radius * 1.1))
    x_front_2 = max(x_front_1 + 1, round(center_x - params.head_radius * 0.15))
    x_back_1 = min(params.canvas_size - 1, round(center_x + params.head_radius * 0.15))
    x_back_2 = min(params.canvas_size, round(center_x + params.head_radius * 1.2))
    y1 = max(0, round(params.head_top_pad))
    y2 = min(params.canvas_size, round(params.head_top_pad + params.head_radius * 1.9))

    hair_mid = np.array(appearance.hair_pal[1], dtype=np.float64)
    skin_mid = np.array(appearance.skin_pal[1], dtype=np.float64)

    front_count = _hair_like_count(
        sprite[y1:y2, x_front_1:x_front_2, :], hair_mid, skin_mid
    )
    back_count = _hair_like_count(
        sprite[y1:y2, x_back_1:x_back_2, :], hair_mid, skin_mid
    )
    mirrored_front_count = _hair_like_count(
        mirrored[y1:y2, x_front_1:x_front_2, :], hair_mid, skin_mid
    )
    mirrored_back_count = _hair_like_count(
        mirrored[y1:y2, x_back_1:x_back_2, :], hair_mid, skin_mid
    )

    assert back_count > front_count
    assert mirrored_front_count > mirrored_back_count


def test_side_tall_hair_reads_as_profile_and_mirror_swaps_back_side() -> None:
    """Side tall-hair profile should keep back-side hair mass and mirror correctly."""
    appearance = replace(
        CharacterAppearance.from_seed(1341, 20),
        hair_style_idx=4,  # tall
    )
    sprite = draw_character_pose(
        appearance, POSE_LEFT_STAND
    )  # side stand (authored side)
    mirrored = np.ascontiguousarray(sprite[:, ::-1, :])
    params = appearance.body_params

    center_x = round(params.canvas_size / 2.0)
    x_front_1 = max(0, round(center_x - params.head_radius * 1.1))
    x_front_2 = max(x_front_1 + 1, round(center_x - params.head_radius * 0.15))
    x_back_1 = min(params.canvas_size - 1, round(center_x + params.head_radius * 0.15))
    x_back_2 = min(params.canvas_size, round(center_x + params.head_radius * 1.2))
    y1 = max(0, round(params.head_top_pad))
    y2 = min(params.canvas_size, round(params.head_top_pad + params.head_radius * 1.9))

    hair_mid = np.array(appearance.hair_pal[1], dtype=np.float64)
    skin_mid = np.array(appearance.skin_pal[1], dtype=np.float64)

    front_count = _hair_like_count(
        sprite[y1:y2, x_front_1:x_front_2, :], hair_mid, skin_mid
    )
    back_count = _hair_like_count(
        sprite[y1:y2, x_back_1:x_back_2, :], hair_mid, skin_mid
    )
    mirrored_front_count = _hair_like_count(
        mirrored[y1:y2, x_front_1:x_front_2, :], hair_mid, skin_mid
    )
    mirrored_back_count = _hair_like_count(
        mirrored[y1:y2, x_back_1:x_back_2, :], hair_mid, skin_mid
    )

    assert back_count > front_count
    assert mirrored_front_count > mirrored_back_count

    # Back-side head region should not leak skin-toned pixels ("bald spot").
    leak_x1 = min(params.canvas_size - 1, round(center_x + params.head_radius * 0.24))
    leak_x2 = min(params.canvas_size, round(center_x + params.head_radius * 1.05))
    leak_y1 = max(0, round(params.head_top_pad + params.head_radius * 0.05))
    leak_y2 = min(
        params.canvas_size, round(params.head_top_pad + params.head_radius * 1.2)
    )
    leak_patch = sprite[leak_y1:leak_y2, leak_x1:leak_x2, :]
    leak_alpha = leak_patch[:, :, 3] > 0
    assert bool(leak_alpha.any())
    leak_rgb = leak_patch[:, :, :3].astype(np.float64)
    leak_hair_dist = np.linalg.norm(leak_rgb - hair_mid, axis=2)
    leak_skin_dist = np.linalg.norm(leak_rgb - skin_mid, axis=2)
    leaked_skin_like = int(
        np.count_nonzero(leak_alpha & (leak_skin_dist + 2.0 < leak_hair_dist))
    )
    assert leaked_skin_like == 0


def test_side_armor_profile_is_narrower_than_front_profile() -> None:
    """Armor side profile should not keep full front-facing shoulder width."""
    appearance = CharacterAppearance.from_seed(1341, 20)
    front = draw_character_pose(appearance, POSE_STAND)  # front stand
    side = draw_character_pose(appearance, POSE_LEFT_STAND)  # side stand
    params = appearance.body_params

    torso_cy = (
        params.head_radius
        + params.head_top_pad
        + params.head_radius
        + params.neck_gap
        + params.torso_ry * params.torso_cy_factor
    )
    y = min(params.canvas_size - 1, max(0, round(torso_cy)))

    front_x = np.where(front[y, :, 3] > 0)[0]
    side_x = np.where(side[y, :, 3] > 0)[0]
    assert front_x.size > 0
    assert side_x.size > 0

    front_w = int(front_x.max() - front_x.min() + 1)
    side_w = int(side_x.max() - side_x.min() + 1)
    assert side_w <= front_w - 1


def test_back_belly_view_blends_torso_belly_seam_toward_cloth_mid() -> None:
    """Back-facing belly build should avoid a hard chest/belly split line."""
    appearance = CharacterAppearance.from_seed(1338, 20)
    sprite = draw_character_pose(appearance, POSE_BACK_STAND)  # back stand
    params = appearance.body_params

    center_x = round(params.canvas_size / 2.0)
    seam_y = round(
        params.head_radius
        + params.head_top_pad
        + params.head_radius
        + params.neck_gap
        + params.torso_ry * params.torso_cy_factor
        + params.torso_ry * 0.95
    )

    x1 = max(0, center_x - 1)
    x2 = min(params.canvas_size, center_x + 2)
    y1 = max(0, seam_y - 1)
    y2 = min(params.canvas_size, seam_y + 2)
    patch = sprite[y1:y2, x1:x2, :]
    alpha_mask = patch[:, :, 3] > 0
    assert bool(alpha_mask.any())

    patch_rgb = patch[:, :, :3][alpha_mask].astype(np.float64)
    mean_rgb = patch_rgb.mean(axis=0)
    cloth_shadow = np.array(appearance.cloth_pal[0], dtype=np.float64)
    cloth_mid = np.array(appearance.cloth_pal[1], dtype=np.float64)
    mid_dist = float(np.linalg.norm(mean_rgb - cloth_mid))
    shadow_dist = float(np.linalg.norm(mean_rgb - cloth_shadow))

    assert mid_dist < shadow_dist


def test_side_stand_has_front_toe_protrusion_for_direction_cue() -> None:
    """Side stand should include a small toe extension toward facing direction."""
    appearance = CharacterAppearance.from_seed(1338, 20)
    sprite = draw_character_pose(appearance, POSE_LEFT_STAND)  # side stand
    params = appearance.body_params
    center_x = params.canvas_size / 2.0
    toe_x = round(center_x - max(0.75, params.leg_w1 * 0.75))
    toe_y = round(min(float(params.canvas_size - 1), params.canvas_size - 1 + 0.2))

    x1 = max(0, toe_x - 1)
    x2 = min(params.canvas_size, toe_x + 2)
    y1 = max(0, toe_y - 1)
    y2 = min(params.canvas_size, toe_y + 1)
    patch = sprite[y1:y2, x1:x2, :]
    assert int(np.count_nonzero(patch[:, :, 3])) > 0


def test_right_pose_is_a_mirrored_left_pose_for_same_appearance() -> None:
    """Right-facing stand should match a mirror of the left-facing stand."""
    appearance = CharacterAppearance.from_seed(73, 20)
    left_sprite = draw_character_pose(appearance, POSE_LEFT_STAND)
    right_sprite = draw_character_pose(appearance, POSE_RIGHT_STAND)

    assert np.array_equal(right_sprite, np.ascontiguousarray(left_sprite[:, ::-1, :]))


def test_thin_front_pose_has_no_large_torso_to_leg_vertical_gap() -> None:
    """Thin builds should not render detached feet below an empty vertical gap."""
    appearance = CharacterAppearance.from_seed(1339, 20)
    sprite = draw_character_pose(appearance, POSE_STAND)  # front stand
    alpha = sprite[:, :, 3]

    torso_bottom = (
        appearance.body_params.head_radius
        + appearance.body_params.head_top_pad
        + appearance.body_params.head_radius
        + appearance.body_params.neck_gap
        + appearance.body_params.torso_ry * appearance.body_params.torso_cy_factor
        + appearance.body_params.torso_ry
    )
    start_row = max(0, int(torso_bottom) - 1)
    occupied_rows = np.where(alpha.max(axis=1) > 0)[0]
    lower_rows = occupied_rows[occupied_rows >= start_row]
    assert lower_rows.size > 0
    assert int(np.max(np.diff(lower_rows))) <= 1


def test_thin_front_pose_has_upper_leg_mass_between_legs() -> None:
    """Thin front pose should keep a small connector mass above stick-like legs."""
    appearance = CharacterAppearance.from_seed(1339, 20)
    sprite = draw_character_pose(appearance, POSE_STAND)  # front stand
    alpha = sprite[:, :, 3]
    params = appearance.body_params

    center_x = round(params.canvas_size / 2.0)
    torso_bottom = (
        params.head_radius
        + params.head_top_pad
        + params.head_radius
        + params.neck_gap
        + params.torso_ry * params.torso_cy_factor
        + params.torso_ry
    )
    y = min(params.canvas_size - 1, int(torso_bottom + 1))
    x1 = max(0, center_x - 1)
    x2 = min(params.canvas_size, center_x + 2)
    y1 = max(0, y - 1)
    y2 = min(params.canvas_size, y + 1)
    patch = alpha[y1:y2, x1:x2]
    assert int(np.count_nonzero(patch)) > 0

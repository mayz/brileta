"""Character render pipeline and public sprite generation APIs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from brileta.sprites.primitives import darken_rim
from brileta.types import Facing

from .appearance import POSE_STAND, POSES, BodyParams, CharacterAppearance, Pose
from .body import (
    _layer_back_arm,
    _layer_belly,
    _layer_face_final,
    _layer_front_arm,
    _layer_head,
    _layer_legs,
    _layer_neck,
)
from .clothing import _layer_torso
from .hair import _layer_hair


@dataclass(frozen=True)
class CharacterDrawContext:
    """Shared geometry and appearance values for layer rendering."""

    canvas: np.ndarray
    appearance: CharacterAppearance
    pose: Pose
    params: BodyParams
    facing: Facing
    cx: float
    hr: float
    head_cy: float
    torso_cy: float
    belly_cy: float
    torso_bottom: float
    leg_bottom: int
    leg_top: int
    is_thin_legs: bool


CharacterLayerFn = Callable[[CharacterDrawContext], None]
CharacterLayerStep = tuple[str, CharacterLayerFn]


def _build_character_draw_context(
    canvas: np.ndarray,
    appearance: CharacterAppearance,
    pose: Pose,
) -> CharacterDrawContext:
    """Resolve geometric anchors once for the layer pipeline."""
    params = appearance.body_params
    facing = pose.facing
    cx = params.canvas_size / 2.0
    hr = params.head_radius
    head_cy = hr + params.head_top_pad + pose.body_dy
    torso_cy = head_cy + hr + params.neck_gap + params.torso_ry * params.torso_cy_factor
    belly_cy = torso_cy + params.belly_cy_offset
    torso_bottom = torso_cy + params.torso_ry
    if params.belly_ry > 0:
        torso_bottom = belly_cy + params.belly_ry

    leg_bottom = params.canvas_size - 1
    leg_top = leg_bottom - 2
    is_thin_legs = params.leg_w1 <= 1.05 and params.arm_thickness == 1
    if params.leg_from_torso:
        leg_top = int(torso_bottom - 0.3)
    else:
        # Anchor leg start to torso silhouette to avoid "floating feet" gaps on
        # thin/slender builds while preserving a minimum visible leg length.
        min_leg_len = 4 if is_thin_legs else 3
        anchored_top = int(torso_bottom + 0.4)
        leg_top = min(anchored_top, leg_bottom - min_leg_len)

    return CharacterDrawContext(
        canvas=canvas,
        appearance=appearance,
        pose=pose,
        params=params,
        facing=facing,
        cx=cx,
        hr=hr,
        head_cy=head_cy,
        torso_cy=torso_cy,
        belly_cy=belly_cy,
        torso_bottom=torso_bottom,
        leg_bottom=leg_bottom,
        leg_top=leg_top,
        is_thin_legs=is_thin_legs,
    )


_CHARACTER_LAYER_PIPELINE_SOUTH: tuple[CharacterLayerStep, ...] = (
    # SOUTH-facing keeps torso mass below both arm passes.
    ("legs", _layer_legs),
    ("belly", _layer_belly),
    ("torso", _layer_torso),
    ("back_arm", _layer_back_arm),
    ("front_arm", _layer_front_arm),
    ("neck", _layer_neck),
    ("head", _layer_head),
    ("hair", _layer_hair),
    ("face", _layer_face_final),
)

_CHARACTER_LAYER_PIPELINE_NORTH: tuple[CharacterLayerStep, ...] = (
    # NORTH-facing shares the same z-stack but has different per-layer internals.
    ("legs", _layer_legs),
    ("belly", _layer_belly),
    ("torso", _layer_torso),
    ("back_arm", _layer_back_arm),
    ("front_arm", _layer_front_arm),
    ("neck", _layer_neck),
    ("head", _layer_head),
    ("hair", _layer_hair),
    ("face", _layer_face_final),
)

_CHARACTER_LAYER_PIPELINE_SIDE: tuple[CharacterLayerStep, ...] = (
    # Side profile has explicit arm depth: back arm then front arm.
    ("legs", _layer_legs),
    ("belly", _layer_belly),
    ("torso", _layer_torso),
    ("back_arm", _layer_back_arm),
    ("front_arm", _layer_front_arm),
    ("neck", _layer_neck),
    ("head", _layer_head),
    ("hair", _layer_hair),
    ("face", _layer_face_final),
)

_CHARACTER_LAYER_PIPELINES_BY_FACING: dict[Facing, tuple[CharacterLayerStep, ...]] = {
    Facing.SOUTH: _CHARACTER_LAYER_PIPELINE_SOUTH,
    Facing.NORTH: _CHARACTER_LAYER_PIPELINE_NORTH,
    Facing.WEST: _CHARACTER_LAYER_PIPELINE_SIDE,
    Facing.EAST: _CHARACTER_LAYER_PIPELINE_SIDE,
}


def _draw_character(
    canvas: np.ndarray,
    appearance: CharacterAppearance,
    pose: Pose,
) -> None:
    """Draw one character via an explicit ordered layer pipeline."""
    context = _build_character_draw_context(canvas, appearance, pose)
    layer_pipeline = _CHARACTER_LAYER_PIPELINES_BY_FACING[context.facing]
    for _layer_name, layer_fn in layer_pipeline:
        layer_fn(context)


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------


def roll_character_appearance(seed: int, size: int = 20) -> CharacterAppearance:
    """Roll one deterministic appearance from a seed."""
    return CharacterAppearance.from_seed(seed, size)


def _render_pose(appearance: CharacterAppearance, pose: Pose) -> np.ndarray:
    """Render one pose from a rolled appearance."""
    canvas_size = appearance.body_params.canvas_size
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    _draw_character(canvas, appearance, pose)
    # Side profile drawing is authored left-facing. Materialize EAST-facing
    # sprites by mirroring here so runtime only needs UV selection.
    if pose.facing is Facing.EAST:
        canvas = np.ascontiguousarray(canvas[:, ::-1, :])
    darken_rim(canvas, (35, 35, 25))
    return canvas


def draw_character_pose(appearance: CharacterAppearance, pose: Pose) -> np.ndarray:
    """Render one pose from pre-rolled appearance data."""
    return _render_pose(appearance, pose)


def generate_character(
    seed: int,
    size: int = 20,
    *,
    pose: Pose = POSE_STAND,
    appearance: CharacterAppearance | None = None,
) -> np.ndarray:
    """Generate a single character sprite for one pose."""
    resolved_appearance = appearance or roll_character_appearance(seed, size)
    return _render_pose(resolved_appearance, pose)


def generate_character_pose_set(seed: int, size: int = 20) -> list[np.ndarray]:
    """Generate front/back/left/right standing poses for one seed."""
    appearance = roll_character_appearance(seed, size)
    return [_render_pose(appearance, pose) for pose in POSES]


def generate_character_sprite(seed: int, size: int = 20) -> np.ndarray:
    """Backward-compatible wrapper returning front-standing pose."""
    return generate_character_pose_set(seed, size)[0]


def _generate_caricature_for_build(seed: int, size: int, build_idx: int) -> np.ndarray:
    """Generate the standing pose for a specific body build template."""
    appearance = CharacterAppearance.from_seed(seed, size, forced_build_idx=build_idx)
    return _render_pose(appearance, POSE_STAND)


def generate_caricature_standard(seed: int, size: int = 20) -> np.ndarray:
    """Standard chibi body template."""
    return _generate_caricature_for_build(seed, size, 0)


def generate_caricature_belly(seed: int, size: int = 20) -> np.ndarray:
    """Belly-dominant body template."""
    return _generate_caricature_for_build(seed, size, 1)


def generate_caricature_broad(seed: int, size: int = 20) -> np.ndarray:
    """Broad/shoulder-dominant body template."""
    return _generate_caricature_for_build(seed, size, 2)


def generate_caricature_thin(seed: int, size: int = 20) -> np.ndarray:
    """Thin/narrow body template."""
    return _generate_caricature_for_build(seed, size, 3)


def generate_caricature_child(seed: int, size: int = 20) -> np.ndarray:
    """Child body template."""
    return _generate_caricature_for_build(seed, size, 4)


CARICATURE_TEMPLATES: list[tuple[str, Callable[[int, int], np.ndarray]]] = [
    ("standard", generate_caricature_standard),
    ("belly", generate_caricature_belly),
    ("broad", generate_caricature_broad),
    ("thin", generate_caricature_thin),
    ("child", generate_caricature_child),
]

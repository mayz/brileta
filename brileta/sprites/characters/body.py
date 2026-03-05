"""Body-part layer rendering (legs, belly, arms, neck, head, face)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from brileta import colors
from brileta.sprites.primitives import (
    draw_tapered_trunk,
    draw_thick_line,
    stamp_ellipse,
    stamp_fuzzy_circle,
)
from brileta.types import Facing

from .clothing import _draw_armor_torso, _draw_bare_torso, _draw_belly_mass
from .hair import HAIR_IDX_TALL

if TYPE_CHECKING:
    from .renderer import CharacterDrawContext


def _layer_legs(context: CharacterDrawContext) -> None:
    """Draw leg masses and toe/readability cues."""
    canvas = context.canvas
    appearance = context.appearance
    pose = context.pose
    params = context.params
    facing = context.facing
    cx = context.cx
    leg_bottom = context.leg_bottom
    leg_top = context.leg_top
    p_shadow, p_mid, _p_hi = appearance.pants_pal

    if not appearance.covers_legs:
        if facing in {Facing.EAST, Facing.WEST}:
            leg_positions = [
                (cx + pose.left_leg_dx, leg_bottom + pose.left_leg_dy),
                (cx + pose.right_leg_dx, leg_bottom + pose.right_leg_dy),
            ]
        else:
            leg_positions = [
                (
                    cx - params.leg_spacing + pose.left_leg_dx,
                    leg_bottom + pose.left_leg_dy,
                ),
                (
                    cx + params.leg_spacing + pose.right_leg_dx,
                    leg_bottom + pose.right_leg_dy,
                ),
            ]

        for lx, l_bot in leg_positions:
            l_bot_clamped = min(float(params.canvas_size - 1), l_bot)
            draw_tapered_trunk(
                canvas,
                lx,
                int(l_bot_clamped),
                leg_top,
                params.leg_w1 + 0.2,
                params.leg_w2 + 0.2,
                (*p_shadow, 235),
            )
            draw_tapered_trunk(
                canvas,
                lx,
                int(l_bot_clamped),
                leg_top,
                params.leg_w1,
                params.leg_w2,
                (*p_mid, 230),
            )

        if facing in {Facing.NORTH, Facing.SOUTH} and context.is_thin_legs:
            # Add a tiny upper-leg mass so thin builds do not read as disconnected
            # "sticks" under the torso at gameplay zoom levels.
            thigh_y = min(float(params.canvas_size - 1), leg_top + 1.1)
            stamp_ellipse(
                canvas,
                cx,
                thigh_y,
                max(0.6, params.leg_spacing * 0.62),
                0.42,
                (*p_shadow, 220),
                1.25,
                0.72,
            )
            for side in (-1, 1):
                stamp_fuzzy_circle(
                    canvas,
                    cx + side * max(0.95, params.leg_spacing * 0.92),
                    thigh_y + 0.12,
                    max(0.28, params.leg_w1 * 0.32),
                    (*p_shadow, 205),
                    1.2,
                    0.68,
                )

        if facing in {Facing.EAST, Facing.WEST}:
            # Side profile gets a visible toe protrusion toward facing direction.
            front_sign = -1.0
            toe_base_x = (
                cx + pose.left_leg_dx + front_sign * max(0.55, params.leg_w1 * 0.52)
            )
            toe_tip_x = toe_base_x + front_sign * max(0.7, params.leg_w1 * 0.68)
            toe_y = min(
                float(params.canvas_size - 1.0),
                leg_bottom + pose.left_leg_dy - 0.25,
            )
            draw_thick_line(
                canvas,
                toe_base_x,
                toe_y,
                toe_tip_x,
                toe_y + 0.01,
                (*p_shadow, 235),
                1,
            )
            stamp_fuzzy_circle(
                canvas,
                toe_tip_x + front_sign * 0.08,
                toe_y,
                max(0.28, params.leg_w1 * 0.22),
                (*p_mid, 220),
                1.1,
                0.6,
            )


def _layer_belly(context: CharacterDrawContext) -> None:
    """Draw belly/body mass that sits behind torso clothing."""
    if context.params.belly_ry > 0 and not context.appearance.covers_legs:
        _draw_belly_mass(
            context.canvas,
            context.cx,
            context.belly_cy,
            context.params,
            context.appearance.cloth_pal,
            context.facing,
        )


@dataclass(frozen=True)
class ArmLayerState:
    """Shared arm geometry/colors reused by back/front arm layers."""

    shoulder_y: float
    arm_end_y: float
    shoulder_anchor: float
    hand_inset: float
    arm_alpha: int
    hand_drop: float
    hand_alpha: int
    skin_mid: colors.Color
    arm_shadow_rgb: colors.Color
    arm_mid_rgb: colors.Color


def _build_arm_layer_state(context: CharacterDrawContext) -> ArmLayerState:
    """Resolve arm shading and anchor values for arm layer passes."""
    appearance = context.appearance
    params = context.params
    torso_cy = context.torso_cy
    belly_cy = context.belly_cy

    s_shadow, s_mid, _s_hi = appearance.skin_pal
    c_shadow, c_mid, _c_hi = appearance.cloth_pal
    shoulder_y = torso_cy - params.torso_ry * params.arm_shoulder_factor

    if params.arm_end_from_belly and params.belly_ry > 0:
        arm_end_y = belly_cy + params.belly_ry * params.arm_end_belly_factor
    else:
        arm_end_y = torso_cy + params.torso_ry * params.arm_end_torso_factor

    # Armor reads better when hands sit slightly lower on the torso silhouette.
    if appearance.clothing_fn is _draw_armor_torso:
        arm_end_y += 1.8

    arm_alpha = 220 if params.arm_thickness == 1 else 225
    hand_drop = params.arm_hand_drop
    hand_alpha = params.arm_hand_alpha
    shoulder_anchor = (
        params.torso_rx * params.arm_shoulder_anchor_torso_factor
        + params.shoulder_width * params.arm_shoulder_anchor_shoulder_width_factor
    )

    if appearance.clothing_fn is _draw_bare_torso:
        arm_shadow_rgb = (
            (s_shadow[0] + s_mid[0]) // 2,
            (s_shadow[1] + s_mid[1]) // 2,
            (s_shadow[2] + s_mid[2]) // 2,
        )
        arm_mid_rgb = s_mid
    else:
        arm_shadow_rgb = c_shadow
        arm_mid_rgb = c_mid

    return ArmLayerState(
        shoulder_y=shoulder_y,
        arm_end_y=arm_end_y,
        shoulder_anchor=shoulder_anchor,
        hand_inset=params.arm_hand_inset,
        arm_alpha=arm_alpha,
        hand_drop=hand_drop,
        hand_alpha=hand_alpha,
        skin_mid=s_mid,
        arm_shadow_rgb=arm_shadow_rgb,
        arm_mid_rgb=arm_mid_rgb,
    )


def _draw_frontback_arm(
    context: CharacterDrawContext,
    arm_state: ArmLayerState,
    side: int,
    arm_dy: float,
) -> None:
    """Draw one arm for NORTH/SOUTH facings, preserving prior per-side order."""
    appearance = context.appearance
    params = context.params
    arm_swing_scale = 0.5 if appearance.clothing_fn is _draw_armor_torso else 1.0

    arm_x = context.cx + side * arm_state.shoulder_anchor
    hand_x = arm_x - side * arm_state.hand_inset
    hand_y = arm_state.arm_end_y + arm_dy * arm_swing_scale

    stamp_fuzzy_circle(
        context.canvas,
        arm_x,
        arm_state.shoulder_y,
        max(0.45, params.arm_thickness * 0.38),
        (*arm_state.arm_shadow_rgb, 175),
        1.45,
        0.72,
    )

    # Two-pass arm stroke gives a readable limb edge without looking detached.
    draw_thick_line(
        context.canvas,
        arm_x + side * 0.08,
        arm_state.shoulder_y + 0.05,
        hand_x + side * 0.08,
        hand_y + 0.05,
        (*arm_state.arm_shadow_rgb, max(165, arm_state.arm_alpha - 30)),
        params.arm_thickness,
    )
    draw_thick_line(
        context.canvas,
        arm_x,
        arm_state.shoulder_y,
        hand_x,
        hand_y,
        (*arm_state.arm_mid_rgb, arm_state.arm_alpha),
        params.arm_thickness,
    )
    if params.hand_radius > 0:
        stamp_fuzzy_circle(
            context.canvas,
            hand_x,
            hand_y + arm_state.hand_drop,
            params.hand_radius,
            (*arm_state.skin_mid, arm_state.hand_alpha),
            2.0,
            0.85,
        )


def _layer_back_arm(context: CharacterDrawContext) -> None:
    """Draw the farther arm layer for the current facing."""
    arm_state = _build_arm_layer_state(context)
    if context.facing in {Facing.EAST, Facing.WEST}:
        # Side profile uses a clear foreground arm plus a faint background arm.
        back_sign = 1.0
        far_arm_x = context.cx + back_sign * context.params.torso_rx * 0.34
        draw_thick_line(
            context.canvas,
            far_arm_x,
            arm_state.shoulder_y + 0.05,
            far_arm_x + back_sign * 0.1,
            arm_state.arm_end_y + context.pose.right_arm_dy + 0.05,
            (*arm_state.arm_shadow_rgb, max(145, arm_state.arm_alpha - 55)),
            max(1, context.params.arm_thickness - 1),
        )
        return
    _draw_frontback_arm(context, arm_state, side=-1, arm_dy=context.pose.left_arm_dy)


def _layer_front_arm(context: CharacterDrawContext) -> None:
    """Draw the nearer arm layer for the current facing."""
    arm_state = _build_arm_layer_state(context)
    if context.facing in {Facing.EAST, Facing.WEST}:
        front_sign = -1.0
        front_arm_x = context.cx + front_sign * context.params.torso_rx * 0.62
        front_hand_x = front_arm_x + front_sign * 0.18
        hand_radius = (
            min(0.9, context.params.hand_radius)
            if context.params.hand_radius > 0
            else 0.0
        )

        stamp_fuzzy_circle(
            context.canvas,
            front_arm_x,
            arm_state.shoulder_y,
            max(0.45, context.params.arm_thickness * 0.38),
            (*arm_state.arm_shadow_rgb, 185),
            1.45,
            0.72,
        )
        draw_thick_line(
            context.canvas,
            front_arm_x + front_sign * 0.08,
            arm_state.shoulder_y,
            front_hand_x + front_sign * 0.08,
            arm_state.arm_end_y + context.pose.left_arm_dy,
            (*arm_state.arm_shadow_rgb, max(165, arm_state.arm_alpha - 30)),
            context.params.arm_thickness,
        )
        draw_thick_line(
            context.canvas,
            front_arm_x,
            arm_state.shoulder_y,
            front_hand_x,
            arm_state.arm_end_y + context.pose.left_arm_dy,
            (*arm_state.arm_mid_rgb, arm_state.arm_alpha),
            context.params.arm_thickness,
        )
        if hand_radius > 0:
            stamp_fuzzy_circle(
                context.canvas,
                front_hand_x,
                arm_state.arm_end_y + context.pose.left_arm_dy + arm_state.hand_drop,
                hand_radius,
                (*arm_state.skin_mid, arm_state.hand_alpha),
                2.0,
                0.85,
            )
        return
    _draw_frontback_arm(context, arm_state, side=1, arm_dy=context.pose.right_arm_dy)


def _layer_neck(context: CharacterDrawContext) -> None:
    """Draw neck for visible facings."""
    params = context.params
    appearance = context.appearance
    if context.facing != Facing.NORTH and params.neck_rx > 0 and params.neck_ry > 0:
        neck_y = context.head_cy + context.hr + params.neck_y_offset
        stamp_ellipse(
            context.canvas,
            context.cx,
            neck_y,
            params.neck_rx,
            params.neck_ry,
            (*appearance.skin_pal[1], 225),
            2.0,
            0.85,
        )


def _layer_head(context: CharacterDrawContext) -> None:
    """Draw base head mass before hair."""
    appearance = context.appearance
    facing = context.facing
    hx = context.cx
    hy = context.head_cy
    hr = context.hr
    canvas = context.canvas

    def stamp_head_circle(
        cx: float,
        cy: float,
        radius: float,
        tone_idx: int,
        alpha: int,
        falloff: float,
        hardness: float,
    ) -> None:
        stamp_fuzzy_circle(
            canvas,
            cx,
            cy,
            radius,
            (*appearance.skin_pal[tone_idx], alpha),
            falloff,
            hardness,
        )

    def stamp_head_ellipse(
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        tone_idx: int,
        alpha: int,
        falloff: float,
        hardness: float,
    ) -> None:
        stamp_ellipse(
            canvas,
            cx,
            cy,
            rx,
            ry,
            (*appearance.skin_pal[tone_idx], alpha),
            falloff,
            hardness,
        )

    if facing == Facing.NORTH:
        for cy, radius, tone_idx, alpha, falloff, hardness in (
            (hy + 0.2, hr + 0.2, 0, 240, 2.0, 0.88),
            (hy, hr, 1, 235, 2.0, 0.88),
        ):
            stamp_head_circle(hx, cy, radius, tone_idx, alpha, falloff, hardness)
    elif facing in {Facing.EAST, Facing.WEST}:
        for (
            cx,
            cy,
            rx,
            ry,
            tone_idx,
            alpha,
            falloff,
            hardness,
        ) in (
            (hx, hy + 0.2, hr * 0.85, hr + 0.2, 0, 240, 2.0, 0.88),
            (hx, hy, hr * 0.8, hr, 1, 235, 2.0, 0.88),
            (
                hx + hr * -0.14,
                hy + hr * -0.2,
                hr * 0.45,
                hr * 0.55,
                2,
                210,
                1.6,
                0.75,
            ),
            (
                hx + hr * -0.26,
                hy + hr * -0.12,
                hr * 0.34,
                hr * 0.48,
                2,
                205,
                1.5,
                0.72,
            ),
        ):
            stamp_head_ellipse(cx, cy, rx, ry, tone_idx, alpha, falloff, hardness)
        if appearance.hair_style_idx != HAIR_IDX_TALL:
            stamp_head_ellipse(
                hx + hr * 0.28,
                hy + hr * 0.03,
                hr * 0.3,
                hr * 0.46,
                0,
                195,
                1.5,
                0.72,
            )
        stamp_head_circle(hx + hr * -0.63, hy + hr * -0.02, hr * 0.12, 2, 190, 1.4, 0.7)
    else:
        for cy, radius, tone_idx, alpha, falloff, hardness in (
            (hy + 0.2, hr + 0.2, 0, 240, 2.0, 0.88),
            (hy, hr, 1, 235, 2.0, 0.88),
            (hy + hr * -0.22, hr * 0.6, 2, 210, 1.6, 0.75),
        ):
            stamp_head_circle(hx, cy, radius, tone_idx, alpha, falloff, hardness)


def _layer_face_final(context: CharacterDrawContext) -> None:
    """Draw final side-profile nose overlay."""
    if context.facing in {Facing.EAST, Facing.WEST}:
        front_sign = -1.0
        nose_x = float(round(context.cx + front_sign * context.hr * 0.9))
        nose_y = float(round(context.head_cy - context.hr * 0.02))
        s_shadow = context.appearance.skin_pal[0]
        stamp_fuzzy_circle(
            context.canvas,
            nose_x,
            nose_y,
            max(0.24, context.hr * 0.08),
            (*s_shadow, 230),
            1.0,
            0.9,
        )
        stamp_fuzzy_circle(
            context.canvas,
            nose_x + front_sign * 0.35,
            nose_y + 0.02,
            max(0.2, context.hr * 0.06),
            (*s_shadow, 220),
            1.0,
            0.85,
        )

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
from .specs import (
    CircleStampSpec,
    EllipseStampSpec,
    ShapeContext,
    ToneStampSpec,
    _expr,
    _stamp_tone_specs,
)

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
    arm_alpha: int
    hand_drop: float
    hand_alpha: int
    skin_mid: colors.Color
    arm_shadow_rgb: colors.Color
    arm_mid_rgb: colors.Color
    is_broad: bool
    is_belly: bool
    is_thin: bool
    is_child: bool


def _build_arm_layer_state(context: CharacterDrawContext) -> ArmLayerState:
    """Resolve arm shading and anchor values for arm layer passes."""
    appearance = context.appearance
    params = context.params
    torso_cy = context.torso_cy
    belly_cy = context.belly_cy

    s_shadow, s_mid, _s_hi = appearance.skin_pal
    c_shadow, c_mid, _c_hi = appearance.cloth_pal
    is_broad = params.shoulder_width > params.torso_rx
    is_belly = params.belly_ry > 0
    is_thin = (
        params.arm_thickness == 1 and params.hand_radius > 0 and params.neck_rx > 0
    )
    is_child = params.neck_rx == 0 and params.hand_radius == 0

    if is_child or is_belly:
        shoulder_factor = 0.18
    elif is_thin:
        shoulder_factor = 0.26
    else:
        shoulder_factor = 0.28
    shoulder_y = torso_cy - params.torso_ry * shoulder_factor

    if params.arm_end_from_belly and params.belly_ry > 0:
        arm_end_y = belly_cy + params.belly_ry * 0.22
    elif is_broad:
        arm_end_y = torso_cy + params.torso_ry * 0.62
    elif is_thin:
        arm_end_y = torso_cy + params.torso_ry * 0.68
    elif is_child:
        arm_end_y = torso_cy + params.torso_ry * 0.64
    else:
        arm_end_y = torso_cy + params.torso_ry * 0.66

    # Armor reads better when hands sit slightly lower on the torso silhouette.
    if appearance.clothing_fn is _draw_armor_torso:
        arm_end_y += 1.8

    arm_alpha = 220 if params.arm_thickness == 1 else 225
    hand_drop = 0.3 if is_thin else 0.5
    hand_alpha = 210 if is_thin else 215

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
        arm_alpha=arm_alpha,
        hand_drop=hand_drop,
        hand_alpha=hand_alpha,
        skin_mid=s_mid,
        arm_shadow_rgb=arm_shadow_rgb,
        arm_mid_rgb=arm_mid_rgb,
        is_broad=is_broad,
        is_belly=is_belly,
        is_thin=is_thin,
        is_child=is_child,
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
    shoulder_base = params.shoulder_width if arm_state.is_broad else params.torso_rx
    if arm_state.is_broad:
        shoulder_anchor = (
            params.torso_rx * 0.82 + (params.shoulder_width - params.torso_rx) * 0.12
        )
        hand_inset = 0.24
    elif arm_state.is_belly:
        shoulder_anchor = shoulder_base * 0.85
        hand_inset = 0.2
    elif arm_state.is_child:
        shoulder_anchor = shoulder_base * 0.8
        hand_inset = 0.12
    elif arm_state.is_thin:
        shoulder_anchor = shoulder_base * 0.82
        hand_inset = 0.14
    else:
        shoulder_anchor = shoulder_base * 0.84
        hand_inset = 0.18

    arm_x = context.cx + side * shoulder_anchor
    hand_x = arm_x - side * hand_inset
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
        is_broad = params.shoulder_width > params.torso_rx
        is_belly = params.belly_ry > 0
        is_thin = (
            params.arm_thickness == 1 and params.hand_radius > 0 and params.neck_rx > 0
        )
        if is_belly or is_broad:
            neck_y = context.head_cy + context.hr + 0.1
        elif is_thin:
            neck_y = context.head_cy + context.hr + 0.15
        else:
            neck_y = context.head_cy + context.hr + params.neck_gap * 0.5
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
    values = {"hx": hx, "hy": hy, "hr": hr}

    if facing == Facing.NORTH:
        specs: tuple[ToneStampSpec, ...] = (
            CircleStampSpec(
                cx=_expr("hx"),
                cy=_expr("hy", None, offset=0.2),
                radius=_expr("hr", None, offset=0.2),
                tone_idx=0,
                alpha=240,
                falloff=2.0,
                hardness=0.88,
            ),
            CircleStampSpec(
                cx=_expr("hx"),
                cy=_expr("hy"),
                radius=_expr("hr"),
                tone_idx=1,
                alpha=235,
                falloff=2.0,
                hardness=0.88,
            ),
        )
    elif facing in {Facing.EAST, Facing.WEST}:
        specs_list: list[ToneStampSpec] = [
            EllipseStampSpec(
                cx=_expr("hx"),
                cy=_expr("hy", None, offset=0.2),
                rx=_expr("hr", None, factor=0.85),
                ry=_expr("hr", None, offset=0.2),
                tone_idx=0,
                alpha=240,
                falloff=2.0,
                hardness=0.88,
            ),
            EllipseStampSpec(
                cx=_expr("hx"),
                cy=_expr("hy"),
                rx=_expr("hr", None, factor=0.8),
                ry=_expr("hr"),
                tone_idx=1,
                alpha=235,
                falloff=2.0,
                hardness=0.88,
            ),
            EllipseStampSpec(
                cx=_expr("hx", "hr", -0.14),
                cy=_expr("hy", "hr", -0.2),
                rx=_expr("hr", None, factor=0.45),
                ry=_expr("hr", None, factor=0.55),
                tone_idx=2,
                alpha=210,
                falloff=1.6,
                hardness=0.75,
            ),
            EllipseStampSpec(
                cx=_expr("hx", "hr", -0.26),
                cy=_expr("hy", "hr", -0.12),
                rx=_expr("hr", None, factor=0.34),
                ry=_expr("hr", None, factor=0.48),
                tone_idx=2,
                alpha=205,
                falloff=1.5,
                hardness=0.72,
            ),
        ]
        if appearance.hair_style_idx != 4:
            specs_list.append(
                EllipseStampSpec(
                    cx=_expr("hx", "hr", 0.28),
                    cy=_expr("hy", "hr", 0.03),
                    rx=_expr("hr", None, factor=0.3),
                    ry=_expr("hr", None, factor=0.46),
                    tone_idx=0,
                    alpha=195,
                    falloff=1.5,
                    hardness=0.72,
                )
            )
        specs_list.append(
            CircleStampSpec(
                cx=_expr("hx", "hr", -0.63),
                cy=_expr("hy", "hr", -0.02),
                radius=_expr("hr", None, factor=0.12),
                tone_idx=2,
                alpha=190,
                falloff=1.4,
                hardness=0.7,
            )
        )
        specs = tuple(specs_list)
    else:
        specs = (
            CircleStampSpec(
                cx=_expr("hx"),
                cy=_expr("hy", None, offset=0.2),
                radius=_expr("hr", None, offset=0.2),
                tone_idx=0,
                alpha=240,
                falloff=2.0,
                hardness=0.88,
            ),
            CircleStampSpec(
                cx=_expr("hx"),
                cy=_expr("hy"),
                radius=_expr("hr"),
                tone_idx=1,
                alpha=235,
                falloff=2.0,
                hardness=0.88,
            ),
            CircleStampSpec(
                cx=_expr("hx"),
                cy=_expr("hy", "hr", -0.22),
                radius=_expr("hr", None, factor=0.6),
                tone_idx=2,
                alpha=210,
                falloff=1.6,
                hardness=0.75,
            ),
        )

    _stamp_tone_specs(context.canvas, appearance.skin_pal, specs, ShapeContext(values))


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

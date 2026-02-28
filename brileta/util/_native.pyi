from typing import Any

class SpatialHashGrid:
    cell_size: int
    def __init__(self, cell_size: int = 16) -> None: ...
    def add(self, obj: Any) -> None: ...
    def remove(self, obj: Any) -> None: ...
    def update(self, obj: Any) -> None: ...
    def get_at_point(self, x: int, y: int) -> list[Any]: ...
    def get_in_radius(self, x: int, y: int, radius: int) -> list[Any]: ...
    def get_in_bounds(self, x1: int, y1: int, x2: int, y2: int) -> list[Any]: ...
    def get_in_rect(self, rect: Any) -> list[Any]: ...
    def clear(self) -> None: ...

class _NoiseState:
    seed: int
    frequency: float
    def __init__(
        self,
        seed: int = 1337,
        noise_type: int = 0,
        frequency: float = 0.01,
        fractal_type: int = 0,
        octaves: int = 3,
        lacunarity: float = 2.0,
        gain: float = 0.5,
        weighted_strength: float = 0.0,
        ping_pong_strength: float = 2.0,
        cellular_distance_func: int = 1,
        cellular_return_type: int = 1,
        cellular_jitter_mod: float = 1.0,
        domain_warp_type: int = 0,
        domain_warp_amp: float = 1.0,
    ) -> None: ...
    def sample_2d(self, x: float, y: float) -> float: ...
    def sample_3d(self, x: float, y: float, z: float) -> float: ...
    def sample_2d_array(self, xs: object, ys: object, out: object) -> None: ...
    def sample_3d_array(
        self, xs: object, ys: object, zs: object, out: object
    ) -> None: ...
    def domain_warp_2d(self, x: float, y: float) -> tuple[float, float]: ...
    def domain_warp_3d(
        self, x: float, y: float, z: float
    ) -> tuple[float, float, float]: ...

class WFCContradictionError(Exception): ...

def astar(
    cost: object,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
) -> list[tuple[int, int]]: ...
def fov(
    transparent: object,
    visible: object,
    origin_x: int,
    origin_y: int,
    radius: int,
) -> None: ...
def wfc_solve(
    width: int,
    height: int,
    num_patterns: int,
    propagation_masks: object,
    pattern_weights: object,
    initial_wave: object,
    seed: int,
) -> list[list[int]]: ...

# Sprite drawing primitives (from _native_sprites.c)

def sprite_alpha_blend(
    canvas: object,
    x: int,
    y: int,
    r: int,
    g: int,
    b: int,
    a: int,
) -> None: ...
def sprite_composite_over(
    canvas: object,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
    src_alpha: object,
    r: int,
    g: int,
    b: int,
) -> None: ...
def sprite_draw_line(
    canvas: object,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    r: int,
    g: int,
    b: int,
    a: int,
) -> None: ...
def sprite_draw_thick_line(
    canvas: object,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    r: int,
    g: int,
    b: int,
    a: int,
    thickness: int,
) -> None: ...
def sprite_draw_tapered_trunk(
    canvas: object,
    cx: float,
    y_bottom: int,
    y_top: int,
    w_bottom: float,
    w_top: float,
    r: int,
    g: int,
    b: int,
    a: int,
    root_flare: int,
) -> None: ...
def sprite_stamp_fuzzy_circle(
    canvas: object,
    cx: float,
    cy: float,
    radius: float,
    r: int,
    g: int,
    b: int,
    a: int,
    falloff: float,
    hardness: float,
) -> None: ...
def sprite_stamp_ellipse(
    canvas: object,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    r: int,
    g: int,
    b: int,
    a: int,
    falloff: float,
    hardness: float,
) -> None: ...
def sprite_batch_stamp_ellipses(
    canvas: object,
    ellipses: list[tuple[float, float, float, float]],
    r: int,
    g: int,
    b: int,
    a: int,
    falloff: float,
    hardness: float,
) -> None: ...
def sprite_batch_stamp_circles(
    canvas: object,
    circles: list[tuple[float, float, float]],
    r: int,
    g: int,
    b: int,
    a: int,
    falloff: float,
    hardness: float,
) -> None: ...
def sprite_generate_deciduous_canopy(
    canvas: object,
    seed: int,
    size: int,
    canopy_cx: float,
    canopy_cy: float,
    base_radius: float,
    crown_rx_scale: float,
    crown_ry_scale: float,
    canopy_center_x_offset: float,
    tips: list[tuple[float, float]],
    shadow_r: int,
    shadow_g: int,
    shadow_b: int,
    shadow_a: int,
    mid_r: int,
    mid_g: int,
    mid_b: int,
    mid_a: int,
    highlight_r: int,
    highlight_g: int,
    highlight_b: int,
    highlight_a: int,
) -> list[tuple[float, float]]: ...
def sprite_fill_triangle(
    canvas: object,
    cx: float,
    top_y: int,
    base_width: float,
    height: int,
    r: int,
    g: int,
    b: int,
    a: int,
) -> None: ...
def sprite_paste_sprite(
    sheet: object,
    sprite: object,
    x0: int,
    y0: int,
) -> None: ...
def sprite_darken_rim(
    canvas: object,
    darken_r: int,
    darken_g: int,
    darken_b: int,
) -> None: ...
def sprite_nibble_canopy(
    canvas: object,
    seed: int,
    center_x: float,
    center_y: float,
    canopy_radius: float,
    nibble_prob: float,
    interior_prob: float,
) -> None: ...
def sprite_nibble_boulder(
    canvas: object,
    seed: int,
    nibble_prob: float,
) -> None: ...

# Glyph vertex encoding (from _native_glyph_vertices.c)

def build_glyph_vertices(
    glyph_data: object,
    output: object,
    uv_map: object,
    cp437_map: object,
    tile_w: float,
    tile_h: float,
) -> int: ...

// Glyph rendering shader with foreground/background color mixing
// and optional per-pixel sub-tile brightness noise.
//
// The noise effect uses a PCG hash to create deterministic brightness
// variation within pattern-defined sub-cells per tile, making terrain like
// cobblestone, thatch, and shingles read as different surface textures.

// Vertex input structure
struct VertexInput {
    @location(0) in_vert: vec2<f32>,
    @location(1) in_uv: vec2<f32>,
    @location(2) in_fg_color: vec4<f32>,
    @location(3) in_bg_color: vec4<f32>,
    @location(4) in_noise_amplitude: f32,
    @location(5) in_noise_pattern: u32,
    @location(6) in_edge_neighbor_mask: u32,
    @location(7) in_edge_blend: f32,
    @location(8) in_edge_neighbor_bg_0: vec3<f32>,
    @location(9) in_edge_neighbor_bg_1: vec3<f32>,
    @location(10) in_edge_neighbor_bg_2: vec3<f32>,
    @location(11) in_edge_neighbor_bg_3: vec3<f32>,
    // Sub-tile split for perspective offset boundary tiles
    @location(12) in_split_y: f32,
    @location(13) in_split_bg_color: vec4<f32>,
    @location(14) in_split_fg_color: vec4<f32>,
    @location(15) in_split_noise_amplitude: f32,
    @location(16) in_split_noise_pattern: u32,
}

// Vertex output / Fragment input structure
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) v_uv: vec2<f32>,
    @location(1) v_fg_color: vec4<f32>,
    @location(2) v_bg_color: vec4<f32>,
    @location(3) v_noise_amplitude: f32,
    @location(4) @interpolate(flat) v_noise_pattern: u32,
    @location(5) v_pixel_pos: vec2<f32>,
    @location(6) @interpolate(flat) v_edge_neighbor_mask: u32,
    @location(7) v_edge_blend: f32,
    @location(8) v_edge_neighbor_bg_0: vec3<f32>,
    @location(9) v_edge_neighbor_bg_1: vec3<f32>,
    @location(10) v_edge_neighbor_bg_2: vec3<f32>,
    @location(11) v_edge_neighbor_bg_3: vec3<f32>,
    // Sub-tile split for perspective offset boundary tiles
    @location(12) v_split_y: f32,
    @location(13) v_split_bg_color: vec4<f32>,
    @location(14) v_split_fg_color: vec4<f32>,
    @location(15) v_split_noise_amplitude: f32,
    @location(16) @interpolate(flat) v_split_noise_pattern: u32,
}

// Uniform buffer: texture size + noise parameters + world-space tile offset.
// Layout ordered with vec2s first for natural 8-byte alignment.
struct Uniforms {
    u_texture_size: vec2<f32>,  // offset 0, 8 bytes
    u_tile_size: vec2<f32>,     // offset 8, 8 bytes (x=width, y=height)
    u_tile_offset: vec2<i32>,   // offset 16, 8 bytes - world origin for stable hashing
    u_noise_seed: u32,          // offset 24, 4 bytes (+4 implicit padding to 32)
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var u_atlas: texture_2d<f32>;
@group(0) @binding(2) var u_sampler: sampler;

// Sub-tile pattern IDs (must match brileta.environment.tile_types).
const SUB_TILE_PATTERN_BLOCKS_2X2: u32 = 0u;
const SUB_TILE_PATTERN_FINE_GRAIN: u32 = 1u;
const SUB_TILE_PATTERN_STAGGERED_ROWS: u32 = 2u;
const SUB_TILE_PATTERN_DIAGONAL_BANDS: u32 = 3u;
const SUB_TILE_PATTERN_HORIZONTAL_RIBS: u32 = 4u;
const SUB_TILE_PATTERN_VERTICAL_RIBS: u32 = 5u;
const SUB_TILE_PATTERN_STAGGERED_COLUMNS: u32 = 6u;

// PCG hash - fast, high-quality 32-bit hash for spatial noise.
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Cardinal edge direction index order (matches WorldView transport schema):
// 0=W, 1=N, 2=S, 3=E
fn edge_distance_px(
    dir_idx: u32,
    local_x: f32,
    local_y: f32,
    tile_w: f32,
    tile_h: f32,
) -> f32 {
    let dist_left = max(local_x, 0.0);
    let dist_right = max(tile_w - local_x, 0.0);
    let dist_top = max(local_y, 0.0);
    let dist_bottom = max(tile_h - local_y, 0.0);

    switch dir_idx {
        case 0u: { return dist_left; }
        case 1u: { return dist_top; }
        case 2u: { return dist_bottom; }
        case 3u: { return dist_right; }
        default: { return 1e9; }
    }
}

// Vertex shader stage
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    output.v_uv = input.in_uv;
    output.v_fg_color = input.in_fg_color;
    output.v_bg_color = input.in_bg_color;
    output.v_noise_amplitude = input.in_noise_amplitude;
    output.v_noise_pattern = input.in_noise_pattern;
    output.v_edge_neighbor_mask = input.in_edge_neighbor_mask;
    output.v_edge_blend = input.in_edge_blend;
    output.v_edge_neighbor_bg_0 = input.in_edge_neighbor_bg_0;
    output.v_edge_neighbor_bg_1 = input.in_edge_neighbor_bg_1;
    output.v_edge_neighbor_bg_2 = input.in_edge_neighbor_bg_2;
    output.v_edge_neighbor_bg_3 = input.in_edge_neighbor_bg_3;
    output.v_split_y = input.in_split_y;
    output.v_split_bg_color = input.in_split_bg_color;
    output.v_split_fg_color = input.in_split_fg_color;
    output.v_split_noise_amplitude = input.in_split_noise_amplitude;
    output.v_split_noise_pattern = input.in_split_noise_pattern;
    // Pass raw pixel position through for sub-tile coordinate computation
    output.v_pixel_pos = input.in_vert;

    let x = (input.in_vert.x / uniforms.u_texture_size.x) * 2.0 - 1.0;
    let y = (1.0 - (input.in_vert.y / uniforms.u_texture_size.y)) * 2.0 - 1.0;
    output.position = vec4<f32>(x, y, 0.0, 1.0);

    return output;
}

// Fragment shader stage
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample the character tile from the atlas
    let char_alpha = textureSample(u_atlas, u_sampler, input.v_uv).a;

    // Tile dimensions: width and height may differ (non-square character cells)
    let tile_w = uniforms.u_tile_size.x;
    let tile_h = uniforms.u_tile_size.y;
    // Buffer-space tile index from pixel position
    let buf_tile_x = i32(floor(input.v_pixel_pos.x / tile_w));
    let buf_tile_y = i32(floor(input.v_pixel_pos.y / tile_h));
    // Convert to world-space tile coordinates for stable hashing
    let world_tile_x = u32(buf_tile_x + uniforms.u_tile_offset.x);
    let world_tile_y = u32(buf_tile_y + uniforms.u_tile_offset.y);
    let local_x = input.v_pixel_pos.x - f32(buf_tile_x) * tile_w;
    let local_y = input.v_pixel_pos.y - f32(buf_tile_y) * tile_h;

    // Sub-tile split for perspective offset boundary tiles. When split_y > 0,
    // the tile is divided at y_frac == split_y: above the threshold uses the
    // primary appearance (roof), below uses the split appearance (wall face).
    // The split region also gets a smooth depth gradient - darker toward the
    // bottom - to make wall faces read as 3D surfaces lit from above.
    var fg_color = input.v_fg_color;
    var noise_amplitude = input.v_noise_amplitude;
    var noise_pattern = input.v_noise_pattern;
    let y_frac = local_y / tile_h;
    let in_split = input.v_split_y > 0.0 && y_frac > input.v_split_y;
    if (in_split) {
        fg_color = input.v_split_fg_color;
        noise_amplitude = input.v_split_noise_amplitude;
        noise_pattern = input.v_split_noise_pattern;
    }

    var bg_color = select(input.v_bg_color, input.v_split_bg_color, in_split);

    // Smooth depth gradient within the split region: darken progressively
    // toward the bottom. Only applied when the split region is darker than
    // the primary (wall face under roof). Skipped when the split region is
    // brighter (e.g. chimney body above roof) so the roof portion reads as
    // normal roof surface, not a darkened wall face.
    if (in_split) {
        let primary_lum = dot(input.v_bg_color.rgb, vec3<f32>(0.299, 0.587, 0.114));
        let split_lum = dot(input.v_split_bg_color.rgb, vec3<f32>(0.299, 0.587, 0.114));
        if (split_lum < primary_lum) {
            let split_depth = (y_frac - input.v_split_y) / (1.0 - input.v_split_y);
            let darken = split_depth * 0.18;
            bg_color = vec4<f32>(bg_color.rgb * (1.0 - darken), bg_color.a);
            fg_color = vec4<f32>(fg_color.rgb * (1.0 - darken * 0.5), fg_color.a);
        }
    }
    if (input.v_edge_blend > 0.0 && input.v_edge_neighbor_mask != 0u) {
        let edge_blend = clamp(input.v_edge_blend, 0.0, 1.0);
        // With cardinal-only + one-sided ownership in the CPU mask, we can use a
        // slightly wider band so the contour is actually visible in motion.
        let base_extent = clamp(1.4 + edge_blend * 4.0, 1.6, 3.8);
        // Use pixel-scale hashing so contours look irregular instead of a straight line.
        let local_px_x = u32(clamp(floor(local_x), 0.0, max(tile_w - 1.0, 0.0)));
        let local_px_y = u32(clamp(floor(local_y), 0.0, max(tile_h - 1.0, 0.0)));
        let neighbor_colors = array<vec3<f32>, 4>(
            input.v_edge_neighbor_bg_0,
            input.v_edge_neighbor_bg_1,
            input.v_edge_neighbor_bg_2,
            input.v_edge_neighbor_bg_3,
        );

        var best_strength = 0.0;
        var best_color = bg_color.rgb;

        for (var dir_idx: u32 = 0u; dir_idx < 4u; dir_idx = dir_idx + 1u) {
            if ((input.v_edge_neighbor_mask & (1u << dir_idx)) == 0u) {
                continue;
            }

            let extent = base_extent;
            let dist = edge_distance_px(dir_idx, local_x, local_y, tile_w, tile_h);
            if (dist > extent + 1.0) {
                continue;
            }

            let dir_salt = (dir_idx + 1u) * 2246822519u;
            let pixel_salt = (local_px_x * 73856093u) ^ (local_px_y * 19349663u);
            let h = pcg_hash(
                world_tile_x
                ^ (world_tile_y * 2654435761u)
                ^ dir_salt
                ^ pixel_salt
                ^ uniforms.u_noise_seed
            );
            let noise_01 = f32(h & 0xFFFFu) / 65535.0;
            // Moderate directional jitter in pixel units. The CPU-side
            // restrictions keep this from turning into interior "canals".
            let jitter_px = (noise_01 * 2.0 - 1.0) * (0.35 + edge_blend * 0.95);
            let edge_value = extent - dist + jitter_px;
            let strength = smoothstep(-0.35, 0.35, edge_value);

            if (strength > best_strength) {
                best_strength = strength;
                best_color = neighbor_colors[dir_idx];
            }
        }

        // Add explicit corner rounding when two perpendicular boundary edges
        // meet. Edge strips alone produce fuzzy but still axis-aligned corners.
        let west_bit = 1u << 0u;
        let north_bit = 1u << 1u;
        let south_bit = 1u << 2u;
        let east_bit = 1u << 3u;
        // Corners need a larger radius than edge strips to read as arcs.
        // Clamp to a fraction of tile size so high edge_blend values don't
        // consume the whole tile.
        let corner_radius = min(
            base_extent + (2.8 + edge_blend * 3.0),
            min(tile_w, tile_h) * 0.72,
        );
        // Keep corner jitter lower than edge-strip jitter so corners stay round.
        let corner_jitter_amp = 0.10 + edge_blend * 0.24;
        let corner_softness = 0.24;

        var best_corner_strength = 0.0;
        var best_corner_color = best_color;

        // NW corner: N + W
        if ((input.v_edge_neighbor_mask & (north_bit | west_bit)) == (north_bit | west_bit)) {
            let dist = length(vec2<f32>(max(local_x, 0.0), max(local_y, 0.0)));
            let h = pcg_hash(
                world_tile_x
                ^ (world_tile_y * 2654435761u)
                ^ (local_px_x * 73856093u)
                ^ (local_px_y * 19349663u)
                ^ 0x9E3779B9u
                ^ uniforms.u_noise_seed
            );
            let noise_01 = f32(h & 0xFFFFu) / 65535.0;
            let jitter_px = (noise_01 * 2.0 - 1.0) * corner_jitter_amp;
            let strength = smoothstep(
                -corner_softness,
                corner_softness,
                (corner_radius - dist) + jitter_px,
            );
            if (strength > best_corner_strength) {
                best_corner_strength = strength;
                best_corner_color = (neighbor_colors[0u] + neighbor_colors[1u]) * 0.5;
            }
        }

        // NE corner: N + E
        if ((input.v_edge_neighbor_mask & (north_bit | east_bit)) == (north_bit | east_bit)) {
            let dist = length(vec2<f32>(max(tile_w - local_x, 0.0), max(local_y, 0.0)));
            let h = pcg_hash(
                world_tile_x
                ^ (world_tile_y * 2654435761u)
                ^ (local_px_x * 73856093u)
                ^ (local_px_y * 19349663u)
                ^ 0x85EBCA6Bu
                ^ uniforms.u_noise_seed
            );
            let noise_01 = f32(h & 0xFFFFu) / 65535.0;
            let jitter_px = (noise_01 * 2.0 - 1.0) * corner_jitter_amp;
            let strength = smoothstep(
                -corner_softness,
                corner_softness,
                (corner_radius - dist) + jitter_px,
            );
            if (strength > best_corner_strength) {
                best_corner_strength = strength;
                best_corner_color = (neighbor_colors[1u] + neighbor_colors[3u]) * 0.5;
            }
        }

        // SW corner: S + W
        if ((input.v_edge_neighbor_mask & (south_bit | west_bit)) == (south_bit | west_bit)) {
            let dist = length(vec2<f32>(max(local_x, 0.0), max(tile_h - local_y, 0.0)));
            let h = pcg_hash(
                world_tile_x
                ^ (world_tile_y * 2654435761u)
                ^ (local_px_x * 73856093u)
                ^ (local_px_y * 19349663u)
                ^ 0xC2B2AE35u
                ^ uniforms.u_noise_seed
            );
            let noise_01 = f32(h & 0xFFFFu) / 65535.0;
            let jitter_px = (noise_01 * 2.0 - 1.0) * corner_jitter_amp;
            let strength = smoothstep(
                -corner_softness,
                corner_softness,
                (corner_radius - dist) + jitter_px,
            );
            if (strength > best_corner_strength) {
                best_corner_strength = strength;
                best_corner_color = (neighbor_colors[0u] + neighbor_colors[2u]) * 0.5;
            }
        }

        // SE corner: S + E
        if ((input.v_edge_neighbor_mask & (south_bit | east_bit)) == (south_bit | east_bit)) {
            let dist = length(vec2<f32>(max(tile_w - local_x, 0.0), max(tile_h - local_y, 0.0)));
            let h = pcg_hash(
                world_tile_x
                ^ (world_tile_y * 2654435761u)
                ^ (local_px_x * 73856093u)
                ^ (local_px_y * 19349663u)
                ^ 0x27D4EB2Fu
                ^ uniforms.u_noise_seed
            );
            let noise_01 = f32(h & 0xFFFFu) / 65535.0;
            let jitter_px = (noise_01 * 2.0 - 1.0) * corner_jitter_amp;
            let strength = smoothstep(
                -corner_softness,
                corner_softness,
                (corner_radius - dist) + jitter_px,
            );
            if (strength > best_corner_strength) {
                best_corner_strength = strength;
                best_corner_color = (neighbor_colors[2u] + neighbor_colors[3u]) * 0.5;
            }
        }

        // Slightly favor explicit corner rounding over strip blends so corners
        // don't snap back to a chamfered/right-angle look.
        if (best_corner_strength + 0.16 > best_strength) {
            best_strength = best_corner_strength;
            best_color = best_corner_color;
        }

        if (best_strength > 0.0) {
            bg_color = vec4<f32>(mix(bg_color.rgb, best_color, best_strength), bg_color.a);
        }
    }

    // Mix foreground and background colors based on texture alpha.
    // If char_alpha is 1.0 (opaque pixel), the result is fg_color.
    // If char_alpha is 0.0 (transparent pixel), the result is bg_color.
    var color = mix(bg_color, fg_color, char_alpha);

    // Apply sub-tile brightness noise when amplitude is non-zero.
    // Divides each tile into a pattern-defined set of sub-cells and hashes
    // (world_tile_x, world_tile_y, sub_cell, seed) to produce a brightness offset.
    // Uses world-space tile coordinates (buffer tile + offset uniform) so the
    // noise pattern stays anchored to world tiles regardless of camera scrolling.
    if (noise_amplitude > 0.0) {
        let pattern = noise_pattern;
        var sub_cell: u32;
        var pattern_offset = 0.0;
        var use_tile_hash = true;

        switch pattern {
            case SUB_TILE_PATTERN_BLOCKS_2X2: {
                // 2x2 blocks (backward-compatible default behavior)
                let sub_x = u32(floor(local_x / (tile_w * 0.5)));
                let sub_y = u32(floor(local_y / (tile_h * 0.5)));
                sub_cell = sub_x + sub_y * 2u;
            }
            case SUB_TILE_PATTERN_FINE_GRAIN: {
                // Fine-grain organic noise: hash from world pixel position so the
                // noise field is continuous across tile boundaries (no visible grid).
                // ~2px cells produce per-pixel-scale variation that reads as fuzzy
                // organic texture (straw/fur).
                let world_px_x = world_tile_x * u32(tile_w) + u32(local_x);
                let world_px_y = world_tile_y * u32(tile_h) + u32(local_y);
                let fine_x = world_px_x / 3u;  // ~3px wide (horizontal straw direction)
                let fine_y = world_px_y / 4u;  // ~4px tall
                let fine_hash_input = fine_x ^ (fine_y * 2654435761u) ^ uniforms.u_noise_seed;
                let fine_h = pcg_hash(fine_hash_input);
                let fine_noise_01 = f32(fine_h & 0xFFFFu) / 65535.0;
                let fine_offset = (fine_noise_01 * 2.0 - 1.0) * noise_amplitude;
                color = vec4<f32>(
                    clamp(color.r + fine_offset, 0.0, 1.0),
                    clamp(color.g + fine_offset, 0.0, 1.0),
                    clamp(color.b + fine_offset, 0.0, 1.0),
                    color.a,
                );
                use_tile_hash = false;
            }
            case SUB_TILE_PATTERN_STAGGERED_ROWS: {
                // Staggered rows: 3 rows x 2 cols, odd rows shifted by half tile width
                let row_h = tile_h / 3.0;
                let row = u32(clamp(floor(local_y / row_h), 0.0, 2.0));
                let offset = select(0.0, tile_w * 0.5, (row % 2u) == 1u);
                let col = u32(floor((local_x + offset) / (tile_w * 0.5))) % 2u;
                sub_cell = col + row * 2u;
                let cell_y = local_y - f32(row) * row_h;
                let cell_y_01 = clamp(cell_y / max(row_h, 1.0), 0.0, 1.0);
                // Exposed shingle edges catch slightly more light than the tucked bottom.
                pattern_offset = mix(0.015, -0.015, cell_y_01);
            }
            case SUB_TILE_PATTERN_STAGGERED_COLUMNS: {
                // Rotated variant of staggered rows for vertical shingle courses:
                // 3 cols x 2 rows, odd columns shifted by half tile height.
                let col_w = tile_w / 3.0;
                let col = u32(clamp(floor(local_x / col_w), 0.0, 2.0));
                let offset = select(0.0, tile_h * 0.5, (col % 2u) == 1u);
                let row = u32(floor((local_y + offset) / (tile_h * 0.5))) % 2u;
                sub_cell = row + col * 2u;
                let cell_x = local_x - f32(col) * col_w;
                let cell_x_01 = clamp(cell_x / max(col_w, 1.0), 0.0, 1.0);
                // Exposed shingle edges catch slightly more light than tucked edges.
                pattern_offset = mix(0.015, -0.015, cell_x_01);
            }
            case SUB_TILE_PATTERN_DIAGONAL_BANDS: {
                // Diagonal bands along x + y
                let band = u32(floor((local_x + local_y) / (tile_w * 0.4)));
                sub_cell = band % 6u;
            }
            case SUB_TILE_PATTERN_HORIZONTAL_RIBS: {
                // Horizontal ribs for corrugated metal. Uses world pixel Y so
                // bands stay continuous across tile boundaries.
                let world_px_y = f32(world_tile_y) * tile_h + local_y;
                // Slightly off-integer band count avoids exact pixel alignment
                // that can create stronger moire/aliasing artifacts at some zooms.
                let ribs_per_tile = 4.9;
                let band_period_px = max(tile_h / ribs_per_tile, 1.0);
                let rib_phase = (world_px_y / band_period_px) * 6.28318530718;
                let rib_wave = sin(rib_phase);
                // Sharpen the profile slightly so ridges read as manufactured ribs.
                let rib_profile = sign(rib_wave) * pow(abs(rib_wave), 0.65);
                let rib_offset = rib_profile * noise_amplitude;
                color = vec4<f32>(
                    clamp(color.r + rib_offset, 0.0, 1.0),
                    clamp(color.g + rib_offset, 0.0, 1.0),
                    clamp(color.b + rib_offset, 0.0, 1.0),
                    color.a,
                );
                use_tile_hash = false;
            }
            case SUB_TILE_PATTERN_VERTICAL_RIBS: {
                // Vertical ribs for corrugated metal. Uses world pixel X so
                // bands stay continuous across tile boundaries.
                let world_px_x = f32(world_tile_x) * tile_w + local_x;
                // Keep the same slight off-integer frequency as horizontal ribs
                // to reduce alignment artifacts.
                let ribs_per_tile = 4.9;
                let band_period_px = max(tile_w / ribs_per_tile, 1.0);
                let rib_phase = (world_px_x / band_period_px) * 6.28318530718;
                let rib_wave = sin(rib_phase);
                let rib_profile = sign(rib_wave) * pow(abs(rib_wave), 0.65);
                let rib_offset = rib_profile * noise_amplitude;
                color = vec4<f32>(
                    clamp(color.r + rib_offset, 0.0, 1.0),
                    clamp(color.g + rib_offset, 0.0, 1.0),
                    clamp(color.b + rib_offset, 0.0, 1.0),
                    color.a,
                );
                use_tile_hash = false;
            }
            default: {
                // Fallback to 2x2 blocks.
                let sub_x = u32(floor(local_x / (tile_w * 0.5)));
                let sub_y = u32(floor(local_y / (tile_h * 0.5)));
                sub_cell = sub_x + sub_y * 2u;
            }
        }

        // Tile-based hash for patterns that use sub_cell. Patterns with custom
        // world-space logic disable this via use_tile_hash.
        if (use_tile_hash) {
            let hash_input = world_tile_x ^ (world_tile_y * 2654435761u) ^ (sub_cell * 2246822519u) ^ uniforms.u_noise_seed;
            let h = pcg_hash(hash_input);

            // Convert hash to signed brightness offset in [-amplitude, +amplitude]
            let noise_01 = f32(h & 0xFFFFu) / 65535.0;
            var offset = (noise_01 * 2.0 - 1.0) * noise_amplitude + pattern_offset;

            color = vec4<f32>(
                clamp(color.r + offset, 0.0, 1.0),
                clamp(color.g + offset, 0.0, 1.0),
                clamp(color.b + offset, 0.0, 1.0),
                color.a,
            );
        }
    }

    return color;
}

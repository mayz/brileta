// Rain overlay shader.
// Generates stochastic drops across the full viewport and only excludes the
// player-building interior mask supplied by the CPU.

struct VertexInput {
    @location(0) in_vert: vec2<f32>,
    @location(1) in_uv: vec2<f32>,
    @location(2) in_color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_uv: vec2<f32>,
}

struct RainUniforms {
    // vec4 0: letterbox (offset_x, offset_y, scaled_w, scaled_h)
    letterbox: vec4<f32>,
    // vec4 1: viewport (offset_x, offset_y, size_x, size_y)
    viewport_data: vec4<f32>,
    // vec4 2: intensity, angle, drop_length, drop_speed
    rain_params: vec4<f32>,
    // vec4 3: color_r, color_g, color_b, time
    anim_data: vec4<f32>,
    // vec4 4: tile_w_px, tile_h_px, drop_spacing, stream_spacing
    spacing_data: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: RainUniforms;
@group(0) @binding(1) var rain_exclusion_map: texture_2d<f32>;
@group(0) @binding(2) var texture_sampler: sampler;

fn hash_u32(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
}

fn hash_to_unit(seed: u32) -> f32 {
    return f32(hash_u32(seed)) / 4294967295.0;
}

fn hash_cell(cell_x: i32, cell_y: i32, salt: u32) -> u32 {
    let x = bitcast<u32>(cell_x);
    let y = bitcast<u32>(cell_y);
    return hash_u32((x * 0x1f123bb5u) ^ (y * 0x5f356495u) ^ salt);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let letterbox_x = uniforms.letterbox.x;
    let letterbox_y = uniforms.letterbox.y;
    let letterbox_w = uniforms.letterbox.z;
    let letterbox_h = uniforms.letterbox.w;

    let norm_x = (input.in_vert.x - letterbox_x) / letterbox_w;
    let norm_y = 1.0 - ((input.in_vert.y - letterbox_y) / letterbox_h);

    let x = norm_x * 2.0 - 1.0;
    let y = norm_y * 2.0 - 1.0;

    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.screen_uv = input.in_uv;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let viewport_offset = uniforms.viewport_data.xy;
    let viewport_size = max(uniforms.viewport_data.zw, vec2<f32>(1.0, 1.0));
    let intensity = clamp(uniforms.rain_params.x, 0.0, 1.0);
    let angle = uniforms.rain_params.y;
    let base_drop_length = max(uniforms.rain_params.z, 0.05);
    let base_drop_speed = max(uniforms.rain_params.w, 0.01);
    let rain_color = uniforms.anim_data.xyz / 255.0;
    let time = uniforms.anim_data.w;
    let tile_size_px = max(uniforms.spacing_data.xy, vec2<f32>(1.0, 1.0));
    let base_drop_spacing = max(uniforms.spacing_data.z, 0.12);
    let base_density_spacing = max(uniforms.spacing_data.w, 0.01);

    if (intensity <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let excluded = textureSampleLevel(
        rain_exclusion_map,
        texture_sampler,
        input.screen_uv,
        0.0,
    ).r;
    if (excluded > 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Stochastic overlay space in tile units.
    // Use a world-aligned spawn lattice so gust angle changes do not rotate
    // the entire random field. Angle only steers streak orientation + advection.
    let world_pos = viewport_offset + (input.screen_uv * viewport_size);
    let slant_dir_raw = vec2<f32>(sin(angle), cos(angle));
    let slant_dir = normalize(select(vec2<f32>(0.0, 1.0), slant_dir_raw, length(slant_dir_raw) > 0.0001));
    let perp_dir = vec2<f32>(-slant_dir.y, slant_dir.x);

    // World-space lattice for stable random seeds.
    // Query cells in the "spawn space" (undo advection) so we inspect
    // neighbors that can actually land near this pixel at current time.
    let advection = slant_dir * (time * base_drop_speed);
    let spawn_space_pos = world_pos - advection;
    let base_cell_x = i32(floor(spawn_space_pos.x / base_density_spacing));
    let base_cell_y = i32(floor(spawn_space_pos.y / base_drop_spacing));

    let line_width_px = mix(0.9, 1.6, intensity);
    var max_drop_alpha = 0.0;

    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            let cell_x = base_cell_x + dx;
            let cell_y = base_cell_y + dy;
            let seed = hash_cell(cell_x, cell_y, 0x9e3779b9u);

            // Global spawn occupancy for stochastic drops.
            // A high occupancy quickly turns rain into an opaque sheet even
            // with moderate spacing values, so keep this deliberately low.
            let occupancy = hash_to_unit(seed ^ 0x85ebca6bu);
            if (occupancy > 0.09) {
                continue;
            }

            let length_jitter = mix(0.7, 1.35, hash_to_unit(seed ^ 0x27d4eb2fu));
            let lateral_jitter = (hash_to_unit(seed ^ 0x165667b1u) - 0.5) * base_density_spacing * 0.95;
            let along_jitter = (hash_to_unit(seed ^ 0xd3a2646cu) - 0.5) * base_drop_spacing * 0.95;

            let spawn_x = (f32(cell_x) + 0.5) * base_density_spacing + lateral_jitter;
            let spawn_y = (f32(cell_y) + 0.5) * base_drop_spacing + along_jitter;
            let center_pos = vec2<f32>(spawn_x, spawn_y) + advection;
            let delta_pos = world_pos - center_pos;

            let perp_delta = dot(delta_pos, perp_dir);
            let delta_world = perp_dir * perp_delta;
            let delta_px = vec2<f32>(delta_world.x * tile_size_px.x, delta_world.y * tile_size_px.y);
            let perp_distance_px = length(delta_px);
            let line_mask = 1.0 - smoothstep(line_width_px, line_width_px + 1.0, perp_distance_px);
            if (line_mask <= 0.0) {
                continue;
            }

            let drop_length = max(base_drop_length * length_jitter, 0.05);
            let along_distance = dot(delta_pos, slant_dir);
            if (along_distance < 0.0 || along_distance > drop_length) {
                continue;
            }

            let edge_softness = 0.1;
            let head_fade = smoothstep(0.0, edge_softness, along_distance);
            let tail_fade = 1.0 - smoothstep(
                drop_length - edge_softness,
                drop_length + edge_softness,
                along_distance,
            );
            let drop_alpha = clamp(line_mask * head_fade * tail_fade, 0.0, 1.0);
            max_drop_alpha = max(max_drop_alpha, drop_alpha);
        }
    }

    let alpha = clamp(max_drop_alpha * intensity, 0.0, 1.0);
    if (alpha <= 0.001) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    return vec4<f32>(rain_color, alpha);
}

// Glyph rendering shader with foreground/background color mixing
// and optional per-pixel sub-tile brightness noise.
//
// The noise effect uses a PCG hash to create deterministic brightness
// variation within a 2x2 sub-cell grid per tile, making terrain like
// cobblestone look like individual fitted stones instead of flat color.

// Vertex input structure
struct VertexInput {
    @location(0) in_vert: vec2<f32>,
    @location(1) in_uv: vec2<f32>,
    @location(2) in_fg_color: vec4<f32>,
    @location(3) in_bg_color: vec4<f32>,
    @location(4) in_noise_amplitude: f32,
}

// Vertex output / Fragment input structure
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) v_uv: vec2<f32>,
    @location(1) v_fg_color: vec4<f32>,
    @location(2) v_bg_color: vec4<f32>,
    @location(3) v_noise_amplitude: f32,
    @location(4) v_pixel_pos: vec2<f32>,
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

// PCG hash - fast, high-quality 32-bit hash for spatial noise.
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Vertex shader stage
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    output.v_uv = input.in_uv;
    output.v_fg_color = input.in_fg_color;
    output.v_bg_color = input.in_bg_color;
    output.v_noise_amplitude = input.in_noise_amplitude;
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

    // Mix foreground and background colors based on texture alpha.
    // If char_alpha is 1.0 (opaque pixel), the result is v_fg_color.
    // If char_alpha is 0.0 (transparent pixel), the result is v_bg_color.
    var color = mix(input.v_bg_color, input.v_fg_color, char_alpha);

    // Apply sub-tile brightness noise when amplitude is non-zero.
    // Divides each tile into a 2x2 sub-cell grid and hashes (world_tile_x,
    // world_tile_y, sub_cell, seed) to produce a signed brightness offset.
    // Uses world-space tile coordinates (buffer tile + offset uniform) so the
    // noise pattern stays anchored to world tiles regardless of camera scrolling.
    if (input.v_noise_amplitude > 0.0) {
        // Tile dimensions: width and height may differ (non-square character cells)
        let tile_w = uniforms.u_tile_size.x;
        let tile_h = uniforms.u_tile_size.y;
        // Buffer-space tile index from pixel position
        let buf_tile_x = i32(floor(input.v_pixel_pos.x / tile_w));
        let buf_tile_y = i32(floor(input.v_pixel_pos.y / tile_h));
        // Convert to world-space tile coordinates for stable hashing
        let world_tile_x = u32(buf_tile_x + uniforms.u_tile_offset.x);
        let world_tile_y = u32(buf_tile_y + uniforms.u_tile_offset.y);
        // Which 2x2 sub-cell within the tile (0-3)
        let local_x = input.v_pixel_pos.x - f32(buf_tile_x) * tile_w;
        let local_y = input.v_pixel_pos.y - f32(buf_tile_y) * tile_h;
        let sub_x = u32(floor(local_x / (tile_w * 0.5)));
        let sub_y = u32(floor(local_y / (tile_h * 0.5)));
        let sub_cell = sub_x + sub_y * 2u;

        // Hash world tile position, sub-cell index, and seed for deterministic noise
        let hash_input = world_tile_x ^ (world_tile_y * 2654435761u) ^ (sub_cell * 2246822519u) ^ uniforms.u_noise_seed;
        let h = pcg_hash(hash_input);

        // Convert hash to signed brightness offset in [-amplitude, +amplitude]
        let noise_01 = f32(h & 0xFFFFu) / 65535.0;
        let offset = (noise_01 * 2.0 - 1.0) * input.v_noise_amplitude;

        // Apply brightness offset to RGB, preserving alpha
        color = vec4<f32>(
            clamp(color.r + offset, 0.0, 1.0),
            clamp(color.g + offset, 0.0, 1.0),
            clamp(color.b + offset, 0.0, 1.0),
            color.a,
        );
    }

    return color;
}

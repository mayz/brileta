// Compose dark + light tile textures using the GPU lightmap and fog-of-war masks.
// This keeps overlay blending on-GPU and avoids full lightmap readback.

struct VertexInput {
    @location(0) position: vec2f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
}

struct ComposeUniforms {
    // viewport_width, viewport_height, viewport_offset_x, viewport_offset_y
    viewport_and_offsets: vec4f,
    // tile_width_px, tile_height_px, pad_tiles, spillover_multiplier
    tile_and_compose_params: vec4f,
}

@group(0) @binding(0) var<uniform> uniforms: ComposeUniforms;
@group(0) @binding(1) var dark_texture: texture_2d<f32>;
@group(0) @binding(2) var light_texture: texture_2d<f32>;
@group(0) @binding(3) var lightmap_texture: texture_2d<f32>;
@group(0) @binding(4) var visible_mask: texture_2d<f32>;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = vec4f(input.position, 0.0, 1.0);
    return output;
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let pixel = vec2i(floor(frag_coord.xy));
    let dark_dims = vec2i(textureDimensions(dark_texture));
    if (pixel.x < 0 || pixel.y < 0 || pixel.x >= dark_dims.x || pixel.y >= dark_dims.y) {
        return vec4f(0.0);
    }

    let tile_w = max(1, i32(uniforms.tile_and_compose_params.x));
    let tile_h = max(1, i32(uniforms.tile_and_compose_params.y));
    let pad_tiles = i32(uniforms.tile_and_compose_params.z);
    let spillover_multiplier = uniforms.tile_and_compose_params.w;

    let viewport_offsets = vec2i(uniforms.viewport_and_offsets.zw);
    let lightmap_size = vec2i(textureDimensions(lightmap_texture));

    // Convert output pixel -> padded GlyphBuffer tile index.
    let buffer_tile = vec2i(pixel.x / tile_w, pixel.y / tile_h);
    // Match CPU mapping onto viewport-local lightmap coordinates.
    let local_tile = buffer_tile - viewport_offsets - vec2i(pad_tiles);
    if (local_tile.x < 0 || local_tile.y < 0 ||
        local_tile.x >= lightmap_size.x || local_tile.y >= lightmap_size.y) {
        return vec4f(0.0);
    }

    let dark_pixel = textureLoad(dark_texture, pixel, 0);
    let light_pixel = textureLoad(light_texture, pixel, 0);
    let explored_alpha = max(dark_pixel.a, light_pixel.a);
    // CPU path only writes explored tiles into the overlay buffer. Preserve that
    // behavior here by leaving unexplored pixels fully transparent.
    if (explored_alpha <= 0.001) {
        return vec4f(0.0);
    }

    var light_rgb = clamp(textureLoad(lightmap_texture, local_tile, 0).rgb, vec3f(0.0), vec3f(1.0));

    let visible = textureLoad(visible_mask, buffer_tile, 0).r;
    if (visible < 0.5) {
        light_rgb *= spillover_multiplier;
    }

    // Blend in pixel space. Because glyph rendering is linear, this matches
    // per-tile fg/bg blending done by the CPU path.
    let composed_rgb = light_pixel.rgb * light_rgb + dark_pixel.rgb * (vec3f(1.0) - light_rgb);

    return vec4f(composed_rgb, explored_alpha);
}

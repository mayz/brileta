// Screen renderer shader - handles letterboxing and coordinate transformation
// Corrected version with proper offset subtraction and Y-axis flip.

// Vertex input structure
struct VertexInput {
    @location(0) in_vert: vec2<f32>,   // Input vertex position in PIXELS
    @location(1) in_uv: vec2<f32>,
    @location(2) in_color: vec4<f32>,
    @location(3) in_world_pos: vec2<f32>,
    @location(4) in_actor_light_scale: f32,
    @location(5) in_flags: u32,
    @location(6) in_tile_bg: vec3<f32>,
}

// Vertex output structure (to fragment shader)
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) v_uv: vec2<f32>,
    @location(1) v_color: vec4<f32>,
    @location(2) v_world_pos: vec2<f32>,
    @location(3) v_actor_light_scale: f32,
    @location(4) v_flags: u32,
    @location(5) v_tile_bg: vec3<f32>,
}

// Uniform buffer for letterbox parameters
struct Uniforms {
    u_letterbox: vec4<f32>,   // (offset_x, offset_y, scaled_w, scaled_h)
    // (viewport_world_x, viewport_world_y, actor_lighting_enabled, unused)
    u_actor_light_data: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var u_texture: texture_2d<f32>;
@group(0) @binding(2) var u_sampler: sampler;
@group(0) @binding(3) var u_actor_lightmap: texture_2d<f32>;

// Vertex shader stage
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let letterbox_x = uniforms.u_letterbox.x;
    let letterbox_y = uniforms.u_letterbox.y;
    let letterbox_w = uniforms.u_letterbox.z;
    let letterbox_h = uniforms.u_letterbox.w;

    // Normalize to letterbox coordinates (0.0 to 1.0)
    let norm_x = (input.in_vert.x - letterbox_x) / letterbox_w;
    let norm_y = 1.0 - ((input.in_vert.y - letterbox_y) / letterbox_h);

    // Convert to clip space (-1.0 to 1.0)
    let x = norm_x * 2.0 - 1.0;
    let y = norm_y * 2.0 - 1.0;

    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.v_uv = input.in_uv;
    output.v_color = input.in_color;
    output.v_world_pos = input.in_world_pos;
    output.v_actor_light_scale = input.in_actor_light_scale;
    output.v_flags = input.in_flags;
    output.v_tile_bg = input.in_tile_bg;

    return output;
}

// Color temperature adjustment to match ModernGL's warmer appearance
fn warm_color_correction(color: vec3<f32>) -> vec3<f32> {
    // Subtle warm adjustment: slightly boost red/orange, reduce blue
    // These values are tuned to match ModernGL's implicit gamma warmth
    return vec3<f32>(
        color.r * 1.15,  // Slightly boost red
        color.g * 1.05,  // Slightly boost green
        color.b * 0.95   // Slightly reduce blue
    );
}

const ACTOR_LIGHTING_FLAG: u32 = 1u;
const CONTRAST_TILE_BG_SENTINEL_EPS: f32 = 0.001;
const CONTRAST_LIT_DIST_MIN: f32 = 0.15;
const CONTRAST_LIT_DIST_MAX: f32 = 0.21;
const CONTRAST_BASE_DIST_MIN: f32 = 0.20;
const CONTRAST_BASE_DIST_MAX: f32 = 0.26;
const CONTRAST_TARGET_LUM_DELTA: f32 = 0.14;
const CONTRAST_SAFE_FG_LUM_FLOOR: f32 = 0.01;
const CONTRAST_BRIGHTEN_SCALE_MAX: f32 = 2.2;
const CONTRAST_NUDGE_EPS: f32 = 0.001;

// Fragment shader stage
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var tint = input.v_color;
    let actor_lighting_enabled = uniforms.u_actor_light_data.z > 0.5;
    let actor_flag_set = (input.v_flags & ACTOR_LIGHTING_FLAG) != 0u;

    if (actor_lighting_enabled && actor_flag_set) {
        let lightmap_dims = vec2i(textureDimensions(u_actor_lightmap));
        let viewport_origin = uniforms.u_actor_light_data.xy;
        let local_tile = vec2i(round(input.v_world_pos - viewport_origin));

        var light_rgb = vec3f(1.0);
        if (local_tile.x >= 0 && local_tile.y >= 0 &&
            local_tile.x < lightmap_dims.x && local_tile.y < lightmap_dims.y) {
            light_rgb = clamp(textureLoad(u_actor_lightmap, local_tile, 0).rgb, vec3f(0.0), vec3f(1.0));
        }

        light_rgb *= input.v_actor_light_scale;
        let base_rgb = input.v_color.rgb;
        var actor_rgb = min(
            vec3f(1.0),
            base_rgb * light_rgb * vec3f(0.7) + light_rgb * vec3f(0.3),
        );

        // Contrast nudge goals:
        // 1) improve readability when lit actor and lit tile are too similar,
        // 2) avoid frame-to-frame popping under flicker (soft gates),
        // 3) protect distinct color pairs (e.g., gold on warm brown) from muddying.
        //
        // Note: tile_bg comes from light_appearance_map base bg and is re-lit here.
        // This is a first-order approximation of visible tile bg, not a direct sample
        // of the final rendered scene color behind the actor.
        //
        // tile_bg > 0 means the caller provided bg data (zero = sentinel, skip check).
        let has_tile_bg = any(input.v_tile_bg > vec3f(CONTRAST_TILE_BG_SENTINEL_EPS));
        if (has_tile_bg) {
            // Approximate the lit tile bg using the same lightmap lighting.
            let lit_bg = min(
                vec3f(1.0),
                input.v_tile_bg * light_rgb * vec3f(0.7) + light_rgb * vec3f(0.3),
            );
            let lit_delta = actor_rgb - lit_bg;
            let lit_rgb_dist = sqrt(dot(lit_delta, lit_delta));
            let base_delta = input.v_color.rgb - input.v_tile_bg;
            let base_rgb_dist = sqrt(dot(base_delta, base_delta));

            // Use a soft activation band to avoid flickering when firelight makes
            // the color distance hover around a hard threshold.
            let bg_lum = dot(lit_bg, vec3f(0.2126, 0.7152, 0.0722));
            let fg_lum = dot(actor_rgb, vec3f(0.2126, 0.7152, 0.0722));
            let brighten = fg_lum <= bg_lum;

            // Require base-color similarity only for darkening so distinct pairs
            // (like gold glyph on brown floor) keep their original color, while
            // still allowing low-contrast dark glyphs to brighten.
            let lit_similarity = 1.0 - smoothstep(
                CONTRAST_LIT_DIST_MIN,
                CONTRAST_LIT_DIST_MAX,
                lit_rgb_dist,
            );
            let base_similarity = 1.0 - smoothstep(
                CONTRAST_BASE_DIST_MIN,
                CONTRAST_BASE_DIST_MAX,
                base_rgb_dist,
            );
            // Guard darkening with base-color similarity.
            // Brightening remains available so dark glyphs on mid-gray tiles
            // don't disappear (e.g. "T" in low-light interiors).
            let direction_gate = select(base_similarity, 1.0, brighten);
            let nudge_strength = lit_similarity * direction_gate;
            if (nudge_strength > CONTRAST_NUDGE_EPS) {
                // Preserve hue by scaling RGB toward target luminance instead of
                // blending to white/black, which can wash chroma to gray.
                let desired_lum = select(
                    bg_lum - CONTRAST_TARGET_LUM_DELTA,
                    bg_lum + CONTRAST_TARGET_LUM_DELTA,
                    brighten,
                );
                let safe_fg_lum = max(fg_lum, CONTRAST_SAFE_FG_LUM_FLOOR);
                let scale_raw = desired_lum / safe_fg_lum;
                let scale = select(
                    clamp(scale_raw, 0.0, 1.0),
                    clamp(scale_raw, 1.0, CONTRAST_BRIGHTEN_SCALE_MAX),
                    brighten,
                );
                let target_rgb = min(vec3f(1.0), actor_rgb * scale);
                actor_rgb = mix(actor_rgb, target_rgb, nudge_strength);
            }
        }

        tint = vec4f(actor_rgb, input.v_color.a);
    }

    let sampled_color = textureSample(u_texture, u_sampler, input.v_uv) * tint;
    let corrected_rgb = warm_color_correction(sampled_color.rgb);
    return vec4<f32>(corrected_rgb, sampled_color.a);
}

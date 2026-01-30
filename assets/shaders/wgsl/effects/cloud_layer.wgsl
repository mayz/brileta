// Atmospheric layer shader for cloud shadows and ground mist effects.
// Renders as a full-screen overlay with noise-based patterns masked by sky exposure.

// Vertex input structure - using pixel coordinates like the screen renderer
struct VertexInput {
    @location(0) in_vert: vec2<f32>,   // Vertex position in pixels
    @location(1) in_uv: vec2<f32>,     // UV coordinates
    @location(2) in_color: vec4<f32>,  // Vertex color (unused but required for layout)
}

// Vertex output structure
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_uv: vec2<f32>,
}

// Uniform buffer for atmospheric layer parameters
// Using vec4 packing for reliable alignment across all backends
struct AtmosphericUniforms {
    // vec4 0: Letterbox parameters (offset_x, offset_y, scaled_w, scaled_h)
    letterbox: vec4<f32>,

    // vec4 1: Viewport offset and size (offset_x, offset_y, size_x, size_y) as floats
    viewport_data: vec4<f32>,

    // vec4 2: Map size and blend params (map_w, map_h, blend_mode, strength)
    map_and_blend: vec4<f32>,

    // vec4 3: Tint color and noise scale (r, g, b, noise_scale)
    tint_and_noise: vec4<f32>,

    // vec4 4: Noise thresholds and drift (threshold_low, threshold_high, drift_x, drift_y)
    thresholds_and_drift: vec4<f32>,

    // vec4 5: Turbulence params (turbulence_offset, turbulence_strength, turbulence_scale, sky_threshold)
    turbulence_and_sky: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: AtmosphericUniforms;
@group(0) @binding(1) var sky_exposure_map: texture_2d<f32>;
@group(0) @binding(2) var explored_map: texture_2d<f32>;
@group(0) @binding(3) var visible_map: texture_2d<f32>;
@group(0) @binding(4) var noise_texture: texture_2d<f32>;
@group(0) @binding(5) var texture_sampler: sampler;

// Vertex shader - transforms pixel coordinates to clip space with letterbox handling
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let letterbox_x = uniforms.letterbox.x;
    let letterbox_y = uniforms.letterbox.y;
    let letterbox_w = uniforms.letterbox.z;
    let letterbox_h = uniforms.letterbox.w;

    // Normalize to letterbox coordinates (0.0 to 1.0)
    let norm_x = (input.in_vert.x - letterbox_x) / letterbox_w;
    let norm_y = 1.0 - ((input.in_vert.y - letterbox_y) / letterbox_h);

    // Convert to clip space (-1.0 to 1.0)
    let x = norm_x * 2.0 - 1.0;
    let y = norm_y * 2.0 - 1.0;

    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.screen_uv = input.in_uv;

    return output;
}

// Fragment shader - applies atmospheric effect based on sky exposure mask
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Extract values from packed uniforms
    let viewport_offset = uniforms.viewport_data.xy;
    let viewport_size = uniforms.viewport_data.zw;
    let map_size = uniforms.map_and_blend.xy;
    let blend_mode = u32(uniforms.map_and_blend.z);
    let strength = uniforms.map_and_blend.w;
    let tint_color = uniforms.tint_and_noise.xyz;
    let noise_scale = uniforms.tint_and_noise.w;
    let noise_threshold_low = uniforms.thresholds_and_drift.x;
    let noise_threshold_high = uniforms.thresholds_and_drift.y;
    let drift_offset = uniforms.thresholds_and_drift.zw;
    let turbulence_offset = uniforms.turbulence_and_sky.x;
    let turbulence_strength = uniforms.turbulence_and_sky.y;
    let turbulence_scale = uniforms.turbulence_and_sky.z;
    let sky_exposure_threshold = uniforms.turbulence_and_sky.w;

    // Convert screen UV to world tile position (matching lighting system)
    let world_pos = viewport_offset + input.screen_uv * viewport_size;
    let tile_pos = vec2<i32>(floor(world_pos));

    // Clamp tile position to map bounds for texture sampling
    let map_size_i = vec2<i32>(map_size);
    let clamped_pos = clamp(tile_pos, vec2<i32>(0), map_size_i - vec2<i32>(1));

    // Sample explored/visible masks using integer coordinates (texelFetch equivalent)
    let explored = textureLoad(explored_map, clamped_pos, 0).r;
    let visible = textureLoad(visible_map, clamped_pos, 0).r;
    let sky_exposure = textureLoad(sky_exposure_map, clamped_pos, 0).r;

    // Early out for unexplored or not visible tiles
    if (explored < 0.5 || visible < 0.5) {
        if (blend_mode == 0u) {
            return vec4<f32>(1.0, 1.0, 1.0, 1.0);  // White - no effect with multiply
        } else {
            return vec4<f32>(0.0, 0.0, 0.0, 0.0);  // Transparent - no effect with additive
        }
    }

    // Early out for indoor areas (below sky exposure threshold)
    if (sky_exposure < sky_exposure_threshold) {
        if (blend_mode == 0u) {
            return vec4<f32>(1.0, 1.0, 1.0, 1.0);  // White - no effect with multiply
        } else {
            return vec4<f32>(0.0, 0.0, 0.0, 0.0);  // Transparent - no effect with additive
        }
    }

    // Sample noise with optional turbulence distortion
    var noise_uv = world_pos * noise_scale - drift_offset;
    if (turbulence_strength > 0.0) {
        let turbulence_uv = world_pos * turbulence_scale + turbulence_offset;
        let turbulence_sample = textureSample(noise_texture, texture_sampler, turbulence_uv).rg;
        let turbulence_offset_val = (turbulence_sample - 0.5) * 2.0 * turbulence_strength;
        noise_uv = noise_uv + turbulence_offset_val * 0.1;
    }

    var noise_value = textureSample(noise_texture, texture_sampler, noise_uv).r;

    // Apply smoothstep threshold for sharper cloud edges
    noise_value = smoothstep(noise_threshold_low, noise_threshold_high, noise_value);

    // Calculate effect intensity
    let intensity = noise_value * strength;

    // Normalize tint color from 0-255 to 0-1
    let tint_normalized = tint_color / 255.0;

    // Apply blend mode
    if (blend_mode == 0u) {
        // Darken mode (cloud shadows): output multiplicative factor
        // intensity=0 -> output white (1,1,1) = no darkening
        // intensity=1 -> output tint color = full shadow
        let result_color = mix(vec3<f32>(1.0), tint_normalized, intensity);
        return vec4<f32>(result_color, 1.0);
    } else {
        // Lighten mode (mist): output color with alpha for blending
        // Alpha blend uses intensity as coverage; keep color unscaled to avoid
        // double attenuation.
        let result_color = tint_normalized;
        return vec4<f32>(result_color, intensity);
    }
}

// UI texture rendering shader with letterbox transformation
// Combines vertex and fragment stages in a single WGSL file

// Vertex input structure
struct VertexInput {
    @location(0) in_vert: vec2<f32>,
    @location(1) in_uv: vec2<f32>,
    @location(2) in_color: vec4<f32>,
}

// Vertex output / Fragment input structure
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) v_uv: vec2<f32>,
    @location(1) v_color: vec4<f32>,
}

// Uniform buffer for letterbox parameters
struct Uniforms {
    u_letterbox: vec4<f32>,   // (offset_x, offset_y, scaled_w, scaled_h)
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var u_texture: texture_2d<f32>;
@group(0) @binding(2) var u_sampler: sampler;

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

    return output;
}

// Color temperature adjustment to match ModernGL's warmer appearance
// TODO: This function is duplicated from screen/main.wgsl. Once we implement
// shader preprocessing with includes, move this to a common/color_utils.wgsl file
fn warm_color_correction(color: vec3<f32>) -> vec3<f32> {
    // Subtle warm adjustment: slightly boost red/orange, reduce blue
    // These values are tuned to match ModernGL's implicit gamma warmth
    return vec3<f32>(
        color.r * 1.15,  // Slightly boost red
        color.g * 1.05,  // Slightly boost green
        color.b * 0.95   // Slightly reduce blue
    );
}

// Fragment shader stage
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let sampled_color = textureSample(u_texture, u_sampler, input.v_uv) * input.v_color;
    let corrected_rgb = warm_color_correction(sampled_color.rgb);
    return vec4<f32>(corrected_rgb, sampled_color.a);
}
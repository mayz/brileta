// Screen renderer shader - handles letterboxing and coordinate transformation
// Combines vertex and fragment stages in a single WGSL file

// Vertex input structure
struct VertexInput {
    @location(0) in_vert: vec2<f32>,   // Input vertex position in PIXELS
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
@group(0) @binding(1) var u_atlas: texture_2d<f32>;
@group(0) @binding(2) var u_sampler: sampler;

// Vertex shader stage
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    output.v_uv = input.in_uv;
    output.v_color = input.in_color;
    
    // Adjust coordinates for letterboxing offset
    let adjusted_pos = input.in_vert - uniforms.u_letterbox.xy;
    
    // Normalize to letterbox space, then to clip space
    let x = (adjusted_pos.x / uniforms.u_letterbox.z) * 2.0 - 1.0;
    let y = (1.0 - (adjusted_pos.y / uniforms.u_letterbox.w)) * 2.0 - 1.0;
    
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    
    return output;
}

// Fragment shader stage
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(u_atlas, u_sampler, input.v_uv);
    return tex_color * input.v_color;
}
// Glyph rendering shader with foreground/background color mixing
// Combines vertex and fragment stages in a single WGSL file

// Vertex input structure
struct VertexInput {
    @location(0) in_vert: vec2<f32>,
    @location(1) in_uv: vec2<f32>,
    @location(2) in_fg_color: vec4<f32>,
    @location(3) in_bg_color: vec4<f32>,
}

// Vertex output / Fragment input structure
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) v_uv: vec2<f32>,
    @location(1) v_fg_color: vec4<f32>,
    @location(2) v_bg_color: vec4<f32>,
}

// Uniform buffer for texture size
struct Uniforms {
    u_texture_size: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var u_atlas: texture_2d<f32>;
@group(0) @binding(2) var u_sampler: sampler;

// Vertex shader stage
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    output.v_uv = input.in_uv;
    output.v_fg_color = input.in_fg_color;
    output.v_bg_color = input.in_bg_color;
    
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
    return mix(input.v_bg_color, input.v_fg_color, char_alpha);
}
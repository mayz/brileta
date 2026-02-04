#version 330
// Full-screen quad vertex shader for fragment-based lighting computation
in vec2 in_position;
out vec2 v_uv;
out vec2 v_world_pos;

uniform ivec2 u_viewport_offset;  // World coordinate offset
uniform ivec2 u_viewport_size;    // Viewport dimensions

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);

    // Convert from [-1,1] to [0,1] UV coordinates
    v_uv = in_position * 0.5 + 0.5;

    // Calculate world position for this fragment
    // UV (0,0) = top-left, UV (1,1) = bottom-right
    vec2 pixel_in_viewport = v_uv * vec2(u_viewport_size);
    v_world_pos = vec2(u_viewport_offset) + pixel_in_viewport;
}
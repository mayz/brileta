#version 330
// Full-screen quad vertex shader for atmospheric overlay.
// Positions are provided in pixel coordinates to match the screen renderer.

in vec2 in_vert;
in vec2 in_uv;
in vec4 in_color;

out vec2 v_screen_uv;
out vec4 v_color;

uniform vec4 u_letterbox; // (offset_x, offset_y, scaled_w, scaled_h)

void main() {
    v_screen_uv = in_uv;
    v_color = in_color;

    // Adjust coordinates for letterboxing offset (same as screen/main.vert).
    vec2 adjusted_pos = in_vert - u_letterbox.xy;
    float x = (adjusted_pos.x / u_letterbox.z) * 2.0 - 1.0;
    float y = (1.0 - (adjusted_pos.y / u_letterbox.w)) * 2.0 - 1.0;

    gl_Position = vec4(x, y, 0.0, 1.0);
}

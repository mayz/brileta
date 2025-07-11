#version 330
// Renders vertices to the screen, normalizing pixel coordinates
// to clip space with letterboxing support.
in vec2 in_vert;       // Input vertex position in PIXELS
in vec2 in_uv;
in vec4 in_color;

out vec2 v_uv;
out vec4 v_color;

uniform vec4 u_letterbox;   // (offset_x, offset_y, scaled_w, scaled_h)

void main() {
    v_uv = in_uv;
    v_color = in_color;

    // Adjust coordinates for letterboxing offset
    vec2 adjusted_pos = in_vert - u_letterbox.xy;

    // Normalize to letterbox space, then to clip space
    float x = (adjusted_pos.x / u_letterbox.z) * 2.0 - 1.0;
    float y = (1.0 - (adjusted_pos.y / u_letterbox.w)) * 2.0 - 1.0;

    gl_Position = vec4(x, y, 0.0, 1.0);
}
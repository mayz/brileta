#version 330
in vec2 in_vert;
in vec2 in_uv;
in vec4 in_color;

uniform vec4 u_letterbox;

out vec2 v_uv;
out vec4 v_color;

void main() {
    float letterbox_x = u_letterbox.x;
    float letterbox_y = u_letterbox.y;
    float letterbox_w = u_letterbox.z;
    float letterbox_h = u_letterbox.w;

    // Normalize to letterbox coordinates (0.0 to 1.0)
    float norm_x = (in_vert.x - letterbox_x) / letterbox_w;
    float norm_y = 1.0 - ((in_vert.y - letterbox_y) / letterbox_h);

    // Convert to clip space (-1.0 to 1.0)
    float x = norm_x * 2.0 - 1.0;
    float y = norm_y * 2.0 - 1.0;

    gl_Position = vec4(x, y, 0.0, 1.0);
    v_uv = in_uv;
    v_color = in_color;
}
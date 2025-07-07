#version 330
in vec2 in_vert;
in vec2 in_uv;
in vec4 in_fg_color;
in vec4 in_bg_color;

out vec2 v_uv;
out vec4 v_fg_color;
out vec4 v_bg_color;

uniform vec2 u_texture_size;

void main() {
    v_uv = in_uv;
    v_fg_color = in_fg_color;
    v_bg_color = in_bg_color;

    float x = (in_vert.x / u_texture_size.x) * 2.0 - 1.0;
    float y = (in_vert.y / u_texture_size.y) * 2.0 - 1.0;
    gl_Position = vec4(x, y, 0.0, 1.0);
}
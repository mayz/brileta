#version 330
in vec2 v_uv;
in vec4 v_color;
out vec4 f_color;
uniform sampler2D u_atlas;
void main() {
    vec4 tex_color = texture(u_atlas, v_uv);
    f_color = tex_color * v_color;
}
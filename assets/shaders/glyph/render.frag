#version 330
uniform sampler2D u_atlas;

in vec2 v_uv;
in vec4 v_fg_color;
in vec4 v_bg_color;

out vec4 f_color;

void main() {
    // Sample the character tile from the atlas
    float char_alpha = texture(u_atlas, v_uv).a;

    // Mix foreground and background colors based on texture alpha.
    // If char_alpha is 1.0 (opaque pixel), the result is v_fg_color.
    // If char_alpha is 0.0 (transparent pixel), the result is v_bg_color.
    f_color = mix(v_bg_color, v_fg_color, char_alpha);
}
#version 330
// Debug shader: apply noise-modulated tint only where sky exposure is above threshold.

in vec2 v_screen_uv;

uniform ivec2 u_viewport_offset;
uniform ivec2 u_viewport_size;
uniform ivec2 u_map_size;
uniform float u_sky_exposure_threshold;
uniform float u_noise_scale;
uniform float u_noise_threshold_low;
uniform float u_noise_threshold_high;
uniform float u_strength;
uniform vec3 u_tint_color;
uniform vec2 u_drift_offset;
uniform float u_turbulence_offset;
uniform float u_turbulence_strength;
uniform float u_turbulence_scale;
uniform int u_blend_mode;
uniform sampler2D u_sky_exposure_map;
uniform sampler2D u_explored_map;
uniform sampler2D u_visible_map;
uniform sampler2D u_noise_texture;

out vec4 frag_color;

void main() {
    // Convert screen UV to world tile position (matching lighting system).
    vec2 world_pos = vec2(u_viewport_offset) + v_screen_uv * vec2(u_viewport_size);
    ivec2 tile_pos = ivec2(floor(world_pos));

    // Sample sky exposure using map-space tile coordinates.
    ivec2 clamped_pos = clamp(tile_pos, ivec2(0), u_map_size - ivec2(1));
    float explored = texelFetch(u_explored_map, clamped_pos, 0).r;
    float visible = texelFetch(u_visible_map, clamped_pos, 0).r;
    float sky_exposure = texelFetch(u_sky_exposure_map, clamped_pos, 0).r;

    if (explored < 0.5 || visible < 0.5) {
        if (u_blend_mode == 0) {
            frag_color = vec4(1.0, 1.0, 1.0, 1.0);
        } else {
            frag_color = vec4(0.0, 0.0, 0.0, 0.0);
        }
        return;
    }

    if (sky_exposure < u_sky_exposure_threshold) {
        if (u_blend_mode == 0) {
            frag_color = vec4(1.0, 1.0, 1.0, 1.0);
        } else {
            frag_color = vec4(0.0, 0.0, 0.0, 0.0);
        }
        return;
    }

    vec2 noise_uv = world_pos * u_noise_scale - u_drift_offset;
    if (u_turbulence_strength > 0.0) {
        vec2 turbulence_uv = world_pos * u_turbulence_scale + u_turbulence_offset;
        vec2 turbulence_sample = texture(u_noise_texture, turbulence_uv).rg;
        vec2 turbulence_offset = (turbulence_sample - 0.5) * 2.0 * u_turbulence_strength;
        noise_uv += turbulence_offset * 0.1;
    }
    float noise_value = texture(u_noise_texture, noise_uv).r;
    noise_value = smoothstep(
        u_noise_threshold_low,
        u_noise_threshold_high,
        noise_value
    );
    float intensity = noise_value * u_strength;
    vec3 tint_normalized = u_tint_color / 255.0;
    if (u_blend_mode == 0) {
        vec3 result_color = mix(vec3(1.0), tint_normalized, intensity);
        frag_color = vec4(result_color, 1.0);
    } else {
        // Alpha blend uses intensity as coverage; keep color unscaled to avoid
        // double attenuation.
        vec3 result_color = tint_normalized;
        frag_color = vec4(result_color, intensity);
    }
}

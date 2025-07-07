#version 330
// Fragment shader-based point light computation for OpenGL 4.1 compatibility
in vec2 v_uv;
in vec2 v_world_pos;

out vec4 f_color;

// Light data as uniforms (simpler than storage buffers for OpenGL 4.1)
// We'll handle multiple lights by multiple render passes
uniform int u_light_count;
uniform float u_light_positions[64];  // xy pairs, up to 32 lights
uniform float u_light_radii[32];
uniform float u_light_intensities[32];
uniform vec3 u_light_colors[32];

// Flicker data
uniform float u_light_flicker_enabled[32];
uniform float u_light_flicker_speed[32];
uniform float u_light_min_brightness[32];
uniform float u_light_max_brightness[32];

// Global uniforms
uniform float u_ambient_light;
uniform float u_time;
uniform bool u_tile_aligned;

// Noise function for flicker effects
float noise2d(vec2 coord) {
    vec2 c = floor(coord);
    vec2 f = fract(coord);

    // Hash function for deterministic noise
    float a = sin(dot(c, vec2(12.9898, 78.233))) * 43758.5453;
    float b = sin(dot(c + vec2(1.0, 0.0), vec2(12.9898, 78.233))) * 43758.5453;
    float c_val = sin(dot(c + vec2(0.0, 1.0), vec2(12.9898, 78.233))) * 43758.5453;
    float d = sin(dot(c + vec2(1.0, 1.0), vec2(12.9898, 78.233))) * 43758.5453;

    a = fract(a); b = fract(b); c_val = fract(c_val); d = fract(d);

    // Smooth interpolation
    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(mix(a, b, u.x), mix(c_val, d, u.x), u.y) * 2.0 - 1.0;  // Range [-1, 1]
}

void main() {
    vec2 world_pos = v_world_pos;
    
    // For tile-aligned mode, sample at tile centers
    if (u_tile_aligned) {
        world_pos = floor(world_pos) + vec2(0.5, 0.5);
    }
    
    // Start with ambient lighting
    vec3 final_color = vec3(u_ambient_light);
    
    // Add contribution from each point light
    for (int i = 0; i < u_light_count && i < 32; i++) {
        vec2 light_pos = vec2(u_light_positions[i * 2], u_light_positions[i * 2 + 1]);
        float light_radius = u_light_radii[i];
        float base_intensity = u_light_intensities[i];
        vec3 light_color = u_light_colors[i];
        
        // Calculate distance from light
        vec2 light_vec = world_pos - light_pos;
        float distance = length(light_vec);
        
        // Skip if outside light radius
        if (distance > light_radius) {
            continue;
        }
        
        // Calculate base attenuation (linear falloff to match CPU)
        float attenuation = max(0.0, 1.0 - (distance / light_radius));
        
        // Apply flicker if enabled
        float intensity = base_intensity;
        if (u_light_flicker_enabled[i] > 0.5) {
            // Create flicker using noise - similar to CPU implementation
            float flicker_noise = noise2d(vec2(u_time * u_light_flicker_speed[i], 0.0));
            // Convert from [-1, 1] to [min_brightness, max_brightness]
            float flicker_multiplier = u_light_min_brightness[i] +
                ((flicker_noise + 1.0) * 0.5 * (u_light_max_brightness[i] - u_light_min_brightness[i]));
            intensity *= flicker_multiplier;
        }
        
        attenuation *= intensity;
        
        // Calculate light contribution
        vec3 light_contribution = light_color * attenuation;
        
        // Use brightest-wins blending to match CPU np.maximum behavior
        final_color = max(final_color, light_contribution);
    }
    
    // Clamp to valid range and write output
    final_color = clamp(final_color, 0.0, 1.0);
    f_color = vec4(final_color, 1.0);
}
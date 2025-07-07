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

// Shadow casting uniforms
uniform int u_shadow_caster_count;
uniform float u_shadow_caster_positions[128]; // xy pairs, up to 64 shadow casters
uniform float u_shadow_intensity;
uniform int u_shadow_max_length;
uniform bool u_shadow_falloff_enabled;

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

// Calculate shadow attenuation from shadow casters matching CPU algorithm
float calculateShadowAttenuation(vec2 world_pos, vec2 light_pos) {
    float shadow_factor = 1.0;  // No shadow by default
    
    // Check each shadow caster
    for (int i = 0; i < u_shadow_caster_count && i < 64; i++) {
        vec2 caster_pos = vec2(u_shadow_caster_positions[i * 2], u_shadow_caster_positions[i * 2 + 1]);
        
        // Calculate displacement from light to caster (CPU algorithm)
        vec2 light_to_caster = caster_pos - light_pos;
        float dx = light_to_caster.x;
        float dy = light_to_caster.y;
        
        // Use Chebyshev distance like CPU (max of absolute values)
        float caster_distance = max(abs(dx), abs(dy));
        if (caster_distance < 0.1) {
            continue;  // Skip if too close
        }
        
        // Calculate shadow direction using step function like CPU
        float shadow_dx = dx > 0.0 ? 1.0 : (dx < 0.0 ? -1.0 : 0.0);
        float shadow_dy = dy > 0.0 ? 1.0 : (dy < 0.0 ? -1.0 : 0.0);
        
        // Check if world_pos is in the shadow path from caster
        float max_shadow_length = float(u_shadow_max_length);
        bool in_shadow = false;
        float shadow_intensity = 0.0;
        
        // Check each shadow position (matching CPU loop)
        for (int j = 1; j <= int(max_shadow_length) && j <= 3; j++) {
            vec2 shadow_pos = caster_pos + vec2(shadow_dx * float(j), shadow_dy * float(j));
            
            // Check if world_pos matches this shadow position (core shadow)
            if (abs(world_pos.x - shadow_pos.x) < 0.1 && abs(world_pos.y - shadow_pos.y) < 0.1) {
                // Calculate falloff like CPU
                float distance_falloff = 1.0;
                if (u_shadow_falloff_enabled) {
                    distance_falloff = 1.0 - (float(j - 1) / max_shadow_length);
                }
                shadow_intensity = max(shadow_intensity, u_shadow_intensity * distance_falloff);
                in_shadow = true;
            }
            
            // Check soft edges for first 2 shadow tiles (like CPU)
            if (j <= 2) {
                float edge_intensity = u_shadow_intensity * 0.4; // 40% like CPU
                if (u_shadow_falloff_enabled) {
                    float distance_falloff = 1.0 - (float(j - 1) / max_shadow_length);
                    edge_intensity *= distance_falloff;
                }
                
                // Check 8 adjacent/diagonal positions around core shadow
                vec2 edge_offsets[8] = vec2[8](
                    vec2(0.0, 1.0), vec2(0.0, -1.0), vec2(1.0, 0.0), vec2(-1.0, 0.0),  // Adjacent
                    vec2(1.0, 1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(-1.0, -1.0)   // Diagonal
                );
                
                for (int k = 0; k < 8; k++) {
                    vec2 edge_pos = shadow_pos + edge_offsets[k];
                    if (abs(world_pos.x - edge_pos.x) < 0.1 && abs(world_pos.y - edge_pos.y) < 0.1) {
                        // Don't add edge if it's in the core shadow direction
                        if (!(abs(edge_offsets[k].x - shadow_dx) < 0.1 && abs(edge_offsets[k].y - shadow_dy) < 0.1)) {
                            shadow_intensity = max(shadow_intensity, edge_intensity);
                            in_shadow = true;
                        }
                    }
                }
            }
        }
        
        if (in_shadow) {
            shadow_factor *= (1.0 - shadow_intensity);
        }
    }
    
    return shadow_factor;
}

void main() {
    vec2 world_pos = v_world_pos;
    
    // For lighting calculations, use integer tile coordinates to match CPU
    // The CPU calculates distance from light to integer tile positions
    vec2 tile_pos = floor(world_pos);
    
    // For tile-aligned mode, we still need to determine which tile we're in
    if (u_tile_aligned) {
        // Use the tile position for distance calculations (matching CPU)
        world_pos = tile_pos;
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
        
        // Apply shadow attenuation using tile position for consistency
        float shadow_attenuation = calculateShadowAttenuation(tile_pos, light_pos);
        light_contribution *= shadow_attenuation;
        
        // Use brightest-wins blending to match CPU np.maximum behavior
        final_color = max(final_color, light_contribution);
    }
    
    // Clamp to valid range and write output
    final_color = clamp(final_color, 0.0, 1.0);
    f_color = vec4(final_color, 1.0);
}
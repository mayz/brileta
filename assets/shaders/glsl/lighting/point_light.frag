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

// Shadow grid texture for terrain shadows (texture unit 3)
uniform sampler2D u_shadow_grid;
uniform float u_shadow_intensity;
uniform int u_shadow_max_length;
uniform bool u_shadow_falloff_enabled;

// Actor shadow uniforms (replaces terrain caster array)
uniform int u_actor_shadow_count;
uniform float u_actor_shadow_positions[128];  // xy pairs, 64 actors max

// Directional light uniforms (sun/moon)
uniform vec2 u_sun_direction;
uniform vec3 u_sun_color;
uniform float u_sun_intensity;
uniform sampler2D u_sky_exposure_map;
uniform float u_sky_exposure_power;
uniform float u_sun_shadow_intensity;  // Separate shadow intensity for sun (outdoor shadows)

// Tile emission uniforms (for glowing tiles like acid pools, hot coals)
// Emission texture: RGB = emission color * intensity, A = light radius
uniform sampler2D u_emission_map;
uniform vec2 u_viewport_size;  // Viewport dimensions
uniform ivec2 u_map_size;  // Full map dimensions for sky exposure UV calculation

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

// Compute point light shadow by marching toward the light and sampling shadow grid
float computePointLightShadow(vec2 tile_pos, vec2 light_pos, float light_radius) {
    vec2 to_light = light_pos - tile_pos;
    float dist_to_light = length(to_light);

    // Early exit if outside light influence
    if (dist_to_light > light_radius + float(u_shadow_max_length)) {
        return 1.0;
    }

    // Use sign-based stepping (matches previous discrete behavior)
    vec2 step_dir;
    step_dir.x = to_light.x > 0.0 ? 1.0 : (to_light.x < 0.0 ? -1.0 : 0.0);
    step_dir.y = to_light.y > 0.0 ? 1.0 : (to_light.y < 0.0 ? -1.0 : 0.0);

    vec2 pos = tile_pos;
    int steps = int(max(abs(to_light.x), abs(to_light.y)));
    steps = min(steps, u_shadow_max_length);

    for (int i = 1; i <= steps; i++) {
        pos += step_dir;

        // Stop if we've reached or passed the light
        vec2 to_light_now = light_pos - pos;
        if (dot(to_light_now, to_light) <= 0.0) {
            break;  // Passed the light, no shadow
        }

        ivec2 texel = ivec2(pos);
        if (texel.x >= 0 && texel.x < u_map_size.x &&
            texel.y >= 0 && texel.y < u_map_size.y) {
            float blocks = texelFetch(u_shadow_grid, texel, 0).r;
            if (blocks > 0.5) {
                // Blocker between pixel and light - calculate shadow with falloff
                float distance_falloff = 1.0;
                if (u_shadow_falloff_enabled) {
                    distance_falloff = 1.0 - (float(i - 1) / float(u_shadow_max_length));
                }
                return 1.0 - u_shadow_intensity * distance_falloff;
            }
        }
    }

    return 1.0;  // No blockers found
}

// Compute actor shadow attenuation (for NPCs that block light)
float computeActorShadow(vec2 tile_pos, vec2 light_pos, float light_radius) {
    float shadow_factor = 1.0;

    for (int i = 0; i < u_actor_shadow_count && i < 64; i++) {
        vec2 actor_pos = vec2(
            u_actor_shadow_positions[i * 2],
            u_actor_shadow_positions[i * 2 + 1]
        );

        // Calculate displacement from light to actor
        vec2 light_to_actor = actor_pos - light_pos;
        float dx = light_to_actor.x;
        float dy = light_to_actor.y;

        // Use Chebyshev distance
        float actor_distance = max(abs(dx), abs(dy));
        if (actor_distance < 0.1) {
            continue;  // Skip if too close
        }

        // Calculate shadow direction using step function
        float shadow_dx = dx > 0.0 ? 1.0 : (dx < 0.0 ? -1.0 : 0.0);
        float shadow_dy = dy > 0.0 ? 1.0 : (dy < 0.0 ? -1.0 : 0.0);

        // Check if tile_pos is in the shadow path from actor
        float max_shadow_length = float(u_shadow_max_length);
        bool in_shadow = false;
        float shadow_intensity_val = 0.0;

        for (int j = 1; j <= int(max_shadow_length) && j <= 3; j++) {
            vec2 shadow_pos = actor_pos + vec2(shadow_dx * float(j), shadow_dy * float(j));

            // Check if tile_pos matches this shadow position
            if (abs(tile_pos.x - shadow_pos.x) < 0.1 && abs(tile_pos.y - shadow_pos.y) < 0.1) {
                float distance_falloff = 1.0;
                if (u_shadow_falloff_enabled) {
                    distance_falloff = 1.0 - (float(j - 1) / max_shadow_length);
                }
                shadow_intensity_val = max(shadow_intensity_val, u_shadow_intensity * distance_falloff);
                in_shadow = true;
            }
        }

        if (in_shadow) {
            shadow_factor *= (1.0 - shadow_intensity_val);
        }
    }

    return shadow_factor;
}

// Calculate emission contribution from nearby light-emitting tiles
vec3 calculateEmissionContribution(vec2 uv, vec2 tile_pos) {
    vec3 emission_total = vec3(0.0);

    // Maximum radius to check (must match max light_radius in tile data)
    const int MAX_EMISSION_RADIUS = 4;

    // Size of one texel in UV space
    vec2 texel_size = 1.0 / u_viewport_size;

    // Sample nearby tiles for emission
    for (int dy = -MAX_EMISSION_RADIUS; dy <= MAX_EMISSION_RADIUS; dy++) {
        for (int dx = -MAX_EMISSION_RADIUS; dx <= MAX_EMISSION_RADIUS; dx++) {
            // Calculate UV of neighboring tile
            vec2 neighbor_uv = uv + vec2(float(dx), float(dy)) * texel_size;

            // Skip if outside texture bounds
            if (neighbor_uv.x < 0.0 || neighbor_uv.x > 1.0 ||
                neighbor_uv.y < 0.0 || neighbor_uv.y > 1.0) {
                continue;
            }

            // Sample emission texture
            vec4 emission = texture(u_emission_map, neighbor_uv);

            // Check if this tile emits light (radius > 0)
            float radius = emission.a;
            if (radius <= 0.0) {
                continue;
            }

            // Calculate distance to emitting tile
            float dist = length(vec2(float(dx), float(dy)));

            // Skip if outside emission radius
            if (dist > radius) {
                continue;
            }

            // Calculate falloff (linear, matching point lights)
            float falloff = max(0.0, 1.0 - dist / radius);

            // Add emission contribution (RGB is pre-multiplied by intensity)
            emission_total += emission.rgb * falloff;
        }
    }

    return emission_total;
}

// Compute directional shadow by marching toward sun and sampling shadow grid
float computeDirectionalShadow(vec2 tile_pos, float sky_exposure) {
    if (sky_exposure <= 0.1 || u_sun_intensity <= 0.0) {
        return 1.0;  // No shadow
    }

    // March toward sun (opposite of shadow cast direction)
    // Use sign-based stepping to match discrete tile behavior
    vec2 step_dir;
    step_dir.x = u_sun_direction.x > 0.0 ? 1.0 : (u_sun_direction.x < 0.0 ? -1.0 : 0.0);
    step_dir.y = u_sun_direction.y > 0.0 ? 1.0 : (u_sun_direction.y < 0.0 ? -1.0 : 0.0);

    vec2 pos = tile_pos;
    float shadow_factor = 1.0;

    for (int i = 1; i <= u_shadow_max_length; i++) {
        pos += step_dir;

        // Bounds check
        ivec2 texel = ivec2(pos);
        if (texel.x < 0 || texel.x >= u_map_size.x ||
            texel.y < 0 || texel.y >= u_map_size.y) {
            break;
        }

        // Use texelFetch for pixel-exact sampling (no filtering)
        float blocks = texelFetch(u_shadow_grid, texel, 0).r;
        if (blocks > 0.5) {
            // Found a blocker - calculate shadow with falloff
            float distance_falloff = 1.0;
            if (u_shadow_falloff_enabled) {
                distance_falloff = 1.0 - (float(i - 1) / float(u_shadow_max_length));
            }
            shadow_factor *= (1.0 - u_sun_shadow_intensity * distance_falloff);
            break;  // First blocker wins for directional light
        }
    }

    return shadow_factor;
}

// Compute actor shadows for directional (sun) light
float computeActorDirectionalShadow(vec2 tile_pos, float sky_exposure) {
    if (sky_exposure <= 0.1 || u_sun_intensity <= 0.0) {
        return 1.0;
    }

    // Shadow direction is opposite of sun direction
    float shadow_dx = u_sun_direction.x > 0.0 ? -1.0 : (u_sun_direction.x < 0.0 ? 1.0 : 0.0);
    float shadow_dy = u_sun_direction.y > 0.0 ? -1.0 : (u_sun_direction.y < 0.0 ? 1.0 : 0.0);

    float shadow_factor = 1.0;

    for (int i = 0; i < u_actor_shadow_count && i < 64; i++) {
        vec2 actor_pos = vec2(
            u_actor_shadow_positions[i * 2],
            u_actor_shadow_positions[i * 2 + 1]
        );

        // Check if tile_pos is in the shadow path from this actor
        float max_shadow_length = float(u_shadow_max_length);
        bool in_shadow = false;
        float shadow_intensity_val = 0.0;

        for (int j = 1; j <= int(max_shadow_length); j++) {
            vec2 shadow_pos = actor_pos + vec2(shadow_dx * float(j), shadow_dy * float(j));

            if (abs(tile_pos.x - shadow_pos.x) < 0.1 && abs(tile_pos.y - shadow_pos.y) < 0.1) {
                float distance_falloff = 1.0;
                if (u_shadow_falloff_enabled) {
                    distance_falloff = 1.0 - (float(j - 1) / max_shadow_length);
                }
                shadow_intensity_val = max(shadow_intensity_val, u_sun_shadow_intensity * distance_falloff);
                in_shadow = true;
            }
        }

        if (in_shadow) {
            shadow_factor *= (1.0 - shadow_intensity_val);
        }
    }

    return shadow_factor;
}

void main() {
    vec2 world_pos = v_world_pos;

    // For lighting calculations, use integer tile coordinates to match CPU
    vec2 tile_pos = floor(world_pos);

    // For tile-aligned mode, use tile position for distance calculations
    if (u_tile_aligned) {
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
        // Combine terrain shadows (from grid texture) with actor shadows
        float terrain_shadow = computePointLightShadow(tile_pos, light_pos, light_radius);
        float actor_shadow = computeActorShadow(tile_pos, light_pos, light_radius);
        light_contribution *= terrain_shadow * actor_shadow;
        
        // Use brightest-wins blending to match CPU np.maximum behavior
        final_color = max(final_color, light_contribution);
    }

    // Add emission contribution from glowing tiles (acid pools, hot coals, etc.)
    vec3 emission_contribution = calculateEmissionContribution(v_uv, tile_pos);
    final_color = max(final_color, emission_contribution);

    // Apply directional lighting (sun/moon) if sky exposure is present
    if (u_sun_intensity > 0.0) {
        // Sample sky exposure for this tile
        // Convert world position to map UV coordinates (sky exposure texture covers full map)
        vec2 map_uv = tile_pos / vec2(u_map_size);
        float sky_exposure = texture(u_sky_exposure_map, map_uv).r;
        
        if (sky_exposure > 0.1) {
            // Full outdoor sunlight for high sky exposure
            float effective_exposure = pow(sky_exposure, u_sky_exposure_power);
            
            // Calculate sun contribution
            vec3 sun_contribution = u_sun_color * u_sun_intensity * effective_exposure;
            
            // Apply directional shadows in outdoor areas
            // Combine terrain shadows (from grid texture) with actor shadows
            float terrain_dir_shadow = computeDirectionalShadow(tile_pos, sky_exposure);
            float actor_dir_shadow = computeActorDirectionalShadow(tile_pos, sky_exposure);
            sun_contribution *= terrain_dir_shadow * actor_dir_shadow;
            
            // Use brightest-wins blending with point lights
            final_color = max(final_color, sun_contribution);
        } else if (sky_exposure > 0.0) {
            // Light spillover for areas with minimal sky exposure (open doors, etc.)
            // Increase spillover intensity to make it more visible
            float spillover_multiplier = 3.0; // Make spillover 3x more intense
            float spillover_exposure = sky_exposure * spillover_multiplier;
            
            // Calculate spillover sun contribution
            vec3 spillover_contribution = u_sun_color * u_sun_intensity * spillover_exposure;
            
            // Use brightest-wins blending with point lights
            final_color = max(final_color, spillover_contribution);
        }
    }
    
    // Clamp to valid range and write output
    final_color = clamp(final_color, 0.0, 1.0);
    f_color = vec4(final_color, 1.0);
}
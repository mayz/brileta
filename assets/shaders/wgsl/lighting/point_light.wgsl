// WGSL lighting shader - port of ModernGL fragment shader-based point light computation
// Preserves critical shadow algorithms and directional lighting exactly

struct VertexInput {
    @location(0) position: vec2f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
    @location(1) world_pos: vec2f,
}

// Simplified struct approach - use vec4f arrays for proper alignment
struct LightingUniforms {
    // Viewport uniforms
    viewport_data: vec4f,              // offset_x, offset_y, size_x, size_y
    
    // Light metadata  
    light_count: i32,
    ambient_light: f32,
    time: f32,
    tile_aligned: u32,
    
    // Light data arrays (all vec4f for 16-byte alignment)
    light_positions: array<vec4f, 32>,     // xy in .xy, zw unused
    light_radii: array<vec4f, 32>,         // radius in .x, yzw unused  
    light_intensities: array<vec4f, 32>,   // intensity in .x, yzw unused
    light_colors: array<vec4f, 32>,        // rgb in .xyz, w unused
    
    // Flicker data
    light_flicker_enabled: array<vec4f, 32>,   // enabled in .x, yzw unused
    light_flicker_speed: array<vec4f, 32>,     // speed in .x, yzw unused
    light_min_brightness: array<vec4f, 32>,    // brightness in .x, yzw unused
    light_max_brightness: array<vec4f, 32>,    // brightness in .x, yzw unused
    
    // Actor shadow uniforms (terrain shadows use shadow_grid texture)
    actor_shadow_count: i32,
    shadow_intensity: f32,
    shadow_max_length: i32,
    shadow_falloff_enabled: u32,
    actor_shadow_positions: array<vec4f, 64>, // xy in .xy, zw unused
    
    // Directional light uniforms (sun/moon)
    sun_direction: vec2f,
    _padding1: vec2f, // Padding for alignment
    sun_color: vec3f,
    sun_intensity: f32,
    sky_exposure_power: f32,
    sun_shadow_intensity: f32,  // Separate shadow intensity for sun (outdoor shadows)
    _padding2: vec2f,  // Ensure 16-byte alignment
    map_size: vec2f,  // Full map dimensions for sky exposure UV calculation
    _padding3: vec2f,  // Padding for 16-byte alignment
}

@group(0) @binding(0) var<uniform> uniforms: LightingUniforms;
@group(0) @binding(22) var sky_exposure_map: texture_2d<f32>;
@group(0) @binding(23) var texture_sampler: sampler;
@group(0) @binding(24) var emission_map: texture_2d<f32>;
@group(0) @binding(25) var shadow_grid: texture_2d<f32>;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Full-screen quad vertex processing
    output.clip_position = vec4f(input.position, 0.0, 1.0);
    
    // Convert from [-1,1] to [0,1] UV coordinates
    output.uv = input.position * 0.5 + 0.5;
    
    // Calculate world position for this fragment
    // UV (0,0) = top-left, UV (1,1) = bottom-right
    // But game world has (0,0) at bottom-left, so we need to flip Y
    let pixel_in_viewport = vec2f(
        output.uv.x * uniforms.viewport_data.z,
        (1.0 - output.uv.y) * uniforms.viewport_data.w  // Flip Y coordinate
    );
    output.world_pos = vec2f(uniforms.viewport_data.x, uniforms.viewport_data.y) + pixel_in_viewport;
    
    return output;
}

// Noise function for flicker effects - exact port from GLSL
fn noise2d(coord: vec2f) -> f32 {
    let c = floor(coord);
    let f = fract(coord);

    // Hash function for deterministic noise
    let a = sin(dot(c, vec2f(12.9898, 78.233))) * 43758.5453;
    let b = sin(dot(c + vec2f(1.0, 0.0), vec2f(12.9898, 78.233))) * 43758.5453;
    let c_val = sin(dot(c + vec2f(0.0, 1.0), vec2f(12.9898, 78.233))) * 43758.5453;
    let d = sin(dot(c + vec2f(1.0, 1.0), vec2f(12.9898, 78.233))) * 43758.5453;

    let a_fract = fract(a);
    let b_fract = fract(b);
    let c_fract = fract(c_val);
    let d_fract = fract(d);

    // Smooth interpolation
    let u = f * f * (3.0 - 2.0 * f);

    return mix(mix(a_fract, b_fract, u.x), mix(c_fract, d_fract, u.x), u.y) * 2.0 - 1.0;  // Range [-1, 1]
}

// Compute point light shadow by marching toward the light and sampling shadow grid texture
fn computePointLightShadow(tile_pos: vec2f, light_pos: vec2f, light_radius: f32) -> f32 {
    let to_light = light_pos - tile_pos;
    let dist_to_light = length(to_light);

    // Early exit if outside light influence
    if (dist_to_light > light_radius + f32(uniforms.shadow_max_length)) {
        return 1.0;
    }

    // Use sign-based stepping (matches previous discrete behavior)
    var step_dir: vec2f;
    step_dir.x = select(select(0.0, -1.0, to_light.x < 0.0), 1.0, to_light.x > 0.0);
    step_dir.y = select(select(0.0, -1.0, to_light.y < 0.0), 1.0, to_light.y > 0.0);

    var pos = tile_pos;
    let steps = min(i32(max(abs(to_light.x), abs(to_light.y))), uniforms.shadow_max_length);
    let map_size = vec2i(uniforms.map_size);

    for (var i = 1; i <= steps; i++) {
        pos += step_dir;

        // Stop if we've reached or passed the light
        let to_light_now = light_pos - pos;
        if (dot(to_light_now, to_light) <= 0.0) {
            break;  // Passed the light, no shadow
        }

        let texel = vec2i(pos);
        if (texel.x >= 0 && texel.x < map_size.x &&
            texel.y >= 0 && texel.y < map_size.y) {
            let blocks = textureLoad(shadow_grid, texel, 0).r;
            if (blocks > 0.5) {
                // Blocker between pixel and light - calculate shadow with falloff
                var distance_falloff = 1.0;
                if (uniforms.shadow_falloff_enabled != 0u) {
                    distance_falloff = 1.0 - (f32(i - 1) / f32(uniforms.shadow_max_length));
                }
                return 1.0 - uniforms.shadow_intensity * distance_falloff;
            }
        }
    }

    return 1.0;  // No blockers found
}

// Compute actor shadow attenuation (for NPCs that block light)
fn computeActorShadow(tile_pos: vec2f, light_pos: vec2f, light_radius: f32) -> f32 {
    var shadow_factor = 1.0;

    for (var i = 0; i < uniforms.actor_shadow_count && i < 64; i++) {
        let actor_pos = uniforms.actor_shadow_positions[i].xy;

        // Calculate displacement from light to actor
        let light_to_actor = actor_pos - light_pos;
        let dx = light_to_actor.x;
        let dy = light_to_actor.y;

        // Use Chebyshev distance
        let actor_distance = max(abs(dx), abs(dy));
        if (actor_distance < 0.1) {
            continue;  // Skip if too close
        }

        // Calculate shadow direction using step function
        var shadow_dx = 0.0;
        var shadow_dy = 0.0;
        if (dx > 0.0) { shadow_dx = 1.0; } else if (dx < 0.0) { shadow_dx = -1.0; }
        if (dy > 0.0) { shadow_dy = 1.0; } else if (dy < 0.0) { shadow_dy = -1.0; }

        // Check if tile_pos is in the shadow path from actor
        let max_shadow_length = f32(uniforms.shadow_max_length);
        var in_shadow = false;
        var shadow_intensity_val = 0.0;

        for (var j = 1; j <= i32(max_shadow_length) && j <= 3; j++) {
            let shadow_pos = actor_pos + vec2f(shadow_dx * f32(j), shadow_dy * f32(j));

            // Check if tile_pos matches this shadow position
            if (abs(tile_pos.x - shadow_pos.x) < 0.1 && abs(tile_pos.y - shadow_pos.y) < 0.1) {
                var distance_falloff = 1.0;
                if (uniforms.shadow_falloff_enabled != 0u) {
                    distance_falloff = 1.0 - (f32(j - 1) / max_shadow_length);
                }
                shadow_intensity_val = max(shadow_intensity_val, uniforms.shadow_intensity * distance_falloff);
                in_shadow = true;
            }
        }

        if (in_shadow) {
            shadow_factor *= (1.0 - shadow_intensity_val);
        }
    }

    return shadow_factor;
}

// Compute directional shadow by marching toward sun and sampling shadow grid texture
fn computeDirectionalShadow(tile_pos: vec2f, sky_exposure: f32) -> f32 {
    if (sky_exposure <= 0.1 || uniforms.sun_intensity <= 0.0) {
        return 1.0;  // No shadow
    }

    // March toward sun (opposite of shadow cast direction)
    // Use sign-based stepping to match discrete tile behavior
    var step_dir: vec2f;
    step_dir.x = select(select(0.0, -1.0, uniforms.sun_direction.x < 0.0), 1.0, uniforms.sun_direction.x > 0.0);
    step_dir.y = select(select(0.0, -1.0, uniforms.sun_direction.y < 0.0), 1.0, uniforms.sun_direction.y > 0.0);

    var pos = tile_pos;
    var shadow_factor = 1.0;
    let map_size = vec2i(uniforms.map_size);

    for (var i = 1; i <= uniforms.shadow_max_length; i++) {
        pos += step_dir;

        let texel = vec2i(pos);
        if (texel.x < 0 || texel.x >= map_size.x || texel.y < 0 || texel.y >= map_size.y) {
            break;
        }

        // Use textureLoad for pixel-exact sampling (no filtering)
        let blocks = textureLoad(shadow_grid, texel, 0).r;
        if (blocks > 0.5) {
            // Found a blocker - calculate shadow with falloff
            var distance_falloff = 1.0;
            if (uniforms.shadow_falloff_enabled != 0u) {
                distance_falloff = 1.0 - (f32(i - 1) / f32(uniforms.shadow_max_length));
            }
            shadow_factor *= (1.0 - uniforms.sun_shadow_intensity * distance_falloff);
            break;  // First blocker wins for directional light
        }
    }

    return shadow_factor;
}

// Compute actor shadows for directional (sun) light
fn computeActorDirectionalShadow(tile_pos: vec2f, sky_exposure: f32) -> f32 {
    if (sky_exposure <= 0.1 || uniforms.sun_intensity <= 0.0) {
        return 1.0;
    }

    // Shadow direction is opposite of sun direction
    var shadow_dx = 0.0;
    var shadow_dy = 0.0;
    if (uniforms.sun_direction.x > 0.0) { shadow_dx = -1.0; } else if (uniforms.sun_direction.x < 0.0) { shadow_dx = 1.0; }
    if (uniforms.sun_direction.y > 0.0) { shadow_dy = -1.0; } else if (uniforms.sun_direction.y < 0.0) { shadow_dy = 1.0; }

    var shadow_factor = 1.0;

    for (var i = 0; i < uniforms.actor_shadow_count && i < 64; i++) {
        let actor_pos = uniforms.actor_shadow_positions[i].xy;

        // Check if tile_pos is in the shadow path from this actor
        let max_shadow_length = f32(uniforms.shadow_max_length);
        var in_shadow = false;
        var shadow_intensity_val = 0.0;

        for (var j = 1; j <= i32(max_shadow_length); j++) {
            let shadow_pos = actor_pos + vec2f(shadow_dx * f32(j), shadow_dy * f32(j));

            if (abs(tile_pos.x - shadow_pos.x) < 0.1 && abs(tile_pos.y - shadow_pos.y) < 0.1) {
                var distance_falloff = 1.0;
                if (uniforms.shadow_falloff_enabled != 0u) {
                    distance_falloff = 1.0 - (f32(j - 1) / max_shadow_length);
                }
                shadow_intensity_val = max(shadow_intensity_val, uniforms.sun_shadow_intensity * distance_falloff);
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
fn calculateEmissionContribution(uv: vec2f) -> vec3f {
    var emission_total = vec3f(0.0);

    // Maximum radius to check (must match max light_radius in tile data)
    let MAX_EMISSION_RADIUS = 4;

    // Get texture dimensions and current pixel coordinates
    // The vertex shader computes world_pos.y = viewport.y1 + (1.0 - uv.y) * height
    // This means UV y=0 corresponds to the TOP of the viewport (max world Y)
    // The emission texture has row 0 = vp_y=0 = bottom of viewport (min world Y)
    // So we need: pixel_y = (1.0 - uv.y) * height to match world Y to texture row
    let tex_dims = textureDimensions(emission_map);
    let pixel_coord = vec2i(
        i32(uv.x * f32(tex_dims.x)),
        min(i32((1.0 - uv.y) * f32(tex_dims.y)), i32(tex_dims.y) - 1)  // Flip Y, clamp to bounds
    );

    // Sample nearby tiles for emission using textureLoad (no filtering needed)
    for (var dy = -MAX_EMISSION_RADIUS; dy <= MAX_EMISSION_RADIUS; dy++) {
        for (var dx = -MAX_EMISSION_RADIUS; dx <= MAX_EMISSION_RADIUS; dx++) {
            // Calculate pixel coordinate of neighboring tile
            // Since base pixel_coord is already Y-flipped to match world coords to texture rows,
            // dy in world space (positive = higher world Y = higher vp_y = higher pixel y)
            // maps directly to +dy in pixel space
            let neighbor_coord = pixel_coord + vec2i(dx, dy);

            // Skip if outside texture bounds
            if (neighbor_coord.x < 0 || neighbor_coord.x >= i32(tex_dims.x) ||
                neighbor_coord.y < 0 || neighbor_coord.y >= i32(tex_dims.y)) {
                continue;
            }

            // Load emission texture (no sampler needed)
            let emission = textureLoad(emission_map, neighbor_coord, 0);

            // Check if this tile emits light (radius > 0)
            let radius = emission.a;
            if (radius <= 0.0) {
                continue;
            }

            // Calculate distance to emitting tile (in world space, using original dx/dy)
            let dist = length(vec2f(f32(dx), f32(dy)));

            // Skip if outside emission radius
            if (dist > radius) {
                continue;
            }

            // Calculate falloff (linear, matching point lights)
            let falloff = max(0.0, 1.0 - dist / radius);

            // Add emission contribution (RGB is pre-multiplied by intensity)
            emission_total += emission.rgb * falloff;
        }
    }

    return emission_total;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    var world_pos = input.world_pos;
    let tile_pos = floor(world_pos);

    if (uniforms.tile_aligned != 0u) {
        world_pos = tile_pos;
    }
    
    var final_color = vec3f(uniforms.ambient_light);
    
    // Point Lights
    for (var i = 0; i < uniforms.light_count && i < 32; i++) {
        let light_pos = uniforms.light_positions[i].xy;
        let light_radius = uniforms.light_radii[i].x;
        let base_intensity = uniforms.light_intensities[i].x;
        let light_color = uniforms.light_colors[i].xyz;
        
        let distance = length(world_pos - light_pos);
        if (distance > light_radius) {
            continue;
        }
        
        var attenuation = max(0.0, 1.0 - (distance / light_radius));
        var intensity = base_intensity;
        
        if (uniforms.light_flicker_enabled[i].x > 0.5) {
            let flicker_noise = noise2d(vec2f(uniforms.time * uniforms.light_flicker_speed[i].x, 0.0));
            let min_brightness = uniforms.light_min_brightness[i].x;
            let max_brightness = uniforms.light_max_brightness[i].x;
            let flicker_multiplier = min_brightness + ((flicker_noise + 1.0) * 0.5 * (max_brightness - min_brightness));
            intensity *= flicker_multiplier;
        }
        
        attenuation *= intensity;
        
        var light_contribution = light_color * attenuation;

        // Apply shadow attenuation using tile position for consistency
        // Combine terrain shadows (from grid texture) with actor shadows
        let terrain_shadow = computePointLightShadow(tile_pos, light_pos, light_radius);
        let actor_shadow = computeActorShadow(tile_pos, light_pos, light_radius);
        light_contribution *= terrain_shadow * actor_shadow;

        final_color = max(final_color, light_contribution);
    }

    // Add emission contribution from glowing tiles (acid pools, hot coals, etc.)
    let emission_contribution = calculateEmissionContribution(input.uv);
    final_color = max(final_color, emission_contribution);

    // Directional Light (Sun/Moon)
    if (uniforms.sun_intensity > 0.0) {
        // Convert world position to map UV coordinates (sky exposure texture covers full map)
        let map_uv = tile_pos / uniforms.map_size;
        let sky_exposure = textureSample(sky_exposure_map, texture_sampler, map_uv).r;

        if (sky_exposure > 0.1) {
            let effective_exposure = pow(sky_exposure, uniforms.sky_exposure_power);
            var sun_contribution = uniforms.sun_color * uniforms.sun_intensity * effective_exposure;

            // Apply directional shadows in outdoor areas
            // Combine terrain shadows (from grid texture) with actor shadows
            let terrain_dir_shadow = computeDirectionalShadow(tile_pos, sky_exposure);
            let actor_dir_shadow = computeActorDirectionalShadow(tile_pos, sky_exposure);
            sun_contribution *= terrain_dir_shadow * actor_dir_shadow;

            final_color = max(final_color, sun_contribution);
        } else if (sky_exposure > 0.0) {
            let spillover_multiplier = 3.0;
            let spillover_exposure = sky_exposure * spillover_multiplier;
            let spillover_contribution = uniforms.sun_color * uniforms.sun_intensity * spillover_exposure;
            final_color = max(final_color, spillover_contribution);
        }
    }
    
    final_color = clamp(final_color, vec3f(0.0), vec3f(1.0));
    return vec4f(final_color, 1.0);
}
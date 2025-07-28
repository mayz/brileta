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
    
    // Shadow casting uniforms
    shadow_caster_count: i32,
    shadow_intensity: f32,
    shadow_max_length: i32,
    shadow_falloff_enabled: u32,
    shadow_caster_positions: array<vec4f, 64>, // xy in .xy, zw unused
    
    // Directional light uniforms (sun/moon)
    sun_direction: vec2f,
    _padding1: vec2f, // Padding for alignment
    sun_color: vec3f,
    sun_intensity: f32,
    sky_exposure_power: f32,
    _padding2: vec3f,  // Ensure 16-byte alignment
}

@group(0) @binding(0) var<uniform> uniforms: LightingUniforms;
@group(0) @binding(22) var sky_exposure_map: texture_2d<f32>;
@group(0) @binding(23) var texture_sampler: sampler;

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

// Calculate shadow attenuation from shadow casters - EXACT port from GLSL
fn calculateShadowAttenuation(world_pos: vec2f, light_pos: vec2f, light_radius: f32) -> f32 {
    var shadow_factor = 1.0;  // No shadow by default
    
    // Early exit: Skip shadow computation if pixel is too far from light
    let distance_to_light = distance(world_pos, light_pos);
    let max_shadow_influence = light_radius + f32(uniforms.shadow_max_length);
    if (distance_to_light > max_shadow_influence) {
        return 1.0; // No shadow influence
    }
    
    // Check each shadow caster
    for (var i = 0; i < uniforms.shadow_caster_count && i < 64; i++) {
        let caster_pos = uniforms.shadow_caster_positions[i].xy;
        
        // Calculate displacement from light to caster (CPU algorithm)
        let light_to_caster = caster_pos - light_pos;
        let dx = light_to_caster.x;
        let dy = light_to_caster.y;
        
        // Use Chebyshev distance like CPU (max of absolute values)
        let caster_distance = max(abs(dx), abs(dy));
        if (caster_distance < 0.1) {
            continue;  // Skip if too close
        }
        
        // Calculate shadow direction using step function like CPU
        // CRITICAL: Sign-based (not normalized) direction vectors - preserved exactly
        var shadow_dx = 0.0;
        var shadow_dy = 0.0;
        if (dx > 0.0) { shadow_dx = 1.0; } else if (dx < 0.0) { shadow_dx = -1.0; }
        if (dy > 0.0) { shadow_dy = 1.0; } else if (dy < 0.0) { shadow_dy = -1.0; }
        
        // Check if world_pos is in the shadow path from caster
        let max_shadow_length = f32(uniforms.shadow_max_length);
        var in_shadow = false;
        var shadow_intensity = 0.0;
        
        // Check each shadow position (matching CPU loop) - CRITICAL: discrete tile-based stepping
        for (var j = 1; j <= i32(max_shadow_length) && j <= 3; j++) {
            let shadow_pos = caster_pos + vec2f(shadow_dx * f32(j), shadow_dy * f32(j));
            
            // Check if world_pos matches this shadow position (core shadow)
            if (abs(world_pos.x - shadow_pos.x) < 0.1 && abs(world_pos.y - shadow_pos.y) < 0.1) {
                // Calculate falloff like CPU
                var distance_falloff = 1.0;
                if (uniforms.shadow_falloff_enabled != 0) {
                    distance_falloff = 1.0 - (f32(j - 1) / max_shadow_length);
                }
                shadow_intensity = max(shadow_intensity, uniforms.shadow_intensity * distance_falloff);
                in_shadow = true;
            }
            
            // Check soft edges for first 2 shadow tiles (like CPU)
            if (j <= 2) {
                var edge_intensity = uniforms.shadow_intensity * 0.4; // 40% like CPU
                if (uniforms.shadow_falloff_enabled != 0) {
                    let distance_falloff = 1.0 - (f32(j - 1) / max_shadow_length);
                    edge_intensity *= distance_falloff;
                }
                
                // Check 8 adjacent/diagonal positions around core shadow
                let edge_offsets = array<vec2f, 8>(
                    vec2f(0.0, 1.0), vec2f(0.0, -1.0), vec2f(1.0, 0.0), vec2f(-1.0, 0.0),  // Adjacent
                    vec2f(1.0, 1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, -1.0)   // Diagonal
                );
                
                for (var k = 0; k < 8; k++) {
                    let edge_pos = shadow_pos + edge_offsets[k];
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

// Calculate directional shadow attenuation (sun/moon shadows) - EXACT port preserving critical algorithm
fn calculateDirectionalShadowAttenuation(world_pos: vec2f, sky_exposure: f32) -> f32 {
    // Only apply directional shadows in outdoor areas
    if (sky_exposure <= 0.1 || uniforms.sun_intensity <= 0.0) {
        return 1.0; // No directional shadows
    }
    
    var shadow_factor = 1.0;
    
    // Get shadow direction using discrete steps (matching CPU algorithm)
    // CRITICAL: CPU uses sign-based direction, not normalized vectors - preserved exactly
    var shadow_dx = 0.0;
    var shadow_dy = 0.0;
    
    // CRITICAL: Sign-based (not normalized) direction vectors - preserved exactly from WGPU migration plan
    shadow_dx = select(select(0.0, 1.0, uniforms.sun_direction.x < 0.0), -1.0, uniforms.sun_direction.x > 0.0);
    shadow_dy = select(select(0.0, 1.0, uniforms.sun_direction.y < 0.0), -1.0, uniforms.sun_direction.y > 0.0);
    
    // Check each shadow caster for directional shadows
    for (var i = 0; i < uniforms.shadow_caster_count && i < 64; i++) {
        let caster_pos = uniforms.shadow_caster_positions[i].xy;
        
        // Check if world_pos is in the shadow path from this caster
        let max_shadow_length = f32(uniforms.shadow_max_length); // Use actual config value
        let base_intensity = uniforms.shadow_intensity; // Use actual config value, no multiplier
        
        var in_shadow = false;
        var shadow_intensity = 0.0;
        
        // Cast shadow using discrete tile steps (matching CPU) - CRITICAL: discrete tile-based stepping
        for (var j = 1; j <= i32(max_shadow_length); j++) {
            // Use integer steps like CPU does
            let shadow_pos = caster_pos + vec2f(shadow_dx * f32(j), shadow_dy * f32(j));
            
            // Check if world_pos matches this shadow position (core shadow)
            if (abs(world_pos.x - shadow_pos.x) < 0.1 && abs(world_pos.y - shadow_pos.y) < 0.1) {
                // Calculate falloff with distance
                var distance_falloff = 1.0;
                if (uniforms.shadow_falloff_enabled != 0) {
                    distance_falloff = 1.0 - (f32(j - 1) / max_shadow_length);
                }
                shadow_intensity = max(shadow_intensity, base_intensity * distance_falloff);
                in_shadow = true;
            }
            
            // Add softer edges for first 2 shadow tiles (like CPU)
            if (j <= 2) {
                var edge_intensity = base_intensity * 0.4; // 40% like CPU
                if (uniforms.shadow_falloff_enabled != 0) {
                    let distance_falloff = 1.0 - (f32(j - 1) / max_shadow_length);
                    edge_intensity *= distance_falloff;
                }
                
                // Check 8 adjacent/diagonal positions (matching CPU)
                let edge_offsets = array<vec2f, 8>(
                    vec2f(0.0, 1.0), vec2f(0.0, -1.0), vec2f(1.0, 0.0), vec2f(-1.0, 0.0),  // Adjacent
                    vec2f(1.0, 1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, -1.0)   // Diagonal
                );
                
                for (var k = 0; k < 8; k++) {
                    let edge_pos = shadow_pos + edge_offsets[k];
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
        
        let shadow_attenuation = calculateShadowAttenuation(tile_pos, light_pos, light_radius);
        light_contribution *= shadow_attenuation;
        
        final_color = max(final_color, light_contribution);
    }
    
    // Directional Light (Sun/Moon)
    if (uniforms.sun_intensity > 0.0) {
        let sky_exposure = textureSample(sky_exposure_map, texture_sampler, input.uv).r;
        
        if (sky_exposure > 0.1) {
            let effective_exposure = pow(sky_exposure, uniforms.sky_exposure_power);
            var sun_contribution = uniforms.sun_color * uniforms.sun_intensity * effective_exposure;
            
            let directional_shadow_attenuation = calculateDirectionalShadowAttenuation(tile_pos, sky_exposure);
            sun_contribution *= directional_shadow_attenuation;
            
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
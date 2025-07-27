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

struct LightingUniforms {
    // Viewport uniforms for coordinate calculation
    viewport_offset: vec2i,
    viewport_size: vec2i,
    
    // Light data
    light_count: i32,
    _padding1: vec3i,               // Pad to 16-byte alignment
    light_positions: array<vec4f, 32>,       // Light positions in world space (xy used, zw padding)
    light_radii: array<vec4f, 32>,          // x used, yzw padding
    light_intensities: array<vec4f, 32>,    // x used, yzw padding
    light_colors: array<vec4f, 32>, // Changed from vec3f to vec4f for alignment
    
    // Flicker data
    light_flicker_enabled: array<vec4f, 32>,   // x used, yzw padding
    light_flicker_speed: array<vec4f, 32>,     // x used, yzw padding
    light_min_brightness: array<vec4f, 32>,    // x used, yzw padding
    light_max_brightness: array<vec4f, 32>,    // x used, yzw padding
    
    // Global uniforms
    ambient_light: f32,
    time: f32,
    tile_aligned: u32,
    _padding2: u32,
    
    // Shadow casting uniforms
    shadow_caster_count: i32,
    shadow_intensity: f32,
    shadow_max_length: i32,
    shadow_falloff_enabled: u32,
    shadow_caster_positions: array<vec4f, 64>, // Shadow caster positions (xy used, zw padding)
    
    // Directional light uniforms (sun/moon)
    sun_direction: vec2f,
    sun_color: vec3f,
    sun_intensity: f32,
    sky_exposure_power: f32,
    _padding3: vec3f,               // Pad to 16-byte alignment
}

@group(0) @binding(0) var<uniform> uniforms: LightingUniforms;
@group(0) @binding(1) var sky_exposure_map: texture_2d<f32>;
@group(0) @binding(2) var texture_sampler: sampler;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Full-screen quad vertex processing
    output.clip_position = vec4f(input.position, 0.0, 1.0);
    
    // Convert from [-1,1] to [0,1] UV coordinates
    output.uv = input.position * 0.5 + 0.5;
    
    // Calculate world position for this fragment
    // UV (0,0) = top-left, UV (1,1) = bottom-right
    let pixel_in_viewport = output.uv * vec2f(uniforms.viewport_size);
    output.world_pos = vec2f(uniforms.viewport_offset) + pixel_in_viewport;
    
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
    
    // For lighting calculations, use integer tile coordinates to match CPU
    // The CPU calculates distance from light to integer tile positions
    let tile_pos = floor(world_pos);
    
    // For tile-aligned mode, we still need to determine which tile we're in
    if (uniforms.tile_aligned != 0) {
        // Use the tile position for distance calculations (matching CPU)
        world_pos = tile_pos;
    }
    
    // Start with ambient lighting
    var final_color = vec3f(uniforms.ambient_light);
    
    // Add contribution from each point light
    for (var i = 0; i < uniforms.light_count && i < 32; i++) {
        let light_pos = uniforms.light_positions[i].xy;
        let light_radius = uniforms.light_radii[i].x;
        let base_intensity = uniforms.light_intensities[i].x;
        let light_color = uniforms.light_colors[i].xyz;
        
        // Calculate distance from light
        let light_vec = world_pos - light_pos;
        let distance = length(light_vec);
        
        // Skip if outside light radius
        if (distance > light_radius) {
            continue;
        }
        
        // Calculate base attenuation (linear falloff to match CPU)
        var attenuation = max(0.0, 1.0 - (distance / light_radius));
        
        // Apply flicker if enabled
        var intensity = base_intensity;
        if (uniforms.light_flicker_enabled[i].x > 0.5) {
            // Create flicker using noise - similar to CPU implementation
            let flicker_noise = noise2d(vec2f(uniforms.time * uniforms.light_flicker_speed[i].x, 0.0));
            // Convert from [-1, 1] to [min_brightness, max_brightness]
            let flicker_multiplier = uniforms.light_min_brightness[i].x +
                ((flicker_noise + 1.0) * 0.5 * (uniforms.light_max_brightness[i].x - uniforms.light_min_brightness[i].x));
            intensity *= flicker_multiplier;
        }
        
        attenuation *= intensity;
        
        // Calculate light contribution
        var light_contribution = light_color * attenuation;
        
        // Apply shadow attenuation using tile position for consistency
        let shadow_attenuation = calculateShadowAttenuation(tile_pos, light_pos, light_radius);
        light_contribution *= shadow_attenuation;
        
        // Use brightest-wins blending to match CPU np.maximum behavior
        final_color = max(final_color, light_contribution);
    }
    
    // Apply directional lighting (sun/moon) if sky exposure is present
    if (uniforms.sun_intensity > 0.0) {
        // Sample sky exposure for this tile - CRITICAL: Sky exposure sampling preserved exactly
        // v_uv is already normalized to [0,1] for the viewport, we need to map to full map
        let map_uv = input.uv;  // The texture coordinates should already be correct
        let sky_exposure = textureSample(sky_exposure_map, texture_sampler, map_uv).r;
        
        if (sky_exposure > 0.1) {
            // Full outdoor sunlight for high sky exposure
            let effective_exposure = pow(sky_exposure, uniforms.sky_exposure_power);
            
            // Calculate sun contribution
            var sun_contribution = uniforms.sun_color * uniforms.sun_intensity * effective_exposure;
            
            // Apply directional shadows in outdoor areas
            let directional_shadow_attenuation = calculateDirectionalShadowAttenuation(tile_pos, sky_exposure);
            sun_contribution *= directional_shadow_attenuation;
            
            // Use brightest-wins blending with point lights
            final_color = max(final_color, sun_contribution);
        } else if (sky_exposure > 0.0) {
            // Light spillover for areas with minimal sky exposure (open doors, etc.)
            // Increase spillover intensity to make it more visible
            let spillover_multiplier = 3.0; // Make spillover 3x more intense
            let spillover_exposure = sky_exposure * spillover_multiplier;
            
            // Calculate spillover sun contribution
            let spillover_contribution = uniforms.sun_color * uniforms.sun_intensity * spillover_exposure;
            
            // Use brightest-wins blending with point lights
            final_color = max(final_color, spillover_contribution);
        }
    }
    
    // Clamp to valid range and write output
    final_color = clamp(final_color, vec3f(0.0), vec3f(1.0));
    return vec4f(final_color, 1.0);
}
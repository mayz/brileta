# GPU Lighting System Plan

## Current Architecture

The lighting system is a single-pass WGPU fragment shader pipeline with no CPU fallback. All shaders live in `assets/shaders/wgsl/` with subdirectories:
- `glyph/` - Character rendering (render.wgsl)
- `lighting/` - GPU lighting system (point_light.wgsl)
- `screen/` - Main screen rendering (main.wgsl)
- `effects/` - Atmospheric effects (cloud_layer.wgsl)

### What's Implemented

- WGPU fragment shader pipeline (chose fragment shaders over compute shaders for macOS compatibility)
- Point lights with distance falloff, flicker, and shadow casting
- Directional sun with ray-marched terrain shadows
- Sky exposure for indoor/outdoor transitions
- Tile emission
- Atmospheric effects (cloud shadows, ground mist)
- GPU compositing with fog-of-war integration
- Performance caching (revision tracking, sky exposure texture caching, uniform buffer hashing)

### Core Components

#### GPULightingSystem
- Location: `brileta/backends/wgpu/gpu_lighting.py`
- Purpose: Main class implementing LightingSystem interface
- Key methods:
  - `compute_lightmap()` - Main API, returns numpy array
  - `update()` - Handle time-based effects
  - `on_light_*()` - Cache management

#### Shader Pipeline
- Point Light Shader: Distance-based falloff, color blending
- Directional Light Shader: Sky exposure, directional illumination
- Shadow Shader: Raycast shadows from light sources
- Composite Shader: Combine lights with ambient, apply gamma

#### GPU Memory Layout

The actual uniform struct in `point_light.wgsl` packs light data into parallel `vec4f` arrays for 16-byte alignment (max 32 lights):

```wgsl
struct LightingUniforms {
    viewport_data: vec4f,              // offset_x, offset_y, size_x, size_y

    light_count: i32,
    ambient_light: f32,
    time: f32,
    tile_aligned: u32,

    // Light data arrays (all vec4f for 16-byte alignment)
    light_positions: array<vec4f, 32>,
    light_radii: array<vec4f, 32>,
    light_intensities: array<vec4f, 32>,
    light_colors: array<vec4f, 32>,

    // Flicker data
    light_flicker_enabled: array<vec4f, 32>,
    light_flicker_speed: array<vec4f, 32>,
    light_min_brightness: array<vec4f, 32>,
    light_max_brightness: array<vec4f, 32>,

    // Directional light (sun/moon)
    sun_direction: vec2f,
    sun_color: vec3f,
    sun_intensity: f32,
    sky_exposure_power: f32,
    sun_shadow_intensity: f32,
    sun_shadow_length_scale: f32,
    map_size: vec2f,
    // (padding fields omitted for clarity)
}
```

Bindings: uniform buffer (binding 0), sky exposure map, sampler, emission map, shadow grid.

---

## Future Work

### Multi-Pass Lighting (Optimization)

**Goal**: Separate static and dynamic lighting into distinct passes to avoid re-computing unchanging light contributions every frame.

**Priority**: 5/10 - The single-pass approach works well. This becomes worthwhile if light counts grow significantly or if static scenes dominate.

**Design**:

1.  **Static Lightmap** - A persistent GPU texture caching ambient, static point lights, and directional light contributions. Only re-rendered when:
    - Static light positions, colors, or radii change
    - Directional light properties change (e.g., time of day)
    - Static shadow casters move (map structure changes)

2.  **Dynamic Lightmap** - A per-frame GPU texture for flickering torches, spell effects, and lights attached to moving actors. Dynamic shadow casters (actors) are computed here.

3.  **Composite Pass** - Blends static and dynamic lightmaps (e.g., `max()`) and applies gamma correction.

4.  **Separate Collections** - `static_lights`, `dynamic_lights`, `static_shadow_casters`, `dynamic_shadow_casters` routed to the appropriate pass.

5.  **Cache Invalidation** - A `static_cache_valid` flag tied to `game_map.structural_revision`.

### Continuous Lighting

**Goal**: Sub-tile resolution lighting for smoother visual effects.

**Priority**: 8/10 - Major visual enhancement opportunity.

- Higher resolution lighting computation (2x, 4x, 8x per tile)
- Smooth gradients across tile boundaries
- Configuration option to choose tile-based vs continuous
- Shadows that don't align to tile grid

### Performance Optimizations

**Goal**: Maximize GPU efficiency for higher light counts.

**Priority**: 6/10 - Enables more complex lighting scenarios.

- Frustum culling for lights
- Level-of-detail for distant lights
- Batched light data updates
- GPU memory usage optimization
- More efficient viewport-based light culling algorithms
- Performance monitoring hooks

### Environmental Shadow Effects

**Status**: Core cloud shadows implemented (`AtmosphericLayerSystem`, `cloud_layer.wgsl`). Moving cloud shadow patterns, ground mist, noise-based generation, and sky exposure masking are all working.

**Remaining**:
- Weather-based shadow effects (storm clouds, etc.)
- Configurable shadow pattern textures

### Configuration & Quality Settings

**Priority**: 4/10 - User experience and debugging improvements.

- Quality presets (Performance/Balanced/Quality)
- Debug visualization tools
- Hardware capability detection

---

## Ideas (Someday/Maybe)

- Soft shadows with penumbra
- Volumetric lighting effects
- Color temperature simulation
- Weather-based lighting effects (storm darkening, lightning flashes)
- Fog and atmospheric scattering
- Water reflection and refraction effects

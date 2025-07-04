# ModernGL Rendering Architecture

## Overview

This document describes the architecture of the ModernGL-based rendering system that replaced the original TCOD backend. The system is designed for high performance through persistent GPU resource management, intelligent change detection, and batched rendering operations.

## Core Design Principles

### 1. Persistent GPU Resource Management

All GPU resources (textures, framebuffers, VBOs, VAOs) are created once and reused across frames. This eliminates the performance bottleneck of per-frame GPU resource creation/destruction.

**Key Resources:**
- Atlas texture: Created at startup, contains the tileset
- FBO cache: Pool of reusable framebuffers for UI rendering
- VBO/VAO pairs: Persistent vertex buffers for batched rendering
- Gradient textures: Pre-computed environmental effect textures

**Implementation:** 
- `ModernGLGraphicsContext` manages the FBO cache via `_get_or_create_render_target()`
- Resources are only recreated when dimensions change
- Proper cleanup handled during shutdown

### 2. TCOD-Inspired Change Detection

Static content generates zero CPU overhead after initial render through dirty tile tracking.

**Change Detection System:**
- Vectorized NumPy comparisons identify dirty tiles: `(glyph_buffer.data["ch"] != cache_buffer.data["ch"])`
- Only changed tiles have their vertex data updated
- Completely unchanged buffers skip all rendering operations
- LRU cache prevents memory growth

**Benefits:**
- Static UI panels: Zero CPU work until content changes
- Cursor blinking: Only 1-2 tiles marked dirty per frame
- Menu hover effects: Only highlighted items get updated

### 3. Batched Rendering Pipeline

Minimizes draw calls through efficient vertex batching and shader management.

**Rendering Strategy:**
- Screen renderer: Batches up to 10,000 quads for main game content
- UI texture renderer: Handles multiple textures with batched vertex uploads
- Immediate rendering: Persistent VBOs for single-quad operations
- Dual shader system: Separate pipelines for screen and UI content

## System Architecture

### Core Components

#### ModernGLGraphicsContext
**Location:** `catley/backends/moderngl/graphics.py`

Primary interface implementing the GraphicsContext ABC. Manages all GPU resources and coordinates between rendering subsystems.

**Key responsibilities:**
- FBO cache management for UI rendering
- Atlas texture and UV coordinate mapping
- Letterbox geometry calculations
- Resource lifecycle management

#### ScreenRenderer
**Location:** `catley/backends/moderngl/screen_renderer.py`

Specialized renderer for main game content using the tileset atlas.

**Features:**
- Batches up to 60,000 vertices (10,000 quads) per frame
- Letterbox-aware coordinate transformation
- Persistent VBO with dynamic updates
- Simple but efficient vertex/fragment shaders

#### TextureRenderer
**Location:** `catley/backends/moderngl/texture_renderer.py`

Renders GlyphBuffers to cached textures for UI elements.

**Optimization features:**
- TCOD-style change detection with dirty tile tracking
- Partial VBO updates for changed content only
- Per-Canvas resource isolation
- Vectorized vertex generation

#### UITextureRenderer
**Location:** `catley/backends/moderngl/graphics.py` (UITextureRenderer class)

Batched renderer for pre-rendered UI textures.

**Capabilities:**
- Multiple texture support in single batch
- Efficient vertex aggregation
- Single GPU upload per frame

### Shader Pipeline

#### Screen Vertex Shader
```glsl
// Transforms pixel coordinates to clip space with letterboxing
in vec2 in_vert;       // Pixel position
in vec2 in_uv;         // Atlas texture coordinates
in vec4 in_color;      // RGBA color (0.0-1.0)

uniform vec4 u_letterbox;   // (offset_x, offset_y, scaled_w, scaled_h)

// Normalize to letterbox space, then to clip space
float x = (adjusted_pos.x / u_letterbox.z) * 2.0 - 1.0;
float y = (1.0 - (adjusted_pos.y / u_letterbox.w)) * 2.0 - 1.0;
```

#### Fragment Shader
```glsl
// Simple texture sampling with color multiplication
vec4 tex_color = texture(u_atlas, v_uv);
f_color = tex_color * v_color;
```

**Design rationale:** Simple shaders based on TCOD analysis. Complex lookup-based shaders proved unnecessary for target performance.

## Performance Characteristics

### Optimization Results

**Original bottlenecks identified and resolved:**
- GPU resource churn: Eliminated through persistent resource management
- CPU vertex generation: Optimized through change detection and vectorization
- Memory allocation: Reduced through pooling and caching strategies

**Performance metrics:**
- Static UI content: Zero CPU overhead after initial render
- Dynamic content: Only changed tiles processed
- GPU memory: Efficient reuse prevents allocation overhead

### Comparison with TCOD

**What we adopted from TCOD:**
- Persistent atlas texture approach
- Change detection for static content optimization
- Batched vertex rendering to minimize draw calls
- Simple, efficient shader design

**What we do differently:**
- ModernGL direct GPU access vs SDL abstraction
- FBO-based UI rendering vs immediate mode
- Separate shader pipelines for different content types
- More sophisticated resource caching

**Performance outcome:** Achieved comparable performance to TCOD while gaining additional rendering capabilities for modern effects.

## Resource Management

### FBO Caching System

**Purpose:** Eliminate per-frame creation of framebuffers for UI rendering.

**Implementation:**
```python
# Cache keyed by dimensions
self.fbo_cache: dict[tuple[int, int], tuple[moderngl.Framebuffer, moderngl.Texture]] = {}

def _get_or_create_render_target(self, width: int, height: int):
    cache_key = (width, height)
    if cache_key in self.fbo_cache:
        return self.fbo_cache[cache_key]
    # Create and cache new FBO/texture pair
```

### Texture Management

**Atlas texture:** Single persistent texture containing the complete tileset
- Created once at startup from tileset PNG
- Magenta pixels converted to transparency
- UV coordinates pre-calculated for all 256 characters

**Gradient textures:** Pre-computed environmental effect textures
- Radial gradients for lighting effects
- Created once, reused for all environmental rendering
- Eliminates per-frame NumPy array creation

### Memory Layout

**Vertex format:**
```python
VERTEX_DTYPE = np.dtype([
    ("position", "2f4"),  # (x, y) in pixels
    ("uv", "2f4"),        # (u, v) texture coordinates
    ("color", "4f4"),     # (r, g, b, a) as floats 0.0-1.0
])
```

**Benefits:** Direct float format eliminates CPU-GPU conversion overhead while maintaining precision for smooth animations.

## Integration Points

### Graphics Context Interface

The ModernGL backend implements the abstract `GraphicsContext` interface, ensuring compatibility with the existing rendering pipeline:

- `render_glyph_buffer_to_texture()`: UI element rendering
- `present_texture()`: Batched texture presentation
- `draw_actor_smooth()`: Interpolated actor rendering
- `apply_environmental_effect()`: Lighting and fog effects

### Coordinate Systems

**Root console coordinates:** Tile-based coordinate system (0-79, 0-49)
**Screen pixel coordinates:** Final rendering coordinates after letterboxing
**Texture coordinates:** UV space (0.0-1.0) for atlas sampling

**Transformation pipeline:**
1. Tile coordinates → Screen pixels via `console_to_screen_coords()`
2. Screen pixels → Clip space via vertex shader letterbox transformation
3. UV coordinates → Texture samples via fragment shader

## Future Considerations

### Extensibility Points

The architecture supports future enhancements:

**GPU lighting system:** Compute shaders can leverage existing FBO infrastructure
**Advanced effects:** Environmental rendering pipeline ready for complex shaders
**Multi-texture support:** UITextureRenderer architecture scales to additional texture types

### Performance Scaling

**Current bottlenecks:** System is CPU-bound for complex game logic, not rendering
**Scaling characteristics:** Linear performance with scene complexity due to change detection
**Memory usage:** Bounded by cache sizes and texture pooling

## Lessons Learned

### Key Insights from TCOD Analysis

1. **Simple solutions work:** TCOD achieves 700 FPS with basic shaders and straightforward approaches
2. **Resource reuse is critical:** Persistent GPU resources provide the largest performance gains
3. **Change detection is powerful:** Only processing modified content eliminates most CPU overhead
4. **Batching matters:** Minimizing draw calls through vertex aggregation provides consistent performance

### Architecture Decisions

1. **ModernGL over SDL:** Direct GPU access provides more control and debugging capability
2. **Dual shader approach:** Separate pipelines for screen and UI content improves maintainability
3. **FBO-based UI rendering:** Enables complex UI effects while maintaining performance
4. **Cache-based change detection:** Balances memory usage with performance through LRU eviction

This architecture provides a solid foundation for high-performance 2D rendering while maintaining flexibility for future enhancements like GPU-based lighting and advanced visual effects.

## Appendix: TCOD Performance Analysis

This section documents the detailed analysis of TCOD's rendering pipeline that informed our architectural decisions.

### TCOD's 700 FPS Architecture

Deep examination of Python tcod wrapper and C libtcod source code reveals how TCOD achieves exceptional performance.

#### Key Findings - How TCOD Achieves 700 FPS

**1. Persistent GPU Resource Management**
- Atlas Texture: Created once at startup (`renderer_sdl2.h:56-57`, `renderer_sdl2.c:121`), never recreated per frame
- Cache Console: Persistent console tracking last frame state (`renderer_sdl2.h:73`)
- Cache Texture: Persistent render target texture (`renderer_sdl2.h:74`)
- Recreation Only on Size Change: Resources only recreated when console dimensions change (`renderer_sdl2.c:537-545`)

**2. Intelligent Change Detection**
- Tile-by-Tile Dirty Checking: Only renders tiles that changed since last frame (`renderer_sdl2.c:426-440`)
- Cache Comparison: Compares `console->tiles[i]` vs `cache->tiles[i]` for background and foreground separately
- Skip Unchanged: `continue` for unchanged tiles (line 436) - massive CPU savings

**3. Efficient Data Transfer Patterns**
- NO Direct OpenGL: Uses SDL2 abstraction layer instead of raw OpenGL calls
- SDL_UpdateTexture(): For atlas updates (`renderer_sdl2.c:85`) - equivalent to `glTexSubImage2D`
- SDL_RenderGeometryRaw(): Batched vertex rendering for modern SDL 2.0.18+ (`renderer_sdl2.c:297`)
- Fallback to SDL_RenderCopy(): Per-tile rendering for older SDL versions (`renderer_sdl2.c:502`)

**4. Optimized Rendering Pipeline**
- Two-Pass Rendering: Separate background pass + foreground pass (`renderer_sdl2.c:442-472`)
- Vertex Batching: Up to 10,922 tiles per batch (`BUFFER_TILES_MAX`, line 42)
- Stack-Allocated Buffers: Vertex buffer allocated on stack, reused between background/foreground passes
- Immediate Flush on Overflow: Automatic buffer flushing prevents memory issues

**5. Simple but Effective Shaders**
- Vertex Shader: Basic position transform + pass-through for colors/UVs (`console_grid.glslv`)
- Fragment Shader: Simple `mix(bg_color, fg_color, tile_alpha)` blend (`console_grid.glslf:14`)
- No Complex Lookups: Uses direct color values, not indexed lookups

### Critical Performance Insights

**TCOD's Secret Formula:**
1. Resource Reuse: Atlas texture, cache texture, cache console all persistent
2. Change Detection: Only process tiles that actually changed
3. Batched Rendering: Massive vertex buffers minimize draw calls
4. Stack Allocation: Minimal heap allocation overhead
5. SDL Abstraction: Let SDL handle efficient GPU communication

### Comparison with Our ModernGL Architecture

**What We Adopted from TCOD:**
- Persistent atlas texture approach
- Batched vertex rendering
- Change detection for static content
- Simple, efficient shader design

**What We Do Differently (And Why):**
- ModernGL direct access vs SDL abstraction - gives us more control for advanced effects
- FBO-based UI rendering - enables complex UI composition
- Dual shader pipeline - better separation of concerns
- More sophisticated caching - handles complex UI hierarchies

**What We Initially Missed (But Later Added):**
- Persistent FBO/texture caching - was the critical missing piece
- Change detection/dirty tracking - implemented TCOD-style system
- Batched updates - now only update changed portions of VBOs

### Validation of Our Approach

The TCOD analysis confirmed that our fundamental approach was sound:
- GPU resource reuse is the primary performance factor
- Change detection provides massive optimization for static content
- Simple shaders are sufficient for excellent performance
- Batched rendering minimizes GPU overhead

Our ModernGL system achieves comparable performance to TCOD while providing additional capabilities for modern rendering effects and better integration with GPU-based lighting systems.
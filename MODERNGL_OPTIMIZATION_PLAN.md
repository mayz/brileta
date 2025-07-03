# ModernGL Rendering Performance Optimization Plan

## 1. Executive Summary

The current ModernGL backend suffers from critical performance issues, resulting in ~25 FPS compared to the TCOD backend's ~700 FPS. This is not due to a minor inefficiency but fundamental architectural problems in how GPU resources are managed.

The primary bottleneck is the **per-frame creation and destruction of GPU resources** (Textures and Framebuffer Objects). This must be solved before any other optimizations are attempted.

This plan is structured in three phases, designed to be executed in order. Each phase builds on the last and tackles the next most significant performance bottleneck.

-   **Phase 1: Fix Critical GPU Resource Churn.** This will provide the largest and most immediate performance gain, likely bringing FPS into the 200-400 range.
-   **Phase 2: Optimize Data Transfer.** This will reduce the amount of redundant data sent to the GPU each frame, further improving performance for static or mostly-static UI elements.
-   **Phase 3: Advanced Shader-Side Rendering (profile game at this point to see if still necessary).** Complex shader optimizations that may be unnecessary given TCOD achieves 700 FPS with simple shaders. Only implement if Phases 1-2 don't achieve target performance.

**Execute these phases sequentially.** Do not proceed to the next phase until the previous one is complete and verified.

---

## Foundational Analysis of TCOD's Rendering Pipeline

**Analysis Results:** Deep examination of Python tcod wrapper (`.venv/lib/python3.13/site-packages/tcod/`) and C libtcod source code (`~/Downloads/libtcod-1.24.0/src/libtcod/`) reveals how TCOD achieves 700 FPS performance.

### Key Findings - How TCOD Achieves 700 FPS

**1. Persistent GPU Resource Management ✅ VALIDATES OUR APPROACH**
- **Atlas Texture**: Created once at startup (`renderer_sdl2.h:56-57`, `renderer_sdl2.c:121`), **never recreated per frame**
- **Cache Console**: Persistent console tracking last frame state (`renderer_sdl2.h:73`)
- **Cache Texture**: Persistent render target texture (`renderer_sdl2.h:74`)
- **Recreation Only on Size Change**: Resources only recreated when console dimensions change (`renderer_sdl2.c:537-545`)

**2. Intelligent Change Detection ✅ VALIDATES PHASE 2**
- **Tile-by-Tile Dirty Checking**: Only renders tiles that changed since last frame (`renderer_sdl2.c:426-440`)
- **Cache Comparison**: Compares `console->tiles[i]` vs `cache->tiles[i]` for background and foreground separately
- **Skip Unchanged**: `continue` for unchanged tiles (line 436) - **massive CPU savings**

**3. Efficient Data Transfer Patterns ✅ DIFFERENT FROM OUR APPROACH BUT EFFECTIVE**
- **NO Direct OpenGL**: Uses SDL2 abstraction layer instead of raw OpenGL calls
- **SDL_UpdateTexture()**: For atlas updates (`renderer_sdl2.c:85`) - equivalent to `glTexSubImage2D`
- **SDL_RenderGeometryRaw()**: Batched vertex rendering for modern SDL 2.0.18+ (`renderer_sdl2.c:297`)
- **Fallback to SDL_RenderCopy()**: Per-tile rendering for older SDL versions (`renderer_sdl2.c:502`)

**4. Optimized Rendering Pipeline ✅ DIFFERENT ARCHITECTURE BUT SOUND PRINCIPLES**
- **Two-Pass Rendering**: Separate background pass + foreground pass (`renderer_sdl2.c:442-472`)
- **Vertex Batching**: Up to 10,922 tiles per batch (`BUFFER_TILES_MAX`, line 42)
- **Stack-Allocated Buffers**: Vertex buffer allocated on stack, reused between background/foreground passes
- **Immediate Flush on Overflow**: Automatic buffer flushing prevents memory issues

**5. Simple but Effective Shaders ✅ SIMPLER THAN OUR PHASE 3 PLANS**
- **Vertex Shader**: Basic position transform + pass-through for colors/UVs (`console_grid.glslv`)
- **Fragment Shader**: Simple `mix(bg_color, fg_color, tile_alpha)` blend (`console_grid.glslf:14`)
- **No Complex Lookups**: Uses direct color values, not indexed lookups like our Phase 3 plan

### Critical Performance Insights

**Our Hypothesis CONFIRMED**: The 700 FPS comes from **persistent resource reuse** and **avoiding per-frame GPU resource creation**.

**TCOD's Secret Formula:**
1. **Resource Reuse**: Atlas texture, cache texture, cache console all persistent
2. **Change Detection**: Only process tiles that actually changed
3. **Batched Rendering**: Massive vertex buffers minimize draw calls
4. **Stack Allocation**: Minimal heap allocation overhead
5. **SDL Abstraction**: Let SDL handle efficient GPU communication

### Comparison with Our Architecture

**What We're Doing SIMILARLY (Good):**
- Persistent atlas texture approach
- Batched vertex rendering
- ModernGL for efficient GPU access

**What We're Doing DIFFERENTLY (Some Good, Some Bad):**
- ✅ **Good Different**: Using ModernGL directly vs SDL abstraction - gives us more control
- ✅ **Good Different**: More advanced shader pipeline planned for Phase 3
- ❌ **Bad Different**: Creating/destroying FBOs and textures every frame (the main bottleneck)
- ❌ **Bad Different**: No change detection system yet

**What We're MISSING (Must Fix):**
- ❌ **Critical**: Persistent FBO/texture caching (Phase 1 will fix this)
- ❌ **Important**: Change detection/dirty tracking (Phase 2 will add this)
- ❌ **Minor**: Stack-allocated buffers (acceptable difference)

### Validation of Our Optimization Plan

✅ **Phase 1 targeting GPU resource churn is EXACTLY right** - This is the primary bottleneck
✅ **Phase 2 change detection already exists in TCOD** - We should implement similar dirty tracking
⚠️ **Phase 3 shader optimizations may be overkill** - TCOD uses simple shaders and achieves 700 FPS

**Recommendation:** Proceed to Phase 1 immediately. The persistent resource approach is validated and will provide massive performance gains.

### Performance Targets & Success Metrics

**Current State:** ~25 FPS (ModernGL) vs ~700 FPS (TCOD) = **28x performance gap**

**Phase 1 Target:** 200-400 FPS (8-16x improvement)
- Success Metric: Stable rendering with no FBO/texture creation in profiler
- Measurement: Frame time should drop from ~40ms to ~5ms or less

**Phase 2 Target:** 400-600 FPS (Additional 1.5-2x improvement)
- Success Metric: CPU usage drops significantly for static UI content
- Measurement: Vertex generation time should be near-zero for unchanged content

**Phase 3 Target:** 600+ FPS (Optional - only if needed)
- Success Metric: Match or exceed TCOD's 700 FPS
- Decision Point: Skip if within 80% of TCOD performance after Phase 2

## Phase 1: Fixing Critical GPU Resource Churn (Highest Priority) ✅ COMPLETED

**Goal:** Eliminate all per-frame creation/destruction of Textures and Framebuffer Objects (FBOs). All such resources should be created once and reused.

**Status:** Phase 1 completed successfully. All GPU resource churn has been eliminated through FBO pooling, reusable gradient textures, and persistent VBO/VAO resources. However, initial testing shows no significant FPS improvement, suggesting the bottleneck may be elsewhere (likely CPU-side operations). Profiling recommended before proceeding to Phase 2.

### Task 1.1: Implement FBO & Texture Pooling for UI/Canvas Rendering ✅ COMPLETED

**Problem:** The `catley.backends.moderngl.texture_renderer.TextureRenderer` creates a new `moderngl.Texture` and `moderngl.Framebuffer` every time its `render` method is called. This happens for every UI view and for the `WorldView`'s light overlay, every single frame. This is extremely slow.

**Relevant Code:** `catley/backends/moderngl/texture_renderer.py`

```python
# In TextureRenderer.render()
# BAD: These are created and destroyed every call.
dest_texture = self.mgl_context.texture((width_px, height_px), 4)
fbo = self.mgl_context.framebuffer(color_attachments=[dest_texture])
# ...
fbo.release()
return dest_texture
```

**Solution:** The `ModernGLGraphicsContext` must own a pool of reusable FBOs and their attached textures. The `TextureRenderer` will render *into* a provided FBO instead of creating its own.

**Implementation Steps:**

1.  **Modify `ModernGLGraphicsContext`:**
    -   In `__init__`, add a cache for FBOs:
        `self.fbo_cache: dict[tuple[int, int], tuple[moderngl.Framebuffer, moderngl.Texture]] = {}`
    -   Create a new private helper method `_get_or_create_render_target(self, width: int, height: int) -> tuple[moderngl.Framebuffer, moderngl.Texture]`.
    -   This method checks `self.fbo_cache` for an existing FBO/Texture tuple of the requested `(width, height)`. If it exists, return it. If not, create it, store it in the cache, and then return it. **Do not release it.**

2.  **Refactor `render_glyph_buffer_to_texture`:**
    -   In `catley.backends.moderngl.graphics.ModernGLGraphicsContext`, modify `render_glyph_buffer_to_texture`.
    -   This method should now be the one that calls `_get_or_create_render_target` to get a destination FBO and texture.
    -   It will then call `self.texture_renderer.render`, passing both the `glyph_buffer` and the `target_fbo`.

3.  **Refactor `TextureRenderer.render`:**
    -   In `catley.backends.moderngl.texture_renderer.py`, change the signature of `render` to: `render(self, glyph_buffer: GlyphBuffer, target_fbo: moderngl.Framebuffer) -> None`. It no longer returns a texture.
    -   Remove all FBO and texture creation/release logic from this method.
    -   Use the `target_fbo` that is passed in: `target_fbo.use()`.
    -   After rendering, switch back to the main screen framebuffer: `self.mgl_context.screen.use()`.

4.  **Update `ModernGLCanvas.create_texture`:**
    -   This method in `catley.backends.moderngl.canvas.py` calls `render_glyph_buffer_to_texture`. The call signature doesn't need to change, but the implementation within `ModernGLGraphicsContext` is now different and much more efficient.

**Expected Outcome:** A massive performance increase. FPS should jump from ~25 to well over 100, possibly into the 200-400 range, depending on other factors.

**Actual Outcome:** Implementation completed successfully. FBO pooling system works correctly and eliminates GPU resource creation. However, no significant FPS improvement observed, indicating this was not the primary bottleneck.

### Task 1.2: Implement Reusable Texture for Environmental Effects ✅ COMPLETED

**Problem:** `apply_environmental_effect` in `ModernGLGraphicsContext` creates a NumPy array for a radial gradient, uploads it to a new GPU texture, uses it once, and releases it. This is done for every single light/fog effect on screen, every frame.

**Relevant Code:** `catley/backends/moderngl/graphics.py`

```python
# In ModernGLGraphicsContext.apply_environmental_effect()
# BAD: NumPy array creation and texture upload every call.
effect_data = np.zeros((size, size, 4), dtype=np.uint8)
# ...
effect_texture = self.mgl_context.texture((size, size), 4, effect_data.tobytes())
# ...
effect_texture.release()
```

**Solution:** Create one single, reusable radial gradient texture when the game starts. Render all circular effects by drawing a quad using this one texture, and use vertex colors to apply the tint and intensity.

**Implementation Steps:**

1.  **In `ModernGLGraphicsContext.__init__`:**
    -   Create a new private method `_create_radial_gradient_texture(self, resolution: int) -> moderngl.Texture`. This method will contain the NumPy logic to generate a white radial gradient (from center=white to edge=transparent). It should only be called once.
    -   In `__init__`, call this new method and store the result:
        `self.radial_gradient_texture = self._create_radial_gradient_texture(256)`

2.  **Refactor `apply_environmental_effect`:**
    -   Remove all NumPy array creation, texture creation, and texture release code.
    -   This method should now calculate the destination rectangle (`dest_x`, `dest_y`, `size`) for the effect on screen.
    -   Calculate the final `color_rgba` tuple by combining the `tint_color` and `intensity`. The `intensity` should map to the alpha channel.
    -   Instead of doing a complex draw call, simply call `self.screen_renderer.add_quad()`. Pass the destination rect, the `color_rgba`, and the UV coordinates for the full gradient texture (`(0.0, 0.0, 1.0, 1.0)`).
    -   **Texture Switching Issue**: `self.screen_renderer.add_quad` is designed for the main `atlas_texture`. For environmental effects, either:
        - **Option A (Immediate)**: Flush the current batch, bind `radial_gradient_texture`, draw the effect quad immediately, then rebind `atlas_texture`
        - **Option B (Better)**: Extend the renderer to handle multiple texture types (atlas vs effects) - but this can wait until Phase 2

**Expected Outcome:** FPS will become stable and high even with many lights or fog effects on screen. This completes the critical performance fixes.

**Actual Outcome:** Implementation completed successfully. Reusable gradient texture and persistent VBO/VAO resources eliminate all per-frame GPU resource creation for environmental effects. However, no significant FPS improvement observed, indicating the bottleneck lies elsewhere.

### Additional Optimization (Task 1.3): Eliminate Temporary VBO/VAO Creation ✅ COMPLETED

**Problem Discovered:** During implementation, identified that `_draw_single_texture_immediately()` and `_draw_environmental_effect_immediately()` were creating temporary VBO/VAO resources every frame for background rendering and environmental effects.

**Solution Implemented:** Added persistent `immediate_vbo`, `immediate_vao_screen`, and `immediate_vao_ui` resources in `ModernGLGraphicsContext` to eliminate all temporary GPU resource creation.

**Outcome:** All per-frame GPU resource creation has been eliminated. The optimization is technically successful but reveals that GPU resource churn was not the primary performance bottleneck.

---

## Phase 2: Optimize Data Transfer

**Goal:** Reduce the amount of CPU work and GPU data transfer for UI elements that do not change every frame.

### Task 2.1: Implement TCOD-Style Change Detection for `GlyphBuffer`

**Problem:** `TextureRenderer._encode_glyph_buffer_to_vertices` reconstructs the entire vertex buffer from the `GlyphBuffer` every time it's called, even if only one character has changed (e.g., a blinking cursor).

**TCOD Reference:** TCOD achieves massive CPU savings by comparing each tile against a cached version from the previous frame (`renderer_sdl2.c:426-440`). It only processes tiles where background color, foreground color, or character has changed.

**Our Solution:** Implement similar tile-by-tile change detection, adapted for our `GlyphBuffer` structure.

**Implementation Steps:**

1.  **Modify `ModernGLCanvas` to Add Cache Buffer:**
    -   This class owns the `private_glyph_buffer`. Add a cache buffer: `self.cache_glyph_buffer: GlyphBuffer | None = None`.
    -   After a successful render, copy the current buffer to the cache slot: `self.cache_glyph_buffer = self.private_glyph_buffer.copy()`.

2.  **Implement TCOD-Style Dirty Detection in `TextureRenderer`:**
    -   Add method signature: `render(self, glyph_buffer: GlyphBuffer, cache_buffer: GlyphBuffer | None, target_fbo: moderngl.Framebuffer) -> None`
    -   **Tile-by-Tile Comparison**: For each `(x, y)` position, compare current vs cache:
        ```python
        current_tile = glyph_buffer.data[y, x]
        cached_tile = cache_buffer.data[y, x] if cache_buffer else None

        # Check for changes (following TCOD's approach)
        bg_changed = (cached_tile is None or
                     current_tile.bg_color != cached_tile.bg_color)
        fg_changed = (cached_tile is None or
                     current_tile.character != cached_tile.character or
                     current_tile.fg_color != cached_tile.fg_color)

        if not (bg_changed or fg_changed):
            continue  # Skip unchanged tiles entirely
        ```
    -   **Sparse Vertex Generation**: Only add vertices for changed tiles to the VBO
    -   **Partial VBO Updates**: Use `self.vbo.write(chunk, offset=...)` to update only the sections corresponding to changed tiles

3.  **Optimize for Common UI Patterns:**
    -   **Static Text**: Most UI text doesn't change between frames
    -   **Cursor Blinking**: Only 1-2 characters change per frame
    -   **Status Updates**: Only specific regions (health bars, scores) change
    -   **Background Panels**: Often completely static

**Expected Outcome:** TCOD-level CPU efficiency gains. Massive reduction in vertex data generation and GPU uploads for mostly-static UI content. Should see 50-80% reduction in CPU usage for typical UI rendering scenarios.

**Note:** This implements the same core optimization that gives TCOD its performance advantage - the `continue` statement that skips unchanged content entirely.

---

## Phase 3: Advanced Shader-Side Rendering - Profile the game again before implementing to see if this is still needed

**Goal:** Minimize CPU-to-GPU bandwidth by sending minimal integer data and performing lookups on the GPU. This was the original "endgame" optimization.

**TCOD Analysis Update:** TCOD achieves 700 FPS with **simple shaders** and **direct color values**, not indexed lookups. Their fragment shader is just `mix(bg_color, fg_color, tile_alpha)` with no complex palette lookups.

**Recommendation:** Phase 3 may be unnecessary complexity. After Phases 1-2, we should **measure performance** before proceeding. TCOD proves that simple shaders + persistent resources + change detection is sufficient for excellent performance.

### Task 3.1: Implement Shader-Based Glyph & Color Rendering (IF NEEDED)

**Problem:** The current renderer sends full UV coordinates (8 bytes) and full float RGBA color (16 bytes) for every vertex. This is highly redundant compared to integer indices.

**TCOD Comparison:** TCOD sends similar data (position + colors + UVs) and achieves 700 FPS. The vertex data size may not be the bottleneck we thought it was.

**Our Advanced Solution (More Complex than TCOD):**
Redesign the rendering pipeline to use integer indices and perform lookups in the shaders - going beyond TCOD's approach for potential additional gains.

**Implementation Steps (Only if Phase 1-2 aren't sufficient):**

1.  **New Compact Vertex Format:**
    -   Current: `position(8) + uv(8) + fg_color(16) + bg_color(16) = 48 bytes/vertex`
    -   Proposed: `position(8) + char_id(4) + fg_color_id(4) + bg_color_id(4) = 20 bytes/vertex`
    -   **58% reduction in vertex data size**

2.  **Lookup-Based Shaders (More Advanced than TCOD):**
    -   **Vertex Shader:** Receives `char_id`, looks up UV coordinates from uniform texture
    -   **Fragment Shader:** Receives color IDs, looks up RGBA values from palette texture
    -   **Blending:** Still uses TCOD's proven `mix(bg_color, fg_color, tile_alpha)` approach

3.  **Dynamic Palette Management:**
    -   Map `(R,G,B,A)` tuples to integer IDs
    -   Upload palette as 1D texture to GPU
    -   Handle palette overflow with LRU eviction

4.  **Backward Compatibility:**
    -   Keep current shader pipeline as fallback
    -   Add config option to enable/disable indexed rendering

**Expected Outcome:** Potentially higher performance than TCOD due to reduced vertex bandwidth, but adds significant complexity. **Only implement if Phases 1-2 don't achieve target performance.**

**Key Decision Point:** After Phase 2 completion, benchmark against TCOD. If we're within 80% of TCOD's performance, **skip Phase 3** and focus on other game systems instead.